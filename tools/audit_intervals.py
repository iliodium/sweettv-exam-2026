"""Read-only raw-CSV interval union and temporal protocol audit.

Selects processed rows only to choose cases, then independently reconstructs
their watch seconds from the ORIGINAL CSV view intervals and schedule rows.
It never uses joined_segments or the root SQL merging implementation.
"""
import csv
import argparse
import json
import time
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
RAW=ROOT/'data/raw'
PROCESSED=ROOT/'data/processed'
WINDOWS={
 'train':(datetime(2020,3,9),datetime(2020,6,29)),
 'val':(datetime(2020,6,29),datetime(2020,7,27)),
 'test':(datetime(2020,7,27),datetime(2020,10,19)),
}


def union_intervals(intervals):
    merged=[]
    for start,end in sorted(intervals):
        if end<=start:continue
        if merged and start<=merged[-1][1]:
            merged[-1]=(merged[-1][0],max(merged[-1][1],end))
        else:merged.append((start,end))
    return sum((b-a).total_seconds() for a,b in merged),merged


def main():
    started=time.perf_counter()
    con=duckdb.connect(str(PROCESSED/'data.duckdb'),read_only=True)
    fields=['split','user_id','airing_id','item_id','channel_id','duration_seconds','airing_start','airing_end','watch_seconds','coverage']
    selected=[]
    predicates=[('full','coverage=1'),('partial_positive','coverage>=.8 AND coverage<1'),('near_negative','coverage>=.79 AND coverage<.8'),('short_negative','coverage<.05')]
    for split in ['train','val']:
        for label,predicate in predicates:
            rows=con.execute(f"SELECT * FROM airing_watch WHERE split=? AND {predicate} ORDER BY hash(user_id||':'||airing_id::VARCHAR) LIMIT 4",[split]).fetchall()
            selected.extend({**dict(zip(fields,row)),'selection_stratum':label} for row in rows)
    failure=con.execute("SELECT * FROM airing_watch WHERE user_id='8986460610792053980' AND airing_id=88548").fetchone()
    assert failure is not None
    selected.append({**dict(zip(fields,failure)),'selection_stratum':'known_duplicate_sort_failure'})
    selected=list({(r['user_id'],r['airing_id']):r for r in selected}.values())
    assert len(selected)>=30
    index=defaultdict(list)
    for row in selected:index[(row['user_id'],row['channel_id'])].append(row)
    segments=defaultdict(list)
    matched_raw_lines=defaultdict(list)
    raw_view_count=0;min_start=None;max_stop=None;invalid_raw=0;max_user_id=0
    with (RAW/'dataset11-30.csv').open(encoding='utf-8-sig',newline='') as handle:
        reader=csv.DictReader(handle)
        for line,row in enumerate(reader,2):
            raw_view_count+=1
            start_text,stop_text=row['start_time'],row['stop_time']
            min_start=start_text if min_start is None else min(min_start,start_text)
            max_stop=stop_text if max_stop is None else max(max_stop,stop_text)
            invalid_raw+=int(start_text>=stop_text)
            max_user_id=max(max_user_id,int(row['user_id']))
            cases=index.get((row['user_id'],int(row['vsetv_id'])))
            if not cases:continue
            start=datetime.fromisoformat(start_text);stop=datetime.fromisoformat(stop_text)
            for case in cases:
                overlap_start=max(start,case['airing_start']);overlap_end=min(stop,case['airing_end'])
                if overlap_start<overlap_end:
                    key=(case['user_id'],case['airing_id'])
                    segments[key].append((overlap_start,overlap_end))
                    matched_raw_lines[key].append(line)
    schedule_keys={(r['channel_id'],r['airing_start'],r['duration_seconds'],r['item_id']) for r in selected}
    schedule_matches=Counter();candidate_sets={name:set() for name in WINDOWS}
    time_only_candidate_sets={name:set() for name in WINDOWS}
    fullfit_correct_source_sets={name:set() for name in WINDOWS}
    starts_in_window_sets={name:set() for name in WINDOWS}
    metadata_ids=set();source_counts=[];boundary=Counter()
    for source,name in enumerate(['export_arh_11-20-final.csv','export_arh_21-30-final.csv','export_arh_31-42-final.csv']):
        counts=Counter()
        with (RAW/name).open(encoding='utf-8-sig',newline='') as handle:
            reader=csv.DictReader(handle)
            for row in reader:
                counts['rows']+=1
                item=int(row['tv_show_id']);duration=int(row['duration'])
                if item==0:
                    counts['unlabelled_rows']+=1
                    continue
                counts['labelled_rows']+=1
                if duration<=0:
                    counts['invalid_labelled_duration']+=1
                    continue
                metadata_ids.add(item)
                start=datetime.strptime(row['start_time'],'%d.%m.%Y %H:%M:%S')
                end=start+timedelta(seconds=duration)
                key=(int(row['channel_id']),start,duration,item)
                if key in schedule_keys:schedule_matches[key]+=1
                for split,(lower,upper) in WINDOWS.items():
                    proper_source=(split=='test' and source==2) or (split!='test' and source in [0,1])
                    if proper_source and lower<=start<upper:
                        starts_in_window_sets[split].add(item)
                        candidate_sets[split].add(item)
                    if start>=lower and end<=upper:
                        time_only_candidate_sets[split].add(item)
                        if proper_source:fullfit_correct_source_sets[split].add(item)
                    elif start<upper and end>lower:boundary[split]+=1
        source_counts.append({'source':name,**dict(counts)})
    cases=[]
    for case in selected:
        key=(case['user_id'],case['airing_id'])
        total,merged=union_intervals(segments[key])
        duration=case['duration_seconds'];coverage=total/duration
        lower,upper=WINDOWS[case['split']]
        raw_key=(case['channel_id'],case['airing_start'],duration,case['item_id'])
        result={**case,'independent_union_seconds':total,'independent_coverage':coverage,
          'raw_overlap_segments':len(segments[key]),'raw_distinct_intervals':len(set(segments[key])),
          'raw_sum_seconds':sum((end-start).total_seconds() for start,end in segments[key]),
          'merged_islands':len(merged),'raw_csv_line_numbers':matched_raw_lines[key],
          'matching_raw_schedule_rows':schedule_matches[raw_key],
          'same_seconds':total==case['watch_seconds'],
          'same_coverage':abs(coverage-case['coverage'])<1e-12,
          'same_positive_qualification':(coverage>=.8)==(case['coverage']>=.8),
          'fully_in_declared_fold':lower<=case['airing_start']<case['airing_end']<=upper,
          'no_programme_zero':case['item_id']>0}
        assert all(result[x] for x in ['same_seconds','same_coverage','same_positive_qualification','fully_in_declared_fold','no_programme_zero'])
        assert result['matching_raw_schedule_rows']>=1
        cases.append(result)
    comprehensive={}
    for filename in ['airing_watch.parquet','events.parquet']:
        totals=Counter();mins={};maxs={};max_coverage_error=0.
        for batch in pq.ParquetFile(PROCESSED/filename).iter_batches(batch_size=131072):
            cols={n:batch.column(n) for n in batch.schema.names}
            item=cols['item_id'].to_numpy();duration=cols['duration_seconds'].to_numpy();seconds=cols['watch_seconds'].to_numpy();coverage=cols['coverage'].to_numpy()
            starts=cols['airing_start'].to_numpy();ends=cols['airing_end'].to_numpy()
            totals['rows']+=len(batch)
            totals['invalid_id']+=int((item<=0).sum())
            totals['invalid_duration']+=int((duration<=0).sum())
            totals['impossible_seconds']+=int(((seconds<=0)|(seconds>duration)).sum())
            totals['impossible_coverage']+=int(((coverage<=0)|(coverage>1)|~np.isfinite(coverage)).sum())
            max_coverage_error=max(max_coverage_error,float(np.max(np.abs(coverage-seconds/duration))))
            if filename=='events.parquet':totals['below_positive_threshold']+=int((coverage<.8).sum())
            split_values=cols['split'].to_pylist()
            for split in ['train','val']:
                mask=np.fromiter((x==split for x in split_values),dtype=bool)
                if not mask.any():continue
                totals[split+'_rows']+=int(mask.sum());lower,upper=WINDOWS[split]
                totals[split+'_temporal_violations']+=int(((starts[mask]<np.datetime64(lower))|(ends[mask]>np.datetime64(upper))|(ends[mask]<=starts[mask])).sum())
        comprehensive[filename]={**dict(totals),'max_coverage_formula_difference':max_coverage_error}
        assert totals['invalid_id']==totals['invalid_duration']==totals['impossible_seconds']==totals['impossible_coverage']==totals['train_temporal_violations']==totals['val_temporal_violations']==0
        if filename=='events.parquet':assert totals['below_positive_threshold']==0
        assert max_coverage_error<1e-12
    failure_case=next(x for x in cases if x['selection_stratum']=='known_duplicate_sort_failure')
    assert failure_case['raw_overlap_segments']==3 and failure_case['raw_distinct_intervals']==2 and failure_case['raw_sum_seconds']==3643 and failure_case['independent_union_seconds']==1800
    result={
      'scope':'Independent raw-CSV sample reconstruction with Python interval merging, plus complete parquet temporal/id/range invariants. No joined_segments/root merge SQL used.',
      'all_pass':True,'sample_cases':len(cases),'sample':cases,
      'raw_view_scan':{'rows':raw_view_count,'start_min':min_start,'stop_max':max_stop,'invalid_intervals':invalid_raw,'unsigned_user_id_above_int64_present':max_user_id>2**63-1,'no_test_period_view_rows':max_stop<'2020-07-27 00:00:00'},
      'schedule_scan':source_counts,'independent_candidates':{k:len(v) for k,v in candidate_sets.items()},'independent_metadata_ids':len(metadata_ids),'boundary_labelled_airings':dict(boundary),
      'candidate_source_policy':{'train_val':'source files0/1, airing START inside fold','test':'test schedule source file2, airing START inside test window','time_only_fullfit_counts_including_crossfile_spill':{k:len(v) for k,v in time_only_candidate_sets.items()},'fullfit_candidate_counts_by_correct_source':{k:len(v) for k,v in fullfit_correct_source_sets.items()},'starts_in_window_candidate_counts_by_correct_source':{k:len(v) for k,v in starts_in_window_sets.items()},'validation_ids_added_by_consistent_starts_in_window_rule':sorted(starts_in_window_sets['val']-fullfit_correct_source_sets['val']),'audit_note':'Consistent start-in-window availability after independent audit; labels still separately require full containment. Earlier VAL candidate rule removed12boundary-only items.'},
      'complete_parquet_checks':comprehensive,
      'documented_before_after_failure':{'user_id':'8986460610792053980','airing_id':88548,'item_id':6700482,'old_reported_coverage':2.0,'old_failure_source':'Root-reported first SQL run; not independently rerun or fabricated here.','current_independent_union_seconds':1800,'current_duration_seconds':1800,'current_coverage':1.0,'raw_sum_seconds':3643,'root_fix':'Unique view_id included in BOTH window-function ORDER BY lists.'},
      'limitations':['Raw overlap reconstruction covers the selected cases, not every user-airing. Complete parquet checks cover all exported interactions but cannot alone prove all raw joins exhaustive.','Ground-truth top-five frequency and model-level train-only fitting require separate experiment audit.'],
      'seconds':time.perf_counter()-started}
    assert result['raw_view_scan']['no_test_period_view_rows'] and invalid_raw==0
    assert len(candidate_sets['val'])==3957 and len(candidate_sets['test'])==6632 and len(metadata_ids)==9671
    actual_candidates=json.loads((PROCESSED/'candidates.json').read_text())
    result['root_candidate_sets_match_independent_scan']={name:set(actual_candidates[name])==candidate_sets[name] for name in ['val','test']}
    assert all(result['root_candidate_sets_match_independent_scan'].values())
    (OUT/'interval_audit.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['sample']},indent=2,default=str))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args();ROOT=args.root.resolve();RAW=ROOT/'data/raw';PROCESSED=ROOT/'data/processed';OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=True)
    main()
