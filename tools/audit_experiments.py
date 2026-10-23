"""Independent read-only reconstruction of truth, matrices, MAP and CSV files.

Uses raw positive-event exports and exact Fraction arithmetic, not metrics.py
or the experiment's labels/metric functions. Run only after a grid completes.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy import sparse


def exact_ap(actual,predicted,k=5):
    target=frozenset(actual)
    if not target:return None
    score=Fraction(0)
    for index,item in enumerate(predicted[:k]):
        if item in target and item not in predicted[:index]:
            score+=Fraction(len(target.intersection(predicted[:index+1])),index+1)
    return score/min(k,len(target))


def main(root,output):
    evidence=root/'evidence';art=root/'artifacts';raw=root/'data/raw';proc=root/'data/processed'
    exp=json.loads((evidence/'experiments.json').read_text())
    index=json.loads((art/'indices.json').read_text())
    cands=json.loads((proc/'candidates.json').read_text())
    with (raw/'submission.csv').open(newline='') as handle:
        reader=csv.DictReader(handle);assert reader.fieldnames==['user_id','tv_show_id'];users=[r['user_id'] for r in reader]
    train=Counter();val=Counter();train_airings=set();val_airings=set()
    for batch in pq.ParquetFile(proc/'events.parquet').iter_batches(batch_size=131072):
        for row in batch.to_pylist():
            assert row['item_id']>0 and row['coverage']>=.8
            key=(row['user_id'],int(row['item_id']));air=(row['user_id'],int(row['airing_id']))
            if row['split']=='train':
                assert air not in train_airings;train_airings.add(air);train[key]+=1
            elif row['split']=='val':
                assert air not in val_airings;val_airings.add(air);val[key]+=1
            else:raise AssertionError('Unexpected fold in events: '+row['split'])
    truth_counts=defaultdict(list)
    for (u,item),count in val.items():truth_counts[u].append((item,count))
    truth={u:sorted(values,key=lambda pair:(-pair[1],pair[0]))[:5] for u,values in truth_counts.items()}
    cohort=[u for u in users if truth.get(u)];train_users={u for u,item in train};cold=[u for u in users if u not in train_users]
    checks={}
    def check(name,condition,details=None):checks[name]={'passed':bool(condition),'details':details}
    check('actual_template_strings_preserved',index['target_user_ids']==users and len(users)==1260 and len(set(users))==len(users) and all(isinstance(u,str) for u in users))
    check('training_user_universe_no_validation_only_identities',set(index['user_ids'])==train_users|set(users),{'actual_index_users':len(index['user_ids']),'expected_train_plus_template_users':len(train_users|set(users))})
    check('cohort_and_cold_counts',len(cohort)==1216 and len(users)-len(cohort)==44 and len(cold)==13,{'all_template_users':len(users),'evaluated':len(cohort),'excluded_no_relevant':len(users)-len(cohort),'cold_train_users':len(cold),'cold_evaluable':len(set(cold)&set(cohort))})
    labels=pq.read_table(art/'validation_labels.parquet').to_pylist()
    actual_labels=defaultdict(list)
    for row in labels:actual_labels[row['user_id']].append((int(row['item_id']),int(row['frequency'])))
    for u in actual_labels:actual_labels[u].sort(key=lambda pair:(-pair[1],pair[0]))
    check('all_validation_top5_frequency_truth',dict(actual_labels)==truth,{'users':len(truth),'template_relevant_pairs':sum(len(truth[u]) for u in cohort)})
    matrix=sparse.load_npz(art/'train_matrix.npz').tocsr();uid_index={u:i for i,u in enumerate(index['user_ids'])};iid_index={int(item):i for i,item in enumerate(index['item_ids'])}
    rows=[];cols=[];values=[]
    for (u,item),count in train.items():rows.append(uid_index[u]);cols.append(iid_index[item]);values.append(count)
    independent=sparse.coo_matrix((values,(rows,cols)),shape=matrix.shape).tocsr()
    delta=matrix-independent
    check('entire_training_matrix_reconstructed',delta.nnz==0 and matrix.nnz==len(train),{'positive_airings':sum(train.values()),'distinct_user_item_pairs':len(train),'matrix_shape':list(matrix.shape)})
    check('candidate_indices_match_current_policy',index['validation_candidates']==cands['val'] and index['test_candidates']==cands['test'])
    protocol=json.loads((evidence/'evaluation_protocol.json').read_text())
    check('reported_protocol_counts',protocol['evaluated_users']==len(cohort) and protocol['excluded_no_validation_relevance']==len(users)-len(cohort) and protocol['cold_train_template_users']==len(cold) and protocol['final_models_refit_on_validation'] is False)
    pop=Counter()
    degree=Counter();airing_counts=Counter()
    for (u,item),count in train.items():pop[item]+=1;degree[u]+=1;airing_counts[u]+=count
    expected_cold={s:sorted(cands[s],key=lambda item:(-pop[item],item))[:5] for s in ['val','test']}
    records=[]
    for record in exp['records']:
        key=record['key'];pred=json.loads((art/(key+'_predictions.json')).read_text())
        shape_valid=True
        for split in ['val','test']:
            pool=set(cands[split]);shape_valid &= list(pred[split])==users
            shape_valid &= all(len(items)==5 and len(set(items))==5 and all(isinstance(item,int) and not isinstance(item,bool) and item>0 and item in pool for item in items) for items in pred[split].values())
        ap_values=[exact_ap([x[0] for x in truth[u]],pred['val'][u]) for u in cohort]
        precise=float(sum(ap_values,Fraction(0))/len(cohort))
        hits=[len({x[0] for x in truth[u]}&set(pred['val'][u])) for u in cohort]
        precision=sum(Fraction(h,5) for h in hits)/len(cohort)
        recall=sum(Fraction(h,len(truth[u])) for h,u in zip(hits,cohort))/len(cohort)
        cold_valid=all(pred[s][u]==expected_cold[s] for s in ['val','test'] for u in cold)
        saved=json.loads((art/(key+'_validation.json')).read_text())
        saved_rows={x['user_id']:x for x in saved['per_user']}
        per_user_same=all(abs(saved_rows[u]['ap']-float(ap))<1e-12 for u,ap in zip(cohort,ap_values))
        records.append({'key':key,'family':record['family'],'independent_map':precise,'reported_map':record['map_at_5'],'map_abs_difference':abs(precise-record['map_at_5']),'independent_precision':float(precision),'independent_recall':float(recall),'diagnostic_difference_max':max(abs(float(precision)-record['precision_at_5']),abs(float(recall)-record['recall_at_5'])),'all1260_zero_for_no_relevance_map':float(sum(ap_values,Fraction(0))/len(users)),'valid_1260_ranked_lists_both_splits':shape_valid,'cold_fallback_identical':cold_valid,'all_per_user_aps_match':per_user_same})
    check('all_experiment_metrics_exact_fraction',all(r['map_abs_difference']<1e-12 and r['diagnostic_difference_max']<1e-12 and r['all_per_user_aps_match'] for r in records),{'records':len(records),'max_map_difference':max(r['map_abs_difference'] for r in records)})
    check('all_models_shared_candidates_users_fallback',all(r['valid_1260_ranked_lists_both_splits'] and r['cold_fallback_identical'] for r in records))
    best_record=max(records,key=lambda r:r['independent_map'])
    check('validation_selected_is_maximum_map',exp['validation_selected']['key']==best_record['key'],{'selected':exp['validation_selected']['key'],'independent_best':best_record['key']})
    finite_details=[]
    for path in sorted(art.glob('best_*_scores.np*')):
        if path.suffix=='.npy':arrays={'scores':np.load(path,mmap_mode='r')}
        else:arrays=dict(np.load(path))
        finite_details.append({'file':path.name,'all_finite':all(np.isfinite(v).all() for v in arrays.values())})
    check('saved_family_best_scores_finite',all(x['all_finite'] for x in finite_details),finite_details)
    csv_checks=[]
    for family,record in exp['family_best'].items():
        path=root/record['submission_file'];payload=path.read_bytes()
        with path.open(newline='') as handle:
            reader=csv.DictReader(handle);header=reader.fieldnames;rows=list(reader)
        pred=json.loads((art/(record['key']+'_predictions.json')).read_text())['test']
        exact=header==['user_id','tv_show_id'] and [r['user_id'] for r in rows]==users and all(r['tv_show_id']==' '.join(map(str,pred[r['user_id']])) for r in rows)
        csv_checks.append({'family':family,'file':record['submission_file'],'same_rows_order_header_ids':exact,'sha256':hashlib.sha256(payload).hexdigest(),'hash_matches_record':hashlib.sha256(payload).hexdigest()==record['submission_sha256']})
    check('family_csv_predictions_and_sha_links',all(x['same_rows_order_header_ids'] and x['hash_matches_record'] for x in csv_checks),csv_checks)
    selected=json.loads((evidence/'selected_users.json').read_text())['users']
    warm_cohort=[u for u in cohort if u in train_users];sorted_degrees=sorted(degree[u] for u in warm_cohort)
    selection=[]
    for profile,q in [('light',Fraction(1,10)),('heavy',Fraction(9,10))]:
        at=q*(len(sorted_degrees)-1);lower=at.numerator//at.denominator;fraction=at-lower
        target=Fraction(sorted_degrees[lower])+(sorted_degrees[min(lower+1,len(sorted_degrees)-1)]-sorted_degrees[lower])*fraction
        uid=min(warm_cohort,key=lambda u:(abs(Fraction(degree[u])-target),int(u)))
        selection.append({'profile':profile,'independent_user':uid,'reported_user':selected[profile]['user_id'],'degree':degree[uid],'airings':airing_counts[uid]})
    check('illustrative_users_chosen_from_training_degree_only',all(x['independent_user']==x['reported_user'] for x in selection),selection)
    result={'scope':'Independent Counter-based truth/train matrix and Fraction-based AP/MAP audit. No experiment label/metric function imported.','checks':checks,'checks_count':len(checks),'passed':sum(x['passed'] for x in checks.values()),'all_pass':all(x['passed'] for x in checks.values()),'records':records,'experiment_file_sha256':hashlib.sha256((evidence/'experiments.json').read_bytes()).hexdigest(),'limits':['Uses all positive exported events, whose raw interval correctness is separately sampled in interval_audit.json.','Does not use or infer hidden Kaggle ground truth or private score.']}
    output.mkdir(parents=True,exist_ok=True);(output/'experiment_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'checks':len(checks),'passed':result['passed'],'all_pass':result['all_pass'],'experiments':len(records),'failed':[k for k,v in checks.items() if not v['passed']],'selected':best_record},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True,type=Path);parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent);args=parser.parse_args();main(args.root.resolve(),args.output.resolve())
