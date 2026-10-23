"""Read-only extension check without importing root model/metric/ranking code."""
import argparse
import csv
import hashlib
import json
from collections import Counter,defaultdict
from fractions import Fraction
from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as pq
from scipy import sparse


def ap_exact(actual,predicted):
    target=frozenset(actual)
    if not target:return None
    total=Fraction(0)
    for rank,item in enumerate(predicted[:5],1):
        if item in target and item not in predicted[:rank-1]:
            total+=Fraction(len(target.intersection(predicted[:rank])),rank)
    return total/min(5,len(target))


def main(root,out):
    folder=root/'artifacts/supply_extension';result=json.loads((folder/'supply_extension_results.json').read_text());index=json.loads((root/'artifacts/indices.json').read_text());cands=json.loads((root/'data/processed/candidates.json').read_text())
    with (root/'data/raw/submission.csv').open(newline='') as handle:
        reader=csv.DictReader(handle);assert reader.fieldnames==['user_id','tv_show_id'];users=[r['user_id'] for r in reader]
    counts=Counter()
    for batch in pq.ParquetFile(root/'data/processed/events.parquet').iter_batches():
        for row in batch.to_pylist():
            if row['split']=='val':counts[(row['user_id'],int(row['item_id']))]+=1
    grouped=defaultdict(list)
    for (u,item),frequency in counts.items():grouped[u].append((item,frequency))
    truth={u:[x[0] for x in sorted(items,key=lambda x:(-x[1],x[0]))[:5]] for u,items in grouped.items()}
    cohort=[u for u in users if truth.get(u)];assert len(cohort)==1216
    item_ids=np.asarray(index['item_ids']);imap={int(item):i for i,item in enumerate(item_ids)}
    matrix=sparse.load_npz(root/'artifacts/train_matrix.npz').tocsr();uindex={u:i for i,u in enumerate(index['user_ids'])};cold=np.asarray([matrix.indptr[uindex[u]]==matrix.indptr[uindex[u]+1] for u in users]);pop=np.asarray(matrix.getnnz(axis=0)).ravel()
    connection=duckdb.connect(str(root/'data/processed/data.duckdb'),read_only=True)
    supplies={}
    specs={'train':("source_part<2 AND start_at>=TIMESTAMP '2020-03-09' AND end_at<=TIMESTAMP '2020-06-29'",112),'val':("source_part<2 AND start_at>=TIMESTAMP '2020-06-29' AND start_at<TIMESTAMP '2020-07-27'",28),'test':("source_part=2 AND start_at>=TIMESTAMP '2020-07-27' AND start_at<TIMESTAMP '2020-10-19'",84)}
    for split,(where,_) in specs.items():
        vector=np.zeros(len(item_ids),dtype=np.float64)
        for item,n in connection.execute('SELECT item_id,count(DISTINCT airing_id)FROM airings WHERE '+where+' GROUP BY item_id').fetchall():vector[imap[int(item)]]=n
        supplies[split]=vector
    connection.close()
    checks=[]
    for row in result['records']:
        key=row['key'];pred=json.loads((folder/(key+'_predictions.json')).read_text());report=json.loads((folder/(key+'_validation.json')).read_text());gamma=row['gamma'];family=row['family'];base=json.loads((root/'artifacts'/(row['base_key']+'_predictions.json')).read_text());cache=np.load(root/'artifacts'/('best_'+family+'_scores.npy'),mmap_mode='r')
        numeric_ap=[ap_exact(truth[u],pred['val'][u]) for u in cohort];independent_map=float(sum(numeric_ap,Fraction(0))/len(cohort))
        hits=[len(set(truth[u])&set(pred['val'][u])) for u in cohort]
        precise=float(sum((Fraction(h,5) for h in hits),Fraction(0))/len(cohort));recall=float(sum((Fraction(h,len(truth[u])) for h,u in zip(hits,cohort)),Fraction(0))/len(cohort))
        valid=True;rerank_same={};gamma0={};summary_same=True;fallback_same=True
        for split in ['val','test']:
            ids=np.asarray(cands[split]);assert np.array_equal(ids,np.sort(np.unique(ids)))
            pool=set(ids.tolist());valid &= list(pred[split])==users and all(len(a)==len(set(a))==5 and all(isinstance(i,int) and i>0 and i in pool for i in a) for a in pred[split].values())
            positions=np.asarray([imap[int(i)] for i in ids]);days=specs[split][1]
            # Algebraically independent rate expression: (n_target*Ttrain/Ttarget+4)/(n_train+4).
            ratio=(supplies[split]*(112/days)+4)/(supplies['train']+4)
            multiplier=np.power(np.maximum(.25,np.minimum(4,ratio)),gamma)
            adjusted=(np.asarray(cache[:,positions],dtype=np.float64)*multiplier[positions]).astype(np.float32)
            assert np.isfinite(adjusted).all()
            pop_order=np.argsort(-pop[positions],kind='stable');n_fallback=0;n_filled=0;all_lists_same=True
            for user_index,user in enumerate(users):
                scores=adjusted[user_index];chosen=[]
                if not cold[user_index]:
                    # IDs are ascending; stable score sort independently enforces raw-ID ties.
                    order=np.argsort(-scores,kind='stable');chosen=[int(i) for i in order if scores[i]>0][:5]
                if len(chosen)<5:
                    n_fallback+=1
                    for item_index in pop_order:
                        if int(item_index) not in chosen:chosen.append(int(item_index));n_filled+=1
                        if len(chosen)==5:break
                independent_ids=ids[chosen].tolist();all_lists_same &= independent_ids==pred[split][user]
            rerank_same[split]=bool(all_lists_same)
            fallback_same &= row['fallback'][split]=={'users_needing_fallback':n_fallback,'fallback_slots':n_filled}
            summary=row['multiplier_summary'][split]
            summary_same &= all(abs(summary[k]-v)<1e-12 for k,v in [('min_multiplier',float(multiplier[positions].min())),('median_multiplier',float(np.median(multiplier[positions]))),('max_multiplier',float(multiplier[positions].max()))])
            if gamma==0:gamma0[split]=pred[split]==base[split]
        csv_path=folder/row['submission_file'];payload=csv_path.read_bytes()
        with csv_path.open(newline='') as handle:reader=csv.DictReader(handle);header=reader.fieldnames;csv_rows=list(reader)
        csv_valid=header==['user_id','tv_show_id'] and [r['user_id'] for r in csv_rows]==users and all(r['tv_show_id']==' '.join(map(str,pred['test'][r['user_id']])) for r in csv_rows)
        sha=hashlib.sha256(payload).hexdigest();csv_valid &= sha==row['csv_validation']['sha256']
        checks.append({'key':key,'independent_map':independent_map,'reported_map':row['map_at_5'],'map_abs_difference':abs(independent_map-row['map_at_5']),'independent_precision':precise,'independent_recall':recall,'diagnostic_difference_max':max(abs(precise-row['precision_at_5']),abs(recall-row['recall_at_5'])),'csv_valid':bool(csv_valid),'csv_sha256':sha,'ranked_lists_valid':bool(valid),'independent_formula_and_reranking_matches':rerank_same,'multiplier_summary_matches':bool(summary_same),'fallback_counts_match':bool(fallback_same),'gamma_zero_base_lists_equal':gamma0,'all1260_zero_empty_map':float(sum(numeric_ap,Fraction(0))/len(users))})
    ok=len(checks)==6 and all(x['map_abs_difference']<1e-12 and x['diagnostic_difference_max']<1e-12 and x['csv_valid'] and x['ranked_lists_valid'] and all(x['independent_formula_and_reranking_matches'].values()) and x['multiplier_summary_matches'] and x['fallback_counts_match'] and all(x['gamma_zero_base_lists_equal'].values()) for x in checks)
    supply_hashes={s:hashlib.sha256(a.tobytes()).hexdigest() for s,a in supplies.items()}
    hash_match=supply_hashes==result['provenance']['supply_vector_sha256']
    selected=max(checks,key=lambda x:x['independent_map'])['key'];overall=json.loads((root/'evidence/experiments.json').read_text())['validation_selected']
    final={'all_pass':bool(ok and hash_match),'scope':'No root metric/model/ranking helper imported. Counter ground truth, Fraction AP, independently reconstructed airing-count supply and cached-score re-ranking; no refit.','records':checks,'supply_sha256_match':hash_match,'supply_counts':{s:int(a.sum()) for s,a in supplies.items()},'extension_best':selected,'primary_validation_selected':overall['key'],'primary_selected_map':overall['map_at_5'],'all_extension_settings_below_primary':all(x['independent_map']<overall['map_at_5'] for x in checks),'limitations':['Original history/PPR fits are reused as frozen inputs, not retrained.','Private Kaggle scores/hidden labels are not inputs to this audit.']}
    out.mkdir(parents=True,exist_ok=True);(out/'extension_audit.json').write_text(json.dumps(final,indent=2)+'\n');print(json.dumps(final,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True,type=Path);parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent);args=parser.parse_args();main(args.root.resolve(),args.output.resolve())
