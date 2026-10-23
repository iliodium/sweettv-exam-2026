"""Frozen temporal SWEET.TV evaluation; no validation viewing enters fitting."""
from __future__ import annotations
from pathlib import Path
import argparse,csv,json,time,hashlib,platform,sys
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize
from .models import PopularityModel,ContentModel,ImplicitALSModel,CoViewPPRModel
from .metrics import evaluate_rankings,validate_ranked_lists

def jdefault(x):
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,np.ndarray):return x.tolist()
 raise TypeError(type(x).__name__)
def save(path,value):path.write_text(json.dumps(value,default=jdefault,ensure_ascii=False,indent=2)+'\n')
def rank_rows(scores,ids,pop,cold,k=5):
 """Positive model evidence first, then one identical TRAIN-popularity filler."""
 if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(pop)):raise ValueError('Non-finite scores cannot silently fall back')
 out=[];fallback_users=0;filled=0;pop_order=np.lexsort((ids,-pop))
 for row,is_cold in zip(scores,cold):
  if is_cold:chosen=[]
  else:
   order=np.lexsort((ids,-row));chosen=order[row[order]>0][:k].tolist()
  if len(chosen)<k:
   fallback_users+=1
   for ix in pop_order:
    if int(ix) not in chosen:chosen.append(int(ix));filled+=1
    if len(chosen)==k:break
  out.append([int(ids[x]) for x in chosen])
 return out,{'users_needing_fallback':fallback_users,'fallback_slots':filled}
def rrf_component(scores,ids,cold,rank_constant=10,depth=100):
 """Convert positive ranking evidence to RRF contributions; zeros get no credit."""
 if not np.all(np.isfinite(scores)):raise ValueError('Non-finite RRF input')
 out=np.zeros_like(scores,dtype=np.float32)
 for i,row in enumerate(scores):
  if cold[i]:continue
  order=np.lexsort((ids,-row));order=order[row[order]>0][:depth]
  out[i,order]=1/(rank_constant+np.arange(1,len(order)+1))
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);args=ap.parse_args();R=args.root
 art=R/'artifacts';ev=R/'evidence';outdir=R/'submission';art.mkdir(exist_ok=True);t0=time.perf_counter()
 metadata=json.loads((R/'data/processed/metadata.json').read_text());item_ids=np.array([m['item_id'] for m in metadata],dtype=np.int64);imap={int(x):i for i,x in enumerate(item_ids)}
 events=pd.read_parquet(R/'data/processed/events.parquet');events['user_id']=events.user_id.astype(str)
 with (R/'data/raw/submission.csv').open() as f:target_users=[r['user_id'] for r in csv.DictReader(f)]
 users=sorted(set(events.loc[events.split=='train','user_id']).union(target_users));umap={u:i for i,u in enumerate(users)};ui=np.array([umap[u] for u in target_users],dtype=np.int64)
 train=events.loc[events.split=='train'].copy();val=events.loc[events.split=='val'].copy();shape=(len(users),len(metadata))
 tr=np.array([umap[u] for u in train.user_id]);ti=np.array([imap[int(x)] for x in train.item_id]);ones=np.ones(len(train),dtype=np.float32)
 X=sparse.coo_matrix((ones,(tr,ti)),shape=shape,dtype=np.float32).tocsr();warm=np.diff(X.indptr)>0;cold=~warm[ui]
 counts=val.groupby(['user_id','item_id'],sort=False).size().rename('frequency').reset_index();counts=counts.sort_values(['user_id','frequency','item_id'],ascending=[True,False,True],kind='stable')
 labels=counts.groupby('user_id',sort=False).head(5);truth={str(u):[int(x) for x in g.item_id] for u,g in labels.groupby('user_id',sort=False)}
 cands=json.loads((R/'data/processed/candidates.json').read_text());cid={s:np.array(cands[s],dtype=np.int64) for s in ['val','test']};ci={s:np.array([imap[int(x)] for x in cid[s]],dtype=np.int64) for s in cid}
 popularity=np.asarray(X.getnnz(axis=0)).ravel().astype(np.float32);pop={s:popularity[ci[s]] for s in cid}
 evaluated=[u for u in target_users if truth.get(u)];degree=np.asarray(X.getnnz(axis=1)).ravel();eligible=[u for u in evaluated if warm[umap[u]]]
 # Fix two illustrative users solely by TRAIN history size before scoring any method.
 degrees=np.array([degree[umap[u]] for u in eligible]);selected={}
 for label,q in [('light',.1),('heavy',.9)]:
  point=float(np.quantile(degrees,q));u=min(eligible,key=lambda u:(abs(degree[umap[u]]-point),int(u)))
  selected[label]={'user_id':u,'train_distinct_programmes':int(degree[umap[u]]),'train_positive_airings':int(X[umap[u]].sum()),'quantile':q,'quantile_target':point}
 save(ev/'selected_users.json',{'selection_before_model_scoring':True,'rule':'closest TRAIN distinct-programme degree to template evaluable warm-user 10th/90th percentile; numeric user ID breaks ties','users':selected})
 labels.to_parquet(art/'validation_labels.parquet',index=False);sparse.save_npz(art/'train_matrix.npz',X);save(art/'indices.json',{'user_ids':users,'item_ids':item_ids,'target_user_ids':target_users,'validation_candidates':cid['val'],'test_candidates':cid['test']})
 save(ev/'evaluation_protocol.json',{'train_cutoff':'2020-06-29T00:00:00','validation_end':'2020-07-27T00:00:00','final_models_refit_on_validation':False,'ground_truth':'top5 programme IDs by qualifying distinct airing frequency; ties by ID','coverage_threshold':.8,'repeats_allowed':True,'candidate_policy':'positive-ID airings whose START is within target period; sources0/1 for VAL, source2 for TEST; labels require fully contained airings','common_fallback':'TRAIN distinct-viewer popularity; cold users identical; fill remaining slots after positive model evidence','template_users':len(target_users),'evaluated_users':len(evaluated),'excluded_no_validation_relevance':len(target_users)-len(evaluated),'cold_train_template_users':int(cold.sum()),'cold_train_evaluated_users':sum(not warm[umap[u]] for u in evaluated),'train_shape':shape,'train_pairs':X.nnz,'train_positive_airings':len(train),'validation_positive_airings':len(val),'training_observed_items':int((X.getnnz(axis=0)>0).sum()),'validation_top5_relevant_pairs':len(labels[labels.user_id.isin(target_users)]),'validation_top5_cold_item_pairs':int(sum(popularity[imap[int(item)]]==0 for item in labels.loc[labels.user_id.isin(target_users),'item_id'])),'candidates':{s:len(cid[s]) for s in cid}})
 records=[];best={};all_predictions={}
 def add(key,family,scores,config,seconds,diag=None):
  predictions={};fills={};split_scores={s:np.asarray(scores[s] if isinstance(scores,dict) else scores[:,ci[s]],dtype=np.float32) for s in ci}
  for split in ['val','test']:
   rows,fills[split]=rank_rows(split_scores[split],cid[split],pop[split],cold)
   predictions[split]=dict(zip(target_users,rows));validate_ranked_lists(target_users,predictions[split],cid[split])
  report=evaluate_rankings(truth,predictions['val'],target_users,k=5)
  row={'key':key,'family':family,'map_at_5':report['map'],'precision_at_5':report['precision'],'recall_at_5':report['recall'],'evaluated_users':report['evaluated_users'],'excluded_no_relevant':report['excluded_no_relevant_count'],'map_all_template_empty_zero':report['map']*report['evaluated_users']/len(target_users),'seconds_fit_and_score':seconds,'config':config,'diagnostics':diag or {},'fallback':fills}
  records.append(row);all_predictions[key]=predictions
  save(art/(key+'_predictions.json'),predictions);save(art/(key+'_validation.json'),report)
  if family not in best or row['map_at_5']>best[family]['record']['map_at_5']:
   best[family]={'record':row,'scores':split_scores,'predictions':predictions}
   if isinstance(scores,dict):np.savez_compressed(art/('best_'+family+'_scores.npz'),**scores)
   else:np.save(art/('best_'+family+'_scores.npy'),scores)
  save(ev/'experiment_progress.json',{'completed':records,'running':None});print(json.dumps({k:row[k] for k in ['key','map_at_5','seconds_fit_and_score']},default=jdefault),flush=True)
 def run_model(key,family,model,matrix):
  t=time.perf_counter();model.fit(matrix,metadata);scores=model.score(ui,np.arange(len(metadata),dtype=np.int64));diag=dict(model.diagnostics_)
  if hasattr(model,'last_convergence_'):
   cv=model.last_convergence_;diag['convergence']={'users':len(cv),'all_converged':all(r['converged'] for r in cv),'max_iterations':max((r['iterations'] for r in cv),default=0),'max_residual':max((r['residual'] for r in cv),default=0),'max_mass_error':max((abs(r['mass']-1) for r in cv),default=0)}
   save(art/(key+'_convergence.json'),cv)
   if not diag['convergence']['all_converged']:raise RuntimeError('Nonconverged PPR is not evaluated as a final stationary ranking')
  add(key,family,scores,model.get_config(),time.perf_counter()-t,diag)
 # Reference and repeated-viewing baselines, both evaluated individually.
 run_model('popularity_users','popularity',PopularityModel('users'),X)
 run_model('popularity_events','popularity',PopularityModel('weight'),X)
 cutoff=pd.Timestamp('2020-06-29');ages=(cutoff-pd.to_datetime(train.airing_end)).dt.total_seconds().to_numpy()/86400
 matrices={0:X}
 for half in [14,30,60]:matrices[half]=sparse.coo_matrix((np.exp2(-ages/half).astype(np.float32),(tr,ti)),shape=shape).tocsr()
 for half,matrix in matrices.items():
  t=time.perf_counter();scores=matrix[ui].toarray();add('history_'+str(half),'history',scores,{'method':'personal repeat-frequency baseline','half_life_days':half or None,'repeat_policy':'allowed'},time.perf_counter()-t)
 # Training-only channel context multiplied by known target-period programme supply.
 channel_t=time.perf_counter();context=pd.read_parquet(R/'data/processed/train_channel_context.parquet');supply=pd.read_parquet(R/'data/processed/target_channel_supply.parquet');channels=sorted(set(context.channel_id).union(supply.channel_id));chmap={int(c):i for i,c in enumerate(channels)}
 context=context.loc[context.user_id.astype(str).isin(umap)]
 uc=sparse.coo_matrix((np.maximum(context.view_seconds.to_numpy(dtype=np.float64),0),([umap[str(u)] for u in context.user_id],[chmap[int(c)] for c in context.channel_id])),shape=(len(users),len(channels))).tocsr();uc=normalize(uc,norm='l1',axis=1)
 channel_scores=np.zeros((len(ui),len(metadata)),dtype=np.float32)
 channel_split={}
 for split in ['val','test']:
  ds=supply.loc[supply.split==split];cs=sparse.coo_matrix((ds.airing_count.astype(float),([chmap[int(c)] for c in ds.channel_id],[imap[int(i)] for i in ds.item_id])),shape=(len(channels),len(metadata))).tocsr();cs=normalize(cs,norm='l1',axis=1)
  channel_split[split]=np.asarray((uc[ui]@cs).toarray(),dtype=np.float32)
 # Availability differs by split, so record separate vectors without blending future schedules.
 temp=channel_scores.copy();temp[:,ci['val']]=channel_split['val'][:,ci['val']]
 add('channel_schedule','channel',{s:channel_split[s][:,ci[s]] for s in ci},{'method':'TRAIN channel-view-seconds profile x target schedule airing-frequency','unknown_id_airtime':'used as TRAIN channel context, never a programme ID'},time.perf_counter()-channel_t)
 for half in [0,30]:
  for name,kwargs in [('balanced',{}),('titles',{'title_weight':3,'category_weight':1,'genre_weight':1,'year_weight':0})]:
   run_model(f'content_{name}_{half}','content',ContentModel(**kwargs),matrices[half])
 for f,reg,alpha in [(32,.1,10),(64,.1,10),(64,.05,40),(96,.1,40)]:
  run_model(f'als_f{f}_r{reg}_a{alpha}','als',ImplicitALSModel(factors=f,regularization=reg,alpha=alpha,num_threads=4,iterations=30),X)
 for a in [.2,.5,.85]:run_model(f'ppr_a{a}','ppr',CoViewPPRModel(alpha=a,restart_transform='identity'),X)
 # Family-best hyperparameters are selected exclusively on the same validation cohort.
 families=['history','content','als','ppr','channel'];rrfs={s:{f:rrf_component(best[f]['scores'][s],cid[s],cold,10,100) for f in families} for s in ['val','test']}
 fusion_rows=[]
 for other in ['content','als','ppr','channel']:
  for hw in [.25,.5,.75,.9]:
   weights={'history':hw,other:1-hw};fusion_rows.append(weights)
 for hw,pw,cw in [(.5,.3,.2),(.7,.2,.1),(.4,.4,.2)]:fusion_rows.append({'history':hw,'ppr':pw,'channel':cw})
 best_fusion=None
 for num,weights in enumerate(fusion_rows):
  preds={};scores_by_split={};fill={}
  for split in ['val','test']:
   scores=sum(w*rrfs[split][f] for f,w in weights.items());scores_by_split[split]=scores
   ranked,fill[split]=rank_rows(scores,cid[split],pop[split],cold);preds[split]=dict(zip(target_users,ranked));validate_ranked_lists(target_users,preds[split],cid[split])
  rep=evaluate_rankings(truth,preds['val'],target_users,k=5);key=f'rrf_{num:02d}'
  row={'key':key,'family':'hybrid_rrf','map_at_5':rep['map'],'precision_at_5':rep['precision'],'recall_at_5':rep['recall'],'evaluated_users':rep['evaluated_users'],'excluded_no_relevant':rep['excluded_no_relevant_count'],'map_all_template_empty_zero':rep['map']*rep['evaluated_users']/len(target_users),'weights':weights,'rank_constant':10,'depth':100,'components':{f:best[f]['record']['key'] for f in weights},'fallback':fill}
  records.append(row);all_predictions[key]=preds;save(art/(key+'_predictions.json'),preds);save(art/(key+'_validation.json'),rep)
  if best_fusion is None or row['map_at_5']>best_fusion['record']['map_at_5']:best_fusion={'record':row,'predictions':preds,'scores':scores_by_split}
 best['hybrid_rrf']=best_fusion
 chosen=max([b['record'] for b in best.values()],key=lambda row:row['map_at_5'])
 # Store method comparison and all proposed candidates; no Kaggle submission here.
 for family,b in best.items():
  name=b['record']['key'];rows=b['predictions']['test'];path=outdir/(name+'.csv')
  with path.open('w',newline='') as f:
   writer=csv.writer(f);writer.writerow(['user_id','tv_show_id']);writer.writerows((u,' '.join(map(str,rows[u]))) for u in target_users)
  b['record']['submission_file']=str(path.relative_to(R));b['record']['submission_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
 save(ev/'experiments.json',{'records':records,'family_best':{f:b['record'] for f,b in best.items()},'validation_selected':chosen,'selected_users':selected,'total_seconds':time.perf_counter()-t0,'environment':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__}})
 save(art/'family_predictions.json',{f:b['predictions'] for f,b in best.items()})
 # Human-readable examples preserve full training/held-out evidence separately from hidden TEST predictions.
 examples=[];md={int(m['item_id']):m for m in metadata}
 for label,info in selected.items():
  u=info['user_id'];ix=target_users.index(u);hist=X[umap[u]];order=np.lexsort((item_ids[hist.indices],-hist.data))
  history=[{'item_id':int(item_ids[hist.indices[j]]),'count':float(hist.data[j]),'title':md[int(item_ids[hist.indices[j]])]['title'],'category':md[int(item_ids[hist.indices[j]])]['category']} for j in order[:10]]
  for family,b in best.items():
   for split in ['val','test']:
    ids=b['predictions'][split][u]
    examples.append({'profile':label,'user_id':u,'family':family,'model':b['record']['key'],'split':split,'training_top10':history,'validation_actual_top5':truth.get(u,[]),'recommendations':[{'rank':r+1,'item_id':i,'title':md[i]['title'],'category':md[i]['category'],'score':float(b['scores'][split][ix,np.searchsorted(cid[split],i)]),'train_distinct_viewers':int(popularity[imap[i]]),'used_fallback':bool(cold[ix] or b['scores'][split][ix,np.searchsorted(cid[split],i)]<=0),'in_validation_top5':i in truth.get(u,[]) if split=='val' else None} for r,i in enumerate(ids)]})
 save(ev/'recommendation_examples.json',examples)
 print(json.dumps({'validation_selected':chosen,'family_best':{f:b['record']['map_at_5'] for f,b in best.items()},'seconds':time.perf_counter()-t0},default=jdefault),flush=True)
if __name__=='__main__':main()
