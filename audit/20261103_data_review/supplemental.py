"""Supplemental CPU-only provenance and metric checks. No model forward calls."""
import csv,hashlib,json,pickle,re,subprocess
from pathlib import Path
from datetime import datetime,timezone,timedelta
import numpy as np
R=Path(__file__).resolve().parents[2]; O=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
raw=R/'data/raw/cifar10/cifar-10-batches-py'
records=[]
for f in [*[raw/f'data_batch_{i}' for i in range(1,6)],raw/'test_batch']:
 with f.open('rb') as h: d=pickle.load(h,encoding='bytes')
 x=np.asarray(d[b'data']); y=np.asarray(d[b'labels']); assert x.shape==(10000,3072) and x.dtype==np.uint8 and y.shape==(10000,)
 records.append((x,y))
x=np.concatenate([d[0] for d in records[:5]]); y=np.concatenate([d[1] for d in records[:5]]); tx,ty=records[5]
with np.load(R/'data/splits/cifar10_split_v1.npz') as a: tr=a['train_indices']; va=a['val_indices']
with np.load(R/'data/splits/cifar10_kfold5_v1.npz') as a: folds=a['fold_ids']
# Check exact pixel duplicates across evaluation boundaries, not near-duplicates.
image_hash=[hashlib.sha256(a.tobytes()).digest() for a in x]; test_hash={hashlib.sha256(a.tobytes()).digest() for a in tx}
tr_hash={image_hash[i] for i in tr}; va_hash={image_hash[i] for i in va}
duplicates={'exact_holdout_cross_images':len(tr_hash&va_hash),'exact_train_test_images':len(set(image_hash)&test_hash),'near_duplicates_checked':False}
rows=[]; curves=[]; test_metrics=[]
for cp in sorted((R/'results').glob('*/*/config.json')):
 d=cp.parent; cfg=read(cp); s=read(d/'summary.json')
 with (d/'history.csv').open() as f: h=list(csv.DictReader(f))
 times={p.name:datetime.fromtimestamp(p.stat().st_mtime,timezone(timedelta(hours=8))).isoformat() for p in d.iterdir() if p.is_file()}
 rows.append({'task':str(d.relative_to(R/'results')),'split':cfg['split'],'seed':cfg['seed'],'fold':cfg['fold'],'planned_epochs':cfg['epochs'],'epochs_run':s['epochs_run'],'diverged':s['diverged'],'params':s['params'],'config_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'model':(d/'model.pt').exists(),'val_logits':(d/'val_logits.npz').exists(),'test_logits':(d/'test_logits.npz').exists(),'mtime_beijing':json.dumps(times,ensure_ascii=False)})
 if not s['diverged'] and cfg['split']!='full':
  curves.append({'task':str(d.relative_to(R/'results')),'first_train_eval_acc':float(h[0]['train_eval_acc']),'first_val_acc':float(h[0]['val_acc']),'final_train_eval_acc':s['final_train_eval_acc'],'final_val_acc':s['final_val_acc'],'best_val_acc':s['best_val_acc'],'best_epoch':s['best_epoch'],'best_minus_final_pp':100*(s['best_val_acc']-s['final_val_acc'])})
 if (d/'test_logits.npz').exists():
  with np.load(d/'test_logits.npz') as a: z=a['logits'].astype(np.float64)
  z-=z.max(1,keepdims=True); logp=z-np.log(np.exp(z).sum(1,keepdims=True)); loss=-logp[np.arange(10000),ty]
  test_metrics.append({'task':str(d.relative_to(R/'results')),'nll_stable_float64':float(loss.mean()),'nll_clipped_float64':float(np.minimum(loss,-np.log(1e-12)).mean()),'true_probs_below_1e12':int((loss>-np.log(1e-12)).sum()),'max_abs_logits_centered':float(abs(z).max())})
with (O/'tasks.csv').open('x',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
curves.sort(key=lambda a:-a['best_minus_final_pp'])
current_diff=subprocess.check_output(['git','diff','--','src/run.py','src/train.py'],cwd=R,text=True)
prep_diff=(R/'preparation/20261102_2256/source_changes.diff').read_text()
start=read(R/'preparation/20261102_2256/formal_launch.json')['started_at_beijing']; finish=read(R/'preparation/20261102_2256/formal_completion_verification.json')['finished_at_beijing']
# Confirm the test-cache time range is after every training summary, without treating mtimes as immutable proof.
summary_times=[p.stat().st_mtime for p in (R/'results').glob('*/*/summary.json')]; cache_times=[p.stat().st_mtime for p in (R/'results').glob('*/*/test_logits.npz')]
smoke_caches=list((R/'results_smoke').glob('*/*/test_logits.npz'))
smoke_shapes=[]
for p in smoke_caches:
 with np.load(p) as a: smoke_shapes.append({'path':str(p.relative_to(R)),'shape':list(a['logits'].shape)})
source_status={}
for p in sorted((R/'src').glob('*.py')):
 name=str(p.relative_to(R)); original=subprocess.check_output(['git','show',f'HEAD:{name}'],cwd=R)
 source_status[name]={'equals_HEAD':original==p.read_bytes(),'sha256_now':hashlib.sha256(p.read_bytes()).hexdigest()}
result={'exact_duplicate_checks':duplicates,'fold_validation_overlap_with_search_train':[int(np.isin(np.flatnonzero(folds==i),tr).sum()) for i in range(5)],'fold_validation_overlap_with_search_val':[int(np.isin(np.flatnonzero(folds==i),va).sum()) for i in range(5)],'curves_ranked_by_best_final_difference':curves,'test_nll_crosscheck':test_metrics,'source_status':source_status,'preparation_diff_matches_current':prep_diff==current_diff,'all_test_cache_mtimes_after_training':min(cache_times)>max(summary_times),'start_beijing':start,'finish_beijing':finish,'smoke_test_cache_shapes':smoke_shapes,'source_freeze_scope_note':'Preparation freeze covers config/experiments and split files only; current audit hashes cannot establish historical immutability.'}
with (O/'supplemental.json').open('x') as f: json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps({k:v for k,v in result.items() if k not in ['curves_ranked_by_best_final_difference','source_status','smoke_test_cache_shapes']},ensure_ascii=False,indent=2))
print('SMOKE_CACHES',len(smoke_shapes),'SHAPES',sorted({tuple(a['shape']) for a in smoke_shapes}))
print('TOP_CURVES',curves[:6])
