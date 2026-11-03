"""Read-only formal audit. CPU arithmetic/weight inspection only; never calls inference or training.
Run from repository root with PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python .../check.py.
Writes only inside this new audit directory; refuses to overwrite its outputs.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['MPLCONFIGDIR']=str(__import__('pathlib').Path(__file__).resolve().parent/'mpl_cache')
import csv, hashlib, json, math, pickle, re, subprocess, sys, zlib
from pathlib import Path
from datetime import datetime, timezone, timedelta
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT)); torch.set_num_threads(2)
from src import experiments as ex
from src.models import build_model, count_params
ctx=ex.Context(ROOT/'results',False)
issues=[]; checks=0

def check(ok, msg):
    global checks
    checks+=1
    if not ok: issues.append(msg)
def close(a,b,tol=2e-6):
    if a is None or b is None: return a is b
    return bool(np.isclose(a,b,atol=tol,rtol=tol,equal_nan=True))
def read(p): return json.loads(p.read_text())
def digest(p,alg='sha256'): return hashlib.new(alg,p.read_bytes()).hexdigest()
def dump(name,obj):
    with (OUT/name).open('x') as f: json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def tracked_files():
    return sorted(set([*ROOT.joinpath('src').glob('*.py'),*ROOT.joinpath('data').rglob('*'),*ROOT.joinpath('results').rglob('*'),*ROOT.joinpath('report').rglob('*'),*ROOT.joinpath('preparation').rglob('*')])-{p for p in ROOT.joinpath('preparation').rglob('__pycache__/*')})
protected=[p for p in tracked_files() if p.is_file() and '__pycache__' not in p.parts]
before={str(p.relative_to(ROOT)):digest(p) for p in protected}
dump('protected_before_sha256.json',before)
(OUT/'git_status_before.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True))
# Load labels directly, independently of production split resolver. No image/model inference.
raw=ROOT/'data/raw/cifar10/cifar-10-batches-py'
def labels(p):
    with p.open('rb') as f: return np.asarray(pickle.load(f,encoding='bytes')[b'labels'],dtype=np.int64)
y=np.concatenate([labels(raw/f'data_batch_{i}') for i in range(1,6)]); ty=labels(raw/'test_batch')
with np.load(ROOT/'data/splits/cifar10_split_v1.npz') as a: tr=a['train_indices']; va=a['val_indices']
with np.load(ROOT/'data/splits/cifar10_kfold5_v1.npz') as a: folds=a['fold_ids']
check(len(y)==50000 and len(ty)==10000,'raw counts')
check(len(tr)==45000 and len(va)==5000 and np.array_equal(np.sort(np.r_[tr,va]),np.arange(50000)),'holdout disjoint complete')
for name,idx,n in [('train',tr,4500),('val',va,500)]: check(np.array_equal(np.bincount(y[idx],minlength=10),[n]*10),name+' stratification')
for f in range(5): check(np.array_equal(np.bincount(y[folds==f],minlength=10),[1000]*10),f'fold {f} stratification')
rng=np.random.default_rng(20261102); fresh=np.empty(50000,dtype=int)
for c in range(10):
    idx=np.flatnonzero(y==c); rng.shuffle(idx); fresh[idx]=np.arange(len(idx))%5
check(np.array_equal(fresh,folds),'independent fold recreation')
label_sha=hashlib.sha256(y.astype(np.uint8).tobytes()).hexdigest()
split_sha=hashlib.sha256(tr.tobytes()+va.tobytes()).hexdigest()
for p in (ROOT/'data/splits').glob('*.json'): check(read(p)['label_sha256']==label_sha,str(p)+' label hash')
check(split_sha==read(ROOT/'data/splits/cifar10_split_v1.json')['split_sha256'],'holdout logical split hash')
data_record=read(ROOT/'preparation/20261102_2256/data_check.json')
md5={str(p.relative_to(ROOT)):digest(p,'md5') for p in raw.iterdir() if p.is_file()}
check(md5==data_record['files'],'all 8 raw md5 match preparation')
frozen=read(ROOT/'preparation/20261102_2256/frozen_inputs_sha256.json')
freeze_check={p:digest(ROOT/p)==v for p,v in frozen.items()}; check(all(freeze_check.values()),'freeze mismatch')
# Actual source need only be called to enumerate definitions; no prepare/run/train/evaluator calls.
expected={}; group_counts={}; phases={}; records={}; val_checks=[]; stats={}; all_nonfinite=[]
model_count=val_count=0
for group,(phase,definition) in ex.GROUPS.items():
    configs=definition(ctx); group_counts[group]=len(configs); phases[phase]=phases.get(phase,0)+len(configs)
    for name,cfg in configs.items(): expected[f'{group}/{name}']=cfg
actual={str(p.parent.relative_to(ROOT/'results')) for p in (ROOT/'results').glob('*/*/config.json')}
check(actual==set(expected),'actual task set != derived expected set')
for key,cfg in expected.items():
    d=ROOT/'results'/key
    for f in ['config.json','summary.json','history.csv','train.log']: check((d/f).is_file(),key+'/'+f+' missing')
    got=read(d/'config.json'); s=read(d/'summary.json'); check(got==cfg,key+' config mismatch')
    check(cfg['train_limit'] is None and cfg['val_limit'] is None,key+' smoke cap')
    with (d/'history.csv').open() as f: h=[{k:float(v) for k,v in row.items() if v!=''} for row in csv.DictReader(f)]
    check([r['epoch'] for r in h]==list(range(1,len(h)+1)),key+' epoch order')
    check(s['epochs_run']==len(h),key+' summary epoch count')
    check(s['diverged'] or len(h)==cfg['epochs'],key+' truncated nondiverged')
    check(s['diverged']==(not np.isfinite(h[-1]['train_loss'])),key+' divergence label')
    check(close(sum(r['epoch_time'] for r in h),s['train_time_s']),key+' train time')
    for field in ['train_eval_acc','train_eval_loss','val_acc','val_loss']:
        check(close(s['final_'+field],h[-1].get(field)),key+' final '+field)
    for r in h:
        for k,v in r.items():
            if not np.isfinite(v): all_nonfinite.append({'run':key,'epoch':int(r['epoch']),'field':k})
            if 'acc' in k: check(0<=v<=1,key+' accuracy out of bounds')
    idx=va if cfg['split']=='holdout' else np.flatnonzero(folds==cfg['fold']) if cfg['split']=='fold' else None
    ntr=45000 if cfg['split']=='holdout' else 40000 if cfg['split']=='fold' else 50000
    steps=ntr//cfg['batch_size']; warm=cfg['warmup_epochs']*steps; total=cfg['epochs']*steps
    for r in h:
        step=int(r['epoch'])*steps
        factor=(step+1)/warm if step<warm else .5*(1+math.cos(math.pi*min(1,(step-warm)/max(1,total-warm))))
        check(close(r['lr'],cfg['lr']*factor,1e-12),key+' scheduler mismatch')
    log=(d/'train.log').read_text(); logrows=[json.loads(l) for l in log.splitlines() if l.startswith('{"epoch":')]
    check(len(logrows)==len(h),key+' log count mismatch')
    for a,b in zip(logrows,h):
        check(set(a)==set(b) and all(close(a[k],b[k],1e-12) for k in a),key+' log/history mismatch')
    check(not re.search(r'Traceback|FAILED|resumed at|out of memory|RuntimeError',log,re.I),key+' engineering failure/resume found')
    check(not (d/'checkpoint.pt').exists(),key+' residual checkpoint')
    needmodel=not s['diverged']; needval=needmodel and idx is not None
    check((d/'model.pt').exists()==needmodel,key+' model presence')
    check((d/'val_logits.npz').exists()==needval,key+' val presence')
    if needmodel:
        model=build_model(cfg); state=torch.load(d/'model.pt',map_location='cpu',weights_only=True)
        check(set(state)==set(model.state_dict()) and all(state[k].shape==v.shape for k,v in model.state_dict().items()),key+' state keys/shape')
        check(all(torch.isfinite(v).all().item() for v in state.values()),key+' nonfinite model')
        check(count_params(model)==s['params'],key+' param count')
        model_count+=1; del state,model
    if needval:
        with np.load(d/'val_logits.npz') as a:
            z=a['logits'].astype(np.float64); lab=a['labels']; indices=a['indices']
        check(z.shape==(len(idx),10) and np.isfinite(z).all(),key+' val shape/finite')
        check(np.array_equal(idx,indices) and np.array_equal(lab,y[idx]),key+' val order/labels')
        zz=z-z.max(1,keepdims=True); nll=float(np.mean(np.log(np.exp(zz).sum(1))-zz[np.arange(len(lab)),lab])); acc=float(np.mean(z.argmax(1)==lab))
        check(close(acc,s['final_val_acc'],1e-7) and close(nll,s['final_val_loss']),key+' val logits metrics')
        best=max(h,key=lambda r:r['val_acc'])
        check(close(best['val_acc'],s['best_val_acc']) and best['epoch']==s['best_epoch'],key+' best val fields')
        check(close(s['gap'],s['final_train_eval_acc']-s['final_val_acc']),key+' gap')
        val_checks.append({'run':key,'samples':len(idx),'acc':acc,'nll':nll,'acc_error':abs(acc-s['final_val_acc']),'nll_error':abs(nll-s['final_val_loss'])}); val_count+=1
    records[key]={'config':cfg,'summary':s,'history':h,'n_train':ntr,'n_val':0 if idx is None else len(idx),'steps_per_epoch':steps,'discarded_per_epoch':ntr%cfg['batch_size']}
# Independent selection from saved records, not helper ranking outputs.
search=sorted([(r['summary']['final_val_acc'],key.split('/')[1].split('__')[0]) for key,r in records.items() if key.startswith('e3_search/') and not r['summary']['diverged']],reverse=True)
shortlist=[v for _,v in search[:4]]
cv=[]
for v in shortlist:
    rr=[records[f'e3_cv/{v}__f{f}'] for f in range(5)]
    vals=[r['summary']['final_val_acc'] for r in rr]
    cv.append({'variant':v,'hp':{k:rr[0]['config'][k] for k in ex.TUNED},'fold_val_acc':vals,'mean':float(np.mean(vals)),'std':float(np.std(vals,ddof=1))})
cv.sort(key=lambda r:-r['mean']); winner=cv[0]['variant']; analysis=read(ROOT/'results/analysis/analysis.json')
check(analysis['final']['cv']==cv and analysis['final']['winner']==winner,'independent CV ranking')
# Independently recompute all stored test scores and both ensembles; never evaluate a model.
def prob(z,t=1):
    a=z/t; a=a-a.max(1,keepdims=True); a=np.exp(a); return a/a.sum(1,keepdims=True)
def metrics(p,lab=ty):
    return {'acc':float((p.argmax(1)==lab).mean()),'nll':float(-np.log(np.maximum(p[np.arange(len(lab)),lab],1e-12)).mean())}
def bins(p,lab=ty):
    c=p.max(1); good=p.argmax(1)==lab; rows=[]; ece=0.
    for i in range(15):
        lo=i/15; hi=(i+1)/15; m=(c>lo)&(c<=hi)
        if m.any():
            a=float(good[m].mean()); cf=float(c[m].mean()); ece+=m.mean()*abs(a-cf)
            rows.append({'lo':lo,'hi':hi,'count':int(m.sum()),'acc':a,'conf':cf})
    return float(ece),rows
def match(a,b,label,tol=3e-6):
    if isinstance(a,dict):
        check(set(a)==set(b),label+' keys')
        for k in a: match(a[k],b[k],label+'/'+k,tol)
    elif isinstance(a,list):
        check(len(a)==len(b),label+' len')
        for i,(x,v) in enumerate(zip(a,b)): match(x,v,label+f'/{i}',tol)
    elif isinstance(a,(int,float)): check(close(a,b,tol),label+f': {a} != {b}')
    else: check(a==b,label)
tests={}
expected_test={f'e3_final/{winner}__full',*[f'e3_cv/{winner}__f{i}' for i in range(5)],*[k for k in records if k.startswith(('b1_ablation/','b2_shift/'))]}
actual_test={str(p.parent.relative_to(ROOT/'results')) for p in (ROOT/'results').glob('*/*/test_logits.npz')}
check(actual_test==expected_test,'test cache task set')
for key in sorted(expected_test):
    with np.load(ROOT/'results'/key/'test_logits.npz') as a:
        check(a.files==['logits'],key+' cache keys'); z=a['logits'].copy()
    check(z.shape==(10000,10) and z.dtype==np.float32 and np.isfinite(z).all(),key+' test finite shape dtype')
    tests[key]=z
recomputed={'final':{'winner':winner,'cv':cv}}
final=recomputed['final']; final['final_test']=metrics(prob(tests[f'e3_final/{winner}__full']))
foldprobs=[prob(tests[f'e3_cv/{winner}__f{i}']) for i in range(5)]
final['fold_test']=[metrics(p) for p in foldprobs]; final['cv_ensemble_test']=metrics(np.mean(foldprobs,axis=0))
match(final,analysis['final'],'final')
b1keys=[k for k in records if k.startswith('b1_ablation/')]
b1rows={k.split('/')[1]:{'val_acc':records[k]['summary']['final_val_acc'],'params':records[k]['summary']['params'],'test':metrics(prob(tests[k]))} for k in b1keys}
top3=sorted(b1rows,key=lambda n:-b1rows[n]['val_acc'])[:3]
recomputed['b1']={'runs':b1rows,'ensemble_members':top3,'ensemble_test':metrics(np.mean([prob(tests['b1_ablation/'+n]) for n in top3],axis=0))}
match(recomputed['b1'],analysis['b1'],'b1')
cal={}; cal_validation={}
for key in b1keys:
    name=key.split('/')[1]
    with np.load(ROOT/'results'/key/'val_logits.npz') as a: z=a['logits']; lab=a['labels']
    grid=np.exp(np.linspace(np.log(.05),np.log(20.),600))
    losses=[metrics(prob(z,t),lab)['nll'] for t in grid]; bi=int(np.argmin(losses)); t=float(grid[bi])
    before=prob(tests[key]); after=prob(tests[key],t)
    ec,bb=bins(before); et,bt=bins(after)
    cal[name]={'temperature':t,'acc':metrics(before)['acc'],'nll':metrics(before)['nll'],'ece':ec,'bins':bb,'nll_ts':metrics(after)['nll'],'ece_ts':et,'bins_ts':bt,'mean_conf':float(before.max(1).mean())}
    check(sum(b['count'] for b in bb)==10000 and sum(b['count'] for b in bt)==10000,name+' ECE coverage')
    cal_validation[name]={'grid_index':bi,'boundary':bi in (0,599),'val_nll_raw':metrics(prob(z),lab)['nll'],'val_nll_ts':losses[bi],'test_acc_ts':metrics(after)['acc']}
recomputed['calibration']=cal; match(cal,analysis['calibration'],'calibration')
# Shift predictions were not persisted. Only arithmetic/clean caches and expected bounds are verifiable.
shift_internal={}
for label,key in [('max-none','b1_ablation/b1-0-base__s0'),('max-flipcrop','b1_ablation/b1-1-flipcrop__s0'),('blur-none','b2_shift/blur-none__s0'),('blur-flipcrop','b2_shift/blur-flipcrop__s0')]:
    s=analysis['shift'][label]; weights=[1,8,16,24,32]; mean=sum(weights[int(m)]*v for m,v in s['acc_by_magnitude'].items())/81
    clean=metrics(prob(tests[key]))['acc']; check(close(clean,s['clean_acc'],1e-12),label+' shift clean')
    check(close(mean,s['mean_shifted_acc'],1e-12),label+' shift mean includes clean weighting')
    check(all(0<=v<=1 for v in [s['consistency'],*s['acc_by_magnitude'].values()]),label+' shift bounds')
    shift_internal[label]={'weighted_81_mean':mean,'mean_80_nonzero_shifts':(81*mean-clean)/80,'clean_from_cache':clean,'raw_shift_predictions_available':False}
# Replay only report rendering into audit directory, then compare each table and PDF body.
from src import report
report.ROOT=OUT/'rendered'; report.build(ctx)
table_compare={}; pdf_compare={}
for p in (ROOT/'report/tables').glob('*.tex'):
    q=OUT/'rendered/report/tables'/p.name; table_compare[p.name]=p.read_bytes()==q.read_bytes(); check(table_compare[p.name],p.name+' table mismatch')
def strip_dates(b): return re.sub(rb'/CreationDate \(D:[^)]*\)',b'/CreationDate (NORMALIZED)',b)
for p in (ROOT/'report/figures').glob('*.pdf'):
    q=OUT/'rendered/report/figures'/p.name
    equal=strip_dates(p.read_bytes())==strip_dates(q.read_bytes())
    streams=re.findall(rb'stream\r?\n(.*?)\r?\nendstream',p.read_bytes(),re.S)
    decoded=[]
    for s in streams:
        try: decoded.append(zlib.decompress(s))
        except zlib.error: decoded.append(s)
    pdf_compare[p.name]={'equal_except_creation_date':equal,'bytes':p.stat().st_size,'decoded_streams':len(decoded),'valid_header':p.read_bytes().startswith(b'%PDF-'),'eof':p.read_bytes().rstrip().endswith(b'%%EOF')}
    check(equal,p.name+' PDF mismatch')
# formal log must describe each expected task exactly once.
logpath=ROOT/'preparation/formal_logs/run_20261102_231850.log'; log=logpath.read_text()
completed=re.findall(r'^\[\d+/\d+\] (\S+) ok ',log,re.M)
check(len(completed)==len(expected) and set(completed)==set(expected),'formal log task coverage')
check(log.count('analysis written to')==1 and log.count('tables and figures written to')==1,'analysis/report log count')
# Group/variant statistics and curve summaries for reporting.
for key,r in records.items():
    group,name=key.split('/'); variant=name.split('__')[0]; stats.setdefault(group,{}).setdefault(variant,[]).append(r)
variant_stats={}
for group,vs in stats.items():
    variant_stats[group]={}
    for v,rs in vs.items():
        agg={'n':len(rs),'seeds':[r['config']['seed'] for r in rs]}
        for field in ['final_train_eval_acc','final_val_acc','gap','final_val_loss']:
            vals=[r['summary'].get(field) for r in rs if r['summary'].get(field) is not None]
            agg[field]={'mean':float(np.mean(vals)),'sample_std':float(np.std(vals,ddof=1)) if len(vals)>1 else None} if vals else None
        agg['runs']=[{'run':f'{group}/{v}'+ ('__f'+str(r['config']['fold']) if r['config']['split']=='fold' else '__full' if r['config']['split']=='full' else '__s'+str(r['config']['seed'])),'best_epoch':r['summary'].get('best_epoch'),'best_val_acc':r['summary'].get('best_val_acc'),'final_val_acc':r['summary']['final_val_acc'],'first_train_loss':r['history'][0]['train_loss'] if np.isfinite(r['history'][0]['train_loss']) else 'NaN','max_train_loss':max([h['train_loss'] for h in r['history'] if np.isfinite(h['train_loss'])],default=None),'min_val_acc':min([h['val_acc'] for h in r['history'] if 'val_acc' in h],default=None)} for r in rs]
        variant_stats[group][v]=agg
# Verify no protected file changed during audit. New audit artifacts intentionally excluded.
after={str(p.relative_to(ROOT)):digest(p) for p in protected}; unchanged=before==after; check(unchanged,'protected files changed')
dump('protected_after_sha256.json',after)
dump('validation_recomputed.json',val_checks); dump('analysis_recomputed.json',recomputed); dump('variant_statistics.json',variant_stats)
dump('checks.json',{'checked_at_beijing':datetime.now(timezone(timedelta(hours=8))).isoformat(),'checks':checks,'issues':issues,'groups':group_counts,'phase_counts':phases,'tasks':len(expected),'planned_epochs':sum(c['epochs'] for c in expected.values()),'observed_epochs':sum(r['summary']['epochs_run'] for r in records.values()),'models_checked':model_count,'val_logits_checked':val_count,'test_logits_checked':len(tests),'diverged':[k for k,r in records.items() if r['summary']['diverged']],'nonfinite_history':all_nonfinite,'shortlist':shortlist,'winner':winner,'hp':cv[0]['hp'],'frozen_hashes':freeze_check,'raw_md5':md5,'label_sha256':label_sha,'logical_split_sha256':split_sha,'temperature_validation':cal_validation,'tables':table_compare,'pdfs':pdf_compare,'shift_internal':shift_internal,'protected_files_unchanged':unchanged,'protected_file_count':len(protected),'max_val_acc_error':max(x['acc_error'] for x in val_checks),'max_val_nll_error':max(x['nll_error'] for x in val_checks),'test_cache_limitation':'Caches contain logits only; identity/order is consistent with code and raw labels but lacks per-cache labels, indices and model digest. No inference performed.'})
print(json.dumps({'checks':checks,'issues':issues,'tasks':len(expected),'models':model_count,'val':val_count,'test':len(tests),'tables':table_compare,'pdfs':pdf_compare,'winner':winner,'unchanged':unchanged},ensure_ascii=False,indent=2))
