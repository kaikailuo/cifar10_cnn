"""Isolated train/validation-only throughput probe; no official configurations/results written."""
import os,sys,time,json,subprocess,threading
from pathlib import Path
P=Path(__file__).resolve().parent
ROOT=P.parents[1]
sys.path.insert(0,str(ROOT))

def worker(spec,out,gate,seconds,jobs):
 import torch,numpy as np
 from src.config import make_config
 from src.data import _load_batch,RAW_DIR,DeviceLoader,holdout_split
 from src.models import build_model
 from src.train import make_optimizer,autocast_for,predict
 torch.set_num_threads(max(1,16//jobs)); torch.backends.cudnn.benchmark=True
 cfg=make_config(**spec); device=torch.device('cuda')
 batches=[_load_batch(RAW_DIR/f'data_batch_{i}') for i in range(1,6)]
 x=np.concatenate([a for a,b in batches]); y=np.concatenate([b for a,b in batches]); tr,va=holdout_split(y)
 loader=DeviceLoader(x[tr],y[tr],cfg['batch_size'],device,shuffle=True,drop_last=True,augment=cfg['augment'],cutout=cfg['cutout'])
 val=DeviceLoader(x[va[:1000]],y[va[:1000]],1000,device)
 m=build_model(cfg).to(device,memory_format=torch.channels_last); opt=make_optimizer(m,cfg)
 def step(xb,yb):
  m.train(); opt.zero_grad(set_to_none=True)
  with autocast_for(device): z=m(xb.contiguous(memory_format=torch.channels_last))
  loss=torch.nn.functional.cross_entropy(z.float(),yb,label_smoothing=cfg['label_smoothing']); loss.backward(); opt.step()
  return loss
 it=iter(loader)
 for _ in range(12): step(*next(it))
 predict(m,val,device); torch.cuda.synchronize(); Path(str(out)+'.ready').touch()
 while not gate.exists(): time.sleep(.02)
 torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t=time.perf_counter(); n=0
 while time.perf_counter()-t<seconds:
  try: xb,yb=next(it)
  except StopIteration: it=iter(loader); xb,yb=next(it)
  loss=step(xb,yb); n+=len(yb); torch.cuda.synchronize()
 elapsed=time.perf_counter()-t
 torch.cuda.synchronize(); e=time.perf_counter(); z,yy=predict(m,val,device); torch.cuda.synchronize(); ev=time.perf_counter()-e
 assert torch.isfinite(loss).item() and torch.isfinite(z).all().item()
 r=dict(config=cfg,samples=n,train_s=elapsed,train_images_s=n/elapsed,eval1000_s=ev,peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mb=torch.cuda.max_memory_reserved()/2**20)
 out.write_text(json.dumps(r,indent=2))

def suite():
 cases=[('mixed',j,[dict(norm='none' if i%2==0 else 'bn') for i in range(j)]) for j in (2,4,6)]
 cases += [(name,1,[spec]) for name,spec in [
 ('gn128',dict(norm='gn')),('ln128',dict(norm='ln')),('in128',dict(norm='in')),
 ('bn16',dict(norm='bn',batch_size=16)),('gn64',dict(norm='gn',batch_size=64)),('bn256',dict(norm='bn',batch_size=256)),
 ('wide256',dict(norm='gn',widths=[128,256,512],batch_size=256,augment='flipcrop',cutout=16)),
 ('res56_256',dict(model='resnet',depth=56,norm='gn',batch_size=256,augment='flipcrop',cutout=16)),
 ('res56_bn',dict(model='resnet',depth=56,norm='bn',augment='flipcrop'))]]
 if os.environ.get('EXTRA_PROBE'):
  cases=[('none128',1,[{}]),('bn128',1,[dict(norm='bn')]),('heterogeneous',6,[{},dict(norm='bn'),dict(norm='gn'),dict(norm='ln'),dict(norm='gn',widths=[128,256,512],batch_size=256,augment='flipcrop',cutout=16),dict(model='resnet',depth=56,norm='gn',batch_size=256,augment='flipcrop',cutout=16)])]
 allrows=[]; begin=time.perf_counter()
 for name,j,specs in cases:
  if time.perf_counter()-begin>540: break
  d=P/'bench'/f'{name}_{j}'; d.mkdir(parents=True,exist_ok=True); gate=d/'go'; processes=[]; logs=[]; t=time.perf_counter()
  for i,spec in enumerate(specs):
   log=(d/f'{i}.log').open('w'); logs.append(log)
   env={**os.environ,'OMP_NUM_THREADS':str(max(1,16//j)),'MKL_NUM_THREADS':str(max(1,16//j)),'OPENBLAS_NUM_THREADS':'1'}
   processes.append(subprocess.Popen([sys.executable,__file__,'worker',json.dumps(spec),str(d/f'{i}.json'),str(gate),'12',str(j)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env))
  while not all((d/f'{i}.json.ready').exists() for i in range(j)):
   if any(p.poll() is not None for p in processes) or time.perf_counter()-t>100: break
   time.sleep(.1)
  if not all((d/f'{i}.json.ready').exists() for i in range(j)):
   for p in processes:
    if p.poll() is None: p.terminate()
   for p in processes: p.wait()
   print('FAILED',name,j,flush=True); allrows.append(dict(name=name,jobs=j,failed=True)); break
  t0=time.perf_counter(); gate.touch(); samples=[]
  while any(p.poll() is None for p in processes):
   q=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True)
   try: samples.append([float(v.strip()) for v in q.stdout.strip().split(',')])
   except ValueError: pass
   if time.perf_counter()-t0>60:
    for p in processes:
     if p.poll() is None: p.terminate()
   time.sleep(.25)
  wall=time.perf_counter()-t0; rows=[json.loads((d/f'{i}.json').read_text()) for i in range(j) if (d/f'{i}.json').exists()]
  r=dict(name=name,jobs=j,wall_s=wall,total_wall_including_startup_s=time.perf_counter()-t,workers=rows,aggregate_train_images_s=sum(v['samples'] for v in rows)/max(v['train_s'] for v in rows) if rows else 0,gpu_util_mean=sum(v[0] for v in samples)/len(samples) if samples else None,gpu_used_peak_mb=max(v[1] for v in samples) if samples else None,samples=samples)
  allrows.append(r); (P/('benchmark_extra.json' if os.environ.get('EXTRA_PROBE') else 'benchmark.json')).write_text(json.dumps(allrows,indent=2)); print(json.dumps({k:v for k,v in r.items() if k not in ('workers','samples')}),flush=True)
  for log in logs: log.close()
  if name=='mixed' and j==6 and (len(rows)<6 or r['aggregate_train_images_s']<allrows[-2]['aggregate_train_images_s']): print('6 jobs slower/failed; no further concurrency increase',flush=True)
 (P/('benchmark_extra_elapsed.txt' if os.environ.get('EXTRA_PROBE') else 'benchmark_elapsed.txt')).write_text(str(time.perf_counter()-begin))
if __name__=='__main__':
 if len(sys.argv)>1: worker(json.loads(sys.argv[2]),Path(sys.argv[3]),Path(sys.argv[4]),float(sys.argv[5]),int(sys.argv[6]))
 else: suite()
