"""Two diagnostic epochs on full train/val, isolated from official runs; training batches only read."""
import sys,os,json,time,subprocess
from pathlib import Path
P=Path(__file__).resolve().parent; ROOT=P.parents[1]; sys.path.insert(0,str(ROOT))
if len(sys.argv)>1:
 import numpy as np,torch
 from src import train
 from src.data import _load_batch,RAW_DIR
 torch.set_num_threads(2)
 batches=[_load_batch(RAW_DIR/f'data_batch_{i}') for i in range(1,6)]
 x=np.concatenate([a for a,b in batches]); y=np.concatenate([b for a,b in batches])
 train.load_cifar10=lambda:(x,y,None,None)
 train.train_run(Path(sys.argv[1]),torch.device('cuda'))
else:
 from src.config import make_config
 specs=[('none',{}),('bn',dict(norm='bn')),('gn',dict(norm='gn')),('ln',dict(norm='ln')),('wide',dict(norm='gn',widths=[128,256,512],batch_size=256,augment='flipcrop',cutout=16)),('res56',dict(model='resnet',depth=56,norm='gn',batch_size=256,augment='flipcrop',cutout=16))]
 processes=[]; t=time.perf_counter()
 for name,spec in specs:
  d=P/'full_epoch'/name; d.mkdir(parents=True,exist_ok=True); (d/'config.json').write_text(json.dumps(make_config(**spec,epochs=2,warmup_epochs=1)))
  log=(d/'train.log').open('w'); processes.append(subprocess.Popen([sys.executable,__file__,str(d)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1'}))
 codes=[p.wait() for p in processes]; assert codes==[0]*6,codes
 (P/'full_epoch_wall.json').write_text(json.dumps(dict(wall_s=time.perf_counter()-t,codes=codes),indent=2))
 print('full-data two-epoch probe passed',time.perf_counter()-t)
