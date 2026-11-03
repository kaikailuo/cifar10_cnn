import sys,json
from pathlib import Path
P=Path(__file__).resolve().parent; sys.path.insert(0,str(P.parents[1]))
import torch
from src import train
from src.config import make_config
p=P/'resume_probe'; p.mkdir(exist_ok=True)
p.joinpath('config.json').write_text(json.dumps(make_config(norm='bn',epochs=2,warmup_epochs=1,train_limit=512,val_limit=256)))
torch.set_num_threads(4)
original=torch.save
class Interrupted(Exception): pass
def save_then_interrupt(obj,path,*a,**kw):
 original(obj,path,*a,**kw)
 if Path(path).name=='checkpoint.pt': raise Interrupted()
train.torch.save=save_then_interrupt
try: train.train_run(p,torch.device('cuda'))
except Interrupted: pass
finally: train.torch.save=original
ck=torch.load(p/'checkpoint.pt',map_location='cuda')
try:
 torch.set_rng_state(ck['torch_rng'])
 print('old_map_location_cuda_rng_restore_unexpectedly_passed')
except Exception as exc: print('old_map_location_cuda_rng_restore_failure:',repr(exc))
r=train.train_run(p,torch.device('cuda')); assert r['epochs_run']==2 and not r['diverged']
assert not (p/'checkpoint.pt').exists(); print('GPU checkpoint resume passed')
