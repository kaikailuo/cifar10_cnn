"""Time shift inference on TRAINING images only with smoke weights, never test data."""
import sys,time,json
from pathlib import Path
P=Path(__file__).resolve().parent; ROOT=P.parents[1]; sys.path.insert(0,str(ROOT))
import torch,numpy as np
from src.data import _load_batch,RAW_DIR,DeviceLoader
from src.models import build_model
from src.train import autocast_for
from src.analysis import translate

torch.set_num_threads(2); torch.backends.cudnn.benchmark=True
x,y=_load_batch(RAW_DIR/'data_batch_1'); dev=torch.device('cuda'); loader=DeviceLoader(x,y,1000,dev); rows=[]
for group,name in [('b1_ablation','b1-0-base__s0'),('b1_ablation','b1-1-flipcrop__s0'),('b2_shift','blur-none__s0'),('b2_shift','blur-flipcrop__s0')]:
 d=ROOT/'results_smoke'/group/name; cfg=json.loads((d/'config.json').read_text())
 def one(dy,dx):
  m=build_model(cfg).to(dev,memory_format=torch.channels_last); m.load_state_dict(torch.load(d/'model.pt',map_location=dev)); m.eval()
  with torch.no_grad():
   for xb,yb in loader:
    xb=translate(xb,dy,dx)
    with autocast_for(dev): z=m(xb.contiguous(memory_format=torch.channels_last))
    z.float().cpu()
 one(1,1); torch.cuda.synchronize(); t=time.perf_counter()
 for dy,dx in [(1,2),(-3,1),(4,4)]: one(dy,dx)
 torch.cuda.synchronize(); sec=time.perf_counter()-t
 rows.append(dict(group=group,name=name,norm=cfg['norm'],samples_per_shift=10000,shifts=3,wall_s=sec,estimated_81_shifts_s=sec/3*81)); print(rows[-1],flush=True)
P.joinpath('analysis_probe.json').write_text(json.dumps(rows,indent=2))
