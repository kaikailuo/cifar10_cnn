import json,sys,math,heapq,csv
from pathlib import Path
P=Path(__file__).resolve().parent; sys.path.insert(0,str(P.parents[1]))
from src.experiments import GROUPS,Context,search_space
from src.config import make_config
rows=json.loads((P/'benchmark.json').read_text()); extra=json.loads((P/'benchmark_extra.json').read_text())
solo={r['name']:r['workers'][0] for r in rows+extra if r['jobs']==1}; hetero=next(r for r in extra if r['name']=='heterogeneous')['workers']; mix=next(r for r in rows if r['name']=='mixed' and r['jobs']==6)['workers']
def rate(c):
 n=c['norm']; b=c['batch_size']; kind='wide' if c['widths'][0]==128 else c['model']
 if kind=='wide': return hetero[4]['train_images_s']*min(1,b/256)
 if kind=='resnet':
  if n=='gn': return hetero[5]['train_images_s']*min(1,b/256)*(56/c['depth'])
  return solo['res56_bn']['train_images_s']*(mix[1]['train_images_s']/solo['bn128']['train_images_s'])*(56/c['depth'])*min(2,b/128)
 base={'none':mix[0]['train_images_s'],'bn':mix[1]['train_images_s'],'gn':hetero[2]['train_images_s'],'ln':hetero[3]['train_images_s'],'in':hetero[2]['train_images_s']*solo['in128']['train_images_s']/solo['gn128']['train_images_s']}[n]
 if b==16: return base*solo['bn16']['train_images_s']/solo['bn128']['train_images_s']
 if b==64: return base*(solo['gn64']['train_images_s']/solo['gn128']['train_images_s'] if n=='gn' else .5)
 if b==256: return base*solo['bn256']['train_images_s']/solo['bn128']['train_images_s']
 return base

def cost(c):
 count={'holdout':45000,'fold':40000,'full':50000}[c['split']]; n=count//c['batch_size']*c['batch_size']
 # Evaluation, saving and Python overhead provision. Exact full-data probes are listed separately.
 return c['epochs']*(n/rate(c)*1.15+.2)+10

def schedule(costs):
 slots=[0.]*6
 for value in costs:
  a=heapq.heappop(slots); heapq.heappush(slots,a+value)
 return max(slots)
ctx=Context(P/'unused')
phase1={g:[cost(c) for c in fn(ctx).values()] for g,(phase,fn) in GROUPS.items() if phase==1}
p1=schedule([v for values in phase1.values() for v in values])
cv=[cost(make_config(**hp,epochs=30,split='fold',fold=0)) for hp in search_space()]
p2=[schedule([v for c in chosen for v in [c]*5]) for chosen in [sorted(cv)[:4],sorted(cv)[-4:]]]
p3=[]
for hp in search_space():
 specs=[make_config(**hp,epochs=30,split='full')]
 for ov in [{},{'augment':'flipcrop'},{'augment':'flipcrop','cutout':16},{'augment':'flipcrop','cutout':16,'label_smoothing':.1},{'augment':'flipcrop','cutout':16,'label_smoothing':.1,'widths':[128,256,512]},{'augment':'flipcrop','cutout':16,'label_smoothing':.1,'model':'resnet','depth':56},{'pool':'blur'},{'pool':'blur','augment':'flipcrop'}]:
  specs.append(make_config(**{**hp,'epochs':40,**ov}))
 p3.append(schedule([cost(c)*(1.3 if c['pool']=='blur' else 1) for c in specs]))
r=dict(method='Six-slot schedule using measured concurrent rates; profile interpolation for unmeasured batches/depths, 15% training allowance, 0.2s/epoch evaluation/write allowance and 10s startup/run. CV and phase3 vary over actual search space; no test scores used.',phase1_minutes=p1/60,phase2_minutes=[v/60 for v in p2],phase3_minutes=[min(p3)/60,max(p3)/60],total_minutes=[(p1+min(p2)+min(p3))/60,(p1+max(p2)+max(p3))/60],run_counts=[78,20,9],epochs=[1980,600,350],uncertainties=['CV shortlist/winner unknown','GN/LN/IN Batch16, BN64, GN256 not directly measured','ResNet/Plain20/32 depth scaling estimated from ResNet56','BlurPool overhead provision 30%, not full-data measured','mixed profiles short and thermal/CPU contention can change','tails and dropout/augmentation checkpoint overhead approximate'])
(P/'time_estimate.json').write_text(json.dumps(r,indent=2)); print(json.dumps(r,indent=2))
