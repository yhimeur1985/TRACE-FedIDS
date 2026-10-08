import json,glob,sys,fl
tag,ds,agg,at=sys.argv[1:5]
for f in glob.glob('results/runs/main_*.json'):
    r=json.load(open(f)); c=r['cfg']
    if c['dataset']==ds and c['agg']==agg and c['attack']==at and c['seed']==0:
        cfg={k:v for k,v in c.items() if k not in('categories','env_clients')}
        new=fl.run(cfg)
        print(tag, 'old',{k:round(v,5) for k,v in r['final'].items()}); print(tag,'new',{k:round(v,5) for k,v in new['final'].items()})
        break
