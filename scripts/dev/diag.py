import sys, json, numpy as np, fl
cfg=json.loads(sys.argv[1])
r=fl.run(cfg)
mal=set(r['mal'])
print('final',{k:round(v,3) for k,v in r['final'].items()})
for t,h in enumerate(r['trace_hist']):
    if t%4: continue
    a=np.array(h['a']);v=np.array(h['v']);rep=np.array(h['rep']);za=np.array(h['za']);zv=np.array(h['zv'])
    mi=np.array([i in mal for i in range(len(a))])
    f=lambda x:(round(float(x[mi].mean()),2) if mi.any() else None, round(float(x[~mi].mean()),2), round(float(x[~mi].min()),2), round(float(x[~mi].max()),2))
    print(t+1,'a',f(a),'v',f(v),'rep',f(rep),'za',f(za),*[(k,f(np.array(h[k]))) for k in h if k.startswith('z_')])
print('malweight',[round(w['mal_weight'],3) for w in r['weights']][::4])
