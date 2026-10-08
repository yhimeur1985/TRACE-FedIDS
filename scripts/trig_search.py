import fl, numpy as np, json
from model import predict
out={}
for ds in ['cic','edge']:
    cfg=dict(fl.DEFAULT,dataset=ds,lr=1e-3,batch=128)
    rng=np.random.default_rng(0)
    Xtr,ytr,gtr,Xte,yte,gte,cl,feats=fl.load(cfg,rng)
    m=fl.build_model(cfg,feats,len(cl)); th=m.init(np.random.default_rng(1))
    th=th+fl.local_train(m,th,Xtr,ytr,cfg,rng,steps=2000)
    att=yte!=0; p=predict(m,th,Xte[att]).argmax(1); det=p!=0
    cand=[j for j in range(Xtr.shape[1]) if len(np.unique(Xtr[:5000,j]))>=10]
    res=[]
    for seed in range(40):
        tf=np.random.default_rng(seed).choice(cand,3,replace=False)
        for q in ['min','max']:
            trig=[(int(j),float(Xtr[:,j].min() if q=='min' else Xtr[:,j].max())) for j in tf]
            pt=predict(m,th,fl.apply_trigger(Xte[att][det],trig)).argmax(1)
            res.append((float((pt==0).mean()),seed,q,[str(feats[j]) for j in tf]))
    res.sort()
    print(ds,res[:6])
    out[ds]=res[:10]
json.dump(out,open('results/trigger_search.json','w'),indent=1)
