import fl, numpy as np, json
out={}
for ds in ['edge','cic']:
    for rp in [5,10,20]:
        res=[]
        for seed in [0,1,2]:
            cfg=dict(fl.DEFAULT,dataset=ds,lr=1e-3,batch=64,seed=seed,root_per_class=rp)
            rng=np.random.default_rng(seed)
            Xtr,ytr,gtr,Xte,yte,gte,cl,feats=fl.load(cfg,rng)
            root=fl.take_root(ytr,gtr,cfg,rng)
            m=fl.build_model(cfg,feats,len(cl)); th=m.init(np.random.default_rng(seed+100))
            th=th+fl.local_train(m,th,Xtr[root],ytr[root],cfg,rng,steps=800)
            res.append(fl.evaluate(m,th,Xte,yte,len(cl),None)['f1'])
        out[f'{ds}_{rp}']=res; print(ds,rp,np.round(res,3),np.mean(res),flush=True)
json.dump(out,open('results/root_only.json','w'))
