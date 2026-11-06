"""Synthetic correctness tests only; no local data, benchmark or timing pilot."""
import copy
import core as c


def run_tests():
    target=c.OUT/'self_test.json'
    if target.exists():
        saved=c.read(target)
        for name,digest in saved['code_sha256'].items():
            c.require(c.sha(c.HERE/name)==digest,'Code changed since correctness tests')
        return saved
    checks=[]
    for cfg in c.CONFIGS:
        for names in [('surface__test',),('p700__test',),('surface__a','p925__a','p700__b')]:
            c.set_seed()
            net=c.AtmosphericTabM(list(names),cfg)
            x=c.torch.arange(4*len(names),dtype=c.torch.float32).reshape(4,len(names))/20
            y=net(x)
            c.require(y.shape==(4,cfg['k']) and c.torch.isfinite(y).all().item(),'Shape/finite test')
            loss=y.square().mean()
            loss.backward()
            c.require(all(p.grad is None or c.torch.isfinite(p.grad).all().item() for p in net.parameters()),'Gradient test')
            net.eval()
            a=net(x).detach()
            c.set_seed()
            second=c.AtmosphericTabM(list(names),cfg).eval()
            c.require(c.torch.equal(a,second(x).detach()),'Deterministic construction')
            checks.append(dict(config_id=cfg['id'],features=len(names),status='PASS'))
    members=c.torch.tensor([[-1.,1.]])
    c.require(float(members.square().mean())==1. and float(members.mean(1).square().mean())==0.,'Mean of losses negative control')
    synthetic=c.pd.DataFrame(dict(hour_bjt=c.pd.date_range('2001-01-01',periods=3),surface__a=[1.,3.,float('nan')],p700__b=[2.,4.,6.],clear_sky_attenuation_db=[0.,1.,2.]))
    names=['surface__a','p700__b']
    x,prep=c.fit_preprocess(synthetic,names)
    previous=copy.deepcopy(prep)
    altered=synthetic.copy(); altered['surface__a']=9999.
    c.transform(altered,names,prep)
    c.require(prep==previous and prep['median']==[2.,4.],'Train-only transform test')
    from verify_search import check_preprocess
    check_preprocess(prep,synthetic,names)
    result=dict(status='PASS',research_rows_used=0,timing_pilot=False,optimizer_steps=0,
                synthetic_shape_backward_determinism_checks=checks,
                member_loss_and_train_only_preprocessing='PASS',utc=c.now(),
                code_sha256={name:c.sha(c.HERE/name) for name in c.SOURCE_NAMES})
    c.emit(target,result)
    return result
