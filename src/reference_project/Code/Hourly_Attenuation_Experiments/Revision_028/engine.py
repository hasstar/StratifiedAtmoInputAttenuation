"""Weighted direct fitting: fitting rows only; holdout only scores/checkpoints."""
from __future__ import annotations
import copy
import math
import time
import warnings
import numpy as np
import torch
from torch import nn
import models as legacy

class GatedProfileNet(legacy.VerticalProfileNet):
    def __init__(self, kind, positions, dropout):
        super().__init__(kind, positions, dropout)
        self.fusion_gate = nn.Linear(32,32)
        nn.init.zeros_(self.fusion_gate.bias)

    def forward(self,x):
        base = self.base(x[:,:188])
        n = len(self.active_positions)
        if n:
            h = self.project(x[:,188:].reshape(-1,n,16))
            h = h + self.absolute_pressure_embedding[list(self.active_positions)]
            if self.kind == 'transformer':
                vertical = self.encoder(h).mean(dim=1)
            elif self.kind == 'gcn':
                h = torch.relu(self.gcn1(torch.matmul(self.adjacency,h)))
                vertical = torch.relu(self.gcn2(torch.matmul(self.adjacency,h))).mean(dim=1)
            else:
                _,(last,_) = self.lstm(h)
                vertical = self.lstm_projection(torch.cat((last[-2],last[-1]),dim=1))
            vertical = torch.sigmoid(self.fusion_gate(base))*vertical
        else:
            vertical = torch.zeros_like(base)
        return self.head(torch.cat((base,vertical),dim=1)).flatten()

def weights(frame,policy):
    n = len(frame)
    if policy == 'uniform':
        return np.ones(n)
    rain = frame.rain.to_numpy(bool)
    cloud = frame.dry_overcast.to_numpy(bool)
    if not rain.any() or not cloud.any():
        raise RuntimeError('Requested objective has an empty fitting weather group')
    return (1+n*rain/rain.sum()+n*cloud/cloud.sum())/3

def objective(frame,pred):
    error = np.asarray(pred)-frame.clear_sky_attenuation_db.to_numpy(float)
    return float(np.sqrt(np.mean([np.mean(error**2),np.mean(error[frame.rain.to_numpy(bool)]**2),
                                 np.mean(error[frame.dry_overcast.to_numpy(bool)]**2)])))

def grids(name):
    original = legacy.candidate_configs(name)
    result=[]
    for extended in (False,True):
        for base in original:
            cfg = dict(base)
            extra = {}
            if name in legacy.NEURAL_MODELS:
                cfg['epochs']=160
                if extended:
                    cfg['dropout'] += .1
                    cfg['weight_decay'] *= 10
            elif extended:
                if name == 'elastic_net': cfg['alpha'] *= 10
                elif name == 'svr_rbf': cfg['C'] /= 10
                elif name == 'random_forest': cfg['min_samples_leaf']=max(5,cfg['min_samples_leaf'])
                elif name == 'xgboost': extra={'reg_lambda':10.,'min_child_weight':5.,'colsample_bytree':.6}
                elif name == 'lightgbm': extra={'reg_lambda':10.,'min_child_samples':40}
                elif name == 'catboost': extra={'l2_leaf_reg':10.}
            result.append({'estimator':cfg,'override':extra,'weight_policy':'balanced' if extended else 'uniform',
                           'gated_fusion':bool(extended and name in ('bilstm','transformer','gcn'))})
    return result

def round_key(name):
    if name in legacy.NEURAL_MODELS:return 'epochs'
    return {'random_forest':'n_estimators','xgboost':'n_estimators','lightgbm':'n_estimators','catboost':'iterations'}.get(name)

def predict_at(model,name,x,round_number=None):
    if round_number is None or name in legacy.NEURAL_MODELS or name in ('elastic_net','svr_rbf'):
        return np.asarray(model.predict(x),float)
    if name == 'xgboost':return np.asarray(model.predict(x,iteration_range=(0,round_number)),float)
    if name == 'lightgbm':return np.asarray(model.predict(x,num_iteration=round_number),float)
    if name == 'catboost':return np.asarray(model.predict(x,ntree_end=round_number),float)
    return np.mean([t.predict(x) for t in model.estimators_[:round_number]],axis=0)

def curve_row(number,train,pred,val=None,val_pred=None):
    error=np.asarray(pred)-train.clear_sky_attenuation_db.to_numpy(float)
    row={'round':number,'train_rmse_db':float(np.sqrt(np.mean(error**2))),
         'train_objective_db':objective(train,pred), 'validation_objective_db':None, 'validation_rmse_db':None}
    for group in ('rain','dry_overcast'):
        row[f'train_{group}_rmse_db']=float(np.sqrt(np.mean(error[train[group].to_numpy(bool)]**2)))
    if val is not None:
        e=np.asarray(val_pred)-val.clear_sky_attenuation_db.to_numpy(float)
        row.update(validation_objective_db=objective(val,val_pred),validation_rmse_db=float(np.sqrt(np.mean(e**2))))
        for group in ('rain','dry_overcast'):
            row[f'validation_{group}_rmse_db']=float(np.sqrt(np.mean(e[val[group].to_numpy(bool)]**2)))
    return row

def fit(name,choice,seed,positions,xt,train,xv=None,val=None,fixed_rounds=None):
    if (xv is None)!=(val is None):raise RuntimeError('Holdout mismatch')
    cfg=copy.deepcopy(choice['estimator'])
    key=round_key(name)
    if fixed_rounds is not None:cfg[key]=int(fixed_rounds)
    y=train.clear_sky_attenuation_db.to_numpy(float)
    w=weights(train,choice['weight_policy'])
    start=time.perf_counter()
    best_pred,best_train,best_state=None,None,None
    best_score=math.inf
    selected=None
    history=[]
    reason='solver convergence'
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter('always')
        if name in legacy.NEURAL_MODELS:
            torch.set_num_threads(2)
            torch.manual_seed(seed)
            model=legacy.TorchRegressor(name,cfg,seed,xt.shape[1],positions)
            model.net=legacy.FlatMLP(xt.shape[1],cfg['dropout']) if name=='mlp' else \
                (GatedProfileNet if choice['gated_fusion'] else legacy.VerticalProfileNet)(name,positions,cfg['dropout'])
            model.target_mean=float(np.mean(y));model.target_scale=max(float(np.std(y)),1e-6)
            xx=torch.as_tensor(np.asarray(xt,np.float32))
            yy=torch.as_tensor(np.asarray((y-model.target_mean)/model.target_scale,np.float32))
            ww=torch.as_tensor(np.asarray(w,np.float32))
            optimizer=torch.optim.AdamW(model.net.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
            generator=torch.Generator().manual_seed(seed)
            stale=0
            for epoch in range(1,cfg['epochs']+1):
                model.net.train()
                order=torch.randperm(len(xx),generator=generator)
                for ix in order.split(128):
                    optimizer.zero_grad(set_to_none=True)
                    loss=torch.mean(ww[ix]*(model.net(xx[ix])-yy[ix])**2)
                    loss.backward();optimizer.step()
                model.net.eval()
                pred=model.predict(xt)
                vp=None if val is None else model.predict(xv)
                row=curve_row(epoch,train,pred,val,vp);history.append(row)
                if val is not None:
                    score=row['validation_objective_db']
                    if score < best_score-1e-12:
                        best_score=score;selected=epoch;best_pred=vp.copy();best_train=pred.copy()
                        best_state=copy.deepcopy(model.net.state_dict());stale=0
                    else:stale+=1
                    if stale>=20:
                        reason='validation objective did not improve for20 consecutive epochs';break
            if best_state is not None:model.net.load_state_dict(best_state)
            model.net.eval()
            if val is None:
                selected=len(history);reason='fixed frozen selected epochs; no holdout stopping'
            elif len(history)==cfg['epochs']:reason='reached frozen epoch cap; restored earliest best validation checkpoint'
        else:
            model=legacy.make_estimator(name,cfg,seed,xt.shape[1],positions)
            if choice['override']:model.set_params(**choice['override'])
            options={'sample_weight':w}
            if name=='catboost':options.update(verbose=False,use_best_model=False)
            model.fit(xt,y,**options)
            cap=cfg.get(key,1) if key else 1
            fit_sum=np.zeros(len(train));val_sum=None if val is None else np.zeros(len(val))
            for number in range(1,cap+1):
                if name=='random_forest':
                    fit_sum+=model.estimators_[number-1].predict(xt);pred=fit_sum/number
                    if val is not None:val_sum+=model.estimators_[number-1].predict(xv)
                    vp=None if val is None else val_sum/number
                else:
                    pred=predict_at(model,name,xt,number if key else None)
                    vp=None if val is None else predict_at(model,name,xv,number if key else None)
                row=curve_row(number if key else None,train,pred,val,vp);history.append(row)
                if val is not None and row['validation_objective_db']<best_score-1e-12:
                    best_score=row['validation_objective_db'];selected=number if key else None
                    best_pred=vp.copy();best_train=pred.copy()
            if key:
                if val is None:selected=len(history);reason='fixed frozen tree count; no holdout selection'
                else:reason='fit full preset tree cap; earliest minimum validation prefix selected afterwards'
    if best_train is None:best_train=np.asarray(model.predict(xt),float)
    if val is not None and best_pred is None:raise RuntimeError('No validation checkpoint')
    if not np.isfinite(best_train).all() or (best_pred is not None and not np.isfinite(best_pred).all()):raise RuntimeError('Nonfinite prediction')
    iterations=getattr(model,'n_iter_',None)
    if isinstance(iterations,np.ndarray):iterations=iterations.tolist()
    if isinstance(iterations,np.integer):iterations=int(iterations)
    warnings_out=[{'category':type(t.message).__name__,'message':str(t.message)} for t in warning_list]
    metadata={'trained_rounds':len(history) if key else None,'selected_round':selected,'selection_reason':reason,
              'solver_iterations':iterations,'fit_seconds':time.perf_counter()-start,'weight_min':float(w.min()),
              'weight_max':float(w.max()),'weight_mean':float(w.mean()),'warnings':warnings_out,
              'solver_converged':not any(t['category']=='ConvergenceWarning' for t in warnings_out),
              'effective_choice':dict(choice,estimator=cfg),'holdout_used_for_gradients':False}
    return model,history,best_train,best_pred,metadata
