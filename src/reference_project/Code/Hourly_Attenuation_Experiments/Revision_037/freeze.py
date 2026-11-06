"""Read-only historical checkpoint audit; isolated immutable transfer manifest."""
from __future__ import annotations
import hashlib, json, os, msvcrt, time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent
OUT = ROOT/'Outputs/Experiments/Hourly_Attenuation_Revision_037_UK_Frozen_RFE'
DELIVERY = ROOT/'Outputs/Experiments/Hourly_Attenuation_Revision_034_Final_Delivery/delivery_002'
R32 = ROOT/'Outputs/Experiments/Hourly_Attenuation_Revision_032_Multi_Model_RFE'
MODELS = ['tabm','bai_lstm','bilstm','transformer','mlp','catboost','gcn','random_forest','svr_rbf','elastic_net']

def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def now(): return datetime.now(timezone.utc).isoformat()
def rel(p): return str(Path(p).resolve().relative_to(ROOT))
def need(x,m):
    if not x: raise RuntimeError(m)
def emit(p,x):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f: json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)

def main():
    started=now(); tic=time.monotonic()
    OUT.mkdir(exist_ok=False)
    with (OUT/'execution.lock').open('a+b') as lock:
        lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            guard={}
            def pin(p,expected=None):
                p=Path(p); digest=sha(p)
                need(expected is None or digest==expected,'SHA mismatch '+str(p))
                guard[rel(p)]=digest; return digest
            pin(DELIVERY/'evidence/FINAL_QA.json','33bc8db491146d40278b585fd3bda9508176207daf58767f51630e7f1625da51')
            need(read(DELIVERY/'evidence/FINAL_QA.json')['status']=='PASS','Final QA not PASS')
            roles_file=DELIVERY/'tables/model_roles_6240_with_sha.csv';pin(roles_file)
            roles=pd.read_csv(roles_file); need(len(roles)==6240,'6240 model roles')
            for i,row in roles.iterrows():
                p=ROOT/row.model_file;pin(p,row.model_sha256)
                need(p.stat().st_size==row.bytes,'Model bytes mismatch')
                if i%1000==0: print('checkpoint_hashes_verified',i,flush=True)
            old=read(R32/'manifest.json');pin(R32/'manifest.json')
            for p,digest in old['source_sha256'].items():pin(ROOT/p,digest)
            states=[]; union=set()
            for model in MODELS:
                path=DELIVERY/'paths'/f'{model}_156_to_1.json';pin(path)
                steps=read(path);need(len(steps)==156,'Complete frozen RFE path')
                prev=None
                for step in [s for s in steps if s['k']<=10]:
                    k=int(step['k']); names=step['retained'];need(len(names)==k,'Feature length')
                    if prev is not None:
                        need(k==prev['k']-1 and names==[n for n in prev['retained'] if n!=prev['removed_next']],'Nested independent path')
                    prev=step; union.update(names)
                    folder=R32/'models'/model/f'k_{k:04d}'
                    state_path=folder/'state.json';state=read(state_path);qa=read(folder/'qa.json')
                    state_sha=pin(state_path,qa['state_sha256']);pin(folder/'qa.json')
                    need(qa['status']=='PASS' and state['feature_names']==names,'State/path QA')
                    full=state['full_fit']; need(full['fold']==0 and full['seed']==17,'Full LOCAL role identity')
                    row=roles[(roles.model==model)&(roles.k==k)&(roles.fold==0)]
                    need(len(row)==1 and row.iloc[0].model_sha256==full['model_sha256'],'Delivery role identity')
                    states.append(dict(model=model,k=k,feature_names=names,removed_next=step['removed_next'],
                        model_file=full['model_file'],model_sha256=full['model_sha256'],format=full['format'],
                        state_file=rel(state_path),state_sha256=state_sha,config=state['config'],seed=17,
                        fit_hour_sha256=full['train_hour_sha256']))
            need(len(states)==100 and len(union)==38,'100 states / 38 unique fields')
            data_path=ROOT/'Outputs/Experiments/Combined_Channel_Revision_010/expanded_dataset_with_itu_inputs.parquet'
            weather_path=ROOT/'Outputs/Experiments/Combined_Channel_Revision_008/weather_labels_all.parquet'
            pin(data_path);pin(weather_path)
            data=pd.read_parquet(data_path);weather=pd.read_parquet(weather_path)
            keys=['station','utc_epoch_seconds'];need(len(data)==5664 and not data.duplicated(keys).any(),'UK key cardinality')
            weather=data[keys].merge(weather,on=keys,validate='one_to_one',how='left')
            need(weather.event.notna().all(),'Weather missing')
            need(np.array_equal(data.primary_score_eligible,weather.primary_score_eligible),'Frozen eligibility')
            need(int(data.primary_score_eligible.sum())==5593,'5593 scoring hours')
            p=weather.total_water_mm; s=weather.snow_water_mm
            rain=(p>=.1)&(s<.1); over=(p<.1)&(s<.1)&(weather.tcc>=.8)
            need(np.array_equal(rain,weather.event=='rain'),'Rain rule')
            need(np.array_equal(over,weather.event=='dry_overcast'),'Overcast rule')
            available=sorted(union.intersection(data.columns)); missing=sorted(union-set(available))
            need(np.isfinite(data[available].to_numpy(float)).all(),'Present UK input nonfinite')
            for st in states:
                st['missing_uk_fields']=sorted(set(st['feature_names'])-set(available))
                st['input_gate']='READY' if not st['missing_uk_fields'] else 'MISSING_INPUTS'
            # Reuse the previous physical raw-source audit as evidence, not as a new download.
            inventory=ROOT/'Outputs/Experiments/Combined_Channel_Revision_025/ncar_record_inventory.json';pin(inventory)
            raw_pairs=sorted({(str(q['level_hpa']),q['variable']) for q in read(inventory)})
            for p in ['cds_headers.json','out_of_scope_assets.json','final_verification.json']:pin(inventory.parent/p)
            for stop in (ROOT/'Outputs/Experiments/Combined_Channel_Revision_001/external').rglob('STOP_REQUESTED*'):
                if stop.is_file():pin(stop)
            for file in sorted(CODE.iterdir()):
                if file.is_file():pin(file)
            rows=[]
            for f in sorted(union):
                users=sorted({s['model'] for s in states if f in s['feature_names']})
                rows.append(dict(feature=f,models=users,available_all_5664=f in available,
                                 missing_station_hour_values=5664 if f not in available else 0))
            emit(OUT/'frozen_states_100.json',states)
            emit(OUT/'required_fields_38.json',rows)
            emit(OUT/'protected_sha256.json',guard)
            summary=dict(status='FROZEN_PASS_INPUT_COMPLETION_REQUIRED',started_utc=started,completed_utc=now(),
                model_role_hashes_checked=6240,full_local_models_frozen=1560,uk_required_states=100,
                uk_hours=5664,eligible_hours=5593,rain_hours=int((rain&data.primary_score_eligible).sum()),
                dry_overcast_hours=int((over&data.primary_score_eligible).sum()),
                required_unique_fields=38,available_fields=available,missing_fields=missing,
                ready_states=sum(s['input_gate']=='READY' for s in states),
                cached_NCAR_parameter_level_pairs=raw_pairs,protected_files=len(guard),
                training_calls=0,model_inference_calls=0,new_downloads=0,
                targeted_acquisition_authorized_by_user=True,old_queues_resumed=False,
                elapsed_seconds=time.monotonic()-tic)
            emit(OUT/'freeze_qa.json',summary)
            print(json.dumps({k:v for k,v in summary.items() if k not in ['available_fields','missing_fields','cached_NCAR_parameter_level_pairs']},ensure_ascii=False),flush=True)
        except Exception as e:
            emit(OUT/'freeze_failure.json',dict(utc=now(),error=repr(e)));raise
        finally:
            lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)

if __name__=='__main__':main()
