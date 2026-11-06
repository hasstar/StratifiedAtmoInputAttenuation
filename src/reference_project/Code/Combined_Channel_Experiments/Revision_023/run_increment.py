"""Create Revision023 from frozen timestamp/quality metadata only, once."""
import hashlib
import json
import msvcrt
import os
import platform
import sys
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pyarrow
import pyarrow.parquet as pq

from sampling_support import deduplicate, physical_quality_mask, utc_milliseconds, within_hour_offsets

ROOT = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent
OUT = ROOT/'Outputs/Experiments/Combined_Channel_Revision_023'
OLD = ROOT/'Outputs/Experiments/Combined_Channel_Revision_022'
DATASET = ROOT/'Outputs/Datasets/Revision_002'
RUN = Path('E:/0_Workflow_System/01_PhD/Data/Further analysis of the data results/思源楼高轨卫星信标信号实测数据_再分析结果/Runs/Run_011')
QUALITY = RUN/'metrics/quality_code/quality_code.parquet'
COLUMNS = ['hour_utc','measurement_records_raw','measurement_unique_timestamps',
    'measurement_duplicate_records','physical_valid_records','physical_invalid_records',
    'quality_zero_records','quality_nonzero_records','quality_bit16_records','quality_bit32_records',
    'quality_bits16_and32_records','measurement_coverage_fraction',
    'measurement_first_time_bjt','measurement_last_time_bjt']


def now():
    return datetime.now(timezone.utc).isoformat()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def emit(name, obj):
    with (OUT/name).open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def source_guard():
    guards = read(OLD/'sources_before.json')
    oldfinal = read(OLD/'final_verification.json')
    assert sha(OLD/'delivery_manifest.json') == oldfinal['manifest_sha256']
    for p, info in read(OLD/'delivery_manifest.json')['files'].items():
        assert sha(ROOT/p) == info['sha256'], p
        guards[str(ROOT/p)] = info['sha256']
    for p in [OLD/'delivery_manifest.json', OLD/'final_verification.json']:
        guards[str(p)] = sha(p)
    for p in [DATASET/n for n in ['measurement_hourly.parquet','run_manifest.json',
            'source_snapshot.json','validation_report.json','QA.md']]+[
            ROOT/'Code/Dataset_Build/Revision_002/build_matched_dataset.py',
            RUN/'run_manifest.json',RUN/'validation_report.json',RUN/'independent_verification.json',
            RUN.parent.parent/'Code/Revision_011/analyze_uniform_100ms.py']:
        guards[str(p)] = sha(p)
    for item in read(DATASET/'source_snapshot.json')['direct_inputs']:
        if item['path'].endswith(('quality_code.parquet','peak_power_dbm.parquet')):
            assert sha(item['path']) == item['sha256'].lower(), item['path']
            guards[item['path']] = item['sha256'].lower()
    assert sha(QUALITY) == 'cde9efccdf9b375f6dc1c050c2d71b00720f4104d9628554a4742d70750b76c9'
    assert sha(ROOT/'Code/Dataset_Build/Revision_002/build_matched_dataset.py') == 'b4463302c1b4ad8f7591e027b6a0bfc58e2f4f975c45ff5fe0664f075a07cd29'
    assert {p:sha(p) for p in guards} == guards
    return guards


def main():
    assert not OUT.exists(), 'new output directory required'
    guards = source_guard()
    OUT.mkdir()
    emit('sources_before.json', guards)
    emit('code_before.json', {str(p):sha(p) for p in sorted(CODE.glob('*')) if p.is_file()})
    (OUT/'PROTOCOL.md').open('x',encoding='utf-8').write((CODE/'PROTOCOL.md').read_text(encoding='utf-8'))
    emit('execution.json', {'utc':now(),'pid':os.getpid(),'argv':sys.argv,'python':sys.version,
        'executable':sys.executable,'platform':platform.platform(),
        'numpy':np.__version__,'pyarrow':pyarrow.__version__,
        'preflight_cim_utc':'2026-10-17T17:04:03.3726019Z','matching_workers':0,
        'data_columns_read':{'measurement_hourly':COLUMNS,'quality_code':['time','value']},
        'power_file_hashed_only':True,'measured_power_values_loaded':False})
    lockpath=OUT/'execution.lock'
    with lockpath.open('xb') as f:
        f.write(b'0')
    with lockpath.open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            z=np.load(OLD/'derived_inputs.npz',allow_pickle=False)
            keys=z['campaign_epoch_s'][z['test_member'].astype(bool)].astype(np.int64)
            assert len(keys)==687 and len(np.unique(keys))==687 and np.all(np.diff(keys)>0)
            meta={int(r['hour_utc'].timestamp()):r for r in pq.read_table(
                DATASET/'measurement_hourly.parquet',columns=COLUMNS).to_pylist()}
            assert all(int(k) in meta for k in keys)
            for k in keys:
                m=meta[int(k)]
                assert m['physical_invalid_records']==0
                assert m['physical_valid_records']==m['measurement_unique_timestamps']
            pf=pq.ParquetFile(QUALITY)
            assert str(pf.schema_arrow.field('time').type)=='timestamp[ms]'
            assert str(pf.schema_arrow.field('value').type)=='uint16'
            localstarts=(keys+28800)*1000
            selected=[]
            for g in range(pf.num_row_groups):
                st=pf.metadata.row_group(g).column(0).statistics
                assert st is not None and st.has_min_max and st.null_count==0
                lo=int(np.datetime64(st.min,'ms').astype(np.int64))
                hi=int(np.datetime64(st.max,'ms').astype(np.int64))
                if np.any((localstarts<=hi)&(localstarts+3600000>lo)):
                    selected.append(g)
            times={int(k):[] for k in keys}
            quals={int(k):[] for k in keys}
            selected_rows=0
            for b in pf.iter_batches(batch_size=262144,row_groups=selected,columns=['time','value']):
                assert b.column(0).null_count==b.column(1).null_count==0
                selected_rows+=len(b)
                t=utc_milliseconds(b.column(0).to_numpy().astype('datetime64[ms]').astype(np.int64))
                q=b.column(1).to_numpy()
                h=(t//3600000)*3600
                pos=np.searchsorted(keys,h)
                ok=(pos<len(keys)) & (keys[np.minimum(pos,len(keys)-1)]==h)
                for k in np.unique(h[ok]):
                    m=ok & (h==k)
                    times[int(k)].append(t[m])
                    quals[int(k)].append(q[m])
            offsets=[]; qualities=[]; indptr=[0]; ledger=[]; hist=Counter()
            for k in keys:
                k=int(k); rawt=np.concatenate(times[k]); rawq=np.concatenate(quals[k])
                t,q=deduplicate(rawt,rawq)
                valid=physical_quality_mask(q)
                assert valid.all(), 'new quality-invalid sample: do not change target'
                d=within_hour_offsets(t,k)
                m=meta[k]
                stats={'measurement_records_raw':len(rawt),'measurement_unique_timestamps':len(t),
                    'measurement_duplicate_records':len(rawt)-len(t),'physical_valid_records':int(valid.sum()),
                    'physical_invalid_records':int((~valid).sum()),'quality_zero_records':int((q==0).sum()),
                    'quality_nonzero_records':int((q!=0).sum()),'quality_bit16_records':int(((q&16)!=0).sum()),
                    'quality_bit32_records':int(((q&32)!=0).sum()),'quality_bits16_and32_records':int(((q&48)==48).sum())}
                assert all(v==m[n] for n,v in stats.items()), (k,stats,m)
                assert abs(len(t)/36000-m['measurement_coverage_fraction'])<2e-15
                epoch=datetime(1970,1,1,tzinfo=timezone.utc)
                first=int((m['measurement_first_time_bjt']-epoch)//timedelta(milliseconds=1))
                last=int((m['measurement_last_time_bjt']-epoch)//timedelta(milliseconds=1))
                assert int(t[0])==first and int(t[-1])==last, (k,first,last)
                steps=np.diff(t); values,counts=np.unique(steps,return_counts=True)
                hist.update({int(v):int(c) for v,c in zip(values,counts)})
                ledger.append({'hour_epoch_s':k,'hour_utc':datetime.fromtimestamp(k,timezone.utc).isoformat(),
                    **stats,'coverage_fraction':len(t)/36000,'first_offset_ms':int(d[0]),
                    'last_offset_ms':int(d[-1]),'on_100ms_grid_records':int((d%100==0).sum()),
                    'steps_not_100ms':int((steps!=100).sum()),'steps_over_1s':int((steps>1000).sum()),
                    'largest_timestamp_step_ms':int(steps.max()),
                    'first_moment_offset_ms':int(d.astype(np.int64).sum())})
                offsets.append(d); qualities.append(q); indptr.append(indptr[-1]+len(d))
                del times[k],quals[k]
            with (OUT/'sampling_support.npz').open('xb') as f:
                np.savez_compressed(f,hour_epoch_s=keys,indptr=np.array(indptr,dtype=np.int64),
                    offset_ms=np.concatenate(offsets),quality=np.concatenate(qualities))
            emit('hourly_sampling_ledger.json',ledger)
            emit('timestamp_step_histogram.json',{'unit':'ms','within_hour_only':True,
                'counts':{str(k):v for k,v in sorted(hist.items())}})
            summary={'status':'COMPUTED_PENDING_INDEPENDENT_QA','hours':len(keys),
                'unique_valid_timestamps':indptr[-1],
                'raw_test_records':sum(r['measurement_records_raw'] for r in ledger),
                'duplicate_test_records':sum(r['measurement_duplicate_records'] for r in ledger),
                'physical_invalid_records':0,'min_records':min(np.diff(indptr)).item(),
                'max_records':max(np.diff(indptr)).item(),'nominal_record_deficit':687*36000-indptr[-1],
                'on_100ms_grid_records':sum(r['on_100ms_grid_records'] for r in ledger),
                'steps_not_100ms':sum(r['steps_not_100ms'] for r in ledger),
                'steps_over_1s':sum(r['steps_over_1s'] for r in ledger),
                'largest_timestamp_step_ms':max(r['largest_timestamp_step_ms'] for r in ledger),
                'quality_zero_records':sum(r['quality_zero_records'] for r in ledger),
                'quality_nonzero_records_retained':sum(r['quality_nonzero_records'] for r in ledger),
                'row_groups_total':pf.num_row_groups,'row_groups_selected':len(selected),
                'selected_row_group_rows_read':selected_rows,'source_rows_total':pf.metadata.num_rows,
                'raw_timestamp_unit':'ms','raw_timezone':'naive Asia/Shanghai UTC+08:00',
                'output_offsets_unit':'exact ms since UTC hour; no rounding',
                'sampling_window_position':'NOT_IDENTIFIED_FROM_METADATA',
                'nominal_deficit_is_unobserved_duration':False,'power_finiteness_rechecked':False,
                'power_finiteness_evidence':'inherited frozen Dataset002 physical_invalid=0 and matching all-hour metadata',
                'measured_power_values_loaded':False,'new_predictions':0,'new_scores':0,
                'mts_status':'NOT_COMPUTED_USER_DECISION'}
            emit('sampling_summary.json',summary)
            after={p:sha(p) for p in guards}
            assert after==guards
            emit('sources_after.json',after)
            print(json.dumps(summary,ensure_ascii=False),flush=True)
        finally:
            lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
    emit('execution_complete.json',{'utc':now(),'status':'PASS_PRODUCER_ONLY',
        'protected_sources':len(guards),'changed_sources':0,'execution_lock':'RELEASED'})


if __name__=='__main__':
    main()
