"""Independent full-file scan and Python-map OR; imports no producer functions."""
import hashlib
import json
import msvcrt
import os
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'Outputs/Experiments/Combined_Channel_Revision_023'
OLD=ROOT/'Outputs/Experiments/Combined_Channel_Revision_022'
RUN=Path('E:/0_Workflow_System/01_PhD/Data/Further analysis of the data results/思源楼高轨卫星信标信号实测数据_再分析结果/Runs/Run_011')


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def main():
    assert not (OUT/'independent_qa_v002.json').exists()
    before=read(OUT/'sources_before.json')
    assert {p:sha(p) for p in before}==before
    with (OUT/'execution.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            frozen=np.load(OLD/'derived_inputs.npz',allow_pickle=False)
            expected_keys=frozen['campaign_epoch_s'][frozen['test_member'].astype(bool)].tolist()
            result=np.load(OUT/'sampling_support.npz',allow_pickle=False)
            assert result.files==['hour_epoch_s','indptr','offset_ms','quality']
            assert result['hour_epoch_s'].tolist()==expected_keys
            assert result['offset_ms'].dtype==np.dtype('uint32')
            assert result['quality'].dtype==np.dtype('uint16')
            points=result['offset_ms']; flags=result['quality']; pointers=result['indptr']
            assert pointers.shape==(688,) and pointers[0]==0
            assert int(pointers[-1])==len(points)==len(flags)
            hours=[int(k)//3600 for k in expected_keys]
            hourset=pa.array(hours,type=pa.int64())
            chunks={h:[] for h in hours}
            # Deliberately no Parquet row-group statistics or time filters.
            scan=ds.dataset(RUN/'metrics/quality_code/quality_code.parquet',format='parquet').scanner(
                columns=['time','value'],batch_size=131072,use_threads=False)
            scanned=selected=0
            for batch in scan.to_batches():
                scanned+=batch.num_rows
                assert batch.column(0).null_count==batch.column(1).null_count==0
                bjt=batch.column(0).cast(pa.int64()).to_numpy()
                bh=bjt//3600000-8
                keep=pc.is_in(pa.array(bh),value_set=hourset).to_numpy(zero_copy_only=False)
                if not keep.any():continue
                tx=bjt[keep]-28800000; qx=batch.column(1).to_numpy()[keep]; hx=bh[keep]
                selected+=len(tx)
                for h in set(hx.tolist()):
                    ok=hx==h
                    chunks[h].append((tx[ok],qx[ok]))
            assert scanned==132720993
            columns=['hour_utc','measurement_records_raw','measurement_unique_timestamps',
                'measurement_duplicate_records','physical_valid_records','physical_invalid_records',
                'quality_zero_records','quality_nonzero_records','quality_bit16_records',
                'quality_bit32_records','quality_bits16_and32_records','measurement_coverage_fraction',
                'measurement_first_time_bjt','measurement_last_time_bjt']
            meta={int(r['hour_utc'].timestamp()):r for r in pq.read_table(
                ROOT/'Outputs/Datasets/Revision_002/measurement_hourly.parquet',columns=columns).to_pylist()}
            ledger=read(OUT/'hourly_sampling_ledger.json')
            assert len(ledger)==len(expected_keys)==687
            histogram=Counter(); checked=rawtotal=duptotal=on_grid=nonzero=0; maxcov=0.0
            for i,(h,k,row) in enumerate(zip(hours,expected_keys,ledger)):
                merged={}; raw=0
                for t,q in chunks.pop(h):
                    raw+=len(t)
                    for stamp,flag in zip(t.tolist(),q.tolist()):
                        merged[stamp]=merged.get(stamp,0)|flag
                ts=sorted(merged); qs=[merged[t] for t in ts]; offsets=[t-k*1000 for t in ts]
                a,b=map(int,pointers[i:i+2]); m=meta[k]
                assert points[a:b].tolist()==offsets and flags[a:b].tolist()==qs, k
                assert row['hour_epoch_s']==k and row['hour_utc']==datetime.fromtimestamp(k,timezone.utc).isoformat()
                assert all(0<=d<3600000 for d in offsets)
                assert all((q&15)==0 for q in qs)
                qarray=np.array(qs,dtype=np.uint16)
                counts={'measurement_records_raw':raw,'measurement_unique_timestamps':len(ts),
                    'measurement_duplicate_records':raw-len(ts),'physical_valid_records':len(ts),
                    'physical_invalid_records':0,'quality_zero_records':int((qarray==0).sum()),
                    'quality_nonzero_records':int((qarray!=0).sum()),
                    'quality_bit16_records':int(((qarray&16)!=0).sum()),
                    'quality_bit32_records':int(((qarray&32)!=0).sum()),
                    'quality_bits16_and32_records':int(((qarray&48)==48).sum())}
                assert all(v==m[n]==row[n] for n,v in counts.items()), k
                epoch=datetime(1970,1,1,tzinfo=timezone.utc)
                assert epoch+timedelta(milliseconds=ts[0])==m['measurement_first_time_bjt']
                assert epoch+timedelta(milliseconds=ts[-1])==m['measurement_last_time_bjt']
                steps=[y-x for x,y in zip(ts,ts[1:])]
                histogram.update(steps)
                extras={'first_offset_ms':offsets[0],'last_offset_ms':offsets[-1],
                    'on_100ms_grid_records':sum(d%100==0 for d in offsets),
                    'steps_not_100ms':sum(d!=100 for d in steps),'steps_over_1s':sum(d>1000 for d in steps),
                    'largest_timestamp_step_ms':max(steps),'first_moment_offset_ms':sum(offsets)}
                assert all(row[n]==v for n,v in extras.items())
                err=max(abs(len(ts)/36000-m['measurement_coverage_fraction']),
                    abs(len(ts)/36000-row['coverage_fraction']))
                assert err<2e-15
                maxcov=max(maxcov,err); checked+=len(ts);rawtotal+=raw;duptotal+=raw-len(ts)
                on_grid+=extras['on_100ms_grid_records'];nonzero+=counts['quality_nonzero_records']
            assert checked==24196167 and rawtotal==selected
            assert {str(k):v for k,v in sorted(histogram.items())}==read(OUT/'timestamp_step_histogram.json')['counts']
            summary=read(OUT/'sampling_summary.json')
            expected={'hours':687,'unique_valid_timestamps':checked,'raw_test_records':rawtotal,
                'duplicate_test_records':duptotal,'physical_invalid_records':0,
                'min_records':int(np.diff(pointers).min()),'max_records':int(np.diff(pointers).max()),
                'nominal_record_deficit':687*36000-checked,'on_100ms_grid_records':on_grid,
                'steps_not_100ms':sum(v for k,v in histogram.items() if k!=100),
                'steps_over_1s':sum(v for k,v in histogram.items() if k>1000),
                'largest_timestamp_step_ms':max(histogram),'quality_nonzero_records_retained':nonzero,
                'quality_zero_records':checked-nonzero,'source_rows_total':scanned}
            assert all(summary[n]==v for n,v in expected.items())
            assert summary['measured_power_values_loaded'] is False
            assert summary['power_finiteness_rechecked'] is False
            assert summary['new_predictions']==summary['new_scores']==0
            assert {p:sha(p) for p in before}==before
            for p,d in read(OUT/'code_before.json').items():assert sha(p)==d
            qa={'status':'PASS','utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),
                'executable':sys.executable,'argv':sys.argv,'python':sys.version,
                'method':'complete Arrow Dataset scan plus Python timestamp-to-OR-quality mapping; no producer import',
                'independence':'different implementation by same agent, not independent personnel',
                'source_rows_scanned':scanned,'hours_checked':687,'exact_timestamps_checked':checked,
                'exact_quality_flags_checked':checked,'all_hour_metadata_fields_checked':len(columns)-1,
                'max_coverage_difference':maxcov,'coverage_tolerance':2e-15,
                'timestamp_histogram_exact':True,'nominal_record_deficit':687*36000-checked,
                'protected_sources_unchanged':len(before),'measured_power_values_loaded':False,
                'power_finiteness_rechecked':False,'source_to_frozen_label_power_reconstruction':'NOT_PERFORMED',
                'new_predictions':0,'new_scores':0,'seven_family_work_complete':False,
                'mts_status':'NOT_COMPUTED_USER_DECISION'}
            with (OUT/'independent_qa_v002.json').open('x',encoding='utf-8') as f:
                json.dump(qa,f,ensure_ascii=False,indent=2); f.write('\n')
            print(json.dumps(qa,ensure_ascii=False),flush=True)
        finally:
            lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)


if __name__=='__main__':main()
