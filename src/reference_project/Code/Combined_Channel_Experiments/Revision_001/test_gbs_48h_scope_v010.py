"""Independent scope/count/provenance checks; NOT independent field decoding QA."""
import hashlib
import json
import math
import unittest
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'Outputs/Experiments/Combined_Channel_Revision_001/external/validation_gbs_48h_v010'

def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

class ReducedScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m=load(OUT/'scope_manifest.json')
        cls.table=load(OUT/'local_top10_acquisition.json')
        cls.qa=load(OUT/'availability_qa.json')
        cls.rows=cls.table['rows']

    def test_independent_calendar(self):
        feb=['2004-02-18','2004-02-19','2004-02-20','2004-02-21','2004-02-25','2004-02-26','2004-02-27','2004-02-28']
        dates=feb+[(datetime(2004,3,1)+timedelta(days=i)).strftime('%Y-%m-%d') for i in range(31)]
        epochs={int((datetime.fromisoformat(d).replace(tzinfo=timezone.utc)+timedelta(hours=h)).timestamp()) for d in dates for h in range(24)}
        self.assertEqual(dates,self.m['dates_utc'])
        self.assertEqual(936,len(epochs))
        self.assertEqual(epochs,set(self.m['hours_utc_epoch_seconds']))
        self.assertEqual({(s,t) for s in ('chilbolton','sparsholt') for t in epochs},{(r['station'],r['utc_epoch_seconds']) for r in self.rows})
        self.assertEqual(1872,len(self.rows))

    def test_only_ten_features(self):
        names={'surface__convective_available_potential_energy','surface__maximum_2m_temperature_since_previous_post_processing',
               'p1__geopotential','p1__u_component_of_wind','p5__ozone_mass_mixing_ratio','p5__u_component_of_wind',
               'p500__ozone_mass_mixing_ratio','p650__u_component_of_wind','p850__ozone_mass_mixing_ratio','p850__relative_humidity'}
        self.assertEqual(names,set(self.m['feature_names']))
        self.assertEqual([],self.m['auxiliary_pairs'])
        for r in self.rows:
            self.assertEqual(names,set(r['values']))
            self.assertEqual(names,set(r['provenance']))
            self.assertTrue(all(math.isfinite(x) for x in r['values'].values()))

    def test_exact_key_identity_and_utc(self):
        for r in self.rows:
            self.assertEqual(r['utc_epoch_seconds'],int(datetime.fromisoformat(r['utc']).timestamp()))
            for name,p in r['provenance'].items():
                prefix,var=name.split('__')
                level=0 if prefix=='surface' else int(prefix[1:])
                key=p['key']
                self.assertEqual([var,level,r['utc_epoch_seconds']],key[:3])
                self.assertEqual(5,len(key))
                self.assertTrue(all(isinstance(v,int) for v in key[3:]))
                self.assertIn(p['source'],self.m['source_sha256'])

    def test_deduplicated_count_and_routes(self):
        unique={tuple(p['key']):p['source'] for r in self.rows for p in r['provenance'].values()}
        routes=Counter('NCAR' if 'weather_top10_ncar' in s else 'existing_CDS' for s in unique.values())
        self.assertEqual(11240,len(unique))
        self.assertEqual({'NCAR':3560,'existing_CDS':7680},dict(routes))
        self.assertEqual(len(unique),self.qa['unique_grid_field_hours'])
        self.assertEqual(18720,sum(len(r['values']) for r in self.rows))

    def test_all_actual_source_hashes(self):
        for path,sha in self.m['source_sha256'].items():
            self.assertEqual(sha,digest(ROOT/path),path)
        self.assertEqual(161,len(self.m['source_sha256']))

    def test_immutable_unit_and_artifact_hashes(self):
        for item in self.m['selected_units']+[self.m['local_acquisition_table'],self.m['availability_qa']]:
            self.assertEqual(item['sha256'],digest(ROOT/item['path']))
        self.assertEqual(4,len(self.m['selected_units']))

    def test_old_state_preserved(self):
        old=ROOT/'Outputs/Experiments/Combined_Channel_Revision_001/external/weather_top10_ncar_v009'
        evidence=load(OUT/'stop_verification.json')
        self.assertEqual('STOPPED_BY_HUMAN_SCOPE_REDUCTION',evidence['status'])
        self.assertEqual(0,evidence['actual_downloaders_remaining'])
        for item in evidence['stopped_v009_files']:
            self.assertEqual(item['sha256'],digest(old/item['relative_path']))
            self.assertEqual(item['sha256'],digest(OUT/'stopped_v009_snapshot'/item['relative_path']))

    def test_truthful_gates(self):
        self.assertFalse(self.m['overall_task_complete'])
        self.assertFalse(self.qa['independent_field_qa'])
        self.assertEqual('PENDING',self.qa['target_model_compatibility'])
        self.assertEqual(0,self.qa['missing_grid_field_hours'])
        for data in (self.m,self.qa,self.table):
            self.assertEqual(0,data['new_fit_calls'])
            self.assertEqual(0,data['predictions_executed'])

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ReducedScopeTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report={'status':'PASS' if result.wasSuccessful() else 'FAIL','tests_run':result.testsRun,
            'errors':len(result.errors),'failures':len(result.failures),
            'scope':'Independent scope, count, identity binding, source SHA and history-preservation tests; not independent raw-field decoding or model QA',
            'scope_manifest_sha256':digest(OUT/'scope_manifest.json'),'test_code_sha256':digest(Path(__file__)),
            'checked_utc':datetime.now(timezone.utc).isoformat()}
    with (OUT/'scope_independent_qa.json').open('x',encoding='utf-8') as f:
        json.dump(report,f,ensure_ascii=False,indent=2)
    raise SystemExit(0 if result.wasSuccessful() else 1)
