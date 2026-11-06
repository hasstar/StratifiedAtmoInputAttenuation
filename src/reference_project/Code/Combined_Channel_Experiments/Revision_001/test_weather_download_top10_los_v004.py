"""Offline contract, migration and geometry tests. No network, no training."""
import ast
import contextlib
import copy
import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import weather_download_top10_los_v004 as w
import top10_los_geometry_v003 as geo
from test_weather_download_v002 import RecoveryTests


class ScopeTests(unittest.TestCase):
    def setUp(self): self.plan=copy.deepcopy(w.READ(w.PLAN))
    def bad(self):
        with self.assertRaises(RuntimeError): w.coverage_contract(self.plan)
    def new(self): return next(n for n in self.plan['jobs'] if n['role']=='los_top10' and 'pressure_level' in n['request'])
    def test_exact_original_calendar(self): self.assertEqual(w.coverage_contract(self.plan)['pressure_request_hours'],22824)
    def test_extra_predictor(self): self.plan['feature_names'].append('temperature');self.bad()
    def test_old_vertical_column_rejected(self): self.new()['request']['area']=[51.25,-1.5,51.,-1.5];self.bad()
    def test_removed_geopotential_rejected(self): self.new()['request']['variable'].remove('geopotential');self.bad()
    def test_missing_level_rejected(self): self.new()['request']['pressure_level'].remove('650');self.bad()
    def test_changed_utc_rejected(self): self.new()['request']['time']=['00:00'];self.bad()
    def test_missing_day_rejected(self): self.new()['request']['day'].pop();self.bad()
    def test_duplicate_day_rejected(self): self.new()['request']['day'].append(self.new()['request']['day'][0]);self.bad()
    def test_bad_count_rejected(self): self.new()['expected_hours']+=1;self.bad()
    def test_changed_collection_rejected(self): self.new()['collection_id']='other';self.bad()
    def test_more_slots_rejected(self): self.plan['max_in_flight_requests']=3;self.bad()
    def test_rain_diagnostic_rejected(self):
        n=next(n for n in self.plan['jobs'] if n['role']=='los_top10' and 'pressure_level' not in n['request'])
        n['request']['variable'].append('total_precipitation');self.bad()
    def test_legacy_exact(self): self.plan['adopted'][0]['original_job']['request']['area'][0]+=1;self.bad()
    def test_duplicate_legacy_id(self): self.plan['adopted'][1]['request_id']=self.plan['adopted'][0]['request_id'];self.bad()
    def test_ambiguous_does_not_release(self):
        with self.assertRaises(RuntimeError): w.guard_states([({}, {'status':'AMBIGUOUS_SUBMISSION'})])
    def test_access_block_does_not_release(self):
        with self.assertRaises(RuntimeError): w.guard_states([({}, {'status':'BLOCKED_REMOTE_ACCESS','request_id':'retained'})])
    def test_three_active_ids_rejected(self):
        with self.assertRaises(RuntimeError):w.guard_states([({}, {'status':'WAITING_REMOTE','request_id':str(i)}) for i in range(3)])
    def test_duplicate_active_ids_rejected(self):
        with self.assertRaises(RuntimeError):w.guard_states([({}, {'status':'WAITING_REMOTE','request_id':'same'})]*2)
    def test_two_legacy_ids_prevent_post(self):
        nodes=[a['job'] for a in self.plan['adopted']]+[self.new()]
        def state(n): return {'request_id':n['id'],'status':'WAITING_REMOTE'} if n['role']=='legacy_top1' else {}
        with patch.object(w,'runtime_nodes',return_value=nodes),patch.object(w.transport,'state',side_effect=state),\
             patch.object(w.transport,'poll') as poll,patch.object(w.transport,'submit') as submit,patch.object(w,'progress',return_value={}):
            w.cycle(self.plan,SimpleNamespace());self.assertEqual(poll.call_count,2);submit.assert_not_called()
    def test_legacy_size_never_splits(self):
        with patch.object(w.transport,'save') as save,patch.object(w,'_split_original') as split:
            w.split_guarded({'role':'legacy_top1'},{'request_id':'keep'},'evidence')
            split.assert_not_called();self.assertEqual(save.call_args.kwargs['status'],'BLOCKED_LEGACY_SIZE_DIAGNOSIS')
    def test_legacy_verified_not_coverage(self):
        def state(n):return {'request_id':next(a['request_id'] for a in self.plan['adopted'] if a['job']['id']==n['id']), 'status':'VERIFIED'} if n['role']=='legacy_top1' else {}
        with patch.object(w.transport,'state',side_effect=state):
            p=w.progress(self.plan);self.assertEqual(p['legacy_downloads_verified_not_LOS'],2);self.assertEqual(p['los_downloaded_request_hours'],0)
    def test_no_training_calls(self):
        for path in (w.CODE,geo.Path(geo.__file__),w.HERE/'inventory_top10_model_v002.py'):
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                    self.assertNotIn(node.func.attr,{'fit','partial_fit','fit_transform','backward','train','step'})


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.m=geo.original()
    def test_west_longitude_not_east(self): self.assertEqual(geo.east_longitude(23,'degrees_west'),-23.)
    def test_0360_longitude(self): self.assertAlmostEqual(geo.east_longitude(358.5,'degrees_east'),-1.5)
    def test_unknown_longitude_units_fail(self):
        with self.assertRaises(RuntimeError):geo.east_longitude(23,'unknown')
    def test_north_clockwise_pointing(self):
        _,d=geo.ray_from_angles(0,0,0,90,45,self.m)
        np.testing.assert_allclose(d,[2**-.5,2**-.5,0],atol=1e-12)
    def test_round_trip_angles(self):
        _,d=geo.ray_from_angles(43,6,400,160,38,self.m)
        np.testing.assert_allclose(geo.ray_angles(d,43,6),[160,38],atol=1e-10)
    def test_invalid_downward_ray(self):
        with self.assertRaises(RuntimeError):geo.ray_from_angles(43,6,0,160,-1,self.m)
    def test_bilinear_linear_field_exact(self):
        lat=np.array([50.,51.,52.]);lon=np.array([-4.,-3.,-2.,-1.])
        values=(2*lat[:,None]+3*lon[None,:])[None,:,:]
        result=self.m._bilinear_time_series(values,lat,lon,np.array([51.2]),np.array([-2.3]))
        np.testing.assert_allclose(result,[2*51.2+3*(-2.3)],atol=1e-10)
    def test_outside_grid_no_extrapolation(self):
        result=self.m._bilinear_time_series(np.ones((1,2,2)),np.array([0.,1.]),np.array([0.,1.]),np.array([2.]),np.array([.5]))
        self.assertTrue(np.isnan(result).all())
    def intersection(self,heights):
        config={'station':{'latitude_deg_n':51.1445,'longitude_deg_e':-1.437,'antenna_phase_center_m_amsl':84},
                'satellite':{'nominal_longitude_deg_e':-23,'nominal_altitude_km':35786}}
        lat=np.arange(49,52.25,.25);lon=np.arange(-4,0.25,.25);r=self.m.GEOPOTENTIAL_EARTH_RADIUS_M
        z=np.asarray(heights)*r/(r+np.asarray(heights))*self.m.G0_M_S2
        values=np.broadcast_to(z[:,None,None],(len(z),len(lat),len(lon))).copy()
        return self.m._solve_intersections(values,lat,lon,1,config)
    def test_hourly_intersection_moves_not_vertical(self):
        result=self.intersection([45000.,50000.,55000.]);self.assertTrue(result.los_intersection_valid.all())
        self.assertTrue((result.los_horizontal_displacement_m>1000).all())
        self.assertTrue(np.all(np.diff(result.los_horizontal_displacement_m)>0))
        np.testing.assert_allclose(result.pressure_geometric_height_m_amsl,[45000,50000,55000],atol=.01)
    def test_below_antenna_invalid_not_backward(self): self.assertFalse(self.intersection([20.]).los_intersection_valid.any())
    def test_missing_z_invalid(self): self.assertFalse(self.intersection([np.nan]).los_intersection_valid.any())
    def test_retained_pointing_gap(self):
        g=w.READ(w.DEST/'geometry_manifest.json');self.assertEqual(g['pointing_review_required_ids'],['FRA-00609'])


if __name__=='__main__':
    p=w.load_plan();stream=io.StringIO();suite=unittest.TestSuite()
    for cls in (RecoveryTests,ScopeTests,GeometryTests):suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(cls))
    with contextlib.redirect_stdout(stream):result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    report={'status':'PASS' if result.wasSuccessful() else 'FAIL','checked_utc':w.transport.now(),
            'plan_fingerprint':p['fingerprint'],'code_sha256':w.SHA(w.CODE),'test_sha256':w.SHA(w.TEST),
            'tests_run':result.testsRun,'contract':w.coverage_contract(p),'network_requests':0,'new_fit_calls':0,
            'transcript':stream.getvalue(),'full_field_QA_complete':False}
    if not result.wasSuccessful():
        w.transport.failure('TOP10_LOS_PREFLIGHT_FAILED',report=report);print(stream.getvalue());raise SystemExit(1)
    w.load_plan();w.transport.immutable(w.DEST/'preflight_qa.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='transcript'},indent=2))
