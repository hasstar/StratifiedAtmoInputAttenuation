"""Offline positive/negative tests. No credentials, network or model operations."""
import ast
import copy
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch
import numpy as np
import fixedheight_los_core_v001 as c
import weather_download_top10_fixedheight_v007 as w

PARENT={'id':'GBS_England_2004-02_pressure','collection_id':'reanalysis-era5-pressure-levels',
        'expected_hours':6,'request':{'year':['2004'],'month':['02'],'day':['18','20','25'],
        'time':['00:00','23:00'],'pressure_level':['1']}}


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.link=w.R(w.OLD/'geometry_manifest.json')['links'][0]

    def test_rounding_positive_negative(self):
        self.assertEqual(c.grid_index(51.14,-1.44),(205,-6))

    def test_half_ties(self):
        self.assertEqual(c.grid_index(51.125,-1.375),(205,-5))

    def test_wrap_longitude(self):
        self.assertEqual(c.grid_index(10,359.9),(40,0))
        self.assertEqual(c.grid_index(0,180),(0,-720))

    def test_nonfinite_coordinate_rejected(self):
        with self.assertRaises(RuntimeError):c.grid_index(float('nan'),1)

    def test_forward_height_residual(self):
        lat,lon,s,r=c.intersect_heights(self.link,[1500,5600,48000])
        self.assertTrue(np.all(np.diff(s)>0))
        self.assertLess(np.max(np.abs(r)),.001)
        self.assertLess(lat[-1],self.link['latitude_deg_n']-.2)
        self.assertLess(lon[-1],self.link['longitude_deg_e']-.2)

    def test_below_antenna_rejected(self):
        with self.assertRaises(RuntimeError):c.intersect_heights(self.link,[0])

    def test_high_height_rejected(self):
        with self.assertRaises(RuntimeError):c.intersect_heights(self.link,[80001])

    def test_nonunit_ray_rejected(self):
        link={**self.link,'direction_ecef':[1.,1.,1.]}
        with self.assertRaises(RuntimeError):c.intersect_heights(link,[1000])

    def test_seed_is_not_final(self):
        rows,missing,targets=c.trace(self.link,1,[100],lambda *k:None)
        self.assertEqual(rows[0]['status'],'WAITING_Z')
        self.assertEqual(len(missing),1);self.assertFalse(targets)

    def test_hourly_measured_z_drives_intersection(self):
        H=45000*6371229/(6371229+45000);z=H*9.80665
        rows,missing,targets=c.trace(self.link,1,[100,200],lambda *k:(z,'synthetic.nc'))
        self.assertFalse(missing)
        self.assertEqual([r['status'] for r in rows],['STABLE_NEAREST']*2)
        self.assertAlmostEqual(rows[0]['height_m'],45000,places=5)
        self.assertEqual(len(targets),4)

    def test_invalid_z_retained(self):
        rows,missing,targets=c.trace(self.link,1,[100],lambda *k:(float('nan'),'synthetic.nc'))
        self.assertEqual(rows[0]['status'],'INVALID_PRESSURE_HEIGHT');self.assertFalse(targets)

    def test_high_station_seed_is_forward_only(self):
        link={**self.link,'height_amsl_m':2000.}
        link['origin_ecef_m']=c.original()._geodetic_to_ecef(link['latitude_deg_n'],link['longitude_deg_e'],2000.).tolist()
        self.assertEqual(c.seed_cell(link,850),c.grid_index(link['latitude_deg_n'],link['longitude_deg_e']))
        rows,missing,targets=c.trace(link,850,[100],lambda *key:(14000.,'synthetic.nc'))
        self.assertEqual(rows[0]['status'],'INVALID_FIXED_HEIGHT_NOT_ABOVE_ANTENNA');self.assertFalse(missing);self.assertFalse(targets)

    def test_all_source_seed_locations_forward(self):
        for link in w.R(w.OLD/'geometry_manifest.json')['links']:
            for level in (1,):
                with self.subTest(link=link['id'],level=level):self.assertEqual(len(c.seed_cell(link,level)),2)

    def test_cell_cycle_retained(self):
        seed=c.seed_cell(self.link,1)
        def alternate(link, heights):
            # z identifies which of the two neighbouring candidate cells was read.
            lat=np.array([seed[0]*.25 if h>10000 else (seed[0]+1)*.25 for h in heights])
            return lat,np.full(len(heights),seed[1]*.25),np.ones(len(heights)),np.zeros(len(heights))
        def lookup(var,level,t,i,j):return (5000. if i==seed[0] else 200000.,'synthetic.nc')
        with patch.object(c.base,'seed_cell',return_value=seed),patch.object(c.base,'intersect_heights',side_effect=alternate):
            rows,missing,targets=c.trace(self.link,1,[100],lookup)
        self.assertEqual(rows[0]['status'],'INVALID_NN_CELL_CYCLE');self.assertFalse(targets)


class PackingTests(unittest.TestCase):
    def setUp(self):self.hours=c.parent_hours(PARENT)

    def test_sparse_calendar_not_filled(self):
        self.assertEqual(len(self.hours),6)

    def test_count_mismatch_rejected(self):
        with self.assertRaises(RuntimeError):c.parent_hours({**PARENT,'expected_hours':7})

    def test_exact_rectangle_has_no_holes_filled(self):
        cells={(1,1),(1,2),(2,1)}
        result=c.tiles(cells)
        reconstructed=[(i,j) for n,w,s,e in result for i in range(s,n+1) for j in range(w,e+1)]
        self.assertEqual(Counter(reconstructed),Counter(cells))

    def test_two_variables_only_when_hours_equal(self):
        wanted={('geopotential',1,t,201,-7) for t in self.hours}
        wanted.add(('u_component_of_wind',1,self.hours[0],201,-7))
        requests=c.demands_to_requests(PARENT,wanted)
        self.assertEqual(Counter(k for r in requests for k in c.expand_request(r)),Counter(wanted))

    def test_hour_masks_do_not_cartesian_fill(self):
        wanted={('geopotential',1,self.hours[0],201,-7),('geopotential',1,self.hours[-1],201,-7)}
        requests=c.demands_to_requests(PARENT,wanted)
        self.assertEqual(len(requests),2)
        self.assertEqual(sum(len(list(c.expand_request(r))) for r in requests),2)

    def test_unused_variable_layer_rejected(self):
        with self.assertRaises(RuntimeError):c.demands_to_requests(PARENT,{('relative_humidity',1,self.hours[0],201,-7)})

    def test_out_of_scope_time_rejected(self):
        with self.assertRaises(RuntimeError):c.demands_to_requests(PARENT,{('geopotential',1,0,201,-7)})

    def test_each_auxiliary_z_forbidden(self):
        for p in c.FIXED_LEVELS:
            with self.subTest(level=p),self.assertRaises(RuntimeError):
                c.demands_to_requests(PARENT,{('geopotential',p,self.hours[0],201,-7)})


    def test_no_spatial_extra_cells(self):
        wanted={('geopotential',1,t,i,j) for i,j in ((201,-7),(202,-7),(202,-6)) for t in self.hours}
        requests=c.demands_to_requests(PARENT,wanted)
        self.assertEqual(Counter(k for r in requests for k in c.expand_request(r)),Counter(wanted))

    def test_out_of_grid_rectangle_rejected(self):
        request=c.demands_to_requests(PARENT,{('geopotential',1,self.hours[0],201,-7)})[0]
        request['area'][0]+=.1
        with self.assertRaises(RuntimeError):list(c.expand_request(request))

    def test_cross_level_request_rejected(self):
        request=c.demands_to_requests(PARENT,{('geopotential',1,self.hours[0],201,-7)})[0]
        request['pressure_level'].append('5')
        with self.assertRaises(RuntimeError):list(c.expand_request(request))

    def test_packing_deterministic(self):
        wanted=[('geopotential',1,t,i,-7) for i in (201,202) for t in self.hours]
        self.assertEqual(c.demands_to_requests(PARENT,wanted),c.demands_to_requests(PARENT,list(reversed(wanted))))


class SafetyTests(unittest.TestCase):
    def test_two_inherited_slots_never_post_third(self):
        nodes=[{'id':str(i),'role':'adopted'} for i in range(2)]
        saved={'status':'WAITING_REMOTE','request_id':'remote'}
        with patch.object(w,'runtime_nodes',return_value=nodes),patch.object(w.io,'state',side_effect=lambda n:{**saved,'request_id':n['id']}),patch.object(w.io,'poll') as polling,patch.object(w.io,'submit') as posting,patch.object(w,'advance',return_value=False):
            w.cycle({},object())
            self.assertEqual(polling.call_count,2);self.assertFalse(posting.called)

    def test_two_active_requests(self):
        self.assertEqual(len(w.guard([({}, {'request_id':str(k),'status':'WAITING_REMOTE'}) for k in range(2)])),2)

    def test_third_slot_forbidden(self):
        with self.assertRaises(RuntimeError):w.guard([({}, {'request_id':str(k),'status':'WAITING_REMOTE'}) for k in range(3)])

    def test_duplicate_id_forbidden(self):
        with self.assertRaises(RuntimeError):w.guard([({}, {'request_id':'id','status':'WAITING_REMOTE'})]*2)

    def test_ambiguous_forbids_submission(self):
        for status in ('SUBMITTING','AMBIGUOUS_SUBMISSION','BLOCKED_REMOTE_ACCESS','BLOCKED_HTTP'):
            with self.subTest(status=status),self.assertRaises(RuntimeError):w.guard([({}, {'status':status})])

    def test_inherited_size_never_split(self):
        with patch.object(w.io,'save') as saved,patch.object(w,'_split') as splitting:
            w.split_guarded({'role':'adopted'}, {}, 'synthetic')
            self.assertFalse(splitting.called)
            self.assertEqual(saved.call_args.kwargs['status'],'BLOCKED_LEGACY_SIZE_DIAGNOSIS')

    def test_only_new_size_may_split(self):
        with patch.object(w,'_split') as splitting:
            w.split_guarded({'role':'fixedheight007'}, {}, 'synthetic')
            self.assertTrue(splitting.called)

    def test_pressure_pair_count(self):
        self.assertEqual(sum(map(len,c.LEVEL_VARS.values())),8)
        self.assertEqual(sum(map(len,c.LEVEL_VARS.values()))+len(c.SURFACE),10)

    def test_no_training_or_interpolation_calls(self):
        forbidden={'fit','partial_fit','fit_transform','backward','step','bilinear','interp','interpolate','load_state_dict'}
        for path in (w.CODE,Path(c.__file__)):
            tree=ast.parse(path.read_text(encoding='utf-8'))
            calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
            self.assertFalse(set(calls)&forbidden)

    def test_current_plan_fixed_inputs(self):
        inv=w.R(w.OLD/'model_inventory.json')
        feature_set={f'p{p}__{v}' for p,vs in c.LEVEL_VARS.items() for v in vs}|{'surface__'+v for v in c.SURFACE}
        self.assertEqual(feature_set,set(inv['feature_names']))

    def test_flagged_link_not_dropped(self):
        geom=w.R(w.OLD/'geometry_manifest.json')
        self.assertEqual(len(geom['links']),218)
        self.assertIn('FRA-00609',geom['pointing_review_required_ids'])


class FieldGateTests(unittest.TestCase):
    def setUp(self):
        scratch=w.DEST/'preflight_scratch';scratch.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=scratch);self.path=Path(self.temp.name)/'sample.nc'
        self.parent=copy.deepcopy(PARENT);self.parent['expected_hours']=1
        self.parent['request']['day']=['18'];self.parent['request']['time']=['00:00']
        self.epoch=c.parent_hours(self.parent)[0]
        self.request=c.demands_to_requests(self.parent,{('geopotential',1,self.epoch,201,-7)})[0]

    def tearDown(self):self.temp.cleanup()

    def sample(self,unit='m**2 s**-2',level=1.,lat=50.25,epoch=None,value=400000.,extra=False):
        with w.netCDF4.Dataset(self.path,'w') as ds:
            for name,n in [('valid_time',1),('pressure_level',1),('latitude',1),('longitude',1)]:ds.createDimension(name,n)
            t=ds.createVariable('valid_time','i8',('valid_time',));t.units='seconds since 1970-01-01';t.calendar='proleptic_gregorian';t[:]=self.epoch if epoch is None else epoch
            p=ds.createVariable('pressure_level','f8',('pressure_level',));p.units='hPa';p[:]=level
            la=ds.createVariable('latitude','f8',('latitude',));la.units='degrees_north';la[:]=lat
            lo=ds.createVariable('longitude','f8',('longitude',));lo.units='degrees_east';lo[:]=-1.75
            v=ds.createVariable('z','f8',('valid_time','pressure_level','latitude','longitude'));v.units=unit;v.GRIB_paramId=129;v[:]=value
            if extra:
                v=ds.createVariable('t','f8',('valid_time','pressure_level','latitude','longitude'));v.units='K';v.GRIB_paramId=130;v[:]=250
        node={'role':'fixedheight007','request':self.request,'expected_hours':1}
        return {'path':self.path,'sha256':w.SHA(self.path),'node':node}

    def load(self,**kwargs):
        item=self.sample(**kwargs)
        with patch.object(w,'payloads',return_value=[item]):return w.load_cache({},self.parent)

    def test_exact_field_accepted(self):
        cache,hashes=self.load();self.assertEqual(len(cache),1);self.assertEqual(len(hashes),1)

    def test_wrong_unit_rejected(self):
        with self.assertRaises(RuntimeError):self.load(unit='m')

    def test_wrong_level_rejected(self):
        with self.assertRaises(RuntimeError):self.load(level=5.)

    def test_wrong_coordinate_rejected(self):
        with self.assertRaises(RuntimeError):self.load(lat=50.5)

    def test_wrong_time_rejected(self):
        with self.assertRaises(RuntimeError):self.load(epoch=self.epoch+3600)

    def test_nonfinite_rejected(self):
        with self.assertRaises(RuntimeError):self.load(value=float('nan'))

    def test_extra_parameter_rejected(self):
        with self.assertRaises(RuntimeError):self.load(extra=True)



class FixedHeightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.link=w.R(w.OLD/'geometry_manifest.json')['links'][0]

    def test_standard_heights_reference_ranges(self):
        # Independent rounded reference values, metres; no test-derived fit.
        expected={5:35979.,500:5579.3,650:3592.7,850:1457.6}
        for p,h in expected.items():
            self.assertLess(abs(c.standard_height(p)['geometric_height_m']-h),.1)

    def test_unauthorized_fixed_layer_rejected(self):
        for level in (1,10,1000):
            with self.subTest(level=level),self.assertRaises(RuntimeError):c.standard_height(level)

    def test_fixed_layers_never_read_z(self):
        def forbidden(*args):raise AssertionError('Fixed layer cannot read auxiliary z')
        for p in c.FIXED_LEVELS:
            rows,missing,targets=c.trace(self.link,p,[100,200],forbidden)
            self.assertFalse(missing)
            self.assertEqual(rows[0],rows[1])
            self.assertEqual(rows[0]['status'],'STABLE_FIXED_HEIGHT_NEAREST')
            self.assertFalse(rows[0]['actual_hourly_pressure_height_verified'])
            self.assertFalse(any(k[0]=='geopotential' for k in targets))
            self.assertEqual(len(targets),2*len(c.LEVEL_VARS[p]))

    def test_real_high_station_invalid_retained(self):
        link=next(x for x in w.R(w.OLD/'geometry_manifest.json')['links'] if x['id']=='FRA-00289')
        rows,missing,targets=c.trace(link,850,[100,200],lambda *k:None)
        self.assertEqual([r['status'] for r in rows],['INVALID_FIXED_HEIGHT_NOT_ABOVE_ANTENNA']*2)
        self.assertFalse(missing);self.assertFalse(targets)
        self.assertLess(rows[0]['height_m'],link['height_amsl_m'])

    def test_equal_height_not_moved_above_antenna(self):
        link={**self.link,'height_amsl_m':c.FIXED_HEIGHTS[850]['geometric_height_m']}
        self.assertEqual(c.fixed_location(link,850)['status'],'INVALID_FIXED_HEIGHT_NOT_ABOVE_ANTENNA')

    def test_request_whitelist_rejects_auxiliary_and_diagnostics(self):
        for p,v in ((5,'geopotential'),(500,'geopotential'),(650,'geopotential'),(850,'geopotential'),(0,'total_precipitation')):
            r={'variable':[v]}
            if p:r['pressure_level']=[str(p)]
            with self.subTest(p=p,v=v),self.assertRaises(RuntimeError):c.validate_request(r)

    def test_request_no_regrid_or_duplicate_fields(self):
        for request in ({'variable':['geopotential'],'pressure_level':['1'],'grid':[.25,.25]},
                        {'variable':['geopotential','geopotential'],'pressure_level':['1']}):
            with self.assertRaises(RuntimeError):c.validate_request(request)

    def test_all_allowed_pairs_pass(self):
        for p,vs in c.LEVEL_VARS.items():c.validate_request({'pressure_level':[str(p)],'variable':list(vs)})
        c.validate_request({'variable':list(c.SURFACE)})

    def test_pre_post_gate_stops_auxiliary(self):
        node={'id':'bad','role':'fixedheight007','request':{'variable':['geopotential'],'pressure_level':['5']}}
        with patch.object(w,'runtime_nodes',return_value=[node]),patch.object(w.io,'state',return_value={}),patch.object(w.io,'submit') as submit:
            with self.assertRaises(RuntimeError):w.cycle({},object())
            self.assertFalse(submit.called)


class RetiredRequestTests(unittest.TestCase):
    def setUp(self):
        self.node={'id':'retired','role':'adopted','download_policy':'poll_only_retired_auxiliary','collection_id':'pressure'}
        self.saved={'status':'WAITING_REMOTE','request_id':'same-id'}
        self.remote=unittest.mock.Mock(collection_id='pressure',status='accepted')
        self.client=unittest.mock.Mock()
        self.client.get_remote.return_value=self.remote

    def run_poll(self):
        def save(node,old,**fields):
            self.saved={**old,**fields};return self.saved
        with patch.object(w.io,'state',side_effect=lambda n:self.saved),patch.object(w.io,'save',side_effect=save),patch.object(w.io,'failure',return_value='evidence'),patch.object(w.io,'download_result') as dl:
            w.poll_retired(self.node,self.client)
            self.assertFalse(dl.called)
        self.assertFalse(self.remote.get_results.called)
        self.assertFalse(self.client.submit.called)

    def test_accepted_keeps_slot_and_id(self):
        self.run_poll()
        self.assertEqual(self.saved['status'],'WAITING_REMOTE')
        self.assertEqual(self.saved['request_id'],'same-id')
        self.assertEqual(len(w.guard([(self.node,self.saved)])),1)

    def test_success_releases_slot_without_payload(self):
        self.remote.status='successful';self.run_poll()
        self.assertEqual(self.saved['status'],'RETIRED_REMOTE_SUCCESS_NO_DOWNLOAD')
        self.assertFalse(self.saved['payload_downloaded'])
        self.assertFalse(self.saved['counted_as_new_coverage'])
        self.assertFalse(w.guard([(self.node,self.saved)]))

    def test_terminal_failure_no_retry_or_split(self):
        self.remote.status='failed';self.run_poll()
        self.assertEqual(self.saved['status'],'RETIRED_REMOTE_TERMINAL_NO_RETRY')
        self.assertFalse(w.guard([(self.node,self.saved)]))

    def test_access_error_does_not_release_slot(self):
        for code in (401,403,404,410):
            self.saved={'status':'WAITING_REMOTE','request_id':'same-id'}
            exc=RuntimeError('synthetic denied');exc.response=unittest.mock.Mock(status_code=code)
            self.client.get_remote.side_effect=exc;self.run_poll()
            self.assertEqual(self.saved['status'],'BLOCKED_REMOTE_ACCESS')
            self.assertFalse(w.terminal(self.saved))
            with self.assertRaises(RuntimeError):w.guard([(self.node,self.saved)])

    def test_transient_retries_same_id_only(self):
        self.client.get_remote.side_effect=TimeoutError('synthetic timeout');self.run_poll()
        self.assertEqual(self.saved['status'],'RETRY_SAME_ID')
        self.assertEqual(self.saved['request_id'],'same-id')
        self.assertFalse(w.terminal(self.saved))

    def test_backoff_does_not_call_remote(self):
        self.saved['retry_after_epoch']=w.time.time()+100
        self.run_poll();self.assertFalse(self.client.get_remote.called)

    def test_retired_payload_never_in_new_payloads(self):
        with patch.object(w,'runtime_nodes',return_value=[self.node]),patch.object(w.io,'state',return_value={'status':'RETIRED_REMOTE_SUCCESS_NO_DOWNLOAD'}):
            self.assertFalse(w.payloads({'reusable':[]}))

    def test_retired_not_verified_progress(self):
        with patch.object(w,'runtime_nodes',return_value=[self.node]),patch.object(w.io,'state',return_value={'status':'RETIRED_REMOTE_SUCCESS_NO_DOWNLOAD'}),patch.object(w,'waves',return_value=[]):
            p=w.progress({'fingerprint':'synthetic'})
        self.assertEqual(p['verified_requests'],0)
        self.assertEqual(p['retired_auxiliary_terminal_requests'],1)
        self.assertFalse(p['pending_or_blocked'])


if __name__=='__main__':unittest.main(verbosity=2)
