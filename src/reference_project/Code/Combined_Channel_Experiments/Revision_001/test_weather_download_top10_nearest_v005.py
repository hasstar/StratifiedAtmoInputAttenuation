"""Offline positive/negative tests. No credentials, network or model operations."""
import ast
import copy
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch
import numpy as np
import nearest_los_core_v001 as c
import weather_download_top10_nearest_v005 as w

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

    def test_cell_cycle_retained(self):
        seed=c.seed_cell(self.link,1)
        def alternate(link, heights):
            # z identifies which of the two neighbouring candidate cells was read.
            lat=np.array([seed[0]*.25 if h>10000 else (seed[0]+1)*.25 for h in heights])
            return lat,np.full(len(heights),seed[1]*.25),np.ones(len(heights)),np.zeros(len(heights))
        def lookup(var,level,t,i,j):return (5000. if i==seed[0] else 200000.,'synthetic.nc')
        with patch.object(c,'seed_cell',return_value=seed),patch.object(c,'intersect_heights',side_effect=alternate):
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

    def test_auxiliary_z_approved(self):
        wanted={('geopotential',p,self.hours[0],201,-7) for p in c.LEVEL_VARS}
        requests=c.demands_to_requests(PARENT,wanted)
        self.assertEqual(len(requests),5)
        self.assertTrue(all(len(r['pressure_level'])==1 for r in requests))

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
            w.split_guarded({'role':'nearest005'}, {}, 'synthetic')
            self.assertTrue(splitting.called)

    def test_pressure_pair_count(self):
        self.assertEqual(sum(map(len,c.LEVEL_VARS.values())),8)
        self.assertEqual(sum(map(len,c.LEVEL_VARS.values()))+4,12)

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
        node={'role':'nearest005','request':self.request,'expected_hours':1}
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


if __name__=='__main__':unittest.main(verbosity=2)
