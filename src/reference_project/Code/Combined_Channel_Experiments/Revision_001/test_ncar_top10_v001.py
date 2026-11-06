"""Offline positive/negative controls for the NCAR-only route."""
import ast
import copy
import itertools
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
import numpy as np
import ncar_top10_transport_v001 as t
import weather_download_top10_ncar_v008 as d

def epoch(text):return int(datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp())

def sample_node(variable='u_component_of_wind',level=1):
    keys={(variable,level,epoch('2004-02-20T00:00:00')+3600*h,202,-8) for h in (0,1,2)}
    node=t.plan_requests(keys)[0]
    node.update(id='unit_node',parent_id='test',target='payloads/unit_node.ascii')
    return node

def das(node):
    nc,pid,_,_,units=t.FIELDS[node['variable']]
    names={'time':'hours since 1900-01-01 00:00:00','forecast_initial_time':'hours since 1900-01-01 00:00:00','forecast_hour':'hour',
           'latitude':'degrees_north','longitude':'degrees_east','level':'hPa'}
    blocks=[f'{nc} {{ Int32 ecmwf_parameter {pid}; String units "{units[0]}"; Float32 _FillValue 9.999E20; }}']
    blocks.extend(f'{n} {{ String units "{names[n]}"; }}' for n in node['dimensions'])
    return 'Attributes {\n'+'\n'.join(blocks)+'\n}'

def ascii_body(node):
    nc=node['nc_variable'];shape=tuple(len(a) for a in node['axes'])
    lines=['Dataset { };','---------------------------------------------',nc+'.'+nc+''.join(f'[{n}]' for n in shape)]
    for idx in np.ndindex(*shape[:-1]):
        lines.append(''.join(f'[{x}]' for x in idx)+', '+', '.join(str(float(x+1)) for x in range(shape[-1])))
    lines.append('')
    for n in node['dimensions']:
        lines.extend([f'{nc}.{n}[{len(node["coords"][n])}]',', '.join(str(x) for x in node['coords'][n]),''])
    return '\n'.join(lines)+'\n'

class TransportTests(unittest.TestCase):
    def test_all_ten_pairs(self):
        pairs=[(v,k) for k,vs in t.core.LEVEL_VARS.items() for v in vs]+[(v,0) for v in t.core.SURFACE]
        self.assertEqual(len(pairs),10)
        for v,k in pairs:
            node=sample_node(v,k);rows,contract=t.decode(ascii_body(node),das(node),node)
            self.assertEqual(len(rows),node['expected_values']);self.assertEqual(contract['parameter_id'],t.FIELDS[v][1])
    def test_auxiliary_rejected(self):
        for k in (5,500,650,850):
            with self.assertRaises(ValueError):t.locate('geopotential',k,epoch('2022-08-01T00:00:00'))
    def test_other_field_rejected(self):
        with self.assertRaises(ValueError):t.locate('total_precipitation',0,0)
    def test_pressure_indices(self):
        for k,idx in [(1,0),(5,3),(500,21),(650,24),(850,30)]:
            node=sample_node(t.core.LEVEL_VARS[k][0],k);self.assertEqual(node['axes'][1],[idx])
    def test_negative_longitude(self):
        n=sample_node();self.assertEqual(n['coords']['longitude'],[358.0]);self.assertEqual(n['coords']['latitude'],[50.5])
    def test_longitude_wrap_exact(self):
        keys={('geopotential',1,0,0,j) for j in (-1,0,1)}
        nodes=t.plan_requests(keys);self.assertEqual(set(k for n in nodes for k in t.expected_keys(n)),keys)
        self.assertTrue(all(n['expected_values']<=2 for n in nodes))
    def test_no_extra_holes_dates_hours(self):
        keys={('u_component_of_wind',1,epoch('2004-02-20T00:00:00')+3600*h,i,j) for h in (0,2,7,8,23,48) for i,j in ((201,-8),(202,-8),(202,-7))}
        nodes=t.plan_requests(keys);flat=[k for n in nodes for k in t.expected_keys(n)]
        self.assertEqual(set(flat),keys);self.assertEqual(len(flat),len(keys))
    def test_multi_lat_lon_rows(self):
        keys={('relative_humidity',850,0,i,j) for i,j in itertools.product((201,202),(-8,-7))}
        n=t.plan_requests(keys)[0];rows,_=t.decode(ascii_body(n),das(n),n)
        self.assertEqual(len(rows),4);self.assertEqual(set(tuple(k) for k,v in rows),keys)
    def test_cape_leap_month(self):
        n=t.plan_requests({(t.core.SURFACE[0],0,epoch('2004-02-29T23:00:00'),202,-8)})[0]
        self.assertIn('2004020100_2004022923',n['file']);self.assertEqual(n['axes'][0],[695])
    def test_forecast_boundaries(self):
        for valid,initial,lead in [('2022-08-01T00:00:00','2022-07-31T18:00:00',6),('2022-08-01T06:00:00','2022-07-31T18:00:00',12),
                                   ('2022-08-01T07:00:00','2022-08-01T06:00:00',1),('2022-08-16T06:00:00','2022-08-15T18:00:00',12),
                                   ('2022-08-16T07:00:00','2022-08-16T06:00:00',1)]:
            n=t.plan_requests({(t.core.SURFACE[1],0,epoch(valid),202,-8)})[0]
            self.assertEqual(t.EPOCH1900+n['coords']['forecast_initial_time'][0]*3600,epoch(initial))
            self.assertEqual(n['coords']['forecast_hour'],[lead]);self.assertEqual(next(t.expected_keys(n))[2],epoch(valid))
    def test_forecast_full_month_no_overlap(self):
        keys={(t.core.SURFACE[1],0,epoch('2022-08-01T00:00:00')+h*3600,202,-8) for h in range(744)}
        ns=t.plan_requests(keys);flat=[k for n in ns for k in t.expected_keys(n)]
        self.assertEqual(set(flat),keys);self.assertEqual(len(flat),744)
        for n in ns:self.assertEqual(len(t.decode(ascii_body(n),das(n),n)[0]),n['expected_values'])
    def test_wrong_unit(self):
        n=sample_node()
        with self.assertRaises(ValueError):t.decode(ascii_body(n),das(n).replace('m s**-1','K'),n)
    def test_wrong_parameter(self):
        n=sample_node()
        with self.assertRaises(ValueError):t.decode(ascii_body(n),das(n).replace('131;','129;'),n)
    def test_wrong_time_unit(self):
        n=sample_node()
        with self.assertRaises(ValueError):t.decode(ascii_body(n),das(n).replace('hours since','days since'),n)
    def test_wrong_time_values(self):
        n=sample_node();bad=ascii_body(n).replace(str(n['coords']['time'][0]),'9999999')
        with self.assertRaises(ValueError):t.decode(bad,das(n),n)
    def test_wrong_level_values(self):
        n=sample_node();bad=ascii_body(n).replace('U.level[1]\n1','U.level[1]\n5')
        with self.assertRaises(ValueError):t.decode(bad,das(n),n)
    def test_wrong_grid(self):
        n=sample_node();bad=ascii_body(n).replace('358.0','358.25')
        with self.assertRaises(ValueError):t.decode(bad,das(n),n)
    def test_nan_and_fill_rejected(self):
        n=sample_node()
        for bad in ('nan','inf','9.999E20'):
            with self.assertRaises(ValueError):t.decode(ascii_body(n).replace('[0][0][0], 1.0','[0][0][0], '+bad),das(n),n)
    def test_truncated_row(self):
        n=sample_node()
        with self.assertRaises(ValueError):t.decode(ascii_body(n).replace('[1][0][0], 1.0\n',''),das(n),n)
    def test_duplicate_row(self):
        n=sample_node()
        with self.assertRaises(ValueError):t.decode(ascii_body(n).replace('[1][0][0]','[0][0][0]'),das(n),n)
    def test_html_rejected(self):
        with self.assertRaises(ValueError):t.parse_ascii('<html>login</html>')
    def test_additional_variable_rejected(self):
        n=sample_node()
        with self.assertRaises(ValueError):t.decode(ascii_body(n)+'RAIN[1]\n1\n\n',das(n),n)
    def test_no_global_or_credentials_or_post(self):
        for p in (d.CODE,Path(t.__file__)):
            tree=ast.parse(p.read_text(encoding='utf-8'))
            calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
            self.assertFalse(set(calls)&{'post','fit','partial_fit','fit_transform','train','client','get_results'})
        n=sample_node();self.assertTrue(n['url'].startswith(t.BASE));self.assertIn('.ascii?',n['url']);self.assertNotIn('grid=',n['url'])
    def test_fixed_height_definition_unchanged(self):
        self.assertEqual(set(t.core.FIXED_HEIGHTS),{5,500,650,850})
        self.assertEqual(d.SHA(d.HERE/'fixedheight_los_core_v001.py'),'565e480b78fff3ff5a5b21eee40e4ebb9d46e7d70e3dfb41acbd47aee665feb5')

class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name);self.old=d.DEST;self.oldio=d.io.DEST
        d.DEST=self.base;d.io.DEST=self.base
        # Provenance paths need a common root in these isolated tests.
        self.oldroot=d.ROOT;d.ROOT=self.base
        self.node=sample_node();self.plan={'fingerprint':'test-plan'}
    def tearDown(self):d.DEST=self.old;d.io.DEST=self.oldio;d.ROOT=self.oldroot
    def fake_meta(self,node):
        path=self.base/'meta.das'
        if not path.exists():path.write_text(das(node),encoding='utf-8')
        return path,das(node)
    def test_complete_part_recovered_without_network(self):
        p=self.base/'payloads/unit_node.ascii.part';p.parent.mkdir();p.write_text(ascii_body(self.node),encoding='utf-8')
        with patch.object(d,'metadata',self.fake_meta),patch.object(d,'get_body',side_effect=AssertionError('network must not occur')):
            d.download(self.node,self.plan)
        self.assertEqual(d.state(self.node)['status'],'VERIFIED_SUBSET');self.assertFalse(p.exists())
    def test_invalid_partial_preserved_same_get(self):
        p=self.base/'payloads/unit_node.ascii.part';p.parent.mkdir();p.write_text('partial',encoding='utf-8')
        def fake_get(url,path,node):
            self.assertEqual(url,node['url']);path.write_text(ascii_body(node),encoding='utf-8');return {'bytes':path.stat().st_size,'seconds':1}
        with patch.object(d,'metadata',self.fake_meta),patch.object(d,'get_body',fake_get):d.download(self.node,self.plan)
        self.assertEqual(d.state(self.node)['status'],'VERIFIED_SUBSET')
        self.assertEqual(len(list((self.base/'partial_history').glob('*.part'))),1)
    def test_timeout_retries_same_get(self):
        with patch.object(d,'metadata',side_effect=d.requests.ReadTimeout('test')):d.download(self.node,self.plan)
        self.assertEqual(d.state(self.node)['status'],'RETRY_SAME_GET')
        self.assertGreater(d.state(self.node)['next_retry_epoch'],0)
    def test_access_failure_not_success(self):
        for status in (401,403,404,410):
            with patch.object(d,'metadata',side_effect=d.HttpFailure(status,'test')):d.download(self.node,self.plan)
            self.assertEqual(d.state(self.node)['status'],'BLOCKED_DOWNLOAD_INTEGRITY_OR_ACCESS')
    def test_payload_tamper_detected(self):
        p=self.base/'payloads/unit_node.ascii';p.parent.mkdir();p.write_text(ascii_body(self.node),encoding='utf-8')
        with patch.object(d,'metadata',self.fake_meta):d.download(self.node,self.plan)
        p.write_text('tampered',encoding='utf-8')
        with self.assertRaises(RuntimeError):d.read_rows(self.node,d.state(self.node))
    def test_rate_limit_respected(self):
        with patch.object(d,'metadata',side_effect=d.HttpFailure(429,'limit',600)):d.download(self.node,self.plan)
        self.assertGreaterEqual(d.state(self.node)['next_retry_epoch']-d.time.time(),590)

if __name__=='__main__':unittest.main()
