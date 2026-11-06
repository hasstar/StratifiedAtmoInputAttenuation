"""Offline controls for new rain identity, masks, stop and frozen Top10 reuse."""
import copy
import unittest
from collections import Counter
from datetime import datetime, timezone
from unittest.mock import patch
import weather_download_gbs_extend_v011 as d
import test_ncar_top10_v001 as original
t=d.transport

def epoch(s): return int(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp())
def rain_das(node):
    with patch.object(original,'t',t):
        body=original.das(node)
    return body.replace('Int32 ecmwf_parameter', f'String long_name "'+({'MTPR':'Mean total precipitation rate','MSR':'Mean snowfall rate'}[node['nc_variable']])+'"; Int32 ecmwf_local_table 235; Int32 ecmwf_parameter')

class ExtensionTests(unittest.TestCase):
    def node(self,var=None):
        return t.plan_requests({(var or t.RAIN[0],0,epoch('2004-03-01T00:00'),204,-6)})[0]
    def test_all_original_top10_unchanged(self):
        pairs=[(v,p) for p,vs in d.core.LEVEL_VARS.items() for v in vs]+[(v,0) for v in d.core.SURFACE]
        for v,p in pairs:
            keys={(v,p,epoch('2004-03-01T00:00'),202,-8)}
            self.assertEqual(t.plan_requests(keys),original.t.plan_requests(keys))
    def test_original_ten_decoder_unchanged(self):
        for p,vs in d.core.LEVEL_VARS.items():
            for v in vs:
                node=t.plan_requests({(v,p,epoch('2004-03-01T00:00'),202,-8)})[0]
                self.assertEqual(t.decode(original.ascii_body(node),original.das(node),node), original.t.decode(original.ascii_body(node),original.das(node),node))
    def test_rain_fields_identity(self):
        for v,param in zip(t.RAIN,(235055,235031)):
            n=self.node(v); rows,c=t.decode(original.ascii_body(n),rain_das(n),n)
            self.assertEqual(c['param_id'],param); self.assertEqual(len(rows),1)
    def test_rain_prior_month_and_lead(self):
        n=self.node(); self.assertIn('/200402/',n['file']); self.assertEqual(n['coords']['forecast_hour'],[6]); self.assertEqual(n['layout'],'forecast')
    def test_rain_wrong_local_table(self):
        n=self.node()
        with self.assertRaises(ValueError):t.decode(original.ascii_body(n),rain_das(n).replace('table 235','table 128'),n)
    def test_rain_wrong_name(self):
        n=self.node()
        with self.assertRaises(ValueError):t.decode(original.ascii_body(n),rain_das(n).replace('Mean total precipitation rate','Mean snowfall rate'),n)
    def test_rain_wrong_units(self):
        n=self.node()
        with self.assertRaises(ValueError):t.decode(original.ascii_body(n),rain_das(n).replace('kg m**-2 s**-1','m'),n)
    def test_rain_pressure_forbidden(self):
        with self.assertRaises(ValueError): t.locate(t.RAIN[0],850,0)
    def test_auxiliary_forbidden(self):
        for p in (5,500,650,850):
            with self.assertRaises(ValueError):t.locate('geopotential',p,0)
    def test_unapproved_diagnostic_forbidden(self):
        with self.assertRaises(ValueError):t.locate('total_cloud_cover',0,0)
    def test_forecast_masks_no_extra(self):
        keys={(v,0,epoch('2004-02-29T23:00')+h*3600,204,-6) for v in t.RAIN for h in (0,1,2,7,8,48)}
        nodes=t.plan_requests(keys); self.assertEqual(Counter(k for n in nodes for k in t.expected_keys(n)),Counter(keys))
    def test_unequal_station_dates(self):
        evidence=[{'station':'chilbolton','date_utc':'2004-03-01'},{'station':'sparsholt','date_utc':'2004-03-02'}]
        pp=d.make_parents(evidence,set())
        self.assertEqual(len(pp),6)
        self.assertTrue(all(len(p['link_ids'])==1 for p in pp))
    def test_identical_masks_shared(self):
        pp=d.make_parents([{'station':s,'date_utc':'2004-03-01'} for s in d.STATIONS],set())
        self.assertEqual(len(pp),3);self.assertTrue(all(len(p['link_ids'])==2 for p in pp))
    def test_rain_interval_end_shift(self):
        pp=d.make_parents([{'station':'chilbolton','date_utc':'2004-03-31'}],set())
        r=next(p for p in pp if p['kind']=='rain');s=next(p for p in pp if p['kind']=='surface')
        self.assertEqual(d.hours(r),[x+3600 for x in d.hours(s)]);self.assertEqual(d.hours(r)[-1],epoch('2004-04-01T00:00'))
    def test_baseline_first_no_duplicate_dates(self):
        pp=d.make_parents([{'station':'chilbolton','date_utc':x} for x in ('2003-08-19','2004-02-18')],{'2004-02-18'})
        self.assertEqual(pp[0]['phase'],'baseline39');self.assertEqual(sum(p['expected_hours'] for p in pp if p['kind']=='surface'),48)
    def test_stop_prevents_new_get(self):
        with patch.object(d,'stopped',return_value=True),patch.object(d,'raw_get_body') as remote:
            with self.assertRaises(RuntimeError): d.get_body('ignored',None,None)
            remote.assert_not_called()
    def test_stop_prevents_run(self):
        with patch.object(d,'stopped',return_value=True),patch.object(d.app,'load_plan') as load:
            with self.assertRaises(RuntimeError):d.run()
            load.assert_not_called()

if __name__=='__main__': unittest.main()
