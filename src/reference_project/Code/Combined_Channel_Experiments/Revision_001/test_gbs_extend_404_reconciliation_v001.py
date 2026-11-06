"""Offline negative controls; no network, live state, model or data writes."""
import copy
import json
import unittest
from pathlib import Path
import reconcile_gbs_extend_404_v001 as fix


class GuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[3]
        dest = root/'Outputs/Experiments/Combined_Channel_Revision_001/external/weather_gbs_extend_v011'
        snapshot = dest/fix.RECOVERY/'before/jobs'/(fix.JOB_ID+'.json')
        fixture = snapshot if snapshot.exists() else dest/'jobs'/(fix.JOB_ID+'.json')
        cls.original = json.loads(fixture.read_text())

    def run_guard(self, saved=None, **kwargs):
        args = dict(saved=saved or copy.deepcopy(self.original), plan_fp=fix.PLAN_FP,
                    request_fp=fix.REQUEST_FP, stopped=False, locked=False)
        args.update(kwargs)
        fix.guard(**args)

    def test_exact_state_passes(self):
        self.run_guard()

    def test_stop_blocks(self):
        with self.assertRaises(RuntimeError): self.run_guard(stopped=True)

    def test_occupied_lock_blocks(self):
        with self.assertRaises(RuntimeError): self.run_guard(locked=True)

    def test_other_plan_blocks(self):
        with self.assertRaises(RuntimeError): self.run_guard(plan_fp='wrong')

    def test_other_request_blocks(self):
        with self.assertRaises(RuntimeError): self.run_guard(request_fp='wrong')

    def test_completed_state_blocks(self):
        s=copy.deepcopy(self.original); s['status']='VERIFIED_SUBSET'
        with self.assertRaises(RuntimeError): self.run_guard(s)

    def test_repeated_attempt_blocks(self):
        s=copy.deepcopy(self.original); s['attempts']=2
        with self.assertRaises(RuntimeError): self.run_guard(s)

    def test_other_http_blocks(self):
        s=copy.deepcopy(self.original); s['error']['message']='NCAR HTTP 403'
        with self.assertRaises(RuntimeError): self.run_guard(s)

    def test_changed_field_blocks(self):
        s=copy.deepcopy(self.original); s['job']['variable']='geopotential'
        with self.assertRaises(RuntimeError): self.run_guard(s)

    def test_expanded_scope_blocks(self):
        s=copy.deepcopy(self.original); s['job']['axes'][1]=[24,25]
        with self.assertRaises(RuntimeError): self.run_guard(s)

    def test_changed_url_blocks(self):
        s=copy.deepcopy(self.original); s['job']['url']=fix.URL+'&extra'
        with self.assertRaises(RuntimeError): self.run_guard(s)

    def test_path_escape_blocks(self):
        s=copy.deepcopy(self.original); s['job']['target']='../elsewhere'
        with self.assertRaises(RuntimeError): self.run_guard(s)

    def test_dds_exact_structure(self):
        fix.check_dds('Float32 U[time = 24][level = 1][latitude = 1][longitude = 1]; files/d633000/'+fix.FILE)

    def test_dds_wrong_shape_blocks(self):
        with self.assertRaises(RuntimeError):
            fix.check_dds('Float32 U[time = 25][level = 1][latitude = 1][longitude = 1]; files/d633000/'+fix.FILE)

    def test_dds_wrong_dataset_blocks(self):
        with self.assertRaises(RuntimeError):
            fix.check_dds('Float32 U[time = 24][level = 1][latitude = 1][longitude = 1]; other')


if __name__ == '__main__':
    unittest.main(verbosity=2)
