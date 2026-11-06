"""v009 regression plus unchanged 30 transport tests, all offline."""
import threading
import unittest
from unittest.mock import patch
import weather_download_top10_ncar_v009 as fix
import test_ncar_top10_v001 as original
from test_ncar_top10_v001 import TransportTests,RecoveryTests

class StateRegressionTests(RecoveryTests):
    # Inherit the six recovery controls and add concurrent Windows regression.
    def test_transient_read_sharing_retry(self):
        with patch.object(fix,'raw_read',side_effect=[PermissionError('sharing'),{'ok':1}]) as mocked:
            self.assertEqual(fix.read('ignored'),{'ok':1});self.assertEqual(mocked.call_count,2)
    def test_transient_write_sharing_retry(self):
        with patch.object(fix,'raw_atomic',side_effect=[PermissionError('sharing'),None]) as mocked:
            fix.atomic('ignored',{});self.assertEqual(mocked.call_count,2)
    def test_serial_state_read_write(self):
        node=self.node;plan=self.plan;errors=[]
        def writer():
            try:
                for i in range(150):fix.save(node,plan,status='TEST',sequence=i)
            except BaseException as e:errors.append(e)
        def reader():
            try:
                for i in range(250):fix.state(node)
            except BaseException as e:errors.append(e)
        a=threading.Thread(target=writer);b=threading.Thread(target=reader);a.start();b.start();a.join();b.join()
        self.assertFalse(errors);self.assertEqual(fix.state(node)['sequence'],149)
    def test_writer_failure_mapping(self):
        record=fix.safe_error(RuntimeError('test'))
        self.assertEqual(record['error_type'],'RuntimeError')
        self.assertEqual({**record}['message'],'test')
        evidence=fix.app.io.failure('TEST_WORKER_ERROR',exc=RuntimeError('test'))
        self.assertTrue((self.base/evidence).exists())
    def test_permanent_permission_not_hidden(self):
        with patch.object(fix,'raw_read',side_effect=PermissionError('persistent')),patch.object(fix.time,'sleep'):
            with self.assertRaises(PermissionError):fix.read('ignored')

if __name__=='__main__':unittest.main()
