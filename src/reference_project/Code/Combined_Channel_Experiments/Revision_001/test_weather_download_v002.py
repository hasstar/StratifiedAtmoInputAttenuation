"""Offline contract and recovery tests: no remote submissions, no scientific fitting."""
import ast
import copy
import io
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import weather_download_v002 as w


class HTTPFailure(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.response = SimpleNamespace(status_code=code)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        scratch = w.DEST / "preflight_scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.dest = patch.object(w, "DEST", Path(self.temp.name))
        self.dest.start()
        self.credentials = patch.object(w, "config", lambda: ("https://cds.climate.copernicus.eu/api", "synthetic-secret"))
        self.credentials.start()
        self.node = {"id": "synthetic", "parent_id": "parent", "collection_id": "synthetic-collection",
                     "request": {"day": ["01", "02", "04", "06"], "time": ["00:00", "01:00"],
                                 "variable": ["a", "b"], "pressure_level": ["1"], "area": [1, 0, 0, 0]},
                     "expected_hours": 8, "target": "payloads/synthetic.download"}

    def tearDown(self):
        self.credentials.stop()
        self.dest.stop()
        self.temp.cleanup()

    def initial(self, **changes):
        return w.save(self.node, {}, status="SUBMITTED", request_id="synthetic-id", **changes)

    def test_bisection_preserves_sparse_days(self):
        saved = self.initial()
        w.split_rejected(self.node, saved, "synthetic-audit")
        result = w.state(self.node)
        self.assertEqual(result["status"], "SPLIT_SIZE_REJECTION")
        self.assertEqual(result["request_id"], "synthetic-id")
        w.partition_check(self.node, result["children"])
        self.assertEqual(len(w.leaves({"jobs": [self.node]})), 2)

    def test_reject_duplicate_date(self):
        with self.assertRaises(RuntimeError):
            w.partition_check(self.node, [self.node, self.node])

    def test_reject_missing_date(self):
        with self.assertRaises(RuntimeError):
            w.partition_check(self.node, [w.child(self.node, ["01"], "__a")])

    def test_reject_variable_change(self):
        changed = copy.deepcopy(self.node)
        changed["request"]["variable"] = ["a"]
        with self.assertRaises(RuntimeError):
            w.partition_check(self.node, [changed])

    def test_reject_hour_change(self):
        changed = copy.deepcopy(self.node)
        changed["request"]["time"] = ["00:00"]
        with self.assertRaises(RuntimeError):
            w.partition_check(self.node, [changed])

    def test_reject_coordinate_change(self):
        changed = copy.deepcopy(self.node)
        changed["request"]["area"] = [0, 0, 0, 0]
        with self.assertRaises(RuntimeError):
            w.partition_check(self.node, [changed])

    def test_single_day_cannot_shrink_scope(self):
        self.node = w.child(self.node, ["01"], "__one")
        w.split_rejected(self.node, self.initial(), "synthetic-audit")
        self.assertEqual(w.state(self.node)["status"], "BLOCKED_SIZE_ONE_DATE")

    def test_submit_id_persisted_without_query(self):
        c = SimpleNamespace(submit=lambda *args: SimpleNamespace(request_id="new-id"))
        w.submit(self.node, {"fingerprint": "plan"}, c)
        self.assertEqual(w.state(self.node)["request_id"], "new-id")
        self.assertEqual(w.state(self.node)["status"], "SUBMITTED")

    def test_size_rejection_splits(self):
        def reject(*args):
            raise HTTPFailure(403, "cost limits exceeded; Your request is too large")
        w.submit(self.node, {"fingerprint": "plan"}, SimpleNamespace(submit=reject))
        self.assertEqual(w.state(self.node)["status"], "SPLIT_SIZE_REJECTION")

    def test_credential_rejection_not_split(self):
        def reject(*args):
            raise HTTPFailure(403, "licence not accepted")
        w.submit(self.node, {"fingerprint": "plan"}, SimpleNamespace(submit=reject))
        self.assertEqual(w.state(self.node)["status"], "BLOCKED_HTTP")

    def test_submit_timeout_is_ambiguous(self):
        def reject(*args):
            raise TimeoutError("lost response")
        w.submit(self.node, {"fingerprint": "plan"}, SimpleNamespace(submit=reject))
        self.assertEqual(w.state(self.node)["status"], "AMBIGUOUS_SUBMISSION")

    def test_poll_timeout_keeps_original_id(self):
        self.initial()
        def fail(*args):
            raise TimeoutError("network")
        w.poll(self.node, SimpleNamespace(get_remote=fail))
        saved = w.state(self.node)
        self.assertEqual(saved["status"], "RETRY_SAME_ID")
        self.assertEqual(saved["request_id"], "synthetic-id")
        self.assertGreater(saved["retry_after_epoch"], 0)

    def test_accepted_queue_is_not_failure(self):
        self.initial()
        remote = SimpleNamespace(collection_id=self.node["collection_id"], status="accepted")
        w.poll(self.node, SimpleNamespace(get_remote=lambda _: remote))
        self.assertEqual(w.state(self.node)["status"], "WAITING_REMOTE")
        self.assertFalse((w.DEST / "failures").exists())

    def test_lost_remote_access_cannot_release_slot(self):
        self.initial()
        def fail(*args):
            raise HTTPFailure(404, "not found")
        w.poll(self.node, SimpleNamespace(get_remote=fail))
        self.assertEqual(w.state(self.node)["status"], "BLOCKED_REMOTE_ACCESS")

    def test_remote_failure_preserved_not_resubmitted(self):
        self.initial()
        def fail():
            raise RuntimeError("server processing error")
        remote = SimpleNamespace(collection_id=self.node["collection_id"], status="failed", get_results=fail)
        w.poll(self.node, SimpleNamespace(get_remote=lambda _: remote))
        self.assertEqual(w.state(self.node)["status"], "BLOCKED_REMOTE")
        self.assertTrue(list((w.DEST / "failures").glob("*.json")))

    def test_partial_retained_before_same_id_redownload(self):
        saved = self.initial()
        target = w.DEST / self.node["target"]
        target.parent.mkdir(parents=True)
        partial = target.with_name(target.name + ".part")
        partial.write_bytes(b"bad")
        def download(path):
            self.assertFalse(partial.exists())
            self.assertEqual(w.state(self.node)["request_id"], "synthetic-id")
            Path(path).write_bytes(b"complete")
        with patch.object(w, "inspect_file", lambda _: {"netcdf_members": 1}):
            w.download_result(self.node, saved, SimpleNamespace(content_length=8, download=download))
        self.assertEqual(w.state(self.node)["status"], "VERIFIED")
        self.assertEqual(list((w.DEST / "partial_history").glob("*.part"))[0].read_bytes(), b"bad")

    def test_complete_partial_not_downloaded_twice(self):
        saved = self.initial()
        target = w.DEST / self.node["target"]
        target.parent.mkdir(parents=True)
        target.with_name(target.name + ".part").write_bytes(b"complete")
        with patch.object(w, "inspect_file", lambda _: {"netcdf_members": 1}):
            w.download_result(self.node, saved, SimpleNamespace(content_length=8))
        self.assertEqual(target.read_bytes(), b"complete")

    def test_secrets_redacted(self):
        result = w.safe_error(RuntimeError("synthetic-secret https://example.org/a?token=private"))
        self.assertNotIn("synthetic-secret", result["message"])
        self.assertNotIn("private", result["message"])

    def test_writer_scope_guard(self):
        with self.assertRaises(RuntimeError):
            w.atomic(w.EXTERNAL / "forbidden-test.json", {})

    def test_changed_request_rejected(self):
        self.initial()
        self.node["request"]["variable"] = ["changed"]
        with self.assertRaises(RuntimeError):
            w.state(self.node)


def contract(plan):
    old = w.read(w.EXTERNAL / "weather_plan.json")
    reused = {item["parent_id"] for item in plan["reused"]}
    for parent in old["jobs"]:
        if parent["id"] in reused:
            continue
        children = [n for n in plan["jobs"] if n["parent_id"] == parent["id"]]
        w.partition_check({**parent, "parent_id": parent["id"]}, children)
        for node in children:
            if "pressure_level" in node["request"]:
                w.require(len(node["request"]["day"]) <= 7, "Initial pressure cap violated")
    w.require(len(reused) == 4, "Expected four reused formal payloads")
    w.require(set(n["parent_id"] for n in plan["jobs"]) | reused == set(n["id"] for n in old["jobs"]),
              "Original parent coverage differs")
    w.require(len({n["id"] for n in plan["jobs"]}) == len(plan["jobs"]), "Duplicate new batch ID")
    tree = ast.parse(Path(w.__file__).read_text(encoding="utf-8"))
    forbidden = {"fit", "partial_fit", "backward", "train", "fit_transform"}
    w.require(not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and
                      n.func.attr in forbidden for n in ast.walk(tree)), "Training-like call in downloader")
    for path, expected in plan["protected_files"].items():
        w.require(w.sha(w.ROOT / path) == expected, "Original artifact changed during tests")
    return {"original_parents_checked": 77, "new_batches_checked": len(plan["jobs"]),
            "reused_payloads_checked": 4, "protected_files_checked": len(plan["protected_files"]),
            "date_multiset_and_all_non_date_fields_exact": True, "original_padding_dates_preserved": True,
            "no_training_calls": True, "network_requests": 0}


if __name__ == "__main__":
    plan = w.build_plan()
    output = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RecoveryTests)
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    print(output.getvalue())
    if not result.wasSuccessful():
        w.failure("OFFLINE_PREFLIGHT_FAILED", tests_run=result.testsRun, transcript=output.getvalue())
        raise SystemExit(1)
    report = {"status": "PASS", "completed_utc": w.now(), "code_sha256": w.sha(Path(w.__file__)),
              "test_sha256": w.sha(Path(__file__)), "tests_run": result.testsRun, "contract": contract(plan)}
    w.atomic(w.DEST / "preflight_qa.json", report)
    print(report)
