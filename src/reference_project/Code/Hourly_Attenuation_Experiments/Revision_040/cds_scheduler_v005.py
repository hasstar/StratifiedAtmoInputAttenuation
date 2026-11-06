"""Persistent bounded CDS deferral. Same thread, no worker or request concurrency."""
import time
from repair_common_v005 import Journal, need, now, stopped, OUT_V5, MANIFEST_V5, sha
from cds_isolated_v005 import CDSTransientError, CDSSafeError, step

CDS_COOLDOWNS = (60, 180, 600)
CDS_MAX_TRANSIENTS = 64
DONE = 'ALL_DOWNLOADED_NOT_YET_INPUT_QA'
BLOCKED = 'CDS_BLOCKED_REQUIRES_INSPECTION_NCAR_CAN_CONTINUE'


class PersistentCDSTicker:
    def __init__(self, journal, operation=step, report=lambda **kw: None, clock=time.time, guard=stopped):
        self.journal, self.operation, self.report = journal, operation, report
        self.clock, self.guard = clock, guard
        self.state = dict(journal.entries.get('CDS', dict(status='NEW', consecutive_failures=0,
            total_transients=0, operations=0, next_allowed=0, observation=None)))
        need(self.state['status'] not in ('ATTEMPT', 'FATAL'), 'Unresolved or fatal CDS operation; inspection required')

    def save(self, kind):
        self.journal.record('CDS', kind, self.state)

    def __call__(self):
        self.guard()
        if self.state['status'] in (DONE, BLOCKED):
            return dict(self.state)
        if self.clock() < self.state['next_allowed']:
            return dict(self.state)
        self.state.update(status='ATTEMPT', operations=self.state['operations'] + 1)
        self.save('CDS_ATTEMPT')
        try:
            observed = self.operation()
        except CDSTransientError as exc:
            n = self.state['consecutive_failures'] + 1
            total = self.state['total_transients'] + 1
            blocked = n >= 4 or total >= CDS_MAX_TRANSIENTS
            self.state.update(status=BLOCKED if blocked else 'CDS_RETRY_WAIT_NCAR_CAN_CONTINUE',
                              consecutive_failures=n, total_transients=total, observation=exc.record,
                              next_allowed=0 if blocked else self.clock() + CDS_COOLDOWNS[n - 1])
            self.save('CDS_BLOCKED' if blocked else 'CDS_RETRY_WAIT')
        except Exception as exc:
            # Only explicitly safe records or type names cross the stage-lock boundary.
            safe = exc.record if isinstance(exc, CDSSafeError) else dict(
                utc=now(), phase='LOCAL_STATE', error_type=type(exc).__name__, raw_exception_retained=False)
            self.state.update(status='FATAL', observation=safe)
            self.save('CDS_FATAL')
            self.report(cds=dict(self.state), cds_observed_utc=now())
            raise CDSSafeError(safe) from None
        else:
            need(isinstance(observed, dict) and 'status' in observed, 'Invalid CDS operation result')
            self.state.update(status=DONE if observed['status'] == DONE else 'CDS_WAIT',
                              consecutive_failures=0, observation=observed, next_allowed=self.clock() + 60)
            self.save('CDS_OBSERVED')
        self.report(cds=dict(self.state), cds_observed_utc=now())
        return dict(self.state)


def ticker(report=lambda **kw: None):
    return PersistentCDSTicker(Journal(OUT_V5 / 'cds_journal.jsonl', sha(MANIFEST_V5)), report=report)
