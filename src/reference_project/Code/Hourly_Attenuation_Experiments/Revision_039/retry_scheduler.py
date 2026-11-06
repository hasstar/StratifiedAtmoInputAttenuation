"""Persistent bounded retries. No threads, parallel requests or worker restarts."""
import hashlib
import json
import os
import time
from pathlib import Path
from scheduler_common import need, now, stopped

COOLDOWNS = (60, 180, 600)
MAX_TRANSIENTS = 512


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


class Journal:
    """Append-before-attempt WAL. A partial or altered log stops, not resets budget."""
    def __init__(self, path, scope):
        self.path, self.scope = Path(path), scope
        self.sequence, self.previous, self.entries, self.transients = 0, '0' * 64, {}, 0
        if self.path.exists():
            with self.path.open(encoding='utf-8') as stream:
                for line in stream:
                    need(line.endswith('\n'), 'Incomplete scheduler journal; inspection required')
                    event = json.loads(line)
                    signature = event.pop('sha256')
                    need(event['sequence'] == self.sequence + 1 and event['previous_sha256'] == self.previous, 'Scheduler journal chain mismatch')
                    need(event['scope'] == scope and digest(event) == signature, 'Scheduler journal SHA/scope mismatch')
                    self._apply(event, signature)

    def _apply(self, event, signature):
        self.sequence, self.previous = event['sequence'], signature
        self.entries[event['key']] = event['state']
        if event['kind'] in ('RETRY_WAIT', 'EXHAUSTED'):
            self.transients += 1

    def record(self, key, kind, state):
        event = dict(sequence=self.sequence + 1, previous_sha256=self.previous, scope=self.scope,
                     utc=now(), key=key, kind=kind, state=dict(state))
        signature = digest(event)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(dict(event, sha256=signature), sort_keys=True, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        self._apply(event, signature)


class RetryScheduler:
    def __init__(self, journal, transient_type, tick=lambda: None, report=lambda **kw: None,
                 guard=stopped, clock=time.time, sleep=time.sleep):
        self.journal, self.transient_type = journal, transient_type
        self.tick, self.report, self.guard, self.clock, self.sleep = tick, report, guard, clock, sleep

    def call(self, url, physical_get, cache_exists=False, daily=False):
        key = hashlib.sha256(url.encode()).hexdigest()
        self.guard()
        state = dict(self.journal.entries.get(key, dict(attempts=0, status='NEW', next_allowed=0)))
        if cache_exists:
            result = physical_get()  # sealed producer validates receipt/URL/bytes/SHA/values, no GET
            if state['status'] != 'SUCCEEDED':
                state['status'] = 'SUCCEEDED'
                self.journal.record(key, 'VERIFIED_CACHE', state)
            return result
        if state['status'] == 'FALLBACK' and daily:
            raise self.transient_type('Previously recorded daily fallback; no repeated daily GET')
        need(state['status'] not in ('ATTEMPT', 'FATAL', 'EXHAUSTED', 'SUCCEEDED'), 'Unresolved or terminal scheduler state requires inspection')
        limit = 1 if daily else 4
        need(state['attempts'] < limit, 'Persistent per-URL attempt budget exhausted')
        while state['attempts'] < limit:
            self.guard()
            while self.clock() < state['next_allowed']:
                self.guard()
                self.tick()
                self.sleep(min(5, max(0, state['next_allowed'] - self.clock())))
            self.guard()
            need(self.journal.transients < MAX_TRANSIENTS, 'v004 total transient ceiling exhausted')
            self.tick()  # finishes before the NCAR GET; never overlaps
            self.guard()
            state.update(attempts=state['attempts'] + 1, status='ATTEMPT')
            self.journal.record(key, 'ATTEMPT', state)
            self.report(scheduler_state='NCAR_ATTEMPT', attempt=state['attempts'], attempt_limit=limit, url_sha256=key)
            try:
                result = physical_get()
            except self.transient_type as exc:
                # The upstream exception contains only a sanitized fixed message.
                if 'HTTP 429' in str(exc):
                    state['status'] = 'FATAL'
                    self.journal.record(key, 'FATAL', state)
                    raise RuntimeError('HTTP 429 requires inspection of service policy; no blind retry') from None
                if daily:
                    state['status'] = 'FALLBACK'
                    self.journal.record(key, 'DAILY_FALLBACK', state)
                    raise
                if state['attempts'] == limit:
                    state['status'] = 'EXHAUSTED'
                    self.journal.record(key, 'EXHAUSTED', state)
                    raise RuntimeError('Four bounded attempts exhausted for exact NCAR input; inspection required') from None
                delay = COOLDOWNS[state['attempts'] - 1]
                state.update(status='RETRY_WAIT', next_allowed=self.clock() + delay)
                self.journal.record(key, 'RETRY_WAIT', state)
                self.report(scheduler_state='NCAR_COOLDOWN_CDS_SERIAL', attempt=state['attempts'],
                            attempt_limit=limit, retry_after_epoch=state['next_allowed'], cooldown_seconds=delay,
                            url_sha256=key)
            except Exception:
                state['status'] = 'FATAL'
                self.journal.record(key, 'FATAL', state)
                raise
            else:
                state.update(status='SUCCEEDED', next_allowed=0)
                self.journal.record(key, 'SUCCEEDED', state)
                return result


class CDSTicker:
    def __init__(self, step, report=lambda **kw: None, clock=time.monotonic):
        self.step, self.report, self.clock = step, report, clock
        self.last = -float('inf')
        self.done = False

    def __call__(self):
        if not self.done and self.clock() - self.last >= 60:
            state = self.step()
            self.last = self.clock()
            self.done = state['status'] == 'ALL_DOWNLOADED_NOT_YET_INPUT_QA'
            self.report(cds=state, cds_observed_utc=now())
