"""Serialized CDS transitions, sanitized failures, no listing or ambiguous POST retry."""
import re
import uuid
import warnings
import requests
from repair_common_v005 import read, sha, now, emit, need, stopped, OUT_V5
from cds_three import DEST, COLLECTION, VARIABLES, fingerprint, client

TRANSIENT_HTTP = {408, 500, 502, 503, 504}
REMOTE_WAIT = {'accepted', 'running', 'queued'}
REMOTE_FAILED = {'failed', 'dismissed', 'deleted', 'cancelled', 'canceled'}


class CDSSafeError(RuntimeError):
    def __init__(self, record):
        self.record = dict(record)
        super().__init__('CDS requires inspection: ' + record['error_type'] + ' / ' + record['phase'])


class CDSTransientError(CDSSafeError):
    pass


def failure_record(exc, phase, job):
    """Never emit str/repr(exc), a URL, headers, key, or SDK response body."""
    response = getattr(exc, 'response', None)
    status = getattr(response, 'status_code', None)
    status = status if isinstance(status, int) and 100 <= status <= 599 else None
    # Only inspect the local text for proxy-auth markers; it is never persisted.
    proxy_auth = isinstance(exc, requests.exceptions.ProxyError) and any(
        token in str(exc).lower() for token in ('407', 'proxy authentication', 'proxy authorization'))
    transient = phase in {'CLIENT_GET', 'POLL_GET', 'DOWNLOAD_GET'} and not proxy_auth and not isinstance(exc, requests.exceptions.SSLError) and (
        isinstance(exc, requests.exceptions.Timeout) or
        isinstance(exc, requests.exceptions.ConnectionError) or
        isinstance(exc, requests.exceptions.HTTPError) and status in TRANSIENT_HTTP)
    return dict(utc=now(), phase=phase, job=job, error_type=type(exc).__name__,
                http_status=status, proxy_auth_indication=proxy_auth, transient=bool(transient),
                raw_exception_retained=False, repair='v005')


def raise_safe(exc, phase, job, folder):
    record = failure_record(exc, phase, job)
    path = folder / 'observations' / ('e5_' + uuid.uuid4().hex + '.json')
    try:
        emit(path, record)
    except Exception as storage_error:
        safe = dict(utc=now(), phase='LOCAL_ERROR_RECEIPT', job=job, error_type=type(storage_error).__name__,
                    cause_error_type=record['error_type'], transient=False, raw_exception_retained=False)
        raise CDSSafeError(safe) from None
    if record['transient']:
        raise CDSTransientError(record) from None
    raise CDSSafeError(record) from None


def quiet_client():
    # The installed SDK warns with raw exception strings during its messages GET.
    # Suppress those in memory; auth and signed URLs never enter stderr/logs.
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        return client()


def checkpoint():
    """Small local identity/byte check on every tick. Full numeric audit is separate."""
    p = read(DEST / 'plan.json')
    need(len(p['jobs']) == 12, 'Exact CDS batch count required')
    active, next_job, done = [], None, 0
    for job in p['jobs']:
        need(job['collection_id'] == COLLECTION and job['request']['variable'] == VARIABLES, 'CDS scope mismatch')
        need(fingerprint(job['request']) == job['request_sha256'], 'CDS request SHA mismatch')
        folder = DEST / 'jobs' / job['id']
        intent, saved, downloaded = [folder / (name + '.json') for name in ('intent', 'submitted', 'download')]
        need(not (folder / 'failure.json').exists(), 'Recorded CDS failure requires inspection')
        if intent.exists():
            i = read(intent)
            need(i['plan_sha256'] == sha(DEST / 'plan.json') and i['request_sha256'] == job['request_sha256'], 'CDS intent identity mismatch')
            need(i['request'] == job['request'] and i['collection_id'] == COLLECTION, 'CDS intent scope mismatch')
            need(saved.exists(), 'Unacknowledged CDS POST; no automatic retry')
            s = read(saved)
            need(s['request_sha256'] == job['request_sha256'], 'CDS submitted identity mismatch')
            need(re.fullmatch(r'[A-Za-z0-9-]{8,100}', str(s['request_id'])) is not None, 'CDS request ID invalid')
        else:
            need(not saved.exists() and not downloaded.exists(), 'Orphan CDS receipt')
        if downloaded.exists():
            d = read(downloaded)
            path = (DEST / d['file']).resolve()
            need(path.is_relative_to(folder.resolve()), 'CDS file outside registered batch')
            need(d['request_id'] == s['request_id'] and d['request_sha256'] == job['request_sha256'], 'CDS download identity mismatch')
            need(path.stat().st_size == d['bytes'] and sha(path) == d['sha256'], 'CDS cached bytes changed')
            done += 1
        elif intent.exists():
            active.append(job)
        elif next_job is None:
            next_job = job
    need(len(active) <= 1, 'Multiple unresolved CDS requests')
    return p, active, next_job, done


def step():
    stopped()
    p, active, next_job, done = checkpoint()
    if done == len(p['jobs']):
        return dict(status='ALL_DOWNLOADED_NOT_YET_INPUT_QA', done=done, total=12)
    job = active[0] if active else next_job
    folder = DEST / 'jobs' / job['id']
    try:
        c = quiet_client()
    except Exception as exc:
        raise_safe(exc, 'CLIENT_GET', job['id'], folder)
    try:
        if not active:
            # WAL before POST. Any ambiguous response is terminal, never retried.
            emit(folder / 'intent.json', dict(utc=now(), request=job['request'],
                 request_sha256=job['request_sha256'], collection_id=COLLECTION, plan_sha256=sha(DEST / 'plan.json')))
            try:
                remote = c.submit(COLLECTION, job['request'])
                ident = str(remote.request_id)
                need(re.fullmatch(r'[A-Za-z0-9-]{8,100}', ident) is not None, 'Invalid CDS response ID')
                emit(folder / 'submitted.json', dict(utc=now(), request_id=ident, request_sha256=job['request_sha256']))
                return dict(status='SUBMITTED_NEW_BOUNDED_JOB', job=job['id'], done=done, total=12)
            except Exception as exc:
                record = failure_record(exc, 'SUBMIT', job['id'])
                try:
                    emit(folder / 'failure.json', record)
                except Exception as storage_error:
                    record.update(phase='SUBMIT_RECEIPT_WRITE', error_type=type(storage_error).__name__)
                raise CDSSafeError(record) from None
        saved = read(folder / 'submitted.json')
        phase = 'POLL_GET'
        try:
            remote = c.get_remote(saved['request_id'])
            need(str(remote.request_id) == saved['request_id'], 'CDS remote ID mismatch')
            need(remote.collection_id == COLLECTION, 'CDS remote collection mismatch')
            status = str(remote.status)
            need(status in REMOTE_WAIT | REMOTE_FAILED | {'successful'}, 'Unknown CDS status')
            emit(folder / 'observations' / (uuid.uuid4().hex + '.json'), dict(utc=now(), status=status, repair='v005'))
            if status in REMOTE_FAILED:
                emit(folder / 'failure.json', dict(utc=now(), phase='REMOTE', status=status, repair='v005'))
                raise RuntimeError('Registered CDS request failed remotely')
            if status != 'successful':
                return dict(status=status, job=job['id'], done=done, total=12)
            phase = 'DOWNLOAD_GET'
            part = folder / ('payload_' + uuid.uuid4().hex + '.download')
            remote.download(str(part))
            need(part.is_file() and part.stat().st_size > 0, 'Empty CDS payload')
            receipt = dict(utc=now(), file=str(part.relative_to(DEST)), sha256=sha(part),
                           bytes=part.stat().st_size, request_sha256=job['request_sha256'], request_id=saved['request_id'])
            # Reject corrupt/wrong-coordinate/wrong-field downloads before counting.
            phase = 'LOCAL_PAYLOAD_VALIDATION'
            from assemble_inputs import read_cds
            values, _ = read_cds(job, receipt)
            need(len(values) == 3 * len(job['epochs']), 'CDS exact value coverage')
            emit(folder / 'download.json', receipt)
            return dict(status='DOWNLOADED_NOT_YET_INPUT_QA', job=job['id'], done=done + 1, total=12)
        except Exception as exc:
            raise_safe(exc, phase, job['id'], folder)
    finally:
        # Session lifetime is one serial transition, no leaked connections.
        session = getattr(c, 'session', None)
        if session is not None:
            try:
                session.close()
            except Exception:
                pass
