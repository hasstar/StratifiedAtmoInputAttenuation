"""Lossless metadata-only timestamp support. No measured powers or model fitting."""
import numpy as np

HOUR_MS = 3_600_000
BJT_OFFSET_MS = 8 * HOUR_MS


def utc_milliseconds(bjt_naive_ms):
    a = np.asarray(bjt_naive_ms)
    if a.dtype.kind not in 'iu':
        raise TypeError('exact integer milliseconds required')
    return a.astype(np.int64) - BJT_OFFSET_MS


def deduplicate(time_ms, quality):
    t, q = np.asarray(time_ms), np.asarray(quality)
    if t.ndim != 1 or q.ndim != 1 or t.shape != q.shape:
        raise ValueError('one-dimensional matching metadata required')
    if t.dtype.kind not in 'iu' or q.dtype.kind not in 'iu':
        raise TypeError('integer metadata required')
    if np.any(q < 0) or np.any(q > 65535):
        raise ValueError('quality outside uint16')
    if t.size == 0:
        return t.astype(np.int64), q.astype(np.uint16)
    order = np.argsort(t, kind='stable')
    ts, qs = t[order].astype(np.int64), q[order].astype(np.uint16)
    starts = np.r_[0, np.flatnonzero(np.diff(ts) != 0) + 1]
    return ts[starts], np.bitwise_or.reduceat(qs, starts)


def physical_quality_mask(q):
    return (np.asarray(q, dtype=np.uint16) & 15) == 0


def within_hour_offsets(time_ms, hour_epoch_s):
    d = np.asarray(time_ms, dtype=np.int64) - int(hour_epoch_s) * 1000
    if np.any(d < 0) or np.any(d >= HOUR_MS):
        raise ValueError('timestamp outside exact UTC hour')
    if d.size > 1 and np.any(np.diff(d) <= 0):
        raise ValueError('offsets must be unique and increasing')
    return d.astype(np.uint32)
