"""Bounded public HTML capture; no weather, credentials, external code or models."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit
import requests

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_026'
PREV = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_025'
URLS = {
    'nasa_gsdsu': 'https://earth.gsfc.nasa.gov/meso/models/g-sdsu',
    'nuwrf_11_5': 'https://astg.pages.smce.nasa.gov/nu-wrf/nu-wrf-dev/userguide/nuwrf_userguide_11.5.html',
    'luini_software': 'https://luini.deib.polimi.it/?page_id=81',
    'gsdsu_v2_release': 'http://opensource.gsfc.nasa.gov/projects/G-SDSU/',
}
HOSTS = {urlsplit(u).hostname for u in URLS.values()}
MAX_BYTES = 2 * 1024 * 1024

def utc():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')

class VisibleHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip = 0
        self.parts = []
        self.links = []
    def handle_starttag(self, tag, attrs):
        if tag in {'script', 'style'}:
            self.skip += 1
        if tag == 'a':
            href = dict(attrs).get('href')
            if href:
                self.links.append(href)
    def handle_endtag(self, tag):
        if tag in {'script', 'style'}:
            self.skip = max(0, self.skip - 1)
    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)

def capture(name, initial):
    record = {'name': name, 'requested_url': initial, 'started_utc': utc(), 'redirects': []}
    current = initial
    try:
        for step in range(4):
            assert urlsplit(current).hostname in HOSTS, 'unapproved redirect host'
            assert urlsplit(current).scheme in {'http', 'https'}
            # Each independent request has no browser session, cookie jar or auth header.
            session = requests.Session()
            session.trust_env = False  # No implicit .netrc credentials or credentialed proxies.
            with session, session.get(current, timeout=20, allow_redirects=False, stream=True,
                                      headers={'User-Agent': 'T012-public-source-audit/026'}) as response:
                record.update(status_code=response.status_code, final_url=current,
                              content_type=response.headers.get('Content-Type', ''))
                if response.status_code in {301, 302, 303, 307, 308}:
                    target = urljoin(current, response.headers['Location'])
                    record['redirects'].append({'status': response.status_code, 'from': current, 'to': target})
                    if step == 3:
                        raise RuntimeError('redirect limit reached')
                    current = target
                    continue
                chunks, size = [], 0
                for chunk in response.iter_content(65536):
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise RuntimeError('response exceeds 2 MiB')
                    chunks.append(chunk)
                payload = b''.join(chunks)
                raw = OUT / 'capture_run' / (name + '.response')
                with raw.open('xb') as stream:
                    stream.write(payload)
                record.update(raw_path=str(raw.relative_to(ROOT)), bytes=len(payload), sha256=sha(raw))
                if 'html' in record['content_type'].lower():
                    # Captured document is data only. Do not execute HTML/JavaScript.
                    parser = VisibleHTML()
                    parser.feed(payload.decode(response.encoding or 'utf-8', errors='replace'))
                    record['visible_text'] = ' '.join(' '.join(parser.parts).split())
                    record['links'] = sorted(set(urljoin(current, h) for h in parser.links))
                record['access'] = 'HTTP_OK' if response.status_code == 200 else 'HTTP_NON_SUCCESS'
                break
    except Exception as exc:
        record.update(access='ACCESS_UNRESOLVED', error_type=type(exc).__name__, error=str(exc))
    record['ended_utc'] = utc()
    write_json(OUT / 'capture_run' / (name + '.json'), record)
    return {k: v for k, v in record.items() if k not in {'visible_text', 'links'}}

def main():
    OUT.mkdir(exist_ok=True)
    # Never reuse this execution directory, even if a prior run failed.
    (OUT / 'capture_run').mkdir(exist_ok=False)
    write_json(OUT / 'execution.json', {'started_utc': utc(), 'pid': os.getpid(),
               'argv': sys.argv, 'python': sys.version, 'platform': platform.platform(),
               'requests': requests.__version__, 'scope': 'PUBLIC_HTML_ONLY',
               'thread': '01a0bda1-92d1-7832-9508-60f4e0ec1634', 'lock': 'LOCKED'})
    guard = json.loads((PREV / 'sources_before.json').read_text(encoding='utf-8'))
    for path, expected in guard.items():
        assert sha(path) == expected, path
    manifest = json.loads((PREV / 'delivery_manifest.json').read_text(encoding='utf-8'))
    for name, item in manifest['files'].items():
        path = ROOT / name
        assert sha(path) == item['sha256'] and path.stat().st_size == item['bytes'], name
    addendum = json.loads((PREV / 'POST_SEAL_ADDENDUM_001/manifest.json').read_text(encoding='utf-8'))
    for name, expected in addendum['hashes'].items():
        assert sha(ROOT / name) == expected
    for directory in [PREV, ROOT / 'Code/Combined_Channel_Experiments/Revision_025',
                      ROOT / 'Code/Combined_Channel_Experiments/Revision_011']:
        for path in directory.rglob('*'):
            if path.is_file():
                guard[str(path)] = sha(path)
    write_json(OUT / 'sources_before.json', guard)
    write_json(OUT / 'code_before.json', {str(p.relative_to(ROOT)): sha(p) for p in Path(__file__).parent.iterdir() if p.is_file()})
    records = []
    for name, url in URLS.items():
        records.append(capture(name, url))
        print(json.dumps(records[-1], ensure_ascii=False), flush=True)
    changed = [p for p, h in guard.items() if sha(p) != h]
    assert not changed, changed
    write_json(OUT / 'capture_complete.json', {'utc': utc(), 'records': records,
               'protected_sources': len(guard), 'source_changes': changed,
               'new_model_predictions': 0, 'new_scores': 0, 'weather_requests': 0,
               'public_html_get_only': True, 'lock': 'RELEASED'})

if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        print(traceback.format_exc(), file=sys.stderr)
        if OUT.is_dir():
            write_json(OUT / ('failure_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '.json'),
                       {'utc': utc(), 'traceback': traceback.format_exc(), 'lock': 'RELEASED_PROCESS_EXIT'})
        raise
