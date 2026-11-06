"""Bounded anonymous public artifact acquisition, never uploads local data."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import requests

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_006'
SESSION = requests.Session()
SESSION.trust_env = False
SESSION.headers['User-Agent'] = 'Frozen-comparator-provenance/1.0'
ALLOWED = {'pypi.org', 'files.pythonhosted.org', 'huggingface.co',
           'api.github.com', 'raw.githubusercontent.com'}

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()

def emit(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')

def get(url, path, expected_sha=None, size=None, cap=128*1024*1024):
    from urllib.parse import urlsplit
    if urlsplit(url).scheme != 'https' or urlsplit(url).hostname not in ALLOWED:
        raise RuntimeError('Unapproved initial URL')
    if path.exists():
        if expected_sha and sha(path) == expected_sha and (size is None or path.stat().st_size == size):
            return dict(url=url, path=str(path.relative_to(ROOT)), sha256=sha(path), bytes=path.stat().st_size, reused=True)
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.partial')
    with SESSION.get(url, stream=True, timeout=(10,180)) as r:
        r.raise_for_status()
        # Signed CDN redirects are supplied by the public author repository, not guessed.
        if not r.url.startswith('https://'): raise RuntimeError('Non-HTTPS redirect')
        length = r.headers.get('Content-Length')
        if length and int(length)>cap: raise RuntimeError('Oversize Content-Length')
        n = 0
        with partial.open('xb') as f:
            for block in r.iter_content(1048576):
                n += len(block)
                if n>cap: raise RuntimeError('Size cap')
                f.write(block)
        if length and n != int(length): raise RuntimeError('Content-Length mismatch')
        if size is not None and n != size: raise RuntimeError('Repository size mismatch')
        digest = sha(partial)
        if expected_sha and digest != expected_sha: raise RuntimeError('SHA mismatch')
        partial.rename(path)
        return dict(url=url, final_host=urlsplit(r.url).hostname, http_status=r.status_code,
                    content_length=int(length) if length else None,
                    path=str(path.relative_to(ROOT)), sha256=digest, bytes=n)

def json_get(url, path):
    get(url, path, cap=8*1024*1024)
    return json.loads(path.read_text(encoding='utf-8'))

def sources():
    records=[]
    versions={'tabpfn':'2.0.9', 'tabicl':'2.2.0', 'scikit-learn':'1.6.1',
              'huggingface-hub':'0.33.4', 'einops':'0.8.1'}
    for name,version in versions.items():
        meta=json_get(f'https://pypi.org/pypi/{name}/{version}/json', OUT/'references'/f'{name}_{version}_pypi.json')
        wheels=[v for v in meta['urls'] if v['filename'].endswith('.whl') and
                ('py3-none-any' in v['filename'] or 'cp312-cp312-win_amd64' in v['filename'])]
        if len(wheels)!=1: raise RuntimeError('Ambiguous wheel '+name)
        w=wheels[0]
        path=HERE/'artifacts'/w['filename']
        row=get(w['url'],path,w['digests']['sha256'],w['size'],cap=64*1024*1024)
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                dest=(HERE/'vendor'/info.filename).resolve()
                if not dest.is_relative_to((HERE/'vendor').resolve()) or info.filename.endswith('.pth'):
                    raise RuntimeError('Unsafe wheel member '+info.filename)
                if dest.exists(): raise FileExistsError(dest)
            z.extractall(HERE/'vendor')
        row.update(package=name,version=version,requirements=meta['info']['requires_dist'])
        records.append(row)
        print('OFFICIAL_WHEEL '+name+' '+version, flush=True)
    model=json_get('https://huggingface.co/api/models/Prior-Labs/TabPFN-v2-reg?blobs=true',OUT/'references'/'tabpfn_hub.json')
    if model.get('gated') or model.get('disabled'): raise RuntimeError('Model access gate')
    revision=model['sha']
    for name in ('README.md','LICENSE.txt'):
        records.append(get(f'https://huggingface.co/Prior-Labs/TabPFN-v2-reg/resolve/{revision}/{name}',OUT/'references'/('tabpfn_'+name)))
    emit(OUT/'sources_acquired.json',dict(utc=datetime.now(timezone.utc).isoformat(),files=records))

def checkpoint(repo, revision, filename, expected, size, alias):
    meta=json_get(f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true',OUT/'references'/f'{alias}_pinned_hub.json')
    if meta.get('gated') or meta.get('disabled'): raise RuntimeError('Model access gate')
    item=next(s for s in meta['siblings'] if s['rfilename']==filename)
    if meta['sha']!=revision or item['lfs']['sha256']!=expected or item['size']!=size:
        raise RuntimeError('Pinned metadata changed')
    row=get(f'https://huggingface.co/{repo}/resolve/{revision}/{filename}',HERE/'weights'/filename,expected,size,cap=256*1024*1024)
    emit(OUT/f'{alias}_checkpoint_acquired.json',row)
    print(json.dumps(row),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['sources','checkpoint'])
    parser.add_argument('--repo'); parser.add_argument('--revision');parser.add_argument('--filename')
    parser.add_argument('--sha');parser.add_argument('--size',type=int);parser.add_argument('--alias')
    a=parser.parse_args()
    if a.action=='sources':sources()
    else:checkpoint(a.repo,a.revision,a.filename,a.sha,a.size,a.alias)
