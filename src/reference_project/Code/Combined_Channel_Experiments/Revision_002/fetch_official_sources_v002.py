"""Exact-path correction; preserve first download attempt and its three files."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import requests

HERE = Path(__file__).resolve().parent
DEST = HERE / 'vendor'

def main():
    assert not (DEST / 'source_manifest.json').exists()
    records = []
    for repo, commit, pairs in [
        ('yandex-research/tabm', '28e47ae301c92ec37787dde1ce923a0793f405b4',
         [('tabm.py','tabm.py'),('LICENSE','TABM_LICENSE'),('README.md','TABM_README.md')]),
        ('yandex-research/rtdl-num-embeddings','a8fc25025c83f2321c63ff127a3bcef83bb1bfb5',
         [('package/rtdl_num_embeddings.py','rtdl_num_embeddings.py'),('package/LICENSE','EMBEDDINGS_LICENSE')])]:
        for remote, local in pairs:
            url = f'https://raw.githubusercontent.com/{repo}/{commit}/{remote}'
            r = requests.get(url, timeout=(10,60), allow_redirects=False)
            r.raise_for_status()
            assert r.status_code == 200 and 0 < len(r.content) < 2*1024*1024
            r.content.decode('utf-8')
            path = DEST / local
            reused = path.exists()
            if reused:
                assert path.read_bytes() == r.content, 'Previously downloaded source differs'
            else:
                with path.open('xb') as f:
                    f.write(r.content)
            records.append(dict(repository=repo, commit=commit, url=url, local=local,
                                sha256=hashlib.sha256(r.content).hexdigest(), bytes=len(r.content),
                                http_status=200, reused_exact_bytes=reused, modified=False))
            print(local,len(r.content),flush=True)
    with (DEST/'source_manifest.json').open('x',encoding='utf-8') as f:
        json.dump(dict(retrieved_utc=datetime.now(timezone.utc).isoformat(), files=records,
                       prior_attempt='Unconfirmed dependency root-file path returned404; official Git tree established package/ path. No old file overwritten.'),f,indent=2)

if __name__ == '__main__':
    main()
