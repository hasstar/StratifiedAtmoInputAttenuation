"""Fetch exact public author-source files; no installation or code execution."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
VENDOR = HERE / 'vendor'
SOURCES = {
    'tabm': ('yandex-research/tabm', '28e47ae301c92ec37787dde1ce923a0793f405b4',
             [('tabm.py', 'tabm.py'), ('LICENSE', 'TABM_LICENSE'), ('README.md', 'TABM_README.md')]),
    'embeddings': ('yandex-research/rtdl-num-embeddings', 'a8fc25025c83f2321c63ff127a3bcef83bb1bfb5',
                   [('rtdl_num_embeddings.py', 'rtdl_num_embeddings.py'), ('LICENSE', 'EMBEDDINGS_LICENSE')]),
}


def main():
    manifest_path = VENDOR / 'source_manifest.json'
    if manifest_path.exists():
        raise RuntimeError('Sources already recorded; do not replace')
    VENDOR.mkdir(exist_ok=True)
    records = []
    for name, (repo, commit, files) in SOURCES.items():
        for remote, local in files:
            path = VENDOR / local
            if path.exists():
                raise RuntimeError('Refusing to overwrite ' + str(path))
            url = f'https://raw.githubusercontent.com/{repo}/{commit}/{remote}'
            with requests.get(url, timeout=(10, 60), stream=True, allow_redirects=False) as response:
                response.raise_for_status()
                assert response.status_code == 200
                chunks, size = [], 0
                for block in response.iter_content(65536):
                    size += len(block)
                    if size > 2 * 1024 * 1024:
                        raise RuntimeError('Source exceeds 2MiB cap')
                    chunks.append(block)
                blob = b''.join(chunks)
                blob.decode('utf-8')
                with path.open('xb') as f:
                    f.write(blob)
                records.append(dict(package=name, repository=repo, commit=commit, url=url,
                                    local=local, bytes=size, sha256=hashlib.sha256(blob).hexdigest(),
                                    http_status=response.status_code, modified=False))
                print(local, size, flush=True)
    with manifest_path.open('x', encoding='utf-8') as f:
        json.dump(dict(retrieved_utc=datetime.now(timezone.utc).isoformat(), files=records,
                       purpose='User-authorized scientific comparison; public author source only'), f, indent=2)


if __name__ == '__main__':
    main()
