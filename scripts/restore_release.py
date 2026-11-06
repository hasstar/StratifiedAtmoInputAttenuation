"""Verify and restore the private GitHub release assets to their original paths."""
from pathlib import Path
import argparse, hashlib, json, os, shutil, zipfile

root=Path(__file__).resolve().parents[1]
if os.name=='nt':
    root=Path('\\\\?\\'+str(root))
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--archives',type=Path,required=True)
args=ap.parse_args()
manifest=json.loads((root/'results/logs/RELEASE_ASSETS.json').read_text('utf-8'))
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()
for asset in manifest['assets']:
    path=args.archives/asset['file']
    if path.stat().st_size!=asset['bytes'] or digest(path)!=asset['sha256']:
        raise ValueError('Archive verification failed: '+asset['file'])
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            relative=Path(member.filename)
            if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe archive member')
            if (member.external_attr>>16)&0o170000==0o120000:raise ValueError('Symlinks are not accepted')
            if not member.filename.startswith(('results/tables/rfe_state_summaries/','results/tables/reference_project/')):
                raise ValueError('Unexpected artifact destination')
            target=(root/relative).resolve();target.relative_to(root)
            if member.is_dir():target.mkdir(parents=True,exist_ok=True);continue
            if target.exists():
                with archive.open(member) as stream:h=hashlib.sha256(stream.read()).hexdigest()
                if digest(target)!=h:raise FileExistsError('Existing different file: '+str(relative))
                continue
            target.parent.mkdir(parents=True,exist_ok=True)
            with archive.open(member) as src,target.open('xb') as dest:shutil.copyfileobj(src,dest)
    print('Verified and restored '+asset['file'])
print('Complete. Local observations, local generator data and SVR support vectors remain intentionally unavailable.')
