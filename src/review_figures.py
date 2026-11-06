"""Redraw the figures whose numerical inputs are included in the GitHub release."""
from pathlib import Path
from datetime import datetime,timezone
import subprocess,sys,json,time,argparse
r=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path);a=ap.parse_args()
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
out=a.output.resolve() if a.output else r/'results/figures'/('review_'+stamp)
out.relative_to(r);out.mkdir(parents=True,exist_ok=False)
log=r/'results/logs'/('review_figures_'+stamp+'.log');t=time.monotonic()
scripts=['plot_fig06_rfe.py','plot_fig07_independent_rfe.py','plot_fig08_uk.py','plot_fig10_fast_temporal.py']
with log.open('x',encoding='utf-8') as f:
    for script in scripts:
        print('Rendering '+script,flush=True)
        p=subprocess.run([sys.executable,'-X','utf8','-B',str(r/'figsrc'/script),'--out',str(out/Path(script).stem)],stdout=f,stderr=subprocess.STDOUT)
        f.flush()
        if p.returncode:raise RuntimeError('Figure failed; inspect '+str(log))
record={'status':'PASS','elapsed_seconds':time.monotonic()-t,'scripts':scripts,'output':str(out.relative_to(r)),'scope':'Redraw from distributed aggregate/UK results; no training or generation.'}
(out/'render_manifest.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record),flush=True)
