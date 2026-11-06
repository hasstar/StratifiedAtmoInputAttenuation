"""Portable entry points for the exact current figure sources and editable art."""
from pathlib import Path
import argparse, json, os, shutil, subprocess, sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import package_support as s

def main(kind):
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--export-ppt',action='store_true',help='Re-export editable PPT with installed PowerPoint; otherwise retain the verified published vector PDF')
    a=ap.parse_args();out=a.out.absolute();out.mkdir(parents=True,exist_ok=True)
    upstream=s.path('figure_sources');inputs=s.path('figure_inputs');python=sys.executable
    if kind=='itu':
        pkg=upstream/'ITU小时观测图_Revision_003'
        subprocess.run([python,'-B',str(pkg/'code/build_figure.py'),'--package',str(pkg),'--output-dir',str(out)],check=True)
    elif kind=='two_paths':
        subprocess.run([python,'-B',str(upstream/'build_fig07.py'),'--source',str(inputs/'common/frozen_exports/tabm_same_k.csv'),'--out',str(out)],check=True)
    elif kind=='uk':
        subprocess.run([python,'-B',str(upstream/'build_fig08_and_audit.py'),
            '--metrics',str(inputs/'common/frozen_exports/uk_metrics.csv'),
            '--predictions',os.environ.get('TVT_UK_PREDICTIONS',str(s.original('Outputs/Experiments/Hourly_Attenuation_Revision_037_UK_Frozen_RFE/evaluation/all_hourly_predictions.parquet'))),
            '--out',str(out)],check=True)
    elif kind=='shap':
        subprocess.run([python,'-B',str(upstream/'TabM_K08_SHAP_20261104_002/code/plot_shap_v002.py'),'--output-root',str(out)],check=True)
    else:
        assets={
            'measurement':('高轨卫星实测图.pdf',None),
            'atmosphere':('Fig02_Stratified_Atmosphere_Editable.pdf','Fig02_Editable_Revision_002/Fig02_Stratified_Atmosphere_Editable.pptx'),
            'tabm':('Fig05_TabM_Architecture_And_Protocol.pdf','Fig05_TabM_Revision_003/editable/Fig05_TabM_Architecture_And_Protocol_Editable.pptx')}
        pdf,ppt=assets[kind];source=s.path('figures')/'published'/pdf;target=out/pdf
        if target.exists():raise FileExistsError(target)
        if a.export_ppt and ppt:
            if sys.platform!='win32':raise RuntimeError('PowerPoint export needs Windows; retain the supplied PDF or export the editable PPT using your presentation software.')
            deck=s.PACKAGE/'figsrc/editable'/ppt
            def q(x):return "'"+str(x).replace("'","''")+"'"
            command="$ErrorActionPreference='Stop'; $pptApp=New-Object -ComObject PowerPoint.Application; $deck=$null; try { $deck=$pptApp.Presentations.Open("+q(deck)+",$true,$false,$false); $deck.ExportAsFixedFormat("+q(target)+",2) } finally { if($deck){$deck.Close()}; [System.Runtime.InteropServices.Marshal]::ReleaseComObject($pptApp) | Out-Null }"
            subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',command],check=True)
            method='PPT vector export'
        else:
            shutil.copyfile(source,target);assert s.sha(source)==s.sha(target)
            method='Lossless publication asset reuse; no numerical plotting applies to this schematic'
        s.write(out/(kind+'_asset_provenance.json'),dict(method=method,pdf_source=s.rel(source),pdf_sha256=s.sha(source),editable_ppt=ppt,output_sha256=s.sha(target)))
