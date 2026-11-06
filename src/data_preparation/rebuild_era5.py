"""Re-extract local surface and pressure-surface ray intersections from raw ERA5."""
from pathlib import Path
import argparse,subprocess,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import package_support as s
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    if a.out.exists():raise FileExistsError('Choose a new output folder')
    code=Path(__file__).parent/'reference_era5/prepare_era5_los_csv.py'
    subprocess.run([sys.executable,'-B',str(code),'--config',str(Path(__file__).parent/'requests/local_campaign.json'),
        '--raw-dir',str(s.path('era5_local')/'raw'),'--output-dir',str(a.out)],check=True)
    s.write(a.out/'reproduction.json',dict(source_code_sha256=s.sha(code),raw_location=s.rel(s.path('era5_local')/'raw'),
        purpose='Raw ERA5 interpolation audit. The archived 156/50 feature identities and measured-data split remain fixed.'))
if __name__=='__main__':main()
