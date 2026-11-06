"""Seal only this new plotting delivery and compare a clean rendering replay."""
from pathlib import Path
import argparse,json,hashlib,shutil,sys,datetime,importlib.metadata as md
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def dump(p,r):p.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf8')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--package',type=Path,default=ROOT/'最终提交的修改材料/论文图表_20261030_001');ap.add_argument('--replay',type=Path,required=True);a=ap.parse_args();p=a.package.resolve();replay=a.replay.resolve();q=p/'05_QA'
 assert read(q/'FINAL_QA.json')['status']=='PASS'
 assert read(replay/'05_QA/FINAL_QA.json')['status']=='PASS'
 assert read(replay/'05_QA/REPRODUCE_RECEIPT.json')['output']==str(replay)
 matched=[]
 for folder,pattern in [('05_QA/previews','*.png'),('02_Tables_LaTeX','*.tex'),('04_Source_Data/table_data','*.csv')]:
  for f in (p/folder).glob(pattern):
   relative=f.relative_to(p);assert sha(f)==sha(replay/relative),str(relative);matched.append(relative.as_posix())
 old=q/'REPRODUCIBILITY_QA.json';history=q/'REPRODUCIBILITY_QA_pre_axis_fix.json'
 if old.exists() and not history.exists():shutil.copy2(old,history)
 dump(old,{'status':'PASS','fresh_output_directory':str(replay),'end_to_end_reproduce_script_completed':True,'byte_identical_preview_images_tables_and_numeric_exports':matched,'expected_binary_metadata_variation':'PDF/PPTX creation metadata and IDs can differ; values, table sources and rendered previews are compared.'})
 shutil.copy2(replay/'05_QA/REPRODUCE_RECEIPT.json',q/'REPRODUCE_RECEIPT.json')
 code=p/'06_Generation_Code';code.mkdir(exist_ok=True);code_records=[]
 files=[f for f in HERE.iterdir() if f.is_file() and f.suffix in ['.py','.mjs','.ps1','.md']]+list((HERE/'report_helpers').glob('*.py'))
 for f in files:
  dest=code/f.relative_to(HERE);dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.resolve()!=f.resolve():shutil.copy2(f,dest)
  assert sha(dest)==sha(f);code_records.append({'original':str(f),'snapshot':dest.relative_to(p).as_posix(),'sha256':sha(f)})
 dump(q/'GENERATION_CODE_MANIFEST.json',code_records)
 versions={name:md.version(name) for name in ['numpy','pandas','matplotlib','seaborn','pillow','pyarrow']}
 dump(q/'runtime_versions.json',{'scientific_python_executable':sys.executable,'python_version':sys.version,'packages':versions,'document_python':'C:/Users/SDR_HH/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe','node':'C:/Users/SDR_HH/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe','latex':'TeX Live 2024 / pdfTeX 1.40.26','notes':'Existing runtimes reused; no package or environment installation. Artifact Tool runtime/font adapter paths are explicit in the supplied MJS sources.'})
 records=[]
 for f in sorted(p.rglob('*')):
  if f.is_file() and f.name!='DELIVERY_SHA256.json':records.append({'file':f.relative_to(p).as_posix(),'bytes':f.stat().st_size,'sha256':sha(f)})
 dump(q/'DELIVERY_SHA256.json',{'status':'SEALED','time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':records,'manifest_self_excluded':True})
 print(json.dumps({'status':'SEALED','files':len(records),'code_files':len(code_records),'identical_replay_outputs':len(matched)},ensure_ascii=False))
if __name__=='__main__':main()
