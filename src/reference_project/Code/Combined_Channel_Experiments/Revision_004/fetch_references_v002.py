"""Resume the initial reference-only request after sandbox socket denial.

Original script/failure/snapshots retained. No URL, proxy, retry or formula change.
Reuse and verify the before snapshot; all reference destination files must be new.
"""
import fetch_references as prior
from common import *
import traceback

def reuse_snapshot():
    guard=read(OUT/'source_sha256_before.json');protected_verify(guard)
    for key,digest in read(OUT/'index_sha256_before.json').items():
        require(sha(ROOT/key)==digest,'Fixed index changed during reference acquisition')
    require(not (OUT/'references_complete.json').exists(),'References already complete')
    for name in prior.FILES:require(not (OUT/'references'/name).exists(),'Unexpected existing reference')
    emit(OUT/'reference_resume_002.json',{'time':now(),'reason':'Initial sandbox WinError10013, no source response or bytes acquired; authorized network escalation for same four official references','sha256':sha(Path(__file__))})
    return guard

if __name__=='__main__':
    prior.snapshot=reuse_snapshot
    try:prior.main()
    except Exception:
        emit(OUT/('failure_reference_v002_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.json'),{'time':now(),'traceback':traceback.format_exc()});raise
