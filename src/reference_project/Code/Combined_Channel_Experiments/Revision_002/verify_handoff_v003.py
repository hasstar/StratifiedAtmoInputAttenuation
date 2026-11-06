"""Final checks v003: UTF-8 paths and final post-link-repair index snapshots."""
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    root = Path(__file__).resolve().parents[3]
    out = root / 'Outputs/Experiments/Combined_Channel_Revision_002'
    receipt = out / 'handoff_verification.json'
    if receipt.exists():
        raise FileExistsError(receipt)
    protected = json.loads((out / 'source_sha256_before.json').read_text(encoding='utf-8-sig'))
    assert len(protected) == 548
    for path, sha in protected.items():
        assert digest(Path(path)) == sha, path
    indexes = ['Tasks/T012_Combined_Channel_And_External_Validation.md',
               'Code/Combined_Channel_Experiments/Revision_001/PROGRESS.md']
    index_checks = []
    for relative in indexes:
        current = root / relative
        before = out / 'document_snapshots/before' / relative
        after = out / 'document_snapshots/after_v002' / relative
        assert str(current) not in protected
        assert digest(current) == digest(after)
        original = before.read_text(encoding='utf-8-sig')
        revised = current.read_text(encoding='utf-8-sig')
        assert original.split('\n', 1)[0] == revised.split('\n', 1)[0]
        assert original.split('\n', 1)[1].strip() in revised
        assert len(revised) > len(original)
        index_checks.append(dict(path=relative, before_sha256=digest(before),
                                 after_sha256=digest(after), original_body_preserved=True))
    selection = json.loads((out / 'frozen_selection.json').read_text(encoding='utf-8-sig'))
    checkpoint_count = 0
    for entry in selection['models']:
        if entry['model_type'] == 'constant':
            continue
        assert digest(out / entry['path']) == entry['sha256'], entry['id']
        checkpoint_count += 1
    assert checkpoint_count == 18
    assert json.loads((out / 'independent_qa.json').read_text(encoding='utf-8-sig'))['status'] == 'PASS'
    assert digest(out / 'figures/plot_values.csv') == digest(out / 'figures_v002/plot_values.csv')
    assert 'Status: PASS' in (out / 'figures_v002/visual_qa.md').read_text(encoding='utf-8')
    automation = Path('C:/Users/SDR_HH/.codex/automations/tvt/automation.toml').read_text(encoding='utf-8')
    assert re.search(r'^status\s*=\s*"PAUSED"\s*$', automation, re.M)
    stop = root / 'Outputs/Experiments/Combined_Channel_Revision_001/external/weather_gbs_extend_v011/STOP_REQUESTED.json'
    assert stop.exists() and str(stop) in protected and digest(stop) == protected[str(stop)]
    docs = [out / 'HANDOFF.md', out / 'COMPARISON_REPORT.md',
            out / 'interpretation_001/INTERPRETATION_ADDENDUM.md',
            root / 'Code/Combined_Channel_Experiments/Revision_002/MODEL_REGISTRY.md']
    links = []
    for doc in docs:
        for target in re.findall(r'\]\(([^)]+)\)', doc.read_text(encoding='utf-8-sig')):
            if target.startswith(('https://', 'http://', '#')):
                continue
            resolved = (doc.parent / target.split('#', 1)[0]).resolve()
            assert resolved == receipt or resolved.exists(), (doc, target)
            links.append(dict(document=str(doc.relative_to(root)), target=target))
    for relative in indexes:
        doc = root / relative
        before = out / 'document_snapshots/before' / relative
        text = doc.read_text(encoding='utf-8-sig')
        old_body = before.read_text(encoding='utf-8-sig').split('\n', 1)[1].strip()
        new_prefix = text.split(old_body, 1)[0]
        for target in re.findall(r'\]\(([^)]+)\)', new_prefix):
            assert (doc.parent / target).resolve().exists(), (doc, target)
            links.append(dict(document=relative, target=target))
    artifacts = {str(p.relative_to(out)): digest(p) for p in out.rglob('*') if p.is_file()}
    code_dir = root / 'Code/Combined_Channel_Experiments/Revision_002'
    sources = {str(p.relative_to(code_dir)): digest(p) for p in code_dir.rglob('*') if p.is_file()}
    result = dict(status='PASS', stage='COMPARISON_BATCH_001_VERIFIED',
                  verified_utc=datetime.now(timezone.utc).isoformat(),
                  execution_lock='RELEASED', task_T012='IN_PROGRESS',
                  human_scientific_acceptance='PENDING', protected_sources_unchanged=len(protected),
                  checkpoint_count=checkpoint_count, saved_artifact_sha256=artifacts,
                  code_sha256=sources, fixed_indexes=index_checks, local_links_checked=links,
                  plot_values_unchanged=True, adopted_figure='figures_v002',
                  weather_stop_preserved=True, tvt_status='PAUSED',
                  process_observation={'time_bjt': '2026-10-16T15:04:34.3269569+08:00',
                                       'source': 'successful elevated Win32_Process CIM query via exec_command',
                                       'all_python_count': 0, 'weather_or_comparison_count': 0})
    with receipt.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: result[k] for k in ['status', 'stage', 'execution_lock',
                                            'protected_sources_unchanged', 'checkpoint_count',
                                            'weather_stop_preserved', 'tvt_status']}))


if __name__ == '__main__':
    main()
