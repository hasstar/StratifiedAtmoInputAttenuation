"""Separate read-only delivery checks, then append one final QA record."""
import hashlib
import json
import msvcrt
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_012'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(name):
    return json.loads((OUT / name).read_text(encoding='utf-8-sig'))


def main():
    need(not (OUT / 'final_verification.json').exists(), 'Already published final QA')
    guards = [read('sources_before.json'), read('publication_sources.json'), read('delivery_manifest.json')]
    for guard in guards:
        for name, value in guard.items():
            need(sha(ROOT / name) == value, 'Changed source or output: ' + name)
    need(read('verification.json')['status'] == 'PASS', 'Calculation QA failed')
    text = (OUT / 'REPORT.md').read_text(encoding='utf-8')
    lines = text.splitlines()
    rows = read('seven_family_rankings.json')
    cohort_ids = ['LOCAL_test', 'UK_full_common', 'UK_chilbolton_full_common', 'UK_sparsholt_full_common']
    groups = [(c, e) for c in cohort_ids for e in ['rain', 'dry_overcast']]
    parsed_groups = []
    block = None
    for line in lines:
        if line.startswith('| 名次 | 模型 | N'):
            block = []
            parsed_groups.append(block)
        elif block is not None and line.startswith('|') and not line.startswith('|---'):
            cells = [x.strip() for x in line.strip('|').split('|')]
            block.append(cells)
        elif block is not None and not line.startswith('|'):
            block = None
    need(len(parsed_groups) == 8, 'Expected all eight report metric tables')
    verified = 0
    for (cohort, event), table in zip(groups, parsed_groups):
        expected = [r for r in rows if r['cohort'] == cohort and r['event'] == event]
        mapping = {r['name']: r for r in expected}
        need(len(table) == 7 and len({x[1] for x in table}) == 7, 'Seven unique models per table')
        n = next(r['n'] for r in expected if r['status'] == 'SCORED_FROZEN')
        for cells in table:
            need(len(cells) == 6, 'Report columns')
            r = mapping[cells[1]]
            need(cells[0] == (str(r['rank']) if r['rank'] is not None else '—'), 'Report rank')
            need(cells[2] == f'{r["n"]}/{n}', 'Report denominator')
            for i, key in enumerate(['rmse_db', 'mae_db', 'bias_db'], 3):
                if r[key + '_mean'] is None:
                    need(cells[i] == '未计算', 'Blocked metric fabricated')
                else:
                    values = cells[i].split(' ± ')
                    need(abs(float(values[0]) - r[key + '_mean']) <= 0.000000500001, 'Report metric rounding')
                    sd = r.get(key + '_sd')
                    need(len(values) == (1 if sd is None else 2), 'Report SD policy')
                    if sd is not None:
                        need(abs(float(values[1]) - sd) <= 0.000000500001, 'Report SD rounding')
            verified += 1
    expected_phrases = ['七类比较完成', '尚未产生合格预测', '5593', '5538', '5664',
                        '不是退回原 39 天', '不是置信区间', '不等于显著性检验结论',
                        '本文模型不是跨全部数据集和天气条件的最优模型']
    for phrase in expected_phrases:
        need(phrase in text, 'Missing coverage/disclosure: ' + phrase)
    matrix = (OUT / 'MODEL_INPUT_MATRIX.md').read_text(encoding='utf-8')
    need('28,320' in matrix and '不是 HTTP 请求数' in matrix, 'Candidate vs request count')
    need('不能将 RMOP 的五类水凝物要求照搬' in matrix, 'PRM/RMOP distinction')
    for name in ['REPORT.md', 'HANDOFF.md', 'MODEL_INPUT_MATRIX.md', 'SOURCE_LOOKUP.md']:
        need((OUT / name).stat().st_size > 1000, 'Missing deliverable')
    with (OUT / 'execution.lock').open('r+b') as f:
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
    result = {
        'status': 'PASS', 'utc': datetime.now(timezone.utc).isoformat(),
        'stage': 'four_frozen_families_scoring_and_seven_family_readiness',
        'all_seven_complete': False, 'report_metric_rows_verified': verified,
        'scored_rows': 32, 'blocked_rows': 24, 'guard_sets_verified': [len(g) for g in guards],
        'core_calculation_qa': 'PASS', 'delivery_sha_qa': 'PASS',
        'execution_lock': 'acquired and released without modifying bytes',
        'no_new_training': True, 'no_new_model_inference': True, 'no_weather_downloads': True,
        'automation_modified_this_round': False,
        'delivery_manifest_sha256': sha(OUT / 'delivery_manifest.json'),
        'verification_script_sha256': sha(Path(__file__)),
        'note': 'Independent arithmetic QA is verify.py; this script checks publication consistency, not an external peer review.'
    }
    with (OUT / 'final_verification.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
