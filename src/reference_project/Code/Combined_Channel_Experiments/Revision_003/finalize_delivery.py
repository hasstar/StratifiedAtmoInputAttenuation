"""Check and seal the offline delivery; never import or execute a model."""
from __future__ import annotations
import os
import re
from pathlib import Path

import pandas as pd
from common import (ROOT, OUT, HERE, ACQ, INDEXES, emit, read, sha, now,
                    require, protected_verify, offline, lock)


def main():
    offline()
    require(not (OUT / 'delivery_verification.json').exists(), 'Delivery already sealed')
    with lock():
        audit = read(OUT / 'independent_qa.json')
        require(audit['status'] == 'PASS', 'Independent numeric verification')
        guard = read(OUT / 'source_sha256_used.json')
        protected_verify(guard)
        require(len(guard) == 6981, 'Frozen source count')
        root_readme = ROOT.parents[2] / 'README.md'
        require(sha(root_readme) == '39741aa8c69136661b4b02948b29714a3654f3f989ab7b488a54f0c02f93f7c7',
                'Previously fully read root rules changed')

        runtime = read(OUT / 'runtime_observation.json')
        for sample in runtime['samples']:
            require(sample['all_python_count'] == 0 and sample['missing_command_line_count'] == 0,
                    'Pre-delivery OS process snapshot ambiguous')
        auto_path = Path(runtime['automation_path'])
        require(sha(auto_path) == runtime['automation_sha256'], 'Paused automation changed')
        auto_text = auto_path.read_text(encoding='utf-8')
        require(re.search(r'^status\s*=\s*"PAUSED"\s*$', auto_text, re.M), 'Automation is not paused')
        stop_path = ACQ / 'STOP_REQUESTED.json'
        require(read(stop_path)['reason'] == 'Direct human stop instruction', 'Stop marker missing')
        require(sha(stop_path) == guard[str(stop_path.relative_to(ROOT))], 'Stop marker changed')

        report = (OUT / 'COMPARISON_REPORT.md').read_text(encoding='utf-8')
        summary = pd.read_csv(OUT / 'family_summary.csv')
        metrics = pd.read_csv(OUT / 'metrics.csv')
        require(len(summary) == 562 and len(metrics) == 1112, 'Table row count')
        family_labels = [
            ('ours_frozen_top10', '本文 Top10 BiLSTM，冻结', '本文 Top10 BiLSTM'),
            ('train_mean', '训练均值常数', '训练均值常数'),
            ('ridge', 'Ridge', 'Ridge'),
            ('elastic_net', 'ElasticNet', 'ElasticNet'),
            ('svr_rbf', 'RBF-SVR', 'RBF-SVR'),
            ('random_forest', '随机森林', '随机森林'),
            ('xgboost', 'XGBoost', 'XGBoost'),
            ('bai2022_mlp_adapted', 'Bai2022 MLP，输入/实现适配', 'Bai2022 MLP 适配'),
            ('bai2022_lstm_t0_adapted', 'Bai2022 LSTM-T0，输入/实现适配', 'Bai2022 LSTM-T0 适配'),
            ('tabm2025_official_budgeted', 'TabM2025，官方架构/固定预算', 'TabM2025 固定预算')]

        def value(cohort, family):
            rows = summary[(summary.cohort == cohort) & (summary.group == 'overall') &
                           (summary.family == family)]
            require(len(rows) == 1, 'Unique summary family/cohort')
            row = rows.iloc[0]
            text = f'{row.rmse_mean_db:.4f}'
            if row.runs > 1:
                text += f' ± {row.rmse_seed_sd_db:.4f}'
            return text

        for family, main_label, period_label in family_labels:
            require(f'| {main_label} | {value("LOCAL_test", family)} | {value("UK_all", family)} |' in report,
                    'Report main table disagrees with CSV: ' + family)
            require(f'| {period_label} | {value("UK_original39", family)} | {value("UK_extension2003", family)} |' in report,
                    'Report period table disagrees with CSV: ' + family)

        data = pd.read_parquet(OUT / 'predictions_expanded.parquet')
        require(len(data) == 5664 and int(data.primary_score_eligible.sum()) == 5593, 'Report coverage')
        require(data[['station', 'hour_utc']].duplicated().sum() == 0, 'Unique station hours')
        require(data.hour_utc.dt.date.nunique() == 143, 'Distinct UTC-date count')
        for station, hours, eligible in [('chilbolton', 3432, 3382), ('sparsholt', 2232, 2211)]:
            d = data[data.station == station]
            require(len(d) == hours and int(d.primary_score_eligible.sum()) == eligible, 'Station report count')
        actual_bands = data.loc[data.primary_score_eligible, 'rain_band'].value_counts().to_dict()
        require(actual_bands == {'negative_difference': 55, 'below_0.1': 4793, '0.1_to_1': 648, 'at_least_1': 97},
                'Report diagnostic counts')
        for family in ('ours_frozen_top10', 'elastic_net_seed17'):
            rows = [r for r in read(OUT / 'date_cluster_bootstrap.json')['results']
                    if r['cohort'] == 'UK_all' and r['reference'] == 'train_mean' and r['model'] == family]
            require(len(rows) == 1, 'Interval report identity')
            r = rows[0]
            require(f'+{r["delta_rmse_db"]:.4f}' in report and
                    f'[{r["unadjusted_ci95_db"][0]:.4f}, {r["unadjusted_ci95_db"][1]:.4f}]' in report,
                    'Reported interval does not match actual bootstrap')

        for name in ('plotted_summary.csv', 'plotted_runs.csv'):
            require((OUT / 'figures' / name).read_bytes() == (OUT / 'figures_v002' / name).read_bytes(),
                    'Figure layout revision altered plotted data')
        for name in ('local_and_expanded_uk', 'original_vs_new_uk'):
            require((OUT / 'figures_v002' / (name + '.png')).read_bytes().startswith(b'\x89PNG\r\n\x1a\n'),
                    'PNG delivery signature')
            require((OUT / 'figures_v002' / (name + '.pdf')).read_bytes().startswith(b'%PDF-'),
                    'PDF delivery signature')

        indexes_before = read(OUT / 'index_sha256_before.json')
        index_evidence = {}
        index_blocks = []
        for source in INDEXES:
            rel = str(source.relative_to(ROOT))
            original = OUT / 'document_snapshots/before' / source.relative_to(ROOT)
            require(sha(original) == indexes_before[rel], 'Before snapshot hash')
            current = source.read_bytes()
            pattern = rb'<!-- REV003-BEGIN -->\r?\n.*?<!-- REV003-END -->\r?\n\r?\n'
            blocks = re.findall(pattern, current, re.S)
            require(len(blocks) == 1, 'One scoped index insertion')
            require(re.sub(pattern, b'', current, flags=re.S) == original.read_bytes(),
                    'Old index body bytes were changed')
            index_blocks.append((source, blocks[0].decode('utf-8')))
            index_evidence[rel] = {'before_sha256': indexes_before[rel], 'after_sha256': sha(source),
                                   'old_body_byte_identical_after_removing_new_block': True}

        documents = [(p, p.read_text(encoding='utf-8')) for p in sorted(OUT.glob('*.md'))]
        documents += [(OUT / 'figures_v002/visual_qa.md', (OUT / 'figures_v002/visual_qa.md').read_text(encoding='utf-8'))]
        documents += index_blocks
        links = []
        for source, text in documents:
            require('{{' not in text and 'TODO' not in text, 'Unfilled delivery placeholder')
            for raw in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', text):
                target = raw.strip().strip('<>')
                if target.startswith(('https://', 'http://', '#')):
                    continue
                resolved = (source.parent / target.split('#')[0]).resolve()
                require(resolved.is_relative_to(ROOT), 'Unexpected external local link')
                is_final_receipt = resolved == (OUT / 'delivery_verification.json').resolve()
                require(resolved.exists() or is_final_receipt, f'Broken link: {source} => {raw}')
                links.append({'source': str(source.relative_to(ROOT)), 'target': str(resolved.relative_to(ROOT)),
                              'created_by_final_step': is_final_receipt})

        # Snapshot only after verifying that original historical bytes remain intact.
        for source in INDEXES:
            dest = OUT / 'document_snapshots/after' / source.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open('xb') as f:
                f.write(source.read_bytes())
        emit(OUT / 'index_sha256_after.json', index_evidence)
        emit(OUT / 'delivery_link_check.json', {'status': 'PASS', 'local_links': links,
                                               'external_urls_not_refetched': True})
        protected_verify(guard)
        files = {}
        for directory in (HERE, OUT):
            for p in sorted(directory.rglob('*')):
                if p.is_file() and p.suffix != '.lock' and '__pycache__' not in p.parts:
                    files[str(p.relative_to(ROOT))] = {'sha256': sha(p), 'bytes': p.stat().st_size}
        emit(OUT / 'delivery_manifest.json', {'created': now(), 'files': files,
                                            'excludes': ['evaluation.lock', 'delivery_manifest.json',
                                                         'delivery_verification.json']})
        receipt = {
            'status': 'PASS', 'scope_status': 'COMPLETED',
            'scope': 'Available-data LOCAL test and expanded-UK frozen hourly comparison package',
            'historical_T012_all_goals_complete': False, 'scientific_acceptance': 'Awaiting user review',
            'created': now(), 'finalizer_pid': os.getpid(),
            'protected_source_count': len(guard), 'source_sha256_unchanged': True,
            'independent_numeric_qa_sha256': sha(OUT / 'independent_qa.json'),
            'delivery_manifest_sha256': sha(OUT / 'delivery_manifest.json'),
            'document_local_link_count': len(links), 'main_report_numeric_rows_checked': 20,
            'index_original_body_byte_preserved': True,
            'figure_revision_data_identical': True, 'adopted_figure_version': 'figures_v002',
            'new_fit_calls': 0, 'new_downloads': 0,
            'download_stop_marker_sha256': sha(stop_path), 'tvt_status': 'PAUSED',
            'tvt_configuration_sha256': sha(auto_path),
            'os_process_observation': 'runtime_observation.json',
            'no_background_worker_started_by_finalizer': True,
            'uk_physical_models': 'Not executed: unaudited rain/zero-degree height and matching polarization',
            'failures_and_first_figure_version_preserved': True,
            'execution_lock': 'RELEASED', 'finalizer_sha256': sha(Path(__file__))}
    # A successful context exit has released the OS byte lock; seal afterwards.
    emit(OUT / 'delivery_verification.json', receipt)
    require((OUT / 'delivery_verification.json').exists(), 'Final receipt exists')
    print({'status': receipt['status'], 'scope_status': receipt['scope_status'],
           'protected_sources': len(guard), 'artifact_files': len(files),
           'checked_links': len(links), 'execution_lock': 'RELEASED'}, flush=True)


if __name__ == '__main__':
    main()
