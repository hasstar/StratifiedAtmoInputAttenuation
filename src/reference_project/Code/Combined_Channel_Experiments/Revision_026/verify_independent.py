"""Offline evidence QA using BeautifulSoup, never imports the capture program."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
from urllib.parse import urljoin, urlsplit
import bs4
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_026'

def sha(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def main():
    assert not (OUT / 'independent_qa.json').exists(), 'one-shot QA output exists'
    records = {}
    pages = {}
    for name in ('nasa_gsdsu', 'nuwrf_11_5', 'luini_software', 'gsdsu_v2_release'):
        record = read(OUT / 'capture_run' / (name + '.json'))
        records[name] = record
        if record['access'] == 'HTTP_OK':
            path = ROOT / record['raw_path']
            assert sha(path) == record['sha256'] and path.stat().st_size == record['bytes']
            assert record['status_code'] == 200 and not record['redirects']
            soup = BeautifulSoup(path.read_bytes(), 'html.parser')
            for node in soup(['script', 'style']):
                node.decompose()
            visible = ' '.join(soup.get_text(separator=' ').split())
            assert visible == record['visible_text'], name
            links = sorted({urljoin(record['final_url'], a['href']) for a in soup.find_all('a', href=True) if a['href']})
            assert links == record['links'], name
            pages[name] = {'text': visible, 'links': links}
    assert set(pages) == {'nasa_gsdsu', 'nuwrf_11_5', 'luini_software'}
    nasa = pages['nasa_gsdsu']
    assert 'V2 of the G-SDSU CORE module' in nasa['text']
    assert 'V3 and beyond' in nasa['text'] and 'NASA Unified WRF' in nasa['text']
    assert records['gsdsu_v2_release']['requested_url'] in nasa['links']
    failed = records['gsdsu_v2_release']
    assert failed['access'] == 'ACCESS_UNRESOLVED'
    assert failed['error_type'] == 'ConnectionError' and 'Failed to resolve' in failed['error']
    assert 'status_code' not in failed and 'raw_path' not in failed
    manual = pages['nuwrf_11_5']['text']
    for phrase in ('requires users to sign a Software Usage Agreement',
                   'To get access the repository contact', 'version 4.0'):
        assert phrase in manual, phrase
    author = pages['luini_software']
    for phrase in ('Analysis tool for radio link budget', 'Conversion of rainfall statistics',
                   'Fitting complex analytic laws'):
        assert phrase in author['text'], phrase
    zip_links = [u for u in author['links'] if urlsplit(u).path.lower().endswith('.zip')]
    assert len(zip_links) == 9
    markers = ('radiosatmet', '3455567', 'parameterized radiopropagation', 'parametrized radiopropagation')
    marker_counts = {m: author['text'].lower().count(m) for m in markers}
    assert not any(marker_counts.values())  # Page-only diagnostic, not a universal absence proof.
    guard = read(OUT / 'sources_before.json')
    changed = [p for p, h in guard.items() if sha(p) != h]
    assert not changed, changed
    for p, h in read(OUT / 'code_before.json').items():
        assert sha(ROOT / p) == h, p
    for name, expected in {'local': '8efbb2d5981c0405940bfddbd7e7c6a3cdce4f62577f92efff46b1f6b0b81cae',
                           'uk': 'f91f9c7831a012ce29043eb477e54d4a4ddb5a5b4749f3558c081c2439e41408'}.items():
        assert sha(ROOT / ('Outputs/Experiments/Combined_Channel_Revision_011/predictions_' + name + '.parquet')) == expected
    state = read(OUT / 'model_status.json')
    expected_ids = ['proposed', 'bai_lstm', 'tabm', 'itu_combined', 'e_sst', 'rmop', 'nwp_chain']
    assert [m['id'] for m in state['models']] == expected_ids
    assert sum(m['hourly_status'] == 'COMPUTED_REUSED' for m in state['models']) == 4
    for model in state['models'][4:6]:
        assert model['numeric_results'] is None and model['rank'] is None
    prm = state['models'][6]['branches']['prm_hourly']
    mts = state['models'][6]['branches']['mts_seconds']
    assert prm['status'] == 'BLOCKED_NOT_COMPUTED' and prm['numeric_results'] is None
    assert mts['status'] == 'NOT_COMPUTED_USER_DECISION' and mts['numeric_results'] is None and mts['rank'] is None
    assert not mts['awaiting_user_materials'] and not mts['required_for_current_delivery']
    assert state['data_scope']['LOCAL_hours'] == 687 and state['data_scope']['uk_weather_hours'] == 5664
    assert state['data_scope']['uk_eligible_targets'] == 5593 and state['data_scope']['uk_common_physics_mask'] == 5538
    assert not state['all_seven_original_models_complete'] and not state['current_required_work_complete']
    for key in ('new_training', 'new_predictions', 'new_physical_attenuation_runs', 'new_scores', 'new_weather_requests'):
        assert state[key] == 0
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'status': 'PASS_SOURCE_EVIDENCE_ONLY',
              'independent_parser': 'BeautifulSoup ' + bs4.__version__, 'producer_not_imported': True,
              'successful_html_captures': len(pages), 'raw_bytes': sum(records[n]['bytes'] for n in pages),
              'text_and_link_roundtrip_exact': True, 'author_zip_links_inspected_as_links_only': zip_links,
              'author_page_marker_counts_not_universal_absence_proof': marker_counts,
              'v2_core_release_link_resolved_from_official_page': True,
              'v2_access_dns_failure_not_http_denial': True, 'protected_sources': len(guard),
              'changed_sources': changed, 'rev011_predictions_sha_match': True,
              'model_count': 7, 'reused_computed_count': 4, 'new_metrics_recomputed': 0,
              'all_seven_complete': False}
    with (OUT / 'independent_qa.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'author_zip_links_inspected_as_links_only'}, ensure_ascii=False))

if __name__ == '__main__':
    main()
