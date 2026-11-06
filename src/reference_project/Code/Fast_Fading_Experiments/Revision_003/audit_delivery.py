"""Independent exported-table arithmetic checks; visual acceptance remains manual."""
import csv,math,json
from common import *

def main():
    need(read(OUT/'independent_numeric_qa.json')['status']=='PASS','Numeric gate')
    need(read(DELIVERY/'RFE_WORKBOOK_QA.json')['status']=='PASS','RFE workbook gate')
    with (DELIVERY/'快变与TabM组合_逐小时结果.csv').open(encoding='utf-8-sig',newline='') as f:hours=list(csv.DictReader(f))
    with (DELIVERY/'快变与TabM组合_分天气汇总.csv').open(encoding='utf-8-sig',newline='') as f:summary=list(csv.DictReader(f))
    need(len(hours)==6623 and len(summary)==25,'Export count')
    need(len({(r['dataset'],r['split'],r['station'],r['hour']) for r in hours})==6623,'Unique exported hours')
    metrics=['fast_crps_db','fast_zero_crps_db','combined_crps_db','slow_only_crps_db','combined_minus_slow_db','observed_sd_db','generated_sd_mean_db','sd_absolute_error_db','fast_coverage90','combined_coverage90']
    maxdiff=0
    for s in summary:
        part=[r for r in hours if (('LOCAL_'+r['split']==s['cohort'] and r['dataset']=='LOCAL') or (s['cohort']=='UK' and r['dataset']=='UK')) and (s['station']=='pooled' or r['station']==s['station']) and (s['event']=='overall' or r['event']==s['event'])]
        need(len(part)==int(s['hours']),'Summary support count')
        need(sum(int(r['n_seconds']) for r in part)==int(s['valid_seconds']),'Summary second count')
        need(len({r['date'] for r in part})==int(s['dates']),'Summary date count')
        for m in metrics:
            expected=math.fsum(float(r[m]) for r in part)/len(part)
            diff=abs(expected-float(s[m]));maxdiff=max(maxdiff,diff);need(diff<1e-11,'Summary mean '+m)
        expected=100*(float(s['slow_only_crps_db'])-float(s['combined_crps_db']))/float(s['slow_only_crps_db'])
        need(abs(expected-float(s['combined_improvement_percent']))<1e-10,'Percent arithmetic')
    with (DELIVERY/'快变_ACF分天气汇总.csv').open(encoding='utf-8-sig',newline='') as f:lags=list(csv.DictReader(f))
    need(len(lags)==275,'ACF export count')
    need(len(list((DELIVERY/'图表').glob('*.png')))==6 and len(list((DELIVERY/'图表').glob('*.pdf')))==6,'Six figure pairs')
    for name in ('快变与TabM组合_分天气汇总.csv','快变与TabM组合_逐小时结果.csv','快变_ACF分天气汇总.csv','结果说明.md'):
        need((DELIVERY/name).stat().st_size>100,'Missing delivery file')
    for p,h in read(OUT/'manifest.json')['sources'].items():need(sha(ROOT/p)==h,'Protected source changed')
    result={'status':'PASS','utc':now(),'hourly_rows':6623,'summary_groups':25,'lag_rows':275,'exported_arithmetic_max_abs_diff':maxdiff,'source_hashes_unchanged':True,'figure_pairs':6,'visual_qa_separate':True,'auditor_sha256':sha(CODE/'audit_delivery.py')}
    write(DELIVERY/'SUMMARY_QA.json',result)
    print(result)
if __name__=='__main__':main()
