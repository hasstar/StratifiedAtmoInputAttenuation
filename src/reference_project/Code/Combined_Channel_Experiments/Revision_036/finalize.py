"""Final artifact QA and write-once seal; does not rerun any experiment."""
from pathlib import Path
from datetime import datetime, timezone, timedelta, date
from collections import Counter
from urllib.parse import unquote
import hashlib
import json
import os
import re
import subprocess
import traceback

ROOT = Path(__file__).resolve().parents[3]
CODE = Path(__file__).parent
OUT = ROOT/"Outputs/Experiments/Combined_Channel_Revision_036"
def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()
def put(name, data):
    with (OUT/name).open("x", encoding="utf-8") as f:
        json.dump(data,f,indent=2,ensure_ascii=False,allow_nan=False)
        f.write("\n")
def text(p):
    return Path(p).read_text(encoding="utf-8-sig")
def need(condition, message):
    if not condition:
        raise AssertionError(message)
def close_display(cell, mean, deviation=None):
    pieces = cell.split("±")
    need(abs(float(pieces[0])-mean) <= 0.000000501, "Rendered mean mismatch")
    if deviation is None:
        need(len(pieces)==1, "Single-run fake SD")
    else:
        need(len(pieces)==2 and abs(float(pieces[1])-deviation)<=0.000000501, "Rendered SD mismatch")


def main():
    need(not (OUT/"final_verification.json").exists(), "Seal already exists")
    qa = read(OUT/"verification.json")
    need(qa["status"]=="PASS_FINAL_NUMERIC_CONSOLIDATION", "Missing numerical QA")
    sources = {**read(OUT/"sources_before.json"), **read(OUT/"additional_source_pins.json")}
    for p,h in sources.items():
        need(sha(p)==h, "Protected source changed: "+p)
    for name,h in read(OUT/"calculation_complete.json")["output_sha256"].items():
        need(sha(OUT/name)==h, "Consolidated numeric output changed: "+name)
    pub = read(OUT/"publication_build.json")
    formatting = []
    for name,h in pub["rendered_sha256"].items():
        if sha(OUT/name)!=h:
            before = OUT/"draft_snapshots"/name
            need(before.exists() and sha(before)==h, "Rendered draft not preserved")
            need([x for x in text(before).splitlines() if x.strip()] == [x for x in text(OUT/name).splitlines() if x.strip()], "Non-formatting change after rendering")
            formatting.append({"path":str(OUT/name),"before_sha256":h,"after_sha256":sha(OUT/name),"kind":"blank-line-only; original draft retained"})
    needed = ["REPORT.md","MODEL_INPUT_MATRIX.md","HOURLY_RANKINGS.md","DYNAMIC_RESULTS.md","COVERAGE_AND_MISSING.md","QA.md","HANDOFF.md","PROTOCOL.md","COMMANDS.md","all_hour_ledger.json","coverage_by_date.json","native_calendar_ledger.json","sample_membership.json","model_source_index.json","claims.json","verification.json"]
    need(all((OUT/p).is_file() for p in needed), "Missing contracted artifact")
    links = []
    for p in OUT.glob("*.md"):
        value = text(p)
        need("\ufffd" not in value, "Unicode replacement in document")
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)",value):
            if target.startswith(("http://","https://","#")):
                continue
            q = (p.parent/unquote(target.split("#")[0].strip("<>"))).resolve()
            need(q.exists() or q in (OUT/"manifest.json",OUT/"final_verification.json"), "Broken local link: "+str(q))
            links.append(str(q))

    names = {"本文冻结Top10":"ours_frozen_top10", "Bai LSTM LOCAL适配":"bai2022_lstm_t0_adapted", "TabM LOCAL适配":"tabm2025_official_budgeted", "组合ITU近似":"itu_combined_geometric_approx"}
    cohort = {"LOCAL液态雨输入有效补充（684小时母集）":"LOCAL_liquid_input_valid_supplement", "LOCAL原687小时历史参考（含3小时输入异常）":"LOCAL_legacy_687_reference", "英国两站合并":"UK_full_common", "Chilbolton":"UK_chilbolton_full_common", "Sparsholt":"UK_sparsholt_full_common"}
    event = {"雨小时":"rain","干燥阴小时":"dry_overcast"}
    hours = {(r["cohort"],r["event"],r["family"]):r for r in read(OUT/"hourly_family_metrics.json")}
    seen = set()
    for line in text(OUT/"HOURLY_RANKINGS.md").splitlines():
        if line.startswith("## "):
            c = cohort[line[3:]]
        elif line.startswith("### "):
            e = event[line[4:]]
        elif re.match(r"\| [1-4] \|",line):
            x = [a.strip() for a in line.strip("|").split("|")]
            k = c,e,names[x[1]]
            need(k not in seen, "Duplicate rendered hourly row")
            seen.add(k)
            r = hours[k]
            need(int(x[0])==r["rank"] and int(x[5])==r["n"],"Rendered hourly rank/count")
            for col,m in zip((2,3,4),("rmse_db","mae_db","bias_db")):
                close_display(x[col],r[m+"_mean"],r[m+"_sd"])
    need(seen==set(hours) and len(seen)==40,"All 40 hourly rows rendered")

    dynamic={(r["cohort"],r["event"],r["weighting"],r["family"]):r for r in read(OUT/"dynamic_family_metrics.json")}
    dyncohort={**cohort,"英国两站合并":"UK","Chilbolton":"chilbolton","Sparsholt":"sparsholt"}
    short={"本文冻结Top10":"ours","Bai LSTM LOCAL适配":"bai_lstm","TabM LOCAL适配":"tabm","组合ITU近似":"itu"}
    dseen=set()
    trows={(r["station"],r["event"],r["weighting"],r["candidate"]):r for r in read(OUT/"temporal_reuse.json")["rows"]}
    tseen=set()
    temporal=False
    for line in text(OUT/"DYNAMIC_RESULTS.md").splitlines():
        if line=="## 时间结构：英国两站":
            temporal=True
        if not temporal:
            if line in ("## 日期等权","## 小时等权"):
                w="date_equal" if "日期" in line else "hour_equal"
            elif line.startswith("### ") and line[4:] in dyncohort:
                c=dyncohort[line[4:]]
            elif line.startswith(("| 雨小时 |","| 干燥阴小时 |")):
                x=[a.strip() for a in line.strip("|").split("|")]
                k=c,event[x[0]],w,short[x[2].removesuffix("＋冻结快变")]
                need(k not in dseen,"Duplicate rendered dynamic row")
                dseen.add(k);r=dynamic[k]
                need(x[1]==str(r["n_station_hours"])+"/"+str(r["n_dates"]) and int(x[7])==r["rank_crps201"],"Rendered dynamic counts/rank")
                for col,m in ((3,"crps201_db"),(4,"crps401_db")):
                    close_display(x[col],r[m],r[m+"_sd"])
                for col,m in ((5,"central90_coverage"),(6,"central98_coverage")):
                    close_display(x[col],r[m])
        else:
            if line.startswith("### "):
                w="date_equal" if "日期" in line else "hour_equal"
            elif line.startswith(("| 英国合并 |","| Chilbolton |","| Sparsholt |")):
                x=[a.strip() for a in line.strip("|").split("|")]
                s={"英国合并":"pooled","Chilbolton":"chilbolton","Sparsholt":"sparsholt"}[x[0]]
                for col,candidate in ((3,"iid"),(4,"block_0256")):
                    k=s,event[x[1]],w,candidate
                    need(k not in tseen,"Duplicate temporal display")
                    tseen.add(k);r=trows[k]
                    need(x[2]==str(r["n_station_hours"])+"/"+str(r["n_dates"]),"Temporal display counts")
                    close_display(x[col],r["vs32"])
    need(dseen==set(dynamic) and len(dseen)==80,"All 80 dynamic rows displayed")
    need(tseen==set(trows) and len(tseen)==24,"All temporal rows displayed")

    claims=read(OUT/"claims.json")
    facts=claims["claims"][0]["facts"]
    for fact in facts:
        r=hours[(fact["cohort"],fact["event"],fact["family"])]
        need(r["rank"]==1 and fact["rmse_db_mean"]==r["rmse_db_mean"] and fact["n"]==r["n"],"Winner claim evidence")
    winner_seen=set()
    for line in text(OUT/"REPORT.md").splitlines():
        if not line.startswith("| "):
            continue
        x=[a.strip() for a in line.strip("|").split("|")]
        if x[0] not in cohort or x[1] not in event:
            continue
        k=cohort[x[0]],event[x[1]],names[x[2]]
        r=hours[k]
        need(r["rank"]==1 and int(x[4])==r["n"],"Report winner/count")
        close_display(x[3],r["rmse_db_mean"])
        winner_seen.add(k)
    need(len(winner_seen)==8,"Eight winner statements")

    cal=read(OUT/"native_calendar_ledger.json")
    plan=read(cal["source"])
    need(sha(cal["source"])==cal["source_sha256"],"Calendar source hash")
    native={(r["station"],r["date_utc"]) for r in plan["native_days"]}
    included={(r["station"],r["date"]) for r in read(OUT/"coverage_by_date.json") if r["station"]!="LOCAL"}
    cm={(r["station"],r["date_utc"]):r for r in cal["rows"]}
    need(len(cm)==len(cal["rows"])==1132,"Calendar uniqueness")
    need({k for k,r in cm.items() if r["native_file_present"]}==native,"Native calendar identity")
    need({k for k,r in cm.items() if r["in_completed_weather_cohort"]}==included,"Completed calendar identity")
    for s,span in plan["scope_counts"].items():
        ds=sorted(date.fromisoformat(k[1]) for k in cm if k[0]==s)
        need(ds[0].isoformat()==span["first"] and ds[-1].isoformat()==span["last"],"Calendar endpoints")
        need(all(b-a==timedelta(days=1) for a,b in zip(ds,ds[1:])),"Calendar gaps explicitly listed")
    need(Counter(r["status"] for r in cal["rows"])==cal["counts"],"Calendar status totals")

    # Read actual related Python process identities; no stopping or remote calls.
    command = "Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|pythonw' -and $_.CommandLine -match 'Combined_Channel|weather_download|trial.py' } | Select-Object ProcessId,Name,CreationDate,CommandLine | ConvertTo-Json -Depth 3 -Compress"
    proc=subprocess.run(["powershell.exe","-NoProfile","-Command",command],capture_output=True,text=True,encoding="utf-8",errors="replace",check=True)
    actual=json.loads(proc.stdout) if proc.stdout.strip() else []
    if isinstance(actual,dict):
        actual=[actual]
    other=[p for p in actual if p["ProcessId"]!=os.getpid()]
    need(not other,"Another related worker is active")
    stop=ROOT/"Outputs/Experiments/Combined_Channel_Revision_001/external/weather_gbs_extend_v011/STOP_REQUESTED.json"
    need(sha(stop)=="0d629e054904ca58dde38630af24899bf89620607410da97a9b49f259253864a","STOP changed")
    for p,h in sources.items():
        need(sha(p)==h,"Source changed during file QA: "+p)
    put("delivery_qa.json",{"status":"PASS_READY_TO_SEAL","utc":datetime.now(timezone.utc).isoformat(),
        "rendered_hourly_rows":40,"rendered_dynamic_rows":80,"rendered_temporal_candidate_rows":24,"report_winner_rows":8,
        "calendar_rows":1132,"local_links_checked":len(links),"protected_source_pins":len(sources),
        "formatting_only_changes":formatting,"other_related_workers":other,"self_pid":os.getpid(),
        "stop_sha256":sha(stop),"numerical_verification_sha256":sha(OUT/"verification.json"),"finalizer_sha256":sha(__file__),
        "human_manuscript_approval":False,"independent_human_review_claimed":False})
    files={str(p):sha(p) for directory in (CODE,OUT) for p in directory.rglob("*") if p.is_file() and p.name not in ("manifest.json","final_verification.json")}
    put("manifest.json",files)
    need(all(sha(p)==h for p,h in read(OUT/"manifest.json").items()),"Manifest readback mismatch")
    put("final_verification.json",{"status":"PASS_FINAL_FOUR_FAMILY_PACKAGE","utc":datetime.now(timezone.utc).isoformat(),
        "four_family_final_package_complete":True,"formal_family_count":4,"all_contracted_deliverables_present":True,
        "scientific_numeric_qa":"PASS_FINAL_NUMERIC_CONSOLIDATION","file_qa":"PASS_READY_TO_SEAL",
        "protected_source_pins":len(sources),"manifest_files":len(files),"manifest_sha256":sha(OUT/"manifest.json"),
        "lock_status":"RELEASED","other_related_workers":other,"self_pid_exits_after_seal":os.getpid(),
        "new_training":0,"new_predictions":0,"new_trajectories":0,"new_weather_calls":0,
        "esst":"EXCLUDED_USER_DECISION_DEADLINE","rmop":"EXCLUDED_USER_DECISION","nwp_prm":"EXCLUDED_USER_DECISION","mts":"NOT_COMPUTED_USER_DECISION",
        "automation_next_action":"PAUSE_EXISTING_TVT_ONLY; separate before/after control change record required",
        "prior_pending_snapshots_are_immutable":True,"human_manuscript_approval":"NOT_CLAIMED"})
    need(all(sha(p)==h for p,h in files.items()),"Post-seal file mismatch")
    need(all(Path(p).exists() for p in links),"Final local link unresolved")
    print("PASS_FINAL_FOUR_FAMILY_PACKAGE",len(files),"sealed files",len(sources),"source pins")


if __name__=="__main__":
    try:
        main()
    except BaseException:
        stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
        put("finalize_failure_"+stamp+".json",{"traceback":traceback.format_exc()})
        raise
