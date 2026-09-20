#!/usr/bin/env python3
"""Measure a judge's self-agreement: identical claims, judged repeatedly, uncached.

A verdict cache makes a score reproducible; it does not make the pinned verdict
correct. This measures what the score would have been had the judge sampled
differently, which is the error bar on any comparison the benchmark makes.

    python3 tools/judge_agreement_probe.py --run-dir results/<run> --sample 120 --repeats 3

Reports the flip rate per claim kind. The number worth watching is the split: claims
checked against a page are stable, while the structural judgements (`is_program`,
`reputation`) flip often enough to matter, and they carry most of the score.
"""

import importlib.util, json, os, pathlib, random, statistics, sys
from concurrent.futures import ThreadPoolExecutor
REPO = pathlib.Path.cwd(); sys.path.insert(0, str(REPO/"src"))
for line in (REPO/".env").read_text().splitlines():
    line=line.strip()
    if line and not line.startswith("#") and "=" in line:
        k,v=line.split("=",1); os.environ.setdefault(k,v)
from benchkit.case import Prediction
from benchkit.judge import (REPUTATION_CLAIM, VERDICT_SYSTEM, is_program_claim,
                            relevance_claim, batch_prompt)
from benchkit.llm import LIMITER, LiteLLMClient
from benchkit.run import as_prediction
from benchkit.sources import SourceText, attach_sources
spec=importlib.util.spec_from_file_location("m", REPO/"benchmarks/us-startup-programs/benchmark.py")
m=importlib.util.module_from_spec(spec); sys.modules["m"]=m; spec.loader.exec_module(m)
b=m.StartupPrograms(); st=SourceText(REPO/"results/.sources-r6", allow_fetch=False)
import argparse
_ap=argparse.ArgumentParser()
_ap.add_argument("--run-dir", required=True)
_ap.add_argument("--sample", type=int, default=120)
_ap.add_argument("--repeats", type=int, default=3)
_args=_ap.parse_args()
RUN_DIR=_args.run_dir; SAMPLE, REPEATS = _args.sample, _args.repeats

calls=[]
for line in open(REPO/RUN_DIR/"responses.jsonl"):
    row=json.loads(line)
    if row.get("status")!="ok": continue
    recs,_=b._merge_duplicates(b._records(as_prediction(row)))
    attach_sources(recs, st)
    for r in recs:
        if not r.get("source_text"): continue
        for f,w in m.CLAIM_WEIGHTS.items():
            if r.get(f): calls.append((f, w, f"{r.get('name')} {f}: {r.get(f)}", r, None))
        if r.get("url"): calls.append(("reputation", 0.5, REPUTATION_CLAIM, r, None))
        calls.append(("is_program", 1.0, is_program_claim(r.get("name")), r, None))
random.Random(11).shuffle(calls)
sample=calls[:SAMPLE]
print(f"claims judged in this run: {len(calls)}; sampled {len(sample)}; repeats {REPEATS}")

client=LiteLLMClient(os.environ.get("BENCHKIT_JUDGE_MODEL") or "deepseek/deepseek-flash")
if os.environ.get("BENCHKIT_JUDGE_BASE_URL"):
    client.base_url=os.environ["BENCHKIT_JUDGE_BASE_URL"]
    client.api_key=os.environ.get(os.environ.get("BENCHKIT_JUDGE_API_KEY_ENV") or "", client.api_key)
print(f"judge: {client.model} via {getattr(client,'base_url','default')}")

def once(args):
    kind, weight, claim, record, question = args
    # The same prompt shape scoring sends, or the flip rate measured here would
    # describe a question nothing asks any more.
    payload=batch_prompt(record, {"claim": claim}, record.get("source_text"), record.get("url"), question)
    for _ in range(3):
        try:
            got=str(client.judge(VERDICT_SYSTEM, payload).get("verdict") or "").strip().casefold()
            return got or "unusable"
        except Exception:
            continue
    return "unusable"

def score_of(kind, weight, verdict):
    if kind=="is_program":
        return {"supported": 1.0, "contradicted": -1.0}.get(verdict, 0.0)
    if kind=="reputation":
        return {"supported": 1.0, "contradicted": -0.5}.get(verdict, 0.0) * weight
    return {"supported": 1.0, "contradicted": -1.0, "not_published": 0.5}.get(verdict, 0.0) * weight

jobs=[(c, i) for c in sample for i in range(REPEATS)]
with ThreadPoolExecutor(max_workers=16) as pool:
    verdicts=list(pool.map(lambda j: once(j[0]), jobs))

per_claim=[]
for n, c in enumerate(sample):
    vs=[verdicts[n*REPEATS+i] for i in range(REPEATS)]
    per_claim.append((c[0], c[1], vs, [score_of(c[0], c[1], v) for v in vs]))

unanimous=sum(1 for _,_,vs,_ in per_claim if len(set(vs))==1)
stdevs=[statistics.stdev(s) for _,_,_,s in per_claim if len(set(s))>1]
mean_sd=statistics.mean(stdevs) if stdevs else 0.0
print(f"unanimous across {REPEATS}: {unanimous}/{len(per_claim)} ({unanimous/len(per_claim)*100:.0f}%)")
print(f"per-claim score stdev where they differed: mean {mean_sd:.3f}, max {max(stdevs) if stdevs else 0:.3f}")
se_total=(sum(sd*sd for sd in stdevs)+ (len(per_claim)-len(stdevs))*0.0)**0.5
print(f"\nper-claim agreement across the whole run of {len(calls)} claims:")
print(f"  estimated SE of the claims total = sqrt(sum of per-claim variance) = +/- {se_total:.2f} points")
print(f"  (scaled to the full run: the sample is {len(sample)}/{len(calls)} of the claims)")
kind_flip={}
for k,w,vs,s in per_claim:
    kind_flip.setdefault(k, [0,0])
    kind_flip[k][1]+=1
    if len(set(vs))>1: kind_flip[k][0]+=1
for k,(flip,tot) in sorted(kind_flip.items()):
    print(f"    {k:12} flips {flip:3}/{tot:3}")
