import json,glob,statistics as st,hashlib
D={}
for f in sorted(glob.glob("locomo_20260930_200456_C*_s200.rescored_q34b.json")):
    arm=f.split("200456_")[1].split("_s200")[0]; d=json.load(open(f)); r=d["results"]
    D[arm]=r
    real=sum(1 for x in r if (x.get("predicted") or "").strip())
    think=sum(1 for x in r if "<think" in (x.get("predicted") or ""))
    ov=d["overall"]
    print(f"VALIDITY {arm}: rows={len(r)} real_predictions={real} think_tags={think}")
    for k in ("judge","judge_tolerant","judge_rejudged"):
        print(f"  {k} mean={sum(x.get(k) or 0 for x in r)/len(r):.4f}")
    print(f"  overall_field={json.dumps(ov)[:200]}")
    print(f"  gen_ms mean={st.mean(x['gen_ms'] for x in r):.1f} median={st.median(x['gen_ms'] for x in r):.1f} retrieval_ms mean={st.mean(x['retrieval_ms'] for x in r):.1f}")
def key(x): return (x["conversation_id"],x["question"])
def ok(x): return (x.get("judge_rejudged") or 0)>=0.5
def paired(a,b):
    A={key(x):ok(x) for x in D[a]}; B={key(x):ok(x) for x in D[b]}
    assert A.keys()==B.keys()
    co=sum(1 for k in A if A[k] and not B[k]); ro=sum(1 for k in A if B[k] and not A[k])
    print(f"PAIRED {a} vs {b}: n={len(A)} cand_only={co} ref_only={ro} net={co-ro}")
arms=sorted(D)
for c,b,t in [("C0","C1","C2"),("C3","C4","C5"),("C6","C7","C8")]:
    n=lambda p:[a for a in arms if a.startswith(p)][0]
    paired(n(c),n(b)); paired(n(t),n(b)); paired(n(t),n(c))
paired([a for a in arms if a.startswith("C4")][0],[a for a in arms if a.startswith("C1")][0])
for f in sorted(glob.glob("locomo_20260930_200456_C*.json")):
    print(hashlib.sha256(open(f,"rb").read()).hexdigest()+"  "+f)
