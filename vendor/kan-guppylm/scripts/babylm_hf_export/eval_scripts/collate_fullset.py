"""Full-set BLiMP + supplement: per-arch mean/CI + Welch vs MLP (n=10)."""
import re, math
from pathlib import Path
from scipy import stats as ss
RESULTS = Path("/Users/felippealves/Documents/GitHub/babylm-eval/results")
ARCHS = ["mlp","swiglu","chebyshev","grkan"]
LABEL={"mlp":"MLP-4x-GELU","swiglu":"SwiGLU","chebyshev":"Chebyshev d3 g8","grkan":"GR-KAN canonical"}
VAL={"mlp":3.8199,"swiglu":3.7700,"chebyshev":3.7809,"grkan":3.7997}
def read(a,s,ds):
    p=RESULTS/f"{a}_s{s}"/"main"/"zero_shot"/"causal"/"blimp"/ds/"best_temperature_report.txt"
    if not p.exists(): return None
    m=re.search(r"### AVERAGE ACCURACY\s*\n([0-9.]+)",p.read_text()); return float(m.group(1)) if m else None
def coll(ds): return {a:[v for s in range(42,52) if (v:=read(a,s,ds)) is not None] for a in ARCHS}
def st(xs):
    n=len(xs);m=sum(xs)/n;sd=(sum((x-m)**2 for x in xs)/(n-1))**.5;ci=ss.t.ppf(.975,n-1)*sd/math.sqrt(n);return m,sd,ci,n
def welch(a,b):
    ma,va,na=sum(a)/len(a),sum((x-sum(a)/len(a))**2 for x in a)/(len(a)-1),len(a)
    mb,vb,nb=sum(b)/len(b),sum((x-sum(b)/len(b))**2 for x in b)/(len(b)-1),len(b)
    se=math.sqrt(va/na+vb/nb);t=(ma-mb)/se
    df=(va/na+vb/nb)**2/((va/na)**2/(na-1)+(vb/nb)**2/(nb-1));return ma-mb,t,2*ss.t.sf(abs(t),df)
for ds,title in [("blimp_filtered","BLiMP FULL (n=10, 59,875 pairs)"),("supplement_filtered","BLiMP-SUPPLEMENT FULL (n=10, 5,218)")]:
    d=coll(ds); print("="*64); print(title); print("="*64)
    print(f"{'Architecture':<18}{'val CE':>8}{'mean':>8}{'sd':>6}{'95% CI':>15}")
    for a in sorted(ARCHS,key=lambda a:VAL[a]):
        m,sd,ci,n=st(d[a]); print(f"{LABEL[a]:<18}{VAL[a]:>8.4f}{m:>8.2f}{sd:>6.2f}   [{m-ci:.2f},{m+ci:.2f}]")
    print("  vs MLP:")
    for a in ["swiglu","chebyshev","grkan"]:
        dd,t,p=welch(d[a],d["mlp"]); v="n.s." if p>=.05 else ("BETTER" if dd>0 else "WORSE")
        print(f"    {LABEL[a]:<18} Δ={dd:+.2f}  p={p:.3f} -> {v}")
    print()
