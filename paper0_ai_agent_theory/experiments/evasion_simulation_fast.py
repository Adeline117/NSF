#!/usr/bin/env python3
"""Fast adversarial evasion simulation for P0.
Simplified: skip MI recomputation (slow), focus on Cohen's d and classifier."""
import json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
import math, warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
FEAT = ROOT / "paper1_onchain_agent_id" / "data" / "features_with_taxonomy.parquet"
AI_JSON = ROOT / "paper3_ai_sybil" / "experiments" / "real_ai_features.json"
OUT = Path(__file__).with_name("evasion_simulation_results.json")

ORIG_23 = ["tx_interval_mean","tx_interval_std","tx_interval_skewness",
    "active_hour_entropy","night_activity_ratio","weekend_ratio","burst_frequency",
    "gas_price_round_number_ratio","gas_price_trailing_zeros_mean","gas_limit_precision",
    "gas_price_cv","eip1559_priority_fee_precision","gas_price_nonce_correlation",
    "unique_contracts_ratio","top_contract_concentration","method_id_diversity",
    "contract_to_eoa_ratio","sequential_pattern_score","unlimited_approve_ratio",
    "approve_revoke_ratio","unverified_contract_approve_ratio",
    "multi_protocol_interaction_count","flash_loan_usage"]
AI_8 = ["gas_price_precision","hour_entropy","behavioral_consistency",
    "action_sequence_perplexity","error_recovery_pattern","response_latency_variance",
    "gas_nonce_gap_regularity","eip1559_tip_precision"]
ALL_31 = ORIG_23 + AI_8
TIMING = ["tx_interval_mean","tx_interval_std","tx_interval_skewness",
    "burst_frequency","response_latency_variance"]
LLM, DEFI, SEED = 3, 2, 42
SIGMAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

def cohens_d(g1, g2):
    g1, g2 = np.asarray(g1,float), np.asarray(g2,float)
    g1, g2 = g1[~np.isnan(g1)], g2[~np.isnan(g2)]
    if len(g1)<2 or len(g2)<2: return float('nan')
    ps = np.sqrt(((len(g1)-1)*g1.std(ddof=1)**2+(len(g2)-1)*g2.std(ddof=1)**2)/(len(g1)+len(g2)-2))
    return float((g1.mean()-g2.mean())/ps) if ps>0 else float('nan')

def joint_mi_knn(X, y, k=3):
    Xs = StandardScaler().fit_transform(X)
    nn = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(Xs)
    _, idx = nn.kneighbors(Xs)
    idx = idx[:,1:]
    nl = y[idx]
    nc = len(np.unique(y))
    ce = np.zeros(len(y))
    for c in range(nc):
        p = (nl==c).sum(axis=1)/k
        m = p>0
        ce[m] -= p[m]*np.log2(p[m])
    h_cond = float(ce.mean())
    _, cts = np.unique(y, return_counts=True)
    p = cts/cts.sum()
    h_y = float(-np.sum(p*np.log2(p+1e-30)))
    return max(0,h_y-h_cond), h_y, h_cond

def fano_ceil(h_y, mi, K):
    hc = max(0, h_y-mi)
    lo, hi = 0.0, (K-1)/K
    def rhs(pe):
        hb = -(pe*math.log2(pe)+(1-pe)*math.log2(1-pe)) if 0<pe<1 else 0
        return hb + pe*math.log2(K-1)
    if rhs(hi)<hc: return 1-hi
    for _ in range(200):
        mid = (lo+hi)/2
        if rhs(mid)>=hc: hi=mid
        else: lo=mid
    return 1-hi

def main():
    print("Loading data...")
    df = pd.read_parquet(FEAT)
    df = df[df["label"]==1].copy()
    with open(AI_JSON) as f: ai = json.load(f)
    rows = []
    for a, feats in ai["per_address"].items():
        r = {"address": a}
        for c in AI_8: r[c] = feats.get(c, np.nan)
        rows.append(r)
    df_ai = pd.DataFrame(rows).set_index("address")
    df = df.join(df_ai[AI_8], how="left")
    y = df["taxonomy_index"].values.astype(int)
    X = df[ALL_31].values.astype(float)
    # Impute + clip
    for j in range(X.shape[1]):
        col = X[:,j]
        med = np.nanmedian(col)
        col[np.isnan(col)] = med if not np.isnan(med) else 0
        lo,hi = np.percentile(col,[1,99])
        X[:,j] = np.clip(col,lo,hi)
    X = np.nan_to_num(X)

    timing_idx = [ALL_31.index(f) for f in TIMING]
    rt_idx = ALL_31.index("tx_interval_mean")
    llm_mask, defi_mask = y==LLM, y==DEFI
    defi_means = X[defi_mask][:,timing_idx].mean(axis=0)
    defi_stds = X[defi_mask][:,timing_idx].std(axis=0)
    rng = np.random.RandomState(SEED)

    print(f"N={len(y)}, LLM={llm_mask.sum()}, DeFi={defi_mask.sum()}")
    print(f"\n{'sigma':<8}{'d':<10}{'acc':<10}{'LLM_F1':<10}{'MI':<10}{'ceiling':<10}")
    print("-"*58)

    results = []
    evasion_cost = None
    for sigma in SIGMAS:
        Xe = X.copy()
        if sigma > 0:
            for k, fi in enumerate(timing_idx):
                blend = min(sigma, 1.0)
                noise = rng.normal(0, defi_stds[k]*sigma, size=llm_mask.sum())
                Xe[llm_mask,fi] = (1-blend)*X[llm_mask,fi] + blend*defi_means[k] + noise*0.5

        d = cohens_d(Xe[llm_mask,rt_idx], Xe[defi_mask,rt_idx])

        # GBM CV (5-fold for speed)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        mdl = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
            min_samples_leaf=5, subsample=0.8, random_state=SEED)
        yt, yp = [], []
        for tr, te in skf.split(Xe, y):
            sc = StandardScaler()
            clf = clone(mdl)
            clf.fit(sc.fit_transform(Xe[tr]), y[tr])
            p = clf.predict(sc.transform(Xe[te]))
            yt.extend(y[te].tolist()); yp.extend(p.tolist())
        yt, yp = np.array(yt), np.array(yp)
        acc = accuracy_score(yt, yp)
        tp = ((yt==LLM)&(yp==LLM)).sum()
        fp = ((yt!=LLM)&(yp==LLM)).sum()
        fn = ((yt==LLM)&(yp!=LLM)).sum()
        pr = tp/(tp+fp) if tp+fp>0 else 0
        rc = tp/(tp+fn) if tp+fn>0 else 0
        f1 = 2*pr*rc/(pr+rc) if pr+rc>0 else 0

        # MI + Fano (only at sigma=0 and 2.0 for speed, interpolate rest)
        if sigma in [0.0, 1.0, 2.0]:
            mi, hy, hc = joint_mi_knn(Xe, y, k=3)
            ceil = fano_ceil(hy, mi, 8)
        else:
            mi, ceil = float('nan'), float('nan')

        print(f"{sigma:<8.2f}{d:<10.3f}{acc:<10.3f}{f1:<10.3f}{mi if mi==mi else '---':<10}{ceil if ceil==ceil else '---':<10}")

        if evasion_cost is None and abs(d) < 0.2 and sigma > 0:
            evasion_cost = sigma

        results.append({"sigma": sigma, "cohens_d": round(d,4), "accuracy": round(acc,4),
            "llm_f1": round(f1,4), "mi_bits": round(mi,4) if mi==mi else None,
            "fano_ceiling": round(ceil,4) if ceil==ceil else None})

    out = {"sweep": results, "evasion_cost_sigma": evasion_cost,
        "n_total": int(len(y)), "n_llm": int(llm_mask.sum()), "n_defi": int(defi_mask.sum()),
        "timing_features_targeted": TIMING}
    with open(OUT, "w") as f: json.dump(out, f, indent=2)
    print(f"\nEvasion cost (d<0.2): sigma={evasion_cost}")
    print(f"Results → {OUT}")

if __name__=="__main__": main()
