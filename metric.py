# 시험 지표 계산기: configs/<TC>.json의 산정식·판정 기준대로 원시 결과(results_on_*/<TC>/)를 읽어
# 결과 폴더 안 metrics/ 에 표(csv), 판정(criteria.json), 그림을 쓴다. 원시 결과는 바꾸지 않는다.
# Usage: python metric.py --exp TC1 [--tag MI300X]              (TC1, TC2: results_on_<tag>/<TC>/)
#        python metric.py --exp TC3 [--backend fake_nighthawk]   (TC3: results_on_<backend>/TC3/ 가 있는 backend 전부, 또는 하나)
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "classical"))
from common import device_tag, save_json  # noqa: E402

argv = sys.argv
exp = argv[argv.index("--exp") + 1]
cfg = json.loads((REPO / "configs" / f"{exp}.json").read_text())
tag = argv[argv.index("--tag") + 1] if "--tag" in argv else device_tag()
only_backend = argv[argv.index("--backend") + 1] if "--backend" in argv else None
crit = cfg["criteria"]


def judge(values):
    # 판정 기준 키는 "<값 이름>_<gt|ge|le>" 꼴 (configs/<TC>.json의 criteria). 값이 없으면(None) 판정도 None
    checks = {}
    for key, thr in crit.items():
        base, op = key.rsplit("_", 1)
        v = values[base]
        checks[key] = {"value": v, "op": {"gt": ">", "ge": ">=", "le": "<="}[op], "threshold": thr,
                       "pass": None if v is None else bool(v > thr if op == "gt" else v >= thr if op == "ge" else v <= thr)}
    return checks


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def show(checks):
    for key, c in checks.items():
        v = "없음" if c["value"] is None else (f"{c['value']:.4f}" if isinstance(c["value"], float) else str(c["value"]))
        print(f"  {'PASS' if c['pass'] else ('----' if c['pass'] is None else 'FAIL'):4s}  {key:36s} 값 {v:>10s}  기준: 값 {c['op']} {c['threshold']}")


# ============================================================ TC1, TC2 (고전 시뮬레이션)
if cfg["kind"] == "classical":
    root = REPO / f"results_on_{tag}" / cfg["name"]
    T, T_max, SHOTS = cfg["exact_steps"], cfg["standard_long_max_steps"], cfg["shots_per_circuit"]
    out = root / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    print(f"===== {cfg['name']} 지표: {root.relative_to(REPO)} → metrics/", flush=True)

    # ---- 쌍(인스턴스 × 초기값)마다 문서의 측정 항목
    rows, status, recs_all = [], {}, {}
    for inst in cfg["instances"]:
        name = f"{cfg['grid']}_inst{inst}_seed{cfg['seed_base'] + inst}"
        recs = {}
        for run in ("standard", "pado_safe", "standard_long"):
            p = root / name / run / "params.json"
            recs[run] = json.loads(p.read_text()) if p.exists() else None
            status[f"{name}/{run}"] = recs[run]["status"] if recs[run] else "missing"
        recs_all[name] = recs
        if recs["standard"] is None or recs["pado_safe"] is None:
            print(f"  {name}: standard 또는 pado_safe 결과 없음 → 건너뜀", flush=True)
            continue
        long_by_c = {r["c"]: r for r in recs["standard_long"]["inits"]} if recs["standard_long"] else {}
        for rs, rp in zip(recs["standard"]["inits"], recs["pado_safe"]["inits"]):
            ar_s, ar_p = np.array(rs["ar"]), np.array(rp["ar"])
            N, n_act = rs["n_active"], rp["n_active"]
            A_star, A_0 = float(ar_p[-1]), float(ar_p[0])            # 목표 품질 = PADO 최종 AR, 시작점 품질 = PADO 시작 AR
            tau_p = int(np.flatnonzero(ar_p >= A_star)[0])          # PADO가 A*에 처음 닿는 step
            rl = long_by_c.get(rs["c"])
            tau_s = rl["steps_to_target"] if rl else None            # standard가 A*에 처음 닿는 step (None = 미도달 또는 연장 실행 없음)
            hit0 = np.flatnonzero(np.array(rl["ar"]) >= A_0) if rl else np.array([])
            C_p = tau_p * (2 * n_act + 1) + 1                        # 회로 수: step × (2N+1) + 최종 에너지 확인 1
            C_s = tau_s * (2 * N + 1) + 1 if tau_s is not None else None
            k = min(len(rs["energy"]), len(rl["energy"])) if rl else 0
            rows.append({
                "problem": name, "c": rs["c"], "N": N, "N_act": n_act, "D": n_act / N,
                "AR_std_0": float(ar_s[0]), f"AR_std_{T}": float(ar_s[-1]), "AR_pado_0": A_0, f"AR_pado_{T}": A_star,
                "AR_pado_lwpp_before_distill": (rp["exact_energy_lwpp_final"] - recs["pado_safe"]["e_max"]) / (recs["pado_safe"]["e_min"] - recs["pado_safe"]["e_max"]),
                f"delta_AR{T}": A_star - float(ar_s[-1]), "pado_start_ge_std_last": bool(A_0 >= ar_s[-1]),
                "tau_pado_Astar": tau_p, "tau_std_Astar": tau_s, "reached": tau_s is not None,
                "tau_std_A0": int(hit0[0]) if hit0.size else None,
                "C_pado": C_p, "C_std": C_s, "R": (1 - C_p / C_s) if C_s else None,
                f"shots_pado_{SHOTS}": C_p * SHOTS, f"shots_std_{SHOTS}": C_s * SHOTS if C_s else None,
                "max_dE_long_vs_standard": float(np.abs(np.array(rs["energy"][:k]) - np.array(rl["energy"][:k])).max()) if rl else None,
                "lwpp_compile_s": recs["pado_safe"]["lwpp"]["compile_s"], "lwpp_train_s": recs["pado_safe"]["lwpp"]["train_s"],
                "lwpp_peak_gb": max(recs["pado_safe"]["lwpp"]["compile_peak_gb"], recs["pado_safe"]["lwpp"]["train_peak_gb"]),
                "status_standard": status[f"{name}/standard"], "status_pado_safe": status[f"{name}/pado_safe"],
                "status_standard_long": status[f"{name}/standard_long"],
            })
    if not rows:
        raise SystemExit("계산할 쌍이 없음 (results 없음). 먼저 run.py를 실행")
    write_csv(out / "pairs.csv", rows)

    # ---- 판정 기준 값
    d_ar = [r[f"delta_AR{T}"] for r in rows]
    reached = [r for r in rows if r["reached"]]
    tau0 = [r["tau_std_A0"] for r in rows if r["tau_std_Astar"] is not None or r["tau_std_A0"] is not None]
    longs = [r["standard_long"] for r in recs_all.values() if r["standard_long"]]
    if longs:
        T_max = longs[0]["config"]["max_steps"]        # 실제 연장 실행에 쓰인 상한 (config와 다를 수 있다)
    values = {
        "n_pairs": len(rows), "T": T, "T_max": T_max,
        "delta_ar20_min": float(min(d_ar)), "delta_ar20_median": float(np.median(d_ar)),
        "n_pairs_pado_start_ge_std_last": int(sum(r["pado_start_ge_std_last"] for r in rows)),
        "D_max": float(max(r["D"] for r in rows)),
        "n_long_pairs": int(sum(r["status_standard_long"] != "missing" for r in rows)),
        "n_reached": len(reached),
        "R_median": float(np.median([r["R"] for r in reached])) if reached else None,
        "R_min": float(min(r["R"] for r in reached)) if reached else None,
        "tau_std_A0_min": min(t for t in tau0 if t is not None) if any(t is not None for t in tau0) else None,
        "n_tau_std_A0_unreached": int(sum(t is None for t in tau0)),   # 상한 안에 PADO 시작 AR에도 못 닿은 쌍 (step 수를 T_max 이상으로 본다)
    }
    checks = judge(values)
    all_pass = all(c["pass"] for c in checks.values())
    save_json(out / "criteria.json", {"experiment": cfg["name"], "title": cfg["title"], "results_root": str(root.relative_to(REPO)),
                                      "values": values, "checks": checks, "all_pass": all_pass, "status": status})

    print(f"쌍 {len(rows)}개, 연장 실행 있는 쌍 {values['n_long_pairs']}개, 도달 {values['n_reached']}개")
    print("판정 (configs/%s.json의 criteria):" % exp)
    show(checks)
    print(f"  => 전체 {'PASS' if all_pass else 'FAIL'}", flush=True)

    # ---- 그림: exact step 대비 에너지 (두 방식, 전수 조사 E_min 기준선), 연장 실행이 있으면 그 그림도
    probs = [p for p in recs_all if recs_all[p]["standard"] and recs_all[p]["pado_safe"]]
    for long_fig in (False, True):
        ps = [p for p in probs if recs_all[p]["standard_long"]] if long_fig else probs
        if not ps:
            continue
        fig, axes = plt.subplots(1, len(ps), figsize=(4.2 * len(ps), 3.6), squeeze=False)
        for ax, p in zip(axes[0], ps):
            S = recs_all[p]["standard_long" if long_fig else "standard"]
            P = recs_all[p]["pado_safe"]
            for i, (rs, rp) in enumerate(zip(S["inits"], P["inits"])):
                ax.plot(rs["energy"], color="black", alpha=0.6, linewidth=1, label=("standard (until PADO final AR)" if long_fig else "standard") if i == 0 else None)
                ax.plot(rp["energy"], color="tab:orange", alpha=0.8, linewidth=1, label="PADO (pado_safe)" if i == 0 else None)
            ax.axhline(S["e_min"], color="tab:blue", linestyle="--", linewidth=1, label="brute-force E_min")
            ax.set_title(p, fontsize=9)
            ax.set_xlabel("exact step")
        axes[0][0].set_ylabel("exact energy")
        axes[0][0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / ("energy_curves_long.png" if long_fig else "energy_curves.png"), dpi=150)
    print(f"saved {out.relative_to(REPO)}: pairs.csv, criteria.json, energy_curves*.png")

# ============================================================ TC3 (양자컴퓨터 실행)
else:
    case = f"{cfg['grid']}_inst{cfg['instance']}_seed{cfg['seed_base'] + cfg['instance']}"
    case_dirs = sorted(p.parents[1] for p in REPO.glob(f"results_on_*/{cfg['name']}/{case}/standard/params.json"))
    ref_dir = REPO / f"results_on_{tag}" / "TC2" / case          # 고전 기준선 (TC2의 같은 쌍)
    if not case_dirs:
        raise SystemExit(f"results_on_*/{cfg['name']}/{case}/standard/params.json 없음. 먼저 run.py --exp {exp} 실행")
    for case_dir in case_dirs:
        backend = case_dir.parents[1].name.replace("results_on_", "")
        if only_backend and backend != only_backend:
            continue
        out = case_dir.parents[1] / cfg["name"] / "metrics"
        out.mkdir(parents=True, exist_ok=True)
        print(f"\n===== {cfg['name']} 지표: backend {backend} ({case_dir.relative_to(REPO)}) → metrics/", flush=True)
        recs = {run: json.loads((case_dir / run / "params.json").read_text())
                for run in ("pado_safe", "standard") if (case_dir / run / "params.json").exists()}
        if len(recs) < 2:
            print("  pado_safe 또는 standard 결과 없음 → 건너뜀")
            continue
        shots = recs["standard"]["config"]["shots"]

        # ---- step별 표와 방식별 합계
        steps_rows, totals = [], {}
        for run, rec in recs.items():
            pts = rec["steps"] + ([rec["final"]] if rec["final"] is not None else [])
            for p in pts:
                steps_rows.append({"backend": backend, "run": run, "t": p["t"], "energy": p["energy"], "energy_std": p["energy_std"],
                                   "AR": p["ar"], "energy_noiseless": p["energy_noiseless"], "AR_noiseless": p["ar_noiseless"],
                                   "G": abs(p["ar"] - p["ar_noiseless"]), "n_circuits": p["n_circuits"], "shots": p["n_circuits"] * shots,
                                   "job_id": p["job_id"], "usage_s": p.get("usage_s"), "wall_s": p["wall_s"]})
            usage = [p.get("usage_s") for p in pts]
            totals[run] = {"status": rec["status"] if "status" in rec else ("ok" if rec["final"] is not None else "incomplete"),
                           "updates": len(rec["steps"]), "planned_updates": rec["config"]["steps"], "complete": rec["final"] is not None,
                           "n_params": rec["config"]["n_params"], "circuits_per_step": rec["config"]["circuits_per_step"],
                           "circuits": int(sum(p["n_circuits"] for p in pts)), "shots": int(sum(p["n_circuits"] for p in pts) * shots),
                           "AR_first": pts[0]["ar"], "AR_last": pts[-1]["ar"],
                           "AR_noiseless_first": pts[0]["ar_noiseless"], "AR_noiseless_last": pts[-1]["ar_noiseless"],
                           "G_max": max(abs(p["ar"] - p["ar_noiseless"]) for p in pts),
                           "usage_s_total": float(sum(u for u in usage if u is not None)) if any(u is not None for u in usage) else None,
                           "physical_qubits": rec["config"].get("physical_qubits")}
        write_csv(out / "steps.csv", steps_rows)
        pre_path = case_dir / "pado_safe" / "pre_distill.json"
        pre = json.loads(pre_path.read_text()) if pre_path.exists() else None
        P, S = totals["pado_safe"], totals["standard"]
        values = {"backend": backend, "shots_per_circuit": shots,
                  "AR_pado_0": P["AR_first"], "AR_pado_last": P["AR_last"], "AR_std_0": S["AR_first"], "AR_std_last": S["AR_last"],
                  "AR_pado_lwpp_before_distill": pre["ar"] if pre else None,
                  "C_pado": P["circuits"], "C_std": S["circuits"], "R_C": P["circuits"] / S["circuits"],
                  "shots_pado": P["shots"], "shots_std": S["shots"],
                  "AR_pado_last_minus_AR_std_last": P["AR_last"] - S["AR_last"],
                  "AR_pado_0_minus_AR_std_last": P["AR_first"] - S["AR_last"],
                  "G_max_pado": P["G_max"], "G_max_std": S["G_max"],
                  "usage_s_total": (P["usage_s_total"] or 0) + (S["usage_s_total"] or 0) if (P["usage_s_total"] or S["usage_s_total"]) else None}
        checks = judge(values)
        complete = P["complete"] and S["complete"]
        all_pass = all(c["pass"] for c in checks.values()) if complete else None   # 끝나지 않은 실행은 판정 보류
        backend_info = json.loads((case_dir / "backend.json").read_text()) if (case_dir / "backend.json").exists() else None
        save_json(out / "criteria.json", {"experiment": cfg["name"], "title": cfg["title"], "backend": backend, "case": case,
                                          "results_dir": str(case_dir.relative_to(REPO)), "complete": complete,
                                          "values": values, "totals": totals, "pre_distill": pre and {k: pre[k] for k in ("ar", "ar_noiseless", "energy", "job_id")},
                                          "checks": checks, "all_pass": all_pass, "backend_info": backend_info})
        for run in ("pado_safe", "standard"):
            t = totals[run]
            print(f"  {run:10s} step {t['updates']}/{t['planned_updates']}{'' if t['complete'] else ' (미완료)'}  회로 {t['circuits']}  shots {t['shots']}"
                  f"  AR {t['AR_first']:.4f} → {t['AR_last']:.4f}  (noiseless {t['AR_noiseless_first']:.4f} → {t['AR_noiseless_last']:.4f})"
                  f"  usage {t['usage_s_total']}")
        print(f"  R_C = {values['R_C']:.4f}" + (f", 증류 전 AR {pre['ar']:.4f}" if pre else ""))
        print("판정 (configs/%s.json의 criteria):" % exp)
        show(checks)
        print(f"  => 전체 {'PASS' if all_pass else ('보류(미완료)' if all_pass is None else 'FAIL')}", flush=True)

        # ---- 그림 1: 단계별 AR 막대 (진한 막대 = 기기 측정, 연한 막대 = 같은 θ의 노이즈 없는 값)
        bars = [("standard\nstart", S["AR_first"], S["AR_noiseless_first"], "black"),
                (f"standard\nafter {S['updates']} steps", S["AR_last"], S["AR_noiseless_last"], "black"),
                ("PADO\nstart (= theta_init)", S["AR_first"], S["AR_noiseless_first"], "tab:orange")]
        if pre:
            bars.append(("PADO\nafter pretrain\n(before distill)", pre["ar"], pre["ar_noiseless"], "tab:orange"))
        bars += [("PADO\nafter distill", P["AR_first"], P["AR_noiseless_first"], "tab:orange"),
                 (f"PADO\nafter {P['updates']} steps", P["AR_last"], P["AR_noiseless_last"], "tab:orange")]
        x = np.arange(len(bars))
        fig, ax = plt.subplots(figsize=(9, 4.4))
        for i, (name, a_meas, a_nl, color) in enumerate(bars):
            ax.bar(x[i] - 0.19, a_meas, 0.38, color=color, label="measured" if i == 0 else None)
            ax.bar(x[i] + 0.19, a_nl, 0.38, color=color, alpha=0.3, label="same theta, noiseless" if i == 0 else None)
            ax.text(x[i] - 0.19, a_meas + 0.01, f"{a_meas:.3f}", ha="center", va="bottom", fontsize=8, bbox=dict(facecolor="white", edgecolor="none", pad=1))
            ax.text(x[i] + 0.19, a_nl + 0.01, f"{a_nl:.3f}", ha="center", va="bottom", fontsize=8, alpha=0.7, bbox=dict(facecolor="white", edgecolor="none", pad=1))
        ax.axhline(1.0, color="tab:blue", linestyle="--", linewidth=1, label="brute-force optimum (AR = 1)")
        ax.set_xticks(x)
        ax.set_xticklabels([b[0] for b in bars], fontsize=8)
        ax.set_ylim(0, 1.22)
        ax.set_ylabel("approximation ratio")
        ax.set_title(f"{case}, {backend}, {shots} shots per circuit", fontsize=9)
        ax.legend(fontsize=7, loc="upper left", ncol=3)
        fig.tight_layout()
        fig.savefig(out / "summary_ar.png", dpi=150)

        # ---- 그림 2: step 대비 에너지 (측정 ●, 노이즈 없는 ×, 고전 기준선 TC2 파선)
        fig, ax = plt.subplots(figsize=(7, 4))
        for run, color, label in (("standard", "black", "standard"), ("pado_safe", "tab:orange", "PADO")):
            pts = recs[run]["steps"] + ([recs[run]["final"]] if recs[run]["final"] is not None else [])
            ax.plot([p["energy"] for p in pts], "o-", color=color, label=f"{label} ({backend}, measured)")
            ax.plot([p["energy_noiseless"] for p in pts], "x:", color=color, alpha=0.6, label=f"{label} (same theta, noiseless)")
        if (ref_dir / "pado_safe" / "params.json").exists() and (ref_dir / "standard_long" / "params.json").exists():
            ref_p = [r for r in json.loads((ref_dir / "pado_safe" / "params.json").read_text())["inits"] if r["c"] == cfg["const"]][0]
            ref_s = [r for r in json.loads((ref_dir / "standard_long" / "params.json").read_text())["inits"] if r["c"] == cfg["const"]][0]
            ax.plot(ref_p["energy"], "--", color="tab:orange", alpha=0.4, label="PADO (TC2 classical)")
            ax.plot(ref_s["energy"][:len(ref_p["energy"])], "--", color="black", alpha=0.4, label="standard (TC2 classical)")
        ax.axhline(recs["standard"]["e_min"], color="tab:blue", linestyle="--", linewidth=1, label="brute-force E_min")
        ax.set_xlabel("exact step (parameter-shift update)")
        ax.set_ylabel("energy")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "energy_curves.png", dpi=150)
        print(f"saved {out.relative_to(REPO)}: steps.csv, criteria.json, summary_ar.png, energy_curves.png")
