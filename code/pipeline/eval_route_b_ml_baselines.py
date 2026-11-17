"""Route-B pure-ML baselines under the current strict evaluation protocol.

Baselines:
  - MLP regression: scalar demand -> 68-generator active dispatch.
  - XGBoost regression: one scalar-demand regressor per generator, if xgboost is installed.
  - GAT-only: included from route_b_pipeline_eval.json or evaluated separately there.

All predicted active-power dispatches are projected to generator bounds and
power balance before strict AC-PF + LODF N-1 verification.
"""

import argparse
import json
import pathlib
import sys
import time
from collections import Counter, defaultdict

import numpy as np

CODE_DIR = pathlib.Path(__file__).resolve().parent
PROJ_DIR = CODE_DIR.parents[1]
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(1, str(PROJ_DIR))

from eval_route_b_pipeline import dispatch_cost, is_pass, set_default_voltage, verify_current
from train_rlv35_complete import SCEDEnvV21, project_box_sum, split_samples_stratified


def resolve_path(value):
    path = pathlib.Path(value)
    return path if path.is_absolute() else PROJ_DIR / path


def load_split(samples, split_path):
    if split_path.exists():
        split = json.load(open(split_path, "r", encoding="utf-8"))
        return split["train_ids"], split.get("val_ids", []), split["test_ids"], str(split_path)
    train_ids, val_ids, test_ids = split_samples_stratified(samples)
    return train_ids, val_ids, test_ids, "generated_by_split_samples_stratified"


def make_xy(samples, ids):
    x = np.array([[float(samples[i]["demand_mw"])] for i in ids], dtype=np.float64)
    y = []
    for i in ids:
        demand = float(samples[i]["demand_mw"])
        dispatch = np.asarray(samples[i]["dispatch_68"], dtype=np.float64)
        y.append(dispatch / (demand + 1e-9))
    return x, np.asarray(y, dtype=np.float64)


def standardize_train_test(x_train, x_test):
    mu = x_train.mean(axis=0, keepdims=True)
    sd = x_train.std(axis=0, keepdims=True)
    sd[sd < 1e-9] = 1.0
    return (x_train - mu) / sd, (x_test - mu) / sd, {"mean": mu.ravel().tolist(), "std": sd.ravel().tolist()}


def train_mlp(x_train, y_train):
    from sklearn.neural_network import MLPRegressor

    model = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        alpha=1e-6,
        max_iter=3000,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=80,
        random_state=42,
    )
    t0 = time.perf_counter()
    model.fit(x_train, y_train)
    return model, time.perf_counter() - t0, {"n_iter": int(getattr(model, "n_iter_", -1))}


def train_xgboost(x_train, y_train):
    try:
        from xgboost import XGBRegressor
    except Exception as exc:
        return None, 0.0, {"skipped": f"xgboost_not_available: {exc}"}

    models = []
    t0 = time.perf_counter()
    for j in range(y_train.shape[1]):
        model = XGBRegressor(
            n_estimators=180,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=1.0,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1,
        )
        model.fit(x_train, y_train[:, j])
        models.append(model)
    return models, time.perf_counter() - t0, {"n_models": len(models)}


def predict_model(model, x, demand_mw):
    if isinstance(model, list):
        y = np.array([m.predict(x)[0] for m in model], dtype=np.float64)
    else:
        y = np.asarray(model.predict(x)[0], dtype=np.float64)
    return y * float(demand_mw)


def eval_static_prediction(env, sample, sid, pred):
    env.reset(sample_idx=sid)
    set_default_voltage(env)
    p = project_box_sum(
        np.asarray(pred, dtype=np.float64),
        env.PMIN,
        env.PMAX * 0.99,
        float(sample["demand_mw"]),
    )
    env.current_power = p.astype(np.float64).copy()
    info, runtime = verify_current(env)
    return {
        "passed": is_pass(info),
        "info": info,
        "runtime_sec": runtime,
        "dispatch": env.current_power.copy(),
        "cost": dispatch_cost(sample, env.current_power),
    }


def summarize(name, rows, train_time):
    n = len(rows)
    passed = sum(r["passed"] for r in rows)
    reasons = Counter(r["reason"] for r in rows if not r["passed"])
    by_demand = defaultdict(lambda: {"passed": 0, "n": 0})
    for r in rows:
        key = f"{r['demand_gw']:g}"
        by_demand[key]["n"] += 1
        by_demand[key]["passed"] += int(r["passed"])
    return {
        "name": name,
        "n": n,
        "passed": passed,
        "pass_rate_pct": round(100.0 * passed / max(n, 1), 3),
        "train_time_sec": round(float(train_time), 3),
        "avg_runtime_sec": round(float(np.mean([r["runtime_sec"] for r in rows])), 4),
        "avg_cost": round(float(np.mean([r["cost"] for r in rows])), 3),
        "avg_v_violations": round(float(np.mean([r["v_violations"] for r in rows])), 3),
        "avg_v_gap": round(float(np.mean([r["v_gap"] for r in rows])), 6),
        "reasons": dict(reasons),
        "by_demand": dict(sorted(by_demand.items(), key=lambda kv: float(kv[0]))),
    }


def evaluate_method(name, model, env, samples, test_ids, x_test_std):
    rows = []
    for local_i, sid in enumerate(test_ids):
        sample = samples[sid]
        pred = predict_model(model, x_test_std[local_i : local_i + 1], sample["demand_mw"])
        out = eval_static_prediction(env, sample, sid, pred)
        info = out["info"]
        rows.append({
            "sid": sid,
            "sample_id": sample.get("sample_id"),
            "demand_gw": float(sample.get("demand_gw", sample["demand_mw"] / 1000.0)),
            "passed": bool(out["passed"]),
            "reason": info.get("reason"),
            "converged": bool(info.get("converged", False)),
            "v_violations": int(info.get("v_violations", 999)),
            "v_gap": float(info.get("v_violation_max_pu", 0.0) or 0.0),
            "n1_violations": int(info.get("n1_violations", 999)),
            "runtime_sec": float(out["runtime_sec"]),
            "cost": float(out["cost"]),
        })
        print(f"  {name} {local_i+1:02d}/{len(test_ids)} {sample.get('sample_id')} {'OK' if out['passed'] else 'FAIL'} {info.get('reason')}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="case300_gt_v10_dataset_lodf_corrected_277.json")
    ap.add_argument("--split-file", default="rl_scratch_sac_lodf277_pure_stable_v2/split_ids.json")
    ap.add_argument("--output", default="paper_results/route_b_ml_baselines_lodf277.json")
    args = ap.parse_args()

    dataset_path = resolve_path(args.dataset)
    split_path = resolve_path(args.split_file)
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = [s for s in data["samples"] if s.get("is_complete_sced")]
    if split_path.exists():
        split_meta = json.load(open(split_path, "r", encoding="utf-8"))
        recorded_dataset = split_meta.get("dataset")
        if recorded_dataset and pathlib.Path(recorded_dataset).resolve() != dataset_path.resolve():
            raise ValueError(f"Split dataset mismatch: split={recorded_dataset}, requested={dataset_path}")
    train_ids, val_ids, test_ids, split_source = load_split(samples, split_path)

    # Keep validation out of training so the supervised baselines use the same
    # training partition as the RL policy.
    x_train, y_train = make_xy(samples, train_ids)
    x_test, _ = make_xy(samples, test_ids)
    x_train_std, x_test_std, scaler = standardize_train_test(x_train, x_test)

    env = SCEDEnvV21(samples, test_ids, dataset_meta=data)
    results = {
        "dataset": str(dataset_path.resolve()),
        "split_source": split_source,
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
        "input": "standardized scalar demand_mw",
        "target": "dispatch_68 / demand_mw",
        "projection": "project_box_sum to PMIN, 0.99*PMAX, and demand balance",
        "scaler": scaler,
        "methods": {},
    }

    print("Training MLP baseline...")
    mlp, mlp_time, mlp_meta = train_mlp(x_train_std, y_train)
    print(f"MLP trained in {mlp_time:.1f}s {mlp_meta}")
    mlp_rows = evaluate_method("MLP", mlp, env, samples, test_ids, x_test_std)
    results["methods"]["MLP"] = {
        "meta": mlp_meta,
        "summary": summarize("MLP", mlp_rows, mlp_time),
        "rows": mlp_rows,
    }

    print("\nTraining XGBoost baseline...")
    xgb, xgb_time, xgb_meta = train_xgboost(x_train_std, y_train)
    if xgb is None:
        print(f"XGBoost skipped: {xgb_meta['skipped']}")
        results["methods"]["XGBoost"] = {"meta": xgb_meta}
    else:
        print(f"XGBoost trained in {xgb_time:.1f}s {xgb_meta}")
        xgb_rows = evaluate_method("XGBoost", xgb, env, samples, test_ids, x_test_std)
        results["methods"]["XGBoost"] = {
            "meta": xgb_meta,
            "summary": summarize("XGBoost", xgb_rows, xgb_time),
            "rows": xgb_rows,
        }

    out = resolve_path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    printable = {k: v.get("summary", v.get("meta", {})) for k, v in results["methods"].items()}
    print(json.dumps(printable, indent=2))
    print("Wrote:", out)


if __name__ == "__main__":
    main()
