"""The self-improvement ML loop.

Mirrors the user's request: a model that "knows nothing" iteratively improves
until it hits a target on held-out, leakage-controlled data, or saturates.

Each round escalates: it widens the feature set (temporal -> +spectral ->
+cross-landmark agreement) and the model family/capacity, evaluates a batch of
candidate configs by *subject-grouped* cross-validation, and promotes the best.
The loop stops when the grouped-CV score reaches the target or improvement
stalls. The winning config is then judged once on a subject-held-out TEST set
that was never seen during the search.

Leakage controls baked in:
* every split is grouped by subject_id (no subject in two folds / in both
  train and test);
* the test subjects are isolated before the search begins;
* `label_shuffle_null` (in leakage.py) is run on the winning features.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
                              GradientBoostingClassifier,
                              HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def clf_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "adjacent_acc": float(np.mean(np.abs(y_true - y_pred) <= 1)),  # ordinal +/-1 bin
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def reg_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = np.abs(y_true - y_pred)
    ss = np.sum((y_true - y_true.mean()) ** 2) + 1e-12
    return {
        "mae_hz": float(err.mean()),
        "within_1hz": float(np.mean(err <= 1.0)),   # Pintea et al. 2018 criterion
        "within_0p5hz": float(np.mean(err <= 0.5)),
        "r2": float(1 - np.sum((y_true - y_pred) ** 2) / ss),
    }


# ---------------------------------------------------------------------------
# Model zoo
# ---------------------------------------------------------------------------
def make_model(task, name, hp, seed=0):
    if task == "classification":
        zoo = {
            "logreg": lambda: make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, C=hp.get("C", 1.0))),
            "rf": lambda: RandomForestClassifier(n_estimators=hp.get("n", 300),
                        max_depth=hp.get("depth"), min_samples_leaf=hp.get("leaf", 1),
                        random_state=seed, n_jobs=-1),
            "extra": lambda: ExtraTreesClassifier(n_estimators=hp.get("n", 400),
                        max_depth=hp.get("depth"), min_samples_leaf=hp.get("leaf", 1),
                        random_state=seed, n_jobs=-1),
            "histgb": lambda: HistGradientBoostingClassifier(
                        max_iter=hp.get("n", 300), learning_rate=hp.get("lr", 0.08),
                        max_depth=hp.get("depth"), l2_regularization=hp.get("l2", 0.0),
                        random_state=seed),
            "gb": lambda: GradientBoostingClassifier(n_estimators=hp.get("n", 200),
                        learning_rate=hp.get("lr", 0.08), max_depth=hp.get("depth", 3),
                        random_state=seed),
        }
    else:
        zoo = {
            "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=hp.get("alpha", 1.0))),
            "rf": lambda: RandomForestRegressor(n_estimators=hp.get("n", 300),
                        max_depth=hp.get("depth"), min_samples_leaf=hp.get("leaf", 1),
                        random_state=seed, n_jobs=-1),
            "extra": lambda: ExtraTreesRegressor(n_estimators=hp.get("n", 400),
                        max_depth=hp.get("depth"), min_samples_leaf=hp.get("leaf", 1),
                        random_state=seed, n_jobs=-1),
            "histgb": lambda: HistGradientBoostingRegressor(
                        max_iter=hp.get("n", 300), learning_rate=hp.get("lr", 0.08),
                        max_depth=hp.get("depth"), l2_regularization=hp.get("l2", 0.0),
                        random_state=seed),
        }
    return zoo[name]()


def _grouped_cv_score(task, model_spec, X, y, groups, primary, seed=0, n_splits=4):
    name, hp = model_spec
    if task == "classification":
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(X, y, groups)
    else:
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(X, y, groups)
    vals = []
    for tr, te in split_iter:
        mdl = make_model(task, name, hp, seed)
        mdl.fit(X[tr], y[tr])
        pred = mdl.predict(X[te])
        m = clf_metrics(y[te], pred) if task == "classification" else reg_metrics(y[te], pred)
        vals.append(m[primary])
    return float(np.mean(vals))


def _round_configs(round_idx, task, best_family):
    """Feature-set escalation + candidate model configs for this round."""
    if round_idx == 0:
        fset = ["temporal"]
        cands = ([("logreg", {"C": 1.0}), ("rf", {"n": 200})] if task == "classification"
                 else [("ridge", {"alpha": 1.0}), ("rf", {"n": 200})])
    elif round_idx == 1:
        fset = ["temporal", "spectral"]
        cands = ([("rf", {"n": 300}), ("extra", {"n": 400}), ("histgb", {"n": 300, "lr": 0.08})]
                 if task == "classification"
                 else [("rf", {"n": 300}), ("extra", {"n": 400}), ("histgb", {"n": 300, "lr": 0.08})])
    elif round_idx == 2:
        fset = ["temporal", "spectral", "agreement"]
        cands = [("histgb", {"n": 400, "lr": 0.06, "l2": 0.0}),
                 ("extra", {"n": 600, "leaf": 1}),
                 ("rf", {"n": 500, "leaf": 1})]
    else:
        # Hyperparameter refinement around the best family found so far.
        fset = ["temporal", "spectral", "agreement"]
        fam = best_family or "histgb"
        grid = {
            "histgb": [{"n": 500, "lr": 0.05, "l2": l2, "depth": d}
                       for l2 in (0.0, 1.0) for d in (None, 6)],
            "extra": [{"n": n, "leaf": leaf} for n in (600, 900) for leaf in (1, 2)],
            "rf": [{"n": n, "leaf": leaf} for n in (500, 800) for leaf in (1, 2)],
            "gb": [{"n": 300, "lr": lr, "depth": d} for lr in (0.05, 0.1) for d in (3, 4)],
            "logreg": [{"C": c} for c in (0.3, 1.0, 3.0, 10.0)],
            "ridge": [{"alpha": a} for a in (0.1, 1.0, 10.0)],
        }.get(fam, [{"n": 500}])
        cands = [(fam, hp) for hp in grid]
    return fset, cands


def select_features(feature_names, groups_active, feature_groups_map):
    cols = []
    for g in groups_active:
        cols += feature_groups_map.get(g, [])
    keep = [i for i, n in enumerate(feature_names) if n in set(cols)]
    return keep


def run_self_improvement(Xfull, y, groups, feature_names, feature_groups_map, task,
                         primary, target, max_rounds=6, patience=2, seed=0, log=print):
    """Iterative search. Returns history + winning model refit on all train data.

    Xfull, y, groups are the TRAIN partition only (test is held out by caller).
    """
    history = []
    best = {"score": -np.inf, "config": None, "fset": None}
    stale = 0
    best_family = None
    for r in range(max_rounds):
        fset, cands = _round_configs(r, task, best_family)
        keep = select_features(feature_names, fset, feature_groups_map)
        X = Xfull[:, keep]
        round_best = None
        for spec in cands:
            score = _grouped_cv_score(task, spec, X, y, groups, primary, seed)
            history.append({"round": r, "features": fset, "model": spec[0],
                            "hparams": spec[1], "cv_score": round(score, 4)})
            log(f"  [round {r}] {spec[0]:7s} {str(spec[1]):38s} "
                f"feat={'+'.join(fset):28s} cv_{primary}={score:.4f}")
            if round_best is None or score > round_best[0]:
                round_best = (score, spec, fset, keep)
        if round_best[0] > best["score"] + 1e-4:
            best.update(score=round_best[0], config=round_best[1], fset=round_best[2],
                        keep=round_best[3])
            best_family = round_best[1][0]
            stale = 0
            log(f"  -> new best: {best_family} cv_{primary}={best['score']:.4f}")
        else:
            stale += 1
        if best["score"] >= target:
            log(f"  TARGET REACHED: cv_{primary}={best['score']:.4f} >= {target}")
            break
        if stale >= patience:
            log(f"  saturated (no improvement for {patience} rounds); stopping")
            break

    # Refit winner on all train data.
    final = make_model(task, best["config"][0], best["config"][1], seed)
    final.fit(Xfull[:, best["keep"]], y)
    return {"history": history, "best": {k: best[k] for k in ("score", "config", "fset")},
            "model": final, "keep": best["keep"]}
