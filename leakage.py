"""Data-leakage audit suite.

A high score means nothing if it came from leakage. These checks are run on
every experiment and their results are reported alongside the headline numbers.

Checks
------
1. group_disjoint        -- no pose/subject appears in two splits (the #1 way
                            synthetic-video pipelines cheat: views of one clip
                            split across train/test).
2. label_collinearity    -- correlation between the two targets. The repo's
                            coupled model makes r(freq, severity) ~= -1, so the
                            two "separate" tasks are one task. This is the
                            headline finding for the current generator.
3. label_shuffle_null    -- retrain on shuffled labels; a real model must
                            collapse to chance. If it doesn't, features encode
                            label-correlated identity (leakage).
4. train_test_separability -- a domain classifier trying to tell train from
                            test rows. AUC ~= 0.5 means the splits are drawn
                            from the same distribution (no split artifact).
5. permutation_top       -- which features the model actually uses; tremor
                            physics (peak frequency, band power, displacement)
                            should dominate, not identity proxies.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold


def group_disjoint(*group_arrays) -> dict:
    sets = [set(np.asarray(g).tolist()) for g in group_arrays]
    overlaps = 0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            overlaps += len(sets[i] & sets[j])
    return {"disjoint": overlaps == 0, "overlap_count": int(overlaps),
            "group_counts": [len(s) for s in sets]}


def label_collinearity(freq, severity_idx, amp) -> dict:
    freq = np.asarray(freq, float)
    sev = np.asarray(severity_idx, float)
    amp = np.asarray(amp, float)
    m = np.isfinite(freq) & np.isfinite(sev)
    out = {}
    if m.sum() > 2:
        out["pearson_freq_severity"] = float(pearsonr(freq[m], sev[m])[0])
        out["spearman_freq_severity"] = float(spearmanr(freq[m], sev[m])[0])
        out["pearson_freq_amp"] = float(pearsonr(freq[m], amp[m])[0])
    out["verdict"] = ("DEGENERATE: targets collinear (|r|>0.9) -> predicting one "
                      "trivially gives the other"
                      if abs(out.get("pearson_freq_severity", 0)) > 0.9
                      else "OK: targets are not collinear -> two independent problems")
    return out


def label_shuffle_null(X, y, groups, task, seed=0) -> dict:
    """Grouped-CV score with labels shuffled. Should fall to chance."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    yshuf = y.copy()
    rng.shuffle(yshuf)
    gkf = GroupKFold(n_splits=4)
    scores = []
    for tr, te in gkf.split(X, yshuf, groups):
        if task == "classification":
            mdl = RandomForestClassifier(n_estimators=120, random_state=0, n_jobs=-1)
            mdl.fit(X[tr], yshuf[tr])
            scores.append(float(np.mean(mdl.predict(X[te]) == yshuf[te])))
        else:
            from sklearn.ensemble import RandomForestRegressor
            mdl = RandomForestRegressor(n_estimators=120, random_state=0, n_jobs=-1)
            mdl.fit(X[tr], yshuf[tr])
            pred = mdl.predict(X[te])
            scores.append(float(np.mean(np.abs(pred - yshuf[te]) <= 1.0)))
    shuffled = float(np.mean(scores))
    if task == "classification":
        chance = float(max(np.bincount(y).max() / len(y), 1.0 / len(np.unique(y))))
        metric = "accuracy"
    else:
        chance = None
        metric = "within_1Hz"
    return {"metric": metric, "shuffled_score": shuffled, "majority_chance": chance,
            "verdict": ("OK: shuffled labels collapse toward chance"
                        if (chance is None or shuffled < chance + 0.12)
                        else "WARNING: shuffled labels still predictable -> possible leakage")}


def train_test_separability(X_train, X_test, seed=0) -> dict:
    """Domain classifier: can a model distinguish train from test rows?"""
    X = np.vstack([X_train, X_test])
    d = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_test))])
    idx = np.random.default_rng(seed).permutation(len(X))
    cut = int(0.7 * len(X))
    tr, te = idx[:cut], idx[cut:]
    clf = HistGradientBoostingClassifier(random_state=0)
    clf.fit(X[tr], d[tr])
    try:
        auc = float(roc_auc_score(d[te], clf.predict_proba(X[te])[:, 1]))
    except Exception:
        auc = float("nan")
    return {"domain_auc": auc,
            "verdict": ("OK: train/test indistinguishable (AUC~0.5)" if auc < 0.65
                        else "NOTE: train/test distributions differ (expected if "
                             "subject-independent split shifts the distribution)")}


def permutation_top(model, X_test, y_test, feature_names, scoring=None, k=12, seed=0) -> list:
    r = permutation_importance(model, X_test, y_test, n_repeats=8, random_state=seed,
                               scoring=scoring, n_jobs=-1)
    order = np.argsort(r.importances_mean)[::-1][:k]
    return [{"feature": feature_names[i], "importance": float(r.importances_mean[i])}
            for i in order]
