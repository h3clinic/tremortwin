"""Dataset generators built on the validated landmark twin.

Two modes:

* ``coupled``  -- reproduces the repo's current label model EXACTLY:
  frequency, amplitude and severity are all deterministic functions of one
  ``intensity`` scalar. Faithful, but frequency and severity are perfectly
  collinear (the leakage audit flags r ~= -1). Predicting either is the same
  problem; high accuracy here is degenerate, not evidence of a useful tool.

* ``decoupled`` -- the honest version. Tremor *type* sets the frequency band
  (Parkinsonian rest 4-6 Hz, essential/postural 5-9 Hz, enhanced-physiologic
  8-12 Hz; grounded in Lenka & Jankovic 2021 / Zhang 2017). Amplitude is
  sampled *independently*, and severity is amplitude-driven (as clinical TETRAS
  amplitude rating is). Now frequency and severity are statistically
  independent -> two genuinely separate prediction problems.

We also simulate multiple "subjects" (per-subject hand-shape variation + noise
level) so the evaluation can be subject-independent -- the only split that does
not leak identity.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from twin import (HandRig, ObsNoise, VIEWS, clinical_bin, random_static_pose,
                  render_clip, tremor_params_coupled, CLASS_IDX)

# Amplitude (rad) -> severity band edges. Aligned to the coupled model's bin
# boundaries (intensity 15/35/65/85 -> amp 0.0225/0.0525/0.0975/0.1275) so the
# two modes share one severity scale.
AMP_EDGES = [0.0, 0.0225, 0.0525, 0.0975, 0.1275, 0.15]
SEVERITY_BY_AMP = ["Slight", "Mild", "Moderate", "Marked", "Severe"]

# Decoupled tremor types -> frequency band (Hz). 'none' = no tremor.
TREMOR_TYPES = {
    "none": None,
    "parkinson_rest": (4.0, 6.0),
    "essential_postural": (5.0, 9.0),
    "enhanced_physiologic": (8.0, 12.0),
}
TYPE_IDX = {t: k for k, t in enumerate(TREMOR_TYPES)}


def severity_from_amp(amp_rad: float) -> str:
    if amp_rad <= 1e-6:
        return "Absent"
    for k in range(1, len(AMP_EDGES)):
        if amp_rad <= AMP_EDGES[k] + 1e-9:
            return SEVERITY_BY_AMP[k - 1]
    return "Severe"


@dataclass
class Subject:
    sid: int
    shape_scale: np.ndarray      # anisotropic hand-shape factor (3,)
    noise: ObsNoise


@dataclass
class Clip:
    pose_id: int
    subject_id: int
    view: str
    traj: np.ndarray             # (T, 21, 2)
    freq_hz: float
    amp_rad: float
    intensity: float
    severity: str
    severity_idx: int
    tremor_type: str
    has_tremor: bool


def make_subjects(n: int, rng: np.random.Generator) -> list[Subject]:
    subs = []
    for sid in range(n):
        shape = np.exp(rng.normal(0, 0.07, size=3)) * rng.uniform(0.88, 1.12)
        # Per-subject tracking quality varies (camera, distance, lighting proxy).
        noise = ObsNoise(
            jitter_std=float(rng.uniform(0.003, 0.007)),
            global_jitter_std=float(rng.uniform(0.001, 0.004)),
            dropout_prob=float(rng.uniform(0.0, 0.02)),
        )
        subs.append(Subject(sid, shape, noise))
    return subs


def _apply_subject_shape(rig: HandRig, shape_scale: np.ndarray):
    """Return a per-subject FK closure: landmarks scaled anisotropically about
    the wrist in the hand frame (consistent across that subject's clips)."""
    def lm(extra):
        pts = rig.landmarks(extra)
        wrist = pts[0].copy()
        return (pts - wrist) * shape_scale + wrist
    return lm


def _render_clip_subject(rig, lm_fn, pose_extra, amp_rad, freq_hz, n_frames, fps,
                         view, rng, noise, phase_offset):
    """Like twin.render_clip but uses a subject-shaped landmark function."""
    from twin import WRIST, project, add_observation_noise, wrist_tremor_euler
    az, el = VIEWS[view]
    frames = []
    for f in range(n_frames):
        t = phase_offset + f / fps
        vx, vy, vz = wrist_tremor_euler(amp_rad, freq_hz, t, rng)
        extra = dict(pose_extra)
        extra[WRIST] = (vx, vy, vz)
        frames.append(project(lm_fn(extra), az, el))
    return add_observation_noise(np.stack(frames, 0), rng, noise)


def generate(mode: str, n_poses: int, n_frames: int, fps: int, n_subjects: int,
             seed: int, views=None, p_no_tremor: float = 0.12):
    """Yield Clip objects. One pose -> one tremor state, rendered from each view.

    Grouping: a pose_id's clips share frequency/amplitude/severity and must stay
    together in any split; a subject_id's poses share hand shape + noise.
    """
    rng = np.random.default_rng(seed)
    rig = HandRig()
    views = views or list(VIEWS.keys())
    subjects = make_subjects(n_subjects, rng)

    for pose_id in range(n_poses):
        subj = subjects[pose_id % n_subjects]
        lm_fn = _apply_subject_shape(rig, subj.shape_scale)
        pose_extra = random_static_pose(rig, rng)
        phase_offset = float(rng.uniform(0.0, 10.0))

        if mode == "coupled":
            intensity = float(rng.integers(1, 101))
            amp_rad, freq_hz, _ = tremor_params_coupled(intensity)
            sev, _, _ = clinical_bin(intensity)
            ttype = "coupled"
            has_tremor = True
        elif mode == "decoupled":
            if rng.random() < p_no_tremor:
                ttype, freq_hz, amp_rad = "none", 0.0, 0.0
                sev = "Absent"
                has_tremor = False
                intensity = 0.0
            else:
                ttype = rng.choice([t for t in TREMOR_TYPES if t != "none"])
                lo, hi = TREMOR_TYPES[ttype]
                freq_hz = float(rng.uniform(lo, hi))
                # Amplitude independent of frequency. Physiologic capped low.
                if ttype == "enhanced_physiologic":
                    amp_rad = float(np.clip(rng.uniform(0.003, 0.045), 0, 0.15))
                else:
                    amp_rad = float(np.clip(np.exp(rng.uniform(np.log(0.006), np.log(0.15))), 0, 0.15))
                sev = severity_from_amp(amp_rad)
                has_tremor = True
                intensity = float(np.interp(amp_rad, [0, 0.15], [0, 100]))
        else:
            raise ValueError(mode)

        sev_idx = CLASS_IDX[sev]
        for view in views:
            traj = _render_clip_subject(rig, lm_fn, pose_extra, amp_rad, freq_hz,
                                        n_frames, fps, view, rng, subj.noise, phase_offset)
            yield Clip(pose_id=pose_id, subject_id=subj.sid, view=view, traj=traj,
                       freq_hz=freq_hz, amp_rad=amp_rad, intensity=intensity,
                       severity=sev, severity_idx=sev_idx, tremor_type=ttype,
                       has_tremor=has_tremor)
