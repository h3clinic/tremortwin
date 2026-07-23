"""Landmark-level digital twin of the handharness Blender tremor pipeline.

Why this exists
---------------
The downstream CV model in this project never sees pixels. It sees 21 MediaPipe
hand landmarks per frame. Skin tone / lighting / render quality only affect
whether MediaPipe *detects* the hand (the detectability gate on the render
side). Once landmarks exist, the entire tremor signal -- frequency and
amplitude -- lives in the landmark *trajectory*.

This module reproduces that trajectory directly from the real rig
(`Do_Hand_DetailedRiggedAnimated_shared_16022026.glb`) using forward
kinematics, applying the *exact* tremor model from
`scripts/generate_tremor_dataset.py` / `scripts/blender_tremor_sequence.py`:
only the wrist bone (`radius_ulna`) oscillates; fingers are held static for a
clip. So the posed hand is a rigid body swinging about the wrist.

It lets us run the full "blind CV predicts injected tremor" experiment and the
data-leakage audit on this machine, faithfully, without Blender.

It does NOT model appearance, so it cannot answer the *detectability* question
(does MediaPipe detect a dark-skinned hand under oblique light?) -- that still
requires the real renders. We are explicit about that boundary everywhere.
"""
from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_ASSET = "input/hand_base/extracted/source/Do_Hand_DetailedRiggedAnimated_shared_16022026.glb"


def _find_glb() -> Path:
    """Locate the rigged-hand GLB.

    The asset is NOT redistributed with this repository (it belongs to the
    upstream handharness project). Resolution order:
      1. $HANDHARNESS_GLB           -- explicit path to the .glb
      2. $HANDHARNESS_DIR/<asset>   -- path to a handharness checkout
      3. ../handharness/<asset>     -- sibling checkout
      4. ./<asset>                  -- asset copied next to this file
    See docs/DATA.md for how to obtain it.
    """
    env = os.environ.get("HANDHARNESS_GLB")
    if env:
        return Path(env)
    d = os.environ.get("HANDHARNESS_DIR")
    if d:
        return Path(d) / _ASSET
    for cand in (HERE.parent / "handharness" / _ASSET, HERE / _ASSET):
        if cand.exists():
            return cand
    return HERE.parent / "handharness" / _ASSET   # reported in the error message


DEFAULT_GLB = _find_glb()

# 21 rig joints, ordered roughly like MediaPipe (wrist first, then fingers
# base->tip). The rig has no separate knuckle for thumb CMC vs others, but the
# anatomy is real; this ordering is only used for stable indexing.
LANDMARK_ORDER = [
    "radius_ulna",
    "thumb_trapez", "thumb_meta", "thumb_prox", "thumb_dist",
    "index_meta", "index_prox", "index_midd", "index_dist",
    "midd_meta", "midd_prox", "midd_midd", "midd_dist",
    "ring_meta", "ring_prox", "ring_midd", "ring_dist",
    "pinky_meta", "pinky_prox", "pinky_midd", "pinky_dist",
]
FINGER_CHAINS = {
    "thumb": ["thumb_trapez", "thumb_meta", "thumb_prox", "thumb_dist"],
    "index": ["index_meta", "index_prox", "index_midd", "index_dist"],
    "middle": ["midd_meta", "midd_prox", "midd_midd", "midd_dist"],
    "ring": ["ring_meta", "ring_prox", "ring_midd", "ring_dist"],
    "pinky": ["pinky_meta", "pinky_prox", "pinky_midd", "pinky_dist"],
}
WRIST = "radius_ulna"


def _quat_to_mat(q) -> np.ndarray:
    """glTF quaternion [x, y, z, w] -> 3x3 rotation matrix."""
    if q is None:
        return np.eye(3)
    x, y, z, w = q
    n = (x * x + y * y + z * z + w * w) ** 0.5
    if n == 0:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _euler_to_mat(rx, ry, rz) -> np.ndarray:
    """Euler XYZ -> 3x3. Small-angle regime (tremor < 0.15 rad), so the precise
    intrinsic/extrinsic convention is immaterial; we use Rz @ Ry @ Rx."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _mat4(R=None, t=None) -> np.ndarray:
    M = np.eye(4)
    if R is not None:
        M[:3, :3] = R
    if t is not None:
        M[:3, 3] = t
    return M


@dataclass
class RigNode:
    name: str
    translation: np.ndarray
    rest_R: np.ndarray
    scale: np.ndarray
    children: list
    parent: int = -1


class HandRig:
    """Forward-kinematics skeleton parsed straight from the GLB."""

    def __init__(self, glb_path: Path = DEFAULT_GLB):
        nodes_json = self._read_gltf_nodes(glb_path)
        self.nodes = []
        for i, n in enumerate(nodes_json):
            t = n.get("translation") or [0.0, 0.0, 0.0]
            s = n.get("scale") or [1.0, 1.0, 1.0]
            self.nodes.append(RigNode(
                name=n.get("name", f"node_{i}"),
                translation=np.array(t, dtype=float),
                rest_R=_quat_to_mat(n.get("rotation")),
                scale=np.array(s, dtype=float),
                children=list(n.get("children", [])),
            ))
        for i, n in enumerate(self.nodes):
            for c in n.children:
                self.nodes[c].parent = i
        self.name2idx = {n.name: i for i, n in enumerate(self.nodes)}
        self.roots = [i for i, n in enumerate(self.nodes) if n.parent == -1]
        # Topological order (parents before children) for FK.
        self._order = self._topo_order()

    @staticmethod
    def _read_gltf_nodes(glb_path: Path):
        p = Path(glb_path)
        if not p.exists():
            raise FileNotFoundError(
                f"Rigged-hand GLB not found at: {p}\n"
                "This asset is not redistributed with this repository. Point at your\n"
                "handharness checkout, e.g.\n"
                "  export HANDHARNESS_DIR=/path/to/handharness\n"
                "  export HANDHARNESS_GLB=/path/to/Do_Hand_...glb\n"
                "See docs/DATA.md.")
        data = p.read_bytes()
        assert data[:4] == b"glTF", "not a GLB file"
        json_len = struct.unpack("<I", data[12:16])[0]
        gltf = json.loads(data[20:20 + json_len].decode("utf-8"))
        return gltf.get("nodes", [])

    def _topo_order(self):
        order, stack = [], list(self.roots)
        while stack:
            i = stack.pop()
            order.append(i)
            stack.extend(self.nodes[i].children)
        return order

    def fk(self, extra_rot: dict | None = None) -> dict:
        """Compute world-space origins for every node.

        extra_rot maps node name -> (rx, ry, rz) euler applied in the bone's
        local frame *after* its rest rotation -- exactly Blender pose-bone
        `rotation_euler` semantics. Returns name -> world xyz (np.array(3)).
        """
        extra_rot = extra_rot or {}
        world = [None] * len(self.nodes)
        for i in self._order:
            n = self.nodes[i]
            R = n.rest_R
            if n.name in extra_rot:
                R = R @ _euler_to_mat(*extra_rot[n.name])
            # glTF local transform = T * R * S.
            local = _mat4(R @ np.diag(n.scale), n.translation)
            world[i] = local if n.parent == -1 else world[n.parent] @ local
        return {n.name: world[i][:3, 3].copy() for i, n in enumerate(self.nodes)}

    def landmarks(self, extra_rot: dict | None = None) -> np.ndarray:
        """Return (21, 3) array in LANDMARK_ORDER."""
        w = self.fk(extra_rot)
        return np.array([w[name] for name in LANDMARK_ORDER])


# ---------------------------------------------------------------------------
# Tremor model -- copied verbatim in spirit from the repo generators so the
# twin's labels ARE the project's labels.
# ---------------------------------------------------------------------------

def tremor_params_coupled(intensity: float):
    """EXACT mapping from scripts/blender_tremor_sequence.py tremor_params().

    NOTE: frequency and amplitude are both deterministic functions of a single
    `intensity` scalar -> they are perfectly collinear. The leakage audit flags
    this. Kept here to reproduce the project's current behaviour faithfully.
    """
    i = max(0.0, min(100.0, intensity))
    amplitude_rad = (i / 100.0) * 0.15
    amplitude_cm = i * 0.12
    frequency_hz = 10.0 - (i / 100.0) * 6.0
    return amplitude_rad, frequency_hz, amplitude_cm


def clinical_bin(intensity: float):
    """EXACT 6-band clinical mapping from the repo (UPDRS / TETRAS references)."""
    i = max(0.0, min(100.0, intensity))
    if i <= 0:
        return "Absent", 0, 0.0
    if i <= 15:
        return "Slight", 1, 1.0
    if i <= 35:
        return "Mild", 2, 2.0
    if i <= 65:
        return "Moderate", 3, 3.0
    if i <= 85:
        return "Marked", 4, 4.0
    return "Severe", 4, 4.5


CLASS_NAMES = ["Absent", "Slight", "Mild", "Moderate", "Marked", "Severe"]
CLASS_IDX = {c: k for k, c in enumerate(CLASS_NAMES)}


def wrist_tremor_euler(amp_rad, freq_hz, t, rng: np.random.Generator):
    """Per-frame wrist Euler rotation (rad). Matches compute_tremor_rotation():
    primary + 2nd-harmonic on flexion axis, phase-shifted sinusoids on the
    other two axes, plus inter-cycle micro-irregularity noise."""
    p = 2.0 * np.pi * freq_hz * t
    primary = np.sin(p)
    secondary = 0.18 * np.sin(2.0 * p + 0.5)
    vx = amp_rad * (primary + secondary) + rng.uniform(-0.05, 0.05) * amp_rad
    vy = amp_rad * np.sin(p + 1.2) + rng.uniform(-0.05, 0.05) * amp_rad
    vz = amp_rad * np.sin(p + 2.4)
    return vx, vy, vz


def random_static_pose(rig: HandRig, rng: np.random.Generator) -> dict:
    """Sample a static finger configuration (held fixed across a clip), mirroring
    apply_random_finger_pose(): per-joint flexion on local X, spread on the
    metacarpal. Returns extra_rot dict for the finger joints only."""
    extra = {}
    for finger, chain in FINGER_CHAINS.items():
        is_thumb = finger == "thumb"
        spread = rng.uniform(-0.30, 0.30) if is_thumb else rng.uniform(-0.18, 0.18)
        if is_thumb:
            flex_limits = [(0.05, 0.55), (0.05, 0.80), (0.05, 0.70)]
        else:
            flex_limits = [(0.00, 0.60), (0.00, 1.00), (0.00, 0.90)]
        for idx, bone in enumerate(chain[:3]):
            lo, hi = flex_limits[min(idx, len(flex_limits) - 1)]
            rx = rng.uniform(lo, hi)
            rz = spread if idx == 0 else 0.0
            extra[bone] = (rx, 0.0, rz)
    return extra


# ---------------------------------------------------------------------------
# Camera projection + MediaPipe-style observation noise
# ---------------------------------------------------------------------------

VIEWS = {
    "equatorial_front": (0.0, 0.0),
    "equatorial_left": (90.0, 0.0),
    "equatorial_right": (270.0, 0.0),
    "dorsal_fingertip": (0.0, 55.0),
}


def _view_matrix(azimuth_deg, elevation_deg):
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    caz, saz = np.cos(az), np.sin(az)
    cel, sel = np.cos(el), np.sin(el)
    Rz = np.array([[caz, -saz, 0], [saz, caz, 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, cel, -sel], [0, sel, cel]])
    return Rx @ Rz


def project(landmarks3d: np.ndarray, azimuth, elevation, wrist_idx=0) -> np.ndarray:
    """Weak-perspective projection of (21,3) -> (21,2) normalized image coords.

    Hand is centered on the wrist and scaled by its own size, mimicking the
    scale-normalized normalized-image-coordinate output MediaPipe produces."""
    R = _view_matrix(azimuth, elevation)
    pts = landmarks3d @ R.T
    pts = pts - pts[wrist_idx]
    scale = np.linalg.norm(pts, axis=1).max() + 1e-9
    img = pts[:, :2] / (2.5 * scale)        # keep within ~[-0.4, 0.4]
    img = img + np.array([0.5, 0.5])         # MediaPipe-like [0,1] frame
    return img


@dataclass
class ObsNoise:
    """MediaPipe observation-noise model (in normalized image units)."""
    jitter_std: float = 0.004        # per-landmark gaussian tracking jitter
    global_jitter_std: float = 0.002 # whole-hand wobble (subject not perfectly still)
    dropout_prob: float = 0.01       # fraction of frames where a landmark is lost
    quantize: float = 0.0            # optional coordinate quantization step


def add_observation_noise(traj: np.ndarray, rng: np.random.Generator, noise: ObsNoise) -> np.ndarray:
    """traj: (T, 21, 2) -> noised copy. Dropouts are forward-filled (as a real
    tracker's last-good-value smoothing would do)."""
    T, L, _ = traj.shape
    out = traj + rng.normal(0, noise.jitter_std, traj.shape)
    out = out + rng.normal(0, noise.global_jitter_std, (T, 1, 2))
    if noise.dropout_prob > 0:
        mask = rng.random((T, L)) < noise.dropout_prob
        for li in range(L):
            last = out[0, li]
            for ti in range(T):
                if mask[ti, li]:
                    out[ti, li] = last
                else:
                    last = out[ti, li]
    if noise.quantize > 0:
        out = np.round(out / noise.quantize) * noise.quantize
    return out


def render_clip(rig: HandRig, pose_extra: dict, amp_rad: float, freq_hz: float,
                n_frames: int, fps: int, view: str, rng: np.random.Generator,
                noise: ObsNoise, phase_offset: float = 0.0) -> np.ndarray:
    """Produce one (T, 21, 2) landmark trajectory for a pose+tremor+view.

    This is the faithful stand-in for: Blender render -> MediaPipe landmarks.
    """
    az, el = VIEWS[view]
    frames = []
    for f in range(n_frames):
        t = phase_offset + f / fps
        vx, vy, vz = wrist_tremor_euler(amp_rad, freq_hz, t, rng)
        extra = dict(pose_extra)
        extra[WRIST] = (vx, vy, vz)
        lm3d = rig.landmarks(extra)
        frames.append(project(lm3d, az, el))
    traj = np.stack(frames, axis=0)
    return add_observation_noise(traj, rng, noise)


# ---------------------------------------------------------------------------
# Self-test: FK must reproduce the GLB rest geometry.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rig = HandRig()
    rest = rig.fk()
    # Reference world origins from trimesh scene graph (probe_glb.json).
    ref = {
        "radius_ulna": [0.0, -0.07584, 0.6177],
        "index_dist": [-0.01768, 18.91601, -6.35006],
        "pinky_dist": [-0.58063, 15.75665, 3.26776],
        "thumb_dist": [0.19015, 9.44487, -10.78763],
        "midd_dist": [-0.43319, 20.1474, -2.76678],
    }
    print("FK rest-pose validation (max abs error vs trimesh graph):")
    max_err = 0.0
    for name, r in ref.items():
        err = float(np.abs(rest[name] - np.array(r)).max())
        max_err = max(max_err, err)
        print(f"  {name:12s} fk={np.round(rest[name],4)}  ref={r}  err={err:.5f}")
    print(f"MAX ERROR = {max_err:.6f}  -> {'PASS' if max_err < 0.05 else 'FAIL'}")

    # Quick tremor sanity: index fingertip displacement amplitude vs intensity.
    print("\nFingertip displacement vs intensity (front view, 2s @ 60fps):")
    for intensity in (10, 30, 60, 90):
        amp_rad, freq_hz, amp_cm = tremor_params_coupled(intensity)
        g = np.random.default_rng(0)
        pose = random_static_pose(rig, g)
        traj = render_clip(rig, pose, amp_rad, freq_hz, 120, 60, "equatorial_front",
                           g, ObsNoise(jitter_std=0.0, global_jitter_std=0.0, dropout_prob=0.0))
        tip = traj[:, LANDMARK_ORDER.index("index_dist"), :]
        disp = np.linalg.norm(tip - tip.mean(0), axis=1).max()
        print(f"  intensity={intensity:3d}  f={freq_hz:.2f}Hz  A={amp_rad:.4f}rad  "
              f"tip_disp(norm)={disp:.4f}")
