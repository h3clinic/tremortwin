# Render-side directions: fix black triangles → skin tones → 3000 poses

This is the **appearance/detectability** half of the pipeline (Blender → MP4 →
MediaPipe). It is separate from the ML experiment in this folder, which operates
on landmarks and therefore does not depend on rendering. But the renders must be
*MediaPipe-detectable* before any rendered clip becomes a training sample.

## 1. Why you get "black triangles / no 3D"

Two concrete causes in the current repo scripts (not guesses — read from the code):

**(a) EEVEE engine string is invalid on Blender ≥ 4.2.**
`scripts/blender_tremor_sequence.py:327` and `scripts/generate_tremor_dataset.py:110`
both do:
```python
scene.render.engine = "BLENDER_EEVEE"
```
Blender 4.2 renamed this enum to `BLENDER_EEVEE_NEXT`. On a recent Blender the
assignment throws / leaves the scene in a broken render state. Fix by version:
```python
def set_eevee(scene):
    engines = scene.render.bl_rna.properties['engine'].enum_items.keys()
    scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
```

**(b) The 3000-pose generator assigns no material.**
`scripts/blender_tremor_sequence.py` builds a skin `Principled BSDF`
(`setup_scene_for_render`, lines 311-324) — good. But
`scripts/generate_tremor_dataset.py` (the `--num-positions 3000` script)
**never assigns a material**. If the GLB's PBR materials don't survive glTF
import into EEVEE, every face renders unlit/near-black against the 0.10 world →
"black triangles." Add the same skin-material block to `ensure_render_scene()`.

Also worth a one-line guard: after import, recompute normals so flipped faces
don't render dark.
```python
for obj in [o for o in bpy.context.scene.objects if o.type == 'MESH']:
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False); bpy.ops.object.mode_set(mode='OBJECT')
```

## 2. Randomized skin tones (drop-in)

Add to both render scripts and call once per pose with a sampled tone:
```python
SKIN_TONES = {  # Fitzpatrick-spanning; linear-ish sRGB base + subsurface tint
 "I_very_light":(0.93,0.80,0.72),  "II_light":(0.87,0.72,0.60),
 "III_light_med":(0.78,0.62,0.48), "IV_medium":(0.68,0.50,0.36),
 "V_olive":(0.62,0.48,0.32),       "VI_tan":(0.50,0.35,0.24),
 "VII_med_dark":(0.40,0.28,0.20),  "VIII_dark":(0.30,0.20,0.14),
 "IX_deep":(0.20,0.13,0.09),       "X_deeper":(0.14,0.09,0.06),
 "XI_deepest":(0.10,0.07,0.05),    "XII_cool_deep":(0.12,0.08,0.07),
}

def apply_skin_material(rng, tone_name):
    base = SKIN_TONES[tone_name]
    jit = [max(0.02, min(0.98, c + rng.uniform(-0.03, 0.03))) for c in base]
    rough = rng.uniform(0.35, 0.55)
    for obj in [o for o in bpy.context.scene.objects if o.type == 'MESH']:
        mat = bpy.data.materials.new(f"Skin_{obj.name}")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        bsdf.inputs["Base Color"].default_value = (*jit, 1.0)
        bsdf.inputs["Roughness"].default_value = rough
        # Blender 4.x: subsurface is "Subsurface Weight"; guard for both.
        for k in ("Subsurface Weight", "Subsurface"):
            if k in bsdf.inputs: bsdf.inputs[k].default_value = 0.15; break
        obj.data.materials.clear(); obj.data.materials.append(mat)
```
Record the tone name in `labels`/manifest so the detectability report can break
out per-tone pass rates. **Expect darker tones + oblique cameras to have lower
MediaPipe detection** — that is the real finding to measure, and is why each
rendered clip must pass detectability before it is kept.

## 3. 3000 unique poses with tremor — the right order

The generator already does 3000: `scripts/generate_tremor_dataset.py
--num-positions 3000` (random static finger pose per id + wrist tremor burst).
Add per-pose skin-tone + camera + tremor-type sampling, then gate on MediaPipe.

**Do not render 3000 first.** Sequence:
1. Fix §1 (black triangles). Render 5–10, confirm MediaPipe detects them.
2. `mini50`: 50 clips spanning the expanded conditions; require mean valid-frame
   rate ≥ 0.85, each clip ≥ 0.80, ≥ 45/50 accepted, and no skin-tone / camera /
   pose group below its floor (0.75 / 0.75 / 0.70). Print `MINI50 PASS` only if all hold.
3. Only then render the full set, **over-render ~15%** (≈3450) and keep the first
   3000 that pass detectability, since darker-tone/oblique clips will be rejected
   more often. **A rendered clip is not a training sample until MediaPipe detects
   it reliably.**

## 4. Important: decouple frequency from severity in the labels (see audit)

The current label model (`tremor_params`) sets
`frequency = 10 − 0.06·intensity` and `amplitude = 0.0015·intensity`, so
frequency and severity are the *same* axis (Pearson r ≈ −1). For the 3000-set to
support two *independent* claims (predict frequency AND predict severity), sample
a tremor **type** → frequency band and an **amplitude** independently, exactly as
`tremor_cv/generator.py` (`decoupled` mode) does. Otherwise a model that "predicts
both at 99%" has only measured one number. This is the headline data-leakage
finding from the audit below.
