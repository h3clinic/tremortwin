"""Probe the rigged hand GLB to recover joint/landmark rest geometry.

We want the rest-pose 3D positions of the wrist + finger joints so the
landmark-level digital twin uses the *real* rig proportions rather than
a generic canonical hand. Falls back gracefully if the GLB has no usable
skeleton (e.g. trimesh cannot read glTF skins)."""
import json
import sys
from pathlib import Path

import numpy as np

GLB = Path("input/hand_base/extracted/source/Do_Hand_DetailedRiggedAnimated_shared_16022026.glb")


def probe_trimesh():
    import trimesh
    scene = trimesh.load(str(GLB), force="scene")
    info = {"kind": type(scene).__name__, "geometry": {}, "graph_nodes": []}
    try:
        for name, geom in scene.geometry.items():
            info["geometry"][name] = {
                "vertices": int(geom.vertices.shape[0]),
                "bounds": geom.bounds.tolist(),
            }
    except Exception as e:
        info["geometry_error"] = repr(e)
    # Scene graph node names (often the bone/empty names)
    try:
        for node in scene.graph.nodes:
            T, _ = scene.graph.get(node)
            origin = np.array(T)[:3, 3].tolist()
            info["graph_nodes"].append({"node": str(node), "origin": [round(x, 5) for x in origin]})
    except Exception as e:
        info["graph_error"] = repr(e)
    return info


def probe_pygltflib():
    """Read raw glTF JSON for node hierarchy + skin joints (no mesh needed)."""
    import struct
    data = GLB.read_bytes()
    assert data[:4] == b"glTF", "not a GLB"
    # GLB: 12-byte header, then chunks. First chunk = JSON.
    json_len = struct.unpack("<I", data[12:16])[0]
    gltf = json.loads(data[20:20 + json_len].decode("utf-8"))
    nodes = gltf.get("nodes", [])
    out = {"num_nodes": len(nodes), "nodes": [], "skins": [], "animations": []}
    for i, n in enumerate(nodes):
        out["nodes"].append({
            "i": i,
            "name": n.get("name", f"node_{i}"),
            "translation": n.get("translation"),
            "rotation": n.get("rotation"),
            "children": n.get("children", []),
        })
    for s in gltf.get("skins", []):
        out["skins"].append({"name": s.get("name"), "joints": s.get("joints", [])})
    for a in gltf.get("animations", []):
        out["animations"].append({"name": a.get("name"), "channels": len(a.get("channels", []))})
    return out


if __name__ == "__main__":
    print("GLB exists:", GLB.exists(), "size_bytes:", GLB.stat().st_size if GLB.exists() else 0)
    result = {}
    try:
        result["gltf_json"] = probe_pygltflib()
    except Exception as e:
        result["gltf_json_error"] = repr(e)
    try:
        result["trimesh"] = probe_trimesh()
    except Exception as e:
        result["trimesh_error"] = repr(e)
    Path("tremor_cv/glb_probe.json").write_text(json.dumps(result, indent=2))
    # Compact stdout summary
    gj = result.get("gltf_json", {})
    print("num_nodes:", gj.get("num_nodes"))
    names = [n["name"] for n in gj.get("nodes", [])]
    print("node names (first 60):", names[:60])
    print("skins:", gj.get("skins"))
    print("animations:", gj.get("animations"))
    tm = result.get("trimesh", {})
    if "geometry" in tm:
        print("geometry:", {k: v["vertices"] for k, v in tm["geometry"].items()})
