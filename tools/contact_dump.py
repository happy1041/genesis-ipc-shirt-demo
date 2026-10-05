"""Optional raw libuipc contact diagnostics; no contact schema/force units assumed.

Call after IPC ``world.retrieve()`` on selected diagnostic frames. This helper
does not advance the world or change simulation geometry. It catches Python
exceptions so an unavailable exporter or output directory cannot stop a run.
Native process failures (e.g. a backend assertion) cannot be caught in Python.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path


def _error(exc):
    return {"type": type(exc).__name__, "message": str(exc)}


def _json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported diagnostic JSON value: {type(value).__name__}")


def _schema(value):
    """Describe actual exported keys without guessing collection names."""
    if isinstance(value, dict):
        return {str(key): _schema(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return {"type": "list", "length": len(value),
                "first_item": _schema(value[0]) if value else None}
    return type(value).__name__


def _slot_record(slot, **identity):
    record = dict(identity)
    try:
        geometry = slot.geometry()
        record.update(slot_id=int(slot.id()), meta=geometry.meta().to_json(),
                      instances=geometry.instances().to_json())
        # These values preserve global_vertex_offset wherever the backend puts
        # it (normally meta), along with raw link_idx/entity_idx/env_idx.
        if hasattr(geometry, "vertices"):
            record["vertex_count"] = int(geometry.vertices().size())
        if hasattr(geometry, "triangles"):
            record["triangle_count"] = int(geometry.triangles().size())
        record["schema"] = _schema({"meta": record["meta"],
                                    "instances": record["instances"]})
    except Exception as exc:
        record["error"] = _error(exc)
    return record


def _geometry_mapping(coupler):
    records, errors = [], []
    for link, slots in getattr(coupler, "_abd_slots_by_link", {}).items():
        for env, slot in enumerate(slots):
            records.append(_slot_record(
                slot, kind="abd", link_name=str(link.name),
                link_idx=int(link.idx), env_idx=env))
    # FEM geometry slots are not cached by the coupler. Resolve its documented
    # object names from _add_fem_entities_to_ipc, never assume contiguous IDs.
    try:
        scene = coupler._ipc_scene
        for entity_index, entity in enumerate(coupler.fem_solver.entities):
            offsets = getattr(coupler, "_fem_vertex_offsets", {}).get(entity, [])
            for env, fem_offset in enumerate(offsets):
                for prefix in ("cloth", "fem"):
                    name = f"{prefix}_{entity_index}_{env}"
                    for obj in scene.objects().find(name):
                        for geometry_id in obj.geometries().ids():
                            slot, _ = scene.geometries().find(int(geometry_id))
                            records.append(_slot_record(
                                slot, kind=prefix, object_name=name,
                                entity_idx=entity_index, env_idx=env,
                                fem_state_vertex_offset=int(fem_offset)))
    except Exception as exc:
        errors.append(_error(exc))
    return {"geometries": records, "errors": errors,
            "note": "fem_state_vertex_offset is NOT the contact global_vertex_offset; "
                    "use native geometry meta/instances to map contact IDs."}


def dump_contacts(coupler, output_path):
    """Write a new raw diagnostic JSON and return its status (never overwrite).

    Energy and gradient outputs are separate native Geometry.to_json() exports.
    No claim about per-pair IDs, activity, force units, or grasp success is made
    until the actual native schema has been inspected. Both normal and friction
    primitive types are exported using names returned by the feature itself.
    """
    report = {"format": "libuipc_raw_contact_dump_v1",
              "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "status": "started", "primitive_types": [], "contacts": [],
              "errors": [], "force_units": "unverified_native_gradient",
              "grasp_verdict": "not_evaluated"}
    try:
        from uipc.core import ContactSystemFeature
        from uipc.geometry import Geometry

        report["geometry_mapping"] = _geometry_mapping(coupler)
        world = coupler._ipc_world
        if world is None:
            raise RuntimeError("IPC world is not initialized")
        feature = world.features().find(ContactSystemFeature)
        if feature is None:
            raise RuntimeError("ContactSystemFeature is unavailable")
        types = list(feature.contact_primitive_types())
        report["primitive_types"] = types
        for primitive_type in types:
            entry = {"primitive_type": primitive_type}
            for kind in ("energy", "gradient"):
                try:
                    result = Geometry()
                    getattr(feature, "contact_" + kind)(primitive_type, result)
                    raw = result.to_json()
                    entry[kind] = {"raw": raw, "schema": _schema(raw)}
                except Exception as exc:
                    entry[kind] = {"error": _error(exc)}
                    report["errors"].append({"primitive_type": primitive_type,
                                             "operation": kind, **_error(exc)})
            report["contacts"].append(entry)
        report["status"] = "partial" if report["errors"] else "exported"
    except Exception as exc:
        report["status"] = "error"
        report["errors"].append(_error(exc))
    try:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(report, ensure_ascii=False, indent=2,
                             default=_json_default)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(payload + "\n")
        return {"status": report["status"], "path": str(path),
                "primitive_types": report["primitive_types"],
                "errors": report["errors"]}
    except Exception as exc:
        return {"status": "write_error", "errors": report["errors"] + [_error(exc)]}
