"""Minimal benchmarking harness for CAD + load-case evaluation.

Input:
- STEP file path
- Load case JSON path (from ./data)

Metrics:
1) Design-space violation fraction (outside volume / design-space volume)
2) Load-path feasibility (binary via connectivity precondition)
3) FEA success + safety factor
4) Volume (proxy for cost)

-------------------------------------------------------------------------------
Usage
-------------------------------------------------------------------------------

Basic (single STEP file against a load case):

    python vendor/structbench/benchmark.py \\
        path/to/model.step \\
        vendor/structbench/data/json/l_bracket.json

Save the JSON result to a file:

    python vendor/structbench/benchmark.py \\
        path/to/model.step \\
        vendor/structbench/data/json/l_bracket.json \\
        --output-dir results/l_bracket \\
        --output-json results/l_bracket/result.json

Run against every load case in the data directory (shell loop):

    for json in vendor/structbench/data/json/*.json; do
        name=$(basename "$json" .json)
        python vendor/structbench/benchmark.py \\
            path/to/model.step "$json" \\
            --output-dir "results/$name" \\
            --output-json "results/$name/result.json"
    done

Available load cases (vendor/structbench/data/json/):
    a_frame_lateral.json          arch_bridge.json
    cantilever_bridge.json        crane_hook.json
    csg_bracket_with_hole.json    cylindrical_cantilever_lateral.json
    cylindrical_pressure_vessel.json  double_arch_bridge.json
    fixed_beam_area_load.json     fixed_beam_point_load.json
    frame_side_pressure.json      gravity_dam.json
    l_bracket_domain.json         l_bracket.json
    mast_side_load.json           plate_corners_fixed.json
    rectangular_plate_short_sides_fixed.json  t_beam_bending.json
    t_shaped_column.json

CLI arguments:
    step_file           Path to the STEP file to evaluate.
    load_case           Path to the load case JSON file.
    --output-dir DIR    Directory for mesh/FEA artefacts (default: ./results).
    --output-json PATH  Optional path to write the full JSON result.

Environment variables:
    RUNS_BASE_DIR       Base directory for run artefacts (default: ./runs).
    MESH_NODES          Target node count used to auto-size the mesh.
    MESH_SIZE           Fallback mesh element size in mm.
    MESHER              Mesher backend, e.g. gmsh-subprocess (default).

Output JSON structure:
    {
      "step_file": "<abs path>",
      "load_case_file": "<abs path>",
      "problem_id": "<id from JSON meta>",
      "metrics": {
        "design_space_violation_fraction_of_design_space_volume": <float>,
        "design_space_violation_fraction_of_generated_volume": <float>,
        "design_space_outside_volume_mm3": <float>,
        "design_space_volume_mm3": <float>,
        "load_path_ok": <bool>,
        "fea_success": <bool>,
        "safety_factor": <float|null>,
        "max_stress_mpa": <float|null>,
        "max_displacement_mm": <float|null>,
        "volume_mm3": <float>
      },
      "details": { ... }
    }
-------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
import types
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

logger = logging.getLogger(__name__)


def _load_module_from_path(module_name: str, module_path: Path):
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _import_fea_helpers():
    fea_tool_path = ROOT_DIR / "tools" / "fea_tool.py"

    try:
        fea_tool_module = _load_module_from_path("structbench_fea_tool", fea_tool_path)
        return (
            fea_tool_module.check_design_space,
            fea_tool_module.run_fea_analysis,
            fea_tool_module.validate_mesh,
        )
    except ModuleNotFoundError as exc:
        if exc.name != "pyvista":
            raise

        logger.warning(
            "pyvista not installed; using a lightweight stub for non-visual benchmark execution"
        )
        pv_stub = types.ModuleType("pyvista")
        pv_stub.OFF_SCREEN = True
        pv_stub.__file__ = "pyvista_stub.py"

        class _PVPlaceholder:
            pass

        class _PlotterStub:
            def __init__(self, *args, **kwargs):
                raise RuntimeError(
                    "pyvista Plotter unavailable in this environment"
                )

        def _pv_getattr(name: str):
            if name.startswith("__"):
                raise AttributeError(name)
            if name == "Plotter":
                return _PlotterStub
            return _PVPlaceholder

        pv_stub.Plotter = _PlotterStub
        pv_stub.UnstructuredGrid = _PVPlaceholder
        pv_stub.PolyData = _PVPlaceholder
        pv_stub.DataSet = _PVPlaceholder
        pv_stub.__getattr__ = _pv_getattr
        sys.modules["pyvista"] = pv_stub
        if "structbench_fea_tool" in sys.modules:
            del sys.modules["structbench_fea_tool"]

        fea_tool_module = _load_module_from_path("structbench_fea_tool", fea_tool_path)
        return (
            fea_tool_module.check_design_space,
            fea_tool_module.run_fea_analysis,
            fea_tool_module.validate_mesh,
        )


def _parse_load_case(load_case_path: str):
    parser_module = _load_module_from_path(
        "structbench_load_case_parser", ROOT_DIR / "tools" / "load_case_parser.py"
    )
    return parser_module.parse_load_case(load_case_path)


def _sum_volume(workplane_obj: Any) -> float:
    try:
        if hasattr(workplane_obj, "solids"):
            solids = workplane_obj.solids().vals()
            if solids:
                total = 0.0
                for solid in solids:
                    if hasattr(solid, "Volume"):
                        total += float(solid.Volume())
                if total > 0:
                    return total
        cq_obj = workplane_obj.val() if hasattr(workplane_obj, "val") else workplane_obj
        if hasattr(cq_obj, "Volume"):
            return float(cq_obj.Volume())
    except Exception:
        pass
    return 0.0


def _load_step_geometry(step_path: str) -> tuple[Any, dict[str, float], float]:
    import cadquery as cq

    wp = cq.importers.importStep(step_path)
    bbox = wp.val().BoundingBox()
    bounds = {
        "x_min": float(bbox.xmin),
        "x_max": float(bbox.xmax),
        "y_min": float(bbox.ymin),
        "y_max": float(bbox.ymax),
        "z_min": float(bbox.zmin),
        "z_max": float(bbox.zmax),
    }
    volume_mm3 = _sum_volume(wp)
    return wp, bounds, volume_mm3


def _compute_design_space_volume(load_case: Any) -> float:
    if getattr(load_case, "domain", None) is not None:
        try:
            domain_geometry = load_case.domain.build_geometry()
            volume = _sum_volume(domain_geometry)
            if volume > 0:
                return volume
        except Exception as e:
            logger.warning("Failed to compute domain geometry volume: %s", e)

    bounds = getattr(load_case, "bounds", None) or {}
    try:
        dx = max(0.0, float(bounds["x_max"]) - float(bounds["x_min"]))
        dy = max(0.0, float(bounds["y_max"]) - float(bounds["y_min"]))
        dz = max(0.0, float(bounds["z_max"]) - float(bounds["z_min"]))
        return dx * dy * dz
    except Exception:
        return 0.0


def _save_mesh_debug_plots(fea_analyzer: Any, output_dir: str) -> dict[str, Any]:
    """Save mesh/load-overlap debug plots using FEAAnalyzer.show.

    Returns paths (or errors) for generated debug artifacts.
    """
    debug_info: dict[str, Any] = {}

    debug_plot = os.path.join(output_dir, "mesh_overlap_debug.png")
    try:
        fea_analyzer.show(
            interactive=False,
            filename=debug_plot,
            display="debug",
            camera_position="iso",
        )
        debug_info["mesh_overlap_debug_plot"] = debug_plot
    except Exception as e:
        logger.warning("Failed to generate debug overlap plot: %s", e)
        debug_info["mesh_overlap_debug_plot_error"] = str(e)

    conditions_plot = os.path.join(output_dir, "mesh_overlap_conditions.png")
    try:
        fea_analyzer.show(
            interactive=False,
            filename=conditions_plot,
            display="conditions",
            camera_position="iso",
        )
        debug_info["mesh_overlap_conditions_plot"] = conditions_plot
    except Exception as e:
        logger.warning("Failed to generate conditions overlap plot: %s", e)
        debug_info["mesh_overlap_conditions_plot_error"] = str(e)

    return debug_info


def _resolve_mesh_size(load_case: Any, num_nodes: int, default_mesh_size: float) -> float:
    """Resolve mesh size using canonical LoadCase behavior with legacy fallback.

    Primary path: call `load_case.calc_mesh_size(num_nodes)` when available.
    Fallback path: compute using the same formula used by RapidCadPy's
    `LoadCase.calc_mesh_size` from bounds/domain.
    """
    if num_nodes <= 0:
        return float(default_mesh_size)

    if hasattr(load_case, "calc_mesh_size"):
        try:
            return float(load_case.calc_mesh_size(num_nodes))
        except Exception as e:
            logger.warning("LoadCase.calc_mesh_size failed, using fallback: %s", e)

    bounds = None
    domain = getattr(load_case, "domain", None)
    if domain is not None and hasattr(domain, "get_bounding_box"):
        try:
            bounds = domain.get_bounding_box()
        except Exception:
            bounds = None
    if bounds is None:
        bounds = getattr(load_case, "bounds", None)

    if not bounds:
        return float(default_mesh_size)

    try:
        dx = float(bounds.get("x_max", 0.0) - bounds.get("x_min", 0.0))
        dy = float(bounds.get("y_max", 0.0) - bounds.get("y_min", 0.0))
        dz = float(bounds.get("z_max", 0.0) - bounds.get("z_min", 0.0))

        positive_dims = [d for d in (dx, dy, dz) if d > 0.0]
        if not positive_dims:
            return float(default_mesh_size)

        min_dim = min(positive_dims)
        if dx > 0.0 and dy > 0.0 and dz > 0.0:
            volume = dx * dy * dz
            h_est = (volume / float(num_nodes)) ** (1.0 / 3.0)
        else:
            h_est = min_dim

        h_max = 0.5 * min_dim
        h_min = 1e-3
        h = max(h_min, min(h_est, h_max))
        return float(h)
    except Exception:
        return float(default_mesh_size)


def benchmark_step(step_path: str, load_case_path: str, output_dir: str) -> dict[str, Any]:
    from config import Config

    check_design_space, run_fea_analysis, validate_mesh = _import_fea_helpers()

    os.makedirs(output_dir, exist_ok=True)

    parsed_load_case = _parse_load_case(load_case_path)
    _, geometry_bounds, volume_mm3 = _load_step_geometry(step_path)

    design_check = check_design_space(
        cad=step_path,
        load_case=parsed_load_case,
        precomputed_bounds=geometry_bounds,
    )

    design_space_volume_mm3 = _compute_design_space_volume(parsed_load_case)
    outside_volume_mm3 = float(design_check.violation_volume_mm3 or 0.0)
    outside_fraction_design_volume = (
        outside_volume_mm3 / design_space_volume_mm3 if design_space_volume_mm3 > 0 else 0.0
    )

    mesh_size = _resolve_mesh_size(
        load_case=parsed_load_case,
        num_nodes=Config.MESH_NODES,
        default_mesh_size=Config.MESH_SIZE,
    )
    fea = parsed_load_case.get_fea_analyzer(
        mesher=Config.MESHER,
    )
    fea.mesh_size = mesh_size
    fea.shape = step_path

    mesh_result = validate_mesh(fea, output_dir)
    load_path_ok = bool(mesh_result.success and mesh_result.is_connected)
    mesh_debug_info: dict[str, Any] = {}
    if not load_path_ok:
        mesh_debug_info = _save_mesh_debug_plots(fea, output_dir)

    fea_success = False
    safety_factor = None
    max_stress_mpa = None
    max_displacement_mm = None
    fea_error = None
    fea_attempts = 0

    if load_path_ok:
        fea_result = run_fea_analysis(fea)
        fea_success = bool(fea_result.success)
        fea_attempts = int(fea_result.attempts)
        if fea_result.success:
            safety_factor = float(fea_result.safety_factor)
            max_stress_mpa = float(fea_result.max_stress_mpa)
            max_displacement_mm = float(fea_result.max_displacement_mm)
        else:
            fea_error = fea_result.error
    else:
        fea_error = mesh_result.error or "Load path/connectivity precondition failed"

    return {
        "step_file": os.path.abspath(step_path),
        "load_case_file": os.path.abspath(load_case_path),
        "problem_id": getattr(parsed_load_case, "problem_id", ""),
        "metrics": {
            "design_space_violation_fraction_of_design_space_volume": outside_fraction_design_volume,
            "design_space_violation_fraction_of_generated_volume": float(design_check.outside_ratio or 0.0),
            "design_space_outside_volume_mm3": outside_volume_mm3,
            "design_space_volume_mm3": float(design_space_volume_mm3),
            "load_path_ok": load_path_ok,
            "fea_success": fea_success,
            "safety_factor": safety_factor,
            "max_stress_mpa": max_stress_mpa,
            "max_displacement_mm": max_displacement_mm,
            "volume_mm3": float(volume_mm3),
        },
        "details": {
            "design_space_ok": bool(design_check.success),
            "design_space_error": design_check.error,
            "geometry_bounds": design_check.geometry_bounds,
            "design_bounds": design_check.design_bounds,
            "design_space_violations": design_check.violations,
            "mesh_error": mesh_result.error,
            "fea_error": fea_error,
            "fea_attempts": fea_attempts,
            **mesh_debug_info,
        },
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run structbench minimal benchmark")
    parser.add_argument("step_file", help="Path to input STEP file")
    parser.add_argument("load_case", help="Path to load case JSON file")
    parser.add_argument(
        "--output-dir",
        default="./results",
        help="Directory for temporary/output artifacts",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write full JSON benchmark result",
    )
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_arg_parser().parse_args()

    result = benchmark_step(args.step_file, args.load_case, args.output_dir)

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2))
        print(f"Wrote benchmark result to {output_json}")

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
