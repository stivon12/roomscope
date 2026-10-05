"""One command per capture: `roomscope run <capture>` -> JSON + rendered plan."""
from __future__ import annotations

import json
from pathlib import Path

import typer

from .detect import detect_tier

app = typer.Typer(add_completion=False, no_args_is_help=True)
SCHEMA = Path(__file__).resolve().parents[2] / "schema" / "plan.schema.json"


@app.command()
def run(
    capture: Path = typer.Argument(..., exists=True, help="Stray export dir, video file/dir, or folder of per-room photo folders"),
    out: Path = typer.Option(Path("out"), help="Output directory"),
    tier: str = typer.Option("auto", help="auto | photo | video | lidar"),
    drift: bool = typer.Option(True, "--drift/--no-drift", help="Drift correction (off only for the ablation)"),
    depth_correction: bool = typer.Option(True, "--depth-correction/--no-depth-correction",
                                          help="LiDAR depth-scale correction (off only for the ablation)"),
    depth_scale: float = typer.Option(None, help="Override depth scale a = measured/true (e.g. from a tape distance)"),
    device: str = typer.Option(None, help="Device model, selects its entry in config/depth_scale.yaml"),
):
    """Run the full pipeline on one capture."""
    from .pipeline import run_capture

    tier = detect_tier(capture) if tier == "auto" else tier
    typer.echo(f"[roomscope] tier={tier} capture={capture}")
    result_path = run_capture(capture, tier=tier, out_dir=out, drift=drift, depth_scale=depth_scale,
                              depth_correction=depth_correction, device=device)
    typer.echo(f"[roomscope] wrote {result_path}")


@app.command("calibrate-depth")
def calibrate_depth(
    capture: Path = typer.Argument(..., exists=True, help="Stray export of a room you can measure"),
    device: str = typer.Option(..., help="Phone model, e.g. 'iPhone 15 Pro' (Settings > General > About)"),
    measured: list[str] = typer.Option(..., help="Tape-measured reference, e.g. R1:ceiling=2.95 or R1-W3=3.42 (repeatable)"),
    ref_sd: float = typer.Option(0.003, help="Standard deviation of the tape reading (m)"),
    write: bool = typer.Option(True, "--write/--dry-run", help="Write the entry to config/depth_scale.yaml"),
    out: Path = typer.Option(Path("out/calibrate_depth"), help="Where the uncorrected run is written"),
):
    """Measure this phone's LiDAR depth scale from one capture plus one or more tape measurements, and
    store it under the device name so later runs with --device use it."""
    from .core.depth_calib import scale_from_references, write_device_entry
    from .pipeline import run_capture

    refs = {}
    for m in measured:
        k, _, v = m.partition("=")
        refs[k.strip()] = float(v)
    res = json.loads(run_capture(capture, tier="lidar", out_dir=out, depth_correction=False, calibrate=False,
                                 device=device).read_text())
    a, se, notes = scale_from_references(res, refs, ref_sd)
    for n in notes:
        typer.echo(f"  {n}")
    typer.echo(f"[roomscope] {device}: depth scale a = {a:.5f} ({100 * (a - 1):+.2f}%), se {100 * se:.2f}%")
    if write:
        write_device_entry(device, a, se, f"calibrate-depth on {capture.name}: " + "; ".join(notes))
        typer.echo("[roomscope] written to config/depth_scale.yaml")


@app.command()
def validate(result: Path = typer.Argument(..., exists=True)):
    """Validate a result JSON against the published schema."""
    import jsonschema

    jsonschema.validate(json.loads(result.read_text()), json.loads(SCHEMA.read_text()))
    typer.echo(f"{result}: valid")


WEIGHTS = ["depth-anything/DA3-BASE", "depth-anything/DA3METRIC-LARGE", "tue-mps/ade20k_semantic_eomt_large_512",
           "IDEA-Research/grounding-dino-base", "google/siglip2-base-patch16-384", "facebook/sam2.1-hiera-small"]
# each check runs in its own interpreter: pycolmap and torch must never share a process on macOS
CHECKS = {
    "core (LiDAR tier)": "import numpy, scipy, cv2, open3d, shapely, skimage, yaml, jsonschema",
    "torch": "import torch; print('mps' if torch.backends.mps.is_available() else 'cpu')",
    "Depth Anything 3": "import roomscope.frontends.da3_worker; import depth_anything_3.api",
    "COLMAP (pycolmap)": "import pycolmap",
    "EoMT (transformers)": "from transformers import EomtForUniversalSegmentation",
}


@app.command()
def doctor():
    """Check the install: Python, packages per tier, GPU, model weights. Exit code 1 if the LiDAR tier
    cannot run; video/photo problems are reported but do not fail."""
    import subprocess
    import sys

    ok_core = sys.version_info >= (3, 11)
    typer.echo(f"{'ok' if ok_core else 'FAIL':4}  python {sys.version.split()[0]} (needs >= 3.11)")
    for name, code in CHECKS.items():
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        good = r.returncode == 0
        if name.startswith("core"):
            ok_core &= good
        note = r.stdout.strip() if good else (r.stderr.strip().splitlines() or ["failed"])[-1]
        status = "ok" if good else ("FAIL" if name.startswith("core") or "ModuleNotFound" not in note else "--")
        if status == "--":
            note = "not installed (uv sync --extra video)"
        typer.echo(f"{status:4}  {name}{': ' + note if note else ''}")
    try:
        from huggingface_hub import try_to_load_from_cache
        for repo in WEIGHTS:
            hit = isinstance(try_to_load_from_cache(repo, "model.safetensors"), str)
            typer.echo(f"{'ok' if hit else 'miss':4}  weights {repo}" + ("" if hit else "  (scripts/fetch_weights.sh)"))
    except ImportError:
        typer.echo("miss  weights: huggingface_hub not installed (video/photo extra)")
    typer.echo("LiDAR tier: " + ("ready" if ok_core else "NOT ready") +
               "; video/photo tiers need every line above to be ok")
    raise typer.Exit(0 if ok_core else 1)


if __name__ == "__main__":
    app()
