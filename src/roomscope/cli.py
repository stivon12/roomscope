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


if __name__ == "__main__":
    app()
