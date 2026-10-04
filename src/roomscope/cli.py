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


@app.command()
def validate(result: Path = typer.Argument(..., exists=True)):
    """Validate a result JSON against the published schema."""
    import jsonschema

    jsonschema.validate(json.loads(result.read_text()), json.loads(SCHEMA.read_text()))
    typer.echo(f"{result}: valid")


if __name__ == "__main__":
    app()
