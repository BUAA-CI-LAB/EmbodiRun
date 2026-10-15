"""Render the published SO-101 medians; this script performs no measurements."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).with_name("recorded-summary.json")
OUTPUT = ROOT / "docs/assets/performance/so101-engine-comparison.svg"


def render(data: dict) -> str:
    """Draw separate latency panels without adding independently computed medians."""
    rows = data["results"]
    if data["evidence_kind"] != "published-summary" or len(rows) != 4:
        raise ValueError("expected the four published configuration summaries")
    conditions = data["conditions"]
    if conditions["statistic"] != "median":
        raise ValueError("expected recorded medians")
    title = html.escape(f"{conditions['robot']} · π0.5 · {conditions['inference_host']}")
    sample_label = html.escape(
        f"Median of {conditions['chunks_per_run']} chunks; "
        f"{conditions['runs_per_configuration']} run per configuration; lower is better"
    )
    description = html.escape(
        f"{conditions['inference_host']}, {conditions['chunks_per_run']} chunks per configuration. "
        f"Inference medians (ms): {', '.join(str(row['inference_ms']) for row in rows)}. "
        f"Full-chunk medians (ms): {', '.join(str(row['full_chunk_ms']) for row in rows)}. "
        "Source: published SO-101 engine comparison."
    )
    settings = html.escape(
        f"{conditions['chunk_steps']} actions at {conditions['control_hz']} Hz; "
        f"{conditions['denoising_steps']} denoising steps; {conditions['cameras']} cameras; "
        "engine optimization settings differ."
    )
    colors = ("#0072b2", "#56b4e9", "#d55e00", "#666666")
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1020 360" role="img" '
        'aria-labelledby="title description">',
        '<title id="title">SO-101 engine comparison: recorded medians</title>',
        f'<desc id="description">{description}</desc>',
        '<rect width="1020" height="360" rx="12" fill="#fff"/>',
        '<g font-family="sans-serif" fill="#172b4d">',
        f'<text x="24" y="30" font-size="20" font-weight="bold">{title}</text>',
        f'<text x="24" y="54" font-size="13">{sample_label}</text>',
    ]
    for column, (field, title, maximum) in enumerate(
        (("inference_ms", "Inference latency (ms)", 1200), ("full_chunk_ms", "Full chunk time (ms)", 4000))
    ):
        left = 24 + column * 500
        bar_start = left + 174
        bar_width = 250
        parts.append(f'<text x="{left}" y="88" font-size="16" font-weight="bold">{title}</text>')
        for tick in range(5):
            x = bar_start + tick * bar_width / 4
            parts.append(f'<path d="M{x:g} 102V270" stroke="#e5eaf0"/>')
            parts.append(f'<text x="{x:g}" y="292" text-anchor="middle" font-size="11">{maximum * tick // 4}</text>')
        for row_index, (row, color) in enumerate(zip(rows, colors)):
            value = row[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"invalid {field}")
            if not 0 <= value <= maximum:
                raise ValueError(f"{field} outside the plotted range")
            y = 112 + row_index * 40
            label = html.escape(f"{row['engine']} / {row['transport']}")
            width = value / maximum * bar_width
            parts.append(f'<text x="{left}" y="{y + 17}" font-size="11">{label}</text>')
            parts.append(f'<rect x="{bar_start}" y="{y}" width="{width:.3f}" height="24" fill="{color}"/>')
            parts.append(f'<text x="{bar_start + width + 5:.3f}" y="{y + 17}" font-size="12">{value:,}</text>')
    parts.extend(
        [
            f'<text x="24" y="324" font-size="12">{settings}</text>',
            '<text x="24" y="345" font-size="12" fill="#52647b">Source: SO-101 engine comparison. Data and measurement details are linked in the report.</text>',
            "</g></svg>",
        ]
    )
    return "\n".join(parts) + "\n"


def main() -> int:
    """Generate the tracked chart, or fail if it differs from its recorded data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify the tracked SVG without writing")
    args = parser.parse_args()
    svg = render(json.loads(DATA.read_text()))
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text() != svg:
            parser.exit(1, "Chart is stale; run python benchmarks/engine-comparison/generate_plot.py\n")
        print("Recorded-summary chart is current.")
    else:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(svg)
        print(OUTPUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
