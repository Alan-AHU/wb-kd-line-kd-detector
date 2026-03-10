from __future__ import annotations

import argparse
from pathlib import Path

from src.kd_detector import analyze_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect gray/black bands in blot-like images and convert y to kD."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("data/input_image.png"),
        help="Input image path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--drop-marker-lane",
        action="store_true",
        help="Exclude marker lane (M) from final outputs",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    result = analyze_image(
        image_path=args.image,
        output_dir=args.output_dir,
        include_marker_lane=(not args.drop_marker_lane),
    )

    print(f"Image: {result['image_path']}")
    print(f"Overlay: {result['overlay_path']}")
    print(f"CSV: {result['csv_path']}")
    print(f"JSON: {result['json_path']}")
    print(f"Detected lanes: {len(result['lane_centers'])}")
    print(f"Detected bands: {len(result['bands'])}")

    counts: dict[str, int] = {}
    for band in result["bands"]:
        lane = band["lane_label"]
        counts[lane] = counts.get(lane, 0) + 1

    for lane in sorted(counts.keys(), key=lambda s: (s != "M", int(s) if s != "M" else -1)):
        print(f"  lane {lane}: {counts[lane]} bands")


if __name__ == "__main__":
    main()
