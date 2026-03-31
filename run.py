from __future__ import annotations

import argparse
from pathlib import Path

from src.kd_detector import analyze_image, write_batch_summary_excel

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect gray/black bands in blot-like images and convert y to kD."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Input image path (single image mode)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Input directory path (batch mode)",
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


def list_images(input_dir: Path) -> list[Path]:
    files = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(files, key=lambda p: str(p))


def print_result(result: dict) -> None:
    print(f"Image: {result['image_path']}")
    print(f"Annotated: {result['overlay_path']}")
    print(f"CSV: {result['csv_path']}")
    print(f"JSON: {result['json_path']}")
    print(f"XLSX: {result['xlsx_path']}")
    print(f"Detected lanes: {len(result['lane_centers'])}")
    print(f"Detected bands: {len(result['bands'])}")

    counts: dict[str, int] = {}
    for band in result["bands"]:
        lane = band["lane_label"]
        counts[lane] = counts.get(lane, 0) + 1

    for lane in sorted(counts.keys(), key=lambda s: int(s)):
        print(f"  lane {lane}: {counts[lane]} bands")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.input_dir is not None:
        input_dir = args.input_dir
        if not input_dir.exists() or not input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        images = list_images(input_dir)
        if not images:
            raise RuntimeError(f"No image files found in: {input_dir}")

        print(f"Batch mode: {len(images)} images")
        all_rows: list[dict] = []
        for idx, image_path in enumerate(images, start=1):
            print(f"[{idx}/{len(images)}] Processing: {image_path.name}")
            result = analyze_image(
                image_path=image_path,
                output_dir=args.output_dir,
                include_marker_lane=(not args.drop_marker_lane),
            )
            print_result(result)
            for band in result["bands"]:
                row = dict(band)
                row["image_name"] = image_path.name
                all_rows.append(row)

        summary_xlsx = args.output_dir / "batch_summary.xlsx"
        write_batch_summary_excel(all_rows, summary_xlsx)
        print(f"Batch summary XLSX: {summary_xlsx}")
        return

    image_path = args.image or Path("data/input_image.png")
    result = analyze_image(
        image_path=image_path,
        output_dir=args.output_dir,
        include_marker_lane=(not args.drop_marker_lane),
    )
    print_result(result)


if __name__ == "__main__":
    main()
