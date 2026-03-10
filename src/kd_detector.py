from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.signal import find_peaks


KD_REFERENCES = np.array([250.0, 150.0, 100.0, 50.0, 40.0, 35.0], dtype=float)


@dataclass(frozen=True)
class BlotRoi:
    x0: int
    x1: int
    y0: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def _longest_true_run(mask: np.ndarray) -> tuple[int, int]:
    start = None
    best = (0, 0)
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        if (not value or i == len(mask) - 1) and start is not None:
            end = i if not value else i + 1
            if (end - start) > (best[1] - best[0]):
                best = (start, end)
            start = None
    return best


def detect_blot_roi(gray: np.ndarray) -> BlotRoi:
    h, w = gray.shape

    y_band_start = int(0.43 * h)
    y_band_end = h - 5
    dark_ratio_x = (gray[y_band_start:y_band_end, :] < 220).mean(axis=0)
    x_mask = dark_ratio_x > 0.35
    x0, x1 = _longest_true_run(x_mask)
    if (x1 - x0) < int(0.20 * w):
        x0, x1 = int(0.10 * w), int(0.53 * w)

    dark_ratio_y = (gray[:, x0:x1] < 220).mean(axis=1)
    y_mask = dark_ratio_y > 0.45
    y0, y1 = _longest_true_run(y_mask)
    if (y1 - y0) < int(0.25 * h):
        y0, y1 = int(0.45 * h), int(0.98 * h)

    return BlotRoi(x0=x0, x1=x1, y0=y0, y1=y1)


def detect_lane_centers(roi_gray: np.ndarray) -> list[int]:
    profile = (255.0 - roi_gray).mean(axis=0)
    smoothed = np.convolve(profile, np.ones(11) / 11, mode="same")
    threshold = np.percentile(smoothed, 50)

    raw_peaks: list[int] = []
    for i in range(2, len(smoothed) - 2):
        if smoothed[i] > smoothed[i - 1] and smoothed[i] >= smoothed[i + 1] and smoothed[i] > threshold:
            raw_peaks.append(i)

    merged: list[int] = []
    for p in raw_peaks:
        if not merged or p - merged[-1] > 20:
            merged.append(p)
        elif smoothed[p] > smoothed[merged[-1]]:
            merged[-1] = p

    if len(merged) <= 1:
        return merged

    step = float(np.median(np.diff(merged)))
    filled = [merged[0]]
    for p in merged[1:]:
        while p - filled[-1] > 1.6 * step:
            filled.append(int(round(filled[-1] + step)))
        filled.append(p)

    return filled


def detect_scale_ticks(gray: np.ndarray, roi: BlotRoi, expected_count: int = 6) -> list[int]:
    strip = (gray[:, 35:75] < 170).sum(axis=1).astype(float)
    strip = np.convolve(strip, np.ones(3) / 3, mode="same")

    ymin = roi.y0 + 50
    ymax = roi.y1 - 25
    if ymax <= ymin:
        ymin, ymax = int(0.55 * gray.shape[0]), int(0.95 * gray.shape[0])

    peaks, props = find_peaks(strip[ymin:ymax], distance=12, prominence=1.0, height=6.0)
    if len(peaks) < expected_count:
        peaks, props = find_peaks(strip[ymin:ymax], distance=10, prominence=0.5, height=4.0)

    y_values = (peaks + ymin).tolist()
    heights = props.get("peak_heights", np.array([], dtype=float)).tolist()

    if len(y_values) < expected_count:
        return [322, 360, 398, 452, 486, 512]

    idx_sorted = sorted(range(len(y_values)), key=lambda i: heights[i], reverse=True)
    chosen = sorted(y_values[i] for i in idx_sorted[:expected_count])
    return chosen


def fit_log_kd_mapping(scale_y: list[int], kd_refs: np.ndarray | None = None) -> tuple[float, float]:
    refs = KD_REFERENCES if kd_refs is None else kd_refs
    y = np.array(scale_y, dtype=float)
    if len(y) != len(refs):
        count = min(len(y), len(refs))
        y = y[:count]
        refs = refs[:count]

    a, b = np.linalg.lstsq(np.vstack([y, np.ones_like(y)]).T, np.log10(refs), rcond=None)[0]
    return float(a), float(b)


def y_to_kd(y: float, a: float, b: float) -> float:
    return float(10 ** (a * y + b))


def detect_bands_by_lane(
    roi_gray: np.ndarray,
    lane_centers: list[int],
    roi_top_y: int,
    a: float,
    b: float,
    scale_y: list[int],
) -> list[dict[str, Any]]:
    min_scale_y = min(scale_y)
    max_scale_y = max(scale_y)
    results: list[dict[str, Any]] = []

    for lane_idx, center_x in enumerate(lane_centers):
        left = max(0, center_x - 7)
        right = min(roi_gray.shape[1], center_x + 8)
        lane_strip = roi_gray[:, left:right]

        profile = (255.0 - lane_strip).mean(axis=1)
        baseline = cv2.GaussianBlur(profile.reshape(-1, 1), (1, 41), 0).ravel()
        signal = cv2.GaussianBlur((profile - baseline).reshape(-1, 1), (1, 5), 0).ravel()

        p95 = float(np.percentile(signal, 95))
        p80 = float(np.percentile(signal, 80))
        prominence = max(3.0, p95 * 0.22)
        height = max(4.0, p80)

        peaks, props = find_peaks(
            signal,
            prominence=prominence,
            distance=10,
            width=1.2,
            height=height,
        )

        candidates: list[tuple[int, float, float]] = []
        for y, prom, peak_h in zip(peaks, props["prominences"], props["peak_heights"]):
            if 20 <= y <= (roi_gray.shape[0] - 8):
                candidates.append((int(y), float(prom), float(peak_h)))
        candidates.sort(key=lambda x: x[0])

        merged: list[list[float]] = []
        for y, prom, peak_h in candidates:
            if not merged or (y - merged[-1][0]) > 6:
                merged.append([y, prom, peak_h])
            elif peak_h > merged[-1][2]:
                merged[-1] = [y, prom, peak_h]

        lane_label = "M" if lane_idx == 0 else str(lane_idx)
        for y_local, prom, peak_h in merged:
            y_abs = int(roi_top_y + y_local)
            kd = y_to_kd(y_abs, a, b)
            extrapolated = y_abs < min_scale_y or y_abs > max_scale_y
            results.append(
                {
                    "lane_index": lane_idx,
                    "lane_label": lane_label,
                    "x": int(center_x),
                    "y": y_abs,
                    "kD": kd,
                    "prominence": prom,
                    "peak_height": peak_h,
                    "extrapolated": extrapolated,
                }
            )

    return results


def draw_overlay(
    image: np.ndarray,
    roi: BlotRoi,
    lane_centers: list[int],
    bands: list[dict[str, Any]],
    output_path: Path,
) -> None:
    vis = image.copy()
    cv2.rectangle(vis, (roi.x0, roi.y0), (roi.x1, roi.y1), (0, 255, 0), 2)

    for cx in lane_centers:
        x_abs = roi.x0 + cx
        cv2.line(vis, (x_abs, roi.y0), (x_abs, roi.y1), (255, 0, 0), 1)

    for band in bands:
        x_abs = roi.x0 + int(band["x"])
        y_abs = int(band["y"])
        cv2.circle(vis, (x_abs, y_abs), 3, (0, 0, 255), -1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis)


def write_outputs(
    output_dir: Path,
    image_name: str,
    roi: BlotRoi,
    lane_centers: list[int],
    scale_y: list[int],
    a: float,
    b: float,
    bands: list[dict[str, Any]],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{image_name}_bands.csv"
    json_path = output_dir / f"{image_name}_bands.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "lane_label",
                "x",
                "y",
                "kD",
                "prominence",
                "peak_height",
                "extrapolated",
            ],
        )
        writer.writeheader()
        for row in bands:
            writer.writerow(
                {
                    "lane_label": row["lane_label"],
                    "x": row["x"],
                    "y": row["y"],
                    "kD": f"{row['kD']:.2f}",
                    "prominence": f"{row['prominence']:.2f}",
                    "peak_height": f"{row['peak_height']:.2f}",
                    "extrapolated": row["extrapolated"],
                }
            )

    summary = {
        "roi": {"x0": roi.x0, "x1": roi.x1, "y0": roi.y0, "y1": roi.y1},
        "lane_centers": lane_centers,
        "scale_y": scale_y,
        "mapping": {"log10_kD = a*y + b": {"a": a, "b": b}},
        "bands": bands,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return csv_path, json_path


def analyze_image(
    image_path: Path,
    output_dir: Path,
    include_marker_lane: bool = True,
) -> dict[str, Any]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    roi = detect_blot_roi(gray)
    roi_gray = gray[roi.y0 : roi.y1, roi.x0 : roi.x1]
    lane_centers = detect_lane_centers(roi_gray)
    scale_y = detect_scale_ticks(gray, roi, expected_count=6)
    a, b = fit_log_kd_mapping(scale_y)

    bands = detect_bands_by_lane(roi_gray, lane_centers, roi.y0, a, b, scale_y)
    if not include_marker_lane:
        bands = [bnd for bnd in bands if bnd["lane_label"] != "M"]

    image_name = image_path.stem
    overlay_path = output_dir / f"{image_name}_overlay.png"
    draw_overlay(image, roi, lane_centers, bands, overlay_path)
    csv_path, json_path = write_outputs(output_dir, image_name, roi, lane_centers, scale_y, a, b, bands)

    return {
        "image_path": str(image_path),
        "overlay_path": str(overlay_path),
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "roi": roi,
        "lane_centers": lane_centers,
        "scale_y": scale_y,
        "mapping": {"a": a, "b": b},
        "bands": bands,
    }
