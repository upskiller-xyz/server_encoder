"""
Visualize encoder output .npz files as PNG images.

Reads .npz files produced by generate_training_data.py and saves
per-channel PNG images + a combined overview for visual inspection.

Usage:
    python scripts/visualize_encoding.py --input ./training_output/ --output ./training_viz/
"""
import os
import sys
import argparse
from typing import Dict

import numpy as np
import cv2


class EncodingVisualizer:
    """Visualizes encoder RGBA numpy arrays as human-readable PNGs."""

    CHANNEL_NAMES = ["Red", "Green", "Blue", "Alpha"]

    @staticmethod
    def visualize_npz(npz_path: str, output_dir: str) -> None:
        """
        Visualize all arrays in a .npz file.

        Args:
            npz_path: Path to .npz file
            output_dir: Directory to save PNG images
        """
        basename = os.path.splitext(os.path.basename(npz_path))[0]
        dp_dir = os.path.join(output_dir, basename)
        os.makedirs(dp_dir, exist_ok=True)

        data = np.load(npz_path)

        for key in data.files:
            arr = data[key]

            if "mask" in key:
                EncodingVisualizer._save_mask(arr, os.path.join(dp_dir, f"{key}.png"))
            elif "image" in key:
                EncodingVisualizer._save_image(arr, dp_dir, key)
            elif arr.ndim == 2:
                cv2.imwrite(os.path.join(dp_dir, f"{key}.png"), arr)

        print(f"  {basename}: {len(data.files)} arrays visualized -> {dp_dir}")

    @staticmethod
    def _save_mask(arr: np.ndarray, output_path: str) -> None:
        """Save a binary mask as green/black PNG."""
        mask_bgr = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
        mask_bgr[arr > 0] = [0, 200, 0]
        cv2.imwrite(output_path, mask_bgr)

    @staticmethod
    def _save_image(arr: np.ndarray, dp_dir: str, key: str) -> None:
        """Save an RGBA image as full PNG + individual colormapped channel PNGs."""
        if arr.ndim != 3 or arr.shape[2] != 4:
            return

        # Save full RGBA as PNG (BGRA for OpenCV)
        bgra = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(os.path.join(dp_dir, f"{key}_rgba.png"), bgra)

        # Save individual channels as colormapped images
        channel_imgs = []
        for ch in range(4):
            channel = arr[:, :, ch]
            colored = cv2.applyColorMap(channel, cv2.COLORMAP_VIRIDIS)
            label = EncodingVisualizer.CHANNEL_NAMES[ch]
            cv2.putText(
                colored, label, (2, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1
            )
            channel_imgs.append(colored)
            cv2.imwrite(
                os.path.join(dp_dir, f"{key}_ch{ch}_{label.lower()}.png"),
                colored
            )

        # Combined overview: 4 channels side by side
        overview = np.hstack(channel_imgs)
        cv2.imwrite(os.path.join(dp_dir, f"{key}_overview.png"), overview)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize encoder output .npz files as PNG images"
    )
    parser.add_argument(
        "--input", required=True, help="Directory with .npz files"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory for PNGs (default: <input>_viz)"
    )

    args = parser.parse_args()

    if args.output is None:
        args.output = args.input.rstrip("/\\") + "_viz"

    os.makedirs(args.output, exist_ok=True)

    npz_files = sorted([
        f for f in os.listdir(args.input) if f.endswith(".npz")
    ])

    if not npz_files:
        print(f"No .npz files found in {args.input}")
        return

    print(f"Visualizing {len(npz_files)} files...")
    for npz_file in npz_files:
        npz_path = os.path.join(args.input, npz_file)
        EncodingVisualizer.visualize_npz(npz_path, args.output)

    print(f"\nDone! Images saved to: {args.output}")


if __name__ == "__main__":
    main()
