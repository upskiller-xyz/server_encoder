#!/usr/bin/env python3
"""
Download the latest dataset from S3 using the manifest-based system.

SAFE DOWNLOAD MECHANISM:
1. Fetch latest.json manifest (contains pointer to current snapshot)
2. Verify manifest has required fields
3. Download the referenced timestamped snapshot
4. Verify checksum matches manifest
5. Decompress if needed

Usage:
    python s3-download.py --output ./dataset.sqlite

Environment variables required:
    S3_ENDPOINT, S3_REGION, S3_BUCKET, S3_PREFIX (optional)
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def get_env(name: str, default: str = None) -> str:
    """Get environment variable or exit with error."""
    value = os.environ.get(name, default)
    if value is None:
        print(f"Error: Missing required environment variable: {name}")
        sys.exit(1)
    return value


def calculate_md5(file_path: Path) -> str:
    """Calculate MD5 checksum of a file."""
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


def download_latest(output_path: Path, decompress: bool = True, verify: bool = True) -> None:
    """Download the latest dataset from S3 using manifest."""
    endpoint = get_env("S3_ENDPOINT")
    region = get_env("S3_REGION")
    bucket = get_env("S3_BUCKET")
    prefix = os.environ.get("S3_PREFIX", "")

    # Build S3 path
    s3_base = f"s3://{bucket}"
    if prefix:
        s3_base = f"{s3_base}/{prefix}"

    # Step 1: Fetch the manifest
    manifest_url = f"{s3_base}/latest.json"
    print(f"[1/4] Fetching manifest from {manifest_url}...")

    try:
        result = subprocess.run(
            [
                "aws", "s3", "cp",
                manifest_url, "-",
                "--endpoint-url", endpoint,
                "--region", region,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        manifest = json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error: Could not fetch manifest: {e.stderr}")
        sys.exit(1)
    except json.JSONDecodeError:
        print("Error: Invalid manifest format")
        sys.exit(1)

    # Step 2: Validate manifest
    print("[2/4] Validating manifest...")
    required_fields = ["snapshot_key", "size_bytes", "md5"]
    missing = [f for f in required_fields if f not in manifest]
    if missing:
        print(f"Error: Manifest missing required fields: {missing}")
        sys.exit(1)

    print(f"  Timestamp: {manifest.get('timestamp', 'unknown')}")
    print(f"  Snapshot: {manifest['snapshot_key']}")
    print(f"  Datapoints: {manifest.get('datapoint_count', 'unknown')}")
    print(f"  Size: {manifest['size_bytes'] / 1024 / 1024:.1f} MB")
    print(f"  Compressed: {manifest.get('compressed', False)}")
    print(f"  MD5: {manifest['md5']}")

    # Step 3: Download the referenced snapshot
    snapshot_key = manifest["snapshot_key"]
    s3_path = f"{s3_base}/{snapshot_key}"
    is_compressed = snapshot_key.endswith(".zst")

    if is_compressed:
        download_path = output_path.with_suffix(".sqlite.zst")
    else:
        download_path = output_path

    print(f"[3/4] Downloading {snapshot_key}...")

    subprocess.run(
        [
            "aws", "s3", "cp",
            s3_path, str(download_path),
            "--endpoint-url", endpoint,
            "--region", region,
        ],
        check=True,
    )

    # Step 4: Verify checksum
    if verify:
        print("[4/4] Verifying checksum...")
        local_md5 = calculate_md5(download_path)
        expected_md5 = manifest["md5"]

        if local_md5 != expected_md5:
            print(f"Error: Checksum mismatch!")
            print(f"  Expected: {expected_md5}")
            print(f"  Got:      {local_md5}")
            download_path.unlink()  # Delete corrupted file
            sys.exit(1)

        print(f"  Checksum verified: {local_md5}")
    else:
        print("[4/4] Skipping checksum verification")

    # Decompress if needed
    if is_compressed and decompress:
        print(f"Decompressing to {output_path}...")
        subprocess.run(
            ["zstd", "-d", "--rm", str(download_path), "-o", str(output_path)],
            check=True,
        )
        final_file = output_path
    else:
        final_file = download_path

    # Final verification
    if final_file.exists():
        size_mb = final_file.stat().st_size / 1024 / 1024
        print(f"\nDownload complete: {final_file} ({size_mb:.1f} MB)")

        # Quick SQLite verification
        if final_file.suffix == ".sqlite":
            result = subprocess.run(
                ["sqlite3", str(final_file), "PRAGMA integrity_check"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and "ok" in result.stdout:
                # Count datapoints
                result = subprocess.run(
                    ["sqlite3", str(final_file), "SELECT COUNT(*) FROM datapoints"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    count = result.stdout.strip()
                    print(f"Database verified: {count} datapoints")
            else:
                print("Warning: Database integrity check failed")
    else:
        print("Error: Download failed - file not found")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download latest dataset from S3 (manifest-based, safe download)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("./dataset.sqlite"),
        help="Output path for the database (default: ./dataset.sqlite)",
    )
    parser.add_argument(
        "--no-decompress",
        action="store_true",
        help="Keep the file compressed (don't decompress .zst)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip checksum verification (not recommended)",
    )
    args = parser.parse_args()

    download_latest(
        args.output,
        decompress=not args.no_decompress,
        verify=not args.no_verify,
    )


if __name__ == "__main__":
    main()
