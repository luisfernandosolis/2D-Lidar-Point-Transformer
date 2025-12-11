"""
Small download helper module for datasets.
"""
import argparse
import os
import zipfile
from pathlib import Path
import subprocess
import sys

def download_from_gdrive(file_id: str, out_path: str):
    try:
        import gdown
    except Exception:
        print("gdown not found. You can install it with: pip install gdown")
        print(f"To download manually use: https://drive.google.com/uc?id={file_id} -> save to {out_path}")
        return False
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading {url} -> {out_path} using gdown...")
    gdown.download(url, out_path, quiet=False)
    return os.path.exists(out_path)

def unzip_file(zip_path: str, out_dir: str):
    print(f"Extracting {zip_path} -> {out_dir}")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(out_dir)
    print("Extraction completed.")

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    outputs = []
    if args.gdrive_ids:
        for gid in args.gdrive_ids:
            base = gid.get("id")
            name = gid.get("name", f"{base}.zip")
            dest = os.path.join(args.output_dir, name)
            ok = download_from_gdrive(base, dest)
            outputs.append((ok, dest))
            if ok and zipfile.is_zipfile(dest) and args.unzip:
                unzip_file(dest, args.output_dir)
    if args.raw_urls:
        for url in args.raw_urls:
            if url.startswith("http"):
                name = Path(url).name
                dest = os.path.join(args.output_dir, name)
                print(f"Downloading (curl/wget fallback) {url} -> {dest}")
                if sys.platform.startswith("win"):
                    cmd = ["powershell", "Invoke-WebRequest", "-Uri", f"'{url}'", "-OutFile", f"'{dest}'"]
                else:
                    cmd = ["curl", "-L", url, "-o", dest]
                subprocess.run(cmd)
                outputs.append((os.path.exists(dest), dest))
                if os.path.exists(dest) and zipfile.is_zipfile(dest) and args.unzip:
                    unzip_file(dest, args.output_dir)
    print("Download results:")
    for ok, dest in outputs:
        print(f" - {dest} : {'OK' if ok else 'MISSING'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download dataset files and optionally unzip them")
    parser.add_argument("--output-dir", "-o", type=str, default="dataset_download", help="Output directory")
    parser.add_argument("--unzip", action="store_true", help="Unzip if zip file(s) detected")
    parser.add_argument("--gdrive-ids", type=lambda s: eval(s), default=None,
                        help="A python list of dicts: [{'id': '1VQ3...', 'name': 'images.zip'}]")
    parser.add_argument("--raw-urls", type=lambda s: eval(s), default=None, help="A python list of raw URLs")
    args = parser.parse_args()
    main(args)
