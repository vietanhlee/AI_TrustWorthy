"""
Script tu dong tai dataset COCO val2017 (5000 anh) va annotations chinh thuc tu cocodataset.org.
Chay doc lap tren moi moi truong: Kaggle, Colab, Linux Server, Windows.
"""

import os
import sys
import urllib.request
import zipfile
import shutil
from pathlib import Path

# URLs chinh thuc cua tap du lieu COCO 2017
COCO_VAL2017_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_VAL2017_ANNO_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

class DownloadProgressBar:
    def __init__(self, description: str):
        self.description = description
        self.pbar = None
        self.last_percent = -1

    def __call__(self, block_num: int, block_size: int, total_size: int):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, int(downloaded * 100 / total_size))
            if percent != self.last_percent and percent % 5 == 0:
                self.last_percent = percent
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                sys.stdout.write(f"\r{self.description}: {percent}% [{mb_downloaded:.1f}/{mb_total:.1f} MB]")
                sys.stdout.flush()
        else:
            mb_downloaded = downloaded / (1024 * 1024)
            sys.stdout.write(f"\r{self.description}: {mb_downloaded:.1f} MB downloaded")
            sys.stdout.flush()

def download_and_extract(url: str, dest_zip: Path, extract_dir: Path, description: str):
    print(f"\n[*] {description} tu: {url}")
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    
    if not dest_zip.exists():
        urllib.request.urlretrieve(url, str(dest_zip), DownloadProgressBar(description))
        print(f"\n[+] Da tai xong: {dest_zip.name} ({dest_zip.stat().st_size / (1024*1024):.1f} MB)")
    else:
        print(f"[*] File zip da ton tai: {dest_zip.name}, bo qua buoc download.")

    print(f"[*] Giai nen {dest_zip.name} vao {extract_dir}...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest_zip, 'r') as zf:
        zf.extractall(extract_dir)
    print(f"[+] Giai nen hoan tat: {dest_zip.name}")

def main():
    root_dir = Path(__file__).resolve().parent
    target_dir = root_dir / "coco5k"
    temp_dir = root_dir / "temp_coco_download"
    
    val2017_dir = target_dir / "val2017"
    anno_file = target_dir / "annotations" / "instances_val2017.json"
    
    # Kiem tra neu da co du 5000 anh va anno
    if val2017_dir.exists() and anno_file.exists():
        imgs = list(val2017_dir.glob("*.jpg"))
        if len(imgs) >= 5000:
            print(f"[+] COCO 5000 da ton tai day du tai: {target_dir} ({len(imgs)} anh, annotation hop le).")
            return

    temp_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download Images (val2017.zip ~780MB)
    images_zip = temp_dir / "val2017.zip"
    download_and_extract(COCO_VAL2017_IMAGES_URL, images_zip, target_dir, "Tai anh COCO val2017")

    # 2. Download Annotations (~241MB)
    anno_zip = temp_dir / "annotations_trainval2017.zip"
    anno_extract_temp = temp_dir / "anno_extract"
    download_and_extract(COCO_VAL2017_ANNO_URL, anno_zip, anno_extract_temp, "Tai annotations COCO")

    # Di chuyen instances_val2017.json vao coco5k/annotations/
    source_anno = anno_extract_temp / "annotations" / "instances_val2017.json"
    target_anno_dir = target_dir / "annotations"
    target_anno_dir.mkdir(parents=True, exist_ok=True)
    if source_anno.exists():
        shutil.copy2(source_anno, target_anno_dir / "instances_val2017.json")
        print(f"[+] Da luu annotation: {target_anno_dir / 'instances_val2017.json'}")

    # Don dep thu muc tam
    print("[*] Don dep file zip tam...")
    shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"\n========================================================")
    print(f"[SUCCESS] Dataset COCO 5000 da duoc cai dat hoan tat tai:")
    print(f"  - Images: {val2017_dir} ({len(list(val2017_dir.glob('*.jpg')))} anh)")
    print(f"  - Annotations: {anno_file}")
    print(f"========================================================")

if __name__ == "__main__":
    main()
