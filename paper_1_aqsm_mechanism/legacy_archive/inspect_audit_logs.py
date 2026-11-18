# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.experiments.inspect_audit_logs
Mục đích: Công cụ tiện ích phục vụ Audit và Tái sử dụng dữ liệu thực nghiệm OFFLINE.
Người dùng có thể chạy script này để:
  1. Đọc và parse toàn bộ các file log (.log), audit stream (.jsonl), raw CSV, và summary JSON.
  2. Hiển thị báo cáo kiểm toán chi tiết từng task mà không cần chạy lại mô hình.
  3. Trích xuất metrics của từng step / epoch / batch để vẽ đồ thị hội tụ hoặc phân tích sâu.
  4. Xuất bảng Markdown chuẩn báo cáo.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUTS_DIR = WORKSPACE_DIR / "paper_1_aqsm_mechanism" / "outputs"
AUDIT_DIR = OUTPUTS_DIR / "audit_streams"
SUMMARY_DIR = OUTPUTS_DIR / "summary_json"
RAW_CSV_DIR = OUTPUTS_DIR / "raw_csv"
LOGS_DIR = WORKSPACE_DIR / "paper_1_aqsm_mechanism" / "logs"


def list_available_audit_data() -> Dict[str, Any]:
    """Liệt kê toàn bộ các tài nguyên kiểm toán hiện có trên đĩa."""
    data = {
        "audit_streams": [str(p.name) for p in AUDIT_DIR.glob("*.jsonl")],
        "summary_json": [str(p.name) for p in SUMMARY_DIR.glob("*.json")],
        "raw_csv": [str(p.name) for p in RAW_CSV_DIR.glob("*.csv")],
        "logs": [str(p.name) for p in LOGS_DIR.glob("*.log")],
    }
    return data


def inspect_raw_csv(csv_name: str, max_rows: int = 5) -> None:
    """Đọc và in thông tin bảng dữ liệu thô CSV kèm mẫu các dòng đầu tiên."""
    cp = RAW_CSV_DIR / csv_name
    if not cp.exists():
        print(f"[LỖI] Không tìm thấy file {csv_name} tại {RAW_CSV_DIR}")
        return

    df = pd.read_csv(cp)
    print(f"\n=======================================================")
    print(f"  CHI TIẾT BẢNG DỮ LIỆU THÔ CSV: {csv_name}")
    print(f"  Tổng số dòng: {len(df):,} | Tổng số cột: {len(df.columns)}")
    print(f"  Danh sách cột: {list(df.columns)}")
    print(f"=======================================================")
    print(df.head(max_rows).to_string(index=False))


def inspect_summary_json(json_name: str) -> None:
    """Đọc và in cấu trúc tóm tắt của một file summary JSON."""
    jp = SUMMARY_DIR / json_name
    if not jp.exists():
        print(f"[LỖI] Không tìm thấy file {json_name} tại {SUMMARY_DIR}")
        return

    with open(jp, mode="r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"\n=======================================================")
    print(f"  CHI TIẾT KIỂM TOÁN SUMMARY JSON: {json_name}")
    print(f"=======================================================")
    print(json.dumps(data, indent=2, ensure_ascii=False)[:3000])
    if len(json.dumps(data)) > 3000:
        print("\n... (đã cắt bớt phần sau, file gốc còn đầy đủ) ...")


def parse_audit_stream_sample(stream_name: str, max_records: int = 5) -> None:
    """Đọc và in mẫu các bản ghi trong audit stream JSONL."""
    sp = AUDIT_DIR / stream_name
    if not sp.exists():
        print(f"[LỖI] Không tìm thấy stream {stream_name} tại {AUDIT_DIR}")
        return

    records = []
    with open(sp, mode="r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= max_records:
                break
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"\n=======================================================")
    print(f"  MẪU BẢN GHI AUDIT STREAM (.JSONL): {stream_name}")
    print(f"  Hiển thị {len(records)} bản ghi đầu tiên:")
    print(f"=======================================================")
    for idx, r in enumerate(records):
        print(f"--- Bản ghi #{idx + 1} (Type: {r.get('record_type', 'UNKNOWN')}) ---")
        print(json.dumps(r, indent=2, ensure_ascii=False))


def generate_master_audit_report() -> None:
    """Tạo bảng đối chiếu kiểm toán toàn diện cho tất cả các bài toán."""
    print("\n" + "=" * 80)
    print("  BÁO CÁO TỔNG QUAN HỆ THỐNG DỮ LIỆU KIỂM TOÁN (AUDIT REPOSITORY)")
    print("=" * 80)

    avail = list_available_audit_data()
    print(f"\n1. Tệp Audit Streams (.jsonl) phục vụ parse theo dòng/step ({len(avail['audit_streams'])} tệp):")
    for f in avail["audit_streams"]:
        f_size = (AUDIT_DIR / f).stat().st_size / 1024
        print(f"   - {f:45s} ({f_size:.1f} KB)")

    print(f"\n2. Tệp Bảng Dữ Liệu Thô (.csv) phục vụ phân tích chi tiết từng ảnh ({len(avail['raw_csv'])} tệp):")
    for f in avail["raw_csv"]:
        f_size = (RAW_CSV_DIR / f).stat().st_size / 1024
        print(f"   - {f:45s} ({f_size:.1f} KB)")

    print(f"\n3. Tệp Báo Cáo Thống Kê Tổng Hợp (.json) phục vụ lập bảng biểu ({len(avail['summary_json'])} tệp):")
    for f in avail["summary_json"]:
        f_size = (SUMMARY_DIR / f).stat().st_size / 1024
        print(f"   - {f:45s} ({f_size:.1f} KB)")

    print(f"\n4. Tệp Nhật Ký Thực Thi (.log) phục vụ truy vết quá trình ({len(avail['logs'])} tệp):")
    for f in avail["logs"]:
        f_size = (LOGS_DIR / f).stat().st_size / 1024
        print(f"   - {f:45s} ({f_size:.1f} KB)")

    print("\n" + "=" * 80)
    print("  TẤT CẢ DỮ LIỆU ĐÃ SẴN SÀNG ĐỂ ĐỌC TRỰC TIẾP MÀ KHÔNG CẦN CHẠY LẠI MÔ HÌNH!")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Audit Inspector & Data Reuse Tool")
    parser.add_argument("--report", action="store_true", help="In báo cáo tổng quan toàn bộ repository kiểm toán")
    parser.add_argument("--json", type=str, default="", help="Tên file summary JSON cần kiểm tra")
    parser.add_argument("--csv", type=str, default="", help="Tên file raw CSV cần kiểm tra")
    parser.add_argument("--stream", type=str, default="", help="Tên file audit stream .jsonl cần đọc mẫu")
    args = parser.parse_args()

    if args.json:
        inspect_summary_json(args.json)
    elif args.csv:
        inspect_raw_csv(args.csv)
    elif args.stream:
        parse_audit_stream_sample(args.stream)
    else:
        generate_master_audit_report()


if __name__ == "__main__":
    main()
