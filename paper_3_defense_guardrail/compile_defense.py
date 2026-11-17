# -*- coding: utf-8 -*-
"""
Script: compile_defense.py
Biên dịch tự động bài báo Defense từ main_defense.tex -> main_defense.pdf
"""

import subprocess
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

DEFENSE_DIR = Path(__file__).resolve().parent

def run_cmd(cmd_list):
    print(f"Executing: {' '.join(cmd_list)}", flush=True)
    res = subprocess.run(cmd_list, cwd=str(DEFENSE_DIR), capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[!] Warning/Error during {cmd_list[0]}:")
        print(res.stdout[-1500:] if len(res.stdout) > 1500 else res.stdout)
        print(res.stderr[-1000:] if len(res.stderr) > 1000 else res.stderr)
        return False
    else:
        print(f"  + {cmd_list[0]} finished successfully.")
        return True

def main():
    print("=" * 80)
    print("BIÊN DỊCH BÀI BÁO DEFENSE (PDFLATEX + BIBTEX)")
    print("=" * 80)

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    job_name = "main_defense_build"

    # Lần 1: pdflatex
    if not run_cmd(["pdflatex", "-interaction=nonstopmode", f"-jobname={job_name}", "main_defense.tex"]):
        print("[!] Pass 1 hoàn tất với cảnh báo.")
    
    # Lần 2: bibtex
    run_cmd(["bibtex", job_name])

    # Lần 3: pdflatex
    run_cmd(["pdflatex", "-interaction=nonstopmode", f"-jobname={job_name}", "main_defense.tex"])

    # Lần 4: pdflatex để resolve hoàn toàn mọi trích dẫn và ref
    run_cmd(["pdflatex", "-interaction=nonstopmode", f"-jobname={job_name}", "main_defense.tex"])

    pdf_out = DEFENSE_DIR / f"{job_name}.pdf"
    target_pdf = DEFENSE_DIR / "main_defense.pdf"
    if pdf_out.exists():
        if target_pdf.exists():
            try:
                target_pdf.unlink()
            except Exception:
                pass
        try:
            pdf_out.rename(target_pdf)
            print(f"\n--> BÀI BÁO DEFENSE ĐÃ BIÊN DỊCH THÀNH CÔNG: {target_pdf} (Size: {target_pdf.stat().st_size:,} bytes)")
        except Exception:
            print(f"\n--> BÀI BÁO DEFENSE ĐÃ BIÊN DỊCH: {pdf_out}")
            target_pdf = pdf_out
            
        # Đồng bộ sang SUBMISSION_PDFS
        sub_dir = Path("G:/Edge_Sec/SUBMISSION_PDFS")
        if sub_dir.exists():
            dest = sub_dir / "5_Defense_Paper__main_defense.pdf"
            import shutil
            shutil.copyfile(target_pdf, dest)
            print(f"--> ĐÃ ĐỒNG BỘ VÀO: {dest}")

if __name__ == "__main__":
    main()
