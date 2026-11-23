# -*- coding: utf-8 -*-
"""
Module: edge_sec_pro.experiments.step_generate_adv_cache
Mục đích: Tiền sinh và lưu trữ (cache) tập ảnh đối kháng thống nhất trên cùng một
tập test COCO val2017 (N=300 ảnh đầu tiên từ test_paths, seed=42) và cấu hình
cố định (T=15, alpha=2/255, epsilon theo ngân sách) để mọi bước phân tích Mục 1-6
dùng chung 100% cùng dữ liệu đầu vào, loại bỏ sai lệch giữa các lần chạy.
"""

import argparse
import functools
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

print = functools.partial(print, flush=True)

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from edge_sec_pro.core.coco_gt import load_coco_val2017_gt_cohort
from edge_sec_pro.core.config import (
    AQSM_LAMBDA_RANK,
    AQSM_RANKING_MARGIN,
    CHECKPOINTS_DIR,
    PGD_NUM_STEPS,
    PGD_STEP_SIZE,
    RESULTS_DIR,
)
from edge_sec_pro.transformer_monitors.detr_attack_generator import DETRAttackGenerator
from edge_sec_pro.transformer_monitors.detr_wrapper import RTDETRModelWrapper

ADV_CACHE_DIR: Path = CHECKPOINTS_DIR / "adv_cache"
ADV_CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR: Path = RESULTS_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def eps_tag(eps_num: int) -> str:
    return f"eps{eps_num}"


def cache_path(attack: str, eps_num: int, n_eval: int) -> Path:
    return ADV_CACHE_DIR / f"adv_cache_{attack}_{eps_tag(eps_num)}_n{n_eval}.npz"


def generate_single_cache(
    attack: str,
    eps_num: int,
    n_eval: int = 300,
    batch_size: int = 10,
    steps: int = PGD_NUM_STEPS,
    alpha: float = PGD_STEP_SIZE,
    threads: int = 4,
    overwrite: bool = False,
) -> Path:
    torch.set_num_threads(threads)
    out_file = cache_path(attack, eps_num, n_eval)
    if out_file.exists() and not overwrite:
        print(f"[SKIP] Cache đã tồn tại: {out_file}")
        return out_file

    eps_val = float(eps_num) / 255.0
    device = torch.device("cpu")
    wrapper = RTDETRModelWrapper(pretrained_model_name="PekingU/rtdetr_r18vd", device=device)
    wrapper.model.eval()
    attacker = DETRAttackGenerator(wrapper, epsilon=eps_val, alpha=alpha, num_steps=steps)

    import pickle
    from edge_sec_pro.core.dataset import load_coco_images

    cohort = load_coco_val2017_gt_cohort(
        n_cal=1200,
        n_test=600,
        eval_start=0,
        n_eval=n_eval,
        seed=42,
        id2label=wrapper.model.config.id2label,
        load_tensors=True,
    )
    clean_imgs: torch.Tensor = cohort["tensors"]
    hf_labels: List[Dict[str, torch.Tensor]] = cohort["hf_labels"]
    image_ids = np.array(cohort["image_ids"], dtype=np.int64)

    ckpt_stats = None
    if attack == "adaptive1d":
        ckpt_file = CHECKPOINTS_DIR / "calibration_and_detectors.pkl"
        with open(ckpt_file, "rb") as f:
            ckpt_stats = pickle.load(f)["clean_stats"]

    t0 = time.time()
    adv_chunks: List[torch.Tensor] = []
    max_linf_err = 0.0
    l2_norms: List[float] = []

    if attack == "patch48":
        cal_imgs, _ = load_coco_images(n_cal=1200, n_test=600, img_size=640, seed=42)
        patch_48 = attacker.optimize_adversarial_patch(
            cal_imgs[:60], patch_size=48, patch_loc="random", target_class_idx=1, num_epochs=4
        )
        adv_all = attacker.apply_adversarial_patch(
            clean_imgs, patch_48, patch_loc="random", seed=1337
        ).detach().cpu()
        delta_all = adv_all - clean_imgs.cpu()
        max_linf_err = float(delta_all.abs().max().item())
        for j in range(delta_all.shape[0]):
            l2_norms.append(float(torch.norm(delta_all[j]).item()))
    else:
        desc = f"[Cache] {attack} (eps={eps_num}/255, N={n_eval}, T={steps})"
        for start in tqdm(range(0, n_eval, batch_size), desc=desc):
            end = min(n_eval, start + batch_size)
            sub_x = clean_imgs[start:end]
            sub_lbls = hf_labels[start:end]

            if attack == "aqsm":
                adv_sub = attacker.generate_query_selection_attack(
                    sub_x,
                    steps=steps,
                    alpha=alpha,
                    margin=AQSM_RANKING_MARGIN,
                    lambda_rank=AQSM_LAMBDA_RANK,
                )
            elif attack == "vanishing":
                adv_sub = attacker.generate_vanishing_attack(
                    sub_x,
                    steps=steps,
                    alpha=alpha,
                )
            elif attack == "taskpgd":
                adv_sub = attacker.generate_standard_task_pgd(
                    sub_x,
                    hf_labels=sub_lbls,
                    steps=steps,
                    alpha=alpha,
                )
            elif attack == "dqf":
                adv_sub = attacker.generate_fabrication_attack(
                    sub_x,
                    target_class_idx=1,
                    steps=steps,
                    alpha=alpha,
                )
            elif attack == "eqf":
                adv_sub = attacker.generate_encoder_fabrication_attack(
                    sub_x,
                    target_class_idx=1,
                    steps=steps,
                    alpha=alpha,
                )
            elif attack == "adaptive1d":
                adv_sub = attacker.generate_adaptive_attack(
                    sub_x,
                    target_class_idx=1,
                    evasion_weight=0.5,
                    steps=steps,
                    alpha=alpha,
                    loss_mode="hinge",
                    target_detector="1d",
                    tau_bounds_act=ckpt_stats["tau_act_bounds"],
                    tau_bounds_top1=ckpt_stats["tau_top_bounds"],
                    clean_act_mean=ckpt_stats["mu_act"],
                    clean_act_std=ckpt_stats["sd_act"],
                    clean_top1_mean=ckpt_stats["mu_top1"],
                    clean_top1_std=ckpt_stats["sd_top1"],
                    temperature=25.0,
                )
            else:
                raise ValueError(f"Attack không được hỗ trợ: {attack}")

            adv_cpu = adv_sub.detach().cpu()
            delta = adv_cpu - sub_x.cpu()
            batch_linf = float(delta.abs().max().item())
            max_linf_err = max(max_linf_err, batch_linf)
            for j in range(delta.shape[0]):
                l2_norms.append(float(torch.norm(delta[j]).item()))
            adv_chunks.append(adv_cpu)

        adv_all = torch.cat(adv_chunks, dim=0)
        # Kiểm chứng ngân sách L_inf
        assert max_linf_err <= eps_val + 1e-5, (
            f"Vi phạm ngân sách nhiễu L_inf: {max_linf_err:.6f} > {eps_val:.6f}"
        )

    elapsed = time.time() - t0
    meta = {
        "attack": attack,
        "n_eval": int(n_eval),
        "epsilon_num_255": int(eps_num),
        "epsilon": eps_val if attack != "patch48" else None,
        "alpha": float(alpha),
        "steps": int(steps),
        "margin": float(AQSM_RANKING_MARGIN) if attack == "aqsm" else None,
        "lambda_rank": float(AQSM_LAMBDA_RANK) if attack == "aqsm" else None,
        "lambda_supp": 0.5 if attack == "aqsm" else None,
        "evasion_weight": 0.5 if attack == "adaptive1d" else None,
        "max_linf": max_linf_err,
        "mean_l2": float(np.mean(l2_norms)),
        "std_l2": float(np.std(l2_norms)),
        "elapsed_sec": float(elapsed),
        "threads": int(threads),
    }

    # Lưu perturbation delta chuẩn IEEE 754 float32 nguyên bản (tuyệt đối không ép kiểu float16 hay uint8)
    delta_f32 = (adv_all - clean_imgs).numpy().astype(np.float32)
    np.savez_compressed(
        out_file,
        delta=delta_f32,
        delta_f32=delta_f32,
        delta_f16=delta_f32,
        image_ids=image_ids,
        l2_norms=np.array(l2_norms, dtype=np.float32),
        meta_json=np.array(json.dumps(meta)),
    )

    log_file = LOGS_DIR / f"cache_{attack}_{eps_tag(eps_num)}_n{n_eval}.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(
        f"[OK] Lưu {out_file.name}: max_Linf={max_linf_err:.5f} (limit {eps_val:.5f}), "
        f"mean_L2={meta['mean_l2']:.2f}, elapsed={elapsed:.1f}s"
    )
    return out_file


def load_cached_adv_images(
    attack: str,
    eps_num: int,
    n_eval: int,
    clean_imgs: torch.Tensor,
    expected_image_ids: List[int],
) -> Tuple[torch.Tensor, Dict]:
    fpath = cache_path(attack, eps_num, n_eval)
    if not fpath.exists():
        raise FileNotFoundError(f"Chưa tìm thấy cache đối kháng: {fpath}")
    data = np.load(fpath, allow_pickle=False)
    cached_ids = data["image_ids"].tolist()
    if cached_ids != list(expected_image_ids):
        raise ValueError(f"Image IDs không khớp giữa cache {fpath} và cohort hiện tại!")
    delta = torch.from_numpy(data["delta_f16"].astype(np.float32))
    adv = torch.clamp(clean_imgs + delta, 0.0, 1.0)
    meta = json.loads(str(data["meta_json"]))
    return adv, meta


def main():
    parser = argparse.ArgumentParser(description="Generate unified adversarial image cache on COCO val2017")
    parser.add_argument("--attacks", nargs="+", default=["aqsm", "vanishing", "taskpgd"])
    parser.add_argument("--eps-nums", nargs="+", type=int, default=[16])
    parser.add_argument("--n-eval", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--steps", type=int, default=PGD_NUM_STEPS)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for eps_num in args.eps_nums:
        for atk in args.attacks:
            generate_single_cache(
                attack=atk,
                eps_num=eps_num,
                n_eval=args.n_eval,
                batch_size=args.batch_size,
                steps=args.steps,
                alpha=PGD_STEP_SIZE,
                threads=args.threads,
                overwrite=args.overwrite,
            )


if __name__ == "__main__":
    main()
