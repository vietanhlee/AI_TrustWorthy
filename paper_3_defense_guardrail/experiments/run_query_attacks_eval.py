# -*- coding: utf-8 -*-
"""
Bước 5: Đánh giá Toàn diện Các Đòn Tấn công Query (Query Attacks Evaluation)
Bao gồm:
1. Encoder Query Fabrication (EQF, 15-step PGD) -> Bơm 300 candidates giả tại Encoder
2. Decoder Query Vanishing (Vanishing, 15-step PGD) -> Triệt tiêu query phát hiện vật thể
3. Decoder Query Fabrication (DQF, 15-step PGD) -> Kích hoạt query rác tạo box ma
Xuất log chi tiết, file CSV mẫu kiểm toán và file JSON metrics kèm tọa độ ROC.
"""

import os
import sys
import time
import pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
from tqdm import tqdm

# Đảm bảo import đúng workspace
EXPERIMENTS_DIR = Path(__file__).resolve().parent
PAPER3_DIR = EXPERIMENTS_DIR.parent
WORKSPACE_DIR = PAPER3_DIR.parent

if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))
if str(WORKSPACE_DIR / "edge_sec_pro") not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR / "edge_sec_pro"))

from edge_sec_pro.experiments.common_exp import (
    load_coco_images,
    clopper_pearson_interval,
    compute_auroc_and_bootstrap_ci,
    calculate_task_level_asr,
    make_serializable,
    RTDETRModelWrapper,
    RTDETRDeepFeatureExtractor,
    DETRAttackGenerator,
    CHECKPOINTS_DIR,
)

from paper_3_defense_guardrail.experiments.exp_logger import (
    setup_experiment_logger,
    SampleAuditTracker,
    save_audit_json,
    compute_roc_coordinates,
    compute_confusion_stats,
    PAPER3_DIR,
    RESULTS_DIR,
)


def run_step5():
    logger, log_file = setup_experiment_logger("step5_query_attacks")
    logger.info("BƯỚC 5: ĐÁNH GIÁ CÁC ĐÒN TẤN CÔNG QUERY (EQF, VANISHING, DQF) (N=300)")

    ckpt_file = CHECKPOINTS_DIR / "calibration_and_detectors.pkl"
    if not ckpt_file.exists():
        err_msg = f"Chưa chạy Bước 1! Không tìm thấy checkpoint: {ckpt_file}"
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)

    with open(ckpt_file, "rb") as f:
        ckpt = pickle.load(f)

    detectors = ckpt["detectors"]
    cal_thresholds = ckpt["cal_thresholds"]
    clean_test_scores = {k: np.array(v) for k, v in ckpt["clean_test_scores"].items()}

    device = torch.device("cpu")
    logger.info(f"Khởi tạo phần cứng: Device = {device}, PyTorch Threads = {torch.get_num_threads()}")

    wrapper = RTDETRModelWrapper(pretrained_model_name="PekingU/rtdetr_r18vd", device=device)
    wrapper.model.eval()
    extractor = RTDETRDeepFeatureExtractor(conf_threshold=0.35, top_k_queries=10, projection_dim=64)
    attacker = DETRAttackGenerator(wrapper, epsilon=16.0 / 255.0, alpha=2.0 / 255.0, num_steps=15)

    _, test_imgs = load_coco_images(n_cal=1200, n_test=600, img_size=320, seed=42)
    N_pgd = 300
    eval_imgs = test_imgs[:N_pgd]
    logger.info(f"Nạp dữ liệu COCO: Cohort N={N_pgd} ảnh kiểm thử độc lập.")

    # 1. Clean Baseline Forward Pass
    logger.info(f"\n1. Chạy Clean Forward Pass trên {N_pgd} ảnh...")
    batch_size = 15
    clean_logits_l = []
    t0_clean = time.time()
    for i in range(0, N_pgd, batch_size):
        b = eval_imgs[i : i + batch_size].to(device)
        with torch.no_grad():
            st = wrapper(b)
            clean_logits_l.append(st.logits.cpu())
            del st
    clean_logits = torch.cat(clean_logits_l, dim=0)
    logger.info(f"   -> Clean Pass hoàn tất trong {time.time() - t0_clean:.2f}s.")

    # Danh sách 3 đòn tấn công
    attacks = [
        ("eqf", "Tier 1: Encoder Query Fabrication (EQF, 15-step PGD)", "Fabrication", 1),
        ("vanishing", "Tier 2: Deformable Decoder Query Vanishing (15-step PGD)", "Vanishing", None),
        ("dqf", "Tier 2: Decoder Query Fabrication (DQF, 15-step PGD)", "Fabrication", 1),
    ]

    all_attack_results = {}
    audit_tracker = SampleAuditTracker("step5_query_attacks", output_dir=RESULTS_DIR)

    for atk_key, atk_desc, asr_type, target_cls in attacks:
        logger.info("\n" + "=" * 85)
        logger.info(f"BẮT ĐẦU ĐÁNH GIÁ: {atk_desc}")
        logger.info("=" * 85)

        # Kiểm tra Adv Cache
        adv_cache_file = CHECKPOINTS_DIR / "adv_cache" / f"adv_cache_{atk_key}_eps16_n{N_pgd}.npz"
        if adv_cache_file.exists():
            logger.info(f"-> Tìm thấy Precomputed Adversarial Cache tại: {adv_cache_file.name}")
            data_cache = np.load(adv_cache_file, allow_pickle=False)
            delta = torch.from_numpy(data_cache["delta_f16"].astype(np.float32))
            adv_imgs = torch.clamp(eval_imgs + delta, 0.0, 1.0)
            logger.info(f"   -> Đã nạp thành công {len(adv_imgs)} ảnh đối kháng từ cache.")
        else:
            logger.info(f"-> Sinh đòn tấn công {atk_key.upper()} (15 steps PGD, N={N_pgd})...")
            t0_gen = time.time()
            adv_list = []
            for i in tqdm(range(0, N_pgd, batch_size), desc=f"  Sinh {atk_key.upper()}"):
                sub_imgs = eval_imgs[i : i + batch_size]
                if atk_key == "eqf":
                    adv_sub = attacker.generate_encoder_query_fabrication_attack(sub_imgs, target_class_idx=target_cls, steps=15)
                elif atk_key == "vanishing":
                    adv_sub = attacker.generate_query_vanishing_attack(sub_imgs, steps=15)
                elif atk_key == "dqf":
                    adv_sub = attacker.generate_query_fabrication_attack(sub_imgs, target_class_idx=target_cls, steps=15)
                adv_list.append(adv_sub)
            adv_imgs = torch.cat(adv_list, dim=0)
            logger.info(f"   -> Sinh {atk_key.upper()} hoàn tất trong {time.time() - t0_gen:.1f}s.")

        # Forward pass trạng thái bị tấn công
        t0_adv_pass = time.time()
        adv_feats = []
        adv_logits_l = []
        for i in range(0, N_pgd, batch_size):
            b_adv = adv_imgs[i : i + batch_size].to(device)
            with torch.no_grad():
                st = wrapper(b_adv)
                adv_feats.extend(extractor.extract_from_state(st))
                adv_logits_l.append(st.logits.cpu())
                del st
        adv_logits = torch.cat(adv_logits_l, dim=0)
        logger.info(f"   -> Forward pass {atk_key.upper()} hoàn tất trong {time.time() - t0_adv_pass:.2f}s.")

        # Tính ASR
        asr, succ_cnt, valid_cnt, succ_mask = calculate_task_level_asr(asr_type, clean_logits, adv_logits, target_class_idx=target_cls)
        asr_ci = clopper_pearson_interval(succ_cnt, valid_cnt)
        logger.info(f"   -> Task-level Attack Success Rate (ASR) = {asr*100:5.1f}% [{asr_ci[0]*100:4.1f}%, {asr_ci[1]*100:4.1f}%] ({succ_cnt}/{valid_cnt})")

        # Đánh giá các detectors
        adv_scores = {name: det.score(adv_feats) for name, det in detectors.items()}
        method_metrics: Dict[str, Any] = {}
        roc_curves: Dict[str, Any] = {}

        logger.info(f"\n   KẾT QUẢ ĐÁNH GIÁ BỘ GIÁM SÁT TRƯỚC ĐÒN {atk_key.upper()}:")
        logger.info("   " + "-" * 90)
        logger.info(f"   {'Detector':<20} | {'AUROC':<10} | {'95% AUROC CI':<20} | {'TPR @ 5% FPR':<14} | {'95% Clopper-Pearson CI':<24}")
        logger.info("   " + "-" * 90)

        for m_name, det in detectors.items():
            s_neg = clean_test_scores[m_name]
            s_pos = adv_scores[m_name]
            labels = np.concatenate([np.zeros(len(s_neg)), np.ones(len(s_pos))])
            all_scores = np.concatenate([s_neg, s_pos])

            auroc, auroc_ci = compute_auroc_and_bootstrap_ci(labels, all_scores)
            roc_coords = compute_roc_coordinates(labels, all_scores, num_points=100)
            roc_curves[m_name] = roc_coords

            if m_name in ["ActiveQ_Split", "Top1Mean_Split", "ESM_Dropoff", "ESM_Margin"]:
                preds = det.predict(adv_feats)
            else:
                preds = (s_pos >= cal_thresholds[m_name]).astype(int)

            tp_cnt = int(np.sum(preds))
            tpr = tp_cnt / float(N_pgd)
            tpr_ci = clopper_pearson_interval(tp_cnt, N_pgd)
            conf = compute_confusion_stats(np.ones(N_pgd, dtype=int), preds)

            method_metrics[m_name] = {
                "auroc": round(float(auroc), 4),
                "auroc_ci": [round(float(auroc_ci[0]), 4), round(float(auroc_ci[1]), 4)],
                "tpr": round(float(tpr), 5),
                "tpr_ci": [round(float(tpr_ci[0]), 5), round(float(tpr_ci[1]), 5)],
                "tp_count": tp_cnt,
                "total_samples": N_pgd,
                "threshold": round(float(cal_thresholds[m_name]), 4),
                "confusion": conf,
            }

            ci_auroc_str = f"[{auroc_ci[0]:.3f}, {auroc_ci[1]:.3f}]"
            ci_tpr_str = f"[{tpr_ci[0]*100:4.1f}%, {tpr_ci[1]*100:4.1f}%]"
            logger.info(f"   {m_name:<20} | {auroc:5.3f}      | {ci_auroc_str:<20} | {tpr*100:5.1f}%        | {ci_tpr_str:<24}")

        logger.info("   " + "-" * 90)

        # Ghi nhận chi tiết từng mẫu
        for idx in range(N_pgd):
            s_dict = {name: float(adv_scores[name][idx]) for name in detectors}
            p_dict = {
                name: int(det.predict([adv_feats[idx]])[0]) if name in ["ActiveQ_Split", "Top1Mean_Split", "ESM_Dropoff", "ESM_Margin"]
                else int(adv_scores[name][idx] >= cal_thresholds[name])
                for name, det in detectors.items()
            }
            is_succ = int(succ_mask[idx]) if succ_mask is not None and idx < len(succ_mask) else -1
            audit_tracker.record_sample(
                sample_idx=idx,
                condition=f"query_attack_{atk_key}",
                ground_truth=1,
                scores=s_dict,
                preds=p_dict,
                is_attack_success=is_succ,
                extra_info={
                    "active_query_soft": float(adv_feats[idx].active_query_soft),
                    "query_top1_mean": float(adv_feats[idx].query_top1_mean),
                    "enc_dropoff": float(adv_feats[idx].enc_dropoff),
                    "enc_margin": float(adv_feats[idx].enc_margin),
                }
            )

        all_attack_results[atk_key] = {
            "attack_description": atk_desc,
            "task_asr": {
                "asr": round(float(asr), 5),
                "ci_95": [round(float(asr_ci[0]), 5), round(float(asr_ci[1]), 5)],
                "success_count": int(succ_cnt),
                "valid_count": int(valid_cnt),
            },
            "detectors_eval": method_metrics,
            "roc_curves": roc_curves,
        }

    # Xuất file CSV kiểm toán
    samples_csv_path = audit_tracker.save_csv()
    logger.info(f"\n-> Đã lưu bảng kiểm toán từng mẫu (Sample Audit CSV): {samples_csv_path}")

    # Xuất Audit JSON
    res_json_paper3 = RESULTS_DIR / "step5_query_attacks_metrics.json"
    res_json_global = CHECKPOINTS_DIR / "res_step5_query_attacks.json"

    export_payload = {
        "experiment": "step5_query_attacks",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {"device": str(device), "num_threads": torch.get_num_threads()},
        "eval_cohort_size": N_pgd,
        "attacks": all_attack_results,
        "artifacts": {
            "samples_csv": str(samples_csv_path),
            "log_file": str(log_file),
        }
    }

    save_audit_json(res_json_paper3, export_payload)
    save_audit_json(res_json_global, export_payload)

    logger.info(f"-> Đã lưu Audit JSON (Paper 3) : {res_json_paper3}")
    logger.info(f"-> Đã đồng bộ JSON toàn dự án  : {res_json_global}")
    logger.info("\n>>> BƯỚC 5 ĐÃ HOÀN TẤT THÀNH CÔNG VỚI ĐẦY ĐỦ DỮ LIỆU KIỂM TOÁN! <<<")


if __name__ == "__main__":
    run_step5()
