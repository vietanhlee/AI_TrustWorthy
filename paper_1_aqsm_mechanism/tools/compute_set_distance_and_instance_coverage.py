# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.tools.compute_set_distance_and_instance_coverage
Mục đích: Đo đạc chính xác các metric khoảng cách tập hợp bất biến hoán vị
(Permutation-Invariant Set Distances: Hungarian Matching Distance, Bidirectional Chamfer Distance)
và độ bao phủ thực tế đối tượng Ground Truth (GT Object Coverage: Cov_center, Cov_IoU>=0.30)
cho 4 điều kiện kiểm chứng không gian (Clean Baseline, Random Reassignment, Matched Random Shift, AQSM).

Ý nghĩa học thuật:
  1. Phản bác triệt để sự ngộ nhận về metric "Reference Drift / Center Displacement" ghép theo slot:
     Random Reassignment có khoảng cách ghép slot lên tới 0.3803 nhưng Hungarian/Chamfer distance
     bằng chính xác 0.0000, giải thích tại sao mAP giữ được 92.9% hiệu năng.
  2. Khắc phục mâu thuẫn giữa Sec 5.5 và Sec 5.7:
     Phân định rõ ràng giữa Anchor Foreground Hit Rate (Hit_FG) và GT Object Coverage (Cov_center).
  3. Bác bỏ giả thuyết hậu nghiệm "tình cờ chạm vật thể ngoại vi" của Random Shift bằng số liệu
     thực nghiệm định lượng về độ bao phủ đối tượng và độ phân tán không gian (Spatial Variance, ANND).
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from tqdm import tqdm

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from paper_1_aqsm_mechanism.core.adv_cache import load_cached_adv_images
from paper_1_aqsm_mechanism.core.dataset_loader import FlexibleCOCODataset
from paper_1_aqsm_mechanism.core.detr_wrapper import RTDETRModelWrapper
from paper_1_aqsm_mechanism.core.mechanism_analyzer import (
    _box_iou_np,
    _cxcywh_to_xyxy_np,
    compute_spatial_clustering_metrics,
    extract_encoder_components,
)
from paper_1_aqsm_mechanism.experiments.config import (
    RANDOM_SEED,
    get_adv_cache_dir,
    get_coco_ann_file,
    get_coco_val_dir,
    get_epsilon_l_inf,
    get_img_size,
    get_model_name,
    get_pgd_num_steps,
    get_summary_json_dir,
    resolve_model_name,
    update_experiment_paths,
)
from paper_1_aqsm_mechanism.experiments.tasks.task4_controlled_misrouting import sample_exact_matched_shift_4d


def run_set_distance_and_coverage_evaluation(
    max_samples: int = 100,
    device_str: str = "cpu",
    output_path: Optional[Path] = None,
    folder_image: Optional[str] = None,
    annotations_file: Optional[str] = None,
    cache_dir: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    device = torch.device(device_str)
    resolved_model = resolve_model_name(model_name or get_model_name())
    wrapper = RTDETRModelWrapper(pretrained_model_name=resolved_model, device=device)
    model = wrapper.model.eval()
    img_size = get_img_size()

    f_img = folder_image or str(get_coco_val_dir())
    f_ann = annotations_file or str(get_coco_ann_file())
    c_dir = Path(cache_dir) if cache_dir else get_adv_cache_dir()

    ds = FlexibleCOCODataset(
        folder_image=f_img,
        annotations_file=f_ann,
        img_size=img_size,
        id2label=model.config.id2label,
        lazy_tensors=True,
        device=device,
        max_samples=max_samples,
    )

    n_eval = len(ds)
    adv_aqsm_all, _ = load_cached_adv_images(
        attack="aqsm",
        eps_num=int(round(get_epsilon_l_inf() * 255.0)),
        n_eval=n_eval,
        clean_imgs=ds.tensors,
        expected_image_ids=ds.image_ids,
        cache_dir=c_dir,
        wrapper=wrapper,
        allow_onthefly=True,
        model_name=resolved_model,
        steps=get_pgd_num_steps(),
        hf_labels=getattr(ds, "hf_labels", None),
    )

    conditions = ["Clean_Baseline", "Random_Reassignment", "Random_Shift", "Decoupled_Random_Shift", "AQSM_Spatial_Misrouting"]
    stats: Dict[str, Dict[str, List[float]]] = {
        c: {
            "slot_center_disp": [],
            "hungarian_center_dist": [],
            "chamfer_center_dist": [],
            "anchor_fg_hit_pct": [],
            "gt_object_cov_center_pct": [],
            "gt_object_cov_iou30_pct": [],
            "annd": [],
            "spatial_variance": [],
        }
        for c in conditions
    }

    rng = np.random.RandomState(RANDOM_SEED)

    print("\n=======================================================")
    print(f"--> [Evaluation] Đo khoảng cách tập hợp và GT coverage trên N={n_eval} ảnh...")
    print("=======================================================")

    for i in tqdm(range(n_eval), desc="Set Distance & Coverage"):
        img_c = ds.tensors[i : i + 1].to(device)
        img_a = adv_aqsm_all[i : i + 1].to(device)
        gt_boxes = ds.gt_list[i].get("boxes_xyxy", np.zeros((0, 4), dtype=np.float32))

        with torch.no_grad():
            c_comp = extract_encoder_components(model, img_c)
            a_comp = extract_encoder_components(model, img_a)

        c_ref = torch.sigmoid(c_comp["init_reference_points"])[0].detach().cpu().numpy()
        a_ref = torch.sigmoid(a_comp["init_reference_points"])[0].detach().cpu().numpy()
        n_queries = int(c_ref.shape[0])

        perm = rng.permutation(n_queries)
        re_ref = c_ref[perm]

        sh_ref = np.zeros_like(c_ref)
        for j in range(n_queries):
            sh_ref[j] = sample_exact_matched_shift_4d(c_ref[j], a_ref[j], rng)

        cond_boxes = {
            "Clean_Baseline": c_ref,
            "Random_Reassignment": re_ref,
            "Random_Shift": sh_ref,
            "Decoupled_Random_Shift": sh_ref[perm],
            "AQSM_Spatial_Misrouting": a_ref,
        }

        for cname, ref in cond_boxes.items():
            cost = cdist(c_ref[:, :2], ref[:, :2])
            slot_d = float(np.mean(np.linalg.norm(c_ref[:, :2] - ref[:, :2], axis=1)))
            r_idx, c_idx = linear_sum_assignment(cost)
            hung_d = float(np.mean(cost[r_idx, c_idx]))
            cham_d = float(0.5 * (np.mean(np.min(cost, axis=1)) + np.mean(np.min(cost, axis=0))))

            clust = compute_spatial_clustering_metrics(ref[:, :2])

            if len(gt_boxes) > 0:
                cx, cy = ref[:, 0], ref[:, 1]
                in_x = (cx[:, None] >= gt_boxes[None, :, 0]) & (cx[:, None] <= gt_boxes[None, :, 2])
                in_y = (cy[:, None] >= gt_boxes[None, :, 1]) & (cy[:, None] <= gt_boxes[None, :, 3])
                fg_hit = float(np.mean(np.any(in_x & in_y, axis=1))) * 100.0
                cov_cen = float(np.mean(np.any(in_x & in_y, axis=0))) * 100.0

                ref_xyxy = _cxcywh_to_xyxy_np(ref)
                ious = _box_iou_np(ref_xyxy, gt_boxes)
                cov_iou30 = float(np.mean(ious.max(axis=0) >= 0.30)) * 100.0
            else:
                fg_hit = cov_cen = cov_iou30 = float("nan")

            stats[cname]["slot_center_disp"].append(slot_d)
            stats[cname]["hungarian_center_dist"].append(hung_d)
            stats[cname]["chamfer_center_dist"].append(cham_d)
            stats[cname]["anchor_fg_hit_pct"].append(fg_hit)
            stats[cname]["gt_object_cov_center_pct"].append(cov_cen)
            stats[cname]["gt_object_cov_iou30_pct"].append(cov_iou30)
            stats[cname]["annd"].append(clust["annd"])
            stats[cname]["spatial_variance"].append(clust["spatial_variance"])

    summary: Dict[str, Any] = {
        "metadata": {
            "n_samples": n_eval,
            "device": device_str,
            "description": "Permutation-invariant set distances and empirical GT instance coverage across controlled conditions",
        },
        "results": {},
    }

    for cname, metrics in stats.items():
        summary["results"][cname] = {}
        for m, l in metrics.items():
            arr = [x for x in l if not np.isnan(x)]
            summary["results"][cname][m] = {
                "mean": round(float(np.mean(arr)), 4),
                "std": round(float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0, 4),
            }

    if output_path is None:
        output_path = get_summary_json_dir() / "set_distance_and_instance_coverage.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n>>> [XUẤT KẾT QUẢ] Đã lưu summary vào: {output_path}")
    print("=" * 80)
    for cname, res in summary["results"].items():
        print(f"Condition: {cname}")
        for k, v in res.items():
            print(f"  - {k:26s}: {v['mean']:.4f} +- {v['std']:.4f}")
    print("=" * 80)
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Đo khoảng cách tập hợp bất biến hoán vị và GT Instance Coverage")
    parser.add_argument("--folder_image", type=str, default=str(get_coco_val_dir()))
    parser.add_argument("--annotations", type=str, default=str(get_coco_ann_file()))
    parser.add_argument("--cache_dir", type=str, default=str(get_adv_cache_dir()))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model", type=str, default=get_model_name())
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    update_experiment_paths(
        folder_image=args.folder_image,
        annotations=args.annotations,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        model=args.model,
    )
    run_set_distance_and_coverage_evaluation(
        max_samples=args.max_samples,
        device_str=args.device,
        folder_image=args.folder_image,
        annotations_file=args.annotations,
        cache_dir=args.cache_dir,
        model_name=args.model,
    )
