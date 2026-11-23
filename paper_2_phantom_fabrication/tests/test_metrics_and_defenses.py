# -*- coding: utf-8 -*-
"""
Unit Tests for Paper 2: "Adversarial Object Fabrication in RT-DETR"
===================================================================
Kiểm thử tự động tính đúng đắn khoa học của:
  1. Strict Phantom Query Matching (IoU < 0.3 với GT cùng target class).
  2. Bóc tách phân loại lỗi: TP, FP_target, FP_other.
  3. Multi-level Attack Success Rate (ASR >= 1, ASR >= 3, ASR >= 5).
  4. DIoU-NMS và Soft-NMS.
  5. SQD-Filter: Giải quyết nghịch lý hình học (tâm gần nhau nhưng IoU thấp).
  6. JPEG Compression Defense: Bảo toàn không gian màu RGB và độ chuẩn hóa.
  7. Representation Guardrails: Hiệu chuẩn Prototype và tính AUROC phát hiện đối kháng.
"""

import os
import sys
from pathlib import Path

# Đảm bảo xử lý lỗi OpenMP duplicate runtime trên môi trường Windows / Anaconda
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Tự động nạp thư mục gốc của repository vào sys.path để pytest / unittest chạy trực tiếp
TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import unittest
import numpy as np
import torch

from paper_2_phantom_fabrication.core.matcher import _match_single_image, match_detections_against_gt
from paper_2_phantom_fabrication.core.metrics import (
    compute_disentangled_precision_breakdown,
    compute_phantom_metrics,
)
from paper_2_phantom_fabrication.defenses.baseline_defenses import (
    JPEGCompressionDefense,
    containment_nms_single_image,
    diou_nms_single_image,
    soft_nms_single_image,
)
from paper_2_phantom_fabrication.defenses.query_guardrail import (
    QueryRepresentationGuardrail,
    evaluate_guardrail_performance,
)
from paper_2_phantom_fabrication.defenses.sqd_filter import apply_sqd_filter


class TestPaper2MetricsAndDefenses(unittest.TestCase):

    def test_strict_phantom_matching(self):
        """Kiểm tra logic phân loại TP, FP_target (Phantom) và FP_other."""
        # GT: 1 Person (cls=1) tại [0.1, 0.1, 0.3, 0.3], 1 Car (cls=2) tại [0.7, 0.7, 0.9, 0.9]
        gt_boxes = np.array([
            [0.1, 0.1, 0.3, 0.3],
            [0.7, 0.7, 0.9, 0.9]
        ], dtype=np.float64)
        gt_labels = np.array([1, 2], dtype=int)

        # Preds:
        # 1. Person thật: [0.11, 0.11, 0.31, 0.31], conf=0.95 -> TP
        # 2. Person ma (Phantom): [0.4, 0.4, 0.5, 0.5], conf=0.88 -> FP_target (IoU với person GT = 0)
        # 3. Car giả (lớp khác): [0.3, 0.3, 0.4, 0.4], conf=0.75 -> FP_other
        pred_boxes = np.array([
            [0.11, 0.11, 0.31, 0.31],
            [0.40, 0.40, 0.50, 0.50],
            [0.30, 0.30, 0.40, 0.40]
        ], dtype=np.float64)
        pred_scores = np.array([0.95, 0.88, 0.75], dtype=np.float64)
        pred_labels = np.array([1, 1, 2], dtype=int)

        res = _match_single_image(
            pred_boxes=pred_boxes,
            pred_scores=pred_scores,
            pred_labels=pred_labels,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            target_cls=1,
            iou_thresh=0.50,
            phantom_iou_max=0.30,
        )

        self.assertEqual(res["tp"], 1, "Cần khớp chính xác 1 True Positive")
        self.assertEqual(res["fp"], 2, "Cần khớp chính xác 2 False Positives tổng cộng")
        self.assertEqual(res["fp_target"], 1, "Cần tách chính xác 1 FP thuộc lớp mục tiêu")
        self.assertEqual(res["fp_other"], 1, "Cần tách chính xác 1 FP thuộc lớp khác")
        self.assertEqual(res["phantoms"], 1, "Phải nhận diện đúng 1 Phantom Query")
        self.assertEqual(res["is_asr_1"], 1, "Ảnh có >= 1 phantom nên is_asr_1 phải bằng 1")
        self.assertEqual(res["is_asr_3"], 0, "Chỉ có 1 phantom nên is_asr_3 phải bằng 0")
        self.assertEqual(res["is_asr_5"], 0, "Chỉ có 1 phantom nên is_asr_5 phải bằng 0")

    def test_multilevel_asr_aggregation(self):
        """Kiểm tra tổng hợp ASR đa cấp độ (>=1, >=3, >=5) qua nhiều ảnh."""
        # 4 ảnh giả lập:
        # Ảnh 0: 0 phantoms
        # Ảnh 1: 1 phantom
        # Ảnh 2: 3 phantoms
        # Ảnh 3: 6 phantoms
        matched_results = [
            {"active": 5, "tp": 2, "fp": 3, "phantoms": 0, "is_asr_1": 0, "is_asr_3": 0, "is_asr_5": 0},
            {"active": 6, "tp": 2, "fp": 4, "phantoms": 1, "is_asr_1": 1, "is_asr_3": 0, "is_asr_5": 0},
            {"active": 8, "tp": 2, "fp": 6, "phantoms": 3, "is_asr_1": 1, "is_asr_3": 1, "is_asr_5": 0},
            {"active": 12, "tp": 2, "fp": 10, "phantoms": 6, "is_asr_1": 1, "is_asr_3": 1, "is_asr_5": 1},
        ]

        metrics = compute_phantom_metrics(matched_results)
        self.assertAlmostEqual(metrics["asr_1_pct"], 75.0, places=1, msg="ASR >= 1 phải là 75% (3/4 ảnh)")
        self.assertAlmostEqual(metrics["asr_3_pct"], 50.0, places=1, msg="ASR >= 3 phải là 50% (2/4 ảnh)")
        self.assertAlmostEqual(metrics["asr_5_pct"], 25.0, places=1, msg="ASR >= 5 phải là 25% (1/4 ảnh)")
        self.assertAlmostEqual(metrics["fp_per_image"], 2.5, places=1, msg="Số phantom TB/ảnh phải là 10/4 = 2.5")

    def test_diou_and_soft_nms(self):
        """Kiểm tra thuật toán DIoU-NMS và Soft-NMS."""
        # 2 boxes đè lên nhau (IoU ~ 0.85): Box 0 có điểm 0.90, Box 1 có điểm 0.80
        boxes = np.array([
            [0.10, 0.10, 0.40, 0.40],
            [0.12, 0.12, 0.41, 0.41]
        ], dtype=np.float64)
        scores = np.array([0.90, 0.80], dtype=np.float64)

        # DIoU-NMS với threshold 0.50 phải loại bỏ Box 1
        keep_diou = diou_nms_single_image(boxes, scores, iou_thresh=0.50)
        self.assertEqual(keep_diou, [0], "DIoU-NMS phải giữ lại duy nhất Box 0 có điểm cao nhất")

        # Soft-NMS với Gaussian decay: ở score_thresh=0.10, cả 2 hộp được giữ lại nhưng điểm hộp 1 giảm
        b_soft, s_soft, keep_soft = soft_nms_single_image(boxes, scores, sigma=0.50, score_thresh=0.10)
        self.assertEqual(len(b_soft), 2, "Soft-NMS với threshold thấp giữ 2 hộp nhưng làm giảm điểm hộp trùng")
        self.assertLess(s_soft[1], 0.80, "Điểm số của hộp thứ 2 phải bị suy giảm qua trọng số Gaussian")

        # Ở score_thresh chuẩn 0.35, điểm số của hộp thứ 2 (~0.19) bị rơi xuống dưới ngưỡng nên bị loại
        b_soft_oper, _, _ = soft_nms_single_image(boxes, scores, sigma=0.50, score_thresh=0.35)
        self.assertEqual(len(b_soft_oper), 1, "Ở threshold 0.35, hộp thứ 2 bị loại do điểm suy giảm")

    def test_containment_nms(self):
        """
        Kiểm tra cơ chế Containment-NMS (Intersection-over-Min-Area):
        Hộp nhỏ đồng tâm nằm lọt trong hộp lớn:
        IoU < 0.25 (IoU-NMS thất bại) nhưng IoMin = 1.0 (Containment-NMS thành công).
        """
        # Box 0: Hộp lớn [0.2, 0.2, 0.8, 0.8], area = 0.36, score = 0.90
        # Box 1: Hộp nhỏ lọt trong [0.4, 0.4, 0.6, 0.6], area = 0.04, score = 0.80
        # Box 2: Hộp riêng biệt [0.0, 0.0, 0.1, 0.1], area = 0.01, score = 0.85
        boxes = np.array([
            [0.20, 0.20, 0.80, 0.80],
            [0.40, 0.40, 0.60, 0.60],
            [0.00, 0.00, 0.10, 0.10],
        ], dtype=np.float64)
        scores = np.array([0.90, 0.80, 0.85], dtype=np.float64)

        # IoU giữa Box 0 và Box 1 là 0.04 / 0.36 = 0.111 < 0.50 -> Standard NMS không gộp
        # IoMin giữa Box 0 và Box 1 là 0.04 / min(0.36, 0.04) = 1.0 >= 0.50 -> Containment-NMS gộp
        keep = containment_nms_single_image(boxes, scores, iomin_thresh=0.50)
        self.assertEqual(keep, [0, 2], "Containment-NMS phải gộp Box 1 vào Box 0 và giữ lại Box 0 cùng Box 2")

    def test_sqd_filter_spatial_geometry_paradox(self):
        """
        Kiểm chứng Hiện tượng Nghịch lý Hình học Không gian:
        Hai query đa tầng có tâm gần nhau (cùng focal patch) nhưng tỷ lệ hộp khác nhau dẫn đến IoU < 0.10.
        IoU-NMS thất bại nhưng SQD-Filter thành công.
        """
        # Box 1: [cx=0.5, cy=0.5, w=0.1, h=0.1] -> vuông nhỏ
        # Box 2: [cx=0.51, cy=0.49, w=0.5, h=0.08] -> chữ nhật dẹt dài
        # Tâm gần nhau: khoảng cách tâm ~ sqrt(0.01^2 + 0.01^2) = 0.014 < 0.15
        pred_boxes = torch.tensor([[[0.50, 0.50, 0.10, 0.10], [0.51, 0.49, 0.50, 0.08]]], dtype=torch.float32)
        # 80 classes, cả 2 query đều dự đoán lớp 1 (person) với điểm cao
        logits = torch.full((1, 2, 80), -10.0, dtype=torch.float32)
        logits[0, 0, 1] = 5.0  # conf ~ 0.99
        logits[0, 1, 1] = 4.0  # conf ~ 0.98

        res = apply_sqd_filter(logits, pred_boxes, score_thresh=0.35, dist_thresh=0.15)
        self.assertEqual(len(res[0]["boxes"]), 1, "SQD-Filter phải nhận diện tâm gần nhau và khử hộp thứ 2")
        self.assertEqual(res[0]["scores"][0], float(torch.sigmoid(torch.tensor(5.0))), "Phải giữ lại hộp có điểm cao nhất")

    def test_jpeg_compression_defense_fidelity(self):
        """Kiểm tra bộ lọc tiền xử lý JPEG: không làm méo hình dạng, chiều kênh hay chuẩn hóa [0, 1]."""
        defense = JPEGCompressionDefense(quality=75)
        dummy_img = torch.rand((2, 3, 64, 64), dtype=torch.float32)

        out_img = defense(dummy_img)
        self.assertEqual(out_img.shape, dummy_img.shape, "Kích thước ảnh sau nén JPEG phải đồng nhất")
        self.assertTrue(out_img.min() >= 0.0 and out_img.max() <= 1.0, "Giá trị pixel phải nằm trong [0.0, 1.0]")
        mse = torch.nn.functional.mse_loss(out_img, dummy_img).item()
        self.assertLess(mse, 0.10, "Độ lệch MSE của JPEG Q=75 trên ảnh chuẩn phải nhỏ (< 0.10)")

    def test_query_representation_guardrail(self):
        """Kiểm tra QueryRepresentationGuardrail: hiệu chuẩn prototype và tính AUROC."""
        guardrail = QueryRepresentationGuardrail(num_classes=80, feature_dim=64)

        # 1. Tạo features sạch cho lớp 1 (person): phân phối chuẩn quanh vector u
        u = torch.randn(64)
        clean_feats = u + torch.randn(50, 64) * 0.10
        clean_labels = torch.full((50,), 1, dtype=torch.long)
        guardrail.calibrate_prototypes(clean_feats, clean_labels)

        # 2. Tạo test query sạch và test query đối kháng (vector ngẫu nhiên hoặc ngược hướng)
        test_clean = u + torch.randn(20, 64) * 0.10
        test_adv = -u + torch.randn(20, 64) * 0.10

        clean_scores = guardrail.compute_anomaly_scores(test_clean, torch.full((20,), 1, dtype=torch.long))
        adv_scores = guardrail.compute_anomaly_scores(test_adv, torch.full((20,), 1, dtype=torch.long))

        # Điểm anomaly của ảnh đối kháng phải cao hơn ảnh sạch rõ rệt
        self.assertLess(clean_scores.mean().item(), 0.30, "Điểm bất thường của query sạch phải thấp (< 0.30)")
        self.assertGreater(adv_scores.mean().item(), 1.50, "Điểm bất thường của query đối kháng phải cao (> 1.50)")

        # 3. Đo AUROC
        perf = evaluate_guardrail_performance(clean_scores.detach().numpy(), adv_scores.detach().numpy())
        self.assertGreater(perf["auroc"], 0.95, "AUROC phát hiện đối kháng phải đạt trên 0.95 trên synthetic benchmark")

    def test_pyramid_distribution_mapping(self):
        """Kiểm tra logic phân loại anchor index sang tầng Feature Pyramid S3, S4, S5."""
        # RT-DETR native 640x640:
        # S3: 80x80 = 6400 anchors (indices 0..6399)
        # S4: 40x40 = 1600 anchors (indices 6400..7999)
        # S5: 20x20 = 400 anchors  (indices 8000..8399)
        sample_indices = [0, 3200, 6399, 6400, 7200, 7999, 8000, 8200, 8399]
        counts = {"S3": 0, "S4": 0, "S5": 0}
        for idx in sample_indices:
            if idx < 6400:
                counts["S3"] += 1
            elif idx < 8000:
                counts["S4"] += 1
            else:
                counts["S5"] += 1

        self.assertEqual(counts["S3"], 3, "Phải có đúng 3 indices thuộc S3")
        self.assertEqual(counts["S4"], 3, "Phải có đúng 3 indices thuộc S4")
        self.assertEqual(counts["S5"], 3, "Phải có đúng 3 indices thuộc S5")


if __name__ == "__main__":
    unittest.main()
