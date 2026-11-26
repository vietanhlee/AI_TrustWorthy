# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.tests.test_refactored_mechanism
Mục đích: Unit test và Integration test toàn diện cho các cải tiến phương pháp
theo nhận xét phản biện của Reviewer (Paper 1 - RT-DETR AQSM Mechanism Breakdown):
  1. AQSM Margin Ranking Loss trong không gian Logit & Hinge-Loss Inactivity.
  2. Bổ sung các biến thể tấn công đối chứng: Memory-Isolated Query & Pure Memory.
  3. Chuẩn hóa Metric: Exact Union Area Null Baseline & Spatial Clustering (ANND, Variance).
  4. Khớp chính xác độ dịch chuyển trong Controlled Misrouting (Post-clipping matching).
  5. Hệ thống phòng thủ: AUROC thực tế (ROC curve) & Spatial-Consistency Monitor.
  6. Cơ chế gia cố SQA: Static Query Augmenter & Ablation forward pass.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from paper_1_aqsm_mechanism.core.coco_gt import (
    compute_image_gt_union_area,
    compute_dataset_gt_union_null_baseline,
)
from paper_1_aqsm_mechanism.core.mechanism_analyzer import (
    compute_spatial_clustering_metrics,
    _cxcywh_to_xyxy_np,
)
from paper_1_aqsm_mechanism.experiments.tasks.task4_controlled_misrouting import (
    sample_exact_matched_shift_4d,
    sample_exact_matched_shift_4d_with_diagnostics,
)
from paper_1_aqsm_mechanism.core.defense_engine import (
    SingleFeatureMonitor,
    EncoderSelectionMonitor,
    SpatialConsistencyMonitor,
)
from paper_1_aqsm_mechanism.core.hardening_prototype import (
    StaticQueryAugmenter,
)
from paper_1_aqsm_mechanism.core.dataset_loader import (
    OnTheFlyAdvImageBatchProxy,
)


class TestAQSMRevampedComponents(unittest.TestCase):
    """Bộ kiểm thử đơn vị cho các thành phần toán học và thuật toán mới cải tiến."""

    def test_01_logit_margin_ranking_loss_hinge_inactivity(self):
        """
        Kiểm tra Margin Ranking Loss trong không gian Logit:
        Loss = ReLU(z_j - z_(300) + gamma_logit).
        Khi thỏa biên margin (z_j <= z_(300) - gamma_logit), số hạng hinge-loss trở nên
        bất hoạt (hinge-loss inactivity) với giá trị 0.0 và gradient bằng 0.0.
        """
        # Giả lập logits của foreground (z_j) và ngưỡng candidate 300 (z_300)
        gamma_logit = 1.0

        # Trường hợp 1: z_j nhỏ hơn z_300 đáng kể (vượt ngưỡng margin an toàn) -> loss=0, grad=0
        z_j_safe = torch.tensor([-2.5, -3.0, -1.8], requires_grad=True)
        z_300 = torch.tensor([-0.5, -0.5, -0.5])
        loss_satisfied = F.relu(z_j_safe - z_300 + gamma_logit).mean()
        self.assertEqual(
            loss_satisfied.item(),
            0.0,
            "Hinge-loss term phải bất hoạt (giá trị 0) khi thỏa mãn margin z_j <= z_300 - gamma",
        )
        loss_satisfied.backward()
        self.assertIsNotNone(z_j_safe.grad)
        self.assertTrue(torch.all(z_j_safe.grad == 0.0))

        # Trường hợp 2: z_j chưa bị dìm (vẫn cao hơn z_300), loss > 0 và có gradient
        z_j_violated = torch.tensor([1.5, 2.0, 0.8], requires_grad=True)
        loss_violated = F.relu(z_j_violated - z_300 + gamma_logit).mean()
        self.assertGreater(loss_violated.item(), 0.0)

        loss_violated.backward()
        self.assertIsNotNone(z_j_violated.grad)
        self.assertTrue(torch.all(z_j_violated.grad > 0.0))

    def test_02b_per_query_center_size_diagnostics_corner_cases(self):
        """
        Kiểm tra khớp đồng thời Center L2 và Size L2 cấp từng query ngay cả gần biên góc [0, 1]^2.
        """
        rng = np.random.RandomState(123)
        for _ in range(100):
            c = rng.uniform(0.01, 0.99, size=4)
            a = rng.uniform(0.01, 0.99, size=4)
            shifted, diag = sample_exact_matched_shift_4d_with_diagnostics(c, a, rng)
            self.assertTrue(np.all(shifted >= 0.0) and np.all(shifted <= 1.0))
            self.assertLess(diag["center_l2_err"], 1e-4)
            self.assertLess(diag["size_l2_err"], 1e-4)

    def test_02_sample_exact_matched_shift_post_clipping(self):
        """
        Kiểm tra độ chính xác của sample_exact_matched_shift_4d:
        Độ dịch chuyển thực tế sau clipping (post-clipping displacement)
        phải khớp với độ dịch chuyển AQSM mục tiêu với sai số cực nhỏ (< 1e-5).
        """
        rng = np.random.RandomState(42)
        n_samples = 150
        max_center_err = 0.0
        max_l2_err = 0.0

        for _ in range(n_samples):
            # Tạo clean box ngẫu nhiên trong (0.1, 0.9)
            clean_box = rng.uniform(0.15, 0.85, size=4)
            # Tạo aqsm box bị dịch chuyển
            shift_delta = rng.uniform(-0.15, 0.15, size=4)
            aqsm_box = np.clip(clean_box + shift_delta, 0.05, 0.95)

            target_center_d = float(np.linalg.norm(aqsm_box[:2] - clean_box[:2]))
            target_l2_d = float(np.linalg.norm(aqsm_box - clean_box))

            # Sinh shifted box đối chứng isotropic
            shifted_box = sample_exact_matched_shift_4d(clean_box, aqsm_box, rng, scale_factor=1.0)

            # Kiểm tra ràng buộc hộp nằm trong canvas [0, 1]
            self.assertTrue(np.all(shifted_box >= 0.0) and np.all(shifted_box <= 1.0))

            realized_center_d = float(np.linalg.norm(shifted_box[:2] - clean_box[:2]))
            realized_l2_d = float(np.linalg.norm(shifted_box - clean_box))

            err_center = abs(realized_center_d - target_center_d)
            err_l2 = abs(realized_l2_d - target_l2_d)

            max_center_err = max(max_center_err, err_center)
            max_l2_err = max(max_l2_err, err_l2)

        self.assertLess(max_center_err, 1e-5, f"Sai số dịch chuyển Center quá lớn: {max_center_err}")
        self.assertLess(max_l2_err, 1e-5, f"Sai số dịch chuyển L2 quá lớn: {max_l2_err}")

    def test_03_exact_gt_union_area_and_null_baseline(self):
        """
        Kiểm tra tính toán diện tích hợp hình học (Exact Union Area)
        và phân phối thực nghiệm của Null Baseline.
        """
        # Hộp rỗng
        empty_area = compute_image_gt_union_area(np.zeros((0, 4)), resolution=640)
        self.assertEqual(empty_area, 0.0)

        # Hai hộp hoàn toàn tách biệt
        box1 = np.array([0.1, 0.1, 0.3, 0.3])  # w=0.2, h=0.2, area=0.04
        box2 = np.array([0.6, 0.6, 0.8, 0.8])  # w=0.2, h=0.2, area=0.04
        boxes_disjoint = np.stack([box1, box2], axis=0)
        disjoint_area = compute_image_gt_union_area(boxes_disjoint, resolution=640)
        self.assertAlmostEqual(disjoint_area, 0.08, places=3)

        # Hai hộp trùng nhau hoàn toàn
        boxes_identical = np.stack([box1, box1], axis=0)
        identical_area = compute_image_gt_union_area(boxes_identical, resolution=640)
        self.assertAlmostEqual(identical_area, 0.04, places=3)

        # Hai hộp giao nhau một phần: box1 [0.1, 0.1, 0.3, 0.3], box3 [0.2, 0.1, 0.4, 0.3]
        # union = 0.04 + 0.04 - 0.1*0.2 = 0.08 - 0.02 = 0.06
        box3 = np.array([0.2, 0.1, 0.4, 0.3])
        boxes_overlap = np.stack([box1, box3], axis=0)
        overlap_area = compute_image_gt_union_area(boxes_overlap, resolution=640)
        self.assertAlmostEqual(overlap_area, 0.06, places=3)

        # Kiểm tra compute_dataset_gt_union_null_baseline
        dummy_cohort = [
            {"boxes_xyxy": boxes_disjoint, "iscrowd": np.zeros(2, dtype=bool)},
            {"boxes_xyxy": boxes_identical, "iscrowd": np.zeros(2, dtype=bool)},
            {"boxes_xyxy": boxes_overlap, "iscrowd": np.zeros(2, dtype=bool)},
        ]
        baseline = compute_dataset_gt_union_null_baseline(dummy_cohort, resolution=640)
        self.assertIn("mean_pct", baseline)
        self.assertIn("std_pct", baseline)
        self.assertIn("median_pct", baseline)
        self.assertIn("ci95_pct", baseline)
        self.assertGreater(baseline["mean_pct"], 0.0)

    def test_04_spatial_clustering_annd_and_dispersion(self):
        """
        Kiểm tra các chỉ số gom cụm không gian ANND và Spatial Variance.
        """
        # Phân bố gom cụm cao: 300 điểm bị dồn cục tại góc (0.1, 0.1)
        rng = np.random.RandomState(42)
        clustered_centers = rng.normal(loc=0.1, scale=0.01, size=(300, 2))
        m_clustered = compute_spatial_clustering_metrics(clustered_centers)

        # Phân bố đều: 300 điểm trải đều khắp canvas [0.1, 0.9]
        dispersed_centers = rng.uniform(low=0.1, high=0.9, size=(300, 2))
        m_dispersed = compute_spatial_clustering_metrics(dispersed_centers)

        # ANND của tập gom cụm phải nhỏ hơn nhiều so với tập phân tán
        self.assertLess(m_clustered["annd"], m_dispersed["annd"] / 2.0)
        # Spatial variance của tập gom cụm phải nhỏ hơn nhiều
        self.assertLess(m_clustered["spatial_variance"], m_dispersed["spatial_variance"] / 5.0)

    def test_05_sqa_static_query_augmenter_strategies(self):
        """
        Kiểm tra StaticQueryAugmenter với các chiến lược khởi tạo khác nhau
        (uniform_grid, kmeans, random) và kích thước mở rộng queries (300 + N_static).
        """
        for strategy in ["uniform_grid", "kmeans", "random"]:
            for n_static in [10, 20, 30]:
                sqa = StaticQueryAugmenter(num_static_anchors=n_static, strategy=strategy, hidden_dim=256)
                self.assertEqual(sqa.static_reference_boxes.shape, (n_static, 4))
                self.assertEqual(sqa.static_content_embeddings.shape, (n_static, 256))

                # Giả lập 300 dynamic queries từ batch size 2
                dynamic_content = torch.randn(2, 300, 256)
                dynamic_boxes = torch.rand(2, 300, 4)

                aug_content, aug_boxes = sqa.augment(dynamic_content, dynamic_boxes)
                self.assertEqual(aug_content.shape, (2, 300 + n_static, 256))
                self.assertEqual(aug_boxes.shape, (2, 300 + n_static, 4))

                # Kiểm tra 300 queries đầu tiên được giữ nguyên vẹn
                self.assertTrue(torch.allclose(aug_content[:, :300, :], dynamic_content))
                self.assertTrue(torch.allclose(aug_boxes[:, :300, :], dynamic_boxes))

    def test_06_defense_engine_auroc_and_spatial_consistency(self):
        """
        Kiểm tra EncoderSelectionMonitor và SpatialConsistencyMonitor:
        - Tính toán AUROC thực tế qua toàn bộ score curve thay vì gán cứng 0.5000.
        - SpatialConsistencyMonitor phát hiện drift và dispersion hiệu quả.
        """
        # Giả lập logits phân lớp cho calibration và test
        rng = np.random.RandomState(42)
        n_cal = 40
        n_test = 30
        S = 300

        # Clean logits: s_1 cao (dropoff lớn, margin tự nhiên)
        # S = 350 tokens (để có cả candidate 300 và 301)
        S = 350
        clean_logits = torch.full((n_cal + n_test, S, 1), -5.0)
        # Giả lập phân bố clean tự nhiên: các tokens đầu có score cao, giảm dần về nền
        for i in range(10):
            clean_logits[:, i, 0] = torch.from_numpy(3.0 + np.random.uniform(0.0, 1.0, size=(n_cal + n_test,))).float()
        clean_logits[:, 299, 0] = -3.0
        clean_logits[:, 300, 0] = -4.5

        cal_list = [clean_logits[:n_cal]]
        esm = EncoderSelectionMonitor(target_fpr=0.05)
        esm.fit(cal_list)
        self.assertTrue(esm.is_fitted)

        test_clean = clean_logits[n_cal:]
        scores_clean = esm.score(test_clean)

        # Adv logits: phân phối bị bẹt hoàn toàn (s_1 bị dìm, s_300 bị đẩy lên xấp xỉ nhau)
        adv_logits = torch.full((n_test, S, 1), -1.5)
        scores_adv = esm.score(adv_logits)

        # Tính AUROC qua sklearn
        import sklearn.metrics as skm
        labels = np.concatenate([np.zeros(n_test), np.ones(n_test)])
        all_scores = np.concatenate([scores_clean, scores_adv])
        auroc = float(skm.roc_auc_score(labels, all_scores))
        self.assertGreater(auroc, 0.85, f"AUROC tính toán thực tế phải cao trên mẫu giả lập phân tách: {auroc}")

        # Kiểm tra SpatialConsistencyMonitor
        scm = SpatialConsistencyMonitor(target_fpr=0.05)
        clean_refs = [torch.rand(n_cal, 300, 4)]
        clean_preds = [clean_refs[0] + torch.randn(n_cal, 300, 4) * 0.01]  # ít drift
        scm.fit(clean_refs, clean_preds)
        self.assertTrue(scm.is_fitted)

        test_refs = torch.rand(n_test, 300, 4)
        test_clean_preds = test_refs + torch.randn(n_test, 300, 4) * 0.01
        test_adv_preds = test_refs + 0.35  # drift lớn

        scm_clean_sc = scm.score(test_refs, test_clean_preds)
        scm_adv_sc = scm.score(test_refs, test_adv_preds)
        scm_auroc = float(skm.roc_auc_score(labels, np.concatenate([scm_clean_sc, scm_adv_sc])))
        self.assertGreater(scm_auroc, 0.80)

    def test_07_onthefly_adv_proxy_routing(self):
        """
        Kiểm tra định tuyến của OnTheFlyAdvImageBatchProxy cho các đòn tấn công mới:
        - memory_isolated_query (aqsm_mem_constrained)
        - pure_memory (pure_memory_attack)
        """
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 80, kernel_size=1)

            def forward(self, pixel_values, return_dict=True):
                feat = self.conv(pixel_values).mean(dim=[-1, -2])  # [B, 80]
                logits = feat.unsqueeze(1).expand(-1, 300, -1)
                enc_h = pixel_values.mean(dim=1, keepdim=True).expand(-1, 256, -1, -1)

                class DummyOut:
                    pass

                out = DummyOut()
                out.logits = logits
                out.enc_outputs_class = logits
                out.enc_outputs_coord_logits = logits[..., :4]
                out.pred_boxes = torch.sigmoid(logits[..., :4])
                out.encoder_last_hidden_state = [enc_h]
                return out

        class DummyWrapper:
            def __init__(self, model):
                self.model = model
                self.device = torch.device("cpu")
                self.is_sigmoid = True

        dummy_wrapper = DummyWrapper(DummyModel())
        clean_imgs = torch.rand(2, 3, 32, 32)
        proxy_mem_iso = OnTheFlyAdvImageBatchProxy(clean_imgs, attack_name="memory_isolated_query", wrapper=dummy_wrapper, steps=2)
        proxy_pure_mem = OnTheFlyAdvImageBatchProxy(clean_imgs, attack_name="pure_memory", wrapper=dummy_wrapper, steps=2)

        # Kiểm tra slice proxy sinh ảnh thành công
        adv_mem_iso = proxy_mem_iso[0:1]
        self.assertEqual(adv_mem_iso.shape, (1, 3, 32, 32))

        adv_pure_mem = proxy_pure_mem[0:1]
        self.assertEqual(adv_pure_mem.shape, (1, 3, 32, 32))

    def test_08_dynamic_table_generation_provenance(self):
        """
        Kiểm tra toàn bộ 12 hàm sinh bảng trong generate_publication_tables:
        - Đọc động 100% từ summary_json/, không lỗi cú pháp hoặc kiểu dữ liệu.
        - Trả về mã LaTeX hợp lệ, Markdown và cấu trúc hàng CSV đầy đủ.
        """
        from paper_1_aqsm_mechanism.tools.generate_publication_tables import (
            generate_table1, generate_table2, generate_table3, generate_table4,
            generate_table5, generate_table6, generate_table7, generate_table8,
            generate_table9, generate_table10, generate_table11, generate_table12,
        )
        summary_dir = Path(WORKSPACE_DIR) / "paper_1_aqsm_mechanism" / "outputs" / "summary_json"
        if not summary_dir.exists() or not (summary_dir / "causal_swap_results_aqsm.json").exists():
            self.skipTest("Bỏ qua test kiểm bảng từ summary_json khi outputs/ đã được dọn sạch chờ chạy mới.")

        table_funcs = [
            generate_table1, generate_table2, generate_table3, generate_table4,
            generate_table5, generate_table6, generate_table7, generate_table8,
            generate_table9, generate_table10, generate_table11, generate_table12,
        ]

        for i, fn in enumerate(table_funcs, 1):
            tex, md, headers, rows = fn(summary_dir)
            self.assertTrue(len(tex) > 50, f"Table {i} tex snippet không được rỗng")
            self.assertTrue(len(headers) >= 2, f"Table {i} headers phải có ít nhất 2 cột")
            self.assertTrue(len(rows) >= 1, f"Table {i} rows không được rỗng")
            self.assertIn(r"\begin{table}", tex, f"Table {i} phải chứa \\begin{{table}}")
            self.assertIn(r"\end{table}", tex, f"Table {i} phải chứa \\end{{table}}")

    def test_09_dynamic_figure_generation_parsing(self):
        """
        Kiểm tra khả năng phân tích dữ liệu động của task7_generate_figures:
        - Đảm bảo load_json và cấu trúc trường số liệu vẽ hình không bị lỗi.
        """
        from paper_1_aqsm_mechanism.utils.io_helpers import load_json
        summary_dir = Path(WORKSPACE_DIR) / "paper_1_aqsm_mechanism" / "outputs" / "summary_json"
        if not summary_dir.exists() or not (summary_dir / "causal_swap_results_aqsm.json").exists():
            self.skipTest("Bỏ qua test phân tích biểu đồ khi outputs/ đã được dọn sạch chờ chạy mới.")

        # Kiểm tra đọc dữ liệu Fig 3
        eqd = load_json(summary_dir / "encoder_query_dynamics.json")
        spec = eqd["results"]["aqsm"]["score_spectrum"]
        self.assertIn("clean_s1", spec)
        self.assertIn("adv_s1", spec)

        # Kiểm tra đọc dữ liệu Fig 4
        c_aqsm = load_json(summary_dir / "causal_swap_results_aqsm.json")
        self.assertIn("point_maps", c_aqsm)
        self.assertIn("Arm_A", c_aqsm["point_maps"])
        self.assertIn("Arm_B", c_aqsm["point_maps"])


if __name__ == "__main__":
    unittest.main()
