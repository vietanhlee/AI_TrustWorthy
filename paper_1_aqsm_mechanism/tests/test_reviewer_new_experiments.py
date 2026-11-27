# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.tests.test_reviewer_new_experiments
Mục đích: Kiểm thử toàn diện các thực nghiệm bổ sung theo phản biện chuyên sâu:
  1. AQSM component ablation weights (lambda_suppr, lambda_rank, lambda_elev) & Encoder Attention Attack [27].
  2. StaticQueryAugmenter sửa lỗi inverse-sigmoid (`logit_calibrated`) + nội suy content từ Encoder (`encoder_sample`).
  3. Pooled COCO Bootstrap 95% CI cấp từng ảnh (`compute_pooled_coco_bootstrap_ci`).
  4. Deformable Telemetry tách biệt rank-paired Δ_p vs permutation-invariant Δ_Hungarian & Entropy TOST.
  5. Interleaved Multi-Image Latency Profiling (Task 6) & CNN+NMS baseline guard.
  6. White-Box Adaptive Evasion thật cho SCM, ActiveBoxCountMonitor (BCM) & Triple-Channel Ensemble (Task 8).
  7. Task 11 End-to-End (Tripartite Loss Ablation + Gaussian Control + Attention Baseline [27]) + Table 1 parser.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from paper_1_aqsm_mechanism.experiments import config as cfg
from paper_1_aqsm_mechanism.tests.test_bugfixes_and_flows import _DummyWrapper


class _DummyAttentionWrapper(_DummyWrapper):
    """Mở rộng _DummyWrapper để trả về thêm encoder_attentions khi `output_attentions=True`."""

    def __init__(self, num_candidates: int = 32, num_queries: int = 10, num_classes: int = 5):
        super().__init__(num_candidates=num_candidates, num_queries=num_queries, num_classes=num_classes)
        orig_model = self.model

        class _WithAttn(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner
                self.config = inner.config

            def forward(self, pixel_values, pixel_mask=None, output_attentions=False, output_hidden_states=False, labels=None, return_dict=True):
                out = self.inner(pixel_values, pixel_mask=pixel_mask, output_hidden_states=output_hidden_states, return_dict=return_dict)
                if output_attentions:
                    bsz = pixel_values.shape[0]
                    # Tạo attention weights có gradient qua feat
                    scores = pixel_values.mean(dim=1).flatten(1)[:, :8]
                    attn = torch.softmax(scores, dim=-1).view(bsz, 1, 1, -1).expand(-1, 2, 4, -1)
                    out.encoder_attentions = (attn,)
                return out

        self.model = _WithAttn(orig_model)


class TestReviewerNewExperiments(unittest.TestCase):
    def test_01_aqsm_component_weights_and_encoder_attention_attack(self):
        from paper_1_aqsm_mechanism.core.attack_generator import DETRAttackGenerator

        wrap = _DummyAttentionWrapper(num_candidates=32, num_queries=10, num_classes=5)
        gen = DETRAttackGenerator(wrap, epsilon=16.0 / 255.0, alpha=4.0 / 255.0, num_steps=2)
        x = torch.rand(2, 3, 16, 16)

        # Thử từng cấu hình lambda độc lập
        adv_suppr = gen.generate_query_selection_attack(x, steps=2, lambda_suppr=1.0, lambda_rank=0.0, lambda_elev=0.0, bg_seed=42)
        adv_elev = gen.generate_query_selection_attack(x, steps=2, lambda_suppr=0.0, lambda_rank=0.0, lambda_elev=0.5, bg_seed=42)
        self.assertEqual(adv_suppr.shape, x.shape)
        self.assertLessEqual(float(torch.max(torch.abs(adv_suppr - x))), 16.0 / 255.0 + 1e-6)
        self.assertEqual(adv_elev.shape, x.shape)

        # Thử Encoder Self-Attention Attack [27] có encoder_attentions và fallback không có encoder_attentions
        adv_attn = gen.generate_encoder_attention_attack(x, steps=2)
        self.assertEqual(adv_attn.shape, x.shape)
        self.assertLessEqual(float(torch.max(torch.abs(adv_attn - x))), 16.0 / 255.0 + 1e-6)

        wrap_no_attn = _DummyWrapper(num_candidates=32, num_queries=10, num_classes=5)
        gen_fb = DETRAttackGenerator(wrap_no_attn, epsilon=16.0 / 255.0, alpha=4.0 / 255.0, num_steps=2)
        adv_fb = gen_fb.generate_encoder_attention_attack(x, steps=2)
        self.assertEqual(adv_fb.shape, x.shape)

        # Gaussian noise với explicit sigma & seed
        noise_1 = gen.generate_random_noise(x, noise_type="gaussian", sigma=16.0 / 255.0, seed=42)
        noise_2 = gen.generate_random_noise(x, noise_type="gaussian", sigma=16.0 / 255.0, seed=42)
        self.assertTrue(torch.allclose(noise_1, noise_2))
        self.assertLessEqual(float(torch.max(torch.abs(noise_1 - x))), 16.0 / 255.0 + 1e-6)

    def test_02_sqa_logit_calibration_and_encoder_content_sampling(self):
        from paper_1_aqsm_mechanism.core.hardening_prototype import StaticQueryAugmenter

        sqa_cal = StaticQueryAugmenter(
            num_static_anchors=10,
            strategy="uniform_grid",
            hidden_dim=16,
            coordinate_space="logit_calibrated",
            content_mode="encoder_sample",
        )
        dyn_c = torch.randn(2, 20, 16)
        dyn_b = torch.randn(2, 20, 4)
        out_mem = torch.randn(2, 32, 16, requires_grad=True)
        anc = torch.randn(1, 32, 4)

        aug_c, aug_b = sqa_cal.augment(dyn_c, dyn_b, output_memory=out_mem, encoder_anchors=anc)
        self.assertEqual(aug_c.shape, (2, 30, 16))
        self.assertEqual(aug_b.shape, (2, 30, 4))

        # Kiểm chứng sau khi đi qua sigmoid(), các static reference boxes phục hồi chuẩn [0.12, 0.88] thay vì co về [0.53, 0.71]
        recovered_static = torch.sigmoid(aug_b[:, 20:, :])
        self.assertTrue(torch.allclose(recovered_static[0], sqa_cal.static_reference_boxes, atol=1e-4))
        self.assertLess(float(recovered_static[..., 0].min()), 0.15)
        self.assertGreater(float(recovered_static[..., 0].max()), 0.85)

        # Đảm bảo gradient truyền được về encoder output_memory khi content_mode='encoder_sample'
        aug_c[:, 20:, :].sum().backward()
        self.assertIsNotNone(out_mem.grad)
        self.assertGreater(float(out_mem.grad.abs().sum()), 0.0)

        # So sánh chế độ raw_uncalibrated giữ nguyên giá trị [0,1] chưa nghịch đảo sigmoid
        sqa_raw = StaticQueryAugmenter(num_static_anchors=10, coordinate_space="raw_uncalibrated", hidden_dim=16)
        _, raw_b = sqa_raw.augment(dyn_c, dyn_b)
        self.assertTrue(torch.allclose(raw_b[0, 20:, :], sqa_raw.static_reference_boxes))

    def test_03_pooled_coco_bootstrap_ci(self):
        from paper_1_aqsm_mechanism.core.coco_gt import compute_pooled_coco_bootstrap_ci

        gts = [
            {"boxes_xyxy": np.array([[0.1, 0.1, 0.4, 0.4]]), "labels": np.array([0]), "iscrowd": np.array([False])},
            {"boxes_xyxy": np.array([[0.5, 0.5, 0.8, 0.8]]), "labels": np.array([0]), "iscrowd": np.array([False])},
            {"boxes_xyxy": np.array([[0.2, 0.2, 0.6, 0.6]]), "labels": np.array([1]), "iscrowd": np.array([False])},
        ]
        preds = [
            {"boxes": np.array([[0.1, 0.1, 0.4, 0.4]]), "scores": np.array([0.95]), "labels": np.array([0])},
            {"boxes": np.array([[0.0, 0.0, 0.1, 0.1]]), "scores": np.array([0.90]), "labels": np.array([0])},  # Miss
            {"boxes": np.array([[0.2, 0.2, 0.6, 0.6]]), "scores": np.array([0.92]), "labels": np.array([1])},
        ]
        ci = compute_pooled_coco_bootstrap_ci(preds, gts, n_boot=50, seed=42)
        self.assertEqual(ci["n_images"], 3)
        self.assertLessEqual(ci["mAP@50_ci95"][0], ci["mAP@50"])
        self.assertLessEqual(ci["mAP@50"], ci["mAP@50_ci95"][1])

    def test_04_hungarian_vs_rankpaired_drift_math(self):
        from paper_1_aqsm_mechanism.experiments.telemetry_analyzer import (
            compute_attention_distribution_details,
            compute_hungarian_reference_drift,
            compute_reference_coordinate_drift,
        )

        clean_pts = np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]])
        permuted_pts = np.array([[0.9, 0.9], [0.1, 0.1], [0.5, 0.5]])
        # Hoán vị thuần làm Δ_p lớn nhưng Δ_Hungarian = 0
        self.assertGreater(compute_reference_coordinate_drift(clean_pts, permuted_pts), 0.5)
        self.assertAlmostEqual(compute_hungarian_reference_drift(clean_pts, permuted_pts), 0.0, places=6)

        shifted_pts = clean_pts + 0.2
        self.assertGreater(compute_hungarian_reference_drift(clean_pts, shifted_pts), 0.15)

        # Kiểm tra head-level deformable attention stats
        w = np.full((10, 8, 3, 4), 1.0 / 12.0, dtype=np.float32)
        det = compute_attention_distribution_details(w)
        self.assertEqual(det["head_entropies"].shape, (8,))
        self.assertEqual(det["marginal_weights"].shape, (8, 12))
        self.assertAlmostEqual(float(det["scale_mass"].sum()), 1.0, places=4)

    def test_05_adaptive_scm_bcm_and_ensemble_evasion_and_task11_end_to_end(self):
        from paper_1_aqsm_mechanism.core.defense_engine import (
            ActiveBoxCountMonitor,
            EncoderSelectionMonitor,
            SpatialConsistencyMonitor,
        )
        from paper_1_aqsm_mechanism.experiments.tasks.task8_defense_evaluation import run_adaptive_monitor_suite_test
        from paper_1_aqsm_mechanism.experiments.tasks.task11_ablation_and_baselines import run_ablation_and_baselines
        from paper_1_aqsm_mechanism.tools.generate_publication_tables import generate_table1

        wrap = _DummyAttentionWrapper(num_candidates=32, num_queries=10, num_classes=5).eval()
        cal_x = torch.rand(6, 3, 16, 16)
        with torch.no_grad():
            out_cal = wrap.model(cal_x)

        esm = EncoderSelectionMonitor(target_fpr=0.1, num_queries=10)
        esm.fit([out_cal.enc_outputs_class])
        scm = SpatialConsistencyMonitor(target_fpr=0.1)
        scm.fit([out_cal.init_reference_points], [out_cal.pred_boxes])
        bcm = ActiveBoxCountMonitor(target_fpr=0.1, conf_threshold=0.35, is_sigmoid=True)
        bcm.fit([out_cal.logits])

        test_x = torch.rand(3, 3, 16, 16)
        gts = [
            {"boxes_xyxy": np.array([[0.1, 0.1, 0.4, 0.4]]), "labels": np.array([0]), "iscrowd": np.array([False]), "image_id": 100},
            {"boxes_xyxy": np.array([[0.3, 0.3, 0.7, 0.7]]), "labels": np.array([1]), "iscrowd": np.array([False]), "image_id": 101},
            {"boxes_xyxy": np.array([[0.2, 0.2, 0.5, 0.5]]), "labels": np.array([2]), "iscrowd": np.array([False]), "image_id": 102},
        ]

        res_scm = run_adaptive_monitor_suite_test(
            wrap, test_x, target_mode="scm", scm_monitor=scm, n_samples=3, lambdas=[1.0], batch_size=2, gt_list=gts, steps=2
        )
        res_bcm = run_adaptive_monitor_suite_test(
            wrap, test_x, target_mode="bcm", box_monitor=bcm, n_samples=3, lambdas=[1.0], batch_size=2, gt_list=gts, steps=2
        )
        res_ens = run_adaptive_monitor_suite_test(
            wrap, test_x, target_mode="ensemble", esm_monitor=esm, scm_monitor=scm, box_monitor=bcm, n_samples=3, lambdas=[1.0], batch_size=2, gt_list=gts, steps=2
        )
        for block in (res_scm, res_bcm, res_ens):
            self.assertIn("lambda_1.0", block)
            self.assertIn("joint_evasion_success_pct", block["lambda_1.0"])
            self.assertIn("post_attack_map50_pct", block["lambda_1.0"])

        # Task 11 End-to-End + đọc bằng generate_table1
        orig_out = cfg.get_outputs_dir()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                cfg.update_experiment_paths(output_dir=tmp)
                sdir = cfg.get_summary_json_dir()
                # Sinh matched_budget_results.json & query_selection_dynamics.json tối thiểu để generate_table1 chạy kèm tripartite_ablation_results.json
                mock_ds = SimpleNamespace(tensors=test_x, gt_list=gts)
                t11 = run_ablation_and_baselines(wrap, mock_ds, batch_size=2, n_boot=20, steps=2, seed=42)
                self.assertTrue((sdir / "tripartite_ablation_results.json").exists())
                self.assertTrue((sdir / "ablation_and_baselines_summary.json").exists())
                self.assertIn("L_suppr", t11["tripartite_loss_ablation"])
                self.assertIn("Full_Tripartite", t11["tripartite_loss_ablation"])
                self.assertIn("memory_isolated_query_control", t11)
                self.assertIn("pure_memory_control", t11)

                trip = json.loads((sdir / "tripartite_ablation_results.json").read_text(encoding="utf-8"))
                (sdir / "matched_budget_results.json").write_text(json.dumps({
                    "n_images": 3,
                    "clean_baseline": {"mAP@50": trip["clean_baseline"]["mAP@50"], "mAP@50:95": trip["clean_baseline"]["mAP@50:95"]},
                    "sweep_results": {"aqsm": {"eps_16": trip["variants"]["Full_Tripartite"]}},
                }), encoding="utf-8")
                (sdir / "query_selection_dynamics.json").write_text(json.dumps({
                    "n_images": 3,
                    "results": {
                        "aqsm": {
                            "top300_overlap_pct": {"mean": trip["variants"]["Full_Tripartite"]["top300_overlap_pct"]},
                            "evicted_top15_fg_pct": {"mean": trip["variants"]["Full_Tripartite"]["evicted_top15_fg_pct"]},
                        }
                    },
                }), encoding="utf-8")
                t1_tex, t1_md, _, t1_rows = generate_table1(sdir)
                self.assertIn("L_suppr + L_rank", t1_md)
                self.assertGreaterEqual(len(t1_rows), 7)
                self.assertIn("tab:tab1_ablation", t1_tex)
        finally:
            cfg.update_experiment_paths(output_dir=orig_out)

    def test_06_interleaved_multi_image_latency_profile_and_cnn_guard(self):
        from paper_1_aqsm_mechanism.experiments.tasks.task6_latency_profiler import (
            _measure_cnn_nms_sponge_baseline,
        )

        # Kiểm tra guard cnn_baseline="none" không tải weights
        x = torch.rand(2, 3, 16, 16)
        guard_res = _measure_cnn_nms_sponge_baseline(x, torch.device("cpu"), cnn_arch="none", n_runs=2, steps=1)
        self.assertEqual(guard_res["status"], "skipped_by_flag")

        # Kiểm tra task registry trong run_all_experiments có Task 11
        from paper_1_aqsm_mechanism.run_all_experiments import _build_task_registry
        args_ns = SimpleNamespace(
            folder_image="img", annotations="ann", output_dir=str(cfg.get_outputs_dir()),
            cache_dir="cache", fig_dir="fig", device="cpu", batch_size=2, threads=1,
            model="r18", max_samples=5, tasks="11", skip_cache_gen=True,
            force_recompute=False, include_sweep=False, run_sqa_ablation=False,
            pgd_steps=2, seeds="42", latency_images=4, cnn_baseline="none", n_boot=20,
        )
        reg = _build_task_registry(Path(WORKSPACE_DIR) / "paper_1_aqsm_mechanism", args_ns, ["--device", "cpu"])
        self.assertIn(11, reg)
        self.assertIn("--n_boot", reg[11]["args_builder"]())
        self.assertIn("--latency_images", reg[6]["args_builder"]())

        # Kiểm tra định tuyến CNN+NMS trong run_multimodel_benchmark
        from paper_1_aqsm_mechanism.tools.run_multimodel_benchmark import _format_arch_display_name, benchmark_single_model
        from unittest.mock import patch
        self.assertEqual(_format_arch_display_name("retinanet_nms"), "RetinaNet-R50-FPN (CNN+NMS)")
        with patch("paper_1_aqsm_mechanism.tools.run_multimodel_benchmark._benchmark_cnn_nms_model", return_value={"model_tag": "retinanet_nms"}) as mock_cnn:
            r_cnn = benchmark_single_model("retinanet_nms", "img", "ann", 5, 2, torch.device("cpu"))
            self.assertTrue(mock_cnn.called)
            self.assertEqual(r_cnn["model_tag"], "retinanet_nms")


if __name__ == "__main__":
    unittest.main()
