# -*- coding: utf-8 -*-
"""
Unit tests kiểm chứng triệt để các lỗi hardcode và luồng nghiệp vụ đã sửa
trong hệ thống Paper 1 (experiments/config, utils/cli, core/*, tasks/*, tools/*, run_all_experiments).
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))


class _DummyRTDETRModel(nn.Module):
    """Mô hình giả lập gọn nhẹ có đầy đủ enc_outputs_class, enc_outputs_coord_logits, logits, pred_boxes, decoder_hidden_states."""

    def __init__(self, num_candidates: int = 32, num_queries: int = 10, num_classes: int = 5, num_layers: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.enc_cls = nn.Linear(8, num_candidates * num_classes)
        self.enc_box = nn.Linear(8, num_candidates * 4)
        self.dec_cls = nn.Linear(8, num_queries * num_classes)
        self.dec_box = nn.Linear(8, num_queries * 4)
        self.S = num_candidates
        self.Nq = num_queries
        self.C = num_classes
        self.L = num_layers
        self.config = SimpleNamespace(
            num_queries=num_queries,
            num_labels=num_classes,
            d_model=8,
            decoder_layers=num_layers,
            id2label={i: f"cls_{i}" for i in range(num_classes)},
        )

    def forward(self, pixel_values, pixel_mask=None, output_hidden_states=False, return_dict=True):
        bsz = pixel_values.shape[0]
        feat = self.conv(pixel_values).mean(dim=(-2, -1))
        enc_cls = self.enc_cls(feat).view(bsz, self.S, self.C)
        enc_box = self.enc_box(feat).view(bsz, self.S, 4)
        dec_cls = self.dec_cls(feat).view(bsz, self.Nq, self.C)
        dec_box = torch.sigmoid(self.dec_box(feat).view(bsz, self.Nq, 4))
        dec_hs = tuple(feat.unsqueeze(1).repeat(1, self.Nq, 1) for _ in range(self.L)) if output_hidden_states else None
        return SimpleNamespace(
            logits=dec_cls,
            pred_boxes=dec_box,
            enc_outputs_class=enc_cls,
            enc_outputs_coord_logits=enc_box,
            encoder_last_hidden_state=[feat.unsqueeze(-1).unsqueeze(-1)],
            decoder_hidden_states=dec_hs,
            init_reference_points=dec_box.clone(),
        )


class _DummyWrapper:
    def __init__(self, num_candidates: int = 32, num_queries: int = 10, num_classes: int = 5, is_sigmoid: bool = True):
        self.device = torch.device("cpu")
        self.model = _DummyRTDETRModel(num_candidates=num_candidates, num_queries=num_queries, num_classes=num_classes)
        self.is_sigmoid = is_sigmoid
        self.num_queries = num_queries
        self.d_model = 8
        self.decoder_layers = 2
        self.pretrained_model_name = "mock/rtdetr_test"

    def eval(self):
        self.model.eval()
        return self


class TestBugfixesAndFlows(unittest.TestCase):
    def test_01_dynamic_config_and_cli_parser(self):
        """Kiểm tra update_experiment_paths cập nhật động parser defaults, NUM_DECODER_LAYERS và NUM_QUERIES."""
        from paper_1_aqsm_mechanism.experiments import config as cfg
        from paper_1_aqsm_mechanism.utils.cli import create_base_parser

        orig_out = cfg.get_outputs_dir()
        orig_model = cfg.get_model_name()
        orig_steps = cfg.get_pgd_num_steps()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                cfg.update_experiment_paths(output_dir=tmp, model="r50", pgd_steps=21)
                self.assertEqual(cfg.get_outputs_dir(), Path(tmp).resolve())
                self.assertEqual(cfg.get_num_decoder_layers(), 6)
                self.assertEqual(cfg.get_pgd_num_steps(), 21)
                self.assertEqual(cfg.infer_num_queries("facebook/detr-resnet-50"), 100)

                parser = create_base_parser("Test dynamic defaults", task_name="Test")
                parsed = parser.parse_args([])
                self.assertEqual(Path(parsed.output_dir), Path(tmp).resolve())
                self.assertEqual(parsed.model, "PekingU/rtdetr_r50vd")
        finally:
            cfg.update_experiment_paths(output_dir=orig_out, model=orig_model, pgd_steps=orig_steps)

    def test_02_coco_91_class_mapping_and_eval(self):
        """Kiểm tra build_coco_category_mapping & evaluate_coco_predictions hỗ trợ cả 80 lớp và 91 lớp HF DETR."""
        from paper_1_aqsm_mechanism.core.coco_gt import build_coco_category_mapping, evaluate_coco_predictions

        cats = []
        id2label_91 = {0: "N/A"}
        # Tạo 80 COCO categories có gap (ví dụ skipped ID 12) tới ID 90
        cid = 1
        while len(cats) < 80:
            if cid in (12, 26, 29, 30, 45, 66, 68, 69, 71, 83):
                id2label_91[cid] = "N/A"
                cid += 1
                continue
            cname = f"obj_{cid}"
            cats.append({"id": cid, "name": cname})
            id2label_91[cid] = cname
            cid += 1
        max_cid = max(c["id"] for c in cats)
        self.assertGreaterEqual(max_cid, 90)

        m91 = build_coco_category_mapping(cats, id2label=id2label_91)
        self.assertEqual(m91[max_cid], max_cid)

        preds = [{"boxes": np.array([[10.0, 10.0, 50.0, 50.0]]), "scores": np.array([0.9]), "labels": np.array([max_cid])}]
        gts = [{"boxes_xyxy": np.array([[10.0, 10.0, 50.0, 50.0]]), "labels": np.array([max_cid]), "iscrowd": np.array([False])}]
        ev = evaluate_coco_predictions(preds, gts)
        self.assertAlmostEqual(ev["mAP@50"], 100.0, places=2)

    def test_03_attack_generator_small_S_batch_gt_and_softmax(self):
        """Kiểm tra AQSM & Mem-Isolated với S < 300, gt_boxes theo từng ảnh b, và adaptive với is_sigmoid=False."""
        from paper_1_aqsm_mechanism.core.attack_generator import DETRAttackGenerator

        wrap = _DummyWrapper(num_candidates=32, num_queries=10, num_classes=5, is_sigmoid=True)
        attacker = DETRAttackGenerator(wrap, epsilon=16.0 / 255.0, alpha=4.0 / 255.0, num_steps=2)
        x = torch.rand(2, 3, 16, 16)
        gt_b = [
            np.array([[0.1, 0.1, 0.4, 0.4]], dtype=np.float32),
            np.array([[0.6, 0.6, 0.9, 0.9]], dtype=np.float32),
        ]

        adv_aqsm = attacker.generate_query_selection_attack(
            x, steps=2, fg_selection_mode="ground_truth_matched", gt_boxes=gt_b, bg_seed=42
        )
        self.assertEqual(adv_aqsm.shape, x.shape)

        adv_mem = attacker.generate_memory_isolated_query_attack(
            x, steps=2, fg_selection_mode="ground_truth_matched", gt_boxes=gt_b, bg_seed=77
        )
        self.assertEqual(adv_mem.shape, x.shape)

        wrap_soft = _DummyWrapper(num_candidates=32, num_queries=10, num_classes=6, is_sigmoid=False)
        attacker_soft = DETRAttackGenerator(wrap_soft, epsilon=16.0 / 255.0, alpha=4.0 / 255.0, num_steps=2)
        adv_adp = attacker_soft.generate_adaptive_attack(x, steps=2)
        adv_fus = attacker_soft.generate_adaptive_fused_attack(x, steps=2)
        self.assertEqual(adv_adp.shape, x.shape)
        self.assertEqual(adv_fus.shape, x.shape)

    def test_04_onthefly_proxy_forwards_hf_labels_and_steps(self):
        """Kiểm tra load_cached_adv_images & OnTheFlyAdvImageBatchProxy chuyển tiếp hf_labels và steps tới Task PGD."""
        from paper_1_aqsm_mechanism.core.adv_cache import load_cached_adv_images
        from paper_1_aqsm_mechanism.core.dataset_loader import LazyImageBatchProxy

        wrap = _DummyWrapper(num_candidates=32, num_queries=10, num_classes=5)
        hf_lbls = [
            {"class_labels": torch.tensor([1]), "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]])},
            {"class_labels": torch.tensor([2]), "boxes": torch.tensor([[0.3, 0.3, 0.2, 0.2]])},
        ]

        class _MockCleanProxy(LazyImageBatchProxy):
            def __init__(self):
                super().__init__([Path("a.jpg"), Path("b.jpg")], img_size=16, device=torch.device("cpu"), hf_labels=hf_lbls)

            def __getitem__(self, item):
                if isinstance(item, slice):
                    n = len(range(*item.indices(len(self.paths))))
                    return torch.rand(n, 3, 16, 16)
                return torch.rand(1, 3, 16, 16)

        clean_p = _MockCleanProxy()
        captured = {}

        def _fake_task_pgd(images, hf_labels=None, steps=None, alpha=None):
            captured["hf_labels"] = hf_labels
            captured["steps"] = steps
            return images.clone()

        with tempfile.TemporaryDirectory() as tmp:
            adv_proxy, meta = load_cached_adv_images(
                attack="taskpgd",
                eps_num=16,
                n_eval=2,
                clean_imgs=clean_p,
                expected_image_ids=[1, 2],
                cache_dir=tmp,
                wrapper=wrap,
                allow_onthefly=True,
                steps=7,
            )
            self.assertEqual(meta["steps"], 7)
            attacker = adv_proxy._get_attacker()
            with patch.object(attacker, "generate_standard_task_pgd", side_effect=_fake_task_pgd):
                _ = adv_proxy[0:2]
            self.assertEqual(captured["steps"], 7)
            self.assertIsNotNone(captured["hf_labels"])
            self.assertEqual(len(captured["hf_labels"]), 2)
            self.assertEqual(int(captured["hf_labels"][1]["class_labels"][0].item()), 2)

    def test_05_defense_esm_small_candidates(self):
        """Kiểm tra EncoderSelectionMonitor không lỗi index khi S < 300."""
        from paper_1_aqsm_mechanism.core.defense_engine import EncoderSelectionMonitor

        cal = [torch.randn(10, 40, 5)]
        esm = EncoderSelectionMonitor(target_fpr=0.05)
        esm.fit(cal)
        self.assertTrue(esm.is_fitted)
        preds = esm.predict(torch.randn(5, 40, 5))
        self.assertEqual(preds.shape, (5,))

    def test_06_sqa_controls_and_tables_no_hardcode(self):
        """Kiểm tra evaluate_sqa_controls (không KeyError) và generate_publication_tables (không số cứng ở Table 4, 10, 11)."""
        from paper_1_aqsm_mechanism.tools import evaluate_sqa_controls, generate_publication_tables

        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            sdir = tmp_p / "summary_json"
            sdir.mkdir(parents=True, exist_ok=True)

            (sdir / "causal_swap_results_aqsm.json").write_text(json.dumps({
                "n_evaluated_images": 50,
                "n_valid_intra_subject_samples": 48,
                "point_maps": {
                    "Arm_A": {"mAP@50": 55.0}, "Arm_B": {"mAP@50": 5.0},
                    "Arm_C_full": {"mAP@50": 35.0}, "Arm_C_index": {"mAP@50": 15.0},
                    "Arm_D_full": {"mAP@50": 25.0}, "Arm_D_index": {"mAP@50": 20.0},
                },
                "anova_repeated_measures_full": {
                    "N": 48,
                    "marginal_effects": {"ME_M": -25.0, "ME_Q": -20.0, "Interaction_MxQ": 5.0, "ME_M_ci95": [-27.0, -23.0], "ME_Q_ci95": [-22.0, -18.0], "Interaction_ci95": [3.0, 7.0]},
                    "attribution_shares": {
                        "partial_eta_squared": {"Memory_M": 0.65, "Query_Q": 0.45, "Interaction_MxQ": 0.12},
                        "anova_condition_ss_share_pct": {"Memory_M": 58.2, "Query_Q": 34.1, "Interaction_MxQ": 7.7},
                    },
                    "anova_table": {
                        "Memory_M": {"SS": 1200.0, "F": 42.0, "p_value": 1e-6},
                        "Query_Q": {"SS": 700.0, "F": 24.0, "p_value": 1e-5},
                        "Interaction_MxQ": {"SS": 150.0, "F": 6.5, "p_value": 0.0123},
                    },
                },
            }), encoding="utf-8")

            (sdir / "hardening_evaluation_results.json").write_text(json.dumps({
                "model": "custom/rtdetr_r50_test",
                "num_dynamic_queries": 300,
                "num_static_anchors": 30,
                "results": {
                    "clean_baseline": {
                        "default_dynamic_only": {"mAP@50": 55.0, "mAP@50:95": 40.0, "recall_pct": 65.0},
                        "hardened_static_augmented": {"mAP@50": 54.8, "mAP@50:95": 39.8, "recall_pct": 64.7, "clean_penalty_pp": -0.2},
                    },
                    "aqsm_attack_eps16": {
                        "default_dynamic_only": {"mAP@50": 5.0, "mAP@50:95": 3.0, "recall_pct": 10.0},
                        "hardened_static_augmented": {"mAP@50": 19.5, "mAP@50:95": 13.5, "recall_pct": 31.0, "recovery_pp": 14.5},
                    },
                    "taskpgd_attack_eps16": {
                        "default_dynamic_only": {"mAP@50": 7.0, "mAP@50:95": 4.0, "recall_pct": 12.0},
                        "hardened_static_augmented": {"mAP@50": 8.0, "mAP@50:95": 4.5, "recall_pct": 13.0},
                    },
                    "controls": {
                        "dynamic_top330": {"mAP@50": 5.2, "mAP@50:95": 3.1, "recall_pct": 10.2, "recovery_pp": 0.2},
                        "random30_static": {"mAP@50": 12.0, "mAP@50:95": 8.0, "recall_pct": 21.0, "recovery_pp": 7.0},
                        "adaptive_evasion_sqa": {"mAP@50": 2.5, "mAP@50:95": 1.2, "recall_pct": 5.0, "bypass_pct": 95.45},
                    },
                    "edge_latency_jetson_orin_nano_ms": {"default_dynamic_only": 11.1, "hardened_static_augmented": 11.5, "overhead_pct": 3.6},
                },
            }), encoding="utf-8")

            # Test evaluate_sqa_controls without KeyError
            with patch.object(sys, "argv", ["evaluate_sqa_controls.py", "--output_dir", str(tmp_p)]):
                evaluate_sqa_controls.main()
            sqa_ctrl_out = json.loads((sdir / "sqa_controls_and_adaptive_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(sqa_ctrl_out["metadata"]["model"], "custom/rtdetr_r50_test")
            self.assertEqual(sqa_ctrl_out["metadata"]["global_benchmark_cohort_n"], 50)
            self.assertEqual(sqa_ctrl_out["causal_memory_bound"]["memory_damage_share_pct"], 58.2)

            # Prepare Table 10 & Table 11 dynamic JSONs
            (sdir / "edge_latency_results.json").write_text(json.dumps({
                "model": "custom/rtdetr_r50_test",
                "n_runs": 25,
                "device_type": "cpu",
                "hardware_metadata": {"processor": "MockCPU"},
                "throughput_fps": 40.0,
                "stages": {
                    "Backbone": {"mean_ms": 5.0, "std_ms": 0.2, "p95_ms": 5.3},
                    "Hybrid_Encoder": {"mean_ms": 8.0, "std_ms": 0.3, "p95_ms": 8.5},
                    "Query_Selection": {"mean_ms": 2.0, "std_ms": 0.1, "p95_ms": 2.2},
                    "Deformable_Decoder": {"mean_ms": 10.0, "std_ms": 0.4, "p95_ms": 10.7},
                    "End_to_End_Total": {"mean_ms": 25.0, "std_ms": 0.8, "p95_ms": 26.5},
                },
                "percentage_breakdown": {"Backbone_pct": 20.0, "Hybrid_Encoder_pct": 32.0, "Query_Selection_pct": 8.0, "Deformable_Decoder_pct": 40.0},
                "adversarial_latency_comparison": {
                    "Clean_Baseline": {"mean_ms": 24.5, "std_ms": 0.5, "p95_ms": 25.2, "throughput_fps": 40.82, "active_boxes_tau35": 4.2},
                    "RT_DETR_Flooding_Sponge": {"mean_ms": 24.6, "std_ms": 0.6, "p95_ms": 25.4, "throughput_fps": 40.65, "active_boxes_tau35": 280.0},
                },
            }), encoding="utf-8")

            (sdir / "matched_budget_sweep.json").write_text(json.dumps({
                "n_images": 50,
                "model": "custom/rtdetr_r50_test",
                "clean_baseline": {"mAP@50": 55.0, "mAP@50:95": 40.0},
                "sweep_results": {
                    "aqsm": {"eps_4": {"mAP@50": 25.0, "mAP@50:95": 17.0, "relative_drop_pct": 54.5}, "eps_8": {"mAP@50": 15.0, "mAP@50:95": 10.0, "relative_drop_pct": 72.7}, "eps_16": {"mAP@50": 5.0, "mAP@50:95": 3.0, "relative_drop_pct": 90.9}},
                    "taskpgd": {"eps_4": {"mAP@50": 28.0, "mAP@50:95": 19.0, "relative_drop_pct": 49.1}, "eps_8": {"mAP@50": 17.0, "mAP@50:95": 11.0, "relative_drop_pct": 69.1}, "eps_16": {"mAP@50": 7.0, "mAP@50:95": 4.0, "relative_drop_pct": 87.3}},
                    "vanishing": {"eps_4": {"mAP@50": 30.0, "mAP@50:95": 20.0, "relative_drop_pct": 45.5}, "eps_8": {"mAP@50": 18.0, "mAP@50:95": 12.0, "relative_drop_pct": 67.3}, "eps_16": {"mAP@50": 6.0, "mAP@50:95": 3.5, "relative_drop_pct": 89.1}},
                },
            }), encoding="utf-8")

            (sdir / "defense_esm_evaluation.json").write_text(json.dumps({
                "target_fpr_pct": 5.0,
                "empirical_clean_fpr_pct": 4.2,
                "clean_mean_anomaly_score": 0.3125,
                "union_bound_pct": 9.1,
                "n_cal": 15,
                "n_test": 35,
                "attacks": {
                    "aqsm": {"tpr_detection_rate_pct": 97.1, "auroc": 0.9912, "mean_anomaly_score": 4.1234},
                    "taskpgd": {"tpr_detection_rate_pct": 91.4, "auroc": 0.9650, "mean_anomaly_score": 3.4567},
                    "vanishing": {"tpr_detection_rate_pct": 99.0, "auroc": 0.9980, "mean_anomaly_score": 5.4321},
                },
                "adaptive_evasion": {
                    "lambda_0.2": {"detected_tpr_pct": 2.0, "raw_monitor_bypass_pct": 98.0, "post_attack_map50_pct": 2.11},
                    "lambda_0.5": {"detected_tpr_pct": 1.0, "raw_monitor_bypass_pct": 99.0, "post_attack_map50_pct": 2.75},
                    "lambda_1.0": {"detected_tpr_pct": 0.0, "raw_monitor_bypass_pct": 100.0, "post_attack_map50_pct": 3.44},
                    "lambda_2.0": {"detected_tpr_pct": 0.0, "raw_monitor_bypass_pct": 100.0, "post_attack_map50_pct": 4.88},
                },
            }), encoding="utf-8")

            tex10, md10, _, _ = generate_publication_tables.generate_table10(sdir)
            self.assertIn("24.500", md10)
            self.assertIn("280.00", md10)
            self.assertNotIn("YOLOv8-S", tex10)
            self.assertNotIn("14.20", tex10)

            tex11, md11, _, _ = generate_publication_tables.generate_table11(sdir)
            self.assertIn("0.3125", md11)
            self.assertIn("2.11%", md11)
            self.assertIn("4.88%", md11)
            self.assertNotIn("0.4950", tex11)
            self.assertNotIn("3.65", tex11)
            self.assertIn("N_{\\text{test}} = 35", tex11)

            # Kiểm tra Table 12 động không hardcode 300/30/R18 nếu JSON khác
            tex12, md12, _, _ = generate_publication_tables.generate_table12(sdir)
            self.assertIn("rtdetr_r50_test", md12)
            self.assertIn("+14.50 pp", md12)

    def test_07_provenance_manifest_handles_missing_r50_and_dynamic_cohort(self):
        """Kiểm tra generate_provenance_manifest dùng n_cal/n_test động và không liệt kê file R50 nếu không tồn tại."""
        from paper_1_aqsm_mechanism.tools.generate_data_provenance_manifest import generate_provenance_manifest

        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            sdir = tmp_p / "summary_json"
            sdir.mkdir(parents=True, exist_ok=True)
            (sdir / "causal_swap_results_aqsm.json").write_text(json.dumps({
                "n_evaluated_images": 40,
                "n_valid_intra_subject_samples": 38,
                "point_maps": {
                    "Arm_A": {"mAP@50": 50.0}, "Arm_B": {"mAP@50": 5.0},
                    "Arm_C_full": {"mAP@50": 30.0}, "Arm_C_index": {"mAP@50": 15.0},
                    "Arm_D_full": {"mAP@50": 20.0}, "Arm_D_index": {"mAP@50": 18.0},
                },
                "anova_repeated_measures_full": {
                    "N": 38,
                    "marginal_effects": {"ME_M": -20.0, "ME_Q": -15.0, "Interaction_MxQ": 4.0},
                    "attribution_shares": {"anova_condition_ss_share_pct": {"Memory_M": 55.0, "Query_Q": 35.0}},
                },
            }), encoding="utf-8")
            (sdir / "matched_budget_results.json").write_text(json.dumps({
                "clean_baseline": {"mAP@50": 50.0, "mAP@50:95": 35.0},
                "sweep_results": {
                    "aqsm": {"eps_4": {"mAP@50": 20.0}, "eps_8": {"mAP@50": 10.0}, "eps_16": {"mAP@50": 5.0, "mAP@50:95": 3.0}},
                    "vanishing": {"eps_16": {"mAP@50": 4.0}},
                    "taskpgd": {"eps_4": {"mAP@50": 22.0}, "eps_8": {"mAP@50": 12.0}, "eps_16": {"mAP@50": 6.0}},
                },
            }), encoding="utf-8")
            (sdir / "encoder_memory_analysis.json").write_text(json.dumps({
                "results": {
                    "aqsm": {
                        "cosine_similarity": {"Overall": {"mean": 0.96}, "S3": {"mean": 0.95}, "S4": {"mean": 0.97}, "S5": {"mean": 0.98}},
                        "relative_l2_perturbation": {"Foreground": {"mean": 0.25}, "Background": {"mean": 0.20}},
                    }
                }
            }), encoding="utf-8")
            (sdir / "encoder_query_dynamics.json").write_text(json.dumps({
                "results": {
                    "aqsm": {
                        "top300_overlap_pct": {"mean": 18.0},
                        "evicted_top15_fg_pct": {"mean": 88.0},
                        "object_coverage": {"center_in_gt_adv_pct": {"mean": 75.0}, "center_in_gt_clean_pct": {"mean": 96.0}},
                        "spearman_rho_all_proposals": {"mean": 0.42},
                        "kendall_tau_all_proposals": {"mean": 0.30},
                    }
                }
            }), encoding="utf-8")
            (sdir / "deformable_attention_telemetry.json").write_text(json.dumps({
                "clean": {"mean_entropy_l1": 0.78, "mean_entropy_all_layers": 0.78, "mean_fg_hit_rate": 0.62, "mean_dropoff_ratio": 35.0},
                "aqsm": {"mean_entropy_l1": 0.79, "mean_entropy_all_layers": 0.79, "mean_fg_hit_rate": 0.41, "mean_dropoff_ratio": 1.25},
                "vanishing": {"mean_entropy_l1": 0.77, "mean_entropy_all_layers": 0.77, "mean_fg_hit_rate": 0.48, "mean_dropoff_ratio": 8.5},
            }), encoding="utf-8")
            (sdir / "controlled_misrouting_results.json").write_text(json.dumps({
                "table8_summary": {
                    "Clean_Baseline": {"mAP@50": 50.0},
                    "Random_Reassignment": {"mAP@50_mean": 46.0},
                    "Matched_L2_Random_Shift": {"mAP@50_mean": 31.0},
                    "Decoupled_Random_Shift": {"mAP@50_mean": 29.5},
                    "AQSM_Spatial_Misrouting": {"mAP@50": 20.0},
                }
            }), encoding="utf-8")
            (sdir / "edge_latency_results.json").write_text(json.dumps({
                "device_type": "cpu",
                "throughput_fps": 20.0,
                "stages": {
                    "Backbone": {"mean_ms": 10.0},
                    "Hybrid_Encoder": {"mean_ms": 15.0},
                    "Query_Selection": {"mean_ms": 5.0},
                    "Deformable_Decoder": {"mean_ms": 20.0},
                    "End_to_End_Total": {"mean_ms": 50.0},
                },
            }), encoding="utf-8")
            (sdir / "defense_esm_evaluation.json").write_text(json.dumps({
                "n_cal": 12,
                "n_test": 28,
                "target_fpr_pct": 5.0,
                "empirical_clean_fpr_pct": 3.57,
                "attacks": {"aqsm": {"tpr_detection_rate_pct": 96.4, "auroc": 0.99}},
                "adaptive_evasion": {"lambda_0.5": {"detected_tpr_pct": 0.0}, "lambda_1.0": {"detected_tpr_pct": 0.0}},
            }), encoding="utf-8")
            (sdir / "hardening_evaluation_results.json").write_text(json.dumps({
                "model": "PekingU/rtdetr_r18vd",
                "results": {
                    "clean_baseline": {"default_dynamic_only": {"mAP@50": 50.0}, "hardened_static_augmented": {"mAP@50": 49.8}},
                    "aqsm_attack_eps16": {"default_dynamic_only": {"mAP@50": 5.0}, "hardened_static_augmented": {"mAP@50": 18.0}},
                    "taskpgd_attack_eps16": {"default_dynamic_only": {"mAP@50": 7.0}, "hardened_static_augmented": {"mAP@50": 7.5}},
                    "edge_latency_jetson_orin_nano_ms": {"default_dynamic_only": 14.0, "hardened_static_augmented": 14.4},
                },
            }), encoding="utf-8")
            (sdir / "multimodel_and_seed_stability_summary.json").write_text(json.dumps({
                "records": {"rtdetr_r18vd_multiseed": {"aqsm_map50_mean": 5.1, "aqsm_map50_std": 0.2, "hungarian_set_drift_mean": 0.2, "anchor_fg_hit_pct_mean": 21.0}}
            }), encoding="utf-8")

            manifest = generate_provenance_manifest(sdir, tmp_p / "data_provenance_manifest.json")
            self.assertEqual(manifest["cohort_definitions"]["defense_calibration_cohort"]["n_calibration"], 12)
            self.assertEqual(manifest["cohort_definitions"]["defense_calibration_cohort"]["n_test"], 28)
            self.assertEqual(manifest["tables"]["tab:tab11_defense_results"]["n_test_images"], 28)
            t13_arts = manifest["tables"]["tab:tab13_multiseed_generalization"]["source_artifacts"]
            self.assertEqual(t13_arts, ["outputs/summary_json/multimodel_and_seed_stability_summary.json"])

    def test_config_img_size_and_uniform_grid_100_queries(self):
        from paper_1_aqsm_mechanism.experiments.config import get_img_size, update_experiment_paths
        from paper_1_aqsm_mechanism.core.coco_gt import compute_image_uniform_grid_fg_hit
        from paper_1_aqsm_mechanism.core.hardening_prototype import StaticQueryAugmenter

        old_size = get_img_size()
        try:
            update_experiment_paths(img_size=512)
            self.assertEqual(get_img_size(), 512)
        finally:
            update_experiment_paths(img_size=old_size)

        # Với toàn canvas [0, 1]^2 là foreground, hit rate phải đạt đúng 1.0 dù num_queries=100
        full_box = np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
        self.assertAlmostEqual(compute_image_uniform_grid_fg_hit(full_box, num_queries=100), 1.0, places=6)

        anchors = StaticQueryAugmenter._generate_uniform_grid_anchors(37)
        self.assertEqual(tuple(anchors.shape), (37, 4))
        # Không bị pad bằng [0.5, 0.5, 0.3, 0.3]
        self.assertFalse(torch.allclose(anchors[-1], torch.tensor([0.5, 0.5, 0.3, 0.3])))

    def test_adv_cache_resolution_mismatch_fallback_and_alpha_preserved(self):
        from paper_1_aqsm_mechanism.core.adv_cache import load_cached_adv_proxy
        from paper_1_aqsm_mechanism.core.dataset_loader import LazyImageBatchProxy, OnTheFlyAdvImageBatchProxy

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_p = Path(tmpdir)
            img_path = tmp_p / "000000000001.jpg"
            from PIL import Image
            Image.new("RGB", (64, 64), color=(120, 120, 120)).save(img_path)

            cache_dir = tmp_p / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                cache_dir / "adv_cache_rtdetr_r18vd_aqsm_eps16_n1.npz",
                image_ids=np.array([1], dtype=np.int64),
                delta=np.zeros((1, 3, 32, 32), dtype=np.float32),
            )
            clean_proxy = LazyImageBatchProxy([img_path], img_size=64, device=torch.device("cpu"))
            dummy_model = _DummyRTDETRModel(num_candidates=16, num_queries=4, num_classes=5)
            wrapper = SimpleNamespace(
                model=dummy_model,
                device=torch.device("cpu"),
                num_queries=4,
                is_sigmoid=True,
                pretrained_model_name="PekingU/rtdetr_r18vd",
            )
            adv_proxy, _ = load_cached_adv_proxy(
                attack="aqsm",
                eps_num=16,
                n_eval=1,
                clean_proxy=clean_proxy,
                expected_image_ids=[1],
                cache_dir=cache_dir,
                wrapper=wrapper,
                allow_onthefly=True,
            )
            self.assertIsInstance(adv_proxy, OnTheFlyAdvImageBatchProxy)
            moved = adv_proxy.to(torch.device("cpu"))
            self.assertEqual(moved.alpha, adv_proxy.alpha)

    def test_adaptive_calibrator_model_tag_and_softmax(self):
        from paper_1_aqsm_mechanism.core.adaptive_calibrator import AdaptiveAttackCalibrator

        with tempfile.TemporaryDirectory() as tmpdir:
            dummy = _DummyRTDETRModel(num_candidates=16, num_queries=6, num_classes=6)
            wrapper = SimpleNamespace(
                model=dummy,
                device=torch.device("cpu"),
                pretrained_model_name="facebook/detr-resnet-50",
                num_queries=6,
                is_sigmoid=False,
            )
            with patch("paper_1_aqsm_mechanism.core.adaptive_calibrator.get_outputs_dir", return_value=Path(tmpdir)):
                cal = AdaptiveAttackCalibrator(wrapper)
                self.assertIn("detr_resnet_50", cal.cache_path.name)
                params = cal.fit(torch.rand(4, 3, 32, 32), batch_size=2, force_recompute=True)
                self.assertEqual(params["num_queries"], 6)
                self.assertTrue(cal.cache_path.exists())

    def test_attack_generator_eqf_small_s_and_taskpgd_short_labels(self):
        from paper_1_aqsm_mechanism.core.attack_generator import DETRAttackGenerator

        # DETR-like (no enc_outputs_class, softmax logits [..., C+1])
        class _SoftmaxMock(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 8, 3, padding=1)
                self.cls = nn.Linear(8, 12 * 6)
                self.box = nn.Linear(8, 12 * 4)

            def forward(self, pixel_values=None, pixel_mask=None, labels=None, output_hidden_states=False, return_dict=True):
                feat = self.conv(pixel_values).mean(dim=(-2, -1))
                B = pixel_values.shape[0]
                return SimpleNamespace(
                    logits=self.cls(feat).view(B, 12, 6),
                    pred_boxes=torch.sigmoid(self.box(feat).view(B, 12, 4)),
                    enc_outputs_class=None,
                    enc_outputs_coord_logits=None,
                    decoder_hidden_states=None,
                    loss=None,
                )

        wrapper = SimpleNamespace(model=_SoftmaxMock(), device=torch.device("cpu"), num_queries=4, is_sigmoid=False)
        gen = DETRAttackGenerator(wrapper, epsilon=16 / 255, alpha=2 / 255, num_steps=1)
        x = torch.rand(2, 3, 32, 32)
        adv_eqf = gen.generate_encoder_fabrication_attack(x, target_class_idx=99, steps=1, k_fake=20)
        self.assertEqual(tuple(adv_eqf.shape), (2, 3, 32, 32))

        short_lbls = [{"class_labels": torch.tensor([1]), "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]])}]
        adv_pgd = gen.generate_standard_task_pgd(x, hf_labels=short_lbls, steps=1)
        self.assertEqual(tuple(adv_pgd.shape), (2, 3, 32, 32))

    def test_esm_num_queries_and_active_box_softmax(self):
        from paper_1_aqsm_mechanism.core.defense_engine import ActiveBoxCountMonitor, EncoderSelectionMonitor

        # Tensor 10 candidates; query thứ 4 (idx 3) có giá trị khác rõ với phần còn lại
        scores = torch.tensor([[10.0, 5.0, 4.0, 2.0, -2.0, -5.0, -6.0, -7.0, -8.0, -9.0]])
        d4, m4 = EncoderSelectionMonitor.extract_features_from_logits(scores, num_queries=4)
        d8, _ = EncoderSelectionMonitor.extract_features_from_logits(scores, num_queries=8)
        self.assertNotAlmostEqual(float(d4[0]), float(d8[0]), places=2)
        self.assertGreater(float(m4[0]), 0.5)

        # ActiveBoxCountMonitor softmax: lớp cuối là no-object logit rất lớn
        lgts = torch.tensor([[[0.0, 0.0, 10.0], [10.0, 0.0, 0.0]]])  # 1 query thực, 1 query no-object
        mon = ActiveBoxCountMonitor(conf_threshold=0.5, is_sigmoid=False)
        cnts = mon.count_active_boxes(lgts)
        self.assertEqual(float(cnts[0]), 1.0)

    def test_hardening_summary_only_and_pending_manifest_and_tables(self):
        from paper_1_aqsm_mechanism.core.hardening_prototype import evaluate_sqa_hardening
        from paper_1_aqsm_mechanism.experiments.tasks.task7_generate_figures import plot_defense_evaluation_figure
        from paper_1_aqsm_mechanism.tools.compute_nonparametric_statistics import main as np_stats_main
        from paper_1_aqsm_mechanism.tools.generate_data_provenance_manifest import generate_provenance_manifest
        from paper_1_aqsm_mechanism.tools.generate_publication_tables import (
            generate_table6,
            generate_table7,
            generate_table11,
            generate_table12,
            generate_table13,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            sdir = out_dir / "summary_json"
            rdir = out_dir / "raw_csv"
            sdir.mkdir(parents=True, exist_ok=True)
            rdir.mkdir(parents=True, exist_ok=True)

            # 1. evaluate_sqa_hardening chỉ có sqa_controls_and_adaptive_summary.json (không có hardening_evaluation_results.json)
            (sdir / "sqa_controls_and_adaptive_summary.json").write_text(json.dumps({
                "metadata": {"model": "PekingU/rtdetr_r18vd", "num_dynamic_queries": 300, "num_static_anchors": 30},
                "baselines": {
                    "clean_nominal": {"dynamic_300_map50": 60.0, "dynamic_300_map50_95": 42.0, "dynamic_300_recall": 75.0},
                    "aqsm_eps16": {"dynamic_300_map50": 5.0, "dynamic_300_map50_95": 3.0, "dynamic_300_recall": 10.0},
                },
                "controls": {
                    "uniform_grid_30_sqa": {"map50": 18.0, "map50_95": 12.0, "recall_pct": 30.0},
                    "dynamic_330_topk": {"map50": 5.2, "map50_95": 3.1, "recall_pct": 10.5},
                    "random_30_static": {"map50": 11.0, "map50_95": 7.0, "recall_pct": 20.0},
                },
                "adaptive_threat_boundary": {"adaptive_map50": 4.0, "adaptive_map50_95": 2.5, "adaptive_recall_pct": 8.0, "evasion_success_pct": 78.0},
            }), encoding="utf-8")

            with patch("paper_1_aqsm_mechanism.experiments.config.get_summary_json_dir", return_value=sdir):
                res_sqa = evaluate_sqa_hardening(summary_dir=sdir)
            self.assertEqual(res_sqa["results"]["aqsm_attack_eps16"]["hardened_static_augmented"]["mAP@50"], 18.0)

            # 2. plot_defense_evaluation_figure khi hardening_json_path=None không crash
            def_p = sdir / "defense_esm_evaluation.json"
            def_p.write_text(json.dumps({
                "n_cal": 5,
                "n_test": 10,
                "target_fpr_pct": 5.0,
                "empirical_clean_fpr_pct": 0.0,
                "clean_mean_anomaly_score": 0.25,
                "attacks": {
                    "aqsm": {"tpr_detection_rate_pct": 95.0, "auroc": 0.99, "mean_anomaly_score": 0.95},
                    "vanishing": {"tpr_detection_rate_pct": 70.0, "auroc": 0.85, "mean_anomaly_score": 0.72},
                    "taskpgd": {"tpr_detection_rate_pct": 40.0, "auroc": 0.65, "mean_anomaly_score": 0.55},
                },
                "adaptive_evasion": {
                    "lambda_0.2": {"detected_tpr_pct": 10.0, "raw_monitor_bypass_pct": 90.0},
                    "lambda_0.5": {"detected_tpr_pct": 5.0, "raw_monitor_bypass_pct": 95.0},
                    "lambda_1.0": {"detected_tpr_pct": 0.0, "raw_monitor_bypass_pct": 100.0},
                    "lambda_2.0": {"detected_tpr_pct": 0.0, "raw_monitor_bypass_pct": 100.0},
                },
            }), encoding="utf-8")
            plot_defense_evaluation_figure(def_p, out_dir / "fig8.pdf", out_dir / "fig8.png", hardening_json_path=None)
            self.assertTrue((out_dir / "fig8.pdf").exists())

            # Table 11 khi chưa có matched_budget_sweep/results vẫn chạy sạch với '---'
            _, _, _, t11_rows_no_ref = generate_table11(sdir)
            self.assertEqual(t11_rows_no_ref[0][5], "---")

            # 3. Các JSON tối thiểu cho Table 6, 7, 11, 12, 13 & Manifest pending SQA & Nonparametric stats
            (sdir / "causal_swap_results_aqsm.json").write_text(json.dumps({
                "n_evaluated_images": 3,
                "anova_repeated_measures_full": {"N": 3, "marginal_effects": {"ME_M": -10.0, "ME_Q": -35.0, "Interaction_MxQ": 5.0}, "attribution_shares": {"anova_condition_ss_share_pct": {"Memory_M": 20.0, "Query_Q": 80.0}}},
                "point_maps": {"Arm_A": {"mAP@50": 50.0}, "Arm_B": {"mAP@50": 5.0}, "Arm_C_full": {"mAP@50": 40.0}, "Arm_D_full": {"mAP@50": 15.0}, "Arm_D_index": {"mAP@50": 14.0}},
            }), encoding="utf-8")
            (sdir / "causal_swap_results_vanishing.json").write_text(json.dumps({
                "n_evaluated_images": 3,
                "point_maps": {"Arm_A": {"mAP@50": 50.0}, "Arm_B": {"mAP@50": 6.0}, "Arm_C_full": {"mAP@50": 42.0}, "Arm_D_full": {"mAP@50": 16.0}},
            }), encoding="utf-8")
            (sdir / "causal_swap_results_taskpgd.json").write_text(json.dumps({
                "n_evaluated_images": 3,
                "point_maps": {"Arm_A": {"mAP@50": 50.0}, "Arm_B": {"mAP@50": 8.0}, "Arm_C_full": {"mAP@50": 35.0}, "Arm_D_full": {"mAP@50": 25.0}},
            }), encoding="utf-8")
            (sdir / "encoder_memory_analysis.json").write_text(json.dumps({
                "results": {
                    "aqsm": {"cosine_similarity": {"Overall": 0.96, "S3": 0.95, "S4": 0.97, "S5": 0.98}, "relative_l2_perturbation": {"Foreground": 0.25, "Background": 0.20}},
                    "vanishing": {"cosine_similarity": {"Overall": 0.94, "S3": 0.93, "S4": 0.95, "S5": 0.96}, "relative_l2_perturbation": {"Foreground": 0.30, "Background": 0.22}},
                    "taskpgd": {"cosine_similarity": {"Overall": 0.90, "S3": 0.89, "S4": 0.91, "S5": 0.92}, "relative_l2_perturbation": {"Foreground": 0.40, "Background": 0.30}},
                }
            }), encoding="utf-8")
            (sdir / "query_selection_dynamics.json").write_text(json.dumps({
                "empirical_null_baseline": {"mean_pct": 15.0, "std_pct": 2.0, "per_image_union_areas": [0.15, 0.16, 0.14]},
                "results": {
                    "aqsm": {"top300_overlap_pct": 18.0, "spearman_rho_all_proposals": 0.42, "kendall_tau_all_proposals": 0.30, "evicted_top15_fg_pct": 88.0, "anchor_foreground_ratio": {"adv_top300_fg_ratio_pct": {"mean": 22.0, "std": 1.5}}, "object_coverage": {"center_in_gt_clean_pct": 96.0, "iou30_clean_pct": 94.0, "center_in_gt_adv_pct": 75.0, "iou30_adv_pct": 70.0}},
                    "vanishing": {"top300_overlap_pct": 30.0, "spearman_rho_all_proposals": 0.55, "kendall_tau_all_proposals": 0.40, "evicted_top15_fg_pct": 65.0, "object_coverage": {"center_in_gt_adv_pct": 82.0, "iou30_adv_pct": 78.0}},
                    "taskpgd": {"top300_overlap_pct": 55.0, "spearman_rho_all_proposals": 0.72, "kendall_tau_all_proposals": 0.56, "evicted_top15_fg_pct": 35.0, "object_coverage": {"center_in_gt_adv_pct": 88.0, "iou30_adv_pct": 84.0}},
                }
            }), encoding="utf-8")
            (sdir / "deformable_attention_telemetry.json").write_text(json.dumps({
                "clean": {"mean_entropy_l1": 0.78, "mean_entropy_all_layers": 0.78, "std_entropy_all_layers": 0.02, "mean_fg_hit_rate": 0.62, "mean_dropoff_ratio": 35.0, "std_dropoff_ratio": 5.0, "mean_score_margin": 0.05, "std_score_margin": 0.01},
                "aqsm": {"mean_entropy_l1": 0.79, "mean_entropy_all_layers": 0.79, "std_entropy_all_layers": 0.02, "mean_fg_hit_rate": 0.41, "mean_dropoff_ratio": 1.25, "std_dropoff_ratio": 0.2, "mean_score_margin": 0.002, "std_score_margin": 0.001, "mean_ref_drift": 0.22, "std_ref_drift": 0.05},
                "vanishing": {"mean_entropy_l1": 0.77, "mean_entropy_all_layers": 0.77, "std_entropy_all_layers": 0.02, "mean_fg_hit_rate": 0.48, "mean_dropoff_ratio": 8.5, "std_dropoff_ratio": 1.5, "mean_score_margin": 0.01, "std_score_margin": 0.005, "mean_ref_drift": 0.14, "std_ref_drift": 0.03},
                "taskpgd": {"mean_entropy_l1": 0.76, "mean_entropy_all_layers": 0.76, "std_entropy_all_layers": 0.02, "mean_fg_hit_rate": 0.55, "mean_dropoff_ratio": 15.0, "std_dropoff_ratio": 2.0, "mean_score_margin": 0.02, "std_score_margin": 0.008, "mean_ref_drift": 0.09, "std_ref_drift": 0.02},
            }), encoding="utf-8")
            (sdir / "matched_budget_results.json").write_text(json.dumps({
                "n_images": 3,
                "clean_baseline": {"mAP@50": 50.0, "mAP@50:95": 35.0},
                "sweep_results": {
                    "aqsm": {"eps_4": {"mAP@50": 20.0, "mAP@50:95": 12.0, "relative_drop_pct": 60.0}, "eps_8": {"mAP@50": 10.0, "mAP@50:95": 6.0, "relative_drop_pct": 80.0}, "eps_16": {"mAP@50": 5.0, "mAP@50:95": 3.0, "relative_drop_pct": 90.0}},
                    "vanishing": {"eps_4": {"mAP@50": 25.0, "mAP@50:95": 16.0, "relative_drop_pct": 50.0}, "eps_8": {"mAP@50": 14.0, "mAP@50:95": 9.0, "relative_drop_pct": 72.0}, "eps_16": {"mAP@50": 7.0, "mAP@50:95": 4.0, "relative_drop_pct": 86.0}},
                    "taskpgd": {"eps_4": {"mAP@50": 22.0, "mAP@50:95": 14.0, "relative_drop_pct": 56.0}, "eps_8": {"mAP@50": 12.0, "mAP@50:95": 7.0, "relative_drop_pct": 76.0}, "eps_16": {"mAP@50": 8.0, "mAP@50:95": 5.0, "relative_drop_pct": 84.0}},
                },
            }), encoding="utf-8")
            # Table 6 & 7 khi chưa có controlled_misrouting_results.json vẫn chạy sạch
            t6_no_mis, _, _, _ = generate_table6(sdir)
            t7_no_mis, _, _, _ = generate_table7(sdir)
            self.assertIn("tab6_query_dynamics", t6_no_mis)
            self.assertIn("tab7_attention_telemetry", t7_no_mis)

            (sdir / "controlled_misrouting_results.json").write_text(json.dumps({
                "n_images": 3,
                "table8_summary": {
                    "Clean_Baseline": {"mAP@50": 50.0, "fg_hit_pct": 57.5},
                    "Random_Reassignment": {"mAP@50_mean": 46.0},
                    "Matched_L2_Random_Shift": {"mAP@50_mean": 31.0},
                    "Decoupled_Random_Shift": {"mAP@50_mean": 29.5},
                    "AQSM_Spatial_Misrouting": {"mAP@50": 20.0, "fg_hit_pct": 22.0, "hungarian_dist": 0.21},
                },
            }), encoding="utf-8")
            (sdir / "edge_latency_results.json").write_text(json.dumps({
                "device_type": "cpu",
                "throughput_fps": 20.0,
                "stages": {"Backbone": {"mean_ms": 10.0}, "Hybrid_Encoder": {"mean_ms": 15.0}, "Query_Selection": {"mean_ms": 5.0}, "Deformable_Decoder": {"mean_ms": 20.0}, "End_to_End_Total": {"mean_ms": 50.0}},
            }), encoding="utf-8")
            (sdir / "multimodel_and_seed_stability_summary.json").write_text(json.dumps({
                "records": {
                    "rtdetr_r101vd_multiseed": {
                        "model": "RT-DETR-R101",
                        "backbone": "ResNet-101",
                        "seeds": [42, 100],
                        "clean_map50": 55.0,
                        "aqsm_map50_mean": 6.0,
                        "aqsm_map50_std": 0.3,
                        "relative_drop_pct": -89.1,
                        "hungarian_set_drift_mean": 0.25,
                        "hungarian_set_drift_std": 0.01,
                        "anchor_fg_hit_pct_mean": 19.0,
                        "anchor_fg_hit_pct_std": 1.2,
                    }
                }
            }), encoding="utf-8")

            # Table 6, 7, 11, 12, 13 (dùng output res_sqa chưa có taskpgd_attack_eps16 cho Table 12)
            (sdir / "hardening_evaluation_results.json").write_text(json.dumps(res_sqa), encoding="utf-8")
            t6_tex, _, _, _ = generate_table6(sdir)
            t7_tex, _, _, _ = generate_table7(sdir)
            t11_tex, _, _, t11_rows = generate_table11(sdir)
            t12_tex, _, _, t12_rows = generate_table12(sdir)
            t13_tex, _, _, t13_rows = generate_table13(sdir)
            self.assertIn("tab6_query_dynamics", t6_tex)
            self.assertIn("tab7_attention_telemetry", t7_tex)
            self.assertEqual(t11_rows[0][5], "50.00%")
            self.assertEqual(t12_rows[-1][1], "---")  # Task PGD không có trong hardening -> '---' thay vì 0.00%
            self.assertTrue(any("RT-DETR-R101" in r[0] for r in t13_rows))

            # Manifest khi hardening đang pending (results: {})
            (sdir / "hardening_evaluation_results.json").write_text(json.dumps({"status": "pending_sqa_controls_run", "results": {}}), encoding="utf-8")
            manifest = generate_provenance_manifest(sdir, sdir / "data_provenance_manifest.json")
            self.assertEqual(manifest["tables"]["tab:tab12_hardening_validation"]["metrics"]["status"], "pending_sqa_controls_run")

            # compute_nonparametric_statistics đọc đúng n_valid_images
            (rdir / "causal_anova_per_image_aqsm.csv").write_text(
                "image_index,image_id,gt_box_count,evicted_ratio,arm_A_ap50,arm_B_ap50,arm_C_full_ap50,arm_C_index_ap50,arm_D_full_ap50,arm_D_index_ap50,arm_A_ap95,arm_B_ap95,arm_C_full_ap95,arm_C_index_ap95,arm_D_full_ap95,arm_D_index_ap95\n"
                "0,1,2,0.5,0.8,0.1,0.6,0.55,0.2,0.18,0.5,0.05,0.4,0.35,0.1,0.09\n"
                "1,2,3,0.6,0.7,0.05,0.5,0.45,0.15,0.12,0.45,0.02,0.3,0.28,0.08,0.07\n"
                "2,3,1,0.7,0.9,0.2,0.7,0.65,0.25,0.22,0.6,0.1,0.45,0.4,0.12,0.1\n",
                encoding="utf-8",
            )
            (rdir / "attention_telemetry_per_image.csv").write_text(
                "image_index,image_id,gt_box_count,clean_ent_l1,clean_ent_l2,clean_ent_l3,clean_fg_hit,clean_dropoff,clean_margin,clean_ref_drift,aqsm_ent_l1,aqsm_ent_l2,aqsm_ent_l3,aqsm_fg_hit,aqsm_dropoff,aqsm_margin,aqsm_ref_drift\n"
                "0,1,2,0.75,0.76,0.77,0.6,30.0,0.05,0.0,0.76,0.77,0.78,0.4,1.2,0.002,0.2\n"
                "1,2,3,0.74,0.75,0.76,0.62,32.0,0.06,0.0,0.75,0.76,0.77,0.42,1.3,0.003,0.21\n"
                "2,3,1,0.78,0.79,0.80,0.65,35.0,0.07,0.0,0.79,0.80,0.81,0.44,1.4,0.004,0.22\n",
                encoding="utf-8",
            )
            (rdir / "tautology_refutation_raw.csv").write_text(
                "image_index,image_id,clean_ap50,aqsm_ap50,reassign_ap50_mean,shift_ap50_mean,decoupled_shift_ap50_mean,aqsm_l2,shift_l2,aqsm_center_l2,aqsm_size_l2,shift_center_l2,shift_size_l2,shift_center_l2_max_err,shift_size_l2_max_err,clean_fg_hit_pct,aqsm_fg_hit_pct,reassign_fg_hit_mean_pct,shift_fg_hit_mean_pct,decoupled_shift_fg_hit_mean_pct,gt_union_null_pct,uniform_grid_300_hit_pct\n"
                "0,1,0.8,0.1,0.7,0.4,0.38,0.2,0.2,0.15,0.05,0.15,0.05,0.001,0.001,55,20,50,35,34,15,25\n"
                "1,2,0.7,0.05,0.65,0.35,0.33,0.22,0.22,0.16,0.06,0.16,0.06,0.001,0.001,58,22,52,36,35,16,26\n"
                "2,3,0.9,0.2,0.8,0.45,0.42,0.19,0.19,0.14,0.05,0.14,0.05,0.001,0.001,60,24,55,38,37,14,24\n",
                encoding="utf-8",
            )
            (rdir / "matched_budget_sweep_raw.csv").write_text(
                "image_index,image_id,clean_fg_hit,clean_iou30,clean_center_cov,aqsm_evicted,aqsm_fg_hit,aqsm_iou30,aqsm_center_cov\n"
                "0,1,55,90,95,0.8,20,70,75\n"
                "1,2,58,92,96,0.85,22,72,76\n"
                "2,3,60,94,97,0.9,24,74,78\n",
                encoding="utf-8",
            )
            with patch.object(sys, "argv", ["compute_nonparametric_statistics.py", "--output_dir", str(out_dir)]):
                np_stats_main()
            np_out = json.loads((sdir / "refined_statistical_methodology_summary.json").read_text(encoding="utf-8"))
            self.assertIn("N=3 valid pairs", np_out["metadata"]["dataset"])


if __name__ == "__main__":
    unittest.main()
