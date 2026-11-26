# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.tests.test_end_to_end_pipeline_smoke
Mục đích: Kiểm thử tích hợp toàn tuyến (End-to-End Pipeline Smoke Integration Test)
chạy qua mã nguồn thực tế của toàn bộ các Task (Task 1, 2, 3, 4, 5, 6, 8, 10)
cùng các công cụ mở rộng (`run_aqsm_sensitivity`, `run_multimodel_benchmark`,
`task7_generate_figures`, `task9_generate_tables`, `generate_data_provenance_manifest`)
trên mô hình `RTDetrForObjectDetection` cấu hình nhỏ (không cần tải mạng/GPU)
và tập dữ liệu COCO val2017 giả lập (N=12 ảnh 640x640) trong thư mục tạm.

Mục tiêu: Phát hiện tức thì 100% các lỗi runtime tiềm ẩn (TypeError, KeyError,
NameError, sai tham số hàm, sai CLI argparse, sai schema JSON) trước khi chạy GPU.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from transformers import RTDetrConfig, RTDetrForObjectDetection, RTDetrResNetConfig

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from paper_1_aqsm_mechanism.core.attack_generator import DETRAttackGenerator
from paper_1_aqsm_mechanism.core.coco_gt import postprocess_rtdetr_predictions
from paper_1_aqsm_mechanism.core.dataset_loader import FlexibleCOCODataset
from paper_1_aqsm_mechanism.core.detr_wrapper import RTDETRModelWrapper
from paper_1_aqsm_mechanism.core.hardening_prototype import StaticQueryAugmenter
from paper_1_aqsm_mechanism.experiments.config import update_experiment_paths


def _build_tiny_rtdetr_config() -> RTDetrConfig:
    """Tạo cấu hình RT-DETR siêu nhỏ trên CPU nhưng giữ nguyên 3 scales, 3 decoder layers và 300 queries."""
    bb_cfg = RTDetrResNetConfig(
        depths=[1, 1, 1, 1],
        hidden_sizes=[8, 16, 32, 64],
        embedding_size=8,
        out_features=["stage2", "stage3", "stage4"],
    )
    return RTDetrConfig(
        backbone_config=bb_cfg,
        d_model=32,
        encoder_hidden_dim=32,
        encoder_layers=1,
        encoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_layers=3,
        decoder_attention_heads=4,
        decoder_ffn_dim=32,
        decoder_in_channels=[32, 32, 32],
        encoder_in_channels=[16, 32, 64],
        num_queries=300,
        num_labels=80,
        num_feature_levels=3,
        decoder_n_points=4,
        _attn_implementation="eager",
    )


def _create_synthetic_coco_dir(root_dir: Path, num_images: int = 12, id2label: dict = None) -> tuple:
    """Tạo tập COCO giả lập hợp lệ (12 ảnh 640x640 + instances_val2017.json)."""
    img_dir = root_dir / "val2017"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_file = root_dir / "instances_val2017.json"

    rng = np.random.default_rng(42)
    images_meta = []
    annotations_meta = []
    categories_meta = [
        {"id": i + 1, "name": str(id2label[i]) if id2label and i in id2label else f"LABEL_{i}", "supercategory": "obj"}
        for i in range(80)
    ]

    ann_id = 1
    for i in range(num_images):
        img_id = 1000 + i
        fname = f"{img_id:012d}.jpg"
        arr = rng.integers(0, 255, size=(640, 640, 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_dir / fname, quality=85)
        images_meta.append({"id": img_id, "file_name": fname, "width": 640, "height": 640})

        # Tạo 2 bounding boxes hợp lệ mỗi ảnh
        for b_k in range(2):
            x = float(60 + b_k * 200)
            y = float(80 + b_k * 120)
            w, h = 140.0, 120.0
            annotations_meta.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": int((i + b_k) % 5 + 1),
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
            })
            ann_id += 1

    with open(ann_file, "w", encoding="utf-8") as f:
        json.dump(
            {"images": images_meta, "annotations": annotations_meta, "categories": categories_meta},
            f,
        )
    return img_dir, ann_file


class TestEndToEndPipelineSmoke(unittest.TestCase):
    """Kiểm tra luồng chạy thật xuyên suốt các Task thực nghiệm trên Tiny RT-DETR."""

    @classmethod
    def setUpClass(cls):
        cls._tmpdir_obj = tempfile.TemporaryDirectory()
        cls.tmpdir = Path(cls._tmpdir_obj.name)
        cls.tiny_cfg = _build_tiny_rtdetr_config()
        cls.wrapper = RTDETRModelWrapper(custom_config=cls.tiny_cfg, device=torch.device("cpu"))
        cls.wrapper.pretrained_model_name = "PekingU/rtdetr_r18vd"
        cls.wrapper.eval()

        cls.img_dir, cls.ann_file = _create_synthetic_coco_dir(
            cls.tmpdir, num_images=12, id2label=cls.wrapper.model.config.id2label
        )
        cls.out_dir = cls.tmpdir / "outputs"
        cls.cache_dir = cls.tmpdir / "cache"
        cls.fig_dir = cls.tmpdir / "fig"
        cls.out_dir.mkdir(parents=True, exist_ok=True)
        cls.cache_dir.mkdir(parents=True, exist_ok=True)
        cls.fig_dir.mkdir(parents=True, exist_ok=True)

        update_experiment_paths(
            folder_image=cls.img_dir,
            annotations=cls.ann_file,
            output_dir=cls.out_dir,
            cache_dir=cls.cache_dir,
            fig_dir=cls.fig_dir,
            model="PekingU/rtdetr_r18vd",
        )

        cls.dataset = FlexibleCOCODataset(
            folder_image=cls.img_dir,
            annotations_file=cls.ann_file,
            img_size=640,
            id2label=cls.wrapper.model.config.id2label,
            lazy_tensors=True,
            device=torch.device("cpu"),
            max_samples=12,
        )

        # Sinh 1 bước nhiễu nhẹ đại diện cho AQSM, Vanishing, TaskPGD
        cls.attacker = DETRAttackGenerator(
            cls.wrapper, epsilon=16.0 / 255.0, alpha=4.0 / 255.0, num_steps=1
        )
        clean_t = cls.dataset.tensors[:].to(torch.device("cpu"))
        cls.clean_imgs = clean_t
        cls.adv_aqsm = cls.attacker.generate_query_selection_attack(clean_t, steps=1)
        cls.adv_van = cls.attacker.generate_vanishing_attack(clean_t, steps=1)
        cls.adv_pgd = cls.attacker.generate_standard_task_pgd(clean_t, steps=1)

    @classmethod
    def tearDownClass(cls):
        try:
            cls._tmpdir_obj.cleanup()
        except Exception:
            pass

    def _make_wrapper_factory(self):
        tiny_w = self.wrapper
        def _factory(*args, **kwargs):
            return tiny_w
        return _factory

    def test_01_task1_causal_surgery_smoke(self):
        from paper_1_aqsm_mechanism.experiments.tasks.task1_causal_surgery import run_causal_surgery

        res = run_causal_surgery(
            wrapper=self.wrapper,
            clean_imgs=self.clean_imgs,
            adv_imgs=self.adv_aqsm,
            gt_list=self.dataset.gt_list,
            attack_name="aqsm",
            batch_size=4,
            n_boot=20,
            seed=42,
        )
        self.assertIn("point_maps", res)
        self.assertIn("Arm_A", res["point_maps"])
        self.assertTrue((self.out_dir / "summary_json" / "causal_swap_results_aqsm.json").exists())

    def test_02_task2_encoder_query_dynamics_smoke(self):
        from paper_1_aqsm_mechanism.experiments.tasks.task2_encoder_query_dynamics import (
            run_encoder_query_dynamics_evaluation,
        )

        with patch(
            "paper_1_aqsm_mechanism.experiments.tasks.task2_encoder_query_dynamics.load_cached_adv_images"
        ) as mock_load:
            atk_map = {"aqsm": self.adv_aqsm, "vanishing": self.adv_van, "taskpgd": self.adv_pgd}
            mock_load.side_effect = lambda attack, **kw: (atk_map[attack], {"source": "smoke"})

            res = run_encoder_query_dynamics_evaluation(
                wrapper=self.wrapper,
                clean_imgs=self.clean_imgs,
                gt_list=self.dataset.gt_list,
                image_ids=self.dataset.image_ids,
                attacks_to_eval=("aqsm", "vanishing", "taskpgd"),
                batch_size=4,
                cache_dir=self.cache_dir,
            )
        self.assertIn("table5_memory", res)
        self.assertIn("table6_query_selection", res)
        self.assertTrue((self.out_dir / "summary_json" / "encoder_query_dynamics.json").exists())
        self.assertTrue((self.out_dir / "raw_csv" / "encoder_query_dynamics_raw.csv").exists())

    def test_03_task3_deformable_telemetry_smoke(self):
        from paper_1_aqsm_mechanism.experiments.tasks import task3_deformable_telemetry

        atk_map = {"aqsm": self.adv_aqsm, "vanishing": self.adv_van, "taskpgd": self.adv_pgd}
        argv = [
            "task3_deformable_telemetry.py",
            "--folder_image", str(self.img_dir),
            "--annotations", str(self.ann_file),
            "--output_dir", str(self.out_dir),
            "--cache_dir", str(self.cache_dir),
            "--device", "cpu",
            "--batch_size", "4",
            "--max_samples", "12",
            "--threads", "2",
        ]
        with patch.object(task3_deformable_telemetry, "RTDETRModelWrapper", side_effect=self._make_wrapper_factory()), \
             patch.object(task3_deformable_telemetry, "load_cached_adv_images", side_effect=lambda attack, **kw: (atk_map[attack], {"source": "smoke"})), \
             patch.object(sys, "argv", argv):
            task3_deformable_telemetry.main()

        tel_file = self.out_dir / "summary_json" / "deformable_attention_telemetry.json"
        self.assertTrue(tel_file.exists())
        tel_data = json.loads(tel_file.read_text(encoding="utf-8"))
        self.assertIn("aqsm", tel_data)
        self.assertIn("std_entropy_all_layers", tel_data["aqsm"])
        self.assertIn("mean_entropy_all_layers", tel_data["aqsm"])

    def test_04_task4_controlled_misrouting_smoke(self):
        from paper_1_aqsm_mechanism.experiments.tasks.task4_controlled_misrouting import (
            run_controlled_misrouting_evaluation,
        )

        res = run_controlled_misrouting_evaluation(
            wrapper=self.wrapper,
            dataset=self.dataset,
            adv_imgs=self.adv_aqsm,
            batch_size=4,
            n_boot=20,
            cache_dir=self.cache_dir,
            seeds=[42, 100],
        )
        self.assertIn("table8_summary", res)
        self.assertIn("AQSM_Spatial_Misrouting", res["table8_summary"])
        self.assertIn("directional_resultant_length", res["table8_summary"]["AQSM_Spatial_Misrouting"])
        self.assertTrue((self.out_dir / "raw_csv" / "tautology_refutation_raw.csv").exists())

    def test_05_task5_matched_sweep_smoke(self):
        from paper_1_aqsm_mechanism.experiments.tasks import task5_matched_sweep

        atk_map = {"aqsm": self.adv_aqsm, "vanishing": self.adv_van, "taskpgd": self.adv_pgd}
        with patch.object(
            task5_matched_sweep,
            "load_cached_adv_images",
            side_effect=lambda attack, **kw: (atk_map[attack], {"source": "smoke"}),
        ):
            res = task5_matched_sweep.run_matched_budget_sweep(
                wrapper=self.wrapper,
                clean_imgs=self.clean_imgs,
                gt_list=self.dataset.gt_list,
                image_ids=self.dataset.image_ids,
                epsilons=[4.0 / 255.0, 8.0 / 255.0, 16.0 / 255.0],
                batch_size=4,
                cache_dir=self.cache_dir,
            )
        self.assertIn("sweep_results", res)
        self.assertIn("aqsm", res["sweep_results"])

    def test_06_task6_latency_profiler_smoke(self):
        from paper_1_aqsm_mechanism.experiments.tasks import task6_latency_profiler

        class _OneStepAttacker(DETRAttackGenerator):
            def __init__(self, wrapper, epsilon=16.0 / 255.0, alpha=2.0 / 255.0, num_steps=15):
                super().__init__(wrapper, epsilon=epsilon, alpha=alpha, num_steps=1)
            def generate_query_selection_attack(self, images, **kwargs):
                kwargs["steps"] = 1
                return super().generate_query_selection_attack(images, **kwargs)
            def generate_standard_task_pgd(self, images, **kwargs):
                kwargs["steps"] = 1
                return super().generate_standard_task_pgd(images, **kwargs)
            def generate_latency_flooding_attack(self, images, **kwargs):
                kwargs["steps"] = 1
                return super().generate_latency_flooding_attack(images, **kwargs)

        argv = [
            "task6_latency_profiler.py",
            "--folder_image", str(self.img_dir),
            "--annotations", str(self.ann_file),
            "--output_dir", str(self.out_dir),
            "--device", "cpu",
            "--n_runs", "2",
            "--warmup", "1",
            "--threads", "2",
        ]
        with patch.object(task6_latency_profiler, "RTDETRModelWrapper", side_effect=self._make_wrapper_factory()), \
             patch.object(task6_latency_profiler, "DETRAttackGenerator", _OneStepAttacker), \
             patch.object(sys, "argv", argv):
            task6_latency_profiler.main()

        lat_file = self.out_dir / "summary_json" / "edge_latency_results.json"
        self.assertTrue(lat_file.exists())
        lat_data = json.loads(lat_file.read_text(encoding="utf-8"))
        self.assertIn("adversarial_latency_comparison", lat_data)
        self.assertIn("RT_DETR_Flooding_Sponge", lat_data["adversarial_latency_comparison"])

    def test_07_task8_defense_evaluation_smoke(self):
        from paper_1_aqsm_mechanism.experiments.tasks import task8_defense_evaluation
        from paper_1_aqsm_mechanism.core import attack_generator

        orig_atk_cls = attack_generator.DETRAttackGenerator

        class _FastOneStepAttacker(orig_atk_cls):
            def __init__(self, wrapper, epsilon=16.0 / 255.0, alpha=2.0 / 255.0, num_steps=15):
                super().__init__(wrapper, epsilon=epsilon, alpha=alpha, num_steps=1)
            def generate_adaptive_evasion_attack(self, images, **kwargs):
                kwargs["steps"] = 1
                return super().generate_adaptive_evasion_attack(images, **kwargs)
            def generate_adaptive_sqa_attack(self, images, sqa_module, **kwargs):
                kwargs["steps"] = 1
                return super().generate_adaptive_sqa_attack(images, sqa_module, **kwargs)

        with patch.object(task8_defense_evaluation, "load_cached_adv_images") as mock_load, \
             patch.object(attack_generator, "DETRAttackGenerator", _FastOneStepAttacker):
            atk_map = {"aqsm": self.adv_aqsm, "vanishing": self.adv_van, "taskpgd": self.adv_pgd}
            mock_load.side_effect = lambda attack, **kw: (atk_map[attack], {"source": "smoke"})
            res = task8_defense_evaluation.run_defense_evaluation(
                wrapper=self.wrapper,
                dataset=self.dataset,
                cal_ratio=0.5,
                target_fpr=0.05,
                n_adaptive=2,
                cache_dir=self.cache_dir,
                run_sqa_ablation=True,
                batch_size=4,
            )
        self.assertIn("box_count_monitor", res)
        self.assertIn("spatial_consistency_monitor", res)
        self.assertIn("sqa_hardening", res)
        self.assertNotEqual(res["sqa_hardening"].get("status"), "pending_sqa_controls_run")
        self.assertTrue((self.out_dir / "summary_json" / "defense_esm_evaluation.json").exists())

    def test_08_task10_multiseed_stability_smoke(self):
        from paper_1_aqsm_mechanism.experiments.tasks import task10_multiseed_stability

        args = SimpleNamespace(
            folder_image=str(self.img_dir),
            annotations=str(self.ann_file),
            output_dir=str(self.out_dir),
            cache_dir=str(self.cache_dir),
            model="PekingU/rtdetr_r18vd",
            seeds="42,100",
            max_samples=4,
            batch_size=2,
            eps_255=16,
            pgd_steps=1,
            device="cpu",
            threads=2,
        )
        with patch.object(task10_multiseed_stability, "RTDETRModelWrapper", side_effect=self._make_wrapper_factory()):
            payload = task10_multiseed_stability.run_multiseed_stability(args)

        self.assertIn("records", payload)
        self.assertIn("rtdetr_r18vd_multiseed", payload["records"])
        self.assertTrue((self.out_dir / "summary_json" / "multimodel_and_seed_stability_summary.json").exists())

    def test_09_tools_sensitivity_and_adaptive_cache_sqa_smoke(self):
        from paper_1_aqsm_mechanism.tools.run_aqsm_sensitivity import build_sweep_configs, evaluate_aqsm_config

        cfgs = build_sweep_configs(full_grid=False)
        self.assertEqual(len(cfgs), 9)
        rec = evaluate_aqsm_config(
            wrapper=self.wrapper,
            attacker=self.attacker,
            clean_proxy=self.clean_imgs[:4],
            gt_list=self.dataset.gt_list[:4],
            n_eval=4,
            batch_size=2,
            device=torch.device("cpu"),
            cfg=cfgs[0],
            steps=1,
            seed=42,
        )
        self.assertIn("mAP@50", rec)

        # Kiểm tra thêm adaptive SQA attack gọi đúng signature
        sqa_mod = StaticQueryAugmenter(num_static_anchors=10, strategy="uniform_grid", hidden_dim=self.wrapper.d_model)
        adv_sqa = self.attacker.generate_adaptive_sqa_attack(self.clean_imgs[:2], sqa_module=sqa_mod, steps=1)
        self.assertEqual(adv_sqa.shape, self.clean_imgs[:2].shape)

    def test_10_multimodel_benchmark_and_tables_smoke(self):
        from paper_1_aqsm_mechanism.tools import run_multimodel_benchmark
        from paper_1_aqsm_mechanism.tools.generate_publication_tables import (
            generate_table2,
            generate_table5,
            generate_table6,
            generate_table7,
            generate_table8,
            generate_table9,
            generate_table10,
            generate_table11,
            generate_table13,
        )

        # Tạo CrossDetectorWrapper giả lập khớp chuẩn interface cho benchmark_single_model
        tiny_m = self.wrapper.model

        class _FakeCrossWrapper:
            def __init__(self, model_name_or_alias, device=None):
                self.device = device or torch.device("cpu")
                self.model = tiny_m
                self.pretrained_model_name = "PekingU/rtdetr_r18vd"
                self.family = "rtdetr"
                self.is_sigmoid = True
                self.has_dynamic_selection = True
                self.num_queries = tiny_m.config.num_queries
                self.num_labels = tiny_m.config.num_labels
                self.d_model = tiny_m.config.d_model
                self.decoder_layers = tiny_m.config.decoder_layers

            def eval(self):
                self.model.eval()
                return self

            def __call__(self, x):
                o = self.model(pixel_values=x, return_dict=True)
                return {
                    "logits": o.logits,
                    "pred_boxes": o.pred_boxes,
                    "enc_outputs_class": getattr(o, "enc_outputs_class", None),
                }

        class _OneStepBenchAttacker(DETRAttackGenerator):
            def __init__(self, wrapper, epsilon=16.0 / 255.0, alpha=2.0 / 255.0, num_steps=15):
                super().__init__(wrapper, epsilon=epsilon, alpha=alpha, num_steps=1)

        with patch.object(run_multimodel_benchmark, "CrossDetectorWrapper", _FakeCrossWrapper), \
             patch.object(run_multimodel_benchmark, "DETRAttackGenerator", _OneStepBenchAttacker):
            bm_res = run_multimodel_benchmark.benchmark_single_model(
                model_name_or_alias="r18",
                folder_image=str(self.img_dir),
                annotations=str(self.ann_file),
                max_samples=4,
                batch_size=2,
                device=torch.device("cpu"),
                pgd_steps=1,
            )
        self.assertIn("clean", bm_res)
        self.assertIn("aqsm", bm_res)
        self.assertIn("task_pgd", bm_res)

        # Kiểm tra các hàm sinh bảng chạy trên summary_json thật vừa tạo bởi test_01..test_08
        sdir = self.out_dir / "summary_json"
        for fn in (
            generate_table2,
            generate_table5,
            generate_table6,
            generate_table7,
            generate_table8,
            generate_table9,
            generate_table10,
            generate_table11,
            generate_table13,
        ):
            tex, md, hdrs, rows = fn(sdir)
            self.assertTrue(len(tex) > 30)
            self.assertTrue(len(rows) >= 1)


if __name__ == "__main__":
    unittest.main()
