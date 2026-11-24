# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.tests.test_statistical_methodology
Mục đích: Unit test và Integration test chuẩn production cho các phương pháp thống kê
và công cụ thực nghiệm mới được nâng cấp theo phản biện Reviewer:
  1. Rank-Biserial Correlation r_rb cho Paired Wilcoxon Signed-Rank Test.
  2. Two One-Sided Tests (TOST) cho kiểm định tương đương hình học (Equivalence Testing).
  3. Phân tích độ mạnh thống kê (Statistical Power Analysis).
  4. Bootstrap BCa Confidence Intervals cho Attribution Shares (Memory vs Query).
  5. Tính nhất quán của điều kiện đối chứng Decoupled Random Shift.
  6. Tính toàn vẹn và đồng bộ dữ liệu giữa JSON artifacts và các bảng xuất bản.
"""

import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from paper_1_aqsm_mechanism.tools.compute_nonparametric_statistics import (
    compute_rank_biserial,
    compute_tost_equivalence,
    compute_statistical_power,
    compute_bootstrap_ci,
)


class TestStatisticalMethodology(unittest.TestCase):
    """Bộ kiểm thử đơn vị cho toàn bộ các hàm thống kê phi tham số và TOST."""

    def test_01_rank_biserial_extreme_cases(self):
        """
        Kiểm tra r_rb trong các trường hợp cực trị và biên:
        - Tất cả hiệu dương: r_rb = +1.0
        - Tất cả hiệu âm: r_rb = -1.0
        - Hiệu đối xứng hoàn hảo: r_rb = 0.0
        - Mảng rỗng hoặc toàn 0: r_rb = 0.0
        """
        # Tất cả hiệu dương
        pos_diff = np.array([1.2, 3.5, 0.4, 2.8, 5.0])
        self.assertAlmostEqual(compute_rank_biserial(pos_diff), 1.0, places=5)

        # Tất cả hiệu âm
        neg_diff = np.array([-1.2, -3.5, -0.4, -2.8, -5.0])
        self.assertAlmostEqual(compute_rank_biserial(neg_diff), -1.0, places=5)

        # Đối xứng hoàn hảo
        sym_diff = np.array([-3.0, -1.0, 1.0, 3.0])
        self.assertAlmostEqual(compute_rank_biserial(sym_diff), 0.0, places=5)

        # Mảng toàn số 0
        zero_diff = np.zeros(10)
        self.assertEqual(compute_rank_biserial(zero_diff), 0.0)

        # Mảng rỗng
        empty_diff = np.array([])
        self.assertEqual(compute_rank_biserial(empty_diff), 0.0)

    def test_02_rank_biserial_with_real_effect_sizes(self):
        """
        Kiểm tra r_rb trên các phân phối mô phỏng tương ứng với các can thiệp thực tế:
        - Arm D_full vs B: Causal recovery mạnh (phần lớn > 0), r_rb > 0.95
        - Arm B vs A: Total attack damage (toàn bộ < 0), r_rb = -1.0
        """
        rng = np.random.RandomState(42)
        # Giả lập Arm D_full vs B: mean ~ +32, std ~ 8, 99% > 0
        recovery_sim = rng.normal(loc=32.0, scale=8.0, size=992)
        r_rb_rec = compute_rank_biserial(recovery_sim)
        self.assertGreater(r_rb_rec, 0.95)

        # Giả lập Arm B vs A: mean ~ -62, std ~ 6, 100% < 0
        attack_sim = rng.normal(loc=-62.0, scale=6.0, size=992)
        r_rb_att = compute_rank_biserial(attack_sim)
        self.assertEqual(r_rb_att, -1.0)

    def test_03_tost_equivalence_logic(self):
        """
        Kiểm tra tính đúng đắn toán học của Two One-Sided Tests (TOST):
        - Trường hợp 1: Chênh lệch nhỏ nằm gọn trong bound hẹp, SE nhỏ -> is_equivalent = True, p_tost < 0.05
        - Trường hợp 2: Chênh lệch vượt ngoài bound -> is_equivalent = False, p_tost >= 0.05
        - Trường hợp 3: Chênh lệch nhỏ nhưng SE quá lớn (chưa đủ bằng chứng kết luận) -> is_equivalent = False
        """
        # Trường hợp 1: Tương đương chặt
        res_equiv = compute_tost_equivalence(
            diff_mean=0.20,
            se=0.25,
            df=999,
            bound=1.5,
            alpha=0.05,
        )
        self.assertTrue(res_equiv["is_equivalent"])
        self.assertLess(res_equiv["p_tost"], 0.05)
        self.assertGreater(res_equiv["t1"], 0.0)  # (0.2 - (-1.5)) / 0.25 = 1.7 / 0.25 = 6.8
        self.assertLess(res_equiv["t2"], 0.0)     # (0.2 - 1.5) / 0.25 = -1.3 / 0.25 = -5.2

        # Trường hợp 2: Bị lệch hẳn ngoài bound (diff_mean = 3.5 > bound = 2.0)
        res_out = compute_tost_equivalence(
            diff_mean=3.5,
            se=0.30,
            df=999,
            bound=2.0,
            alpha=0.05,
        )
        self.assertFalse(res_out["is_equivalent"])
        self.assertGreater(res_out["p_tost"], 0.05)

        # Trường hợp 3: Hiệu nhỏ (diff=0.5, bound=2.0) nhưng mẫu quá nhỏ / SE quá lớn (se=3.0)
        res_noisy = compute_tost_equivalence(
            diff_mean=0.5,
            se=3.0,
            df=15,
            bound=2.0,
            alpha=0.05,
        )
        self.assertFalse(res_noisy["is_equivalent"])

    def test_04_tost_with_paper_empirical_values(self):
        """
        Kiểm tra TOST với đúng thông số thực nghiệm trong bài báo:
        - AQSM Hit: 45.13%
        - Grid Uniform Null: 44.26%, SD: 14.20%, N: 1,000 -> SE = 0.449%
        - Delta = +0.87 pp, Bound = 2.0 pp
        - Kỳ vọng: Bác bỏ phi tương đương với p_tost < 0.01 (chính xác ~0.0060)
        """
        delta = 45.13 - 44.26  # +0.87
        se = 14.20 / math.sqrt(1000)  # ~0.44905
        df = 999
        res = compute_tost_equivalence(diff_mean=delta, se=se, df=df, bound=2.0, alpha=0.05)

        self.assertTrue(res["is_equivalent"], "TOST phải khẳng định tương đương thống kê ở biên +/-2.0 pp")
        self.assertAlmostEqual(res["p_tost"], 0.0060, delta=0.001)
        self.assertGreater(res["t1"], 6.0)
        self.assertLess(res["t2"], -2.0)

    def test_05_statistical_power_analysis(self):
        """
        Kiểm tra phân tích độ mạnh thống kê (Statistical Power Analysis):
        - N=992, d=0.20, alpha=0.01: Power phải > 0.999 (gần như 1.0 tuyệt đối)
        - N=992, d=0.50, alpha=0.01: Power > 0.9999
        - Tính đơn điệu: Power tăng khi N tăng và khi d tăng.
        """
        power_large_sample = compute_statistical_power(n=992, d=0.20, alpha=0.01)
        self.assertGreater(power_large_sample, 0.999)

        power_medium_effect = compute_statistical_power(n=992, d=0.50, alpha=0.01)
        self.assertGreater(power_medium_effect, 0.9999)

        # Tính đơn điệu theo N
        p_n50 = compute_statistical_power(n=50, d=0.20, alpha=0.05)
        p_n200 = compute_statistical_power(n=200, d=0.20, alpha=0.05)
        p_n1000 = compute_statistical_power(n=1000, d=0.20, alpha=0.05)
        self.assertLess(p_n50, p_n200)
        self.assertLess(p_n200, p_n1000)

        # Tính đơn điệu theo d
        p_d01 = compute_statistical_power(n=100, d=0.10, alpha=0.05)
        p_d03 = compute_statistical_power(n=100, d=0.30, alpha=0.05)
        self.assertLess(p_d01, p_d03)

    def test_06_bootstrap_confidence_interval_coverage(self):
        """
        Kiểm tra hàm compute_bootstrap_ci:
        - Khoảng tin cậy [low, high] phải bao bọc giá trị trung bình mẫu.
        - low < mean < high.
        """
        rng = np.random.RandomState(42)
        sample = rng.normal(loc=15.0, scale=3.0, size=500)
        sample_mean = float(np.mean(sample))

        ci_low, ci_high = compute_bootstrap_ci(sample, n_boot=2000, alpha=0.05, seed=42)
        self.assertLess(ci_low, sample_mean)
        self.assertGreater(ci_high, sample_mean)
        self.assertLess(ci_high - ci_low, 2.0)  # SE ~ 3/sqrt(500) ~ 0.134, CI width ~ 0.52

    def test_07_decoupled_random_shift_summary_integrity(self):
        """
        Kiểm tra tính nhất quán dữ liệu của controlled_misrouting_results.json:
        - Điều kiện Decoupled_Random_Shift phải tồn tại.
        - Khoảng cách 5.85 pp giữa Random Shift và AQSM phải được bóc tách chính xác
          thành 4.63 pp decoupling (79.15%) và 1.22 pp clumping (20.85%).
        """
        json_path = (
            WORKSPACE_DIR
            / "paper_1_aqsm_mechanism"
            / "outputs"
            / "summary_json"
            / "controlled_misrouting_results.json"
        )
        if not json_path.exists():
            self.skipTest(f"Tệp {json_path} không tồn tại.")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        summary = data["table8_summary"]
        self.assertIn("Clean_Baseline", summary)
        self.assertIn("Matched_L2_Random_Shift", summary)
        self.assertIn("Decoupled_Random_Shift", summary)
        self.assertIn("AQSM_Spatial_Misrouting", summary)

        shift_map = summary["Matched_L2_Random_Shift"]["mAP@50_mean"]
        decoupled_map = summary["Decoupled_Random_Shift"]["mAP@50_mean"]
        aqsm_map = summary["AQSM_Spatial_Misrouting"]["mAP@50"]

        total_gap = shift_map - aqsm_map
        decoupling_penalty = shift_map - decoupled_map
        clumping_penalty = decoupled_map - aqsm_map

        self.assertAlmostEqual(total_gap, 5.85, places=2)
        self.assertAlmostEqual(decoupling_penalty, 4.63, places=2)
        self.assertAlmostEqual(clumping_penalty, 1.22, places=2)
        self.assertAlmostEqual(decoupling_penalty + clumping_penalty, total_gap, places=5)

    def test_08_refined_statistics_summary_json_integrity(self):
        """
        Kiểm tra tệp refined_statistical_methodology_summary.json:
        - Chứa đầy đủ các trường TOST, Bootstrap attribution CIs, và Power analysis.
        - Tổng tỷ lệ Macro Memory Share + Query Share phải xấp xỉ 100.0%.
        """
        json_path = (
            WORKSPACE_DIR
            / "paper_1_aqsm_mechanism"
            / "outputs"
            / "summary_json"
            / "refined_statistical_methodology_summary.json"
        )
        if not json_path.exists():
            self.skipTest(f"Tệp {json_path} không tồn tại.")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        causal = data["causal_decomposition"]
        shares = causal["attribution_shares"]
        m_share = shares["macro_memory_share_pct"]
        q_share = shares["macro_query_share_pct"]
        self.assertAlmostEqual(m_share + q_share, 100.0, places=2)

        ci_m = shares["macro_memory_share_95ci"]
        ci_q = shares["macro_query_share_95ci"]
        self.assertLess(ci_m[0], m_share)
        self.assertGreater(ci_m[1], m_share)
        self.assertLess(ci_q[0], q_share)
        self.assertGreater(ci_q[1], q_share)

        # TOST checks
        null_res = data["null_controls_and_tost"]
        grid = null_res["grid_uniform_null"]
        self.assertTrue(grid["tost_2pp"]["is_equivalent"])
        self.assertLess(grid["tost_2pp"]["p_tost"], 0.05)

    def test_09_log_scale_multiplicative_decomposition(self):
        """
        Kiểm tra tính nhất quán toán học của phép phân rã nhân tính trên thang log (Log-Scale Multiplicative Decomposition):
        - Tỉ lệ giữ lại hiệu năng: C_full / Clean = 7.45 / 65.50 ≈ 0.11374
        - Tỉ lệ giữ lại hiệu năng: D_full / Clean = 26.58 / 65.50 ≈ 0.40580
        - Tích nhân tính: 0.11374 * 0.40580 ≈ 0.04616
        - Hiệu năng Arm B / Clean: 3.07 / 65.50 ≈ 0.04687
        - Tương tác gamma trong miền log: ln(0.04687 / 0.04616) = +0.0154 ≈ +0.0155 (gần như triệt tiêu hoàn toàn).
        - Trọng số phân bổ Log Memory: ~70.68%, Log Query: ~29.32%.
        """
        y00 = 65.50
        y11 = 3.07
        y10 = 7.45   # Arm C_full (Memory corrupted, Query clean)
        y01 = 26.58  # Arm D_full (Memory clean, Query corrupted)

        ret_b = y11 / y00
        ret_c = y10 / y00
        ret_d = y01 / y00

        pred_mult = ret_c * ret_d
        self.assertAlmostEqual(pred_mult, 0.04616, places=4)
        self.assertAlmostEqual(ret_b, 0.04687, places=4)

        gamma_log = math.log(ret_b / pred_mult)
        self.assertAlmostEqual(gamma_log, 0.0154, places=3)
        self.assertLess(abs(gamma_log), 0.02)  # Tương tác siêu nhỏ, chứng minh mô hình gần như nhân tính hoàn hảo

        log_drop_m = -math.log(ret_c)
        log_drop_q = -math.log(ret_d)
        log_m_share = (log_drop_m / (log_drop_m + log_drop_q)) * 100.0
        log_q_share = (log_drop_q / (log_drop_m + log_drop_q)) * 100.0

        self.assertAlmostEqual(log_m_share, 70.68, places=1)
        self.assertAlmostEqual(log_q_share, 29.32, places=1)
        self.assertAlmostEqual(log_m_share + log_q_share, 100.0, places=5)

    def test_10_order_dependent_causal_mediation(self):
        """
        Kiểm tra tính phụ thuộc thứ tự (Order-Dependent Causal Mediation Paths):
        - Theo đường dẫn 1 (Memory can thiệp trước): ME_M = Y10 - Y00 = -58.05 pp
        - Theo đường dẫn 2 (Query can thiệp trước): ME_Q = Y01 - Y00 = -38.92 pp
        - Giá trị Shapley 2 người chơi chính là kỳ vọng đều qua 2 thứ tự can thiệp.
        """
        y00 = 65.50
        y11 = 3.07
        y10 = 7.45
        y01 = 26.58

        tot_drop = y00 - y11  # 62.43 pp
        # Order 1: M first
        drop_m_first = y00 - y10  # 58.05 pp
        drop_q_second = y10 - y11 # 4.38 pp
        self.assertAlmostEqual(drop_m_first + drop_q_second, tot_drop, places=4)
        share_m_order1 = (drop_m_first / tot_drop) * 100.0

        # Order 2: Q first
        drop_q_first = y00 - y01  # 38.92 pp
        drop_m_second = y01 - y11 # 23.51 pp
        self.assertAlmostEqual(drop_q_first + drop_m_second, tot_drop, places=4)
        share_m_order2 = (drop_m_second / tot_drop) * 100.0

        # Shapley là trung bình cộng của 2 đường dẫn
        shapley_m = (share_m_order1 + share_m_order2) / 2.0
        self.assertGreater(share_m_order1, 90.0)  # Khi M đi trước, M chiếm tới 92.98%
        self.assertLess(share_m_order2, 40.0)     # Khi Q đi trước, M chỉ còn 37.66%
        # Kiểm tra tính toán biên độc lập không phụ thuộc giả định thứ tự
        self.assertAlmostEqual(shapley_m, (drop_m_first + drop_m_second) / (2.0 * tot_drop) * 100.0, places=4)

    def test_11_two_sample_pooled_tost_and_multiseed_stability(self):
        """
        Kiểm tra kiểm định TOST 2 mẫu độc lập và tính ổn định đa seed / kiến trúc:
        - Two-Sample TOST giữa Spatial Null và AQSM empirical distribution (p < 0.05).
        - Multi-seed JSON artifact chứa đủ 5 seeds và độ lệch chuẩn nhỏ (< 0.20 pp mAP).
        - RT-DETR-R50VD duy trì độ suy giảm thảm họa > 94%.
        """
        json_path = (
            WORKSPACE_DIR
            / "paper_1_aqsm_mechanism"
            / "outputs"
            / "summary_json"
            / "multimodel_and_seed_stability_summary.json"
        )
        if not json_path.exists():
            self.skipTest(f"Tệp {json_path} không tồn tại.")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        records = data["records"]
        self.assertIn("rtdetr_r18vd_canonical", records)
        self.assertIn("rtdetr_r18vd_multiseed", records)
        self.assertIn("rtdetr_r50vd_canonical", records)

        r18_ms = records["rtdetr_r18vd_multiseed"]
        self.assertEqual(len(r18_ms["seeds"]), 5)
        self.assertLess(r18_ms["aqsm_map50_std"], 0.20)
        self.assertLess(r18_ms["hungarian_set_drift_std"], 0.01)

        r50_can = records["rtdetr_r50vd_canonical"]
        self.assertLess(r50_can["relative_drop_pct"], -90.0)
        self.assertAlmostEqual(r50_can["aqsm_map50"], 4.15, places=2)

    def test_12_data_provenance_manifest_generator_integrity(self):
        """
        Kiểm tra tính toàn vẹn và khả năng tự động sinh của công cụ generate_data_provenance_manifest.py:
        - Sinh động 100% manifest từ summary_json mà không cần chỉnh sửa hậu nghiệm thủ công.
        - Khởi tạo đầy đủ 13 Bảng biểu (Table 1 đến Table 13) và 7 Hình biểu (Figure 3 đến Figure 10).
        - Toàn bộ source artifacts được khai báo đều phải tồn tại thực tế trên ổ đĩa.
        """
        from paper_1_aqsm_mechanism.tools.generate_data_provenance_manifest import generate_provenance_manifest

        summary_dir = (
            WORKSPACE_DIR
            / "paper_1_aqsm_mechanism"
            / "outputs"
            / "summary_json"
        )
        if not summary_dir.exists():
            self.skipTest(f"Thư mục {summary_dir} không tồn tại.")

        manifest = generate_provenance_manifest(summary_dir)
        self.assertIsInstance(manifest, dict)
        self.assertEqual(len(manifest["tables"]), 13)
        self.assertEqual(len(manifest["figures"]), 7)

        # Kiểm tra Table 13 được sinh tự động chuẩn xác
        self.assertIn("tab:tab13_multiseed_generalization", manifest["tables"])
        t13 = manifest["tables"]["tab:tab13_multiseed_generalization"]
        self.assertEqual(t13["table_number"], 13)
        self.assertEqual(t13["n_evaluated_images"], 1000)
        self.assertAlmostEqual(t13["metrics"]["RTDETR_R18_Clean_mAP50"], 65.50, places=1)
        self.assertAlmostEqual(t13["metrics"]["RTDETR_R18_Canonical_AQSM_mAP50"], 3.07, places=2)
        self.assertAlmostEqual(t13["metrics"]["RTDETR_R50_AQSM_mAP50"], 4.15, places=2)

        # Kiểm tra mọi artifact khai báo đều tồn tại thực tế
        base_paper_dir = WORKSPACE_DIR / "paper_1_aqsm_mechanism"
        for tid, tinfo in manifest["tables"].items():
            for art in tinfo.get("source_artifacts", []):
                art_file = base_paper_dir / art
                self.assertTrue(art_file.exists(), f"[{tid}] Artifact {art} không tồn tại trên đĩa!")


if __name__ == "__main__":
    unittest.main()
