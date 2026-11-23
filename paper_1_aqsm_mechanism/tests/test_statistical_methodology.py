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


if __name__ == "__main__":
    unittest.main()
