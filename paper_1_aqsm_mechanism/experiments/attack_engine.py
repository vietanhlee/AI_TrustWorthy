# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.experiments.attack_engine
Mục đích: Cung cấp bộ sinh mẫu đối kháng gradient chuẩn hóa L_inf
dựa trên DETRAttackGenerator phục vụ chẩn đoán cơ chế kiến trúc nội tại của RT-DETR:
  1. AQSM (Adversarial Query Selection Manipulation): Tấn công phân hạng bù trừ tại Encoder.
  2. Query Vanishing: Tấn công triệt tiêu độ tự tin tại Decoder output.
  3. Standard Task PGD: Tấn công end-to-end phá vỡ đồng thời vị trí và nhãn.
  4. Gaussian Noise Control: Nhiễu tự nhiên đối chứng (sigma = 16/255).
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from paper_1_aqsm_mechanism.core.attack_generator import DETRAttackGenerator
from paper_1_aqsm_mechanism.core.detr_wrapper import RTDETRModelWrapper
from paper_1_aqsm_mechanism.experiments.config import (
    AQSM_LAMBDA_RANK,
    AQSM_RANKING_MARGIN,
    EPSILON_L_INF,
    PGD_NUM_STEPS,
    PGD_STEP_SIZE,
)


class DiagnosticAttackEngine:
    """
    Bộ động cơ sinh mẫu đối kháng chẩn đoán lỗi kiến trúc trên RT-DETR.
    """

    def __init__(
        self,
        model: RTDETRModelWrapper,
        eps: float = EPSILON_L_INF,
        alpha: float = PGD_STEP_SIZE,
        num_steps: int = PGD_NUM_STEPS,
        device: Optional[torch.device] = None,
    ):
        self.wrapper = model
        self.device = device or model.device
        self.attacker = DETRAttackGenerator(
            model_wrapper=model,
            epsilon=eps,
            alpha=alpha,
            num_steps=num_steps,
        )

    def generate_aqsm(
        self,
        x_clean: torch.Tensor,
        track_progress: bool = False,
    ) -> Tuple[torch.Tensor, List[Dict[str, float]]]:
        """
        Sinh mẫu tấn công AQSM (Adversarial Query Selection Manipulation).
        """
        adv_imgs = self.attacker.generate_query_selection_attack(
            images=x_clean,
            steps=self.attacker.num_steps,
            margin=AQSM_RANKING_MARGIN,
            lambda_rank=AQSM_LAMBDA_RANK,
        )
        return adv_imgs, []

    def generate_query_vanishing(
        self,
        x_clean: torch.Tensor,
        track_progress: bool = False,
    ) -> Tuple[torch.Tensor, List[Dict[str, float]]]:
        """
        Sinh mẫu tấn công Query Vanishing.
        """
        adv_imgs = self.attacker.generate_vanishing_attack(
            images=x_clean,
            steps=self.attacker.num_steps,
        )
        return adv_imgs, []

    def generate_task_pgd(
        self,
        x_clean: torch.Tensor,
        track_progress: bool = False,
    ) -> Tuple[torch.Tensor, List[Dict[str, float]]]:
        """
        Sinh mẫu tấn công Standard Task PGD.
        """
        adv_imgs = self.attacker.generate_standard_task_pgd(
            images=x_clean,
            steps=self.attacker.num_steps,
        )
        return adv_imgs, []

    def generate_gaussian_noise(
        self,
        x_clean: torch.Tensor,
        sigma: float = 16.0 / 255.0,
    ) -> torch.Tensor:
        """
        Tạo đối chứng nhiễu tự nhiên Gaussian Noise.
        """
        return self.attacker.generate_random_noise(images=x_clean, sigma=sigma)
