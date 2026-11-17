"""
Phantom Query Attacker for Real-Time Detection Transformers (RT-DETR)
Paper 2: "Phantom Queries: Exploiting NMS-Free Set Prediction for Adversarial Object Fabrication in Real-Time Transformers"

Implements:
1. Encoder Query Fabrication (EQF) Attack: Exploits dynamic selection bottleneck.
2. Decoder Query Fabrication (DQF) Attack: Exploits lack of NMS in deformable decoder.
3. Foreground Proposal Displacement Telemetry: Quantifies eviction of legitimate object anchors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List


class PhantomQueryAttacker:
    """
    Dedicated Adversarial Fabrication Attacker targeting NMS-Free Detection Transformers.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        epsilon: float = 16.0 / 255.0,
        alpha: float = 2.0 / 255.0,
        num_steps: int = 15,
    ):
        self.model = model
        self.device = device
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.model.to(self.device)
        self.model.eval()

    def generate_eqf_attack(
        self,
        images: torch.Tensor,
        target_class_idx: int = 1,
        steps: Optional[int] = None,
        alpha: Optional[float] = None,
        k_fake: int = 50,
        margin: float = 0.25,
        lambda_rank: float = 1.20,
    ) -> torch.Tensor:
        """
        Encoder Query Fabrication (EQF) Attack:
        Targets the Hybrid Encoder candidate selection heads (enc_outputs_class).
        Artificially elevates classification logits of background candidate anchors,
        forcing them across the dynamic top-300 selection bottleneck without backpropagating
        through the deformable attention decoder.
        """
        self.model.eval()
        adv_images = images.clone().detach().to(self.device)
        n_steps = steps if steps is not None else self.num_steps
        step_alpha = alpha if alpha is not None else self.alpha

        # 1. Forward clean pass to identify true background candidates and top-300 cutoff
        with torch.no_grad():
            clean_out = self.model(pixel_values=images.to(self.device), return_dict=True)
            clean_enc_logits = clean_out.enc_outputs_class  # [B, N_cand, C]
            clean_enc_probs = torch.sigmoid(clean_enc_logits)
            clean_max_probs, _ = clean_enc_probs.max(dim=-1)  # [B, N_cand]
            
            # 300th order statistic
            top300_clean_val, _ = torch.topk(clean_max_probs, k=min(300, clean_max_probs.shape[1]), dim=-1)
            tau_300_clean = top300_clean_val[:, -1]  # [B]

            # Low-confidence background candidates
            bg_mask = clean_max_probs < 0.10

        # 2. PGD Optimization loop
        for _ in range(n_steps):
            adv_images.requires_grad_(True)
            outputs = self.model(pixel_values=adv_images, return_dict=True)
            enc_logits = outputs.enc_outputs_class
            target_logits = enc_logits[..., target_class_idx]  # [B, N_cand]

            # Mask out non-background candidates
            masked_target_logits = target_logits.masked_fill(~bg_mask, -1e4)
            top_fake_logits, _ = torch.topk(masked_target_logits, k=k_fake, dim=-1)

            # Maximize target class logit
            loss_fake = -top_fake_logits.mean()
            
            # Ranking margin hinge loss: push past dynamic selection threshold
            min_fake_prob = torch.sigmoid(top_fake_logits[:, -1])
            loss_rank = F.relu(tau_300_clean - min_fake_prob + margin).mean()

            total_loss = loss_fake + lambda_rank * loss_rank

            self.model.zero_grad()
            if adv_images.grad is not None:
                adv_images.grad.zero_()
            total_loss.backward()

            with torch.no_grad():
                grad_sign = adv_images.grad.sign() if adv_images.grad is not None else torch.zeros_like(adv_images)
                adv_images = adv_images - step_alpha * grad_sign
                delta = torch.clamp(adv_images - images.to(self.device), min=-self.epsilon, max=self.epsilon)
                adv_images = torch.clamp(images.to(self.device) + delta, min=0.0, max=1.0).detach()

        return adv_images

    def generate_dqf_attack(
        self,
        images: torch.Tensor,
        target_class_idx: int = 1,
        steps: Optional[int] = None,
        alpha: Optional[float] = None,
        k_fake: int = 15,
    ) -> torch.Tensor:
        """
        Decoder Query Fabrication (DQF) Attack:
        Targets the final Deformable Attention Decoder classification logits.
        Directly optimizes inactive queries into high-confidence phantom detections
        for class target_class_idx, exploiting RT-DETR's lack of NMS deduplication.
        """
        self.model.eval()
        adv_images = images.clone().detach().to(self.device)
        n_steps = steps if steps is not None else self.num_steps
        step_alpha = alpha if alpha is not None else self.alpha

        # Identify queries that are inactive on clean input
        with torch.no_grad():
            clean_out = self.model(pixel_values=images.to(self.device), return_dict=True)
            clean_logits = clean_out.logits  # [B, K, C]
            clean_probs = torch.sigmoid(clean_logits)
            clean_max_p, _ = clean_probs.max(dim=-1)
            inactive_mask = clean_max_p < 0.15

        for _ in range(n_steps):
            adv_images.requires_grad_(True)
            outputs = self.model(pixel_values=adv_images, return_dict=True)
            logits = outputs.logits  # [B, K, C]
            target_logits = logits[..., target_class_idx]  # [B, K]

            # Target inactive decoder queries
            masked_logits = target_logits.masked_fill(~inactive_mask, -1e4)
            top_target_logits, _ = torch.topk(masked_logits, k=min(k_fake, masked_logits.shape[1]), dim=-1)
            loss = -top_target_logits.mean()

            self.model.zero_grad()
            if adv_images.grad is not None:
                adv_images.grad.zero_()
            loss.backward()

            with torch.no_grad():
                grad_sign = adv_images.grad.sign() if adv_images.grad is not None else torch.zeros_like(adv_images)
                adv_images = adv_images - step_alpha * grad_sign
                delta = torch.clamp(adv_images - images.to(self.device), min=-self.epsilon, max=self.epsilon)
                adv_images = torch.clamp(images.to(self.device) + delta, min=0.0, max=1.0).detach()

        return adv_images

    @torch.no_grad()
    def evaluate_displacement_and_phantoms(
        self,
        clean_images: torch.Tensor,
        adv_images: torch.Tensor,
        target_class_idx: int = 1,
        conf_threshold: float = 0.35,
    ) -> Dict[str, float]:
        """
        Measures:
        1. Number of fabricated phantom boxes (target class conf >= conf_threshold).
        2. Top-300 Candidate Displacement Rate (% of clean top-300 anchors evicted).
        3. Mean confidence of injected phantoms.
        """
        self.model.eval()
        clean_out = self.model(pixel_values=clean_images.to(self.device), return_dict=True)
        adv_out = self.model(pixel_values=adv_images.to(self.device), return_dict=True)

        # 1. Check encoder selection displacement
        clean_enc_p = torch.sigmoid(clean_out.enc_outputs_class).max(dim=-1)[0]
        adv_enc_p = torch.sigmoid(adv_out.enc_outputs_class).max(dim=-1)[0]

        _, clean_indices = torch.topk(clean_enc_p, k=300, dim=-1)
        _, adv_indices = torch.topk(adv_enc_p, k=300, dim=-1)

        b_size = clean_images.shape[0]
        eviction_rates = []
        for b in range(b_size):
            clean_set = set(clean_indices[b].tolist())
            adv_set = set(adv_indices[b].tolist())
            overlap = len(clean_set.intersection(adv_set))
            evicted = 300 - overlap
            eviction_rates.append(evicted / 300.0)

        mean_eviction_rate = float(sum(eviction_rates) / len(eviction_rates))

        # 2. Check output phantom detections
        adv_logits = adv_out.logits
        adv_probs = torch.sigmoid(adv_logits)
        target_probs = adv_probs[..., target_class_idx]  # [B, K]

        phantom_counts = (target_probs >= conf_threshold).sum(dim=-1).float()
        mean_phantoms = float(phantom_counts.mean().item())
        asr_fab = float((phantom_counts >= 1).float().mean().item() * 100.0)

        return {
            "mean_phantoms": mean_phantoms,
            "asr_fab": asr_fab,
            "displacement_rate": mean_eviction_rate,
        }
