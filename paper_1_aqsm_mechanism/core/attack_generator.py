"""
Module: detr_attack_generator.py
Mục đích: Bộ sinh các đòn tấn công đối kháng chuyên biệt cho Transformer Object Detectors (RT-DETR).

Kiến trúc các đòn tấn công tinh gọn (Streamlined Attack Suite):
1. Tier 1 (Encoder-to-Decoder Interface):
   - Adversarial Query Selection Manipulation (AQSM / Selection Starvation Attack)
     Thao túng ranking của S candidate predictions từ Encoder (S = 8,400 ở native 640x640), ép object thật rơi khỏi Top-300,
     gây hiện tượng "Bỏ đói thông tin" (Information Starvation) ở Deformable Decoder.
2. Tier 2 (Deformable Decoder Output Space):
   - Query Vanishing Attack: Ép toàn bộ object queries triệt tiêu độ tự tin về mức 0 (No Detection / AEB Failure).
   - Query Fabrication Attack: Khai thác việc RT-DETR loại bỏ NMS (No-NMS design), kích hoạt queries rác
     gây ngập lụt false-positive boxes (TrackFlood / Phantom Braking).
3. Tier 3 (Input Localized Physical Threat):
   - Gradient Adversarial Patch Attack: Gắn patch đối kháng tối ưu hóa gradient (EoT),
     chứng minh điểm mù không gian (Spatial Blind Spot) của các bộ giám sát 1D toàn cục.
4. Tier 4 (Defense-Aware Adaptive Evasion):
   - White-box Adaptive Evasion PGD: Tấn công đa mục tiêu (Task Failure + Monitor Hinge Evasion),
     chứng minh giới hạn tất yếu của các bộ giám sát 1D trước kẻ tấn công có tri thức.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from paper_1_aqsm_mechanism.core.detr_wrapper import DETRModelWrapper, RTDETRModelWrapper


class DETRAttackGenerator:
    def __init__(
        self,
        model_wrapper: Union[DETRModelWrapper, RTDETRModelWrapper],
        epsilon: float = 16.0 / 255.0,
        alpha: float = 2.0 / 255.0,
        num_steps: int = 15,
    ) -> None:
        """
        Khởi tạo bộ sinh tấn công.

        Tham số:
            model_wrapper: Instance của RTDETRModelWrapper hoặc DETRModelWrapper.
            epsilon: Biên độ nhiễu L_infinity tối đa (chuẩn hóa trên [0, 1]).
            alpha: Kích thước bước nhảy mỗi vòng lặp PGD.
            num_steps: Số vòng lặp tối ưu hóa gradient.
        """
        self.wrapper = model_wrapper
        self.model = model_wrapper.model
        self.device = model_wrapper.device
        self.is_sigmoid = getattr(model_wrapper, "is_sigmoid", True)
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps

    @torch.enable_grad()
    def generate_vanishing_attack(
        self,
        images: torch.Tensor,
        steps: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Query Vanishing Attack: Triệt tiêu sự hiện diện của mọi đối tượng trong ảnh.
        Tối ưu hóa nhằm ép xác suất nhận diện của top-k queries về 0.
        """
        self.model.eval()
        adv_images = images.clone().detach().to(self.device)
        n_steps = steps if steps is not None else self.num_steps
        step_alpha = alpha if alpha is not None else self.alpha

        for _ in range(n_steps):
            adv_images.requires_grad_(True)
            outputs = self.model(pixel_values=adv_images, return_dict=True)
            logits = outputs.logits  # [B, N, C]

            if self.is_sigmoid:
                probs = torch.sigmoid(logits)  # [B, N, C]
                max_probs, _ = probs.max(dim=-1)  # [B, N]
                k = min(10, max_probs.shape[1])
                top_probs, _ = torch.topk(max_probs, k=k, dim=-1)
                loss = (top_probs ** 2).mean()
            else:
                probs = F.softmax(logits, dim=-1)
                max_fg_probs, _ = probs[..., :-1].max(dim=-1)
                k = min(10, max_fg_probs.shape[1])
                top_probs, _ = torch.topk(max_fg_probs, k=k, dim=-1)
                loss = (top_probs ** 2).mean()

            self.model.zero_grad()
            if adv_images.grad is not None:
                adv_images.grad.zero_()
            loss.backward()

            with torch.no_grad():
                grad_sign = adv_images.grad.sign() if adv_images.grad is not None else torch.zeros_like(adv_images)
                adv_images = adv_images - step_alpha * grad_sign
                delta = torch.clamp(adv_images - images, min=-self.epsilon, max=self.epsilon)
                adv_images = torch.clamp(images + delta, min=0.0, max=1.0).detach()

        return adv_images

    @torch.enable_grad()
    def generate_query_selection_attack(
        self,
        images: torch.Tensor,
        steps: Optional[int] = None,
        alpha: Optional[float] = None,
        margin: float = 0.20,
        lambda_rank: float = 1.0,
        fg_selection_mode: str = "clean_prediction_proxy",
        gt_boxes: Optional[Sequence[np.ndarray]] = None,
    ) -> torch.Tensor:
        """
        Adversarial Query Selection Manipulation (AQSM / Selection Starvation Attack):
        Tấn công có chủ đích vào mắt xích Encoder Query Selection (đặc thù độc nhất của RT-DETR).
        Thao túng thứ hạng (ranking) của S candidate predictions từ Hybrid Encoder (S = 8,400 ở native 640x640; 2,100 ở 320x320):
        1. Dìm điểm các candidate thuộc về vật thể thật (foreground candidates) để chúng rơi khỏi Top-300.
        2. Kích hoạt và nâng điểm các candidate thuộc vùng nền (background candidates) thế chỗ vào Top-300.
        3. Gây ra hiện tượng "Bỏ đói thông tin" (Information Starvation) ở Deformable Decoder:
           Reference points nạp vào Decoder bị đặt vào khoảng trống rác, khiến Decoder hoàn toàn mất dấu vật thể thật.

        Tham số fg_selection_mode:
            - 'clean_prediction_proxy': (Mặc định chuẩn thực tế test-time)
              Kẻ tấn công không có nhãn ground-truth người gán, sử dụng Top-15 queries có độ tự tin cao nhất
              của mô hình trên ảnh sạch làm proxy đại diện cho vật thể.
            - 'ground_truth_matched': (Nhánh oracle / đối chứng nghiên cứu)
              Sử dụng bounding box thật COCO để gán candidate theo spatial anchor.
        """
        self.model.eval()
        adv_images = images.clone().detach().to(self.device)
        n_steps = steps if steps is not None else self.num_steps
        step_alpha = alpha if alpha is not None else self.alpha

        # Bước 1: Xác định tập candidate đại diện cho vật thể thật
        with torch.no_grad():
            clean_out = self.model(pixel_values=images.to(self.device), return_dict=True)
            if hasattr(clean_out, "enc_outputs_class") and clean_out.enc_outputs_class is not None:
                clean_enc_logits = clean_out.enc_outputs_class  # [B, S, C], S = 8,400 at native 640x640
            else:
                clean_enc_logits = clean_out.logits

            # Tính điểm tối đa trên các lớp cho từng spatial candidate
            clean_enc_probs = torch.sigmoid(clean_enc_logits)  # [B, S, C]
            max_clean_probs, _ = clean_enc_probs.max(dim=-1)  # [B, S]
            B, S = max_clean_probs.shape
            k_fg = min(15, S)

            if fg_selection_mode == "ground_truth_matched" and gt_boxes is not None:
                from paper_1_aqsm_mechanism.core.mechanism_analyzer import compute_token_fg_masks
                anchors_sig = None
                if hasattr(clean_out, "enc_outputs_coord_logits") and clean_out.enc_outputs_coord_logits is not None:
                    anchors_sig = torch.sigmoid(clean_out.enc_outputs_coord_logits[0]).cpu().numpy()
                elif hasattr(clean_out, "pred_boxes"):
                    anchors_sig = clean_out.pred_boxes[0].cpu().numpy()

                topk_fg_list = []
                for b in range(B):
                    gt_b = gt_boxes[b] if b < len(gt_boxes) else np.zeros((0, 4), dtype=np.float32)
                    if len(gt_b) > 0 and anchors_sig is not None:
                        fg_mask = compute_token_fg_masks(anchors_sig, gt_b)
                        fg_idx = np.where(fg_mask)[0]
                        if len(fg_idx) >= k_fg:
                            topk_fg_list.append(torch.tensor(fg_idx[:k_fg], device=self.device))
                        elif len(fg_idx) > 0:
                            pad = np.random.choice(fg_idx, size=k_fg - len(fg_idx), replace=True)
                            topk_fg_list.append(torch.tensor(np.concatenate([fg_idx, pad]), device=self.device))
                        else:
                            _, topk_b = torch.topk(max_clean_probs[b], k=k_fg)
                            topk_fg_list.append(topk_b)
                    else:
                        _, topk_b = torch.topk(max_clean_probs[b], k=k_fg)
                        topk_fg_list.append(topk_b)
                topk_fg_idx = torch.stack(topk_fg_list, dim=0)
            else:
                # Mặc định: Unsupervised Clean-Prediction Proxy
                _, topk_fg_idx = torch.topk(max_clean_probs, k=k_fg, dim=-1)  # [B, k_fg]

            # Chọn background candidates (các candidates xếp hạng thấp ngoài top 500)
            k_bg = min(30, S)
            _, bottom_bg_idx = torch.topk(-max_clean_probs, k=k_bg, dim=-1)  # [B, k_bg]

        # Bước 2: Tối ưu hóa PGD thao túng ranking và dìm điểm
        for _ in range(n_steps):
            adv_images.requires_grad_(True)
            outputs = self.model(pixel_values=adv_images, return_dict=True)
            
            if hasattr(outputs, "enc_outputs_class") and outputs.enc_outputs_class is not None:
                enc_logits = outputs.enc_outputs_class  # [B, S, C]
            else:
                enc_logits = outputs.logits

            enc_probs = torch.sigmoid(enc_logits)  # [B, S, C]

            # 1. Suppression Loss: Ép điểm của real object candidates về 0
            fg_probs = torch.gather(enc_probs, 1, topk_fg_idx.unsqueeze(-1).expand(-1, -1, enc_probs.shape[-1]))
            max_fg_probs, _ = fg_probs.max(dim=-1)  # [B, k_fg]
            loss_suppress = (max_fg_probs ** 2).mean()

            # 2. Ngưỡng điểm của ứng viên thứ 300 trong toàn bộ S candidates (8,400 tại 640x640)
            max_all_probs, _ = enc_probs.max(dim=-1)  # [B, S]
            val_at_300, _ = torch.kthvalue(max_all_probs, k=max_all_probs.shape[1] - 300 + 1, dim=-1)  # [B]
            
            # Ranking Margin Loss: Ép điểm real object candidate phải nhỏ hơn ngưỡng top-300 một biên độ margin
            loss_rank = F.relu(max_fg_probs - val_at_300.unsqueeze(-1) + margin).mean()

            # 3. Đẩy điểm background candidates lên để chiếm chỗ trong Top-300
            bg_probs = torch.gather(enc_probs, 1, bottom_bg_idx.unsqueeze(-1).expand(-1, -1, enc_probs.shape[-1]))
            max_bg_probs, _ = bg_probs.max(dim=-1)  # [B, k_bg]
            loss_bg_boost = -(max_bg_probs).mean()

            total_loss = loss_suppress + lambda_rank * loss_rank + 0.5 * loss_bg_boost

            self.model.zero_grad()
            if adv_images.grad is not None:
                adv_images.grad.zero_()
            total_loss.backward()

            with torch.no_grad():
                grad_sign = adv_images.grad.sign() if adv_images.grad is not None else torch.zeros_like(adv_images)
                adv_images = adv_images - step_alpha * grad_sign
                delta = torch.clamp(adv_images - images, min=-self.epsilon, max=self.epsilon)
                adv_images = torch.clamp(images + delta, min=0.0, max=1.0).detach()

        return adv_images

    @torch.enable_grad()
    def generate_fabrication_attack(
        self,
        images: torch.Tensor,
        target_class_idx: int = 1,
        steps: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Query Fabrication Attack: Khai thác việc RT-DETR loại bỏ NMS.
        Ép các background queries tạo ra dự đoán giả mạo có độ tự tin cao cho lớp mục tiêu.
        """
        self.model.eval()
        adv_images = images.clone().detach().to(self.device)
        n_steps = steps if steps is not None else self.num_steps
        step_alpha = alpha if alpha is not None else self.alpha

        for _ in range(n_steps):
            adv_images.requires_grad_(True)
            outputs = self.model(pixel_values=adv_images, return_dict=True)
            logits = outputs.logits  # [B, N, C]

            target_logits = logits[..., target_class_idx]  # [B, N]
            k = min(5, target_logits.shape[1])
            top_target_logits, _ = torch.topk(target_logits, k=k, dim=-1)
            loss = -top_target_logits.mean()

            self.model.zero_grad()
            if adv_images.grad is not None:
                adv_images.grad.zero_()
            loss.backward()

            with torch.no_grad():
                grad_sign = adv_images.grad.sign() if adv_images.grad is not None else torch.zeros_like(adv_images)
                adv_images = adv_images - step_alpha * grad_sign
                delta = torch.clamp(adv_images - images, min=-self.epsilon, max=self.epsilon)
                adv_images = torch.clamp(images + delta, min=0.0, max=1.0).detach()

        return adv_images

    def generate_random_noise(
        self,
        images: torch.Tensor,
        noise_type: str = "gaussian",
    ) -> torch.Tensor:
        """
        Nhóm đối chứng (Random Noise Control): Thêm nhiễu ngẫu nhiên không có gradient.
        """
        device = images.device
        if noise_type == "gaussian":
            noise = torch.randn_like(images, device=device) * (self.epsilon / 2.0)
            noise = torch.clamp(noise, -self.epsilon, self.epsilon)
        else:
            noise = (torch.rand_like(images, device=device) * 2.0 - 1.0) * self.epsilon

        return torch.clamp(images + noise, 0.0, 1.0)

    def generate_random_patch(
        self,
        images: torch.Tensor,
        patch_size: int = 48,
        patch_loc: Any = "random",
        seed: int = 42,
    ) -> torch.Tensor:
        """
        Nhóm đối chứng (Random Patch Control): Gắn một patch ngẫu nhiên không được tối ưu gradient.
        """
        patched_images = images.clone().detach().to(self.device)
        B, C, H, W = patched_images.shape
        rng = np.random.RandomState(seed)

        for b in range(B):
            if patch_loc == "random":
                max_y = max(0, H - patch_size)
                max_x = max(0, W - patch_size)
                py = int(rng.randint(0, max_y + 1)) if max_y > 0 else 0
                px = int(rng.randint(0, max_x + 1)) if max_x > 0 else 0
            else:
                py, px = patch_loc
                py = max(0, min(py, H - patch_size))
                px = max(0, min(px, W - patch_size))

            rand_patch = torch.rand((C, patch_size, patch_size), device=self.device)
            patched_images[b, :, py : py + patch_size, px : px + patch_size] = rand_patch

        return patched_images

    @torch.enable_grad()
    def optimize_adversarial_patch(
        self,
        calibration_images: torch.Tensor,
        patch_size: int = 48,
        patch_loc: Any = "random",
        target_class_idx: int = 1,
        num_epochs: int = 20,
        lr: float = 0.05,
    ) -> torch.Tensor:
        """
        Huấn luyện Patch Đối kháng thực thụ bằng Gradient Ascent (Gradient-Optimized Adversarial Patch).
        Áp dụng Expectation over Transformation (EoT) dán ngẫu nhiên trên tập calibration.
        """
        self.model.eval()
        B_cal, C, H, W = calibration_images.shape

        patch = torch.full((1, C, patch_size, patch_size), 0.5, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([patch], lr=lr)

        batch_size = min(16, B_cal)
        num_batches = max(1, B_cal // batch_size)
        rng = np.random.RandomState(42)

        for epoch in range(num_epochs):
            perm = torch.randperm(B_cal)
            for b_idx in range(num_batches):
                indices = perm[b_idx * batch_size : (b_idx + 1) * batch_size]
                batch_imgs = calibration_images[indices].clone().to(self.device)
                cur_B = len(indices)

                for b in range(cur_B):
                    if patch_loc == "random":
                        max_y = max(0, H - patch_size)
                        max_x = max(0, W - patch_size)
                        py = int(rng.randint(0, max_y + 1)) if max_y > 0 else 0
                        px = int(rng.randint(0, max_x + 1)) if max_x > 0 else 0
                    else:
                        py, px = patch_loc
                        py = max(0, min(py, H - patch_size))
                        px = max(0, min(px, W - patch_size))

                    batch_imgs[b, :, py : py + patch_size, px : px + patch_size] = patch[0]

                outputs = self.model(pixel_values=batch_imgs, return_dict=True)
                logits = outputs.logits  # [B, N, C_out]

                target_prob = torch.sigmoid(logits[..., target_class_idx])
                k_top = min(3, target_prob.shape[1])
                top_probs, _ = torch.topk(target_prob, k=k_top, dim=-1)
                loss = -torch.log(top_probs + 1e-6).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    patch.clamp_(0.0, 1.0)

        return patch.detach()

    def apply_adversarial_patch(
        self,
        images: torch.Tensor,
        patch: torch.Tensor,
        patch_loc: Any = "random",
        seed: int = 42,
    ) -> torch.Tensor:
        """
        Dán patch đối kháng đã tối ưu hóa lên tập ảnh đầu vào.
        """
        patched_images = images.clone().detach().to(self.device)
        B, C, H, W = patched_images.shape
        _, _, py_size, px_size = patch.shape
        rng = np.random.RandomState(seed)

        for b in range(B):
            if patch_loc == "random":
                max_y = max(0, H - py_size)
                max_x = max(0, W - px_size)
                py = int(rng.randint(0, max_y + 1)) if max_y > 0 else 0
                px = int(rng.randint(0, max_x + 1)) if max_x > 0 else 0
            else:
                py, px = patch_loc
                py = max(0, min(py, H - py_size))
                px = max(0, min(px, W - px_size))

            patched_images[b, :, py : py + py_size, px : px + px_size] = patch[0]

        return patched_images

    def generate_clean_shifts(
        self,
        images: torch.Tensor,
        shift_type: str = "blur",
    ) -> torch.Tensor:
        """
        Tạo biến dạng tự nhiên (Clean-shift Control) kiểm tra False Positive Rate của Monitor:
        - blur: Gaussian Blur 5x5 sigma=1.5
        - brightness: Giảm sáng 30%
        """
        shifted = images.clone().detach().to(self.device)
        if shift_type == "blur":
            kernel_size = 5
            sigma = 1.5
            x = torch.arange(kernel_size, device=self.device, dtype=torch.float32) - (kernel_size - 1) / 2.0
            gauss_1d = torch.exp(-0.5 * (x / sigma) ** 2)
            gauss_2d = gauss_1d[:, None] * gauss_1d[None, :]
            gauss_2d = gauss_2d / gauss_2d.sum()
            kernel = gauss_2d.expand(3, 1, kernel_size, kernel_size)
            shifted = F.conv2d(shifted, kernel, padding=kernel_size // 2, groups=3)
        elif shift_type == "brightness":
            shifted = shifted * 0.70
        return torch.clamp(shifted, 0.0, 1.0)

    @torch.enable_grad()
    def generate_adaptive_attack(
        self,
        images: torch.Tensor,
        target_class_idx: int = 1,
        evasion_weight: float = 0.5,
        steps: Optional[int] = None,
        alpha: Optional[float] = None,
        loss_mode: str = "hinge",
        target_detector: str = "1d",
        num_target_queries: int = 5,
        tau_bounds_act: Optional[Tuple[float, float]] = None,
        tau_bounds_top1: Optional[Tuple[float, float]] = None,
        clean_act_mean: float = 0.08,
        clean_act_std: float = 0.04,
        clean_top1_mean: float = 0.12,
        clean_top1_std: float = 0.05,
        temperature: float = 25.0,
    ) -> torch.Tensor:
        """
        White-box Adaptive Evasion PGD Attack:
        Kẻ tấn công biết rõ hệ thống phòng thủ:
        - target_detector="1d": Né tránh bộ đếm 1D (active_query_ratio, query_top1_mean).
        - target_detector="joint": Né tránh đồng thời bộ đếm 1D và bộ giám sát biểu diễn sâu (Layer Mahalanobis)
          bằng cách phạt khoảng cách biến dạng hidden states của Deformable Decoder.

        Tối ưu hóa đa mục tiêu:
        Loss = Loss_Task + evasion_weight * Loss_Monitor_Evasion
        """
        self.model.eval()
        adv_images = images.clone().detach().to(self.device)
        n_steps = steps if steps is not None else self.num_steps
        step_alpha = alpha if alpha is not None else self.alpha

        # Precompute clean reference hidden states nếu tấn công joint/maha
        clean_h_refs = None
        if target_detector in ["joint", "maha_aware"] and evasion_weight > 0.0:
            with torch.no_grad():
                clean_out = self.model(pixel_values=images.to(self.device), output_hidden_states=True, return_dict=True)
                clean_h_refs = [h.detach() for h in clean_out.decoder_hidden_states]

        # Nạp tham số hiệu chuẩn thực tế nếu chưa được truyền tường minh
        if tau_bounds_act is None or tau_bounds_top1 is None:
            try:
                from paper_1_aqsm_mechanism.core.adaptive_calibrator import get_or_create_calibrated_params
                cal_p = get_or_create_calibrated_params(self.wrapper)
                clean_act_mean = cal_p.get("clean_act_mean", clean_act_mean)
                clean_act_std = cal_p.get("clean_act_std", clean_act_std)
                clean_top1_mean = cal_p.get("clean_top1_mean", clean_top1_mean)
                clean_top1_std = cal_p.get("clean_top1_std", clean_top1_std)
                if tau_bounds_act is None:
                    tau_bounds_act = tuple(cal_p.get("tau_bounds_act", (0.01, 0.16)))
                if tau_bounds_top1 is None:
                    tau_bounds_top1 = tuple(cal_p.get("tau_bounds_top1", (0.05, 0.22)))
            except Exception:
                pass

        if tau_bounds_act is None:
            tau_bounds_act = (max(0.0, clean_act_mean - 2.0 * clean_act_std), clean_act_mean + 2.0 * clean_act_std)
        if tau_bounds_top1 is None:
            tau_bounds_top1 = (max(0.0, clean_top1_mean - 2.0 * clean_top1_std), clean_top1_mean + 2.0 * clean_top1_std)

        tau_act_min, tau_act_max = tau_bounds_act
        tau_top_min, tau_top_max = tau_bounds_top1

        for _ in range(n_steps):
            adv_images.requires_grad_(True)
            need_hidden = (target_detector in ["joint", "maha_aware"] and evasion_weight > 0.0)
            outputs = self.model(pixel_values=adv_images, output_hidden_states=need_hidden, return_dict=True)
            logits = outputs.logits  # [B, N, C]
            probs = torch.sigmoid(logits)  # [B, N, C]

            # 1. Mục tiêu Tác vụ: Ép top-k queries nhạy cảm nhất nhận diện lớp target
            target_logits = logits[..., target_class_idx]  # [B, N]
            k = min(num_target_queries, target_logits.shape[1])
            top_target_logits, _ = torch.topk(target_logits, k=k, dim=-1)
            task_loss = -top_target_logits.mean()

            # 2. Mục tiêu Né tránh Monitor (Evasion Loss)
            if evasion_weight > 0.0:
                max_p, _ = probs.max(dim=-1)  # [B, N]
                top1_mean_batch = max_p.mean(dim=-1)  # [B]
                soft_active_ratio = torch.sigmoid(temperature * (max_p - 0.35)).mean(dim=-1)  # [B]

                if loss_mode == "hinge":
                    pen_act = F.relu(tau_act_min - soft_active_ratio) + F.relu(soft_active_ratio - tau_act_max)
                    pen_top1 = F.relu(tau_top_min - top1_mean_batch) + F.relu(top1_mean_batch - tau_top_max)
                    evasion_1d = ((pen_act / max(1e-4, clean_act_std)) ** 2 + (pen_top1 / max(1e-4, clean_top1_std)) ** 2).mean()
                else:
                    z_act = (soft_active_ratio - clean_act_mean) / max(1e-4, clean_act_std)
                    z_top1 = (top1_mean_batch - clean_top1_mean) / max(1e-4, clean_top1_std)
                    evasion_1d = (z_act ** 2 + z_top1 ** 2).mean()

                if target_detector in ["joint", "maha_aware"] and clean_h_refs is not None:
                    # Phạt độ lệch L2 trên hidden states của Decoder
                    maha_surrogate = 0.0
                    for l_idx, h_cur in enumerate(outputs.decoder_hidden_states):
                        h_ref = clean_h_refs[l_idx]
                        maha_surrogate = maha_surrogate + F.mse_loss(h_cur, h_ref)
                    evasion_loss = evasion_1d + 10.0 * maha_surrogate
                else:
                    evasion_loss = evasion_1d

                total_loss = task_loss + evasion_weight * evasion_loss
            else:
                total_loss = task_loss

            self.model.zero_grad()
            if adv_images.grad is not None:
                adv_images.grad.zero_()
            total_loss.backward()

            with torch.no_grad():
                grad_sign = adv_images.grad.sign() if adv_images.grad is not None else torch.zeros_like(adv_images)
                adv_images = adv_images - step_alpha * grad_sign
                delta = torch.clamp(adv_images - images, min=-self.epsilon, max=self.epsilon)
                adv_images = torch.clamp(images + delta, min=0.0, max=1.0).detach()

        return adv_images

    @torch.enable_grad()
    def generate_encoder_fabrication_attack(
        self,
        images: torch.Tensor,
        target_class_idx: int = 1,
        steps: Optional[int] = None,
        alpha: Optional[float] = None,
        k_fake: int = 20,
        margin: float = 0.25,
        lambda_rank: float = 0.5,
    ) -> torch.Tensor:
        """
        Encoder Query Fabrication (EQF) Attack:
        Nham vao Encoder Selection Head (model.enc_outputs_class).
        Bien cac vi tri nen trong o Encoder thanh ung vien gia mao cho lop muc tieu (target_class_idx),
        ep chung lot vao Top-300 Query Selection ma khong can lan truyen nguoc qua Decoder.
        """
        self.model.eval()
        adv_images = images.clone().detach().to(self.device)
        n_steps = steps if steps is not None else self.num_steps
        step_alpha = alpha if alpha is not None else self.alpha

        with torch.no_grad():
            clean_out = self.model(pixel_values=images.to(self.device), return_dict=True)
            clean_enc_logits = clean_out.enc_outputs_class
            clean_enc_probs = torch.sigmoid(clean_enc_logits)
            clean_max_probs, _ = clean_enc_probs.max(dim=-1)
            top300_clean_val, _ = torch.topk(clean_max_probs, k=min(300, clean_max_probs.shape[1]), dim=-1)
            tau_300_clean = top300_clean_val[:, -1]

            bg_mask = (clean_max_probs < 0.10)

        for _ in range(n_steps):
            adv_images.requires_grad_(True)
            outputs = self.model(pixel_values=adv_images, return_dict=True)
            enc_logits = outputs.enc_outputs_class
            target_logits = enc_logits[..., target_class_idx]

            masked_target_logits = target_logits.masked_fill(~bg_mask, -1e4)
            top_fake_logits, _ = torch.topk(masked_target_logits, k=k_fake, dim=-1)

            loss_fake = -top_fake_logits.mean()
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
                delta = torch.clamp(adv_images - images, min=-self.epsilon, max=self.epsilon)
                adv_images = torch.clamp(images + delta, min=0.0, max=1.0).detach()

        return adv_images

    @torch.enable_grad()
    def generate_adaptive_esm_attack(
        self,
        images: torch.Tensor,
        steps: Optional[int] = None,
        alpha: Optional[float] = None,
        margin: float = 0.20,
        lambda_rank: float = 1.0,
        lambda_esm: float = 2.0,
        clean_dropoff_bounds: Tuple[float, float] = (12.0, 140.0),
        clean_margin_bounds: Tuple[float, float] = (0.015, 0.115),
    ) -> torch.Tensor:
        """
        Adaptive ESM-Aware Evasion Attack:
        Kẻ tấn công thực hiện thao túng tuyển chọn Query (AQSM) nhưng biết rõ hệ thống
        giám sát phân vị Dropoff và Margin tại cửa ngõ Encoder (Encoder Selection Monitor - ESM).
        Tối ưu hóa đa mục tiêu:
        Loss = Loss_AQSM + lambda_esm * Loss_ESM_Evasion
        Hàm loss né tránh ESM hoàn toàn khả vi, ép tỷ số dropoff s_{(1)} / s_{(300)} và
        chênh lệch margin s_{top300} - s_{bg} nằm trong biên độ an toàn của ảnh sạch.
        """
        self.model.eval()
        adv_images = images.clone().detach().to(self.device)
        n_steps = steps if steps is not None else self.num_steps
        step_alpha = alpha if alpha is not None else self.alpha

        # Xác định foreground và background candidates trên ảnh sạch
        with torch.no_grad():
            clean_out = self.model(pixel_values=images.to(self.device), return_dict=True)
            if hasattr(clean_out, "enc_outputs_class") and clean_out.enc_outputs_class is not None:
                clean_enc_logits = clean_out.enc_outputs_class
            else:
                clean_enc_logits = clean_out.logits

            clean_enc_probs = torch.sigmoid(clean_enc_logits)
            max_clean_probs, _ = clean_enc_probs.max(dim=-1)

            k_fg = min(15, max_clean_probs.shape[1])
            _, topk_fg_idx = torch.topk(max_clean_probs, k=k_fg, dim=-1)

            k_bg = min(30, max_clean_probs.shape[1])
            _, bottom_bg_idx = torch.topk(-max_clean_probs, k=k_bg, dim=-1)

        # Nạp tham số hiệu chuẩn thực tế nếu có sẵn
        try:
            from paper_1_aqsm_mechanism.core.adaptive_calibrator import get_or_create_calibrated_params
            cal_p = get_or_create_calibrated_params(self.wrapper)
            clean_dropoff_bounds = tuple(cal_p.get("clean_dropoff_bounds", clean_dropoff_bounds))
            clean_margin_bounds = tuple(cal_p.get("clean_margin_bounds", clean_margin_bounds))
        except Exception:
            pass

        tau_drop_min, tau_drop_max = clean_dropoff_bounds
        tau_marg_min, tau_marg_max = clean_margin_bounds

        for _ in range(n_steps):
            adv_images.requires_grad_(True)
            outputs = self.model(pixel_values=adv_images, return_dict=True)

            if hasattr(outputs, "enc_outputs_class") and outputs.enc_outputs_class is not None:
                enc_logits = outputs.enc_outputs_class
            else:
                enc_logits = outputs.logits

            enc_probs = torch.sigmoid(enc_logits)
            max_all_probs, _ = enc_probs.max(dim=-1)  # [B, S]

            # 1. AQSM Task Loss: Dìm foreground và nâng background
            fg_probs = torch.gather(enc_probs, 1, topk_fg_idx.unsqueeze(-1).expand(-1, -1, enc_probs.shape[-1]))
            max_fg_probs, _ = fg_probs.max(dim=-1)
            loss_suppress = (max_fg_probs ** 2).mean()

            val_at_300, _ = torch.kthvalue(max_all_probs, k=max_all_probs.shape[1] - 300 + 1, dim=-1)
            loss_rank = F.relu(max_fg_probs - val_at_300.unsqueeze(-1) + margin).mean()

            bg_probs = torch.gather(enc_probs, 1, bottom_bg_idx.unsqueeze(-1).expand(-1, -1, enc_probs.shape[-1]))
            max_bg_probs, _ = bg_probs.max(dim=-1)
            loss_bg_boost = -(max_bg_probs).mean()

            task_loss = loss_suppress + lambda_rank * loss_rank + 0.5 * loss_bg_boost

            # 2. ESM Evasion Loss: Giữ dropoff và margin trong vùng sạch chuẩn
            topk_vals, _ = torch.topk(max_all_probs, k=min(300, max_all_probs.shape[1]), dim=-1)
            top1_val = topk_vals[:, 0]
            top300_val = topk_vals[:, -1]
            diff_dropoff = top1_val / (top300_val.clamp(min=1e-4) + 1e-4)

            top300_mean = topk_vals.mean(dim=-1)
            all_sum = max_all_probs.sum(dim=-1)
            topk_sum = topk_vals.sum(dim=-1)
            bg_mean = (all_sum - topk_sum) / float(max(1, max_all_probs.shape[1] - 300))
            diff_margin = top300_mean - bg_mean

            pen_drop = F.relu(tau_drop_min - diff_dropoff) + F.relu(diff_dropoff - tau_drop_max)
            pen_marg = F.relu(tau_marg_min - diff_margin) + F.relu(diff_margin - tau_marg_max)
            loss_esm = (pen_drop ** 2 + 10.0 * (pen_marg ** 2)).mean()

            total_loss = task_loss + lambda_esm * loss_esm

            self.model.zero_grad()
            if adv_images.grad is not None:
                adv_images.grad.zero_()
            total_loss.backward()

            with torch.no_grad():
                grad_sign = adv_images.grad.sign() if adv_images.grad is not None else torch.zeros_like(adv_images)
                adv_images = adv_images - step_alpha * grad_sign
                delta = torch.clamp(adv_images - images, min=-self.epsilon, max=self.epsilon)
                adv_images = torch.clamp(images + delta, min=0.0, max=1.0).detach()

        return adv_images

    @torch.enable_grad()
    def generate_adaptive_fused_attack(
        self,
        images: torch.Tensor,
        target_class_idx: int = 1,
        steps: Optional[int] = None,
        alpha: Optional[float] = None,
        evasion_weight: float = 1.0,
        tau_bounds_act: Optional[Tuple[float, float]] = None,
        tau_bounds_top1: Optional[Tuple[float, float]] = None,
        clean_dropoff_bounds: Tuple[float, float] = (12.0, 140.0),
    ) -> torch.Tensor:
        """
        Comprehensive Adaptive Fused Ensemble Attack:
        Kẻ tấn công am hiểu toàn bộ kiến trúc phòng thủ Calibrated Max-OR Ensemble:
        Đồng thời né tránh:
        - Tier 0: 1D Continuous Active Query Activation & Top1 Confidence
        - Tier 1: ESM Selection Dropoff
        - Tier 2: Layer-wise Decoder Mahalanobis Embeddings
        Tối ưu hóa liên hợp nhằm đánh sập toàn bộ 4 kênh phòng thủ.
        """
        self.model.eval()
        adv_images = images.clone().detach().to(self.device)
        n_steps = steps if steps is not None else self.num_steps
        step_alpha = alpha if alpha is not None else self.alpha

        with torch.no_grad():
            clean_out = self.model(pixel_values=images.to(self.device), output_hidden_states=True, return_dict=True)
            clean_h_refs = [h.detach() for h in clean_out.decoder_hidden_states]

        try:
            from paper_1_aqsm_mechanism.core.adaptive_calibrator import get_or_create_calibrated_params
            cal_p = get_or_create_calibrated_params(self.wrapper)
            if tau_bounds_act is None:
                tau_bounds_act = tuple(cal_p.get("tau_bounds_act", (0.01, 0.08)))
            if tau_bounds_top1 is None:
                tau_bounds_top1 = tuple(cal_p.get("tau_bounds_top1", (0.06, 0.20)))
            clean_dropoff_bounds = tuple(cal_p.get("clean_dropoff_bounds", clean_dropoff_bounds))
        except Exception:
            if tau_bounds_act is None:
                tau_bounds_act = (0.01, 0.08)
            if tau_bounds_top1 is None:
                tau_bounds_top1 = (0.06, 0.20)

        tau_act_min, tau_act_max = tau_bounds_act
        tau_top_min, tau_top_max = tau_bounds_top1
        tau_drop_min, tau_drop_max = clean_dropoff_bounds

        for _ in range(n_steps):
            adv_images.requires_grad_(True)
            outputs = self.model(pixel_values=adv_images, output_hidden_states=True, return_dict=True)
            logits = outputs.logits
            probs = torch.sigmoid(logits)

            # 1. Task Objective (Fabrication of target class)
            target_logits = logits[..., target_class_idx]
            k = min(5, target_logits.shape[1])
            top_target_logits, _ = torch.topk(target_logits, k=k, dim=-1)
            task_loss = -top_target_logits.mean()

            # 2. Tier 0 Evasion (Soft Active Ratio & Top1 Mean)
            max_p, _ = probs.max(dim=-1)
            top1_mean = max_p.mean(dim=-1)
            soft_act = torch.sigmoid(20.0 * (max_p - 0.35)).mean(dim=-1)

            pen_act = F.relu(tau_act_min - soft_act) + F.relu(soft_act - tau_act_max)
            pen_top = F.relu(tau_top_min - top1_mean) + F.relu(top1_mean - tau_top_max)
            loss_t0 = (pen_act ** 2 + pen_top ** 2).mean()

            # 3. Tier 1 Evasion (ESM Dropoff)
            if hasattr(outputs, "enc_outputs_class") and outputs.enc_outputs_class is not None:
                enc_p = torch.sigmoid(outputs.enc_outputs_class)
                max_enc_p, _ = enc_p.max(dim=-1)
                topk_enc, _ = torch.topk(max_enc_p, k=min(300, max_enc_p.shape[1]), dim=-1)
                diff_drop = topk_enc[:, 0] / (topk_enc[:, -1].clamp(min=1e-4) + 1e-4)
                loss_t1 = (F.relu(tau_drop_min - diff_drop) + F.relu(diff_drop - tau_drop_max)).pow(2).mean()
            else:
                loss_t1 = 0.0

            # 4. Tier 2 Evasion (Decoder Manifold L2 Surrogate)
            loss_t2 = 0.0
            for l_idx, h_cur in enumerate(outputs.decoder_hidden_states):
                loss_t2 = loss_t2 + F.mse_loss(h_cur, clean_h_refs[l_idx])

            total_loss = task_loss + evasion_weight * (loss_t0 + 0.5 * loss_t1 + 5.0 * loss_t2)

            self.model.zero_grad()
            if adv_images.grad is not None:
                adv_images.grad.zero_()
            total_loss.backward()

            with torch.no_grad():
                grad_sign = adv_images.grad.sign() if adv_images.grad is not None else torch.zeros_like(adv_images)
                adv_images = adv_images - step_alpha * grad_sign
                delta = torch.clamp(adv_images - images, min=-self.epsilon, max=self.epsilon)
                adv_images = torch.clamp(images + delta, min=0.0, max=1.0).detach()

        return adv_images


    @torch.enable_grad()
    def generate_standard_task_pgd(
        self,
        images: torch.Tensor,
        hf_labels: Optional[List[Dict[str, torch.Tensor]]] = None,
        steps: Optional[int] = None,
        alpha: Optional[float] = None,
        fallback_conf_thresh: float = 0.30,
    ) -> torch.Tensor:
        """
        Standard End-to-End Task PGD Baseline (Mục 6):
        Tối ưu hóa PGD L_inf trực tiếp trên hàm loss phát hiện chuẩn của RT-DETR
        (Hungarian Bipartite Matching + Varifocal Loss + L1 BBox + GIoU) khi có nhãn COCO thật;
        tự động chuyển sang pseudo-target nếu ảnh không có vật thể GT hoặc chưa truyền nhãn.
        """
        self.model.eval()
        orig = images.clone().detach().to(self.device)
        adv_images = orig.clone().detach()
        n_steps = steps if steps is not None else self.num_steps
        step_alpha = alpha if alpha is not None else self.alpha
        bsz = orig.shape[0]

        need_fallback = [
            b for b in range(bsz)
            if hf_labels is None or len(hf_labels[b]["boxes"]) == 0
        ]
        fallback_masks: Dict[int, torch.Tensor] = {}
        fallback_labels: Dict[int, torch.Tensor] = {}
        fallback_boxes: Dict[int, torch.Tensor] = {}
        if need_fallback:
            with torch.no_grad():
                clean_out = self.model(pixel_values=orig, return_dict=True)
                clean_probs = torch.sigmoid(clean_out.logits)
                for b in need_fallback:
                    scores_b, labels_b = clean_probs[b].max(dim=-1)
                    mask_b = scores_b >= fallback_conf_thresh
                    if not mask_b.any():
                        _, top_idx = torch.topk(scores_b, k=min(10, scores_b.shape[0]))
                        mask_b = torch.zeros_like(scores_b, dtype=torch.bool)
                        mask_b[top_idx] = True
                    fallback_masks[b] = mask_b
                    fallback_labels[b] = labels_b[mask_b]
                    fallback_boxes[b] = clean_out.pred_boxes[b][mask_b].detach()

        valid_indices = [
            b for b in range(bsz)
            if hf_labels is not None and len(hf_labels[b]["boxes"]) > 0
        ]

        for _ in range(n_steps):
            adv_images.requires_grad_(True)
            total_loss = torch.zeros((), device=self.device)

            if valid_indices:
                sub_x = adv_images[valid_indices]
                sub_lbls = [
                    {
                        "class_labels": hf_labels[b]["class_labels"].to(self.device),
                        "boxes": hf_labels[b]["boxes"].to(self.device),
                    }
                    for b in valid_indices
                ]
                out_gt = self.model(pixel_values=sub_x, labels=sub_lbls, return_dict=True)
                total_loss = total_loss + out_gt.loss

            if need_fallback:
                sub_fb_x = adv_images[need_fallback]
                out_fb = self.model(pixel_values=sub_fb_x, return_dict=True)
                for local_i, b in enumerate(need_fallback):
                    m_b = fallback_masks[b]
                    probs_b = torch.sigmoid(out_fb.logits[local_i])
                    fg_probs = probs_b[m_b, fallback_labels[b]]
                    cls_drop = -torch.log(fg_probs + 1e-6).mean()
                    coord_drift = F.l1_loss(out_fb.pred_boxes[local_i][m_b], fallback_boxes[b])
                    total_loss = total_loss + cls_drop + 0.5 * coord_drift

            self.model.zero_grad()
            if adv_images.grad is not None:
                adv_images.grad.zero_()
            total_loss.backward()

            with torch.no_grad():
                grad_sign = adv_images.grad.sign() if adv_images.grad is not None else torch.zeros_like(adv_images)
                adv_images = adv_images + step_alpha * grad_sign
                delta = torch.clamp(adv_images - orig, min=-self.epsilon, max=self.epsilon)
                adv_images = torch.clamp(orig + delta, min=0.0, max=1.0).detach()

        return adv_images
