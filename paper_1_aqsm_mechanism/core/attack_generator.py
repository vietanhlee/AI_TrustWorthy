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

    def _get_num_queries(self, total_candidates: int) -> int:
        """Lấy số query hợp lệ dựa theo wrapper.num_queries và giới hạn S candidates."""
        n_q_cfg = int(getattr(self.wrapper, "num_queries", 300) or 300)
        return min(max(1, n_q_cfg), max(1, int(total_candidates)))

    def _class_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Chuyển đổi logits sang xác suất lớp tương thích Sigmoid (RT-DETR/Deformable) và Softmax (DETR)."""
        if self.is_sigmoid:
            return torch.sigmoid(logits)
        return F.softmax(logits, dim=-1)[..., :-1]

    def _class_logits(self, logits: torch.Tensor, is_enc_output: bool = False) -> torch.Tensor:
        """Trích xuất tensor logit thuần của các lớp đối tượng (loại bỏ lớp no-object nếu là DETR Softmax)."""
        if is_enc_output or self.is_sigmoid:
            return logits
        return logits[..., :-1]

    def _select_fg_bg_indices(
        self,
        clean_out: Any,
        max_clean_probs: torch.Tensor,
        top_k_fg: int = 15,
        fg_selection_mode: str = "clean_prediction_proxy",
        gt_boxes: Optional[Sequence[np.ndarray]] = None,
        k_bg: int = 30,
        bg_rank_min: int = 400,
        bg_seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Chọn ứng viên foreground (clean proxy hoặc ground-truth matched theo từng ảnh b) và background."""
        B, S = max_clean_probs.shape
        k_fg = min(max(1, int(top_k_fg)), S)

        if fg_selection_mode == "ground_truth_matched" and gt_boxes is not None:
            from paper_1_aqsm_mechanism.core.mechanism_analyzer import compute_token_fg_masks
            anchors_all = None
            if hasattr(clean_out, "enc_outputs_coord_logits") and clean_out.enc_outputs_coord_logits is not None:
                anchors_all = torch.sigmoid(clean_out.enc_outputs_coord_logits).detach().cpu().numpy()
            elif hasattr(clean_out, "pred_boxes") and clean_out.pred_boxes is not None:
                anchors_all = clean_out.pred_boxes.detach().cpu().numpy()

            topk_fg_list = []
            base_fg_seed = int(bg_seed) if bg_seed is not None else 42
            for b in range(B):
                gt_b = gt_boxes[b] if b < len(gt_boxes) else np.zeros((0, 4), dtype=np.float32)
                anchors_b = anchors_all[b] if anchors_all is not None and b < len(anchors_all) else None
                if len(gt_b) > 0 and anchors_b is not None:
                    fg_mask = compute_token_fg_masks(anchors_b, gt_b)
                    fg_idx = np.where(fg_mask)[0]
                    if len(fg_idx) >= k_fg:
                        topk_fg_list.append(torch.tensor(fg_idx[:k_fg], device=self.device))
                    elif len(fg_idx) > 0:
                        rng_fg = np.random.RandomState(base_fg_seed + b)
                        pad = rng_fg.choice(fg_idx, size=k_fg - len(fg_idx), replace=True)
                        topk_fg_list.append(torch.tensor(np.concatenate([fg_idx, pad]), device=self.device))
                    else:
                        _, topk_b = torch.topk(max_clean_probs[b], k=k_fg)
                        topk_fg_list.append(topk_b)
                else:
                    _, topk_b = torch.topk(max_clean_probs[b], k=k_fg)
                    topk_fg_list.append(topk_b)
            topk_fg_idx = torch.stack(topk_fg_list, dim=0)
        else:
            _, topk_fg_idx = torch.topk(max_clean_probs, k=k_fg, dim=-1)

        k_bg_eff = min(max(1, int(k_bg)), S)
        rank_min_eff = max(0, int(bg_rank_min))
        if bg_seed is not None and S > rank_min_eff + k_bg_eff:
            rng_bg = np.random.RandomState(int(bg_seed))
            sorted_idx = torch.argsort(max_clean_probs, dim=-1, descending=True).cpu().numpy()
            bg_list = []
            for b in range(B):
                pool_bg = sorted_idx[b, rank_min_eff:]
                sampled_bg = rng_bg.choice(pool_bg, size=k_bg_eff, replace=False)
                bg_list.append(torch.from_numpy(sampled_bg).to(self.device))
            bottom_bg_idx = torch.stack(bg_list, dim=0)
        else:
            _, bottom_bg_idx = torch.topk(-max_clean_probs, k=k_bg_eff, dim=-1)

        return topk_fg_idx, bottom_bg_idx

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
        margin: Optional[float] = None,
        margin_logit: float = 1.0,
        lambda_suppr: float = 1.0,
        lambda_rank: float = 1.0,
        lambda_elev: float = 0.5,
        fg_selection_mode: str = "clean_prediction_proxy",
        gt_boxes: Optional[Sequence[np.ndarray]] = None,
        bg_seed: Optional[int] = None,
        top_k_fg: int = 15,
        k_bg: int = 30,
        bg_rank_min: int = 500,
    ) -> torch.Tensor:
        """
        Adversarial Query Selection Manipulation (AQSM / Selection Starvation Attack):
        Tấn công có chủ đích vào mắt xích Encoder Query Selection (cơ chế lựa chọn đề xuất top-K động kế thừa từ two-stage DETR, triển khai trên RT-DETR không dùng NMS).
        Thao túng thứ hạng (ranking) của S candidate predictions từ Hybrid Encoder (S = 8,400 ở native 640x640; 2,100 ở 320x320):
        1. Dìm điểm các candidate thuộc về vật thể thật (foreground candidates) để chúng rơi khỏi Top-300.
        2. Kích hoạt và nâng điểm các candidate thuộc vùng nền (background candidates) thế chỗ vào Top-300.
        3. Gây ra hiện tượng "Bỏ đói thông tin" (Information Starvation) ở Deformable Decoder:
           Reference points nạp vào Decoder bị đặt vào khoảng trống rác, khiến Decoder hoàn toàn mất dấu vật thể thật.

        Đặc tính toán học chuẩn mực:
        - Ranking Margin Loss được tính toán trong KHÔNG GIAN LOGIT (z_j - z_{(300)} + gamma_logit),
          đảm bảo khi thỏa mãn biên margin z_j <= z_{(300)} - gamma_logit thì hàm hinge-loss ReLU(.)
          trở nên bất hoạt (hinge-loss inactivity) và đóng góp gradient bằng 0 cho ứng viên đó
          (thay vì ép gradient vô ích ngoài vùng vi phạm margin; tránh bị lẫn với hiện tượng
          gradient saturation của sigmoid hay toàn bộ objective).
        - Đưa tham số lambda_elev (trọng số nâng nền) vào biểu thức tối ưu hóa tường minh.

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
        effective_margin = margin_logit if margin is None else float(margin)

        # Bước 1: Xác định tập candidate đại diện cho vật thể thật
        with torch.no_grad():
            clean_out = self.model(pixel_values=images.to(self.device), return_dict=True)
            has_enc_clean = hasattr(clean_out, "enc_outputs_class") and clean_out.enc_outputs_class is not None
            clean_enc_logits = self._class_logits(
                clean_out.enc_outputs_class if has_enc_clean else clean_out.logits,
                is_enc_output=has_enc_clean,
            )

            # Tính điểm tối đa trên các lớp cho từng spatial candidate
            clean_enc_probs = torch.sigmoid(clean_enc_logits) if has_enc_clean else self._class_probs(clean_out.logits)
            max_clean_probs, _ = clean_enc_probs.max(dim=-1)  # [B, S]
            topk_fg_idx, bottom_bg_idx = self._select_fg_bg_indices(
                clean_out=clean_out,
                max_clean_probs=max_clean_probs,
                top_k_fg=top_k_fg,
                fg_selection_mode=fg_selection_mode,
                gt_boxes=gt_boxes,
                k_bg=k_bg,
                bg_rank_min=bg_rank_min,
                bg_seed=bg_seed,
            )

        # Bước 2: Tối ưu hóa PGD thao túng ranking và dìm điểm trong không gian logit
        for _ in range(n_steps):
            adv_images.requires_grad_(True)
            outputs = self.model(pixel_values=adv_images, return_dict=True)

            has_enc = hasattr(outputs, "enc_outputs_class") and outputs.enc_outputs_class is not None
            raw_enc_logits = outputs.enc_outputs_class if has_enc else outputs.logits
            enc_logits = self._class_logits(raw_enc_logits, is_enc_output=has_enc)
            enc_probs = torch.sigmoid(enc_logits) if has_enc else self._class_probs(raw_enc_logits)

            # 1. Suppression Loss: Ép xác suất nhận diện của real object candidates về 0
            fg_probs_gathered = torch.gather(enc_probs, 1, topk_fg_idx.unsqueeze(-1).expand(-1, -1, enc_probs.shape[-1]))
            fg_probs, _ = fg_probs_gathered.max(dim=-1)  # [B, k_fg]
            loss_suppress = (fg_probs ** 2).mean()

            fg_logits = torch.gather(enc_logits, 1, topk_fg_idx.unsqueeze(-1).expand(-1, -1, enc_logits.shape[-1]))
            max_fg_logits, _ = fg_logits.max(dim=-1)  # [B, k_fg]

            # 2. Ngưỡng điểm của ứng viên thứ N_q trong toàn bộ S candidates
            max_all_logits, _ = enc_logits.max(dim=-1)  # [B, S]
            n_q_eff = self._get_num_queries(max_all_logits.shape[1])
            val_at_300_logit, _ = torch.kthvalue(max_all_logits, k=max(1, max_all_logits.shape[1] - n_q_eff + 1), dim=-1)  # [B]

            # Margin Ranking Loss trong không gian Logit (-inf, +inf):
            loss_rank = F.relu(max_fg_logits - val_at_300_logit.unsqueeze(-1) + effective_margin).mean()

            # 3. Đẩy điểm background candidates lên để chiếm chỗ trong Top-N_q
            bg_probs_gathered = torch.gather(enc_probs, 1, bottom_bg_idx.unsqueeze(-1).expand(-1, -1, enc_probs.shape[-1]))
            max_bg_probs, _ = bg_probs_gathered.max(dim=-1)  # [B, k_bg]
            loss_bg_boost = -max_bg_probs.mean()

            total_loss = (
                float(lambda_suppr) * loss_suppress
                + float(lambda_rank) * loss_rank
                + float(lambda_elev) * loss_bg_boost
            )

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
    def generate_memory_isolated_query_attack(
        self,
        images: torch.Tensor,
        steps: Optional[int] = None,
        alpha: Optional[float] = None,
        margin_logit: float = 1.0,
        lambda_rank: float = 1.0,
        lambda_elev: float = 0.5,
        lambda_mem: float = 20.0,
        fg_selection_mode: str = "clean_prediction_proxy",
        gt_boxes: Optional[Sequence[np.ndarray]] = None,
        bg_seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Memory-Isolated / Feature-Constrained AQSM Attack (Nhóm đối chứng cô lập kênh Query Selection):
        Khai thác loss phạt độ lệch Feature Memory (Memory Penalty Loss) để ép encoder memory
        giữ nguyên vẹn tối đa so với ảnh sạch (Cosine similarity > 0.95), cô lập thuần túy
        kênh Query Selection để chứng minh độc lập tác động nhân quả của việc thao túng query.
        """
        self.model.eval()
        adv_images = images.clone().detach().to(self.device)
        n_steps = steps if steps is not None else self.num_steps
        step_alpha = alpha if alpha is not None else self.alpha

        # Bước 1: Trích xuất đặc trưng sạch tham chiếu (Clean Memory Reference)
        with torch.no_grad():
            clean_out = self.model(pixel_values=images.to(self.device), return_dict=True)
            has_enc_clean = hasattr(clean_out, "enc_outputs_class") and clean_out.enc_outputs_class is not None
            clean_enc_logits = self._class_logits(
                clean_out.enc_outputs_class if has_enc_clean else clean_out.logits,
                is_enc_output=has_enc_clean,
            )

            clean_enc_probs = torch.sigmoid(clean_enc_logits) if has_enc_clean else self._class_probs(clean_out.logits)
            max_clean_probs, _ = clean_enc_probs.max(dim=-1)
            topk_fg_idx, bottom_bg_idx = self._select_fg_bg_indices(
                clean_out=clean_out,
                max_clean_probs=max_clean_probs,
                top_k_fg=15,
                fg_selection_mode=fg_selection_mode,
                gt_boxes=gt_boxes,
                k_bg=30,
                bg_rank_min=400,
                bg_seed=bg_seed,
            )

            # Lưu giữ hidden states của encoder và coordinate logits sạch
            clean_enc_hidden = (
                [h.detach() for h in clean_out.encoder_last_hidden_state]
                if getattr(clean_out, "encoder_last_hidden_state", None) is not None
                else None
            )
            clean_coords = (
                clean_out.enc_outputs_coord_logits.detach()
                if getattr(clean_out, "enc_outputs_coord_logits", None) is not None
                else None
            )

        # Bước 2: Tối ưu hóa PGD kết hợp hàm phạt bảo toàn Feature Memory
        for _ in range(n_steps):
            adv_images.requires_grad_(True)
            outputs = self.model(pixel_values=adv_images, return_dict=True)

            has_enc = hasattr(outputs, "enc_outputs_class") and outputs.enc_outputs_class is not None
            raw_enc_logits = outputs.enc_outputs_class if has_enc else outputs.logits
            enc_logits = self._class_logits(raw_enc_logits, is_enc_output=has_enc)
            enc_probs = torch.sigmoid(enc_logits) if has_enc else self._class_probs(raw_enc_logits)

            # AQSM Task Objectives (trong không gian Logit)
            fg_probs_gathered = torch.gather(enc_probs, 1, topk_fg_idx.unsqueeze(-1).expand(-1, -1, enc_probs.shape[-1]))
            fg_probs, _ = fg_probs_gathered.max(dim=-1)
            loss_suppress = (fg_probs ** 2).mean()

            fg_logits = torch.gather(enc_logits, 1, topk_fg_idx.unsqueeze(-1).expand(-1, -1, enc_logits.shape[-1]))
            max_fg_logits, _ = fg_logits.max(dim=-1)

            max_all_logits, _ = enc_logits.max(dim=-1)
            n_q_eff = self._get_num_queries(max_all_logits.shape[1])
            val_at_300_logit, _ = torch.kthvalue(max_all_logits, k=max(1, max_all_logits.shape[1] - n_q_eff + 1), dim=-1)
            loss_rank = F.relu(max_fg_logits - val_at_300_logit.unsqueeze(-1) + margin_logit).mean()

            bg_probs_gathered = torch.gather(enc_probs, 1, bottom_bg_idx.unsqueeze(-1).expand(-1, -1, enc_probs.shape[-1]))
            max_bg_probs, _ = bg_probs_gathered.max(dim=-1)
            loss_bg_boost = -max_bg_probs.mean()

            loss_aqsm = loss_suppress + lambda_rank * loss_rank + lambda_elev * loss_bg_boost

            # Feature Memory Preservation Penalty: Ép Memory không bị biến dạng
            loss_mem_pen = torch.zeros((), device=self.device)
            if clean_enc_hidden is not None and getattr(outputs, "encoder_last_hidden_state", None) is not None:
                for h_adv, h_cln in zip(outputs.encoder_last_hidden_state, clean_enc_hidden):
                    cos_sim = F.cosine_similarity(h_adv.flatten(2), h_cln.flatten(2), dim=-1).mean()
                    mse = F.mse_loss(h_adv, h_cln)
                    loss_mem_pen = loss_mem_pen + (1.0 - cos_sim) + 5.0 * mse
            if clean_coords is not None and getattr(outputs, "enc_outputs_coord_logits", None) is not None:
                loss_mem_pen = loss_mem_pen + F.mse_loss(outputs.enc_outputs_coord_logits, clean_coords)

            total_loss = loss_aqsm + lambda_mem * loss_mem_pen

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
    def generate_pure_memory_attack(
        self,
        images: torch.Tensor,
        steps: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Pure Feature Memory Disruption Attack (Nhóm đối chứng phá hủy đặc trưng thuần túy):
        Chỉ tập trung phá hủy độ tương đồng biểu diễn không gian (Feature Memory Cosine Similarity)
        của Hybrid Encoder mà hoàn toàn KHÔNG có hàm mục tiêu xếp hạng Query Selection
        (loại bỏ hoàn toàn loss_suppress, loss_rank, loss_bg_boost).
        """
        self.model.eval()
        adv_images = images.clone().detach().to(self.device)
        n_steps = steps if steps is not None else self.num_steps
        step_alpha = alpha if alpha is not None else self.alpha

        with torch.no_grad():
            clean_out = self.model(pixel_values=images.to(self.device), return_dict=True)
            clean_enc_hidden = (
                [h.detach() for h in clean_out.encoder_last_hidden_state]
                if getattr(clean_out, "encoder_last_hidden_state", None) is not None
                else None
            )

        for _ in range(n_steps):
            adv_images.requires_grad_(True)
            outputs = self.model(pixel_values=adv_images, return_dict=True)

            # Cực tiểu hóa Cosine Similarity giữa adversarial memory và clean memory (tối đa hóa 1 - cosine)
            loss_mem = torch.zeros((), device=self.device)
            if clean_enc_hidden is not None and getattr(outputs, "encoder_last_hidden_state", None) is not None:
                for h_adv, h_cln in zip(outputs.encoder_last_hidden_state, clean_enc_hidden):
                    cos_sim = F.cosine_similarity(h_adv.flatten(2), h_cln.flatten(2), dim=-1).mean()
                    loss_mem = loss_mem + cos_sim  # Tối thiểu hóa cosine bằng gradient descent
            else:
                # Fallback nếu không có encoder_last_hidden_state: phá hủy logits phân loại chung
                logits = outputs.logits
                loss_mem = torch.sigmoid(logits).mean()

            self.model.zero_grad()
            if adv_images.grad is not None:
                adv_images.grad.zero_()
            loss_mem.backward()

            with torch.no_grad():
                grad_sign = adv_images.grad.sign() if adv_images.grad is not None else torch.zeros_like(adv_images)
                adv_images = adv_images - step_alpha * grad_sign
                delta = torch.clamp(adv_images - images, min=-self.epsilon, max=self.epsilon)
                adv_images = torch.clamp(images + delta, min=0.0, max=1.0).detach()

        return adv_images

    @torch.enable_grad()
    def generate_adaptive_sqa_attack(
        self,
        images: torch.Tensor,
        sqa_module: Any,
        steps: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Adaptive White-box Attack on SQA (Kiểm định tấn công thích ứng SQA theo chuẩn Carlini-Tramèr):
        Kẻ tấn công biết rõ sự hiện diện của N_static anchors tĩnh được tiêm vào Deformable Decoder,
        thực hiện backpropagation xuyên suốt qua cả 300 dynamic queries VÀ N_static static queries.
        """
        from paper_1_aqsm_mechanism.core.mechanism_analyzer import (
            extract_encoder_components,
            run_decoder_from_components,
        )

        self.model.eval()
        adv_images = images.clone().detach().to(self.device)
        n_steps = steps if steps is not None else self.num_steps
        step_alpha = alpha if alpha is not None else self.alpha

        for _ in range(n_steps):
            adv_images.requires_grad_(True)
            # Trích xuất components với gradient khả vi
            core_model = getattr(self.wrapper, "model", self.model)
            c_comp = extract_encoder_components(core_model, adv_images, enable_grad=True)

            # Ghép queries động và queries tĩnh (truyền output_memory và anchors nếu content_mode='encoder_sample')
            aug_target, aug_ref = sqa_module.augment(
                c_comp["target"],
                c_comp["init_reference_points"],
                output_memory=c_comp.get("output_memory"),
                encoder_anchors=c_comp.get("anchors"),
            )

            # Chạy decoder qua toàn bộ queries
            dec_out = run_decoder_from_components(
                model=core_model,
                target=aug_target,
                source_flatten=c_comp["source_flatten"],
                init_reference_points=aug_ref,
                spatial_shapes=c_comp["spatial_shapes"],
                spatial_shapes_list=c_comp["spatial_shapes_list"],
                level_start_index=c_comp["level_start_index"],
                enable_grad=True,
            )

            # Tấn công thích ứng: dìm toàn bộ confidence scores của cả dynamic lẫn static queries
            probs = self._class_probs(dec_out["logits"])
            max_probs, _ = probs.max(dim=-1)  # [B, N_q + N_static]
            k_top = min(20, max_probs.shape[1])
            top_p, _ = torch.topk(max_probs, k=k_top, dim=-1)
            loss = (top_p ** 2).mean()

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
        sigma: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Nhóm đối chứng (Random Noise Control): Thêm nhiễu ngẫu nhiên không có gradient.
        Hỗ trợ chỉ định độ lệch chuẩn sigma (mặc định self.epsilon / 2.0) và random seed tất định.
        """
        device = images.device
        gen = None
        if seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(int(seed))

        eff_sigma = float(sigma) if sigma is not None else float(self.epsilon / 2.0)
        if noise_type == "gaussian":
            noise = torch.randn(images.shape, device=device, dtype=images.dtype, generator=gen) * eff_sigma
            noise = torch.clamp(noise, -self.epsilon, self.epsilon)
        else:
            rand_u = torch.rand(images.shape, device=device, dtype=images.dtype, generator=gen)
            noise = (rand_u * 2.0 - 1.0) * self.epsilon

        return torch.clamp(images + noise, 0.0, 1.0)

    @torch.enable_grad()
    def generate_encoder_attention_attack(
        self,
        images: torch.Tensor,
        steps: Optional[int] = None,
        alpha: Optional[float] = None,
        lambda_attn_entropy: float = 1.0,
        lambda_attn_shift: float = 1.0,
        lambda_mem_disrupt: float = 0.5,
    ) -> torch.Tensor:
        """
        Encoder Self-Attention Disruption Attack (Đối chứng thực nghiệm theo hướng tiếp cận
        phá hủy Self-Attention của Encoder trong Transformer Detector - cf. Jayasundara et al. [27]):
        - Cực đại hóa Shannon entropy trên bản đồ self-attention của Hybrid Encoder (AIFI / Encoder layers)
          để làm phẳng phân bố chú ý nội tại (entropy maximization / uniform KL), ĐỒNG THỜI
          tối đa hóa sai lệch biểu diễn chú ý và độ lệch cosine của encoder memory.
        - Nếu kiến trúc không trả về encoder_attentions khi `output_attentions=True`, tự động dùng
          phân tán entropy không gian trên encoder hidden states làm surrogate tương đương.
        """
        self.model.eval()
        orig = images.clone().detach().to(self.device)
        adv_images = orig.clone().detach()
        n_steps = steps if steps is not None else self.num_steps
        step_alpha = alpha if alpha is not None else self.alpha

        # Trích xuất bản đồ self-attention & hidden states sạch của Encoder làm tham chiếu
        with torch.no_grad():
            try:
                clean_out = self.model(
                    pixel_values=orig,
                    output_attentions=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
            except TypeError:
                clean_out = self.model(pixel_values=orig, return_dict=True)

            raw_clean_enc_attn = getattr(clean_out, "encoder_attentions", None)
            clean_enc_attn = (
                [a.detach() for a in raw_clean_enc_attn if a is not None]
                if raw_clean_enc_attn is not None
                else []
            )
            raw_clean_enc_hs = getattr(clean_out, "encoder_last_hidden_state", None)
            if isinstance(raw_clean_enc_hs, (list, tuple)):
                clean_enc_hs = [h.detach() for h in raw_clean_enc_hs if h is not None]
            elif raw_clean_enc_hs is not None:
                clean_enc_hs = [raw_clean_enc_hs.detach()]
            else:
                clean_enc_hs = []

        for _ in range(n_steps):
            adv_images.requires_grad_(True)
            try:
                out = self.model(
                    pixel_values=adv_images,
                    output_attentions=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
            except TypeError:
                out = self.model(pixel_values=adv_images, return_dict=True)

            total_obj = torch.zeros((), device=self.device)
            raw_adv_attn = getattr(out, "encoder_attentions", None)
            adv_attn = (
                [a for a in raw_adv_attn if a is not None]
                if raw_adv_attn is not None
                else []
            )

            # 1. Phá hủy phân bố Encoder Self-Attention (Entropy maximization + KL/MSE khỏi clean attention)
            if adv_attn:
                ent_terms = []
                shift_terms = []
                for idx, a_adv in enumerate(adv_attn):
                    a_safe = a_adv.clamp(min=1e-8)
                    ent = -(a_safe * torch.log(a_safe)).sum(dim=-1).mean()
                    ent_terms.append(ent)
                    if idx < len(clean_enc_attn) and clean_enc_attn[idx].shape == a_adv.shape:
                        shift_terms.append(F.mse_loss(a_adv, clean_enc_attn[idx]))
                # Maximize entropy (+ent) and maximize shift from clean attention (+shift) via gradient ascent (+step_alpha)
                loss_ent = torch.stack(ent_terms).mean()
                total_obj = total_obj + float(lambda_attn_entropy) * loss_ent
                if shift_terms:
                    total_obj = total_obj + float(lambda_attn_shift) * torch.stack(shift_terms).mean()
            else:
                # Fallback cho mô hình mock/kiến trúc không xuất encoder_attentions: phá hủy cấu trúc self-correlation trên memory
                raw_adv_hs = getattr(out, "encoder_last_hidden_state", None)
                adv_hs_fb = (
                    list(raw_adv_hs) if isinstance(raw_adv_hs, (list, tuple))
                    else ([raw_adv_hs] if raw_adv_hs is not None else [])
                )
                for h_adv in adv_hs_fb:
                    h_flat = h_adv.flatten(2).transpose(1, 2) if h_adv.ndim == 4 else h_adv
                    attn_proxy = torch.softmax(torch.bmm(F.normalize(h_flat, dim=-1), F.normalize(h_flat, dim=-1).transpose(1, 2)), dim=-1)
                    ent_p = -(attn_proxy.clamp(min=1e-8) * torch.log(attn_proxy.clamp(min=1e-8))).sum(dim=-1).mean()
                    total_obj = total_obj + float(lambda_attn_entropy) * ent_p

            # 2. Phá hủy độ tương đồng biểu diễn Encoder Memory đi kèm
            raw_adv_hs = getattr(out, "encoder_last_hidden_state", None)
            if isinstance(raw_adv_hs, (list, tuple)):
                adv_enc_hs = [h for h in raw_adv_hs if h is not None]
            elif raw_adv_hs is not None:
                adv_enc_hs = [raw_adv_hs]
            else:
                adv_enc_hs = []

            if adv_enc_hs and clean_enc_hs:
                mem_disrupt_terms = []
                for h_adv, h_cln in zip(adv_enc_hs, clean_enc_hs):
                    h_a_f = h_adv.flatten(2) if h_adv.ndim >= 3 else h_adv
                    h_c_f = h_cln.flatten(2) if h_cln.ndim >= 3 else h_cln
                    cos_sim = F.cosine_similarity(h_a_f, h_c_f, dim=-1).mean()
                    mem_disrupt_terms.append(1.0 - cos_sim)
                total_obj = total_obj + float(lambda_mem_disrupt) * torch.stack(mem_disrupt_terms).mean()

            self.model.zero_grad()
            if adv_images.grad is not None:
                adv_images.grad.zero_()
            total_obj.backward()

            with torch.no_grad():
                grad_sign = adv_images.grad.sign() if adv_images.grad is not None else torch.zeros_like(adv_images)
                # Gradient ascent cực đại hóa entropy và độ lệch attention
                adv_images = adv_images + step_alpha * grad_sign
                delta = torch.clamp(adv_images - orig, min=-self.epsilon, max=self.epsilon)
                adv_images = torch.clamp(orig + delta, min=0.0, max=1.0).detach()

        return adv_images

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
                probs = self._class_probs(logits)
                eff_cls = min(max(0, int(target_class_idx)), probs.shape[-1] - 1)
                target_prob = probs[..., eff_cls]
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
                dec_hs = getattr(clean_out, "decoder_hidden_states", None)
                clean_h_refs = [h.detach() for h in dec_hs] if dec_hs is not None else None

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
            probs = self._class_probs(logits)  # [B, N, C]

            # 1. Mục tiêu Tác vụ: Ép top-k queries nhạy cảm nhất nhận diện lớp target
            cls_logits = self._class_logits(logits, is_enc_output=False)
            eff_cls = min(max(0, int(target_class_idx)), cls_logits.shape[-1] - 1)
            target_logits = cls_logits[..., eff_cls]  # [B, N]
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

                out_dec_hs = getattr(outputs, "decoder_hidden_states", None)
                if target_detector in ["joint", "maha_aware"] and clean_h_refs is not None and out_dec_hs is not None:
                    # Phạt độ lệch L2 trên hidden states của Decoder
                    maha_surrogate = torch.zeros((), device=self.device)
                    for l_idx, h_cur in enumerate(out_dec_hs):
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
            has_enc_clean = hasattr(clean_out, "enc_outputs_class") and clean_out.enc_outputs_class is not None
            clean_enc_logits = self._class_logits(
                clean_out.enc_outputs_class if has_enc_clean else clean_out.logits,
                is_enc_output=has_enc_clean,
            )
            clean_enc_probs = torch.sigmoid(clean_enc_logits) if has_enc_clean else self._class_probs(clean_out.logits)
            clean_max_probs, _ = clean_enc_probs.max(dim=-1)
            n_q_eff = self._get_num_queries(clean_max_probs.shape[1])
            top300_clean_val, _ = torch.topk(clean_max_probs, k=n_q_eff, dim=-1)
            tau_300_clean = top300_clean_val[:, -1]

            bg_mask = (clean_max_probs < 0.10)
            # Bảo đảm mỗi ảnh có ít nhất k_eff ứng viên nền hợp lệ
            k_eff = min(max(1, int(k_fake)), clean_max_probs.shape[1])
            insufficient = bg_mask.sum(dim=-1) < k_eff
            if insufficient.any():
                _, bot_idx = torch.topk(-clean_max_probs, k=k_eff, dim=-1)
                for b_i in range(bg_mask.shape[0]):
                    if insufficient[b_i]:
                        bg_mask[b_i, bot_idx[b_i]] = True

        for _ in range(n_steps):
            adv_images.requires_grad_(True)
            outputs = self.model(pixel_values=adv_images, return_dict=True)
            has_enc = hasattr(outputs, "enc_outputs_class") and outputs.enc_outputs_class is not None
            raw_enc_logits = outputs.enc_outputs_class if has_enc else outputs.logits
            enc_logits = self._class_logits(raw_enc_logits, is_enc_output=has_enc)
            eff_cls = min(max(0, int(target_class_idx)), enc_logits.shape[-1] - 1)
            target_logits = enc_logits[..., eff_cls]

            masked_target_logits = target_logits.masked_fill(~bg_mask, -1e4)
            top_fake_logits, top_fake_idx = torch.topk(masked_target_logits, k=k_eff, dim=-1)

            loss_fake = -top_fake_logits.mean()
            if has_enc or self.is_sigmoid:
                min_fake_prob = torch.sigmoid(top_fake_logits[:, -1])
            else:
                all_probs = self._class_probs(raw_enc_logits)[..., eff_cls]
                min_fake_prob = torch.gather(all_probs, 1, top_fake_idx[:, -1:]) .squeeze(-1)
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
        margin: Optional[float] = None,
        margin_logit: float = 1.0,
        lambda_rank: float = 1.0,
        lambda_elev: float = 0.5,
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
        effective_margin = margin_logit if margin is None else float(margin)

        # Xác định foreground và background candidates trên ảnh sạch
        with torch.no_grad():
            clean_out = self.model(pixel_values=images.to(self.device), return_dict=True)
            has_enc_clean = hasattr(clean_out, "enc_outputs_class") and clean_out.enc_outputs_class is not None
            clean_enc_logits = self._class_logits(
                clean_out.enc_outputs_class if has_enc_clean else clean_out.logits,
                is_enc_output=has_enc_clean,
            )

            clean_enc_probs = torch.sigmoid(clean_enc_logits) if has_enc_clean else self._class_probs(clean_out.logits)
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

            has_enc = hasattr(outputs, "enc_outputs_class") and outputs.enc_outputs_class is not None
            raw_enc_logits = outputs.enc_outputs_class if has_enc else outputs.logits
            enc_logits = self._class_logits(raw_enc_logits, is_enc_output=has_enc)
            enc_probs = torch.sigmoid(enc_logits) if has_enc else self._class_probs(raw_enc_logits)
            max_all_probs, _ = enc_probs.max(dim=-1)  # [B, S]
            max_all_logits, _ = enc_logits.max(dim=-1)

            # 1. AQSM Task Loss: Dìm foreground và nâng background trong không gian Logit
            fg_probs_gathered = torch.gather(enc_probs, 1, topk_fg_idx.unsqueeze(-1).expand(-1, -1, enc_probs.shape[-1]))
            fg_probs, _ = fg_probs_gathered.max(dim=-1)
            loss_suppress = (fg_probs ** 2).mean()

            fg_logits = torch.gather(enc_logits, 1, topk_fg_idx.unsqueeze(-1).expand(-1, -1, enc_logits.shape[-1]))
            max_fg_logits, _ = fg_logits.max(dim=-1)

            n_q_eff = self._get_num_queries(max_all_logits.shape[1])
            val_at_300_logit, _ = torch.kthvalue(max_all_logits, k=max(1, max_all_logits.shape[1] - n_q_eff + 1), dim=-1)
            loss_rank = F.relu(max_fg_logits - val_at_300_logit.unsqueeze(-1) + effective_margin).mean()

            bg_probs_gathered = torch.gather(enc_probs, 1, bottom_bg_idx.unsqueeze(-1).expand(-1, -1, enc_probs.shape[-1]))
            max_bg_probs, _ = bg_probs_gathered.max(dim=-1)
            loss_bg_boost = -max_bg_probs.mean()

            task_loss = loss_suppress + lambda_rank * loss_rank + lambda_elev * loss_bg_boost

            # 2. ESM Evasion Loss: Giữ dropoff và margin trong vùng sạch chuẩn
            topk_vals, _ = torch.topk(max_all_probs, k=n_q_eff, dim=-1)
            top1_val = topk_vals[:, 0]
            top300_val = topk_vals[:, -1]
            diff_dropoff = top1_val / (top300_val.clamp(min=1e-4) + 1e-4)

            top300_mean = topk_vals.mean(dim=-1)
            all_sum = max_all_probs.sum(dim=-1)
            topk_sum = topk_vals.sum(dim=-1)
            bg_count = max_all_probs.shape[1] - n_q_eff
            bg_mean = (all_sum - topk_sum) / float(bg_count) if bg_count > 0 else torch.zeros_like(top300_mean)
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
            dec_hs = getattr(clean_out, "decoder_hidden_states", None)
            clean_h_refs = [h.detach() for h in dec_hs] if dec_hs is not None else None

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
            probs = self._class_probs(logits)

            # 1. Task Objective (Fabrication of target class)
            cls_logits = self._class_logits(logits, is_enc_output=False)
            eff_cls = min(max(0, int(target_class_idx)), cls_logits.shape[-1] - 1)
            target_logits = cls_logits[..., eff_cls]
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
                n_q_eff = self._get_num_queries(max_enc_p.shape[1])
                topk_enc, _ = torch.topk(max_enc_p, k=n_q_eff, dim=-1)
                diff_drop = topk_enc[:, 0] / (topk_enc[:, -1].clamp(min=1e-4) + 1e-4)
                loss_t1 = (F.relu(tau_drop_min - diff_drop) + F.relu(diff_drop - tau_drop_max)).pow(2).mean()
            else:
                loss_t1 = 0.0

            # 4. Tier 2 Evasion (Decoder Manifold L2 Surrogate)
            loss_t2 = torch.zeros((), device=self.device)
            out_dec_hs = getattr(outputs, "decoder_hidden_states", None)
            if clean_h_refs is not None and out_dec_hs is not None:
                for l_idx, h_cur in enumerate(out_dec_hs):
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
            if hf_labels is None or b >= len(hf_labels) or len(hf_labels[b]["boxes"]) == 0
        ]
        fallback_masks: Dict[int, torch.Tensor] = {}
        fallback_labels: Dict[int, torch.Tensor] = {}
        fallback_boxes: Dict[int, torch.Tensor] = {}
        if need_fallback:
            with torch.no_grad():
                clean_out = self.model(pixel_values=orig, return_dict=True)
                clean_probs = self._class_probs(clean_out.logits)
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
            if hf_labels is not None and b < len(hf_labels) and len(hf_labels[b]["boxes"]) > 0
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
                gt_loss = getattr(out_gt, "loss", None)
                if gt_loss is not None:
                    total_loss = total_loss + gt_loss
                else:
                    # Fallback nếu wrapper/mock không tính loss khi nhận labels
                    gt_probs_all = self._class_probs(out_gt.logits)
                    for local_i, b in enumerate(valid_indices):
                        cls_t = hf_labels[b]["class_labels"].to(self.device)
                        top_p = gt_probs_all[local_i, :, cls_t].max(dim=0).values
                        total_loss = total_loss - torch.log(top_p.clamp(min=1e-6)).mean()

            if need_fallback:
                sub_fb_x = adv_images[need_fallback]
                out_fb = self.model(pixel_values=sub_fb_x, return_dict=True)
                for local_i, b in enumerate(need_fallback):
                    m_b = fallback_masks[b]
                    probs_b = self._class_probs(out_fb.logits[local_i])
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

    @torch.enable_grad()
    def generate_latency_flooding_attack(
        self,
        images: torch.Tensor,
        steps: Optional[int] = None,
        alpha: Optional[float] = None,
        conf_threshold: float = 0.35,
    ) -> torch.Tensor:
        """
        Worst-Case Proposal-Flooding / Sponge Attack nhắm thẳng vào RT-DETR:
        Khác với Task PGD hay AQSM, mục tiêu này tối ưu PGD L_inf nhằm ĐẨY CỰC ĐẠI
        độ tự tin phân loại trên toàn bộ 300 decoder queries (và encoder proposals)
        vượt qua ngưỡng conf_threshold, kiểm chứng xem mật độ active boxes cực đại
        có gây suy giảm thông lượng/độ trễ trên RT-DETR hay không.
        """
        self.model.eval()
        orig = images.clone().detach().to(self.device)
        adv_images = orig.clone().detach()
        n_steps = steps if steps is not None else self.num_steps
        step_alpha = alpha if alpha is not None else self.alpha

        for _ in range(n_steps):
            adv_images.requires_grad_(True)
            outputs = self.model(pixel_values=adv_images, return_dict=True)

            dec_probs = self._class_probs(outputs.logits)
            max_dec_probs, _ = dec_probs.max(dim=-1)  # [B, N_q]
            # Phạt mượt (soft hinge + probability boost) đẩy mọi query vượt ngưỡng conf_threshold
            loss_dec_flood = F.relu((conf_threshold + 0.25) - max_dec_probs).mean() - 0.5 * max_dec_probs.mean()

            total_loss = loss_dec_flood
            if hasattr(outputs, "enc_outputs_class") and outputs.enc_outputs_class is not None:
                enc_max_probs, _ = torch.sigmoid(outputs.enc_outputs_class).max(dim=-1)
                n_q_eff = self._get_num_queries(enc_max_probs.shape[1])
                top_enc_probs, _ = torch.topk(enc_max_probs, k=n_q_eff, dim=-1)
                total_loss = total_loss - 0.5 * top_enc_probs.mean()

            self.model.zero_grad()
            if adv_images.grad is not None:
                adv_images.grad.zero_()
            total_loss.backward()

            with torch.no_grad():
                grad_sign = adv_images.grad.sign() if adv_images.grad is not None else torch.zeros_like(adv_images)
                adv_images = adv_images - step_alpha * grad_sign
                delta = torch.clamp(adv_images - orig, min=-self.epsilon, max=self.epsilon)
                adv_images = torch.clamp(orig + delta, min=0.0, max=1.0).detach()

        return adv_images
