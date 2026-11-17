# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.core.detr_wrapper
Mục đích: Lớp bao bọc (Wrapper) kiến trúc Transformer Object Detector (RT-DETR).
Cung cấp khả năng trích xuất toàn bộ trạng thái nội tại của mô hình:
- Phân phối phân loại của N queries (logits)
- Tọa độ bounding box dự đoán (pred_boxes)
- Trạng thái ẩn của các tầng Decoder (decoder_hidden_states)
- Ma trận trọng số Cross-Attention của các tầng Decoder (cross_attentions)

Đặc biệt tối ưu hóa cho RT-DETR (Real-Time DEtection TRansformer) - mô hình thời gian thực
chuẩn công nghiệp được triển khai phổ biến trên thiết bị Edge.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import (
    DetrConfig,
    DetrForObjectDetection,
    RTDetrConfig,
    RTDetrForObjectDetection,
)


@dataclass
class DETRInternalState:
    """
    Cấu trúc dữ liệu chứa toàn bộ thông tin nội tại từ một lượt suy luận Transformer Detector.

    Thuộc tính:
        logits: Tensor [B, N, C] (RT-DETR) hoặc [B, N, C+1] (Vanilla DETR).
        pred_boxes: Tensor [B, N, 4] - Tọa độ hộp [cx, cy, w, h] chuẩn hóa [0, 1].
        decoder_hidden_states: Tuple chứa L+1 tensors [B, N, D] - Biểu diễn vector của queries
                              từ tầng khởi tạo (layer 0) đến tầng decoder cuối cùng (layer L).
        cross_attentions: Tuple chứa L tensors [B, N, H, L, P] (RT-DETR Deformable Attention)
                          hoặc [B, H, N, S] (Vanilla DETR Dense Attention).
        intermediate_logits: Tensor [B, L-1, N, C] - Logits phân loại tại từng tầng Decoder trung gian.
        intermediate_pred_boxes: Tensor [B, L-1, N, 4] - Hộp dự đoán tại từng tầng Decoder trung gian.
        encoder_last_hidden_state: Tensor [B, S, D] - Đặc trưng không gian đầu ra của Encoder.
        is_sigmoid: bool - True nếu mô hình dùng Sigmoid đa nhãn (RT-DETR), False nếu Softmax (DETR).
    """
    logits: torch.Tensor
    pred_boxes: torch.Tensor
    decoder_hidden_states: Tuple[torch.Tensor, ...]
    cross_attentions: Tuple[torch.Tensor, ...]
    intermediate_logits: Optional[torch.Tensor] = None
    intermediate_pred_boxes: Optional[torch.Tensor] = None
    encoder_last_hidden_state: Optional[torch.Tensor] = None
    enc_outputs_class: Optional[torch.Tensor] = None
    enc_topk_logits: Optional[torch.Tensor] = None
    init_reference_points: Optional[torch.Tensor] = None
    is_sigmoid: bool = True


class RTDETRModelWrapper(nn.Module):
    """
    Wrapper chuyên dụng cho RT-DETR (Real-Time DEtection TRansformer).
    Khai thác Deformable Cross-Attention và Sigmoid Head cho Edge AI Security.
    """

    def __init__(
        self,
        pretrained_model_name: Optional[str] = None,
        custom_config: Optional[RTDetrConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Khởi tạo RT-DETR wrapper.
        """
        super().__init__()
        self.device = device or torch.device("cpu")
        self.is_sigmoid = True

        if pretrained_model_name is not None:
            config = RTDetrConfig.from_pretrained(pretrained_model_name, token=False)
            config._attn_implementation = "eager"
            self.model = RTDetrForObjectDetection.from_pretrained(
                pretrained_model_name,
                config=config,
                attn_implementation="eager",
                token=False,
            )
        elif custom_config is not None:
            custom_config._attn_implementation = "eager"
            self.model = RTDetrForObjectDetection(custom_config)
        else:
            light_cfg = RTDetrConfig(
                num_queries=100,
                d_model=128,
                decoder_layers=3,
                encoder_layers=1,
                decoder_attention_heads=4,
                num_labels=80,
                backbone_config=None,
                use_timm_backbone=False,
                _attn_implementation="eager",
            )
            self.model = RTDetrForObjectDetection(light_cfg)

        self.model.to(self.device)
        self.model.eval()

        self.num_queries = self.model.config.num_queries
        self.num_labels = self.model.config.num_labels
        self.d_model = self.model.config.d_model
        self.decoder_layers = self.model.config.decoder_layers

    def forward(self, pixel_values: torch.Tensor) -> DETRInternalState:
        """
        Forward pass và trích xuất trạng thái nội tại RT-DETR.
        """
        pixel_values = pixel_values.to(self.device)

        outputs = self.model(
            pixel_values=pixel_values,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )

        inter_logits = getattr(outputs, "intermediate_logits", None)
        inter_boxes = getattr(outputs, "intermediate_reference_points", None)
        if inter_boxes is None:
            inter_boxes = getattr(outputs, "intermediate_pred_boxes", None)

        return DETRInternalState(
            logits=outputs.logits,
            pred_boxes=outputs.pred_boxes,
            decoder_hidden_states=outputs.decoder_hidden_states,
            cross_attentions=outputs.cross_attentions,
            intermediate_logits=inter_logits,
            intermediate_pred_boxes=inter_boxes,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            enc_outputs_class=getattr(outputs, "enc_outputs_class", None),
            enc_topk_logits=getattr(outputs, "enc_topk_logits", None),
            init_reference_points=getattr(outputs, "init_reference_points", None),
            is_sigmoid=True,
        )

    def get_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Tính xác suất phân loại Sigmoid cho từng lớp độc lập (RT-DETR Focal Loss).
        """
        return torch.sigmoid(logits)


class DETRModelWrapper(nn.Module):
    """
    Wrapper tổng quát cho Transformer Object Detector (mặc định RT-DETR, tương thích cả DETR).
    """

    def __init__(
        self,
        pretrained_model_name: Optional[str] = None,
        custom_config: Optional[Union[DetrConfig, RTDetrConfig]] = None,
        device: Optional[torch.device] = None,
        use_rtdetr: bool = True,
    ) -> None:
        super().__init__()
        self.device = device or torch.device("cpu")
        self.use_rtdetr = use_rtdetr

        if use_rtdetr:
            if isinstance(custom_config, DetrConfig):
                rtdetr_cfg = RTDetrConfig(
                    num_queries=getattr(custom_config, "num_queries", 100),
                    d_model=getattr(custom_config, "d_model", 128),
                    decoder_layers=getattr(custom_config, "decoder_layers", 3),
                    encoder_layers=getattr(custom_config, "encoder_layers", 1),
                    decoder_attention_heads=getattr(custom_config, "decoder_attention_heads", 4),
                    num_labels=getattr(custom_config, "num_labels", 80),
                    backbone_config=None,
                    use_timm_backbone=False,
                    _attn_implementation="eager",
                )
                self.inner_wrapper = RTDETRModelWrapper(custom_config=rtdetr_cfg, device=self.device)
            else:
                self.inner_wrapper = RTDETRModelWrapper(
                    pretrained_model_name=pretrained_model_name,
                    custom_config=custom_config,
                    device=self.device,
                )
            self.model = self.inner_wrapper.model
            self.is_sigmoid = True
        else:
            if pretrained_model_name is not None:
                config = DetrConfig.from_pretrained(pretrained_model_name)
                config._attn_implementation = "eager"
                self.model = DetrForObjectDetection.from_pretrained(
                    pretrained_model_name,
                    config=config,
                    attn_implementation="eager",
                )
            elif isinstance(custom_config, DetrConfig):
                custom_config._attn_implementation = "eager"
                self.model = DetrForObjectDetection(custom_config)
            else:
                light_cfg = DetrConfig(
                    num_queries=50,
                    d_model=128,
                    encoder_layers=2,
                    decoder_layers=3,
                    encoder_attention_heads=4,
                    decoder_attention_heads=4,
                    num_labels=80,
                    _attn_implementation="eager",
                )
                self.model = DetrForObjectDetection(light_cfg)

            self.model.to(self.device)
            self.model.eval()
            self.is_sigmoid = False

        self.num_queries = self.model.config.num_queries
        self.num_labels = self.model.config.num_labels
        self.d_model = self.model.config.d_model
        self.decoder_layers = self.model.config.decoder_layers

    def forward(self, pixel_values: torch.Tensor) -> DETRInternalState:
        pixel_values = pixel_values.to(self.device)
        outputs = self.model(
            pixel_values=pixel_values,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
        inter_logits = getattr(outputs, "intermediate_logits", None)
        inter_boxes = getattr(outputs, "intermediate_reference_points", None)
        if inter_boxes is None:
            inter_boxes = getattr(outputs, "intermediate_pred_boxes", None)

        return DETRInternalState(
            logits=outputs.logits,
            pred_boxes=outputs.pred_boxes,
            decoder_hidden_states=outputs.decoder_hidden_states,
            cross_attentions=outputs.cross_attentions,
            intermediate_logits=inter_logits,
            intermediate_pred_boxes=inter_boxes,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            is_sigmoid=self.is_sigmoid,
        )

    def get_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        if self.is_sigmoid:
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=-1)
