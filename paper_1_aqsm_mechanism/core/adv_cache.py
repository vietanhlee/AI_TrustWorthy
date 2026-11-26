# -*- coding: utf-8 -*-
"""
Module: paper_1_aqsm_mechanism.core.adv_cache
Mục đích: Quản lý đọc nạp và lưu trữ bộ nhớ đệm mẫu đối kháng (adversarial image cache)
đảm bảo 100% tính nhất quán thực nghiệm qua các task phân tích độc lập.
Hỗ trợ:
  1. Caching linh hoạt theo mọi quy mô dataset (N ảnh bất kỳ).
  2. Nạp proxy lười (LazyAdvImageBatchProxy) để tiết kiệm bộ nhớ RAM khi chạy quy mô lớn.
  3. Kiểm tra toàn vẹn định danh ảnh (Image ID Verification) chống lệch dữ liệu.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch

from paper_1_aqsm_mechanism.core.dataset_loader import (
    LazyAdvImageBatchProxy,
    LazyImageBatchProxy,
    OnTheFlyAdvImageBatchProxy,
)


def get_default_cache_dir() -> Path:
    """Truy xuất đường dẫn cache mặc định động từ config."""
    from paper_1_aqsm_mechanism.experiments.config import get_adv_cache_dir
    return get_adv_cache_dir()


get_adv_cache_dir = get_default_cache_dir


def eps_tag(eps_num: int) -> str:
    """Trả về nhãn epsilon dạng xâu chuỗi (ví dụ: eps16)."""
    return f"eps{eps_num}"


def find_cache_file(
    attack: str,
    eps_num: int,
    n_eval: int,
    cache_dir: Optional[Union[str, Path]] = None,
    model_name: Optional[str] = None,
) -> Path:
    """
    Tìm kiếm file cache phù hợp nhất trong thư mục cache:
    1. Ưu tiên file đúng tên kèm model_tag: adv_cache_{model_tag}_{attack}_eps{eps_num}_n{n_eval}.npz
    2. File kèm model_tag chứa số ảnh >= n_eval: adv_cache_{model_tag}_{attack}_eps{eps_num}*.npz
    3. File alias kèm model_tag: adv_cache_{model_tag}_{attack}_eps{eps_num}.npz
    4. Fallback legacy không có model_tag: adv_cache_{attack}_eps{eps_num}_n{n_eval}.npz
    5. Fallback legacy glob: adv_cache_{attack}_eps{eps_num}*.npz
    """
    from paper_1_aqsm_mechanism.experiments.config import get_model_tag
    cdir = Path(cache_dir) if cache_dir is not None else get_default_cache_dir()
    mtag = get_model_tag(model_name)
    ep_str = eps_tag(eps_num)

    # 1. Tìm kiếm file chính xác có model_tag
    target_tagged = cdir / f"adv_cache_{mtag}_{attack}_{ep_str}_n{n_eval}.npz"
    if target_tagged.exists():
        return target_tagged

    # 2. Tìm kiếm các file tagged có n >= n_eval
    tagged_candidates = sorted(list(cdir.glob(f"adv_cache_{mtag}_{attack}_{ep_str}*.npz")), key=lambda p: p.stat().st_size, reverse=True)
    for cand in tagged_candidates:
        try:
            with np.load(cand, mmap_mode="r") as data:
                if len(data["image_ids"]) >= n_eval:
                    return cand
        except Exception:
            continue

    # 3. Fallback legacy (dành cho cache R18 cũ hoặc file không gắn tag)
    target_legacy = cdir / f"adv_cache_{attack}_{ep_str}_n{n_eval}.npz"
    if target_legacy.exists():
        return target_legacy

    legacy_candidates = sorted(list(cdir.glob(f"adv_cache_{attack}_{ep_str}*.npz")), key=lambda p: p.stat().st_size, reverse=True)
    for cand in legacy_candidates:
        # Bỏ qua file thuộc model khác nếu có tag rõ ràng
        cand_name = cand.name
        if any(other_m in cand_name for other_m in ["rtdetr_r50vd", "rtdetr_r18vd"]) and mtag not in cand_name:
            continue
        try:
            with np.load(cand, mmap_mode="r") as data:
                if len(data["image_ids"]) >= n_eval:
                    return cand
        except Exception:
            continue

    return target_tagged


def _make_onthefly_proxy(
    attack: str,
    eps_num: int,
    n_eval: int,
    clean_imgs: Any,
    wrapper: Optional[Any],
    steps: Optional[int],
    hf_labels: Optional[Sequence[Dict[str, Any]]],
    mode_note: str = "on_the_fly_streaming",
) -> Tuple[OnTheFlyAdvImageBatchProxy, Dict[str, Any]]:
    """Khởi tạo OnTheFlyAdvImageBatchProxy thống nhất với hf_labels và steps từ cấu hình."""
    from paper_1_aqsm_mechanism.experiments.config import get_pgd_num_steps
    eff_steps = int(steps) if steps is not None else int(get_pgd_num_steps())
    onthefly_proxy = OnTheFlyAdvImageBatchProxy(
        clean_proxy=clean_imgs,
        attack_name=attack,
        eps_num=eps_num,
        wrapper=wrapper,
        steps=eff_steps,
        hf_labels=hf_labels,
    )
    meta = {
        "attack": attack.lower(),
        "n_eval": n_eval,
        "epsilon_num_255": eps_num,
        "steps": eff_steps,
        "mode": mode_note,
    }
    return onthefly_proxy, meta


def load_cached_adv_proxy(
    attack: str,
    eps_num: int,
    n_eval: int,
    clean_proxy: LazyImageBatchProxy,
    expected_image_ids: Sequence[int],
    cache_dir: Optional[Union[str, Path]] = None,
    wrapper: Optional[Any] = None,
    allow_onthefly: bool = True,
    model_name: Optional[str] = None,
    steps: Optional[int] = None,
    hf_labels: Optional[Sequence[Dict[str, Any]]] = None,
) -> Tuple[Union[LazyAdvImageBatchProxy, OnTheFlyAdvImageBatchProxy], Dict[str, Any]]:
    """
    Nạp đối tượng Proxy lười cho ảnh đối kháng từ cache npz trên đĩa.
    Nếu chưa có cache và allow_onthefly=True: tự động chuyển sang chế độ On-The-Fly VRAM Streaming (0 GB đĩa).
    """
    effective_model = model_name
    if effective_model is None and wrapper is not None and hasattr(wrapper, "pretrained_model_name"):
        effective_model = wrapper.pretrained_model_name

    fpath = find_cache_file(attack, eps_num, n_eval, cache_dir=cache_dir, model_name=effective_model)
    if fpath is None or not fpath.exists():
        if allow_onthefly:
            print(f"[ADV CACHE] Không tìm thấy cache trên đĩa cho {attack.upper()} (eps={eps_num}).")
            print("            --> Kích hoạt On-The-Fly VRAM Streaming (0 GB đĩa, sinh và giải phóng theo batch)!")
            return _make_onthefly_proxy(attack, eps_num, n_eval, clean_proxy, wrapper, steps, hf_labels)
        raise FileNotFoundError(f"Chưa tìm thấy cache đối kháng cho {attack} (eps={eps_num}) tại: {fpath}")

    def _read_npz_validated(
        cache_path: Path,
    ) -> Tuple[List[int], np.ndarray, Dict[str, Any]]:
        """Mở, kiểm tra và giải phóng handle tệp npz an toàn (tránh rò rỉ file descriptor)."""
        try:
            ctx = np.load(cache_path, mmap_mode="r")
        except Exception:
            ctx = np.load(cache_path, allow_pickle=False)
        with ctx as data:
            cached_ids_local = data["image_ids"].tolist()
            delta_key_local = "delta" if "delta" in data else ("delta_f32" if "delta_f32" in data else "delta_f16")
            delta_sub = np.array(data[delta_key_local][:n_eval], copy=True)
            meta_local = json.loads(str(data["meta_json"])) if "meta_json" in data else {}
        return cached_ids_local, delta_sub, meta_local

    try:
        cached_ids, delta_all, meta = _read_npz_validated(fpath)
    except Exception as e:
        if allow_onthefly:
            print(f"[ADV CACHE WARNING] Lỗi đọc file cache {fpath}: {e}")
            print("                    --> Kích hoạt On-The-Fly VRAM Streaming (0 GB đĩa)!")
            return _make_onthefly_proxy(attack, eps_num, n_eval, clean_proxy, wrapper, steps, hf_labels)
        raise

    # Kiểm tra tính khớp của image_ids
    expected_ids = list(expected_image_ids)[:n_eval]
    cached_subset = cached_ids[:n_eval]
    if cached_subset != expected_ids:
        if allow_onthefly:
            print("[ADV CACHE WARNING] Image IDs trong cache không khớp hoàn toàn với cohort hiện tại!")
            print("                    --> Tự động chuyển sang On-The-Fly VRAM Streaming!")
            return _make_onthefly_proxy(attack, eps_num, n_eval, clean_proxy, wrapper, steps, hf_labels)
        raise ValueError(
            f"Image IDs không khớp giữa cache {fpath.name} và cohort hiện tại! "
            f"Cache[:3]={cached_subset[:3]}, Cohort[:3]={expected_ids[:3]}"
        )

    if tuple(delta_all.shape[-2:]) != tuple(clean_proxy.shape[-2:]):
        if allow_onthefly:
            print(
                f"[ADV CACHE WARNING] Kích thước không gian cache {delta_all.shape[-2:]} "
                f"không khớp ảnh sạch {clean_proxy.shape[-2:]}. Chuyển sang On-The-Fly VRAM Streaming!"
            )
            return _make_onthefly_proxy(attack, eps_num, n_eval, clean_proxy, wrapper, steps, hf_labels)
        raise ValueError(
            f"Kích thước không gian không khớp giữa delta {delta_all.shape[-2:]} và clean {clean_proxy.shape[-2:]}!"
        )

    adv_proxy = LazyAdvImageBatchProxy(clean_proxy, delta_all, device=clean_proxy.device)
    return adv_proxy, meta


def load_cached_adv_images(
    attack: str,
    eps_num: int,
    n_eval: int,
    clean_imgs: Union[torch.Tensor, LazyImageBatchProxy],
    expected_image_ids: Sequence[int],
    cache_dir: Optional[Union[str, Path]] = None,
    wrapper: Optional[Any] = None,
    allow_onthefly: bool = True,
    model_name: Optional[str] = None,
    steps: Optional[int] = None,
    hf_labels: Optional[Sequence[Dict[str, Any]]] = None,
) -> Tuple[Union[torch.Tensor, LazyAdvImageBatchProxy, OnTheFlyAdvImageBatchProxy], Dict[str, Any]]:
    """
    Nạp ảnh đối kháng từ file cache npz trên đĩa.
    Hỗ trợ cả Tensor PyTorch chuẩn lẫn LazyImageBatchProxy và OnTheFlyAdvImageBatchProxy (0 GB đĩa).
    """
    effective_model = model_name
    if effective_model is None and wrapper is not None and hasattr(wrapper, "pretrained_model_name"):
        effective_model = wrapper.pretrained_model_name

    if isinstance(clean_imgs, LazyImageBatchProxy):
        return load_cached_adv_proxy(
            attack=attack,
            eps_num=eps_num,
            n_eval=n_eval,
            clean_proxy=clean_imgs,
            expected_image_ids=expected_image_ids,
            cache_dir=cache_dir,
            wrapper=wrapper,
            allow_onthefly=allow_onthefly,
            model_name=effective_model,
            steps=steps,
            hf_labels=hf_labels,
        )

    fpath = find_cache_file(attack, eps_num, n_eval, cache_dir=cache_dir, model_name=effective_model)
    if fpath is None or not fpath.exists():
        if allow_onthefly:
            print(f"[ADV CACHE] Không tìm thấy cache trên đĩa cho {attack.upper()} (eps={eps_num}).")
            print("            --> Kích hoạt On-The-Fly VRAM Streaming (0 GB đĩa, sinh và giải phóng theo batch)!")
            return _make_onthefly_proxy(attack, eps_num, n_eval, clean_imgs, wrapper, steps, hf_labels)
        raise FileNotFoundError(f"Chưa tìm thấy cache đối kháng cho {attack} (eps={eps_num}) tại: {fpath}")

    try:
        try:
            ctx = np.load(fpath, mmap_mode="r")
        except Exception:
            ctx = np.load(fpath, allow_pickle=False)
        with ctx as data:
            cached_ids = data["image_ids"].tolist()
            delta_key = "delta" if "delta" in data else ("delta_f32" if "delta_f32" in data else "delta_f16")
            delta_raw = np.array(data[delta_key][:n_eval], copy=True)
            meta = json.loads(str(data["meta_json"])) if "meta_json" in data else {}
    except Exception as e:
        if allow_onthefly:
            print(f"[ADV CACHE WARNING] Lỗi đọc file cache {fpath}: {e}")
            print("                    --> Kích hoạt On-The-Fly VRAM Streaming (0 GB đĩa)!")
            return _make_onthefly_proxy(attack, eps_num, n_eval, clean_imgs, wrapper, steps, hf_labels)
        raise

    expected_ids = list(expected_image_ids)[:n_eval]
    cached_subset = cached_ids[:n_eval]
    if cached_subset != expected_ids:
        if allow_onthefly:
            print("[ADV CACHE WARNING] Image IDs trong cache không khớp hoàn toàn với cohort hiện tại!")
            print("                    --> Tự động chuyển sang On-The-Fly VRAM Streaming!")
            return _make_onthefly_proxy(attack, eps_num, n_eval, clean_imgs, wrapper, steps, hf_labels)
        raise ValueError(
            f"Image IDs không khớp giữa cache {fpath.name} và cohort hiện tại! "
            f"Cache[:3]={cached_subset[:3]}, Cohort[:3]={expected_ids[:3]}"
        )

    delta = torch.from_numpy(delta_raw if delta_raw.dtype == np.float32 else delta_raw.astype(np.float32))
    if delta.shape[-2:] != clean_imgs.shape[-2:]:
        if allow_onthefly:
            print(
                f"[ADV CACHE WARNING] Kích thước không gian cache {delta.shape[-2:]} "
                f"không khớp ảnh sạch {clean_imgs.shape[-2:]}. Chuyển sang On-The-Fly VRAM Streaming!"
            )
            return _make_onthefly_proxy(attack, eps_num, n_eval, clean_imgs, wrapper, steps, hf_labels)
        raise ValueError(
            f"Kích thước không gian không khớp giữa delta {delta.shape[-2:]} và clean {clean_imgs.shape[-2:]}!"
        )
    if clean_imgs.device != delta.device:
        delta = delta.to(clean_imgs.device)

    adv = torch.clamp(clean_imgs[:n_eval] + delta, 0.0, 1.0)
    return adv, meta
