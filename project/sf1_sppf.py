# -*- coding: utf-8 -*-
"""
SF1-SPPF 模块 (SoftPool Feature SPPF-Fast) - 用于 Ultralytics YOLO 模型

✅ 关键修复：
- 兼容 Ultralytics 对“未知自定义模块”可能直接调用 SF1SPPF(c2, k) 的情况，
  避免把 (256,5) 误解成 (c1=256, c2=5) 导致输出通道=5。

- 若运行时输入通道 != 期望 c1，则用 1x1 Conv adapter 自动对齐。

典型错误修复：
- expected input[...] to have 128 channels, but got 5 channels instead
"""

from __future__ import annotations

from typing import Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftPool2d(nn.Module):
    """SoftPool2d: sum(x*exp(x))/sum(exp(x))"""

    def __init__(
        self,
        kernel_size: int = 2,
        stride: Optional[int] = None,
        padding: int = 0,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(stride) if stride is not None else int(kernel_size)
        self.padding = int(padding)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.exp(x)
        num = F.avg_pool2d(x * w, self.kernel_size, self.stride, self.padding)
        den = F.avg_pool2d(w, self.kernel_size, self.stride, self.padding)
        return num / (den + self.eps)


def _resolve_sf1sppf_args(args: Tuple[Any, ...], kwargs: dict) -> Tuple[int, int, int]:
    """
    兼容多种构造方式，尽量按 YAML 意图解析：

    支持：
    - SF1SPPF(c1, c2, k)
    - SF1SPPF(c1, c2) -> k=5
    - SF1SPPF(c2, k)  -> 视为 [c2, k]（Ultralytics 未补 c1 的常见情况）
    - SF1SPPF(c2,)    -> k=5 且 c1=c2
    """
    k_default = int(kwargs.get("k", 5))

    if len(args) >= 3:
        c1 = int(args[0])
        c2 = int(args[1])
        k = int(args[2])
        return c1, c2, k

    if len(args) == 2:
        a = int(args[0])
        b = int(args[1])

        # b 很像 kernel_size，且 a 看起来像通道数：按 (c2=a, k=b)，并令 c1=c2
        if b in (3, 5, 7, 9, 11, 13, 15) and a >= 16:
            c2 = a
            k = b
            c1 = c2
            return c1, c2, k

        # 否则按 (c1=a, c2=b, k=默认)
        c1 = a
        c2 = b
        k = k_default
        return c1, c2, k

    if len(args) == 1:
        c2 = int(args[0])
        c1 = c2
        k = k_default
        return c1, c2, k

    raise TypeError("SF1SPPF requires at least 1 positional arg.")


class SF1SPPF(nn.Module):
    """
    SF1SPPF: SPPF-like，但使用 SoftPool 替代 MaxPool。

    ✅ 兼容 Ultralytics 自定义模块解析：
    - YAML 写 [c2, k] 但 parse_model 未补 c1 时，会调用 SF1SPPF(c2, k)
      本类会自动理解成 (c1=c2, c2=c2, k=k)，不会把 5 当输出通道。
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        from ultralytics.nn.modules import Conv  # 延迟导入避免循环

        c1, c2, k = _resolve_sf1sppf_args(args, kwargs)

        self.expected_c1 = int(c1)
        self.c2 = int(c2)
        self.k = int(k)

        print(f"[SF1SPPF DEBUG] resolved args: c1={self.expected_c1}, c2={self.c2}, k={self.k}")

        self.adapter: Optional[nn.Module] = None

        c_ = max(1, self.expected_c1 // 2)
        self.cv1 = Conv(self.expected_c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, self.c2, 1, 1)

        p = self.k // 2
        self.pool = SoftPool2d(kernel_size=self.k, stride=1, padding=p)

    def _build_adapter_if_needed(self, x: torch.Tensor) -> None:
        if int(x.shape[1]) == self.expected_c1:
            return
        from ultralytics.nn.modules import Conv
        in_ch = int(x.shape[1])
        print(f"[SF1SPPF] 通道适配: {in_ch} -> {self.expected_c1}")
        self.adapter = Conv(in_ch, self.expected_c1, 1, 1).to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if int(x.shape[1]) != self.expected_c1:
            self._build_adapter_if_needed(x)
            if self.adapter is not None:
                x = self.adapter(x)

        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), 1))


def register_ultralytics_modules() -> None:
    """注册到 Ultralytics 解析器命名空间（给 eval() 找到）"""
    import ultralytics.nn.tasks as tasks
    import ultralytics.nn.modules as modules

    tasks.__dict__["SF1SPPF"] = SF1SPPF
    setattr(tasks, "SF1SPPF", SF1SPPF)

    modules.__dict__["SF1SPPF"] = SF1SPPF
    setattr(modules, "SF1SPPF", SF1SPPF)

    if hasattr(modules, "__all__") and isinstance(modules.__all__, list):
        if "SF1SPPF" not in modules.__all__:
            modules.__all__.append("SF1SPPF")

    try:
        ok = (eval("SF1SPPF", tasks.__dict__) is SF1SPPF)
    except Exception:
        ok = False

    print("[OK] SF1SPPF 已成功注册到 Ultralytics 解析器" if ok else "[WARN] SF1SPPF 注册可能不完整")
    print(f"[DEBUG] 'SF1SPPF' in tasks.__dict__: {'SF1SPPF' in tasks.__dict__}")
    print(f"[DEBUG] 'SF1SPPF' in dir(tasks): {'SF1SPPF' in dir(tasks)}")
    print(f"[DEBUG] 'SF1SPPF' in modules.__dict__: {'SF1SPPF' in modules.__dict__}")

