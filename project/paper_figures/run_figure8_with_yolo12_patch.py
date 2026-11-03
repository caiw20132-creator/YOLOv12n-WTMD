import sys
import runpy

try:
    import torch
    from ultralytics.nn.modules.block import AAttn

    if not getattr(AAttn, "_wtb_qk_to_qkv_forward_patched", False):

        def _forward_qk_qkv_compat(self, x):
            B0, _, H, W = x.shape
            N0 = H * W

            if not hasattr(self, "all_head_dim"):
                self.all_head_dim = self.head_dim * self.num_heads

            all_head_dim = self.all_head_dim
            head_dim = self.head_dim
            num_heads = self.num_heads
            area = self.area

            # 情况1：当前模型有 qkv，并且 qkv 输出为 3C，走标准 YOLOv12 AAttn
            if hasattr(self, "qkv"):
                qkv_raw = self.qkv(x)

                if qkv_raw.shape[1] == all_head_dim * 3:
                    qkv = qkv_raw.flatten(2).transpose(1, 2)

                    B = B0
                    N = N0

                    if area > 1:
                        qkv = qkv.reshape(B * area, N // area, all_head_dim * 3)
                        B, N, _ = qkv.shape

                    q, k, v = (
                        qkv.view(B, N, num_heads, head_dim * 3)
                        .permute(0, 2, 3, 1)
                        .split([head_dim, head_dim, head_dim], dim=2)
                    )

                    attn = (q.transpose(-2, -1) @ k) * (head_dim ** -0.5)
                    attn = attn.softmax(dim=-1)

                    y = v @ attn.transpose(-2, -1)
                    y = y.permute(0, 3, 1, 2)
                    v_pe = v.permute(0, 3, 1, 2)

                    if area > 1:
                        y = y.reshape(B // area, N * area, all_head_dim)
                        v_pe = v_pe.reshape(B // area, N * area, all_head_dim)
                        B, N, _ = y.shape

                    y = y.reshape(B, H, W, all_head_dim).permute(0, 3, 1, 2).contiguous()
                    v_pe = v_pe.reshape(B, H, W, all_head_dim).permute(0, 3, 1, 2).contiguous()

                    y = y + self.pe(v_pe)
                    return self.proj(y)

                # 情况2：qkv 实际只有 2C，多半是旧权重 qk 被错误别名成 qkv
                if qkv_raw.shape[1] == all_head_dim * 2:
                    qk_raw = qkv_raw

                else:
                    raise RuntimeError(
                        f"AAttn qkv channel mismatch: got {qkv_raw.shape[1]}, "
                        f"expected {all_head_dim * 3} or {all_head_dim * 2}"
                    )

            # 情况3：旧 YOLOv12 权重只有 qk，没有 qkv
            elif hasattr(self, "qk"):
                qk_raw = self.qk(x)

            else:
                raise AttributeError("AAttn has neither qkv nor qk")

            # 旧版 qk 兼容逻辑：q 和 k 来自 qk，v 使用输入特征 x
            if qk_raw.shape[1] != all_head_dim * 2:
                raise RuntimeError(
                    f"AAttn qk channel mismatch: got {qk_raw.shape[1]}, "
                    f"expected {all_head_dim * 2}"
                )

            if x.shape[1] != all_head_dim:
                raise RuntimeError(
                    f"AAttn value channel mismatch: input has {x.shape[1]} channels, "
                    f"but all_head_dim is {all_head_dim}"
                )

            qk = qk_raw.flatten(2).transpose(1, 2)
            v0 = x.flatten(2).transpose(1, 2)

            B = B0
            N = N0

            if area > 1:
                qk = qk.reshape(B * area, N // area, all_head_dim * 2)
                v0 = v0.reshape(B * area, N // area, all_head_dim)
                B, N, _ = qk.shape

            q, k = (
                qk.view(B, N, num_heads, head_dim * 2)
                .permute(0, 2, 3, 1)
                .split([head_dim, head_dim], dim=2)
            )

            v = (
                v0.view(B, N, num_heads, head_dim)
                .permute(0, 2, 3, 1)
            )

            attn = (q.transpose(-2, -1) @ k) * (head_dim ** -0.5)
            attn = attn.softmax(dim=-1)

            y = v @ attn.transpose(-2, -1)
            y = y.permute(0, 3, 1, 2)
            v_pe = v.permute(0, 3, 1, 2)

            if area > 1:
                y = y.reshape(B // area, N * area, all_head_dim)
                v_pe = v_pe.reshape(B // area, N * area, all_head_dim)
                B, N, _ = y.shape

            y = y.reshape(B, H, W, all_head_dim).permute(0, 3, 1, 2).contiguous()
            v_pe = v_pe.reshape(B, H, W, all_head_dim).permute(0, 3, 1, 2).contiguous()

            y = y + self.pe(v_pe)
            return self.proj(y)

        AAttn.forward = _forward_qk_qkv_compat
        AAttn._wtb_qk_to_qkv_forward_patched = True

        print("[OK] YOLOv12 AAttn qk/qkv full forward compatibility patch enabled")

except Exception as e:
    print("[WARN] YOLOv12 AAttn compatibility patch failed:", repr(e))

SCRIPT = r"D:\YOLO-WTB\paper_figures\make_figure8_tpfpfn.py"
sys.argv = [SCRIPT] + sys.argv[1:]
runpy.run_path(SCRIPT, run_name="__main__")
