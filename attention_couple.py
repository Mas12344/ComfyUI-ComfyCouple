import math
import torch
import torch.nn.functional as F
import copy
import comfy
import comfy.patcher_extension
from comfy.ldm.modules.attention import optimized_attention

def manual_scaled_dot_product_attention(q, k, v):
    B, Sq, H, D = q.shape
    qh = q.permute(0, 2, 1, 3)
    kh = k.permute(0, 2, 1, 3)
    vh = v.permute(0, 2, 1, 3)
    scale = D ** -0.5
    attn = torch.matmul(qh.float(), kh.float().transpose(-1, -2)) * scale
    attn = attn.softmax(dim=-1)
    out = torch.matmul(attn.to(vh.dtype), vh)
    out = out.permute(0, 2, 1, 3).reshape(B, Sq, H * D)
    return out


def get_masks_from_q(masks, q, original_shape):
    oh, ow = original_shape[2], original_shape[3]
 
    scale = math.sqrt((oh * ow) / q.shape[1])
    feat_h = max(1, round(oh / scale))
    feat_w = max(1, round(ow / scale))
 
    while feat_h * feat_w > q.shape[1] and feat_h > 1:
        feat_h -= 1
    while feat_h * feat_w > q.shape[1] and feat_w > 1:
        feat_w -= 1
    while feat_h * feat_w < q.shape[1]:
        feat_w += 1
 
    ret_masks = []
    for mask in masks:
        if isinstance(mask, torch.Tensor):
            mask_downsample = F.interpolate(
                mask.unsqueeze(0), size=(feat_h, feat_w), mode="nearest"
            )
            mask_downsample = mask_downsample.view(1, -1, 1).repeat(
                q.shape[0], 1, q.shape[2]
            )
            ret_masks.append(mask_downsample)
        else:  # coupling処理なしの場合
            ret_masks.append(torch.ones_like(q))
 
    ret_masks = torch.cat(ret_masks, dim=0)
    return ret_masks

def set_model_patch_replace(model, patch, key):
    to = model.model_options["transformer_options"]
    if "patches_replace" not in to:
        to["patches_replace"] = {}
    if "attn2" not in to["patches_replace"]:
        to["patches_replace"]["attn2"] = {}
    to["patches_replace"]["attn2"][key] = patch

class AttentionCouple:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "mode": (["Attention", "Latent"], ),
            }
        }
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING")
    FUNCTION = "attention_couple"
    CATEGORY = "loaders"

    def attention_couple(self, model, positive, negative, mode):
        if mode == "Latent":
            return (model, positive, negative) # latent coupleの場合は何もしない
        
        self.negative_positive_masks = []
        self.negative_positive_conds = []
        self.negative_positive_extra = []

        new_positive = copy.deepcopy(positive)
        new_negative = copy.deepcopy(negative)

        dtype = model.model.diffusion_model.dtype
        device = comfy.model_management.get_torch_device()

        # maskとcondをリストに格納する
        for conditions in [new_negative, new_positive]:
            conditions_masks = []
            conditions_conds = []
            conditions_extra = []
            if len(conditions) != 1:
                mask_norm = torch.stack([cond[1]["mask"].to(device, dtype=dtype) * cond[1]["mask_strength"] for cond in conditions])
                mask_norm = mask_norm / mask_norm.sum(dim=0) # 合計が1になるように正規化(他が0の場合mask_strengthの効果がなくなる)
                conditions_masks.extend([mask_norm[i] for i in range(mask_norm.shape[0])])
                conditions_conds.extend([cond[0].to(device, dtype=dtype) for cond in conditions])
                conditions_extra.extend([cond[1] for cond in conditions])
                del conditions[0][1]["mask"] # latent coupleの無効化のため
                del conditions[0][1]["mask_strength"]
            else:
                conditions_masks = [False]
                conditions_conds = [conditions[0][0].to(device, dtype=dtype)]
                conditions_extra = [conditions[0][1]]
            self.negative_positive_masks.append(conditions_masks)
            self.negative_positive_conds.append(conditions_conds)
            self.negative_positive_extra.append(conditions_extra)
        self.conditioning_length = (len(new_negative), len(new_positive))

        new_model = model.clone()
        diffusion_model = new_model.model.diffusion_model

        if not hasattr(diffusion_model, "input_blocks") and hasattr(diffusion_model, "blocks"):
            self._patch_dit_blocks(new_model, diffusion_model)
            return (new_model, [new_positive[0]], [new_negative[0]])

        if not hasattr(diffusion_model, "input_blocks"):
            raise Exception(
                f"Attention Couple: unsupported diffusion model architecture '{type(diffusion_model).__name__}'. "
                f"This node supports UNet-style models (SD1.5/SDXL) and Cosmos-Predict2/Anima-style DiT models "
                f"(models exposing a flat `blocks` list)."
            )

        self.sdxl = hasattr(diffusion_model, "label_emb")
        if not self.sdxl:
            for id in [1,2,4,5,7,8]: # id of input_blocks that have cross attention
                set_model_patch_replace(new_model, self.make_patch(new_model.model.diffusion_model.input_blocks[id][1].transformer_blocks[0].attn2), ("input", id))
            set_model_patch_replace(new_model, self.make_patch(new_model.model.diffusion_model.middle_block[1].transformer_blocks[0].attn2), ("middle", 0))
            for id in [3,4,5,6,7,8,9,10,11]: # id of output_blocks that have cross attention
                set_model_patch_replace(new_model, self.make_patch(new_model.model.diffusion_model.output_blocks[id][1].transformer_blocks[0].attn2), ("output", id))
        else:
            for id in [4,5,7,8]: # id of input_blocks that have cross attention
                block_indices = range(2) if id in [4, 5] else range(10) # transformer_depth
                for index in block_indices:
                    set_model_patch_replace(new_model, self.make_patch(new_model.model.diffusion_model.input_blocks[id][1].transformer_blocks[index].attn2), ("input", id, index))
            for index in range(10):
                set_model_patch_replace(new_model, self.make_patch(new_model.model.diffusion_model.middle_block[1].transformer_blocks[index].attn2), ("middle", id, index))
            for id in range(6): # id of output_blocks that have cross attention
                block_indices = range(2) if id in [3, 4, 5] else range(10) # transformer_depth
                for index in block_indices:
                    set_model_patch_replace(new_model, self.make_patch(new_model.model.diffusion_model.output_blocks[id][1].transformer_blocks[index].attn2), ("output", id, index))
        
        return (new_model, [new_positive[0]], [new_negative[0]]) # pool outputは・・・後回し
    
    def make_patch(self, module):
        def patch(q, k, v, extra_options):
            
            len_neg, len_pos = self.conditioning_length # negative, positiveの長さ
            cond_or_uncond = extra_options["cond_or_uncond"] # 0: cond, 1: uncond
            q_list = q.chunk(len(cond_or_uncond), dim=0)
            b = q_list[0].shape[0] # batch_size
            
            masks_uncond = get_masks_from_q(self.negative_positive_masks[0], q_list[0], extra_options["original_shape"])
            masks_cond = get_masks_from_q(self.negative_positive_masks[1], q_list[0], extra_options["original_shape"])

            maxi_prompt_size_uncond = max([cond.shape[1] for cond in self.negative_positive_conds[0]])
            context_uncond = torch.cat([cond.repeat(1, maxi_prompt_size_uncond//cond.shape[1], 1) for cond in self.negative_positive_conds[0]], dim=0)

            maxi_prompt_size_cond = max([cond.shape[1] for cond in self.negative_positive_conds[1]])
            context_cond = torch.cat([cond.repeat(1, maxi_prompt_size_cond//cond.shape[1], 1) for cond in self.negative_positive_conds[1]], dim=0)
            
            k_uncond = module.to_k(context_uncond)
            k_cond = module.to_k(context_cond)
            v_uncond = module.to_v(context_uncond)
            v_cond = module.to_v(context_cond)

            out = []
            for i, c in enumerate(cond_or_uncond):
                if c == 0:
                    masks = masks_cond
                    k = k_cond
                    v = v_cond
                    length = len_pos
                else:
                    masks = masks_uncond
                    k = k_uncond
                    v = v_uncond
                    length = len_neg

                q_target = q_list[i].repeat(length, 1, 1)
                k = torch.cat([k[i].unsqueeze(0).repeat(b,1,1) for i in range(length)], dim=0)
                v = torch.cat([v[i].unsqueeze(0).repeat(b,1,1) for i in range(length)], dim=0)
                
                if k.dtype != q_target.dtype or v.dtype != q_target.dtype:
                    # Ensure all dtypes match
                    k = k.to(q_target.dtype)
                    v = v.to(q_target.dtype)
                qkv = optimized_attention(q_target, k, v, extra_options["n_heads"])
                qkv = qkv * masks
                qkv = qkv.view(length, b, -1, module.heads * module.dim_head).sum(dim=0)

                out.append(qkv)

            out = torch.cat(out, dim=0)
            return out
        return patch

    def _patch_dit_blocks(self, new_model, diffusion_model):
        self._adapt_conds_for_llm_adapter(diffusion_model)
        self._dit_state = {"cond_or_uncond": [0], "original_shape": None}

        def diffusion_model_wrapper(executor, *args, **kwargs):
            transformer_options = kwargs.get("transformer_options", {})
            x_arg = args[0] if len(args) > 0 else kwargs.get("x")
            self._dit_state["cond_or_uncond"] = transformer_options.get("cond_or_uncond", [0])
            if x_arg is not None:
                self._dit_state["original_shape"] = (
                    x_arg.shape[0], x_arg.shape[1], x_arg.shape[-2], x_arg.shape[-1]
                )
            return executor(*args, **kwargs)

        to = new_model.model_options.setdefault("transformer_options", {})
        wrappers = to.setdefault("wrappers", {})
        wrapper_bucket = wrappers.setdefault(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, {})
        wrapper_bucket.setdefault("attention_couple", []).append(diffusion_model_wrapper)

        for i, block in enumerate(diffusion_model.blocks):
            if not (hasattr(block, "cross_attn") and hasattr(block, "self_attn")):
                continue
            patched_forward = self.make_patch_dit(block.cross_attn)
            new_model.add_object_patch(f"diffusion_model.blocks.{i}.cross_attn.forward", patched_forward)

    def _adapt_conds_for_llm_adapter(self, diffusion_model):
        if not hasattr(diffusion_model, "preprocess_text_embeds"):
            return

        adapter_param = next(diffusion_model.llm_adapter.parameters())
        adapter_device = adapter_param.device
        adapter_dtype = adapter_param.dtype
        target_device = comfy.model_management.get_torch_device()
        target_dtype = diffusion_model.dtype

        for branch_idx in range(len(self.negative_positive_conds)):
            adapted_conds = []
            for cond, extra in zip(self.negative_positive_conds[branch_idx], self.negative_positive_extra[branch_idx]):
                t5xxl_ids = extra.get("t5xxl_ids")
                if t5xxl_ids is None:
                    adapted_conds.append(cond)
                    continue

                t5xxl_weights = extra.get("t5xxl_weights")
                if t5xxl_weights is not None:
                    t5xxl_weights = t5xxl_weights.unsqueeze(0).unsqueeze(-1).to(device=adapter_device, dtype=adapter_dtype)

                with torch.no_grad():
                    adapted = diffusion_model.preprocess_text_embeds(
                        cond.to(device=adapter_device, dtype=adapter_dtype),
                        t5xxl_ids.unsqueeze(0).to(device=adapter_device),
                        t5xxl_weights=t5xxl_weights,
                    )
                adapted_conds.append(adapted.to(device=target_device, dtype=target_dtype))
            self.negative_positive_conds[branch_idx] = adapted_conds

    def make_patch_dit(self, module):
        def patched_forward(x, context=None, rope_emb=None, transformer_options=None, **kwargs):
            transformer_options = transformer_options or {}
            len_neg, len_pos = self.conditioning_length
            cond_or_uncond = transformer_options.get(
                "cond_or_uncond", self._dit_state.get("cond_or_uncond", [0])
            )
            original_shape = self._dit_state.get("original_shape")
            if original_shape is None:
                original_shape = (x.shape[0], 1, x.shape[1], 1)

            n_branches = max(len(cond_or_uncond), 1)
            b_total, seq_len, x_dim = x.shape
            b = b_total // n_branches

            def pad_and_stack(conds):
                maxi_len = max(cond.shape[1] for cond in conds)
                padded = []
                for cond in conds:
                    if cond.shape[1] < maxi_len:
                        last_token = cond[:, -1:, :].expand(-1, maxi_len - cond.shape[1], -1)
                        cond = torch.cat([cond, last_token], dim=1)
                    padded.append(cond)
                return torch.cat(padded, dim=0)

            context_uncond = pad_and_stack(self.negative_positive_conds[0]).to(x.dtype)
            context_cond = pad_and_stack(self.negative_positive_conds[1]).to(x.dtype)

            x_list = x.chunk(n_branches, dim=0)
            masks_uncond = get_masks_from_q(self.negative_positive_masks[0], x_list[0], original_shape)
            masks_cond = get_masks_from_q(self.negative_positive_masks[1], x_list[0], original_shape)

            out = []
            for i, c in enumerate(cond_or_uncond):
                if c == 0:
                    masks, ctx_stack, length = masks_cond, context_cond, len_pos
                else:
                    masks, ctx_stack, length = masks_uncond, context_uncond, len_neg

                x_target = x_list[i].repeat(length, 1, 1)
                ctx_expanded = torch.cat(
                    [ctx_stack[j:j + 1].repeat(b, 1, 1) for j in range(length)], dim=0
                )

                q, k, v = module.compute_qkv(x_target, ctx_expanded, rope_emb=None)
                attn_out = manual_scaled_dot_product_attention(q, k, v)
                attn_out = module.output_dropout(module.output_proj(attn_out))

                if not getattr(self, f"_dit_debug_branch{i}_done", False):
                    setattr(self, f"_dit_debug_branch{i}_done", True)
                    print(
                        f"[AttentionCouple/DiT debug/branch{i}] c={c} length={length} "
                        f"q shape={tuple(q.shape)} k shape={tuple(k.shape)} "
                        f"attn_out nan={torch.isnan(attn_out).any().item()} "
                        f"attn_out std={attn_out.float().std().item():.6f}"
                    )

                attn_out = attn_out * masks
                attn_out = attn_out.view(length, b, seq_len, x_dim).sum(dim=0)

                out.append(attn_out)

            out = torch.cat(out, dim=0)

            if not getattr(self, "_dit_debug_done", False):
                self._dit_debug_done = True
                print(
                    f"[AttentionCouple/DiT debug] out stats: "
                    f"mean={out.float().mean().item():.6f} std={out.float().std().item():.6f} "
                    f"nan={torch.isnan(out).any().item()} inf={torch.isinf(out).any().item()}"
                )
            return out
        return patched_forward


NODE_CLASS_MAPPINGS = {
    "Attention couple": AttentionCouple
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Attention couple": "Load Attention couple",
}
