"""Supervising the frames after EOS: `TrainableTTS.forward` with a three-valued mask.

The loader can append digital silence after each target (pocket_tts_ja.loader, post_eos_frames)
and then marks the frames uint8: 1 speech, 2 the silence after it, 0 padding. Here the EOS target
and the EOS loss stay on the speech (1 on its first frame after, as upstream), the flow loss and
the distillation loss cover the speech and the silence, and the latent statistics see the speech
only. With a bool mask this is the upstream forward. scripts/train_ja.py swaps it in.

Why: the stock wheel generates frames_after_eos more frames after EOS fires (5 for Japanese text)
and plays them, but upstream training never supervises a frame after the first invalid one. What
the model says there is whatever it extrapolates; the CC-BY teacher extrapolates "うん" or "えー"
in about 6% of JKYB-mini sentences (docs/plans/ja_finetune_plan.md, B-CC).
"""

from functools import partial

import torch
import torch.nn.functional as F

from pocket_tts.models.flow_lm import FlowLMModel
from training.modules.conditioner import build_sequences_with_conditions
from training.modules.model import TrainableTTS


def forward_with_post_eos(
    self: TrainableTTS,
    latents: torch.Tensor,  # [B, T, C] raw codec latents
    mask: torch.Tensor,  # [B, T] bool valid positions, or uint8 1 speech / 2 silence after EOS
    text_tokens: list[torch.Tensor],
    voice_latents: torch.Tensor,  # [B, Tv, C] raw codec latents
    update_stats: bool = False,
    num_voice_prompt_frames: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if mask.dtype == torch.bool:
        speech = flow_mask = mask
    else:
        speech, flow_mask = mask == 1, mask > 0
    fl = self.flow_lm
    _, T, C = latents.shape
    if update_stats and self.training:
        self._update_latent_stats(latents, speech)
    normalized_latents = (latents - fl.emb_mean) / fl.emb_std

    def backbone_z(module: FlowLMModel, cfg_dropout: bool, force_null: bool = False) -> torch.Tensor:
        x, prefix_lengths = build_sequences_with_conditions(
            self.args,
            normalized_latents,
            text_tokens,
            voice_latents,
            cfg_dropout=cfg_dropout,
            num_voice_prompt_frames=num_voice_prompt_frames,
            fl=module,
            force_null=force_null,
        )
        out = module.out_norm(module.transformer(x, model_state=None))
        idx = prefix_lengths[:, None] + torch.arange(T, device=out.device)[None, :]
        return out.gather(1, idx[:, :, None].expand(-1, -1, out.shape[-1]))

    z = backbone_z(fl, cfg_dropout=self.training)  # [B, T, dim]

    if self.distill_teacher is not None:
        teacher = self.distill_teacher.to(latents.device)
        with torch.no_grad():
            z_cond = backbone_z(teacher, cfg_dropout=False)
            z_null = backbone_z(teacher, cfg_dropout=False, force_null=True)
            z_t = z_null + self.args.distill_cfg_coef * (z_cond - z_null)
        shifted = torch.cat([flow_mask[:, :1], flow_mask[:, :-1]], dim=1)
        loss = ((z - z_t).square().mean(dim=-1) * shifted).sum() / shifted.sum().clamp(min=1)
        return loss, {"distill_mse": loss.detach(), "loss": loss.detach()}

    is_eos = ~speech
    is_eos[:, 0] = False
    eos_logits = fl.out_eos(z).squeeze(-1)
    shifted_speech = torch.cat([speech[:, :1], speech[:, :-1]], dim=1)
    eos_loss = is_eos * F.softplus(-eos_logits) + ~is_eos * F.softplus(eos_logits)
    eos_loss = (eos_loss * shifted_speech).sum() / shifted_speech.sum().clamp(min=1)

    fbm = self.args.flow_batch_multiplier
    z_flat = z.unsqueeze(0).expand(fbm, -1, -1, -1).reshape(-1, z.shape[-1])
    target_flat = normalized_latents.unsqueeze(0).expand(fbm, -1, -1, -1).reshape(-1, C)
    mask_flat = flow_mask.unsqueeze(0).expand(fbm, -1, -1).reshape(-1)
    sel_z = z_flat[mask_flat]
    sel_target = target_flat[mask_flat]
    noise = torch.randn_like(sel_target)
    if getattr(self.flow, "fp32_loss", False):
        with torch.autocast(device_type=latents.device.type, enabled=False):
            flow_loss, metrics, _ = self.flow.loss(partial(fl.flow_net, sel_z.float()), noise.float(), sel_target.float())
    else:
        flow_loss, metrics, _ = self.flow.loss(partial(fl.flow_net, sel_z), noise, sel_target)
    flow_loss = flow_loss.mean()

    loss = flow_loss + self.args.eos_loss_weight * eos_loss
    metrics = {k: v.detach() if torch.is_tensor(v) else v for k, v in metrics.items()}
    metrics.update(flow_loss=flow_loss.detach(), eos_loss=eos_loss.detach(), loss=loss.detach())
    return loss, metrics
