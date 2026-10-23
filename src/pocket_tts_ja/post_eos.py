"""Supervising the frames after EOS: `TrainableTTS.forward` with a three-valued mask.

The loader (pocket_tts_ja.dv_loader) appends the teacher's tail after each target and marks the
frames uint8: 1 speech, 2 the tail after it, 0 padding. Here the EOS target and the EOS loss stay
on the speech (1 on its first frame after, as upstream), the flow loss covers the speech and the
tail, and the latent statistics see the speech only. With a bool mask this is the upstream
forward. Ported from main, now through upstream's `backbone_z` (v4.0.1 added `z_proj`, which
main's copy of the backbone pass skipped).

Why: the stock wheel generates frames_after_eos more frames after EOS fires and plays them, but
upstream training never supervises a frame after the first invalid one.

With a distillation teacher attached (distill_cfg_coef, scripts/train/train_student.py) this is
upstream's latent CFG distillation over the speech and the tail: the backbone regresses onto the
frozen teacher's guidance-combined output, and the flow and EOS heads (frozen) see it as they would
see CFG at generation.

P3c (`TrainableTTS.ja_context_noise`, a ContextNoise): in training, a share of the rows sees its
past latents (the backbone's input) with Gaussian noise of a per-row level, while the flow
targets stay clean, so a generation that drifts from the teacher's latents (the long silences
the student falls into at punctuation) still sees inputs like its training ones. The voice
prompt and a distillation teacher's inputs get the same noisy past.
"""

from dataclasses import dataclass
from functools import partial

import torch
import torch.nn.functional as F

from training.modules.model import TrainableTTS


@dataclass
class ContextNoise:
    prob: float = 0.5  # share of the rows with noise
    max_std: float = 0.5  # per-row std drawn uniformly in [0, max_std], in normalized latent units


def forward_with_post_eos(
    self: TrainableTTS,
    latents: torch.Tensor,  # [B, T, C] raw codec latents
    mask: torch.Tensor,  # [B, T] bool valid positions, or uint8 1 speech / 2 tail after EOS
    text_tokens: list[torch.Tensor],
    voice_latents: torch.Tensor,  # [B, Tv, C] raw codec latents
    update_stats: bool = False,
    num_voice_prompt_frames: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    assert self.head_teacher is None, "head distillation is not supported here"
    if mask.dtype == torch.bool:
        speech = flow_mask = mask
    else:
        speech, flow_mask = mask == 1, mask > 0
    fl = self.flow_lm
    C = latents.shape[-1]
    if update_stats and self.training:
        self._update_latent_stats(latents, speech)
    normalized_latents = (latents - fl.emb_mean) / fl.emb_std
    past = normalized_latents
    noise_spec: ContextNoise | None = getattr(self, "ja_context_noise", None)
    if noise_spec is not None and self.training:
        rows = torch.rand(latents.shape[0], 1, 1, device=latents.device) < noise_spec.prob
        std = noise_spec.max_std * torch.rand(latents.shape[0], 1, 1, device=latents.device) * rows
        past = normalized_latents + std.to(latents.dtype) * torch.randn_like(normalized_latents)
    inputs = (past, text_tokens, voice_latents, num_voice_prompt_frames)
    z = self.backbone_z(fl, *inputs, cfg_dropout=self.training)

    if self.distill_teacher is not None:
        teacher = self.distill_teacher.to(latents.device)
        with torch.no_grad():
            z_t = self.backbone_z(teacher, *inputs, cfg_dropout=False)
            if self.args.distill_cfg_coef != 1.0:
                z_null = self.backbone_z(teacher, *inputs, cfg_dropout=False, force_null=True)
                z_t = z_null + self.args.distill_cfg_coef * (z_t - z_null)
        # every supervised frame and the one after the tail, whose EOS logit stops a generation
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
