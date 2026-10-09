import os
import math
import torch
import warnings
from .model_minimind import *
from typing import Optional, Tuple, List, Union
from torch import nn
from transformers import SiglipImageProcessor, SiglipVisionModel
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

warnings.filterwarnings('ignore')


def pool_visual_tokens(features: torch.Tensor, target_tokens: int) -> torch.Tensor:
    """Spatial average-pooling baseline; the default 64 -> 64 path is identity.

    This reduces decoder input length, not the frozen encoder's input cost.
    It is an ablation baseline, not a learned selector or a quality guarantee.
    """
    source_tokens, dim = features.shape[-2:]
    source_side, target_side = math.isqrt(source_tokens), math.isqrt(target_tokens)
    if target_tokens < 1 or target_side ** 2 != target_tokens:
        raise ValueError('target_tokens must be a positive square')
    if source_side ** 2 != source_tokens or target_tokens > source_tokens:
        raise ValueError('Expected a square patch grid and no token upsampling')
    if source_tokens == target_tokens:
        return features
    grid = features.reshape(-1, source_side, source_side, dim).permute(0, 3, 1, 2)
    pooled = F.adaptive_avg_pool2d(grid, (target_side, target_side))
    return pooled.permute(0, 2, 3, 1).reshape(*features.shape[:-2], target_tokens, dim)


def fuse_visual_tokens(tokens, hidden_states, features, marker, image_mask=None):
    """Replace exactly N*T marker positions, preserving text and sequence length.

    Ordered marker positions, rather than runs, distinguish adjacent images.
    image_mask marks padded images in a variable-image-count batch.
    """
    if features.ndim == 3:
        features = features.unsqueeze(1)
    if features.ndim != 4 or features.shape[0] != tokens.shape[0]:
        raise ValueError('features must have shape [batch, images, tokens, hidden]')
    if hidden_states.shape[:2] != tokens.shape or features.shape[-1] != hidden_states.shape[-1]:
        raise ValueError('Text/vision embedding dimensions do not match')
    if image_mask is None:
        image_mask = (tokens == marker).any(-1, keepdim=True).expand(features.shape[:2])
    image_mask = image_mask.to(device=features.device, dtype=torch.bool)
    if image_mask.shape != features.shape[:2]:
        raise ValueError('image_mask must have shape [batch, images]')
    result = hidden_states.clone()
    for b in range(tokens.shape[0]):
        positions = (tokens[b] == marker).nonzero(as_tuple=True)[0]
        visual = features[b, image_mask[b]].flatten(0, 1)
        if positions.numel() != visual.shape[0]:
            raise ValueError(f'Sample {b}: {positions.numel()} image markers but '
                             f'{visual.shape[0]} visual tokens; check image count and truncation')
        result[b, positions] = visual.to(device=result.device, dtype=result.dtype)
    return result


class VLMConfig(MiniMindConfig):
    model_type = "minimind-v"

    def __init__(self, image_special_token='<|image_pad|>', image_ids=[12], **kwargs):
        self.image_special_token = image_special_token
        self.image_ids = image_ids
        self.image_hidden_size = kwargs.get("image_hidden_size", 768)
        self.image_token_len = kwargs.get("image_token_len", 64)
        if self.image_token_len not in (4, 16, 64):
            raise ValueError('image_token_len must be 4, 16, or 64')
        super().__init__(**kwargs)

class MMVisionProjector(nn.Module):
    def __init__(self, in_dim, out_dim, source_tokens=64, target_tokens=64):
        super().__init__()
        self.target_tokens = target_tokens
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
    def forward(self, x):
        return self.mlp(pool_visual_tokens(x, self.target_tokens))

# 继承自语言模型
class MiniMindVLM(MiniMindForCausalLM):
    config_class = VLMConfig

    def __init__(self, config: VLMConfig = None, vision_model_path="./model/siglip2-base-p32-256-ve"):
        self.config = config or VLMConfig()
        super().__init__(self.config)
        self.vision_encoder, self.processor = self.__class__.get_vision_model(vision_model_path)
        self.vision_proj = MMVisionProjector(self.config.image_hidden_size, self.config.hidden_size, target_tokens=self.config.image_token_len)

    @staticmethod
    def get_vision_model(model_path: str):
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
        if not os.path.exists(model_path):
            return None, None
        try:
            model = SiglipVisionModel.from_pretrained(model_path)
        except (RuntimeError, ValueError):
            return None, None
        processor = SiglipImageProcessor.from_pretrained(model_path)
        # 冻结 vision_encoder 的所有参数
        for param in model.parameters():
            param.requires_grad = False
        return model.eval(), processor

    @staticmethod
    def image2tensor(image, processor):
        if processor is None:
            raise RuntimeError('Vision processor is missing; download the SigLIP2 encoder first')
        image = image.convert('RGB')
        inputs = processor(images=image, return_tensors="pt")
        return inputs

    @staticmethod
    def get_image_embeddings(image_inputs, vision_model):
        if vision_model is None:
            raise RuntimeError('Vision encoder is missing; download the SigLIP2 encoder first')
        if not hasattr(image_inputs, 'keys'):
            image_inputs = {'pixel_values': image_inputs}
        parameter = next(vision_model.parameters())
        image_inputs = {k: v.to(device=parameter.device, dtype=parameter.dtype)
                        if v.is_floating_point() else v.to(parameter.device)
                        for k, v in image_inputs.items()}
        with torch.no_grad():
            outputs = vision_model(**image_inputs)
        return outputs.last_hidden_state

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, 'vision_encoder', None) is not None:
            self.vision_encoder.eval()
        return self

    def encode_images(self, pixel_values, image_mask=None):
        """Return projected [B,N,T,D] features, reusable for same-image inference.

        Cache only in eval mode and invalidate after changing weights or budget.
        Padded images and text-only dummy images can be masked out before ViT.
        """
        features, image_mask = self.encode_image_features(pixel_values, image_mask)
        return self.project_image_features(features, image_mask)

    def encode_image_features(self, pixel_values, image_mask=None):
        """Frozen pre-projector features [B,N,64,Dv] for shared teacher/student use.

        Unlike projected embeddings, these remain valid while the projector trains.
        The encoder, input pixels and preprocessing must remain identical.
        """
        if hasattr(pixel_values, 'keys'):
            image_mask = pixel_values.get('image_mask', image_mask)
            pixels = pixel_values['pixel_values']
        else:
            pixels = pixel_values
        if pixels.ndim == 6 and pixels.shape[2] == 1:
            pixels = pixels.squeeze(2)
        if pixels.ndim == 4:
            pixels = pixels.unsqueeze(1)
        if pixels.ndim != 5:
            raise ValueError('pixel_values must be [B,C,H,W] or [B,N,C,H,W]')
        batch, images = pixels.shape[:2]
        if image_mask is None:
            image_mask = torch.ones((batch, images), dtype=torch.bool, device=pixels.device)
        image_mask = image_mask.to(device=pixels.device, dtype=torch.bool)
        if image_mask.shape != (batch, images):
            raise ValueError('image_mask shape does not match pixel_values')
        parameter = next(self.vision_proj.parameters())
        result = parameter.new_zeros(batch, images, 64, self.config.image_hidden_size)
        if image_mask.any():
            features = self.get_image_embeddings(pixels[image_mask], self.vision_encoder)
            if features.shape[-2:] != (64, self.config.image_hidden_size):
                raise ValueError('BudgetLab requires the configured 64-patch vision encoder')
            result = result.to(device=features.device, dtype=features.dtype)
            result[image_mask.to(result.device)] = features
        return result, image_mask.to(result.device)

    def project_image_features(self, features, image_mask=None):
        """Project only real images; keep this operation inside forward for DDP."""
        if features.ndim != 4 or features.shape[-2:] != (64, self.config.image_hidden_size):
            raise ValueError('vision_features must be [B,N,64,image_hidden_size]')
        if image_mask is None:
            image_mask = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
        image_mask = image_mask.to(device=features.device, dtype=torch.bool)
        if image_mask.shape != features.shape[:2]:
            raise ValueError('image_mask shape does not match vision_features')
        parameter = next(self.vision_proj.parameters())
        result = parameter.new_zeros(*features.shape[:2], self.config.image_token_len, self.config.hidden_size)
        if image_mask.any():
            projected = self.vision_proj(features[image_mask].to(device=parameter.device, dtype=parameter.dtype))
            result = result.to(projected.dtype)
            result[image_mask.to(result.device)] = projected
        return result

    @torch.compiler.disable
    def count_vision_proj(self, tokens, h, vision_tensors=None, seqlen=512, image_mask=None):
        if vision_tensors is None or not self.config.image_ids:
            return h
        return fuse_visual_tokens(tokens, h, vision_tensors, self.config.image_ids[0], image_mask)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                labels: Optional[torch.Tensor] = None,
                pixel_values: Optional[torch.FloatTensor] = None,
                image_embeddings: Optional[torch.Tensor] = None,
                vision_features: Optional[torch.Tensor] = None,
                image_mask: Optional[torch.Tensor] = None,
                **args):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.model.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.model.dropout(self.model.embed_tokens(input_ids))

        if start_pos == 0:
            if sum(value is not None for value in (pixel_values, image_embeddings, vision_features)) > 1:
                raise ValueError('Pass only one of pixel_values, image_embeddings, vision_features')
            if vision_features is not None:
                image_embeddings = self.project_image_features(vision_features, image_mask)
            if pixel_values is not None:
                if hasattr(pixel_values, 'keys'):
                    image_mask = pixel_values.get('image_mask', image_mask)
                pixels = pixel_values['pixel_values'] if hasattr(pixel_values, 'keys') else pixel_values
                images = pixels.shape[1] if pixels.ndim >= 5 else 1
                if image_mask is None:
                    image_mask = (input_ids == self.config.image_ids[0]).any(-1, keepdim=True).expand(-1, images)
                image_embeddings = self.encode_images(pixels, image_mask)
            if image_embeddings is not None:
                hidden_states = self.count_vision_proj(input_ids, hidden_states, image_embeddings, image_mask=image_mask)
            elif (input_ids == self.config.image_ids[0]).any():
                raise ValueError('Image markers require pixel_values or image_embeddings')

        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        if self.model.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            self.model.freqs_cos, self.model.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
        position_embeddings = (
            self.model.freqs_cos[start_pos:start_pos + seq_length],
            self.model.freqs_sin[start_pos:start_pos + seq_length]
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.model.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        hidden_states = self.model.norm(hidden_states)

        aux_loss = sum([l.mlp.aux_loss for l in self.model.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        aux_loss = aux_loss + sum(p.sum() for p in self.vision_proj.parameters()) * 0  # dummy gradient for DDP
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

        output = MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=presents, hidden_states=hidden_states)
        return output

    def generate(self, *args, num_return_sequences=1, **kwargs):
        if num_return_sequences > 1 and 'pixel_values' in kwargs:
            pv = kwargs['pixel_values']
            if hasattr(pv, 'keys'):
                kwargs['pixel_values'] = {k: v.repeat(num_return_sequences, *([1] * (v.ndim - 1))) for k, v in pv.items()}
            else:
                kwargs['pixel_values'] = pv.repeat(num_return_sequences, *([1] * (pv.ndim - 1)))
        for key in ('image_embeddings', 'vision_features', 'image_mask'):
            if num_return_sequences > 1 and kwargs.get(key) is not None:
                value = kwargs[key]
                kwargs[key] = value.repeat(num_return_sequences, *([1] * (value.ndim - 1)))
        return super().generate(*args, num_return_sequences=num_return_sequences, **kwargs)