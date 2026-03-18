from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from arena_ranker.config import ModelConfig
from arena_ranker.hf import load_encoder


def masked_mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    masked = last_hidden_state * mask
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return masked.sum(dim=1) / denom


@dataclass(slots=True)
class ModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None


class PreferenceClassifier(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = load_encoder(config)
        hidden_size = self.encoder.config.hidden_size
        classifier_input = hidden_size * 6
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_size * 2, 3),
        )
        self.loss_fn = nn.CrossEntropyLoss()

        if config.freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def print_trainable_parameters(self) -> None:
        total_params = 0
        trainable_params = 0
        for parameter in self.parameters():
            count = parameter.numel()
            total_params += count
            if parameter.requires_grad:
                trainable_params += count

        ratio = 0.0 if total_params == 0 else trainable_params / total_params * 100
        print(
            f"trainable params: {trainable_params} || all params: {total_params} || "
            f"trainable%: {ratio:.4f}"
        )

    def enable_gradient_checkpointing(self) -> None:
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()
        elif hasattr(self.encoder, "base_model") and hasattr(self.encoder.base_model, "gradient_checkpointing_enable"):
            self.encoder.base_model.gradient_checkpointing_enable()

        if hasattr(self.encoder, "enable_input_require_grads"):
            self.encoder.enable_input_require_grads()
        elif hasattr(self.encoder, "base_model") and hasattr(self.encoder.base_model, "enable_input_require_grads"):
            self.encoder.base_model.enable_input_require_grads()

        encoder_config = getattr(self.encoder, "config", None)
        if encoder_config is not None and hasattr(encoder_config, "use_cache"):
            encoder_config.use_cache = False

        base_model = getattr(self.encoder, "base_model", None)
        base_config = getattr(base_model, "config", None)
        if base_config is not None and hasattr(base_config, "use_cache"):
            base_config.use_cache = False

    def encode(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.encoder(**inputs)
        return masked_mean_pool(outputs.last_hidden_state, inputs["attention_mask"])

    def forward(
        self,
        prompt_inputs: dict[str, torch.Tensor],
        response_a_inputs: dict[str, torch.Tensor],
        response_b_inputs: dict[str, torch.Tensor],
        labels: torch.Tensor | None = None,
    ) -> ModelOutput:
        prompt_emb = self.encode(prompt_inputs)
        response_a_emb = self.encode(response_a_inputs)
        response_b_emb = self.encode(response_b_inputs)

        features = torch.cat(
            [
                prompt_emb,
                response_a_emb,
                response_b_emb,
                response_a_emb - response_b_emb,
                response_a_emb - prompt_emb,
                response_b_emb - prompt_emb,
            ],
            dim=-1,
        )
        logits = self.classifier(self.dropout(features))

        loss = self.loss_fn(logits, labels) if labels is not None else None
        return ModelOutput(logits=logits, loss=loss)
