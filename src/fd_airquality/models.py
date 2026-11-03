"""Forecast adapters with a common (batch, context) -> (batch, horizon) interface."""

from __future__ import annotations

import re
from contextlib import nullcontext
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn

from fd_airquality.adapters import (
    ADAPTER_IMPLEMENTATION,
    LoRAFactors,
    LoRALinear,
    layer_seed,
    target_linears,
)
from fd_airquality.settings import RunSettings

# Immutable Hugging Face revisions; never silently substitute R2 for TTM-R3.
GRANITE = {
    "ttm-r3": (
        "ibm-granite/granite-timeseries-ttm-r3",
        "417bfc83545095031c2a002ddfe73ac6b700ea1a",
        "TinyTimeMixerForDecomposedPrediction",
    ),
    "patchtst-fm-r2": (
        "ibm-granite/granite-timeseries-patchtst-fm-r2",
        "b125275f9204d37cbb81fe47b9cf5e08a521e829",
        "PatchTSTFMForPrediction",
    ),
    "flowstate-r1": (
        "ibm-granite/granite-timeseries-flowstate-r1",
        "05effc6cb39ee16dce9dd0064ed1a76e4b8ff464",
        "FlowStateForPrediction",
    ),
}
MODEL_NAMES = ("mlp", "autoencoder", "darts-nlinear", *GRANITE)


class ForecastMLP(nn.Module):
    """Direct residual forecast, initialized to persistence."""

    def __init__(self, context: int, horizon: int, hidden_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(context, hidden_size), nn.ReLU(), nn.Linear(hidden_size, horizon)
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        anchor = inputs[:, -1:]
        return anchor + self.network(inputs - anchor)


class ForecastAutoencoder(nn.Module):
    """Shared bottleneck for context reconstruction and supervised forecasting."""

    def __init__(self, context: int, horizon: int, hidden: int, latent: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(context, hidden), nn.ReLU(), nn.Linear(hidden, latent))
        self.reconstruction = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, context))
        self.forecast = nn.Linear(latent, horizon)
        nn.init.zeros_(self.forecast.weight)
        nn.init.zeros_(self.forecast.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        anchor = inputs[:, -1:]
        return anchor + self.forecast(self.encoder(inputs - anchor))

    def reconstruction_loss(self, inputs: torch.Tensor) -> torch.Tensor:
        centered = inputs - inputs[:, -1:]
        return nn.functional.mse_loss(self.reconstruction(self.encoder(centered)), centered)


class DartsNLinear(nn.Module):
    """Use the actual Darts NLinear module, not a reimplementation of its formula."""

    def __init__(self, context: int, horizon: int):
        super().__init__()
        import numpy as np
        from darts.models import NLinearModel
        from darts.utils.data.torch_datasets.utils import TorchTrainingSample

        owner = NLinearModel(
            input_chunk_length=context, output_chunk_length=horizon,
            shared_weights=True, normalize=True, use_static_covariates=False,
        )
        # Darts exposes module construction through this version-sensitive hook.
        # No synthetic samples are fitted; these arrays specify shapes only.
        self.module = owner._create_model(TorchTrainingSample(
            past_target=np.zeros((context, 1), dtype=np.float32),
            future_target=np.zeros((horizon, 1), dtype=np.float32),
        ))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        from darts.utils.data.torch_datasets.utils import ModuleStage, PLModuleInput

        stage = ModuleStage.TRAIN if self.training else ModuleStage.PREDICT
        output = self.module(PLModuleInput(past_target=inputs.unsqueeze(-1), stage=stage))
        return output.prediction[:, :, 0, 0]


def validate_model_geometry(geometry: dict, settings: RunSettings) -> None:
    context, horizon = geometry["context"], geometry["horizon"]
    if settings.model == "ttm-r3" and (context != 90 or horizon > 30):
        raise ValueError(
            "El checkpoint TTM-R3 fijado exige contexto 90 y horizonte <= 30. "
            "Ejecuta prepare --context 90 y usa ese mismo manifest para todos los modelos; "
            "no se rellena ni inventa historia."
        )
    if settings.model == "patchtst-fm-r2" and not 16 <= context <= 8192:
        raise ValueError("PatchTST-FM-R2 requiere aquí contexto entre 16 y 8192.")
    if settings.model == "flowstate-r1":
        patch = int(24 / settings.flowstate_scale_factor + 1e-6)
        maximum = min(16384, int(2048 / settings.flowstate_scale_factor)) - max(0, horizon - patch)
        if patch < 1 or context > maximum:
            raise ValueError(
                "FlowState-R1 recortaría el contexto con esta escala/horizonte; "
                f"context debe ser <= {maximum}. No se recorta silenciosamente."
            )


def _patchtst_autocast_context(device: torch.device):
    """Keep upstream BF16 only where CUDA provides native hardware support."""
    if device.type not in {"cuda", "mps"}:
        return nullcontext()
    enabled = True
    if device.type == "cuda":
        with torch.cuda.device(device):
            # Tensor emulation is not enough for cuDNN's BF16 convolutions.
            enabled = torch.cuda.is_bf16_supported(including_emulation=False)
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)


@lru_cache(maxsize=1)
def _load_granite(name: str, revision: str) -> nn.Module:
    import inspect

    import tsfm_public

    identifier, _, class_name = GRANITE[name]
    if name == "ttm-r3":
        parameters = inspect.signature(tsfm_public.TinyTimeMixerConfig.__init__).parameters
        if "decompose" not in parameters:
            raise RuntimeError("Esta versión de granite-tsfm no implementa TTM-R3; ejecuta uv sync --locked.")
    elif name == "patchtst-fm-r2":
        from tsfm_public.models.patchtst_fm import modeling_patchtst_fm

        # PatchTST opens its own nested autocast; an outer FP32 context cannot
        # override it. Replace only this model's precision helper, not cuDNN.
        modeling_patchtst_fm.get_autocast_context = _patchtst_autocast_context
    model_class = getattr(tsfm_public, class_name)
    backbone, info = model_class.from_pretrained(
        identifier, revision=revision, use_safetensors=True, output_loading_info=True,
    )
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        counts = {key: len(value) for key, value in info.items() if value}
        raise RuntimeError(f"Checkpoint Granite incompatible; no se aceptan pesos aleatorios: {counts}")
    return backbone.requires_grad_(False).eval()


def effective_adaptation(settings: RunSettings) -> str:
    return ("lora" if settings.model in GRANITE else "full") if settings.adaptation == "auto" else settings.adaptation


def validate_run_mode(settings: RunSettings, mode: str) -> None:
    algorithm = effective_adaptation(settings)
    if settings.model not in GRANITE:
        if algorithm != "full" or mode == "zero-shot":
            raise ValueError("Los modelos clásicos se entrenan completos; no admiten LoRA ni zero-shot.")
        return
    if mode == "zero-shot":
        if algorithm != "none":
            raise ValueError("zero-shot exige adaptation=none.")
    elif algorithm == "none":
        raise ValueError("adaptation=none está reservado a zero-shot.")
    if algorithm in {"flora", "lora-fair"} and (mode != "federated" or settings.strategy != "fedavg"):
        raise ValueError("FLoRA/LoRA-FAIR especifican su agregación y solo se ejecutan en federación, sin FedProx.")


def _template(settings):
    with torch.random.fork_rng(devices=[]):
        return _load_granite(settings.model, settings.hf_revision or GRANITE[settings.model][1]).requires_grad_(False).eval()


class GraniteForecaster(nn.Module):
    """Pinned pretrained weights with internal adapters, not an output-correction MLP."""

    def __init__(self, geometry: dict, settings: RunSettings, pretrained: bool = True):
        super().__init__()
        self.name, self.horizon = settings.model, geometry["horizon"]
        self.algorithm = effective_adaptation(settings)
        self.scale_factor = settings.flowstate_scale_factor
        self.revision = settings.hf_revision or GRANITE[self.name][1]
        template = _template(settings)
        config = template.config
        if self.name == "ttm-r3" and (geometry["context"] != config.context_length or self.horizon > config.prediction_length):
            raise ValueError("La geometría no coincide con la revisión TTM-R3 solicitada.")
        if self.name == "patchtst-fm-r2" and geometry["context"] > config.context_length:
            raise ValueError("PatchTST recortaría el contexto con esta revisión.")
        if self.name == "flowstate-r1":
            patch = int(config.decoder_patch_len / self.scale_factor + 1e-6)
            maximum = min(16384, int(config.context_length / self.scale_factor)) - max(0, self.horizon - patch)
            if patch < 1 or geometry["context"] > maximum:
                raise ValueError("FlowState recortaría el contexto con esta revisión/escala/horizonte.")
        # Share ONLY immutable pretrained Parameters. Modules, normalization state,
        # adapters and merged FLoRA updates must be independent in every instance.
        self.backbone = deepcopy(template, {id(p): nn.Parameter(p.detach(), requires_grad=False) for p in template.parameters()}) if pretrained else None
        self.factors = nn.ModuleDict()
        self.layer_paths = []
        self.layers = {}
        self.history = {}
        if self.algorithm != "none":
            for i, (path, linear) in enumerate(target_linears(template, self.name, settings.lora_targets).items()):
                factor = LoRAFactors(linear.in_features, linear.out_features, rank=settings.lora_rank,
                                     alpha=settings.lora_alpha, dropout=settings.lora_dropout,
                                     freeze_a=self.algorithm == "ffa-lora", seed=layer_seed(settings.seed, path))
                self.factors[str(i)] = factor
                self.layer_paths.append(path)
                if self.backbone is not None:
                    original = self.backbone.get_submodule(path)
                    layer = LoRALinear(original, factor)
                    parent, _, child = path.rpartition(".")
                    setattr(self.backbone.get_submodule(parent) if parent else self.backbone, child, layer)
                    self.layers[path] = layer
        self.train(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbone is not None:
            self.backbone.eval()
        self.factors.train(mode)
        return self

    def export_state(self, *, include_fixed: bool = False) -> dict[str, torch.Tensor]:
        result = {}
        for path, factor in zip(self.layer_paths, self.factors.values(), strict=True):
            if factor.A.requires_grad or include_fixed:
                result[f"adapter.{path}.lora_A"] = factor.A
            result[f"adapter.{path}.lora_B"] = factor.B
        return result | self.history

    def import_state(self, state: dict[str, torch.Tensor]) -> None:
        expected = {key: value for key, value in self.export_state().items() if key.startswith("adapter.")}
        current = {key: value for key, value in state.items() if key.startswith("adapter.")}
        if set(current) != set(expected):
            raise ValueError("Claves de estado LoRA incompatibles.")
        history = {key: value for key, value in state.items() if key.startswith("history.")}
        if set(state) != set(current) | set(history) or (history and self.algorithm != "flora"):
            raise ValueError("Estado de adaptación desconocido o historia no permitida.")
        with torch.no_grad():
            for key, value in current.items():
                if value.shape != expected[key].shape or value.dtype != expected[key].dtype or not torch.isfinite(value).all():
                    raise ValueError(f"Tensor de adaptación incompatible: {key}")
                expected[key].copy_(value)
        rounds = set()
        for key in history:
            match = re.fullmatch(r"history\.(\d{6})\.(.+)\.lora_([AB])", key)
            if match is None or match[2] not in self.layer_paths:
                raise ValueError("Historia FLoRA incompatible.")
            rounds.add(int(match[1]))
        if rounds and rounds != set(range(1, max(rounds) + 1)):
            raise ValueError("Historia FLoRA incompleta.")
        if len(history) != len(rounds) * len(self.layer_paths) * 2:
            raise ValueError("Historia FLoRA incompleta.")
        for path, factor in zip(self.layer_paths, self.factors.values(), strict=True):
            merged = None
            for round_number in sorted(rounds):
                prefix = f"history.{round_number:06d}.{path}.lora_"
                a, b = history[prefix + "A"], history[prefix + "B"]
                if (a.ndim != 2 or b.ndim != 2 or a.shape[1] != factor.A.shape[1]
                        or b.shape != (factor.B.shape[0], a.shape[0]) or a.dtype != factor.A.dtype
                        or b.dtype != factor.B.dtype or not torch.isfinite(a).all() or not torch.isfinite(b).all()):
                    raise ValueError("Pareja de factores en historia FLoRA incompatible.")
                if self.backbone is not None:
                    update = (b.detach().to(factor.B.device) @ a.detach().to(factor.A.device)) * factor.scale
                    merged = update if merged is None else merged + update
            if self.backbone is not None:
                self.layers[path].merged_delta = merged
        self.history = {key: value.detach().cpu().clone() for key, value in history.items()}

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.backbone is None:
            raise RuntimeError("El servidor solo construye el estado de adaptación; la inferencia pertenece al cliente.")
        options = {"past_values": inputs.unsqueeze(-1), "return_dict": True, "return_loss": False}
        if self.name != "ttm-r3":
            options["prediction_length"] = self.horizon
        if self.name == "flowstate-r1":
            options.update(batch_first=True, scale_factor=self.scale_factor, prediction_type="median")
        output = self.backbone(**options)
        prediction = output.prediction_outputs
        if self.name == "patchtst-fm-r2":
            # Upstream wraps only its point integration in no_grad. Reproduce the
            # SAME integration with autograd, retaining the pretrained quantiles.
            samples = output.quantile_outputs
            levels = torch.as_tensor(self.backbone.config.quantile_levels, device=samples.device, dtype=samples.dtype)
            prediction = torch.trapezoid(samples, levels, dim=1)
            prediction = prediction + samples[:, 0] * levels[0] + samples[:, -1] * (1 - levels[-1])
        if prediction.ndim != 3 or prediction.shape[0] != len(inputs) or prediction.shape[-1] != 1:
            raise ValueError(f"Forma Granite no soportada: {tuple(prediction.shape)}")
        prediction = prediction[:, :self.horizon, 0]
        if prediction.shape != (len(inputs), self.horizon):
            raise ValueError("El checkpoint no cubre el horizonte solicitado.")
        return prediction


def make_model(geometry: dict, settings: RunSettings, *, pretrained: bool = True) -> nn.Module:
    validate_model_geometry(geometry, settings)
    context, horizon = geometry["context"], geometry["horizon"]
    if settings.model == "mlp":
        return ForecastMLP(context, horizon, settings.hidden_size)
    if settings.model == "autoencoder":
        return ForecastAutoencoder(context, horizon, settings.hidden_size, settings.latent_size)
    if settings.model == "darts-nlinear":
        return DartsNLinear(context, horizon)
    return GraniteForecaster(geometry, settings, pretrained=pretrained)


def shared_state(model: nn.Module) -> dict[str, torch.Tensor]:
    # The immutable pretrained backbone, including FlowState complex buffers, is
    # never averaged or transmitted by Flower. Only adaptation weights travel.
    return model.export_state() if isinstance(model, GraniteForecaster) else model.state_dict()


def load_shared_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    if isinstance(model, GraniteForecaster):
        model.import_state(state)
    else:
        model.load_state_dict(state, strict=True)


def model_metadata(geometry: dict, settings: RunSettings) -> dict:
    metadata = {"name": settings.model, "context": geometry["context"], "horizon": geometry["horizon"]}
    if settings.model in GRANITE:
        identifier, revision, _ = GRANITE[settings.model]
        metadata.update(
            repository=identifier, revision=settings.hf_revision or revision,
            adaptation=effective_adaptation(settings), implementation=ADAPTER_IMPLEMENTATION, point_forecast=(
                "median" if settings.model == "flowstate-r1" else "upstream_point_prediction"
            ),
        )
        metadata["trainable_parameters"] = 0
        metadata["base_parameters"] = sum(parameter.numel() for parameter in _template(settings).parameters())
        if effective_adaptation(settings) != "none":
            targets = target_linears(_template(settings), settings.model, settings.lora_targets)
            if any(settings.lora_rank > min(layer.in_features, layer.out_features) for layer in targets.values()):
                raise ValueError("El rango LoRA excede las dimensiones de una capa target; no se recorta.")
            metadata["target_modules"] = {path: [layer.in_features, layer.out_features] for path, layer in targets.items()}
            metadata["trainable_parameters"] = sum(settings.lora_rank * (layer.out_features + (0 if effective_adaptation(settings) == "ffa-lora" else layer.in_features)) for layer in targets.values())
            metadata["base_parameters"] = sum(parameter.numel() for parameter in _template(settings).parameters())
            metadata["flowstate_adaptation_experimental"] = settings.model == "flowstate-r1"
    return metadata


def checkpoint_payload(geometry: dict, settings: RunSettings, state: dict, round_number: int) -> dict:
    fixed = {}
    if settings.model in GRANITE and effective_adaptation(settings) == "ffa-lora":
        server = make_model(geometry, settings, pretrained=False)
        fixed = {key: value.detach().cpu().clone() for key, value in server.export_state(include_fixed=True).items() if key.endswith("lora_A")}
    return {
        "checkpoint_version": 2, "model_state_dict": {key: value.detach().cpu().clone() for key, value in state.items()},
        "fixed_adapter_state": fixed, "round": round_number, "geometry": geometry,
        "hidden_size": settings.hidden_size, "settings": settings.flower_config(), "model": model_metadata(geometry, settings),
    }


def restore_checkpoint(checkpoint: str | Path | dict) -> nn.Module:
    """Restore exact adapter state (including FLoRA history and fixed FFA A)."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True) if not isinstance(checkpoint, dict) else checkpoint
    if payload.get("checkpoint_version") != 2:
        raise ValueError("Checkpoint antiguo incompatible con la adaptación interna.")
    settings = RunSettings.from_mapping(payload["settings"])
    if payload["model"] != model_metadata(payload["geometry"], settings):
        raise ValueError("Metadatos/capas/revisión del checkpoint incompatibles.")
    model = make_model(payload["geometry"], settings)
    if isinstance(model, GraniteForecaster) and model.algorithm == "ffa-lora":
        fixed = payload["fixed_adapter_state"]
        expected = {key: value for key, value in model.export_state(include_fixed=True).items() if key.endswith("lora_A")}
        if set(fixed) != set(expected):
            raise ValueError("Checkpoint FFA sin matrices A congeladas completas.")
        with torch.no_grad():
            for key in fixed:
                if fixed[key].shape != expected[key].shape or not torch.isfinite(fixed[key]).all():
                    raise ValueError("Matriz A congelada incompatible.")
                expected[key].copy_(fixed[key])
    load_shared_state(model, payload["model_state_dict"])
    return model.eval()
