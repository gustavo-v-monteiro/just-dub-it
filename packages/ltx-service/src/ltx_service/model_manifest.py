from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HuggingFaceFileArtifact:
    name: str
    repo_id: str
    filename: str
    relative_path: Path


@dataclass(frozen=True)
class HuggingFaceSnapshotArtifact:
    name: str
    repo_id: str
    relative_path: Path
    required_markers: tuple[str, ...]


CHECKPOINT_ARTIFACT = HuggingFaceFileArtifact(
    name="checkpoint",
    repo_id="Lightricks/LTX-2",
    filename="ltx-2-19b-dev.safetensors",
    relative_path=Path("checkpoint") / "ltx-2-19b-dev.safetensors",
)

JUSTDUBIT_LORA_ARTIFACT = HuggingFaceFileArtifact(
    name="justdubit_lora",
    repo_id="justdubit/justdubit",
    filename="ltx-2-19b-ic-lora-lipdubbing.safetensors",
    relative_path=Path("loras") / "ltx-2-19b-ic-lora-lipdubbing.safetensors",
)

DISTILLED_LORA_ARTIFACT = HuggingFaceFileArtifact(
    name="distilled_lora",
    repo_id="Lightricks/LTX-2",
    filename="ltx-2-19b-distilled-lora-384.safetensors",
    relative_path=Path("loras") / "ltx-2-19b-distilled-lora-384.safetensors",
)

SPATIAL_UPSAMPLER_ARTIFACT = HuggingFaceFileArtifact(
    name="spatial_upsampler",
    repo_id="Lightricks/LTX-2",
    filename="ltx-2-spatial-upscaler-x2-1.0.safetensors",
    relative_path=Path("upscaler") / "ltx-2-spatial-upscaler-x2-1.0.safetensors",
)

GEMMA_ARTIFACT = HuggingFaceSnapshotArtifact(
    name="gemma_root",
    repo_id="google/gemma-3-12b-it-qat-q4_0-unquantized",
    relative_path=Path("gemma") / "gemma-3-12b-it-qat-q4_0-unquantized",
    required_markers=("config.json",),
)

GEMMA_PUBLIC_MIRROR_REPO_ID = "DeepBeepMeep/LTX-2"
GEMMA_PUBLIC_MIRROR_PREFIX = "gemma-3-12b-it-qat-q4_0-unquantized"
GEMMA_PUBLIC_MIRROR_REQUIRED_FILES = (
    "added_tokens.json",
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "gemma-3-12b-it-qat-q4_0-unquantized.safetensors",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
)

FILE_ARTIFACTS = (
    CHECKPOINT_ARTIFACT,
    JUSTDUBIT_LORA_ARTIFACT,
    DISTILLED_LORA_ARTIFACT,
    SPATIAL_UPSAMPLER_ARTIFACT,
)

SNAPSHOT_ARTIFACTS = (GEMMA_ARTIFACT,)
