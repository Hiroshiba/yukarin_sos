from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from yukarin_sos.utility import dataclass_utility
from yukarin_sos.utility.git_utility import get_branch_name, get_commit_id


@dataclass
class DatasetConfig:
    f0_glob: str
    phoneme_glob: str
    phoneme_list_glob: str
    speaker_dict_path: Optional[Path]
    speaker_size: Optional[int]
    test_num: int
    test_trial_num: int = 1
    seed: int = 0


@dataclass
class NetworkConfig:
    phoneme_size: int
    phoneme_embedding_size: int
    speaker_size: int
    speaker_embedding_size: int
    transformer_hidden_size: int
    tranformer_head_num: int
    transformer_encoder_layer_num: int
    transformer_decoder_layer_num: int
    tranformer_linear_size: int


@dataclass
class ModelConfig:
    f0_loss_weight: float
    phoneme_loss_weight: float
    vuv_loss_weight: float
    stop_loss_weight: float


@dataclass
class TrainConfig:
    batch_size: int
    log_iteration: int
    snapshot_iteration: int
    stop_iteration: int
    optimizer: Dict[str, Any]
    weight_initializer: Optional[str] = None
    num_processes: Optional[int] = None
    use_multithread: bool = False


@dataclass
class ProjectConfig:
    name: str
    tags: Dict[str, Any] = field(default_factory=dict)
    category: Optional[str] = None


@dataclass
class Config:
    dataset: DatasetConfig
    network: NetworkConfig
    model: ModelConfig
    train: TrainConfig
    project: ProjectConfig

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        backward_compatible(d)
        return dataclass_utility.convert_from_dict(cls, d)

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_utility.convert_to_dict(self)

    def add_git_info(self):
        self.project.tags["git-commit-id"] = get_commit_id()
        self.project.tags["git-branch-name"] = get_branch_name()


def backward_compatible(d: Dict[str, Any]):
    pass
