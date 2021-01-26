import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import numpy
import pytest
import wandb
import yaml
from acoustic_feature_extractor.data.phoneme import JvsPhoneme
from acoustic_feature_extractor.data.sampling_data import SamplingData
from yaml import SafeLoader
from yukarin_sos.trainer import create_trainer

from tests.utility import get_data_directory

os.environ["WANDB_MODE"] = "dryrun"


@pytest.fixture(params=["train_config.yaml"])
def config_path(request):
    return get_data_directory().joinpath(request.param)


@pytest.fixture()
def config(config_path: Path):
    with config_path.open() as f:
        return yaml.load(f, SafeLoader)


@pytest.fixture()
def dataset_directory():
    return Path("/tmp/yukarin_sos_test_dataset")


@pytest.fixture()
def trained_directory():
    return Path("/tmp/yukarin_sos_test_trained")


def generate_dataset(
    dataset_directory: Path,
    data_num: int,
    f0_rate: int,
    phoneme_rate: int,
    phoneme_size: int,
    speaker_size: int,
):
    if dataset_directory.exists():
        for p in dataset_directory.rglob("*"):
            if not p.is_dir():
                p.unlink()
    else:
        dataset_directory.mkdir()

    f0_dir = dataset_directory.joinpath("f0")
    phoneme_dir = dataset_directory.joinpath("phoneme")
    phoneme_list_dir = dataset_directory.joinpath("phoneme_list")

    f0_dir.mkdir(exist_ok=True)
    phoneme_dir.mkdir(exist_ok=True)
    phoneme_list_dir.mkdir(exist_ok=True)

    speaker_dict = defaultdict(list)
    for i_data in range(data_num):
        speaker_num = i_data % speaker_size
        speaker_dict[str(speaker_num)].append(str(i_data))

        source_length = int(numpy.random.randint(low=10, high=20))
        phoneme_list = numpy.random.randint(
            low=0, high=phoneme_size, size=source_length, dtype=numpy.int32
        )
        phoneme_list_dir.joinpath(f"{i_data}.lab").write_text(
            "\n".join([f"0 0 {JvsPhoneme.phoneme_list[p]}" for p in phoneme_list])
        )

        f0 = phoneme_list.astype(numpy.float32) / 10 + 0.2 + speaker_num / 100
        f0 = numpy.repeat(f0, (phoneme_list + 1) * (f0_rate // phoneme_rate))
        f0[::5] = 0
        SamplingData(array=f0, rate=f0_rate).save(f0_dir.joinpath(f"{i_data}.npy"))

        phoneme = numpy.repeat(phoneme_list, phoneme_list + 1)
        phoneme = numpy.identity(phoneme_size, dtype=numpy.int32)[phoneme]
        SamplingData(array=phoneme, rate=phoneme_rate).save(
            phoneme_dir.joinpath(f"{i_data}.npy")
        )

    json.dump(speaker_dict, dataset_directory.joinpath("speaker_dict.json").open("w"))


def test_train(
    config: Dict[str, Any], dataset_directory: Path, trained_directory: Path
):
    generate_dataset(
        dataset_directory=dataset_directory,
        data_num=500,
        f0_rate=200,
        phoneme_rate=100,
        phoneme_size=config["network"]["phoneme_size"],
        speaker_size=config["dataset"]["speaker_size"],
    )

    config["dataset"]["f0_glob"] = str(dataset_directory.joinpath("f0/*.npy"))
    config["dataset"]["phoneme_glob"] = str(dataset_directory.joinpath("phoneme/*.npy"))
    config["dataset"]["phoneme_list_glob"] = str(
        dataset_directory.joinpath("phoneme_list/*.lab")
    )
    config["dataset"]["speaker_dict_path"] = dataset_directory.joinpath(
        "speaker_dict.json"
    )

    config["train"]["batch_size"] = 8
    config["train"]["log_iteration"] = 10
    config["train"]["snapshot_iteration"] = 100
    config["train"]["stop_iteration"] = 100

    trainer = create_trainer(
        config_dict=config,
        output=trained_directory,
    )
    trainer.run()

    wandb.finish()
