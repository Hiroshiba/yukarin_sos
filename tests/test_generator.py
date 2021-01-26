from pathlib import Path

import numpy
import pytest
import yaml
from yaml.loader import SafeLoader
from yukarin_sos.config import Config
from yukarin_sos.generator import Generator
from yukarin_sos.network.predictor import create_predictor

from tests.utility import get_data_directory


@pytest.fixture()
def train_config_path():
    return get_data_directory() / "train_config.yaml"


@pytest.fixture()
def trained_directory():
    return Path("/tmp/yukarin_sos_test_trained")


@pytest.fixture()
def generated_directory():
    return Path("/tmp/yukarin_sos_test_generated")


def test_generator(
    train_config_path: Path, trained_directory: Path, generated_directory: Path
):
    with train_config_path.open() as f:
        d = yaml.load(f, SafeLoader)

    config = Config.from_dict(d)

    if trained_directory.joinpath("predictor_1000.pth").exists():
        predictor = trained_directory.joinpath("predictor_1000.pth")
    else:
        predictor = create_predictor(config=config.network)

    generator = Generator(
        config=config,
        predictor=predictor,
    )

    source_length = 15
    phoneme_list = numpy.random.randint(
        low=0, high=config.network.phoneme_size, size=source_length, dtype=numpy.int64
    )

    speaker_id = numpy.array(0)
    d = generator.generate(
        phoneme_list=phoneme_list,
        speaker_id=speaker_id,
    )

    generated_directory.mkdir(exist_ok=True)
    numpy.save(generated_directory.joinpath("f0.npy"), d["f0"])
    numpy.save(generated_directory.joinpath("phoneme.npy"), d["phoneme"])
