import argparse
import re
from pathlib import Path
from typing import Optional

import numpy
import yaml
from more_itertools import chunked
from pytorch_trainer.dataset.convert import concat_examples
from torch.utils.data.dataset import ConcatDataset
from tqdm import tqdm
from utility.save_arguments import save_arguments
from yukarin_sos.config import Config
from yukarin_sos.dataset import FeatureDataset, SpeakerFeatureDataset, create_dataset
from yukarin_sos.generator import Generator


def _extract_number(f):
    s = re.findall(r"\d+", str(f))
    return int(s[-1]) if s else -1


def _get_predictor_model_path(
    model_dir: Path,
    iteration: int = None,
    prefix: str = "predictor_",
):
    if iteration is None:
        paths = model_dir.glob(prefix + "*.pth")
        model_path = list(sorted(paths, key=_extract_number))[-1]
    else:
        model_path = model_dir / (prefix + "{}.pth".format(iteration))
        assert model_path.exists()
    return model_path


def generate(
    model_dir: Path,
    model_iteration: Optional[int],
    model_config: Optional[Path],
    time_second: float,
    num_test: int,
    output_dir: Path,
    use_gpu: bool,
):
    if model_config is None:
        model_config = model_dir / "config.yaml"

    output_dir.mkdir(exist_ok=True)
    save_arguments(output_dir / "arguments.yaml", generate, locals())

    config = Config.from_dict(yaml.safe_load(model_config.open()))

    model_path = _get_predictor_model_path(
        model_dir=model_dir,
        iteration=model_iteration,
    )
    generator = Generator(
        config=config,
        predictor=model_path,
        use_gpu=use_gpu,
    )

    sampling_rate = 200
    config.dataset.sampling_length = int(sampling_rate * time_second)

    batch_size = config.train.batch_size

    dataset = create_dataset(config.dataset)["test"]
    if isinstance(dataset, ConcatDataset):
        dataset = dataset.datasets[0]

    if isinstance(dataset.dataset, FeatureDataset):
        phoneme_paths = [inp.phoneme_path for inp in dataset.dataset.inputs[:num_test]]
    elif isinstance(dataset.dataset, SpeakerFeatureDataset):
        phoneme_paths = [
            inp.phoneme_path for inp in dataset.dataset.dataset.inputs[:num_test]
        ]
    else:
        raise ValueError(dataset)

    for data, phoneme_path in zip(
        chunked(tqdm(dataset, desc="generate"), batch_size),
        chunked(phoneme_paths, batch_size),
    ):
        data = concat_examples(data)
        f0s = generator.generate(
            phoneme=data["phoneme"],
            speaker_id=data["speaker_id"] if "speaker_id" in data else None,
        )

        for f0, p in zip(f0s, phoneme_path):
            numpy.save(output_dir.joinpath(p.stem + ".npy"), f0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, type=Path)
    parser.add_argument("--model_iteration", type=int)
    parser.add_argument("--model_config", type=Path)
    parser.add_argument("--time_second", type=float, default=3)
    parser.add_argument("--num_test", type=int, default=10)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--use_gpu", action="store_true")
    generate(**vars(parser.parse_args()))
