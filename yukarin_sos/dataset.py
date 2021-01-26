import json
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy
from acoustic_feature_extractor.data.phoneme import JvsPhoneme
from acoustic_feature_extractor.data.sampling_data import SamplingData
from torch.utils.data._utils.collate import default_convert
from torch.utils.data.dataset import ConcatDataset, Dataset

from yukarin_sos.config import DatasetConfig


def resample(rate: float, data: SamplingData):
    length = int(len(data.array) / data.rate * rate)
    indexes = (numpy.random.rand() + numpy.arange(length)) * (data.rate / rate)
    return data.array[indexes.astype(int)]


@dataclass
class Input:
    f0: SamplingData
    phoneme: SamplingData
    phoneme_list: List[JvsPhoneme]


@dataclass
class LazyInput:
    f0_path: SamplingData
    phoneme_path: SamplingData
    phoneme_list_path: SamplingData

    def generate(self):
        return Input(
            f0=SamplingData.load(self.f0_path),
            phoneme=SamplingData.load(self.phoneme_path),
            phoneme_list=JvsPhoneme.load_julius_list(self.phoneme_list_path),
        )


class FeatureDataset(Dataset):
    def __init__(
        self,
        inputs: List[Union[Input, LazyInput]],
    ):
        self.inputs = inputs

    @staticmethod
    def extract_input(
        f0_data: SamplingData,
        phoneme_data: SamplingData,
        phoneme_list_data: List[JvsPhoneme],
    ):
        rate = f0_data.rate

        f0 = f0_data.array
        phoneme = resample(rate=rate, data=phoneme_data)

        length = min(len(f0), len(phoneme))
        assert numpy.abs(length - len(f0)) < 10
        assert numpy.abs(length - len(phoneme)) < 10

        f0 = f0[:length]
        phoneme = phoneme[:length]

        phoneme_list = numpy.array([p.phoneme_id for p in phoneme_list_data])

        return dict(
            f0=f0,
            phoneme=numpy.argmax(phoneme, axis=1).astype(numpy.int64),
            phoneme_list=phoneme_list.astype(numpy.int64),
        )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        input = self.inputs[i]
        if isinstance(input, LazyInput):
            input = input.generate()

        return self.extract_input(
            f0_data=input.f0,
            phoneme_data=input.phoneme,
            phoneme_list_data=input.phoneme_list,
        )


class SpeakerFeatureDataset(Dataset):
    def __init__(self, dataset: FeatureDataset, speaker_ids: List[int]):
        assert len(dataset) == len(speaker_ids)
        self.dataset = dataset
        self.speaker_ids = speaker_ids

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        d = self.dataset[i]
        d["speaker_id"] = numpy.array(self.speaker_ids[i], dtype=numpy.int64)
        return d


class TensorWrapperDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return default_convert(self.dataset[i])


def create_dataset(config: DatasetConfig):
    f0_paths = {Path(p).stem: Path(p) for p in glob(config.f0_glob)}
    fn_list = sorted(f0_paths.keys())
    assert len(fn_list) > 0

    phoneme_paths = {Path(p).stem: Path(p) for p in glob(config.phoneme_glob)}
    assert set(fn_list) == set(phoneme_paths.keys())

    phoneme_list_paths = {Path(p).stem: Path(p) for p in glob(config.phoneme_list_glob)}
    assert set(fn_list) == set(phoneme_list_paths.keys())

    speaker_ids: Optional[Dict[str, int]] = None
    if config.speaker_dict_path is not None:
        fn_each_speaker: Dict[str, List[str]] = json.loads(
            config.speaker_dict_path.read_text()
        )
        assert config.speaker_size == len(fn_each_speaker)

        speaker_ids = {
            fn: speaker_id
            for speaker_id, (_, fns) in enumerate(fn_each_speaker.items())
            for fn in fns
        }
        assert set(fn_list).issubset(set(speaker_ids.keys()))

    numpy.random.RandomState(config.seed).shuffle(fn_list)

    test_num = config.test_num
    trains = fn_list[test_num:]
    tests = fn_list[:test_num]

    def _dataset(fns, for_test=False):
        inputs = [
            LazyInput(
                f0_path=f0_paths[fn],
                phoneme_path=phoneme_paths[fn],
                phoneme_list_path=phoneme_list_paths[fn],
            )
            for fn in fns
        ]

        dataset = FeatureDataset(inputs=inputs)

        if speaker_ids is not None:
            dataset = SpeakerFeatureDataset(
                dataset=dataset,
                speaker_ids=[speaker_ids[fn] for fn in fns],
            )

        dataset = TensorWrapperDataset(dataset)

        if for_test:
            dataset = ConcatDataset([dataset] * config.test_trial_num)

        return dataset

    return {
        "train": _dataset(trains),
        "test": _dataset(tests, for_test=True),
    }
