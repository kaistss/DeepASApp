
import os
import random
from os.path import *
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
import csv
from natsort import natsorted


def collate_fn(list_data_dict):
    data = {k: [] for k in list_data_dict[0].keys()}
    for ddict in list_data_dict:
        for k in data:
            data[k].append(ddict[k])
    for k in data.keys():
        if type(data[k][0]) is torch.Tensor:
            data[k] = torch.stack(data[k], 0)
    return data


class AuditorySceneAnalysisDataset(Dataset):
    """The Auditory Scene Analysis dataset"""
    def __init__(
        self, 
        asa_dir: str,
        dataset: str,
        sample_rate: int = 16000,
        num_speakers: int = 5,
        audio_time_len: Optional[float] = None,
        n_classes: int = 13
    ) -> None:
        """The Auditory Scene Analysis dataset

        Args:
            asa_dir: directory of the dataset
            sample_rate: sample rate of the audio
            num_speakers: number of speakers
            audio_time_len: cut the audio to `audio_time_len` seconds if given audio_time_len
            n_classes: number of classes
        """
        super().__init__()

        self.asa_dir = str(Path(asa_dir).expanduser())
        self.wav_dir = Path(self.asa_dir) / dataset
        self.files = os.listdir(Path(self.wav_dir) / 'mixed')
        self.wav_files = [file for file in self.files if file.endswith(".wav")]
        assert len(self.wav_files) > 0, f"dir is empty or not exists: {self.asa_dir}"

        self.dataset = dataset
        self.sr = sample_rate
        self.audio_time_len = audio_time_len
        self.n_classes = n_classes

        self.collate_fn = collate_fn

    def __getitem__(self, index_seed: Union[int, Tuple[int, int]]):
        if type(index_seed) == int:
            index = index_seed
            if self.dataset == 'train':
                seed = random.randint(a=0, b=99999999)
            else:
                seed = index
        else:
            index, seed = index_seed
        g = torch.Generator()
        g.manual_seed(seed)
        front = self.wav_files[index].split('.')[0]
        # load audio signals
        mix, sr = sf.read(self.wav_dir / f'mixed/{front}.wav')
        reverb_dir = os.listdir(self.wav_dir / f'reverb/{front}')
        wav_files = [file for file in reverb_dir if file.endswith(".wav")]
        wav_files = natsorted(wav_files)
        # load files
        reverb_wavs = []
        anechoic_wavs = []
        for i in range(min(len(wav_files), 5)):  # load maximum 5 files
            reverb_wavs.append(sf.read(self.wav_dir / f'reverb/{front}' / wav_files[i])[0].T)
            anechoic_wavs.append(sf.read(self.wav_dir / f'anechoic/{front}' / wav_files[i])[0].T)
        n_src = len(reverb_wavs)
        # zero padding
        zeros_wav = np.zeros_like(reverb_wavs[0]) if reverb_wavs else None
        while len(reverb_wavs) < 5:
            reverb_wavs.append(zeros_wav)
            anechoic_wavs.append(zeros_wav)
        reverb = np.stack(reverb_wavs, axis=0)
        anechoic = np.stack(anechoic_wavs, axis=0)
        mix = mix.T
        noise = mix - (anechoic + reverb).sum(axis=0)

        # info
        info_file = f'info/{front}.csv'
        _output_dict = {}
        _fid = open(os.path.join(self.wav_dir, info_file), 'r')
        with open(os.path.join(self.wav_dir, info_file), 'r') as file:
            csv_reader = csv.reader(file)
            headers = next(csv_reader)
            for row in csv_reader:
                _frame_ind = int(float(row[0]))
                if _frame_ind not in _output_dict: _output_dict[_frame_ind] = []
                if len(row) == 5 or len(row) == 6:
                    _output_dict[_frame_ind].append([int(float(row[1])), int(float(row[2])), float(row[3]), float(row[4])]) # src_idx, class, azi, ele
        _desc_file = self.convert_output_format_polar_to_cartesian(_output_dict)

        # Audio tagging
        self.label_len = int(self.audio_time_len * 10)
        cls_label = np.zeros(len(wav_files))  # (S, C)
        for frame_ind, active_event_list in _desc_file.items():
            if frame_ind < self.label_len:
                for active_event in active_event_list:
                    cls_label[active_event[0]] = active_event[1]
        # Zero-padding for cls_label (Ensure length is 5)
        zero_cls = -1 * np.ones(1)
        cls_label = np.concatenate([cls_label] + [zero_cls] * (5 - len(wav_files)), axis=0)
        # Sound event detection
        sed_label = np.zeros((len(wav_files), self.label_len, self.n_classes))  # (S, T, C)
        x_label = np.zeros((len(wav_files), self.label_len))  # (S, T)
        y_label = np.zeros((len(wav_files), self.label_len))  # (S, T)
        z_label = np.zeros((len(wav_files), self.label_len))  # (S, T)
        for frame_ind, active_event_list in _desc_file.items():
            if frame_ind < self.label_len:
                for active_event in active_event_list:
                    sed_label[active_event[0], frame_ind, active_event[1]] = 1
                    x_label[active_event[0], frame_ind] = active_event[2]
                    y_label[active_event[0], frame_ind] = active_event[3]
                    z_label[active_event[0], frame_ind] = active_event[4]
        # Zero-padding for sed_label (Ensure length is 5)
        zeros_sed = np.zeros((1, self.label_len, self.n_classes))
        sed_label = np.concatenate([sed_label] + [zeros_sed] * (5 - len(wav_files)), axis=0)
        # Sound event direction-of-arrival (DOA) labels
        doa_label = np.stack([x_label, y_label, z_label], axis=2)  # (S, T, XYZ)
        zeros_doa = np.zeros((1, self.label_len, 3))
        doa_label = np.concatenate([doa_label] + [zeros_doa] * (5 - len(wav_files)), axis=0)
        # Add silence class (-1 -> 14)
        # cls_label[cls_label == -1] = 13

        paras = {
            'index': front,
            'seed': seed,
            'n_src': n_src,
            'wavname': self.wav_files[index],
            'wavdir': str(self.wav_dir),
            'sample_rate': self.sr,
            'dataset': self.dataset,
            'audio_time_len': self.audio_time_len,
        }

        item = {
            'mixture': torch.as_tensor(mix, dtype=torch.float32),
            'anechoic': torch.as_tensor(anechoic, dtype=torch.float32),
            'reverb': torch.as_tensor(reverb, dtype=torch.float32),
            'noise': torch.as_tensor(noise, dtype=torch.float32),
            'len': torch.as_tensor(len(wav_files), dtype=torch.int64),
            'cls_label': torch.as_tensor(cls_label, dtype=torch.long),
            'sed_label': torch.as_tensor(sed_label, dtype=torch.float32),
            'doa_label': torch.as_tensor(doa_label, dtype=torch.float32),
            'paras': paras,
        }
        return item

    def __len__(self):
        return len(self.wav_files)

    def convert_output_format_polar_to_cartesian(self, in_dict):
        out_dict = {}
        for frame_cnt in in_dict.keys():
            if frame_cnt not in out_dict:
                out_dict[frame_cnt] = []
                for tmp_val in in_dict[frame_cnt]:
                    ele_rad = (tmp_val[3]-90)*np.pi/180.
                    # ele_rad = tmp_val[3]*np.pi/180.
                    azi_rad = tmp_val[2]*np.pi/180
                    tmp_label = np.cos(ele_rad)
                    x = np.cos(azi_rad) * tmp_label
                    y = np.sin(azi_rad) * tmp_label
                    z = np.sin(ele_rad)
                    out_dict[frame_cnt].append([tmp_val[0], tmp_val[1], x, y, z])
        return out_dict
