from tqdm import tqdm
import pickle
import numpy as np
from typing import Tuple, List, Dict, Iterable, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


__HDFS_vocab_sizes__: List[int] = [24, 60, 60, 27799, 2, 9, 79+3445254+27522]
__BGL_vocab_sizes__: List[int] = [69251, 24, 60, 60, 100, 6, 14, 10, 648+359228+156004]
vocab_size_map: Dict[str, List[int]] = {
    'hdfs': __HDFS_vocab_sizes__,
    'bgl': __BGL_vocab_sizes__
}


def get_vocab_size(dataset: str, tabular_cap: int, text_cap: int) -> List[int] :
    vocab_sizes = vocab_size_map[dataset]
    if dataset == 'hdfs':
        vocab_sizes[3] = tabular_cap
    elif dataset =='bgl':
        vocab_sizes[0] = tabular_cap
    else:
        raise ValueError(f"incorrect dataset name {dataset}")
    vocab_sizes[-1] = text_cap
    return vocab_sizes


class LogDataset(Dataset):
    """ Dataset for Log Data """
    def __init__(self, raw_data: np.ndarray, nstep: int,
                 vocab_sizes: List[int], max_seq_len: int):
        self.nstep = nstep
        self.nsamples = len(raw_data) - nstep
        self.max_seq_len = max_seq_len

        # sample format: [tabular; text, label]
        nfields = len(raw_data[0])
        assert (len(vocab_sizes) == nfields-1), \
            f"vocab_sizes {len(vocab_sizes)} and data fields {nfields} not match"
        # offsets for tabular embedding
        offsets = np.zeros(nfields-2, dtype=np.int64)
        offsets[1:] = np.cumsum(vocab_sizes)[:-2]

        # tabular: fields 0~nfield-3
        self.tabular = raw_data[:, :-2].astype(np.int64)
        for col in range(nfields-2):
            # cap at vocab_size
            cap_mask = self.tabular[:, col] >= vocab_sizes[col]-1
            self.tabular[cap_mask, col] = vocab_sizes[col]-1

            # add offset, sharing one embedding layer for all tabular features
            self.tabular[:, col] += offsets[col]
        self.tabular = torch.tensor(self.tabular, dtype=torch.long)

        # text: field nfield-2
        raw_text = raw_data[:, -2]
        self.text = []
        # two special tokens (bos/pad) following regular token idx
        self.bos_idx: int = vocab_sizes[-1]
        self.pad_idx: int = self.bos_idx + 1
        print(f'===>>> processing {self.nsamples} logs ...')
        with tqdm(total=self.nsamples) as pbar:
            for idx in range(len(raw_text)):
                sentence = torch.zeros([len(raw_text[idx])+1], dtype=torch.long)
                sentence[1:] = torch.tensor(raw_text[idx])
                # cap at vocab_size
                cap_mask = sentence >= vocab_sizes[-1]-1
                sentence[cap_mask] = vocab_sizes[-1]-1
                # prepend bos token
                sentence[0] = self.bos_idx

                self.text.append(sentence)
                # update progress bar
                pbar.update(1)

        # label: field -1
        self.y = torch.tensor(raw_data[:, -1].astype(np.int64), dtype=torch.float)

        # vocab_size of (all tabular features, text)
        self.vocab_sizes = (sum(vocab_sizes[:-1]), self.pad_idx + 1)

    def __len__(self):
        return self.nsamples

    def __getitem__(self, idx):
        return {
            'tabular': self.tabular[idx: idx+self.nstep],       # nstep*nfield-2
            'text': self.text[idx: idx+self.nstep],             # [len_1, len_2, ..., len_nstep]
            'y': self.y[idx+self.nstep]                         # 1
        }

    def generate_batch(self, batch: List[Dict[str, Union[Tensor, Iterable]]]) -> Dict[str, Tensor]:
        """
        :param batch:   a batch of samples
        :return:
        """
        bsz = len(batch)
        tabular = torch.stack([sample['tabular'] for sample in batch], dim=0)           # bsz*nstep*nfield-2

        # text
        max_seq_lens: List[int] = [max(map(len, sample['text'])) for sample in batch]
        max_seq_len: int = min(self.max_seq_len, max(max_seq_lens))
        text = torch.full((bsz, self.nstep, max_seq_len),
                          fill_value=self.pad_idx, dtype=torch.long)                    # bsz*nstep*max_seq_len
        for idx in range(bsz):
            for step in range(self.nstep):
                seq = batch[idx]['text'][step]
                seq_len = min(max_seq_len, len(seq))
                text[idx, step, :seq_len] = seq[:seq_len]

        y = torch.stack([sample['y'] for sample in batch], dim=0)                       # bsz
        return {
            'tabular': tabular,
            'text': text,
            'y': y
        }


def _loader(data: np.ndarray, nstep: int, vocab_sizes: List[int],
            max_seq_len: int, bsz: int, workers: int) -> [DataLoader, int]:
    dataset = LogDataset(data, nstep, vocab_sizes, max_seq_len)
    data_loader = DataLoader(dataset, batch_size=bsz, shuffle=True, drop_last=True,
                             num_workers=workers, collate_fn=dataset.generate_batch)
    return data_loader, dataset.vocab_sizes


def log_loader(data_path: str, nstep: int,
               vocab_sizes: List[int], max_seq_len: int, bsz: int,
               valid_perc: float = 0.1,
               test_perc: float = 0.1,
               workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader, Tuple[int, int]]:
    """
    :param data_path:       path to the pickled dataset
    :param nstep:           number of time steps
    :param vocab_sizes:     vocabulary size
    :param max_seq_len:     max_seq_len for text padding
    :param bsz:             batch size
    :param valid_perc:      validation percentage
    :param test_perc:       test percentage
    :param workers:         number of workers to load data
    :return:                train/valid/test data loader, vocabulary sizes tuple (tabular, text)
    """
    data = pickle.load(open(data_path, 'rb'))
    nsamples = len(data)-nstep
    n_train, n_valid, n_test = int(nsamples*(1-valid_perc-test_perc)), \
                              int(nsamples*valid_perc), int(nsamples*test_perc)

    train_loader, ret_vocab_sizes = _loader(data[:n_train+nstep], nstep, vocab_sizes, max_seq_len, bsz, workers)
    valid_loader, _ = _loader(data[n_train:n_train+n_valid+nstep], nstep, vocab_sizes, max_seq_len, bsz, workers)
    test_loader, _ = _loader(data[-n_test-nstep:], nstep, vocab_sizes, max_seq_len, bsz, workers)

    return train_loader, valid_loader, test_loader, ret_vocab_sizes
