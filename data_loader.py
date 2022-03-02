from tqdm import tqdm
import pickle
import numpy as np
from typing import Tuple, List, Dict
from collections import Counter
import random

import torch
from torch import Tensor, LongTensor, FloatTensor
from torch.utils.data import Dataset, DataLoader
torch.multiprocessing.set_sharing_strategy('file_system')

__HDFS_vocab_sizes__: LongTensor = LongTensor([24, 60, 60, 27799, 2, 9, 48])
__BGL_vocab_sizes__: LongTensor = LongTensor([69251, 24, 60, 60, 100, 6, 14, 10, 648+359228+156004])
vocab_size_map: Dict[str, LongTensor] = {
    'hdfs': __HDFS_vocab_sizes__,
    'bgl': __BGL_vocab_sizes__
}


def get_vocab_size(dataset: str, tabular_cap: int) -> LongTensor:
    vocab_sizes = vocab_size_map[dataset]
    if dataset == 'hdfs':
        # pid cap
        vocab_sizes[3] = tabular_cap
    else:
        raise ValueError(f"incorrect dataset name {dataset}")
    return vocab_sizes


class LogDataset(Dataset):
    """ Dataset for Log Data (EventID in the last Field) """
    def __init__(self, data: list, nstep: int, vocab_sizes: LongTensor, session_based: bool):
        self.nstep, self.nsample = nstep, len(data)
        self.session_based, self.nevent = session_based, vocab_sizes[-1]

        # data sample format: [log_seq: [tabular; eventID]; log_seq_label]
        nfields = vocab_sizes.size(0)
        assert (nfields == len(data[0][0][0])), \
            f"vocab_sizes field {nfields} and data fields {len(data[0][0][0])} not match"
        # offsets for tabular embedding
        offsets = torch.zeros(nfields, dtype=torch.int64)                           # nfield
        offsets[1:] = torch.cumsum(vocab_sizes, dim=0)[:-1]
        self.event_offset = offsets[-1]

        print(f'===>>> processing {self.nsample} logs ...')
        # tabular & label
        self.tabular = []
        self.log_seq_y = FloatTensor(self.nsample)                                  # nsample
        with tqdm(total=self.nsample) as pbar:
            for sample_idx in range(self.nsample):
                # tabular
                log_seq = LongTensor(data[sample_idx][0])                           # n_idx*nfield
                ## cap at vocab_sizes
                for field_idx in range(nfields):
                    cap_mask = log_seq[:, field_idx] >= vocab_sizes[field_idx]-1    # n_idx
                    log_seq[cap_mask, field_idx] = vocab_sizes[field_idx]-1
                ## offset
                log_seq += offsets                                                  # n_idx*nfield
                self.tabular.append(log_seq)

                # label
                self.log_seq_y[sample_idx] = data[sample_idx][1]
                # update progress bar
                pbar.update(1)

    def __len__(self):
        return self.nsample

    def __getitem__(self, idx):
        return {
            'tabular': self.tabular[idx],               # n_idx*nfield
            'log_seq_y': self.log_seq_y[idx]            # 1
        }

    def generate_batch(self, batch: List[Dict[str, LongTensor]]) -> Dict[str, Tensor]:
        """
        :param batch:   a batch of samples, {tabular: [n_idx, nfield], LongTensor, log_seq_y: 1, LongTensor}
        :return:        session_based: {event_count, log_event_seq, log_seq_y}
        :return:        window_based: {tabular, eventID_y, nsamples, log_seq_y}
        """
        bsz = len(batch)
        # session-based, return only session features
        if self.session_based:
            log_seq_y, log_event_seq = [], []                                                   # bsz
            event_count = torch.zeros((bsz, self.nevent)).long()                                # bsz*nevent
            for log_idx in range(bsz):
                # event count
                event_seq = batch[log_idx]['tabular'][:, -1]-self.event_offset                  # n_idx
                counter = Counter(event_seq.tolist())
                for eventID in counter:
                    event_count[log_idx][eventID] = counter[eventID]
                # log eventID seq
                log_event_seq.append(event_seq)                                                 # n_idx
                # label
                log_seq_y.append(batch[log_idx]['log_seq_y'])                                   # 1

            return {
                'event_count': event_count,                                                     # bsz*nevent
                'log_event_seq': log_event_seq,                                                 # [n_1, ..., n_bsz]
                'log_seq_y': torch.stack(log_seq_y, dim=0).long()                               # bsz
            }
        # window-based, return both window and session features
        else:
            # nsamples: number of samples of each log_seq
            nsamples, log_seq_y = [], []                                                        # bsz
            tabular, eventID_y = [], []                                                         # N = n_1+\cdot+n_bsz
            for log_idx in range(bsz):
                log_nsample = batch[log_idx]['tabular'].size(0)-self.nstep                      # n_i = n_idx-nstep
                # drop if log_seq not long enough
                if log_nsample <= 0: continue
                # features for all windows
                nsamples.append(log_nsample)
                log_tabular = batch[log_idx]['tabular']
                for window_idx in range(log_nsample):
                    tabular.append(log_tabular[window_idx:window_idx+self.nstep])               # nstep*nfield
                    eventID_y.append(log_tabular[window_idx+self.nstep][-1]-self.event_offset)  # 1
                # session label
                log_seq_y.append(batch[log_idx]['log_seq_y'])                                   # 1

            return {
                'tabular': torch.stack(tabular, dim=0),                                         # N*nstep*nfield
                'eventID_y': torch.stack(eventID_y, dim=0),                                     # N
                'nsamples': torch.LongTensor(nsamples),                                         # bsz
                'log_seq_y': torch.stack(log_seq_y, dim=0).long()                               # bsz
            }


def _loader(data: np.ndarray, nstep: int, vocab_sizes: LongTensor,
            bsz: int, session_based: bool, nworker: int) -> DataLoader:
    dataset = LogDataset(data, nstep, vocab_sizes, session_based)
    data_loader = DataLoader(dataset, batch_size=bsz, shuffle=True, drop_last=True,
                             num_workers=nworker, collate_fn=dataset.generate_batch)
    return data_loader


def log_loader(data_path: str, nstep: int, vocab_sizes: LongTensor, bsz: int,
               shuffle: bool, valid_perc: float, test_perc: float, nenv: int,
               session_based = False, nworker: int = 4) -> Tuple[List[DataLoader], DataLoader, DataLoader]:
    """
    :param data_path:       path to the pickled dataset
    :param nstep:           number of time steps
    :param vocab_sizes:     vocabulary size
    :param bsz:             batch size
    :param valid_perc:      validation percentage
    :param nworker:         number of workers to load data
    :return:                train/valid/test data loader
    """
    with open(data_path, 'rb') as data_file:
        data = pickle.load(data_file)
        # TODO: use event embedding
        embedding_map = pickle.load(data_file)

    if shuffle: random.shuffle(data)

    train_samples = int(len(data) * (1-test_perc))
    valid_samples = int(train_samples * valid_perc)
    train_data, test_data = data[:train_samples], data[train_samples:]
    nsample_per_env = (train_samples-valid_samples) // nenv

    # split env temporally
    train_loaders = [_loader(train_data[env_idx*nsample_per_env:(env_idx+1)*nsample_per_env],
                             nstep, vocab_sizes, bsz, session_based, nworker) for env_idx in range(nenv)]
    valid_loader = _loader(train_data[-valid_samples:], nstep, vocab_sizes, bsz, session_based, nworker)
    test_loader = _loader(test_data, nstep, vocab_sizes, bsz, session_based, nworker)

    return train_loaders, valid_loader, test_loader
