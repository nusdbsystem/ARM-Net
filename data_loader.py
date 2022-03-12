from tqdm import tqdm
import pickle
import numpy as np
from typing import Tuple, List, Dict, Union
from collections import Counter, defaultdict
import random

import torch
from torch import Tensor, LongTensor, FloatTensor
from torch.utils.data import Dataset, DataLoader
torch.multiprocessing.set_sharing_strategy('file_system')

# dataset META info
__HDFS_vocab_sizes__: LongTensor = LongTensor([24, 60, 60, 27799, 2, 9, 48])
__BGL_vocab_sizes__: LongTensor = LongTensor([69251, 24, 60, 60, 100, 6, 14, 10, 648+359228+156004])
vocab_size_map: Dict[str, LongTensor] = {
    'hdfs': __HDFS_vocab_sizes__,
    'bgl': __BGL_vocab_sizes__
}
# format: {eventID; [int]*300}; event embeddings from event NLP preprocessing
event_emb_map = {}
__event_emb_size__ = 300


def get_vocab_size(dataset: str, tabular_cap: int) -> LongTensor:
    vocab_sizes = vocab_size_map[dataset]
    if dataset == 'hdfs':
        # pid cap for hdfs dataset
        vocab_sizes[3] = tabular_cap
    else:
        raise NotImplementedError
    return vocab_sizes


class LogDataset(Dataset):
    """ Dataset for Log Data (EventID in the last Field) """
    def __init__(self, data: Union[list, np.ndarray], nstep: int, vocab_sizes: LongTensor,
                 session_based: bool, feature_code: int):
        self.nfield, self.nstep, self.nsample = vocab_sizes.size(0), nstep, len(data)
        self.session_based, self.nevent = session_based, vocab_sizes[-1]
        # data sample format: [log_seq: [[int]*nfield]*nlog, log_seq_label: int]
        assert (self.nfield == len(data[0][0][0])), \
            f"vocab_sizes field {self.nfield} and data fields {len(data[0][0][0])} not match"
        # feature_code:    [sequential, quantitative, semantic, tabular] <-> [0/1][0/1][0/1][0/1]
        assert 1 <= feature_code <= 15, f'legal feature code 1~15 (0001~1111)'
        self.use_sequential, self.use_quantitative, self.use_semantic, self.use_tabular = \
            feature_code >> 3 & 1, feature_code >> 2 & 1, feature_code >> 1 & 1, feature_code & 1
        # offsets for tabular embedding
        offsets = torch.zeros(self.nfield, dtype=torch.int64)                           # nfield
        offsets[1:] = torch.cumsum(vocab_sizes, dim=0)[:-1]
        self.event_offset = offsets[-1]

        print(f'===>>> processing {self.nsample} logs ...')
        # tabular & label
        self.tabular = []
        self.y = FloatTensor(self.nsample)                                              # nsample
        with tqdm(total=self.nsample) as pbar:
            for sample_idx in range(self.nsample):
                # tabular
                log_seq = LongTensor(data[sample_idx][0])                               # nlog*nfield
                ## cap at vocab_sizes
                for field_idx in range(self.nfield):
                    cap_mask = log_seq[:, field_idx] >= vocab_sizes[field_idx]-1        # nlog
                    log_seq[cap_mask, field_idx] = vocab_sizes[field_idx]-1
                ## offset
                log_seq += offsets                                                      # nlog*nfield
                self.tabular.append(log_seq)
                # label
                self.y[sample_idx] = data[sample_idx][1]
                # update progress bar
                pbar.update(1)

    def __len__(self):
        return self.nsample

    def __getitem__(self, idx):
        return {
            'tabular': self.tabular[idx],                                               # nlog*nfield
            'y': self.y[idx]                                                            # 1
        }

    def log_seq_to_features(self, log_seq: LongTensor) -> {str: Tensor}:
        """
        :param event_seq:       log sequence, [nlog, nfield], LongTensor
        :return:                sequential, quantitative, semantic, tabular features
        """
        sequential, quantitative, semantic, tabular = [], [], [], []
        event_seq = log_seq[:, -1]-self.event_offset                                    # nlog
        if self.use_sequential:
            sequential = event_seq                                                      # nlog
        if self.use_quantitative:
            quantitative = torch.zeros((self.nevent,), dtype=torch.int64)               # nevent
            counter = Counter(event_seq.tolist())
            for eventID in counter:
                quantitative[eventID] = counter[eventID]
        if self.use_semantic:
            semantic = torch.zeros((event_seq.size(0), __event_emb_size__),
                                   dtype=torch.float32)                                 # nlog*nemb
            for event_idx, eventID in enumerate(event_seq):
                semantic[event_idx] = torch.FloatTensor(event_emb_map[eventID.item()])
        if self.use_tabular:
            tabular = log_seq                                                           # nlog*nfield
        return sequential, quantitative, semantic, tabular

    def append_featuers(self, features: defaultdict, sequential: LongTensor = None, quantitative: LongTensor = None,
                        semantic: FloatTensor = None, tabular: LongTensor = None) -> None:
        if self.use_sequential:
            features['sequential'].append(sequential)
        if self.use_quantitative:
            features['quantitative'].append(quantitative)
        if self.use_semantic:
            features['semantic'].append(semantic)
        if self.use_tabular:
            features['tabular'].append(tabular)

    def batchfy_features(self, features: defaultdict) -> Dict[str, Tensor]:
        """
        :param features: only need to pad [sequential, semantic, tabular] for session-based features
        :return:        padded (-1) and stacked features
        """
        sequential, quantitative, semantic, tabular = \
            features['sequential'], features['quantitative'], features['semantic'], features['tabular']
        batchfied_features = {}
        # padding (-1) for [sequential, semantic, tabular]
        if self.session_based:
            if self.use_sequential:
                bsz, seq_len = len(sequential), [event_seq.size(0) for event_seq in sequential]
                tmp = torch.full((bsz, max(seq_len)), fill_value=-1, dtype=torch.int64) # bsz*max_len
                for seq_idx in range(bsz):
                    tmp[seq_idx][:len(sequential[seq_idx])] = sequential[seq_idx]
                batchfied_features['sequential'] = tmp
            if self.use_semantic:
                bsz, seq_len = len(semantic), [emb_seq.size(0) for emb_seq in semantic]
                tmp = torch.full((bsz, max(seq_len), __event_emb_size__),
                                 fill_value=-1., dtype=torch.float32)                   # bsz*max_len*nemb
                for seq_idx in range(bsz):
                    tmp[seq_idx][:semantic[seq_idx].size(0)] = semantic[seq_idx]
                batchfied_features['semantic'] = tmp
            if self.use_tabular:
                bsz, seq_len = len(tabular), [tab_seq.size(0) for tab_seq in tabular]
                tmp = torch.full((bsz, max(seq_len), self.nfield),
                                 fill_value=-1, dtype=torch.int64)                      # bsz*max_len*nfield
                for seq_idx in range(bsz):
                    tmp[seq_idx][:tabular[seq_idx].size(0)] = tabular[seq_idx]
                batchfied_features['tabular'] = tmp
            if self.use_sequential or self.use_semantic or self.use_tabular:
                batchfied_features['seq_len'] = torch.LongTensor(seq_len)               # bsz
        else:
            if self.use_sequential:
                batchfied_features['sequential'] = torch.stack(sequential, dim=0)       # nwindow*nstep
            if self.use_semantic:
                batchfied_features['semantic'] = torch.stack(semantic, dim=0)           # nwindow*nstep*nemb
            if self.use_tabular:
                batchfied_features['tabular'] = torch.stack(tabular, dim=0)             # nwindoe*nstep*nfield
        if self.use_quantitative:
            batchfied_features['quantitative'] = torch.stack(quantitative, dim=0)       # bsz/nwindow*nfield
        return batchfied_features

    def generate_batch(self, batch: List[Dict[str, LongTensor]]) -> Dict[str, Union[Tensor, Dict]]:
        """
        :param batch:   a batch of samples, {tabular: [nlog, nfield], LongTensor, y: 1, LongTensor}
        :return:        # session-based
                            features:   {sequential: bsz*max_len (-1 pad), quantitative: bsz*nfield,
                                        semantic: bsz*max_len*nemb (-1 pad), tabular: bsz*max_len*nfield (-1 pad),
                                        seq_len: bsz}
                            pred_label: bsz
                        # window-based
                            features:   {sequential: nwindow*nstep, quantitative: nwindow*nfield,
                                        semantic: nwindow*nstep*nemb, tabular: nwindow*nstep*nfield}
                            pred_label: nwindow
                            nsamples:   bsz, (used for restoring the session label)
                            label:      bsz
        """
        bsz = len(batch)
        features, pred_label = defaultdict(list), []                                    # bsz/nwindow
        nsamples, label = [], []                                                        # bsz, bsz
        for seq_idx in range(bsz):
            seq_tabular = batch[seq_idx]['tabular']                                     # nlog*nfield
            if self.session_based:
                self.append_featuers(features, *self.log_seq_to_features(seq_tabular))
                pred_label.append(batch[seq_idx]['y'])
            else:
                seq_nsample = seq_tabular.size(0)-self.nstep
                if seq_nsample <= 0: continue
                for window_idx in range(seq_nsample):
                    self.append_featuers(features, *self.log_seq_to_features(
                        seq_tabular[window_idx:window_idx+self.nstep]))
                    pred_label.append(seq_tabular[window_idx+self.nstep][-1]-self.event_offset)
                nsamples.append(seq_nsample)
                label.append(batch[seq_idx]['y'])
        batch = {
            'features': self.batchfy_features(features),
            'pred_label': torch.stack(pred_label, dim=0),                               # bsz/nwindow
        }
        if not self.session_based:
            batch['nsamples'] = torch.LongTensor(nsamples)                              # bsz
            batch['label'] = torch.stack(label, dim=0).long()                           # bsz
        return batch

def _loader(data: np.ndarray, nstep: int, vocab_sizes: LongTensor,
            session_based: bool, feature_code: int, bsz: int, nworker: int) -> DataLoader:
    dataset = LogDataset(data, nstep, vocab_sizes, session_based, feature_code)
    data_loader = DataLoader(dataset, batch_size=bsz, shuffle=True, drop_last=True,
                             num_workers=nworker, collate_fn=dataset.generate_batch)
    return data_loader


def log_loader(data_path: str, nstep: int, vocab_sizes: LongTensor, session_based: bool, feature_code: int,
               shuffle: bool, only_normal: bool, valid_perc: float, test_perc: float, nenv: int,
               bsz: int, nworker: int = 4) -> Tuple[List[DataLoader], DataLoader, DataLoader]:
    """
    :param data_path:       path to the pickled dataset
    :param nstep:           window size (number of window time step)
    :param vocab_sizes:     vocabulary size
    :param session_based:   whether session-based or window-based
    :param feature_code:    see LogDataset.__init__
    :param shuffle:         whether to shuffle the whole dataset
    :param only_normal:     whether to use only normal log sequences for training
    :param valid_perc:      valid set percentage (over test set)
    :param test_perc:       test set percentage (over whole dataset)
    :param nenv:            number of training environments
    :param bsz:             batch size
    :param nworker:         number of data_loader workers
    :return:                train*nenv/valid/test data loader
    """
    with open(data_path, 'rb') as data_file:
        data = pickle.load(data_file)
        global event_emb_map
        event_emb_map = pickle.load(data_file)

    # whether to shuffle data to make it i.i.d. (data leak to train set)
    if shuffle: random.shuffle(data)

    ntrain_sample, nvalid_sample = int(len(data)*(1-test_perc)), int(len(data)*test_perc*valid_perc)
    train_data, test_data = data[:ntrain_sample], data[ntrain_sample:]
    # use only normal log seq for training
    if only_normal:
        train_data = list(filter(lambda log_seq: log_seq[1] == 0, train_data))
        ntrain_sample = len(train_data)
    nsample_per_env = ntrain_sample//nenv

    # split env sequentially (across time)
    train_loaders = [_loader(train_data[env_idx*nsample_per_env:(env_idx+1)*nsample_per_env],
                             nstep, vocab_sizes, session_based, feature_code, bsz, nworker) for env_idx in range(nenv)]
    valid_loader = _loader(test_data[:nvalid_sample], nstep, vocab_sizes, session_based, feature_code, bsz, nworker)
    test_loader = _loader(test_data[nvalid_sample:], nstep, vocab_sizes, session_based, feature_code, bsz, nworker)

    return train_loaders, valid_loader, test_loader
