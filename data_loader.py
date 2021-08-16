import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

import os
import numpy as np
import sklearn.model_selection
from tqdm import tqdm
from scipy.io.arff import loadarff

class UCILibsvmDataset(Dataset):
    """ Dataset loader for loading UCI dataset of Libsvm format """
    def __init__(self, X, y):
        assert X.shape[0] == y.shape[0]
        self.nsamples, self.nfeat = X.shape

        self.feat_ids = torch.LongTensor(self.nsamples, self.nfeat)
        self.feat_vals = torch.FloatTensor(self.nsamples, self.nfeat)
        self.y = torch.FloatTensor(self.nsamples)

        with tqdm(total=self.nsamples) as pbar:
            ids = torch.LongTensor(range(self.nfeat))
            for idx in range(self.nsamples):
                self.feat_ids[idx] = ids
                self.feat_vals[idx] = torch.FloatTensor(X[idx])
                self.y[idx] = y[idx]

                pbar.update(1)
        print(f'Data loader: {self.nsamples} data samples')

    def __len__(self):
        return self.nsamples

    def __getitem__(self, idx):
        return {'ids': self.feat_ids[idx],
                'vals': self.feat_vals[idx],
                'y': self.y[idx]}

def uci_loader(data_dir, batch_size, valid_perc=0., libsvm=False, workers=2, verbose=True):
    '''
    :param data_dir:        Path to load the uci dataset
    :param batch_size:      Batch size
    :param valid_perc:      valid percentage split from train (default 0, whole train set)
    :param libsvm:          Libsvm loader of format {'ids', 'vals', 'y'}
    :param workers:         the number of subprocesses to load data
    :return:                train/valid/test loader, train_loader.nclass/nclass
    '''

    def uci_validation_set(X, y, split_perc):
        return sklearn.model_selection.train_test_split(
            X, y, test_size=split_perc, random_state=0)

    def make_loader(X, y, transformer=None, batch_size=64, drop_last=False):
        if transformer is None:
            transformer = sklearn.preprocessing.StandardScaler()
            transformer.fit(X)
        X = transformer.transform(X)
        if libsvm:
            return DataLoader(UCILibsvmDataset(X, y),
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=workers, pin_memory=True,
                              drop_last=drop_last
                              ), transformer
        else:
            return DataLoader(
                dataset=TensorDataset(*[torch.from_numpy(e) for e in [X, y]]),
                batch_size=batch_size,
                shuffle=transformer is None,
                num_workers=workers,
                pin_memory=True,
                drop_last=drop_last
            ), transformer

    def uci_folder_to_name(f):
        return f.split('/')[-1]

    def line_to_idx(l):
        return np.array([int(e) for e in l.split()], dtype=np.int32)

    def load_uci_dataset(folder, train=True):
        full_file = f'{folder}/{uci_folder_to_name(folder)}.arff'
        if os.path.exists(full_file):
            data = loadarff(full_file)
            train_idx, test_idx = [line_to_idx(l) for l in open(f'{folder}/conxuntos.dat').readlines()]
            assert len(set(train_idx) & set(test_idx)) == 0
            all_idx = list(train_idx) + list(test_idx)
            assert len(all_idx) == np.max(all_idx) + 1
            assert np.min(all_idx) == 0
            if train:
                data = (data[0][train_idx], data[1])
            else:
                data = (data[0][test_idx], data[1])
        else:
            typename = 'train' if train else 'test'
            filename = f'{folder}/{uci_folder_to_name(folder)}_{typename}.arff'
            data = loadarff(filename)
        assert data[1].types() == ['numeric'] * (len(data[1].types()) - 1) + ['nominal']
        X = np.array(data[0][data[1].names()[:-1]].tolist())
        y = np.array([int(e) for e in data[0][data[1].names()[-1]]])
        nclass = len(data[1]['clase'][1])
        return X.astype(np.float32), y, nclass

    Xtrain, ytrain, nclass = load_uci_dataset(data_dir, train=True)
    if valid_perc > 0:
        Xtrain, Xvalid, ytrain, yvalid = uci_validation_set(Xtrain, ytrain, split_perc=valid_perc)
        train_loader, transformer = make_loader(Xtrain, ytrain, batch_size=batch_size, drop_last=False)
        valid_loader, _ = make_loader(Xvalid, yvalid, transformer, batch_size=batch_size, drop_last=False)
    else:
        train_loader, transformer = make_loader(Xtrain, ytrain, batch_size=batch_size, drop_last=False)
        valid_loader = train_loader

    Xtest, ytest, _ = load_uci_dataset(data_dir, train=False)
    test_loader, _ = make_loader(Xtest, ytest, transformer, batch_size=batch_size)
    if verbose:
        print(f'{uci_folder_to_name(data_dir)}: {len(ytrain)} training samples loaded.')
        print(f'{uci_folder_to_name(data_dir)}: {len(ytest)} testing samples loaded.')
    train_loader.nclass, train_loader.nfeat = nclass, Xtrain.shape[-1]
    return train_loader, valid_loader, test_loader