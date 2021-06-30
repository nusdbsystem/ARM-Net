import torch
from torch.utils.data import Dataset, DataLoader
import glob
from tqdm import tqdm

class LibsvmDataset(Dataset):
    """ Dataset loader for Libsvm data format """
    def __init__(self, fname, nfields):

        def decode_libsvm(line):
            columns = line.split(' ')
            map_func = lambda pair: (int(pair[0]), float(pair[1]))
            ids, vals = zip(*map(lambda col: map_func(col.split(':')), columns[1:]))
            sample = {'ids': torch.LongTensor(ids),
                      'vals': torch.FloatTensor(vals),
                      'y': float(columns[0])}
            return sample

        with open(fname) as f:
            sample_lines = sum(1 for line in f)

        self.feat_ids = torch.LongTensor(sample_lines, nfields)
        self.feat_vals = torch.FloatTensor(sample_lines, nfields)
        self.y = torch.FloatTensor(sample_lines)

        self.nsamples = 0
        with tqdm(total=sample_lines) as pbar:
            with open(fname) as fp:
                line = fp.readline()
                while line:
                    try:
                        sample = decode_libsvm(line)
                        self.feat_ids[self.nsamples] = sample['ids']
                        self.feat_vals[self.nsamples] = sample['vals']
                        self.y[self.nsamples] = sample['y']
                        self.nsamples += 1
                    except Exception:
                        print(f'incorrect data format line "{line}" !')
                    line = fp.readline()
                    pbar.update(1)
        print(f'# {self.nsamples} data samples loaded...')

    def __len__(self):
        return self.nsamples

    def __getitem__(self, idx):
        return {'ids': self.feat_ids[idx],
                'vals': self.feat_vals[idx],
                'y': self.y[idx]}

def libsvm_dataloader(args):
    data_dir = args.data_dir + args.dataset
    train_file = glob.glob("%s/tr*libsvm" % data_dir)[0]
    val_file = glob.glob("%s/va*libsvm" % data_dir)[0]
    test_file = glob.glob("%s/te*libsvm" % data_dir)[0]

    train_loader = DataLoader(LibsvmDataset(train_file, args.nfield),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(LibsvmDataset(val_file, args.nfield),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(LibsvmDataset(test_file, args.nfield),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader, test_loader