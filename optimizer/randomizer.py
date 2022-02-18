import random
from typing import List, Generator

from torch.utils.data import DataLoader


class Randomizer():
    """ Randomizer utility class """
    @staticmethod
    def batch_generator(data_loader: DataLoader) -> Generator:
        """ sampling without replacement """
        for idx, batch in enumerate(data_loader):
            yield idx, batch

    @staticmethod
    def data_generator(data_loaders: List[DataLoader], random_type: int, max_nbatch: int=None) -> Generator:
        '''
        :param data_loaders:    a list of DataLoaders
        :param random_type:     random type, 0: sequential, 1: sampling batches from random loaders
        :param max_nbatch:       optional, maximum number of batches
        :return:                a batch generator in format (loader_idx, batch_idx, batch)
        '''
        batch_cnt = 0
        data_generators = [(idx, Randomizer.batch_generator(loader)) for idx, loader in enumerate(data_loaders)]

        if random_type == 0:
            for loader_idx, loader in data_generators:
                for batch_idx, batch in loader:
                    yield loader_idx, batch_idx, batch
                    batch_cnt += 1
                    if max_nbatch is not None and batch_cnt >= max_nbatch: return
        elif random_type == 1:
            while len(data_generators) > 0:
                loader_idx, loader = random.choice(data_generators)
                try:
                    batch_idx, batch = next(loader)
                    yield loader_idx, batch_idx, batch
                    batch_cnt += 1
                    if max_nbatch is not None and batch_cnt >= max_nbatch: return
                except StopIteration:
                    data_generators.remove((loader_idx, loader))
        else:
            raise ValueError(f'random type {random_type} not supported!')

    @staticmethod
    def select_generator(data_loaders: List[DataLoader]) -> Generator:
        return Randomizer.batch_generator(random.choice(data_loaders))
