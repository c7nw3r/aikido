from torch.utils.data.dataset import T_co, Dataset

from aikido.__api__.kata import Kata


class DelegateKata(Kata):

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> T_co:
        return self.dataset[index]
