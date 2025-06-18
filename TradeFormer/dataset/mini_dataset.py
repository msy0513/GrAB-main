import torch
from torch_geometric.data import InMemoryDataset
from grit.dataset.transform import RRWPTransform, SimilarTransform
import argparse
from utils.utils import parse_config


class miniDataset(InMemoryDataset):
    def __init__(
        self,
        root="./dataset",
        use_edge_attr=True,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.root = root
        self.use_edge_attr = use_edge_attr
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def download(self):
        pass

    @property
    def raw_file_names(self):
        filelist = []
        for i in range(1, 50):
            file = "graph_data" + str(i) + ".pt"
            filelist.append(file)
        return filelist

    @property
    def processed_file_names(self):
        return ["mini_graphs.pt"]

    def process(self):
        data_list = []
        for i in range(1, 50):
            file = "graph_data" + str(i) + ".pt"
            loaded_data = torch.load("./subs/" + file)
            data_list += loaded_data
        print(data_list)
        print(len(data_list))
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])


def create_miniDataset(config):
    # pre_transform = RRWPTransform(**config.pos_enc_rrwp)
    if config:
        pre_transform = SimilarTransform(**config.pos_enc_rrwp)
        dataset = miniDataset(pre_transform=pre_transform)
    else:
        print("no_special_config")
        dataset = miniDataset()

    dataset = dataset.shuffle()
    data_len = len(dataset)

    train_dataset = dataset[: int(data_len * 0.7)]
    val_dataset = dataset[int(data_len * 0.7) : int(data_len * 0.9)]
    test_dataset = dataset[int(data_len * 0.9) :]

    import torch_geometric.transforms as T

    split_train = T.RandomNodeSplit(split="train_rest", num_val=0.0, num_test=0.0)
    split_val = T.RandomNodeSplit(num_val=1.0)
    split_test = T.RandomNodeSplit(num_test=1.0, num_val=0.0)

    tmp_train = []
    for data in train_dataset:
        data = split_train(data)
        data.split = "train"
        tmp_train.append(data)
    train_dataset = tmp_train

    tmp_test = []
    for data in test_dataset:
        data = split_test(data)
        data.split = "test"
        tmp_test.append(data)
    test_dataset = tmp_test

    tmp_val = []
    for data in val_dataset:
        data = split_val(data)
        data.split = "val"
        tmp_val.append(data)
    val_dataset = tmp_val

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../task_finetune.yaml")
    args = parser.parse_args()
    config = parse_config(args.config)
    create_miniDataset(config.data)
