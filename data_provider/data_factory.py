from data_provider.data_loader import Dataset_Loader
from torch.utils.data import DataLoader

def data_provider(args, flag):
    assert flag in ["train", "val", "test"]
    # The simplified training entrypoint always uses encoded time features.
    timeenc = 1
    if flag == "test":
        shuffle_flag = False
        drop_last = True
    else:
        shuffle_flag = True
        drop_last = True
    data_set = Dataset_Loader(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        timeenc=timeenc,
        freq=args.freq,
        percent=args.percent,
    )
    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
    )
    return data_set, data_loader
