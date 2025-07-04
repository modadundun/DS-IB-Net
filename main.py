from __future__ import print_function
import os
import torch

from merged_dataset import AnomalyDataset

from torch.utils.data import DataLoader

from train_icic import train

import options
import torch.optim as optim
import datetime
from DS_IB import DS_IB



if __name__ == '__main__':
    torch.backends.cudnn.enabled = False
    args = options.parser.parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda")
    torch.cuda.set_device(args.device)
    time = datetime.datetime.now()

    # 构建保存路径
    save_path = os.path.join(args.model_name, '{}{:02d}{:02d}{:02d}{:02d}{:02d}'.format(time.year, time.month, time.day, time.hour,time.minute, time.second))

    # 生成模型
    model = DS_IB(feature_size=args.feature_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
    if args.pretrained_ckpt is not None:
        model.load_state_dict(torch.load(args.pretrained_ckpt))


    train_dataset = AnomalyDataset(args=args, train=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, pin_memory=True,
                              num_workers=1, shuffle=True)


    test_dataset = AnomalyDataset(args=args, train=False)

    test_loader = DataLoader(dataset=test_dataset, batch_size=10, pin_memory=True,
                             num_workers=1, shuffle=False)

    all_test_loader = [test_loader]

    #
    if not os.path.exists('./ckpt/' + save_path):
        os.makedirs('./ckpt/' + save_path)

    logger = False


    train(epochs=args.max_epoch, train_loader=train_loader, all_test_loader=all_test_loader, args=args, model=model, optimizer=optimizer, logger=logger, device=device, save_path=save_path)