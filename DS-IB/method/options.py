import argparse
parser = argparse.ArgumentParser(description='SimiAtteen_Net')
parser.add_argument('--device', type=int, default=0, help='GPU ID')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate (default: 0.0001)')


parser.add_argument('--model_name', default='DS_IB', help=' ')


parser.add_argument('--loss_type', default='MIL', type=str, help='the type of n_pair loss, max_min_2, max_min, attention, attention_median, attention_H_L or max')
parser.add_argument('--pretrain', type=int, default=0)
parser.add_argument('--pretrained_ckpt', default=None, help='ckpt for pretrained model')  #
# parser.add_argument('--pretrained_ckpt', default=r"***.pkl", help='ckpt for pretrained model')


parser.add_argument('--Lambda', type=str, default='1_5', help='')

parser.add_argument('--testing_path', type=str, default=None, help='time file for test model')
parser.add_argument('--testing_model', type=str, default=None, help='iteration name for testing model')
parser.add_argument('--feature_size', type=int, default=512, help='size of feature (default: 2048)')

parser.add_argument('--batch_size', type=int, default=1, help='number of samples in one iteration')
parser.add_argument('--sample_size', type=int, default=30, help='number of samples in one iteration')

parser.add_argument('--sample_step', type=int, default=1, help='')
# parser.add_argument('--dataset_name', type=str, default='shanghaitech', help='')
parser.add_argument('--dataset_name', type=str, default='ucf-crime', help='')

parser.add_argument('--dataset_path', type=str, default='../dataset', help='path to dir contains anomaly datasets')
parser.add_argument('--feature_modal', type=str, default='rgb', help='features from different input, options contain rgb, flow , combine')

parser.add_argument('--max-seqlen', type=int, default=100, help='maximum sequence length during training (default: 750)')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--max_epoch', type=int, default=100, help='maximum iteration to train (default: 50000)')
parser.add_argument('--k', type=int, default=5, help='value of k')
parser.add_argument('--plot', type=int, default=0, help='whether plot the video anomalous map on testing')

parser.add_argument('--larger_mem', type=int, default=0, help='')
parser.add_argument('--snapshot', type=int, default=20, help='anomaly sample threshold')
# parser.add_argument('--ps', type=str, default='normal_loss_mean')
parser.add_argument('--label_type', type=str, default='unary')

