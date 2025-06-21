import numpy as np
from torch.utils.data import Dataset, DataLoader
import utils
import options
import os
import pickle
import random
import torch

class AnomalyDataset(Dataset):
    def __init__(self, args, train=True):
        self.args = args
        self.dataset_name = args.dataset_name.lower()
        self.dataset_path = args.dataset_path
        self.train = train
        self.t_max = args.max_seqlen

        if self.dataset_name == 'shanghaitech':
            self.feature_path = r"D:\SH_clip_video_features"
        elif self.dataset_name == 'ucf-crime':
            self.feature_path = r"D:\UCFClipFeatures"

        elif self.dataset_name == 'xd':
            self.feature_path = r"E:\XD\XDClipFeatures"

        else:
            raise ValueError(f"no support: {self.dataset_name}. ")

        list_base_path = os.path.join(self.dataset_path, args.dataset_name)
        self.trainlist_path = os.path.join(list_base_path, 'train_split_10crop.txt')
        self.testlist_path = os.path.join(list_base_path, 'test_split_10crop.txt')
        self.gt_path = os.path.join(list_base_path, 'GT', 'video_label_10crop.pickle')



        self.trainlist = self.txt2list(self.trainlist_path)
        self.testlist = self.txt2list(self.testlist_path)
        self.video_label_dict = self.pickle_reader(self.gt_path)

        self.normal_video_train, self.anomaly_video_train = self.p_n_split_dataset(self.video_label_dict, self.trainlist)

        if self.train:
            if len(self.normal_video_train) < args.sample_size:
                raise ValueError(f"normal num ({len(self.normal_video_train)}) less ({args.sample_size})")
            if len(self.anomaly_video_train) < args.sample_size:
                raise ValueError(f"anomaly num ({len(self.anomaly_video_train)}) less ({args.sample_size})")

    def txt2list(self, txtpath=''):
        try:
            with open(file=txtpath, mode='r') as f:
                filelist = f.readlines()
            return filelist
        except FileNotFoundError:
            print("no")
            return []
        except Exception as e:
            print(999)
            return []

    def pickle_reader(self, file=''):
        """读取 pickle 文件"""
        try:
            with open(file=file, mode='rb') as f:
                video_label_dict = pickle.load(f)
            return video_label_dict
        except FileNotFoundError:
            print(f"no find {file}")
            return {}
        except Exception as e:
            print("fail")
            return {}

    def p_n_split_dataset(self, video_label_dict, trainlist):

        normal_video_train = []
        anomaly_video_train = []
        missing_labels = 0
        for t in trainlist:
            video_name = t.strip().replace('Ped', 'ped')
            if not video_name: continue

            if video_name in video_label_dict:
                label = video_label_dict[video_name]
                if label == '[1.0]' or label == 1.0:
                    anomaly_video_train.append(video_name)
                else:
                    normal_video_train.append(video_name)
            else:
                missing_labels += 1


        if missing_labels > 0:
            print("no label")

        return normal_video_train, anomaly_video_train


    def __getitem__(self, index):
        if self.train:
            try:
                normaly_indexs = random.sample(self.normal_video_train, self.args.sample_size)
                anomaly_indexs = random.sample(self.anomaly_video_train, self.args.sample_size)
            except ValueError as e:
                 print("wrong")
                 return [torch.empty(0), torch.empty(0)], [torch.empty(0), torch.empty(0)]

            anomaly_features = []
            normaly_features = []

            for a_i, n_i in zip(anomaly_indexs, normaly_indexs):
                anomaly_data_video_name = a_i.strip()
                normaly_data_video_name = n_i.strip()


                try:
                    anomaly_feature_path = os.path.join(self.feature_path, anomaly_data_video_name + '.npy')
                    anomaly_feature = np.load(file=anomaly_feature_path)
                    anomaly_feature, _ = utils.process_feat_sample(anomaly_feature, self.t_max)
                    anomaly_features.append(torch.from_numpy(anomaly_feature).unsqueeze(0))

                    normaly_feature_path = os.path.join(self.feature_path, normaly_data_video_name + '.npy')
                    normaly_feature = np.load(file=normaly_feature_path)
                    normaly_feature, _ = utils.process_feat(normaly_feature, self.t_max, self.args.sample_step)
                    normaly_features.append(torch.from_numpy(normaly_feature).unsqueeze(0))

                except FileNotFoundError as e:
                    print("no find")

                    continue #
                except Exception as e:
                    print("wrong")
                    continue

            if not anomaly_features or not normaly_features:
                 print("wrong")

                 return [torch.empty(0), torch.empty(0)], [torch.empty(0), torch.empty(0)]

            anomaly_features = torch.cat(anomaly_features, dim=0).float()
            normaly_features = torch.cat(normaly_features, dim=0).float()


            normaly_label = torch.zeros((normaly_features.size(0), 1))
            anomaly_label = torch.ones((anomaly_features.size(0), 1))

            return [anomaly_features, normaly_features], [anomaly_label, normaly_label]

        else:
            data_video_name = self.testlist[index].strip()
            if not data_video_name:
                print("nothing")
                return None, None

            video_feature = None
            video_feature_path = os.path.join(self.feature_path, data_video_name + '.npy')
            try:
                video_feature_np = np.load(file=video_feature_path)
                video_feature = torch.from_numpy(video_feature_np.astype(np.float32))
            except FileNotFoundError:
                print(f"no find: {video_feature_path}")
                return None, data_video_name
            except Exception as e:
                print("wrong")
                # (移除了文本特征的None)
                return None, data_video_name


            return video_feature, data_video_name


    def __len__(self):
        if self.train:

            return len(self.trainlist)
        else:
            return len(self.testlist)

