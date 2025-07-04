import numpy as np  # 导入NumPy库
import os  # 导入os模块
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot模块
plt.switch_backend('agg')
import zipfile
import io
import torch


def random_extract(feat, t_max):
    """
    随机提取特征序列中的子序列
    """
    r = np.random.randint(len(feat)-t_max)
    return feat[r:r+t_max], r


def random_extract_step(feat, t_max, step):
    """
    使用步长随机提取特征序列中的子序列
    """
    if len(feat) - step * t_max > 0:
        r = np.random.randint(len(feat) - step * t_max)
    else:
        r = np.random.randint(step)
    return feat[r:r+t_max:step], r


def random_perturb(feat, length):
    """
    随机扰动特征序列
    """
    samples = np.arange(length) * len(feat) / length
    for i in range(length):
        if i < length - 1:
            if int(samples[i]) != int(samples[i + 1]):
                samples[i] = np.random.choice(range(int(samples[i]), int(samples[i + 1]) + 1))
            else:
                samples[i] = int(samples[i])
        else:
            if int(samples[i]) < length - 1:
                samples[i] = np.random.choice(range(int(samples[i]), length))
            else:
                samples[i] = int(samples[i])
    return feat[samples.astype('int')], samples.astype('int')


def pad(feat, min_len):
    if np.shape(feat)[0] <= min_len:
       return np.pad(feat, ((0, min_len-np.shape(feat)[0]), (0, 0)), mode='constant', constant_values=0)
    else:
       return feat


def process_feat(feat, length, step):

    if len(feat) > length:
        if step and step > 1:
            features, r = random_extract_step(feat, length, step)
            return pad(features, length), r
        else:
            features, r = random_extract(feat, length)
            return features, r
    else:
        return pad(feat, length), 0


def process_feat_sample(feat, length):
    if len(feat) > length:
            features, samples = random_perturb(feat, length)
            return features, samples
    else:
        return pad(feat, length), 0


def scorebinary(scores=None, threshold=0.5):
    """

    """
    scores_threshold = scores.copy()
    scores_threshold[scores_threshold < threshold] = 0  #
    scores_threshold[scores_threshold >= threshold] = 1  #
    return scores_threshold  #


def fill_context_mask(mask, sizes, v_mask, v_unmask):
    """

    """
    mask.fill_(v_unmask)  #
    n_context = mask.size(2)  #
    for i, size in enumerate(sizes):  #
        if size < n_context:  #
            mask[i, :, size:] = v_mask  #
    return mask  # 返回填充后的掩码


def median(attention_logits, args):
    """
    使用中位数对注意力权重进行归一化
    """
    attention_medians = torch.zeros(0).to(args.device)  #
    batch_size = attention_logits.shape[0]  #
    for i in range(batch_size):  #
        attention_logit = attention_logits[i][attention_logits[i] > 0].unsqueeze(0)  #
        attention_medians = torch.cat((attention_medians, attention_logit.median(1, keepdims=True)[0]), dim=0)  #
    attention_medians = attention_medians.unsqueeze(1)  #
    attention_logits_mask = attention_logits.clone()  #
    attention_logits_mask[attention_logits <= attention_medians] = 0  #
    attention_logits_mask[attention_logits > attention_medians] = 1  #
    attention_logits = attention_logits * attention_logits_mask  #
    attention_logits_sum = attention_logits.sum(dim=2, keepdim=True)  #
    attention_logits = attention_logits / attention_logits_sum  #
    return attention_logits  #


def anomap(predict_dict, label_dict, save_path, itr, save_root, zip=False, width=15, height=5):
    """

    """
    if not os.path.exists(os.path.join(save_root, save_path, 'plot')):  #
        os.makedirs(os.path.join(save_root, save_path, 'plot'))  #
    for k, v in predict_dict.items():  #
        predict_np = v.repeat(16)  #
        k = k[:-2]
        label_np = label_dict[k][:len(predict_np)]  #
        x = np.arange(len(predict_np))  #
        plt.figure(figsize=(width, height))  #
        plt.plot(x, predict_np, color='b', label='predicted scores', linewidth=1)  #
        label_np = np.array(label_np)  #
        # plt.fill_between(x, label_np, where=label_np > 0, facecolor="r", alpha=0.3)  #
        plt.fill_between(x, label_np, where=label_np > '0', facecolor="r", alpha=0.3)  #

        plt.yticks(np.arange(0, 1.1, step=0.1))  #
        plt.xlabel('Frames')  #
        plt.ylabel('Anomaly scores')  #
        plt.grid(True, linestyle='-.')  #
        plt.legend()  #
        output_dir = os.path.join(save_root, save_path, 'plot', 'itr_{}'.format(itr))
        if not os.path.exists(output_dir):  #
            os.makedirs(output_dir)  #
        plt.savefig(os.path.join(output_dir, k + '.svg'), format='svg')  #
        plt.close()  #

