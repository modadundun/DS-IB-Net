

import pickle

def load_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

if __name__ == "__main__":
    pickle_file = 'video_label.pickle'

    # 加载 pickle 文件
    loaded_data = load_pickle_file(pickle_file)

    # 打印加载的数据
    print("Loaded data from pickle file:")
    print(loaded_data)
