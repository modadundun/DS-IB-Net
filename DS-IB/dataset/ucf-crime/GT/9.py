import pickle
import os


def convert_pickle_to_txt(input_pickle_path, output_txt_path):
    """
    读取 pickle 文件，将其中的字典数据转换成特定格式的文本文件。

    Args:
        input_pickle_path (str): 输入的 .pkl 文件路径。
        output_txt_path (str): 输出的 .txt 文件路径。
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_pickle_path):
        print(f"错误：找不到 Pickle 文件 '{input_pickle_path}'。")
        print("请确保文件名正确，并且文件与脚本在同一目录下。")
        return

    print(f"正在加载 Pickle 文件: {input_pickle_path}")

    # --- 1. 加载 Pickle 文件 ---
    try:
        with open(input_pickle_path, 'rb') as f:
            # 从 pickle 文件中加载 Python 对象（字典）
            data_dict = pickle.load(f)
    except Exception as e:
        print(f"加载 Pickle 文件时发生错误: {e}")
        return

    if not isinstance(data_dict, dict):
        print("错误：Pickle 文件中的数据不是预期的字典格式。")
        return

    print(f"文件加载成功，包含 {len(data_dict)} 条数据。")

    # --- 2. 格式化数据并准备写入 ---
    lines_to_write = []
    for key, value_list in data_dict.items():
        # 将浮点数列表转换为用点号(.)连接的字符串
        # 例如 [1.0, 1.0, 0.0] -> "1.0.1.0.0.0"
        value_string = ".".join(map(str, value_list))

        # 组装成最终的行格式
        formatted_line = f"{key}:[{value_string}]"
        lines_to_write.append(formatted_line)

    # --- 3. 写入到文本文件 ---
    print(f"正在写入到文本文件: {output_txt_path}")
    try:
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines_to_write))
        print("文件转换成功！")
    except Exception as e:
        print(f"写入文本文件时发生错误: {e}")


# --- 主程序入口 ---
if __name__ == "__main__":
    # 定义输入和输出文件名
    pickle_file = 'video_label_10crop.pickle'
    # 定义一个新的输出文件名，以防覆盖原始文件
    text_file = 'video_label_10crop.txt'

    # 执行转换函数
    convert_pickle_to_txt(pickle_file, text_file)