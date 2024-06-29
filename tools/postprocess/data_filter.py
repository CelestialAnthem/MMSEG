import os
import glob
import json
import matplotlib.pyplot as plt
import numpy as np


s = 0
t = 80000
def get_all_json_files(root_dir):
    # 使用glob模块递归搜索所有的json文件
    json_files = glob.glob(os.path.join(root_dir, '**', '*.json'), recursive=True)
    return json_files

def merge_and_flatten_json_files(json_file_list):
    merged_data = []
    
    for json_file in json_file_list:
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for key, value in data.items():
                if isinstance(value, list):
                    merged_data.extend(value)
                else:
                    merged_data.append(value)
    
    # 按照 image_score 排序
    merged_data.sort(key=lambda x: x.get('image_score', 0), reverse=True)
    
    return merged_data

def load_jsonl(jsonl_file_path):
    with open(jsonl_file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

def match_top_data_with_jsonl(top_data, jsonl_data):
    jsonl_dict = {entry['img_path']: entry for entry in jsonl_data}
    matched_data = [jsonl_dict[entry['img_path']] for entry in top_data if entry['img_path'] in jsonl_dict]
    return matched_data

# 指定根目录
root_directory = '/mnt/ve_share/songyuhao/seg_cleanlab/res'

# 获取所有json文件的完全路径
all_json_files = get_all_json_files(root_directory)

# 过滤包含"processed"的json文件
all_json_list = [json_file for json_file in all_json_files if "processed" in json_file]

# 合并和拉平成单个列表，并按 image_score 排序
flattened_sorted_data = merge_and_flatten_json_files(all_json_list)

# 获取前20000条数据
top_20000_data = flattened_sorted_data[:t]

# 加载jsonl文件
jsonl_file_path = '/share/tengjianing/songyuhao/segmentation/datasets/0617/train_80000_p.jsonl'
jsonl_data = load_jsonl(jsonl_file_path)

# 匹配top20000数据与jsonl数据
matched_data = match_top_data_with_jsonl(top_20000_data, jsonl_data)

# Extract image_scores for histogram
image_scores = [entry.get('image_score', 0) for entry in flattened_sorted_data]

# Plot histogram of image_scores
plt.figure(figsize=(10, 6))
plt.hist(image_scores, bins=50, color='blue', edgecolor='black')
plt.title('Histogram of Image Scores')
plt.xlabel('Image Score')
plt.ylabel('Frequency')
output_image_path = '/root/image_score_histogram.png'
plt.savefig(output_image_path)

# 保存匹配后的json数据
output_file = '/share/tengjianing/songyuhao/segmentation/datasets/0626/top%d_%d___test.json' % (s, t)
with open(output_file, 'w', encoding='utf-8') as outfile:
    json.dump(matched_data, outfile, ensure_ascii=False, indent=4)

print(f"Matched JSON data saved to {output_file}")
