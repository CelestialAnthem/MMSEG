import os
import glob
import logging
import torch
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from mmseg.apis import init_model, inference_model
import numpy as np
import cv2

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

config_path = 'configs/dinov2/dinov2_vitg_mask2former_240k_mapillary_v2-672x672.py'
checkpoint_path = '/share/tengjianing/songyuhao/segmentation/models/0609_data_mapillary_18000/best_mIoU_iter_30000.pth'

id = "map"
img_dir = "/root/mmsegmentation/data/test/standard"

def list_image_files_in_directory(directory):
    jpg_files = glob.glob(os.path.join(directory, '**', '*.jpg'), recursive=True)
    png_files = glob.glob(os.path.join(directory, '**', '*.png'), recursive=True)
    jpeg_files = glob.glob(os.path.join(directory, '**', '*.jpeg'), recursive=True)
    
    return jpg_files + png_files + jpeg_files

img_list = list_image_files_in_directory(img_dir)
print(img_list)

out_dir = "/root/mmsegmentation/data/inf/20k"

os.makedirs(out_dir, exist_ok=True)

# 初始化模型并加载检查点
logger.info('Initializing model...')
model = init_model(config_path, checkpoint_path, device='cuda:2')
logger.info('Model initialized successfully.')

classes = ('Bird', 'Ground Animal', 'Curb', 'Fence', 'Guard Rail', 'Barrier',
           'Wall', 'Bike Lane', 'Crosswalk - Plain', 'Curb Cut', 'Parking',
           'Pedestrian Area', 'Rail Track', 'Road', 'Service Lane',
           'Sidewalk', 'Bridge', 'Building', 'Tunnel', 'Person', 'Bicyclist',
           'Motorcyclist', 'Other Rider', 'Lane Marking - Crosswalk',
           'Lane Marking - General', 'Mountain', 'Sand', 'Sky', 'Snow',
           'Terrain', 'Vegetation', 'Water', 'Banner', 'Bench', 'Bike Rack',
           'Billboard', 'Catch Basin', 'CCTV Camera', 'Fire Hydrant',
           'Junction Box', 'Mailbox', 'Manhole', 'Phone Booth', 'Pothole',
           'Street Light', 'Pole', 'Traffic Sign Frame', 'Utility Pole',
           'Traffic Light', 'Traffic Sign (Back)', 'Traffic Sign (Front)',
           'Trash Can', 'Bicycle', 'Boat', 'Bus', 'Car', 'Caravan',
           'Motorcycle', 'On Rails', 'Other Vehicle', 'Trailer', 'Truck',
           'Wheeled Slow', 'Car Mount', 'Ego Vehicle')
palette = [[165, 42, 42], [0, 192, 0], [196, 196, 196], [190, 153, 153],
           [180, 165, 180], [90, 120, 150], [102, 102, 156], [128, 64, 255],
           [140, 140, 200], [170, 170, 170], [250, 170, 160], [96, 96, 96],
           [230, 150, 140], [128, 64, 128], [110, 110, 110], [244, 35, 232],
           [150, 100, 100], [70, 70, 70], [150, 120, 90], [220, 20, 60],
           [255, 0, 0], [255, 0, 100], [255, 0, 200], [200, 128, 128],
           [255, 255, 255], [64, 170, 64], [230, 160, 50], [70, 130, 180],
           [190, 255, 255], [152, 251, 152], [107, 142, 35], [0, 170, 30],
           [255, 255, 128], [250, 0, 30], [100, 140, 180], [220, 220, 220],
           [220, 128, 128], [222, 40, 40], [100, 170, 30], [40, 40, 40],
           [33, 33, 33], [100, 128, 160], [142, 0, 0], [70, 100, 150],
           [210, 170, 100], [153, 153, 153], [128, 128, 128], [0, 0, 80],
           [250, 170, 30], [192, 192, 192], [220, 220, 0], [140, 140, 20],
           [119, 11, 32], [150, 0, 255], [0, 60, 100], [0, 0, 142],
           [0, 0, 90], [0, 0, 230], [0, 80, 100], [128, 64, 64], [0, 0, 110],
           [0, 0, 70], [0, 0, 192], [32, 32, 32], [120, 10, 10]]

category_colors = {i: tuple(color) for i, color in enumerate(palette)}

def adjust_brightness(color, factor):
    """
    调整颜色的亮度
    """
    return tuple(min(int(c * factor), 255) for c in color)

def is_position_valid(avg_x, avg_y, text_size, existing_labels, buffer=10):
    """
    检查标签位置是否有效，没有遮挡其他标签
    """
    for label_pos in existing_labels:
        lx, ly, lwidth, lheight = label_pos
        if not (avg_x + text_size[0] // 2 + buffer < lx or avg_x - text_size[0] // 2 - buffer > lx + lwidth or avg_y + text_size[1] // 2 + buffer < ly or avg_y - text_size[1] // 2 - buffer > ly + lheight):
            return False
    return True

def tensor_to_image(tensor, color_dict, classes, original_img, alpha=0.8):
    """
    将张量转换为图像，基于color_dict指示类别的颜色，并添加类别标签
    """
    tensor = tensor.squeeze(0).cpu().numpy()
    h, w = tensor.shape
    overlay = Image.new('RGBA', (w, h))
    pixels = overlay.load()

    # 创建绘图对象
    draw = ImageDraw.Draw(overlay)

    # 定义字体
    font = ImageFont.truetype("/root/mmsegmentation/data/Arial.ttf", 48)

    # 设置每个类别的标签位置
    label_positions = {}
    existing_labels = []

    for i in range(h):
        for j in range(w):
            category = int(tensor[i, j])
            pixels[j, i] = (*color_dict.get(category, (0, 0, 0)), int(255 * alpha))  # 增加透明度
            if category in label_positions:
                label_positions[category].append((j, i))
            else:
                label_positions[category] = [(j, i)]

    for category, positions in label_positions.items():
        if category < len(classes):
            # 创建二值图像，标记当前类别的位置
            mask = np.zeros((h, w), dtype=np.uint8)
            for pos in positions:
                mask[pos[1], pos[0]] = 255

            # 查找连通组件
            num_labels, labels_im = cv2.connectedComponents(mask)

            avg_positions = []
            for label in range(1, num_labels):  # 忽略背景
                component_mask = (labels_im == label)
                component_positions = np.column_stack(np.where(component_mask))
                avg_x = np.mean(component_positions[:, 1])
                avg_y = np.mean(component_positions[:, 0])
                avg_positions.append((int(avg_x), int(avg_y)))

            # 检查距离较近的区域并合并
            merged_positions = []
            while avg_positions:
                pos = avg_positions.pop(0)
                merged_cluster = [pos]
                to_remove = []
                for other_pos in avg_positions:
                    if np.linalg.norm(np.array(pos) - np.array(other_pos)) < 1500:  # 距离阈值，可调整
                        merged_cluster.append(other_pos)
                        to_remove.append(other_pos)
                for item in to_remove:
                    avg_positions.remove(item)
                merged_avg_x = int(np.mean([p[0] for p in merged_cluster]))
                merged_avg_y = int(np.mean([p[1] for p in merged_cluster]))
                merged_positions.append((merged_avg_x, merged_avg_y))

            # 如果区域是一个完整的形状或距离较近合并为一个就写一个标签，否则最多写3个标签
            if len(merged_positions) > 3:
                merged_positions = merged_positions[:2]

            # 绘制标签
            for avg_x, avg_y in merged_positions:
                text = classes[category]
                text_size = draw.textsize(text, font=font)

                # 检查标签位置是否有效，没有遮挡其他标签
                if not is_position_valid(avg_x, avg_y, text_size, existing_labels):
                    # 尝试找到一个新的位置
                    for dx, dy in [(-100, 0), (100, 0), (0, -100), (0, 100), (-100, -100), (100, 100), (-100, 100), (100, -100)]:
                        new_x, new_y = avg_x + dx, avg_y + dy
                        if is_position_valid(new_x, new_y, text_size, existing_labels):
                            avg_x, avg_y = new_x, new_y
                            break

                background_pos = (avg_x - (text_size[0]) // 2, avg_y - (text_size[1]) // 2)
                draw.rectangle([background_pos, (background_pos[0] + text_size[0], background_pos[1] + text_size[1])], fill=(0, 0, 0))
                
                bright_text_color = adjust_brightness(category_colors[category], 1.5)
                text_pos = (background_pos[0] + (text_size[0] - text_size[0]) // 2, background_pos[1] + (text_size[1] - text_size[1]) // 2)
                draw.text(text_pos, text, fill=bright_text_color, font=font)
                
                existing_labels.append((background_pos[0], background_pos[1], text_size[0], text_size[1]))

    # 将分割图像附加在原始图像上
    combined_img = Image.alpha_composite(original_img.convert('RGBA'), overlay)
    
    return combined_img.convert('RGB')  # 转换为RGB模式

# 处理图像并显示进度条
for img_path in tqdm(img_list, desc="Processing Images"):
    out_path = img_path.replace("test", "inf").replace("standard", id)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    logger.info(f'Processing image: {img_path}')
    original_img = Image.open(img_path)
    result = inference_model(model, img_path)
    tensor_img = result.pred_sem_seg.data[0]
    img_result = tensor_to_image(tensor_img, category_colors, classes, original_img)
    img_result.save(out_path, 'JPEG')
    logger.info(f'Processed and saved result for image: {out_path}')
