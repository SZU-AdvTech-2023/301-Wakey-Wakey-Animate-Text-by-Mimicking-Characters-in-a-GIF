from PIL import Image, ImageSequence
import numpy as np
from ultralytics import YOLO  # 假设有一个名为 yolo_model.py 的文件包含 YOLO 模型的代码
from segment_anything import SamPredictor
import os
from tools import plot_bboxes
# 加载 YOLO 模型
yolo_model = YOLO("yolov8n.pt")

# 加载 SAM 模型
#segment anything
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = "./notebooks/sam_vit_h_4b8939.pth"
DEVICE = "cpu" #cpu,cuda
from segment_anything import sam_model_registry
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)

def process_frame(image_path):
    original_image = np.asarray(Image.open(image_path))
    # 使用 YOLO 检测物体边界框
    yolo_results = yolo_model.predict(original_image)
    target_box = yolo_results[0].to('cpu').boxes.data[0].numpy()  # 假设取第一个检测到的物体
    # plot_bboxes(original_image, yolo_results[0].boxes.data, score=False)
    # 提取主体部分
    #target_image = original_image.crop((target_box[0], target_box[1], target_box[2], target_box[3]))

    # 使用 SAM 模型进行分割
    mask_predictor = SamPredictor(sam)
    mask_predictor.set_image(original_image)
    mask, _, _ = mask_predictor.predict(box=target_box[:-2])

    # 应用分割结果
    segmented_image = original_image.copy()
    print(np.shape(segmented_image))
    print(np.shape(mask))
    segmented_image[mask[1] == 0] = [255, 255, 255]  # 将背景部分设为白色
    # 显示结果或保存图像
    return Image.fromarray(segmented_image)

# 读取待处理的图像
image_path = "C:\\Users\\talon\\Desktop\\Lufy4.gif"
original_gif = Image.open(image_path)
processed_frames=[]
for idx,frame in enumerate(ImageSequence.Iterator(original_gif)):
    jpeg_path = os.path.join("C:\\Users\\talon\\Desktop\\frames\\", f"frame_{idx:03d}.jpg")
    jpeg_path2 = os.path.join("C:\\Users\\talon\\Desktop\\frames2\\", f"frame_{idx:03d}.jpg")
    frame.convert("RGB").save(jpeg_path, "JPEG")
    processed_frame=process_frame(jpeg_path)
    processed_frame.convert("RGB").save(jpeg_path2, "JPEG")
    processed_frames.append(processed_frame)
processed_frames[0].save(
    "C:\\Users\\talon\\Desktop\\Lufy_R.gif",
    save_all=True,
    append_images=processed_frames[1:],
    duration=original_gif.info['duration'],  # 保留原始GIF的帧速率
    loop=original_gif.info['loop']  # 保留原始GIF的循环设置
)
