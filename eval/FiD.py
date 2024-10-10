import numpy as np
from PIL import Image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
from scipy.linalg import sqrtm
import os
def extract_frames_from_gif(gif_path):
    frames = []
    with Image.open(gif_path) as im:
        try:
            while True:
                frame = im.copy()
                frames.append(frame.convert('RGB'))  # 转换为RGB格式
                im.seek(im.tell() + 1)
        except EOFError:
            pass
    return frames

def get_activations(images, model):
    activations = []
    for img in images:
        img = img.resize((299, 299))  # InceptionV3的输入尺寸
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        act = model.predict(x)
        activations.append(act[0])
    activations = np.array(activations)
    return activations

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    # 计算协方差矩阵的平方根
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    # 处理数值误差导致的复数结果
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # 计算FID
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

# 主程序
path1 = '/raid/home/CAMCA/yl463/Video/TestDataApical_raw/gt/'
path2 = '/raid/home/CAMCA/yl463/Video/TestDataApical_raw/pred/'
files1 = os.listdir(path1)
files2 = os.listdir(path2)
re = []
for i in range(len(files1)):
    gif_path1 = path1 + files1[i]   # 第一个GIF文件路径
    gif_path2 = path2 + files2[i]   # 第二个GIF文件路径

    # 提取两个GIF的帧
    frames1 = extract_frames_from_gif(gif_path1)
    frames2 = extract_frames_from_gif(gif_path2)

    # 加载预训练的InceptionV3模型
    model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

    # 获取激活值
    act1 = get_activations(frames1, model)
    act2 = get_activations(frames2, model)

    # 计算均值和协方差
    mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)

    # 计算FID
    fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    print(f'两个GIF之间的FID值为: {fid_value}')
    re.append(fid_value/11)
print(np.mean(re))
