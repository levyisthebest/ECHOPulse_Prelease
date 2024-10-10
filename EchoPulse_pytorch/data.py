from pathlib import Path
import re

import cv2
from PIL import Image
from functools import partial

from typing import Tuple, List
from beartype.door import is_bearable

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader as PytorchDataLoader
from torchvision import transforms as T, utils

from einops import rearrange

# helper functions


def exists(val):
    return val is not None


def identity(t, *args, **kwargs):
    return t


def pair(val):
    return val if isinstance(val, tuple) else (val, val)


def bgr_to_rgb(video_tensor):
    video_tensor = video_tensor[[2, 1, 0], :, :, :]
    return video_tensor


def cast_num_frames(t, *, frames):
    f = t.shape[1]

    if f == frames:
        return t

    if f > frames:
        return t[:, :frames]

    return F.pad(t, (0, 0, 0, 0, 0, frames - f))


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# image related helpers functions and dataset


class ImageDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts=['jpg', 'jpeg', 'png']
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(
            f'{folder}').glob(f'**/*.{ext}')]

        print(f'{len(self.paths)} training samples found at {folder}')

        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB')
                     if img.mode != 'RGB' else img),
            T.Resize(image_size),
            T.RandomHorizontalFlip(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

# tensor of shape (channels, frames, height, width) -> gif

# handle reading and writing gif


CHANNELS_TO_MODE = {
    1: 'L',
    3: 'RGB',
    4: 'RGBA'
}


def seek_all_images(img, channels=3):
    assert channels in CHANNELS_TO_MODE, f'channels {channels} invalid'
    mode = CHANNELS_TO_MODE[channels]

    i = 0
    while True:
        try:
            img.seek(i)
            yield img.convert(mode)
        except EOFError:
            break
        i += 1

# tensor of shape (channels, frames, height, width) -> gif


def video_tensor_to_pil_first_image(tensor):

    tensor = bgr_to_rgb(tensor)
    images = map(T.ToPILImage(), tensor.unbind(dim=1))
    first_img, *rest_imgs = images

    return first_img


def video_tensor_to_gif(
    tensor,
    path,
    duration=120,
    loop=0,
    optimize=True
):

    tensor = torch.clamp(tensor, min=0, max=1) # clipping underflow and overflow
    #tensor = bgr_to_rgb(tensor)
    images = map(T.ToPILImage(), tensor.unbind(dim=1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all=True, append_images=rest_imgs,
                   loop=loop, optimize=optimize)
    return images

# gif -> (channels, frame, height, width) tensor


def gif_to_tensor(
    path,
    channels=3,
    transform=T.ToTensor()
):
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, channels=channels)))
    return torch.stack(tensors, dim=1)

# handle reading and writing mp4


def video_to_tensor(
    path: str,
    transform,              # Path of the video to be imported
    num_frames=-1,        # Number of frames to be stored in the output tensor
    crop_size=None
) -> torch.Tensor:          # shape (1, channels, frames, height, width)

    video = cv2.VideoCapture(path)

    frames = []
    check = True
    while check:
        check, frame = video.read()

        if not check:
            continue

        frame = transform(frame)

        # if exists(crop_size):
        #    frame = crop_center(frame, *pair(crop_size))

        frames.append(rearrange(frame, '... -> 1 ...'))

    # convert list of frames to numpy array
    frames = np.array(np.concatenate(frames, axis=0))
    frames = rearrange(frames, 'f c h w -> c f h w')

    frames_torch = torch.tensor(frames).float()

    return frames_torch


def tensor_to_video(
    tensor,                # Pytorch video tensor
    path: str,             # Path of the video to be saved
    fps=8,              # Frames per second for the saved video
    video_format=('m', 'p', '4', 'v')
):
    # Import the video and cut it into frames.
    tensor = tensor.cpu()*255.  # TODO: have a better function for that? Not using cv2?

    num_frames, height, width = tensor.shape[-3:]

    # Changes in this line can allow for different video formats.
    fourcc = cv2.VideoWriter_fourcc(*video_format)
    video = cv2.VideoWriter(path, fourcc, fps, (width, height))

    frames = []

    for idx in range(num_frames):
        numpy_frame = tensor[:, idx, :, :].numpy()
        numpy_frame = np.uint8(rearrange(numpy_frame, 'c h w -> h w c'))
        video.write(numpy_frame)

    video.release()

    cv2.destroyAllWindows()

    return video


def crop_center(
    img,        # tensor
    cropx,      # Length of the final image in the x direction.
    cropy       # Length of the final image in the y direction.
) -> torch.Tensor:
    y, x, c = img.shape
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    return img[starty:(starty + cropy), startx:(startx + cropx), :]

def sort_key(file_path):
    # Extract the numerical parts from the file name using regex
    match = re.findall(r'(\d+)', file_path.stem)
    if match:
        return [int(part) for part in match]
    return str(file_path)
# video dataset

class VideoDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        channels=3,
        num_frames=11,
        horizontal_flip=False,
        force_num_frames=True,
        exts=['gif', 'mp4'],
        sample_texts=None  # 新增参数
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.channels = channels
        self.paths = [p for ext in exts for p in Path(
            f'{folder}').glob(f'**/*.{ext}')]
        self.paths.sort(key=sort_key)
        self.sample_texts = sample_texts
        self.transform = T.Compose([
            T.Resize(image_size),
            T.RandomHorizontalFlip() if horizontal_flip else T.Lambda(identity),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

        # TODO: rework so it is faster, for now it works but is bad
        self.transform_for_videos = T.Compose([
            T.ToPILImage(),  # added to PIL conversion because video is read with cv2
            T.Resize(image_size),
            T.RandomHorizontalFlip() if horizontal_flip else T.Lambda(identity),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

        # functions to transform video path to tensor

        self.gif_to_tensor = partial(
            gif_to_tensor, channels=self.channels, transform=self.transform)
        self.mp4_to_tensor = partial(
            video_to_tensor, transform=self.transform_for_videos, crop_size=self.image_size, num_frames=num_frames)

        self.cast_num_frames_fn = partial(
            cast_num_frames, frames=num_frames) if force_num_frames else identity

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        
        if ".mp4" not in str(path):
            print(str(path))
        ext = path.suffix

        if ext == '.gif':
            tensor = self.gif_to_tensor(path)
        elif ext == '.mp4':
            tensor = self.mp4_to_tensor(str(path))
        else:
            raise ValueError(f'unknown extension {ext}')

        tensor = self.cast_num_frames_fn(tensor)

        # 获取对应的文本，如果存在
        text = self.sample_texts[index % len(self.sample_texts)] if self.sample_texts else None

        return tensor, text  # 返回视频张量和对应的文本

# override dataloader to be able to collate strings


def collate_tensors_and_strings(data):
    if is_bearable(data, List[torch.Tensor]):
        return (torch.stack(data, dim=0),)

    data = zip(*data)
    output = []

    for datum in data:
        if is_bearable(datum, Tuple[torch.Tensor, ...]):
            datum = torch.stack(datum, dim=0)
        elif is_bearable(datum, Tuple[str, ...]):
            datum = list(datum)
        else:
            raise ValueError('detected invalid type being passed from dataset')

        output.append(datum)

    return tuple(output)


def DataLoader(*args, **kwargs):
    return PytorchDataLoader(*args, collate_fn=collate_tensors_and_strings, **kwargs)
