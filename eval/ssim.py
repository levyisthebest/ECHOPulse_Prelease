import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import os

def load_gif(path, num_frames=11):
    frames = []
    try:
        with Image.open(path) as gif:
            for frame_index in range(num_frames):
                try:
                    gif.seek(frame_index)  # Move to the correct frame
                    frame = gif.convert('RGB').resize((224, 224))
                    frame = np.array(frame)  # Convert to numpy array
                    frames.append(frame)
                except EOFError:
                    # Not enough frames in GIF, stop reading
                    print(f"Warning: Reached end of GIF {path} at frame {frame_index}")
                    break
    except Exception as e:
        print(f"Error loading GIF {path}: {str(e)}")
    return frames

def calculate_ssim(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score

def gif_ssim(gif1_path, gif2_path, num_frames=11):
    frames1 = load_gif(gif1_path, num_frames)
    frames2 = load_gif(gif2_path, num_frames)

    min_frames = min(len(frames1), len(frames2))
    
    if min_frames == 0:
        print(f"Error: No frames could be loaded from either {gif1_path} or {gif2_path}")
        return 0

    ssim_scores = []
    for frame1, frame2 in zip(frames1[:min_frames], frames2[:min_frames]):
        ssim_score = calculate_ssim(frame1, frame2)
        ssim_scores.append(ssim_score)

    return np.mean(ssim_scores) if ssim_scores else 0  # Avoid division by zero if no frames are loaded

# Main function to calculate SSIM for the two GIFs
path1 = '/raid/home/CAMCA/yl463/Video/TestDataApical_raw/gt/'
path2 = '/raid/home/CAMCA/yl463/Video/TestDataApical_raw/pred/'
files1 = os.listdir(path1)
files2 = os.listdir(path2)

re = []
for i in range(min(len(files1), len(files2))):  # Ensure we have matching files
    gif_path1 = os.path.join(path1, files1[i])   # First GIF file path
    gif_path2 = os.path.join(path2, files2[i])   # Second GIF file path
    average_ssim = gif_ssim(gif_path1, gif_path2)
    print(f"Average SSIM for GIFs {files1[i]} and {files2[i]}: {average_ssim}")
    re.append(average_ssim)

print(f"Overall Average SSIM: {np.mean(re)}")
