import tensorflow as tf
import numpy as np
from PIL import Image
from scipy import linalg
from sklearn.preprocessing import MinMaxScaler

# Step 1: Load and preprocess GIFs
def load_gif(path, max_frames=100):
    frames = []
    with Image.open(path) as gif:
        for frame_index in range(min(gif.n_frames, max_frames)):
            gif.seek(frame_index)
            frame = gif.convert('RGB').resize((224, 224))  # Resize to a smaller dimension
            frame_array = np.array(frame)
            frames.append(frame_array)
    return np.array(frames)

# Step 2: Load pre-trained InceptionV3 model
def load_inception_model():
    base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling='avg')
    return base_model

# Step 3: Extract features
def extract_features(model, frames, batch_size=32):
    features = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i+batch_size]
        batch_features = model.predict(batch)
        features.append(batch_features)
    return np.concatenate(features)

# Step 4: Normalize features to a 0-1 range
def normalize_features(features):
    scaler = MinMaxScaler()
    return scaler.fit_transform(features)

# Step 5: Calculate FVD with normalized features
def calculate_fvd(features_real, features_generated):
    mu_real = np.mean(features_real, axis=0)
    mu_gen = np.mean(features_generated, axis=0)
    
    sigma_real = np.cov(features_real, rowvar=False)
    sigma_gen = np.cov(features_generated, rowvar=False)
    
    diff = mu_real - mu_gen
    covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_gen), disp=False)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fvd = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return fvd

# Main execution
def main():
    # Load GIFs
    gif1_path = '/raid/home/CAMCA/yl463/Video/TestDataApical_raw/gt/A2C.USm.1.2.840.114089.1.0.1.2887499449.1641693197.4144.17402_1_21.mp4.gif'
    gif2_path = '/raid/home/CAMCA/yl463/Video/TestDataApical_raw/pred/A2C.USm.1.2.840.114089.1.0.1.2887499449.1641693197.4144.17402_1_21.mp4_pred.gif' 
    
    frames1 = load_gif(gif1_path)
    frames2 = load_gif(gif2_path)

    # Preprocess frames
    frames1 = tf.keras.applications.inception_v3.preprocess_input(frames1)
    frames2 = tf.keras.applications.inception_v3.preprocess_input(frames2)

    # Load InceptionV3 model
    model = load_inception_model()

    # Extract features
    features1 = extract_features(model, frames1)
    features2 = extract_features(model, frames2)

    # Normalize features
    features1 = normalize_features(features1)
    features2 = normalize_features(features2)

    # Calculate FVD
    fvd_score = calculate_fvd(features1, features2)

    print(f"The FVD: {fvd_score}")

if __name__ == "__main__":
    main()