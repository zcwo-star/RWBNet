import pandas as pd
import glob
import numpy as np
import os
from scipy.fft import fft2, fftfreq
from skimage.feature import graycomatrix, graycoprops
import pywt
from PIL import Image
import cv2

def extract_fft_features(image):
    """Extract FFT spectral patterns from an image."""
    # Convert to grayscale if not already
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Compute 2D FFT
    f = fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)  # Avoid log(0)
    # Extract statistical features from the spectrum
    return {
        'fft_mean': np.mean(magnitude_spectrum),
        'fft_std': np.std(magnitude_spectrum),
        'fft_max': np.max(magnitude_spectrum)
    }

def extract_glcm_features(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """Extract GLCM texture features from an image."""
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Ensure image is in uint8 format for GLCM
    image = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)
    glcm = graycomatrix(image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    features = {
        'glcm_contrast': graycoprops(glcm, 'contrast').flatten(),
        'glcm_dissimilarity': graycoprops(glcm, 'dissimilarity').flatten(),
        'glcm_homogeneity': graycoprops(glcm, 'homogeneity').flatten(),
        'glcm_energy': graycoprops(glcm, 'energy').flatten(),
        'glcm_correlation': graycoprops(glcm, 'correlation').flatten()
    }
    return features

def extract_dwt_features(image, wavelet='db1', level=1):
    """Extract DWT coefficients from an image."""
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Perform DWT
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    cA, (cH, cV, cD) = coeffs if level == 1 else coeffs[0], coeffs[1]
    # Extract statistical features from approximation and detail coefficients
    return {
        'dwt_cA_mean': np.mean(cA),
        'dwt_cA_std': np.std(cA),
        'dwt_cH_mean': np.mean(cH),
        'dwt_cH_std': np.std(cH),
        'dwt_cV_mean': np.mean(cV),
        'dwt_cV_std': np.std(cV),
        'dwt_cD_mean': np.mean(cD),
        'dwt_cD_std': np.std(cD)
    }

def main():
    print("Starting multimodal feature extraction for weld seam detection...")
    # Path for weld seam image files (e.g., PNG, JPG)
    file_path = "/path/to/weld_seam_images"  # Update this path
    all_files = glob.glob(file_path + "/*.png")  # Assuming PNG images; adjust as needed

    # DataFrame to store features
    feature_list = []

    # Iterate through each image file
    for file in all_files:
        path = file
        sample_id = os.path.basename(path).replace('.png', '')  # Adjust extension as needed

        # Load image
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Failed to load image: {path}")
            continue

        # Extract features
        fft_features = extract_fft_features(image)
        glcm_features = extract_glcm_features(image)
        dwt_features = extract_dwt_features(image)

        # Flatten GLCM features (since they are arrays for multiple angles)
        glcm_flat = {
            key: np.mean(value) for key, value in glcm_features.items()
        }

        # Combine all features into a single dictionary
        features = {
            'SAMPLE_ID': sample_id,
            **fft_features,
            **glcm_flat,
            **dwt_features
        }

        # Append to feature list
        feature_list.append(features)

    # Create DataFrame from feature list
    df_master = pd.DataFrame(feature_list)

    # Optionally, add ground truth labels if available (e.g., defect type)
    # For now, assuming no labels are provided
    # df_master['DEFECT_TYPE'] = ... (add logic if you have labels)

    # Save features to CSV
    output_path = "/path/to/output_features"  # Update this path
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, 'weld_seam_features.csv')
    df_master.to_csv(output_file, encoding='utf-8', index=False)
    print(f"Features saved to {output_file}")

if __name__ == "__main__":
    main()