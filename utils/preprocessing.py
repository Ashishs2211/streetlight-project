"""
utils/preprocessing.py
=======================
Image preprocessing and low-light enhancement utilities
for the Streetlight Outage Detection system.

Techniques used:
  - CLAHE (Contrast Limited Adaptive Histogram Equalization)
  - Gamma correction
  - Denoising
  - Brightness/contrast normalization
"""

import cv2
import numpy as np


# ─────────────────────────────────────────────
# 1. CLAHE — Adaptive Contrast Enhancement
# ─────────────────────────────────────────────
def apply_clahe(image: np.ndarray,
                clip_limit: float = 2.0,
                tile_grid: tuple = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE on the L-channel of the image (LAB color space).
    Excellent for improving visibility in dark street scenes.

    Args:
        image      : BGR image (numpy array)
        clip_limit : Threshold for contrast limiting
        tile_grid  : Size of grid for histogram equalization

    Returns:
        Enhanced BGR image
    """
    # Convert BGR → LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Apply CLAHE only to the L (luminance) channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_enhanced = clahe.apply(l_channel)

    # Merge back and convert to BGR
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    return enhanced


# ─────────────────────────────────────────────
# 2. Gamma Correction — Brighten Dark Images
# ─────────────────────────────────────────────
def gamma_correction(image: np.ndarray, gamma: float = 1.5) -> np.ndarray:
    """
    Apply gamma correction to brighten or darken an image.
    gamma > 1.0 → brightens (useful for night-time images)
    gamma < 1.0 → darkens

    Args:
        image : BGR image (numpy array)
        gamma : Gamma value (default 1.5 for mild brightening)

    Returns:
        Gamma-corrected BGR image
    """
    inv_gamma = 1.0 / gamma
    # Build a lookup table for fast pixel-wise transformation
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in np.arange(256)
    ]).astype("uint8")
    return cv2.LUT(image, table)


# ─────────────────────────────────────────────
# 3. Denoising — Remove Noise from Low-Light
# ─────────────────────────────────────────────
def denoise_image(image: np.ndarray,
                  h: int = 10,
                  template_window: int = 7,
                  search_window: int = 21) -> np.ndarray:
    """
    Apply Non-Local Means Denoising (suitable for color images).
    Helps when camera ISO is high in dark environments.

    Args:
        image           : BGR image
        h               : Filter strength (higher = more smoothing)
        template_window : Size of template patch (odd number)
        search_window   : Size of search window (odd number)

    Returns:
        Denoised BGR image
    """
    return cv2.fastNlMeansDenoisingColored(
        image, None, h, h, template_window, search_window
    )


# ─────────────────────────────────────────────
# 4. Auto Brightness & Contrast Normalization
# ─────────────────────────────────────────────
def auto_brightness_contrast(image: np.ndarray,
                              clip_percent: float = 1.0) -> np.ndarray:
    """
    Automatically stretch the histogram to use the full 0–255 range.
    Clips a small percentage of pixels to avoid outlier influence.

    Args:
        image        : BGR image
        clip_percent : Percentage of pixels to clip from each end

    Returns:
        Contrast-normalized BGR image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clip = int(gray.size * clip_percent / 100)

    # Compute histogram and cumulative distribution
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    cumulative = np.cumsum(hist)

    # Find lower and upper bounds after clipping
    lo = np.searchsorted(cumulative, clip)
    hi = np.searchsorted(cumulative, cumulative[-1] - clip)
    hi = max(hi, lo + 1)  # Ensure hi > lo

    # Scale pixel values
    alpha = 255.0 / (hi - lo)
    beta = -lo * alpha
    result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return result


# ─────────────────────────────────────────────
# 5. Composite Low-Light Enhancement Pipeline
# ─────────────────────────────────────────────
def enhance_low_light(image: np.ndarray,
                      use_clahe: bool = True,
                      use_gamma: bool = True,
                      use_denoise: bool = False,
                      gamma_value: float = 1.4) -> np.ndarray:
    """
    Full low-light enhancement pipeline combining multiple techniques.
    Applied automatically during inference for nighttime images.

    Args:
        image      : Input BGR image
        use_clahe  : Apply CLAHE enhancement
        use_gamma  : Apply gamma correction
        use_denoise: Apply denoising (slower, optional)
        gamma_value: Gamma for correction

    Returns:
        Enhanced BGR image
    """
    enhanced = image.copy()

    # Step 1 — Denoise (optional, computationally expensive)
    if use_denoise:
        enhanced = denoise_image(enhanced)

    # Step 2 — Gamma correction to lift dark regions
    if use_gamma:
        enhanced = gamma_correction(enhanced, gamma=gamma_value)

    # Step 3 — CLAHE for adaptive local contrast
    if use_clahe:
        enhanced = apply_clahe(enhanced)

    return enhanced


# ─────────────────────────────────────────────
# 6. Detect Image Brightness (Auto-trigger)
# ─────────────────────────────────────────────
def is_low_light(image: np.ndarray, threshold: int = 80) -> bool:
    """
    Check if an image is dark (low-light condition).
    Used to automatically decide whether to apply enhancement.

    Args:
        image     : BGR image
        threshold : Mean brightness threshold (0–255)

    Returns:
        True if image is considered low-light
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    return mean_brightness < threshold


# ─────────────────────────────────────────────
# 7. Resize for YOLO Input
# ─────────────────────────────────────────────
def preprocess_for_inference(image: np.ndarray,
                              target_size: int = 640,
                              auto_enhance: bool = True) -> np.ndarray:
    """
    Full preprocessing pipeline before feeding image to YOLO.
    Automatically applies low-light enhancement if needed.

    Args:
        image       : Raw BGR image
        target_size : YOLO input size (default 640)
        auto_enhance: Auto-detect and enhance dark images

    Returns:
        Preprocessed BGR image (original size, enhanced if needed)
    """
    processed = image.copy()

    # Auto-apply low-light enhancement if image is dark
    if auto_enhance and is_low_light(processed):
        processed = enhance_low_light(processed)

    return processed
