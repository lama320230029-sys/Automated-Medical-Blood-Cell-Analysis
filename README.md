# Automated Blood Cell Image Enhancement and Classification Using Hybrid Spatial–Frequency Domain Processing and Machine Learning

**Project Phase:** Phase I — Image Preprocessing  
**Subject:** Medical Image Analysis — Blood Cell Classification Pipeline

---

## Abstract

This project presents an automated blood cell image preprocessing pipeline designed as the foundational stage for a machine learning-based classification system. The core objective is to enhance image quality through a combination of spatial and frequency domain techniques, thereby improving the reliability and accuracy of downstream feature extraction and classification. The pipeline covers resizing, noise removal, contrast enhancement, histogram equalization, normalization, and frequency domain filtering via the Fast Fourier Transform (FFT). The output is a standardized set of enhanced images optimized for subsequent machine learning tasks.

---

## 1. Problem Definition

Medical blood cell images are inherently susceptible to several quality degradation factors including sensor noise, low contrast, uneven illumination, and inter-sample inconsistency. These imperfections reduce the discriminative power of extracted features and consequently lower classification accuracy in machine learning models. A robust preprocessing pipeline is therefore critical before any feature extraction or model training is attempted.

---

## 2. Objectives

- Enhance blood cell image quality using established image processing techniques.
- Remove noise using multiple spatial-domain filtering strategies.
- Improve contrast through histogram-based and gamma correction methods.
- Apply frequency domain filtering (low-pass and high-pass) to capture both structural and edge-level information.
- Produce a clean, normalized image dataset ready for feature extraction and classification.

---

## 3. Dataset

| Property | Details |
|---|---|
| **Source** | Kaggle — `unclesamulus/blood-cells-image-dataset` |
| **Version** | Version 2 |
| **Dataset Root** | `/root/.cache/kagglehub/datasets/unclesamulus/blood-cells-image-dataset/versions/2/bloodcells_dataset` |
| **Number of Classes** | 8 |
| **Download Size** | ~268 MB |

### 3.1 Cell Classes

The dataset comprises **8 distinct blood cell categories**, automatically detected by scanning subdirectories containing more than 10 images each:

1. `basophil`
2. `eosinophil`
3. `erythroblast`
4. `ig` (Immature Granulocytes)
5. `lymphocyte`
6. `monocyte`
7. `neutrophil`
8. `platelet`

Five random samples per class were visualized to confirm dataset integrity and class diversity prior to any processing.

---

## 4. Preprocessing Pipeline

The pipeline was applied sequentially to blood cell images. Below is a detailed description of each stage.

### 4.1 Image Loading and Resizing

All images were loaded using OpenCV (`cv2.imread`) with color conversion from BGR to RGB for correct display. Images were then resized to a uniform spatial resolution of **224 × 224 pixels** — a standard input size aligned with widely used CNN architectures such as VGG, ResNet, and EfficientNet.

```python
resized = cv2.resize(img, (224, 224))
normalized = resized / 255.0
```

Following resizing, pixel intensities were normalized to the `[0, 1]` floating-point range by dividing by 255, ensuring consistent scale across all samples.

---

### 4.2 Noise Removal

Three spatial filtering methods were applied and compared:

| Filter | Method | Kernel/Parameters | Characteristic |
|---|---|---|---|
| **Gaussian Blur** | `cv2.GaussianBlur` | 5×5 kernel, σ=1 | Smooths uniform noise; mild blurring |
| **Median Filter** | `cv2.medianBlur` | 5×5 kernel | Effective against salt-and-pepper noise; preserves edges better than Gaussian |
| **Bilateral Filter** | `cv2.bilateralFilter` | d=9, σ_color=75, σ_space=75 | Edge-preserving smoothing; best structural fidelity |

The bilateral filter is particularly well-suited for medical imaging as it reduces noise while retaining the morphological boundaries of blood cell structures.

---

### 4.3 Contrast Enhancement

Contrast enhancement was performed on the grayscale version of the resized image using three distinct methods:

#### 4.3.1 Histogram Equalization
```python
hist_eq = cv2.equalizeHist(gray)
```
Redistributes pixel intensities across the full dynamic range `[0, 255]`. Suitable for globally low-contrast images but can over-enhance certain regions.

#### 4.3.2 CLAHE (Contrast Limited Adaptive Histogram Equalization)
```python
clahe = cv2.createCLAHE(2.0, (8, 8))
clahe_img = clahe.apply(gray)
```
Applies histogram equalization locally on 8×8 tiles with a clip limit of 2.0, preventing noise amplification. This is the **preferred method** for medical images as it enhances local detail without introducing artifacts.

#### 4.3.3 Gamma Correction
```python
gamma = 1.5
gamma_img = np.array(255 * (gray / 255) ** gamma, dtype='uint8')
```
A gamma value of 1.5 brightens the image by applying a non-linear intensity transformation. Useful for images captured under dim or uneven illumination conditions.

---

### 4.4 Edge Enhancement

Edge information is critical for differentiating blood cell types based on membrane shape and nuclear structure.

#### 4.4.1 Sharpening Filter
```python
kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
sharp = cv2.filter2D(resized, -1, kernel)
```
A Laplacian-based sharpening kernel that enhances local contrast at boundaries by subtracting a blurred version from the original.

#### 4.4.2 Sobel Edge Detection
```python
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, 3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, 3)
sobel = cv2.magnitude(sobelx, sobely)
```
Computes gradient magnitude in both horizontal and vertical directions. The combined Sobel response highlights the full boundary map of each cell, which is highly informative for shape-based feature extraction.

---

### 4.5 Frequency Domain Filtering (FFT-Based)

Frequency domain processing was performed using the `scipy.fft` module to analyze and manipulate the spectral content of images.

```python
f = fft2(gray)
fshift = fftshift(f)
```

Two complementary filters were constructed:

| Filter | Mask | Effect |
|---|---|---|
| **Low-Pass Filter** | Square mask of radius 30 around DC center | Retains low-frequency components (smooth regions, cell body); suppresses noise |
| **High-Pass Filter** | Complement of low-pass mask (`1 - low_mask`) | Retains high-frequency components (edges, fine texture, nucleus detail) |

Reconstruction was performed via Inverse FFT:
```python
low  = np.abs(ifft2(ifftshift(fshift * low_mask)))
high = np.abs(ifft2(ifftshift(fshift * high_mask)))
```

The FFT magnitude spectrum was also visualized using a logarithmic scale (`np.log(1 + |F|)`) to reveal the energy distribution across spatial frequencies.

---

## 5. Final Pipeline Summary

The complete preprocessing pipeline is summarized below:

```
Raw Image
    │
    ├─► Resize to 224×224
    ├─► Normalize to [0, 1]
    ├─► Noise Removal (Gaussian / Median / Bilateral)
    ├─► Contrast Enhancement (Hist Eq / CLAHE / Gamma)
    ├─► Edge Enhancement (Sharpening / Sobel)
    └─► Frequency Domain Filtering (FFT Low-Pass / High-Pass)
                │
                ▼
        Enhanced Image Set
        (Ready for Feature Extraction)
```

A 2×3 comparative visualization was produced showing: Original, Gaussian-denoised, Median-denoised, CLAHE-enhanced, Sharpened, and High-Pass filtered outputs side-by-side.

---

## 6. Results

| Processing Stage | Technique Applied | Outcome |
|---|---|---|
| Noise Removal | Gaussian, Median, Bilateral | Successfully reduced noise; bilateral preserved cell boundaries best |
| Contrast Enhancement | Histogram Eq, CLAHE, Gamma (γ=1.5) | Improved visibility of internal cell structures |
| Edge Enhancement | Sharpening kernel, Sobel gradient | Accentuated membrane and nuclear contours |
| Frequency Filtering | FFT-based Low-Pass & High-Pass | Improved structural clarity; separated smooth and textural components |
| Normalization | Pixel scaling to [0,1] | Ensured uniform input range for ML models |

---

## 7. Discussion

The multi-technique approach adopted in this phase ensures that the preprocessing pipeline is both **redundant** (multiple methods addressing the same degradation type) and **complementary** (spatial and frequency domain methods capturing different information). CLAHE was identified as the most medically appropriate contrast enhancement method due to its locality and noise control. The bilateral filter stands out among noise reduction methods for its edge-preservation property — critical when cell morphology is a key diagnostic feature.

The FFT-based high-pass filter, when combined with spatial sharpening, yields images with strong edge responses that are expected to improve the discriminability of shape-based features in subsequent phases.

---

## 8. Conclusion

Phase I successfully establishes a comprehensive and reproducible image preprocessing pipeline for the automated analysis of blood cell images. By applying a combination of spatial and frequency domain techniques, the pipeline significantly improves image quality across all eight cell classes. The resulting enhanced images exhibit reduced noise, improved contrast, and sharper structural detail — all of which are expected to translate into improved feature extraction performance and higher classification accuracy in Phase II.

---

## 9. Tools & Libraries Used

| Library | Version / Role |
|---|---|
| `opencv-python` (cv2) | Image I/O, filtering, resizing, CLAHE |
| `numpy` | Array operations, gamma correction, masking |
| `scipy.fft` | FFT, inverse FFT, frequency domain processing |
| `matplotlib` | Visualization of all processing stages |
| `kagglehub` | Dataset download and management |

---

*End of Phase I Report*
