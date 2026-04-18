# Automated Blood Cell Image Enhancement and Classification Using Hybrid Spatial–Frequency Domain Processing and Machine Learning

**Project Phase:** Phase I — Image Preprocessing  
**Date:** April 2026  
**Subject:** Medical Image Analysis — Blood Cell Classification Pipeline

---

## Abstract

This project presents an automated blood cell image preprocessing pipeline designed as the foundational stage for a machine learning-based classification system. The core objective is to enhance image quality through a combination of spatial and frequency domain techniques, thereby improving the reliability and accuracy of downstream feature extraction and classification. The pipeline covers resizing, noise removal, contrast enhancement, histogram equalization, normalization, and frequency domain filtering via the Fast Fourier Transform (FFT). The dataset used contains 8 blood cell classes sourced from Kaggle, and the output is a standardized set of enhanced images optimized for subsequent machine learning tasks.

---

## Problem Definition

Medical blood cell images are inherently susceptible to several quality degradation factors including sensor noise, low contrast, uneven illumination, and inter-sample inconsistency. These imperfections reduce the discriminative power of extracted features and consequently lower classification accuracy in machine learning models. Raw images also vary in size and intensity scale, making them incompatible with standard ML pipelines without prior preprocessing. A robust preprocessing pipeline is therefore critical before any feature extraction or model training is attempted.

---

## Objectives and Methodology

### Objectives

- Enhance blood cell image quality using established image processing techniques.
- Remove noise using multiple spatial-domain filtering strategies.
- Improve contrast through histogram-based and gamma correction methods.
- Apply frequency domain filtering (low-pass and high-pass) to capture both structural and edge-level information.
- Produce a clean, normalized image dataset ready for feature extraction and classification.

---

### Methodology

#### 1. Dataset Acquisition and Exploration

The dataset was downloaded from Kaggle (`unclesamulus/blood-cells-image-dataset`, Version 2, ~268 MB) using `kagglehub`. A root-detection function `find_root()` was written to automatically locate the correct dataset directory by scanning for subfolders containing more than 10 images. The dataset contains **8 blood cell classes**: basophil, eosinophil, erythroblast, ig, lymphocyte, monocyte, neutrophil, and platelet. Five random samples per class were visualized to confirm dataset integrity.

---

#### 2. Resizing and Normalization

All images were loaded using OpenCV and converted from BGR to RGB. They were resized to a uniform **224 × 224 pixels** — a standard input size aligned with CNN architectures such as VGG and ResNet. Pixel intensities were then normalized to the `[0, 1]` range by dividing by 255, ensuring consistent scale across all samples and stable model training.

```python
resized = cv2.resize(img, (224, 224))
normalized = resized / 255.0
```

---

#### 3. Noise Removal

Three spatial filtering methods were applied and compared:

| Filter | Parameters | Characteristic |
|---|---|---|
| **Gaussian Blur** | 5×5 kernel, σ=1 | Uniform smoothing; reduces random noise |
| **Median Filter** | 5×5 kernel | Removes salt-and-pepper noise; preserves edges better than Gaussian |
| **Bilateral Filter** | d=9, σ_color=75, σ_space=75 | Edge-preserving smoothing; best structural fidelity for medical images |

```python
gaussian  = cv2.GaussianBlur(resized, (5,5), 1)
median    = cv2.medianBlur(resized, 5)
bilateral = cv2.bilateralFilter(resized, 9, 75, 75)
```

---

#### 4. Contrast Enhancement

Three contrast enhancement methods were applied to the grayscale image:

**Histogram Equalization** — redistributes pixel intensities across the full `[0, 255]` range globally.

**CLAHE (Contrast Limited Adaptive Histogram Equalization)** — applies histogram equalization locally on 8×8 tiles with a clip limit of 2.0, preventing noise amplification. This is the preferred method for medical images.

**Gamma Correction** — applies a non-linear intensity transformation with γ = 1.5 to brighten midtone regions.

```python
hist_eq   = cv2.equalizeHist(gray)
clahe     = cv2.createCLAHE(2.0, (8,8)); clahe_img = clahe.apply(gray)
gamma_img = np.array(255 * (gray / 255) ** 1.5, dtype='uint8')
```

---

#### 5. Edge Enhancement

**Sharpening Filter** — a Laplacian-based 3×3 kernel that amplifies the center pixel while subtracting its neighbors, making cell boundaries more defined.

**Sobel Edge Detection** — computes the gradient magnitude in horizontal and vertical directions to produce a full boundary map of each cell.

```python
kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
sharp  = cv2.filter2D(resized, -1, kernel)

sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, 3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, 3)
sobel  = cv2.magnitude(sobelx, sobely)
```

---

#### 6. Frequency Domain Filtering (FFT)

The 2D Fast Fourier Transform was applied to convert images into the frequency domain, where low frequencies represent smooth regions and high frequencies represent fine details and edges.

- A **low-pass filter** (60×60 center mask) was applied to retain smooth structural information and suppress noise.
- A **high-pass filter** (inverse of low-pass) was applied to isolate edges and fine texture detail.

Both filtered outputs were reconstructed via Inverse FFT and visualized alongside the FFT magnitude spectrum (log scale).

```python
f      = fft2(gray);  fshift = fftshift(f)
low    = np.abs(ifft2(ifftshift(fshift * low_mask)))
high   = np.abs(ifft2(ifftshift(fshift * high_mask)))
```

---

## Results and Interpretation

### Summary of Outcomes

| Processing Stage | Technique | Result |
|---|---|---|
| Resizing | 224×224 resize | Uniform spatial dimensions across all samples |
| Normalization | Pixel ÷ 255 | Consistent intensity scale in [0, 1] |
| Noise Removal | Gaussian, Median, Bilateral | Noise successfully reduced; bilateral best preserved cell boundaries |
| Contrast Enhancement | Histogram Eq, CLAHE, Gamma (γ=1.5) | Improved visibility of internal cell structures; CLAHE performed best locally |
| Edge Enhancement | Sharpening kernel, Sobel gradient | Cell membrane and nuclear contours clearly accentuated |
| Frequency Filtering | FFT Low-Pass & High-Pass | Smooth cell body and fine edge detail successfully separated |

### Interpretation

- **Noise Removal:** The bilateral filter proved most effective for medical images as it reduced noise while preserving the morphological boundaries of blood cells — boundaries that are diagnostically critical.
- **Contrast Enhancement:** CLAHE outperformed global histogram equalization by enhancing contrast locally (per 8×8 tile), revealing internal cell structures such as nuclear granularity without introducing artifacts.
- **Edge Enhancement:** The combined use of Sobel edge detection and the sharpening kernel produced images with strong, well-defined cell boundaries, which are expected to significantly improve shape-based feature extraction in Phase II.
- **Frequency Domain Filtering:** The high-pass filtered output clearly isolated edge and texture information invisible in the spatial domain alone. The low-pass output confirmed that smooth cell body structure was cleanly retained. This dual representation enriches the feature space for classification.

### Conclusion

The preprocessing pipeline significantly enhances blood cell image quality across all 8 classes and prepares the data for robust machine learning classification. By systematically addressing noise, contrast, edge definition, and frequency-domain structure, the pipeline ensures that Phase II feature extraction will operate on the highest quality input possible.

---

*End of Phase I Report*
