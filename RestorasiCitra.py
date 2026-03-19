import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from scipy import signal

# ==========================================
# 1. MEMBUAT CITRA ASLI
# ==========================================
def create_image():
    img = np.zeros((256,256), dtype=np.uint8)

    # objek
    cv2.rectangle(img,(50,50),(100,100),200,-1)
    cv2.circle(img,(180,100),30,180,-1)

    # grid
    for i in range(0,256,20):
        cv2.line(img,(i,0),(i,255),150,1)
        cv2.line(img,(0,i),(255,i),150,1)

    # teks
    cv2.putText(img,'TEST',(120,180),
                cv2.FONT_HERSHEY_SIMPLEX,1,220,2)

    return img


# ==========================================
# 2. MEMBUAT PSF MOTION BLUR
# ==========================================
def motion_psf(length=15, angle=30):
    psf = np.zeros((length,length))
    center = length//2
    angle = np.deg2rad(angle)

    x1 = int(center - (length/2)*np.cos(angle))
    y1 = int(center - (length/2)*np.sin(angle))
    x2 = int(center + (length/2)*np.cos(angle))
    y2 = int(center + (length/2)*np.sin(angle))

    cv2.line(psf,(x1,y1),(x2,y2),1,1)
    psf = psf / np.sum(psf)

    return psf


# ==========================================
# 3. DEGRADASI CITRA
# ==========================================
def add_motion_blur(img, psf):
    return cv2.filter2D(img.astype(float), -1, psf)

def add_gaussian_noise(img, sigma=20):
    noise = np.random.normal(0, sigma, img.shape)
    return np.clip(img + noise, 0, 255)

def add_salt_pepper(img, prob=0.05):
    noisy = img.copy()
    rand = np.random.rand(*img.shape)

    noisy[rand < prob/2] = 0
    noisy[rand > 1 - prob/2] = 255

    return noisy


# ==========================================
# 4. RESTORASI
# ==========================================
def inverse_filter(img, psf, eps=1e-3):
    G = np.fft.fft2(img)
    H = np.fft.fft2(psf, s=img.shape)

    F_hat = G / (H + eps)
    result = np.abs(np.fft.ifft2(F_hat))

    return np.clip(result, 0, 255)


def wiener_filter(img, psf, K=0.01):
    G = np.fft.fft2(img)
    H = np.fft.fft2(psf, s=img.shape)

    H_conj = np.conj(H)
    H_abs = np.abs(H)**2

    F_hat = (H_conj / (H_abs + K)) * G
    result = np.abs(np.fft.ifft2(F_hat))

    return np.clip(result, 0, 255)


def lucy_richardson(img, psf, iterations=15):
    img = img.astype(float)
    estimate = img.copy()

    psf_flip = np.flip(psf)

    for i in range(iterations):
        conv = cv2.filter2D(estimate, -1, psf)
        conv[conv == 0] = 1e-8

        ratio = img / conv
        estimate = estimate * cv2.filter2D(ratio, -1, psf_flip)

    return np.clip(estimate, 0, 255)


# ==========================================
# 5. METRIK
# ==========================================
def mse(original, restored):
    return np.mean((original - restored) ** 2)

def psnr(original, restored):
    return 10 * np.log10(255**2 / mse(original, restored))

def ssim(img1, img2):
    mu1 = img1.mean()
    mu2 = img2.mean()

    sigma1 = img1.var()
    sigma2 = img2.var()
    sigma12 = ((img1 - mu1)*(img2 - mu2)).mean()

    C1 = 6.5025
    C2 = 58.5225

    return ((2*mu1*mu2 + C1)*(2*sigma12 + C2)) / \
           ((mu1**2 + mu2**2 + C1)*(sigma1 + sigma2 + C2))


# ==========================================
# 6. SPEKTRUM FREKUENSI
# ==========================================
def show_spectrum(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    spectrum = np.log(1 + np.abs(fshift))
    return spectrum


# ==========================================
# 7. PIPELINE UTAMA
# ==========================================
img = create_image()
psf = motion_psf(15, 30)

# 3 SKENARIO SESUAI SOAL
blur = add_motion_blur(img, psf)
gaussian_blur = add_gaussian_noise(blur, 20)
sp_blur = add_salt_pepper(blur, 0.05)

datasets = {
    "Motion Blur": blur,
    "Gaussian + Blur": gaussian_blur,
    "SaltPepper + Blur": sp_blur
}

# ==========================================
# 8. PROSES & OUTPUT
# ==========================================
for name, data in datasets.items():

    print("\n====================================")
    print("SCENARIO:", name)
    print("====================================")

    # INVERSE
    start = time.time()
    inv = inverse_filter(data, psf)
    t_inv = time.time() - start

    # WIENER
    start = time.time()
    wien = wiener_filter(data, psf, 0.01)
    t_wien = time.time() - start

    # RL
    start = time.time()
    rl = lucy_richardson(data, psf, 15)
    t_rl = time.time() - start

    # METRIK
    print("Inverse -> PSNR:", psnr(img, inv),
          "MSE:", mse(img, inv),
          "SSIM:", ssim(img, inv),
          "Time:", t_inv)

    print("Wiener -> PSNR:", psnr(img, wien),
          "MSE:", mse(img, wien),
          "SSIM:", ssim(img, wien),
          "Time:", t_wien)

    print("RL -> PSNR:", psnr(img, rl),
          "MSE:", mse(img, rl),
          "SSIM:", ssim(img, rl),
          "Time:", t_rl)

    # ======================================
    # VISUALISASI CITRA
    # ======================================
    plt.figure(figsize=(12,6))

    plt.subplot(2,4,1)
    plt.imshow(img, cmap='gray')
    plt.title("Original")

    plt.subplot(2,4,2)
    plt.imshow(data, cmap='gray')
    plt.title("Degraded")

    plt.subplot(2,4,3)
    plt.imshow(inv, cmap='gray')
    plt.title("Inverse")

    plt.subplot(2,4,4)
    plt.imshow(wien, cmap='gray')
    plt.title("Wiener")

    plt.subplot(2,4,5)
    plt.imshow(rl, cmap='gray')
    plt.title("Lucy-Richardson")

    # ======================================
    # SPEKTRUM FREKUENSI
    # ======================================
    plt.subplot(2,4,6)
    plt.imshow(show_spectrum(data), cmap='gray')
    plt.title("Spectrum Degraded")

    plt.subplot(2,4,7)
    plt.imshow(show_spectrum(wien), cmap='gray')
    plt.title("Spectrum Wiener")

    plt.subplot(2,4,8)
    plt.imshow(show_spectrum(rl), cmap='gray')
    plt.title("Spectrum RL")

    plt.tight_layout()
    plt.show()