# geometry_freq_pipeline.py
import cv2
import numpy as np
from scipy.fftpack import fft
from scipy.signal import savgol_filter
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
import math

def load_and_preprocess(path, target_size=256):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # simple thresholding + pad/resize keeping aspect
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # invert if needed (foreground dark)
    if np.mean(bw) > 127:
        bw = 255 - bw
    h, w = bw.shape
    scale = target_size / max(h,w)
    bw = cv2.resize(bw, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_NEAREST)
    pad = target_size - max(bw.shape)
    # pad to square
    top = (target_size - bw.shape[0])//2
    left = (target_size - bw.shape[1])//2
    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    canvas[top:top+bw.shape[0], left:left+bw.shape[1]] = bw
    return canvas

def contour_and_chain(bw):
    # find largest contour
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours)==0:
        return None
    c = max(contours, key=cv2.contourArea).squeeze()
    if c.ndim != 2:
        return None
    return c  # Nx2 array

def angle_histogram(contour):
    # compute turning angles along contour
    diffs = np.diff(contour, axis=0)
    thetas = np.arctan2(diffs[:,1], diffs[:,0])
    # unwrap and compute discrete curvature (delta theta)
    dtheta = np.diff(np.unwrap(thetas))
    # histogram
    ang = np.abs(dtheta)
    return ang, np.mean(ang), np.percentile(ang, 90), np.sum(ang>np.deg2rad(30))

def fourier_descriptor(contour, n_coeff=64):
    # complex sequence
    z = contour[:,0] + 1j*contour[:,1]
    zc = z - np.mean(z)
    fd = fft(zc)
    mag = np.abs(fd[:n_coeff])
    centroid = np.sum(np.arange(len(mag))*mag)/np.sum(mag)
    # normalized high-frequency energy
    hf_energy = np.sum(mag[int(len(mag)*0.5):]) / np.sum(mag)
    return centroid, hf_energy, mag

def curvature_wavelet_energy(contour, window=51):
    # curvature as in angle along chain (smoothed)
    diffs = np.diff(contour, axis=0)
    theta = np.arctan2(diffs[:,1], diffs[:,0])
    k = np.abs(np.diff(np.unwrap(theta)))
    if len(k) < window:
        k_s = k
    else:
        k_s = savgol_filter(k, window_length=window//2*2+1, polyorder=3)
    # compute spectral energy via FFT of curvature signal
    K = np.abs(np.fft.rfft(k_s))
    freqs = np.fft.rfftfreq(len(k_s), d=1.0)
    # bands: low[0:25%], mid[25:75%], high[75:100%]
    N = len(K)
    e_low = np.sum(K[:max(1,int(0.25*N))])
    e_mid = np.sum(K[int(0.25*N):int(0.75*N)])
    e_high = np.sum(K[int(0.75*N):])
    total = e_low+e_mid+e_high+1e-12
    return e_low/total, e_mid/total, e_high/total

def skeleton_branch_features(bw):
    sk = skeletonize(bw>0)
    # label connected skeleton components
    lab = label(sk)
    props = regionprops(lab)
    # branch count ~ number of endpoints / branchpoints approx
    # compute endpoints by convolution
    K = np.array([[1,1,1],[1,10,1],[1,1,1]])
    conv = cv2.filter2D(sk.astype(np.uint8), -1, K)
    endpoints = np.sum((sk) & (conv==11))
    branchpoints = np.sum((sk) & (conv>=13))
    return endpoints, branchpoints, len(props)

# Example usage for a single image
if __name__ == "__main__":
    path = "seal1.jpg"
    bw = load_and_preprocess(path)
    c = contour_and_chain(bw)
    if c is not None:
        ang_arr, ang_mean, ang90, ang_count = angle_histogram(c)
        centroid, hf_energy, mag = fourier_descriptor(c)
        low, mid, high = curvature_wavelet_energy(c)
        endpoints, branchpoints, comps = skeleton_branch_features(bw)
        print("angle_mean, ang90, ang_count:", ang_mean, ang90, ang_count)
        print("fourier centroid, hf_energy:", centroid, hf_energy)
        print("band energies (low,mid,high):", low, mid, high)
        print("skeleton endpoints/branchpoints:", endpoints, branchpoints)
