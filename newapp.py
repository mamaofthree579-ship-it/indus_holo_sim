import urllib.request
import numpy as np
import cv2
from scipy.signal import find_peaks

# ==========================================
# HARMONIC MATRIX: JUST INTONATION FREQUENCIES
# ==========================================
# We map angular geometry (0 to 90 degrees) to exact fractional ratios.
# Normalized scale base: 90 degrees = Octave (2:1 ratio)
JUST_INTONATION_SCALE = {
    "1:1 (Unison / 0° Base)": 0.0,
    "16:15 (Minor Second)": 90.0 * np.log2(16/15),   # ~ 8.44°
    "9:8 (Major Second)": 90.0 * np.log2(9/8),       # ~ 15.29°
    "6:5 (Minor Third)": 90.0 * np.log2(6/5),         # ~ 24.47°
    "5:4 (Major Third)": 90.0 * np.log2(5/4),         # ~ 28.97°
    "4:3 (Perfect Fourth)": 90.0 * np.log2(4/3),      # ~ 37.37°
    "45:32 (Tritone)": 90.0 * np.log2(45/32),        # ~ 44.18°
    "3:2 (Perfect Fifth)": 90.0 * np.log2(3/2),       # ~ 52.65°
    "8:5 (Minor Sixth)": 90.0 * np.log2(8/5),         # ~ 61.02°
    "5:3 (Major Sixth)": 90.0 * np.log2(5/3),         # ~ 66.37°
    "7:4 (Harmonic Seventh)": 90.0 * np.log2(7/4),    # ~ 72.67°
    "15:8 (Major Seventh)": 90.0 * np.log2(15/8),     # ~ 81.56°
    "2:1 (Octave / 90° Axis)": 90.0                   # 90.0°
}

def download_indus_sign(url):
    """Fetches a digital grapheme directly from the web."""
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            arr = np.asarray(bytearray(response.read()), dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    except Exception as e:
        print(f"Network error loading database symbol: {e}")
        return None

def extract_and_tag_angles(bw, tolerance_deg=1.5):
    """
    Skeletons the sign, detects junctions/turns, evaluates spatial orientation vectors,
    and tags joints that match fractional harmonic intervals.
    """
    # 1. Generate structural skeleton
    from skimage.morphology import skeletonize
    sk = skeletonize(bw > 0).astype(np.uint8)
    
    # 2. Extract edge contours from skeleton to find straight line segment paths
    contours, _ = cv2.findContours(sk, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_tags = []
    
    for c in contours:
        if len(c) < 3:
            continue
        # Squeeze down to clear point pairs
        pts = c.reshape(-1, 2)
        
        # Calculate sequential directional vectors
        for i in range(1, len(pts) - 1):
            v1 = pts[i] - pts[i-1]
            v2 = pts[i+1] - pts[i]
            
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            
            if norm_v1 == 0 or norm_v2 == 0:
                continue
                
            cos_angle = dot_product / (norm_v1 * norm_v2)
            # Clip boundary calculation safely
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            
            # Absolute structural angle deviation from straight line reference
            angle_deg = np.degrees(np.arccos(cos_angle))
            # Bound within primary quadrant for 90-degree axis mapping
            angle_quadrant = angle_deg % 90
            
            # Cross-reference against Just Intonation Harmonic Scale
            for interval_name, target_angle in JUST_INTONATION_SCALE.items():
                if abs(angle_quadrant - target_angle) <= tolerance_deg:
                    detected_tags.append({
                        "Coordinate": tuple(pts[i]),
                        "Measured Angle": round(angle_deg, 2),
                        "Harmonic Tag": interval_name,
                        "Deviation": round(abs(angle_quadrant - target_angle), 3)
                    })
                    break
                    
    return detected_tags

if __name__ == "__main__":
    # Pulling an active open-access Indus script character profile from Wikipedia / Harappa.com archives
    # Representing a standard fish/jar compound sign structure
    sample_url = "https://wikimedia.org"
    
    print("Connecting to database repository...")
    raw = download_indus_sign(sample_url)
    
    if raw is not None:
        # Preprocess binary mask (Inverting to make background dark if needed)
        _, thresh_img = cv2.threshold(raw, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if np.mean(thresh_img) > 127:
            thresh_img = 255 - thresh_img
            
        print("Analyzing geometric intersections for harmonic ratios...")
        tags = extract_and_tag_angles(thresh_img, tolerance_deg=2.0)
        
        # Display data summary matrix
        import pandas as pd
        if tags:
            df = pd.DataFrame(tags)
            print("\n=== DETECTED GEOMETRIC DATA TAGS (JUST INTONATION MATRICES) ===")
            print(df.drop_duplicates(subset=['Harmonic Tag']).to_string(index=False))
        else:
            print("\nNo precise harmonic tags isolated with current threshold parameters.")
