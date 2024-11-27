import cv2
import numpy as np
from scipy.stats import entropy
from scipy.ndimage import generic_filter

component_1 = cv2.imread('results/OpenPano/flower/dimg12.jpg')
component_2 = cv2.imread('results/OpenPano/flower/dimg34.jpg')
final_panorama = cv2.imread('results/OpenPano/flower/final_panorama.jpg')

component_1 = cv2.cvtColor(component_1, cv2.COLOR_BGR2GRAY)
component_2 = cv2.cvtColor(component_2, cv2.COLOR_BGR2GRAY)
component_images = [component_1, component_2]
final_panorama = cv2.cvtColor(final_panorama, cv2.COLOR_BGR2GRAY)
downscaled_image = cv2.resize(final_panorama, (final_panorama.shape[1] // 4, final_panorama.shape[0] // 4))
_bins = 128

# ==============================  f2  ==============================

def calculate_global_entropy(image):
    hist, _ = np.histogram(image.ravel(), bins=_bins, range=(0, _bins))
    hist = hist[hist > 0]  # Avoid log(0)
    return entropy(hist, base=2)


def calculate_local_entropy(image, window_size=9):
    # Downscale image to speed up calculation (optional)
    # downscaled_image = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4))

    padded_image = np.pad(image, pad_width=window_size // 2, mode='reflect')
    local_entropies = []

    for i in range(0, image.shape[0], window_size):  # Step by window size to avoid overlapping windows
        for j in range(0, image.shape[1], window_size):
            # Extract a sliding window
            window = padded_image[i:i + window_size, j:j + window_size]
            local_entropies.append(calculate_global_entropy(window))

    return np.array(local_entropies)

def calculate_f2(component_images, panorama_image):
    component_entropies = [calculate_global_entropy(img) for img in component_images]
    avg_entropy_components = np.mean(component_entropies)
    panorama_entropy = calculate_global_entropy(panorama_image)
    f2 = avg_entropy_components - panorama_entropy
    return f2

f2 = calculate_f2(component_images, final_panorama)
print(f"f2 (Differential Entropy): {f2}")

# ==============================  f3  ==============================

def calculate_f3(component_images, panorama_image):
    component_local_entropies = [np.mean(calculate_local_entropy(img)) for img in component_images]
    avg_local_entropy_components = np.mean(component_local_entropies)
    panorama_local_entropy = np.mean(calculate_local_entropy(panorama_image))
    f3 = panorama_local_entropy - avg_local_entropy_components
    return f3

f3 = calculate_f3(component_images, downscaled_image)
print(f"f3 (Average Local Entropy): {f3}")

# ==============================  f4  ==============================

def calculate_f4(component_images, panorama_image):
    component_local_entropies = [np.var(calculate_local_entropy(img)) for img in component_images]
    avg_variance_local_entropy_components = np.mean(component_local_entropies)

    panorama_variance_local_entropy = np.var(calculate_local_entropy(panorama_image))
    f4 = panorama_variance_local_entropy - avg_variance_local_entropy_components
    return f4

f4 = calculate_f4(component_images, final_panorama)
print(f"f4 (Differential Variance of Local Entropy): {f4}")

# ==============================  f9  ==============================

def calculate_f9(component_images, panorama_image):
    component_std_devs = [np.std(img) for img in component_images]
    avg_std_deviation_components = np.mean(component_std_devs)

    panorama_std_deviation = np.std(panorama_image)
    f9 = abs(panorama_std_deviation - avg_std_deviation_components)
    return f9

f9 = calculate_f9(component_images, final_panorama)
print(f"f9 (Absolute Difference of Standard Deviations): {f9}")

