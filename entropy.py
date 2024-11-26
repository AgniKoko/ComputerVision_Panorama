import cv2
import numpy as np
from skimage.measure import shannon_entropy
from skimage.filters.rank import entropy
from skimage.morphology import disk

component_1 = cv2.imread('results/OpenPano/flower/dimg12.jpg', cv2.IMREAD_GRAYSCALE)
component_2 = cv2.imread('results/OpenPano/flower/dimg34.jpg', cv2.IMREAD_GRAYSCALE)
component_images = [component_1, component_2]
final_panorama = cv2.imread('results/OpenPano/flower/final_panorama.jpg', cv2.IMREAD_GRAYSCALE)

# ==============================  f2  ==============================

def global_entropy(image):
    return shannon_entropy(image)

def calculate_f2(component_images, panorama_image):
    entropies = [global_entropy(img) for img in component_images]
    avg_entropy_components = np.mean(entropies)

    panorama_entropy = global_entropy(panorama_image)

    f2 = avg_entropy_components - panorama_entropy
    return f2

f2 = calculate_f2(final_panorama, component_images)
print(f"f2 (Differential Entropy): {f2}")

# ==============================  f3  ==============================

def local_entropy(image, window_size=9):
    return entropy(image, disk(window_size // 2))

def calculate_f3(panorama_image):
    local_entropy_panorama = local_entropy(panorama_image)
    return np.mean(local_entropy_panorama)

f3 = calculate_f3(final_panorama)
print(f"f3 (Average Local Entropy): {f3}")

# ==============================  f4  ==============================

def calculate_f4(component_images, panorama_image, window_size=9):
    local_entropies_components = [local_entropy(img, window_size) for img in component_images]
    variance_components = [np.var(ent) for ent in local_entropies_components]
    avg_variance_components = np.mean(variance_components)

    local_entropy_panorama = local_entropy(panorama_image, window_size)
    variance_panorama = np.var(local_entropy_panorama)

    return variance_panorama - avg_variance_components

f4 = calculate_f4(component_images, final_panorama)
print(f"f4 (Differential Variance of Local Entropy): {f4}")
