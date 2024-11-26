import cv2
import numpy as np
from skimage.measure import shannon_entropy
from skimage.filters.rank import entropy
from skimage.morphology import disk

component_1 = cv2.imread('results/OpenPano/flower/dimg12.jpg', cv2.IMREAD_GRAYSCALE)
component_2 = cv2.imread('results/OpenPano/flower/dimg34.jpg', cv2.IMREAD_GRAYSCALE)
component_images = [component_1, component_2]
final_panorama = cv2.imread('results/OpenPano/flower/final_panorama.jpg', cv2.IMREAD_GRAYSCALE)

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

def local_entropy(image, window_size=9):
    return entropy(image, disk(window_size // 2))

def calculate_f3(panorama_image):
    local_entropy_panorama = local_entropy(panorama_image)
    return np.mean(local_entropy_panorama)

f3 = calculate_f3(final_panorama)
print(f"f3 (Average Local Entropy): {f3}")
