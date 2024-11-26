import cv2
import numpy as np
from skimage.measure import shannon_entropy


def global_entropy(image):
    return shannon_entropy(image)


def calculate_f2(component_images, panorama_image):
    entropies = [global_entropy(img) for img in component_images]
    avg_entropy_components = np.mean(entropies)

    panorama_entropy = global_entropy(panorama_image)

    f2 = avg_entropy_components - panorama_entropy
    return f2
