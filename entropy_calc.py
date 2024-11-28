import cv2
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt

_bins = 128

def calculate_entropy(component_1, component_2, final_panorama, save_plot_path=None):
    component_1 = cv2.cvtColor(component_1, cv2.COLOR_BGR2GRAY)
    component_2 = cv2.cvtColor(component_2, cv2.COLOR_BGR2GRAY)
    component_images = [component_1, component_2]
    final_panorama = cv2.cvtColor(final_panorama, cv2.COLOR_BGR2GRAY)
    downscaled_image = cv2.resize(final_panorama, (final_panorama.shape[1] // 4, final_panorama.shape[0] // 4))

    # ==============================  f2  ==============================

    def calculate_global_entropy(image):
        hist, _ = np.histogram(image.ravel(), bins=_bins, range=(0, _bins))
        hist = hist[hist > 0]  # Avoid log(0)
        prob_dist = hist / hist.sum()
        image_entropy = entropy(prob_dist, base=2)
        return image_entropy, hist


    def calculate_local_entropy(image, window_size=9):
        # Downscale image to speed up calculation (debug)

        padded_image = np.pad(image, pad_width=window_size // 2, mode='reflect')
        local_entropies = []

        # Ensure that the image dimensions are divisible by window_size to avoid mismatched shapes
        height, width = image.shape
        for i in range(0, height - window_size + 1, window_size):  # Step by window size
            for j in range(0, width - window_size + 1, window_size):
                window = padded_image[i:i + window_size, j:j + window_size]
                local_entropies.append(calculate_global_entropy(window)[0])  # Only entropy value

        return np.array(local_entropies)

    def calculate_f2(component_images, panorama_image):
        component_entropies = [calculate_global_entropy(img)[0] for img in component_images]
        avg_entropy_components = np.mean(component_entropies)
        panorama_entropy = calculate_global_entropy(panorama_image)[0]
        f2 = avg_entropy_components - panorama_entropy
        return f2

    f2 = calculate_f2(component_images, final_panorama)
    print(f"f2 (Differential Entropy): {f2}")

    # ==============================  f3  ==============================

    def calculate_f3(component_images, panorama_image):
        component_local_entropies = [np.mean(calculate_local_entropy(img)) for img in component_images]
        panorama_local_entropy = np.mean(calculate_local_entropy(panorama_image))
        f3 = panorama_local_entropy - np.mean(component_local_entropies)
        return f3

    # f3 = calculate_f3(component_images, downscaled_image)
    f3 = calculate_f3(component_images, final_panorama)
    print(f"f3 (Average Local Entropy): {f3}")

    # ==============================  f4  ==============================

    def calculate_f4(component_images, panorama_image):
        component_local_entropies = [calculate_local_entropy(img) for img in component_images]
        panorama_local_entropy = calculate_local_entropy(panorama_image)
        component_variance = np.var([np.mean(local_entropy) for local_entropy in component_local_entropies])
        panorama_variance = np.var(np.mean(panorama_local_entropy))
        f4 = panorama_variance - component_variance
        return f4

    f4 = calculate_f4(component_images, final_panorama)
    print(f"f4 (Differential Variance of Local Entropy): {f4}")

    # ==============================  f9  ==============================

    def calculate_f9(component_images, panorama_image):
        component_std_devs = [np.std(img) for img in component_images]
        panorama_std_dev = np.std(panorama_image)
        f9 = abs(np.mean(component_std_devs) - panorama_std_dev)
        return f9

    f9 = calculate_f9(component_images, final_panorama)
    print(f"f9 (Absolute Difference of Standard Deviations): {f9}")

    # ==============================  plot  ==============================

    if save_plot_path:
        entropy_value, hist = calculate_global_entropy(final_panorama)
        plt.figure(figsize=(10, 6))
        plt.hist(hist, bins=_bins, density=True, color='blue', alpha=0.7)
        plt.title(f"Histogram of Image (Entropy: {entropy_value:.2f})")
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Probability Density')
        plt.savefig(save_plot_path)
        plt.close()

    return f2, f3, f4, f9
