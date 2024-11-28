import cv2
from entropy_calc import calculate_entropy

# ==============================  GES-50  ==============================

component_1 = cv2.imread('results/GES-50/panorama12.jpg')
component_2 = cv2.imread('results/GES-50/panorama34.jpg')
final_panorama = cv2.imread('results/GES-50/final_panorama.jpg')
f2, f3, f4, f9 = calculate_entropy(component_1, component_2, final_panorama, save_plot_path='results/GES-50/plot.jpg')
print("")

# ==============================  NISwGSP  ==============================

component_1 = cv2.imread('results/NISwGSP/panorama12.jpg')
component_2 = cv2.imread('results/NISwGSP/panorama34.jpg')
final_panorama = cv2.imread('results/NISwGSP/final_panorama.jpg')
f2, f3, f4, f9 = calculate_entropy(component_1, component_2, final_panorama, save_plot_path='results/NISwGSP/plot.jpg')
print("")

# ==============================  flower  ==============================

component_1 = cv2.imread('results/OpenPano/flower/panorama12.jpg')
component_2 = cv2.imread('results/OpenPano/flower/panorama34.jpg')
final_panorama = cv2.imread('results/OpenPano/flower/final_panorama.jpg')
f2, f3, f4, f9 = calculate_entropy(component_1, component_2, final_panorama, save_plot_path='results/OpenPano/flower/plot.jpg')
print("")

# ==============================  augo - scene1  ==============================

component_1 = cv2.imread('results/augo/scene1/panorama12.jpg')
component_2 = cv2.imread('results/augo/scene1/panorama34.jpg')
final_panorama = cv2.imread('results/augo/scene1/final_panorama.jpg')
f2, f3, f4, f9 = calculate_entropy(component_1, component_2, final_panorama, save_plot_path='results/augo/scene1/plot.jpg')
print("")

# ==============================  augo - scene2  ==============================

component_1 = cv2.imread('results/augo/scene2/panorama12.jpg')
component_2 = cv2.imread('results/augo/scene2/panorama34.jpg')
final_panorama = cv2.imread('results/augo/scene2/final_panorama.jpg')
f2, f3, f4, f9 = calculate_entropy(component_1, component_2, final_panorama, save_plot_path='results/augo/scene2/plot.jpg')
print("")
