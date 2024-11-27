import cv2
import numpy as np
import cv2 as cv

sift = cv.xfeatures2d_SIFT.create(400)

def resize_image(img, width=999):
    h, w = img.shape[:2]
    aspect_ratio = h / w
    new_height = int(width * aspect_ratio)
    return cv2.resize(img, (width, new_height))

# ==============================  12  ==============================

img1 = cv.imread('augo/scene1/1.jpg')
img1 = resize_image(img1)
cv.namedWindow('img1')
cv.imshow('img1', img1)
cv.waitKey(0)
kp1 = sift.detect(img1)
desc1 = sift.compute(img1, kp1)

img2 = cv.imread('augo/scene1/2.jpg')
img2 = resize_image(img2)
cv.namedWindow('img2')
cv.imshow('img2', img2)
cv.waitKey(0)
kp2 = sift.detect(img2)
desc2 = sift.compute(img2, kp2)

def match_features(desc1, desc2):
    bf = cv.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:  # Lowe's ratio test
            good_matches.append(m)
    return good_matches

matches12 = match_features(desc1[1], desc2[1])

dimg12 = cv.drawMatches(img1, desc1[0], img2, desc2[0], matches12, None)
cv.namedWindow('dimg12')
cv.imshow('dimg12', dimg12)
cv2.imwrite('results/augo/scene1/dimg12.jpg', dimg12)
cv.waitKey(0)

img_pt1 = []
img_pt2 = []

def create_panorama(img1, img2, kp1, kp2, desc1, desc2, matches):
    img_pt1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    img_pt2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    M, mask = cv.findHomography(img_pt2, img_pt1, cv.RANSAC, 5.0)

    panorama = cv.warpPerspective(img2, M, (img1.shape[1] + 1000, img1.shape[0] + 1000))
    panorama[0:img1.shape[0], 0:img1.shape[1]] = img1

    return panorama

panorama12 = create_panorama(img1, img2, kp1, kp2, desc1, desc2, matches12)
cv.namedWindow('panorama12')
cv.imshow('panorama12', panorama12)
cv2.imwrite('results/augo/scene1/panorama12.jpg', panorama12)
kp12 = sift.detect(panorama12)
desc12 = sift.compute(panorama12, kp12)
cv.waitKey(0)

gray12 = cv2.cvtColor(panorama12, cv2.COLOR_BGR2GRAY)
_, thresh12 = cv2.threshold(gray12, 1, 255, cv2.THRESH_BINARY)
x, y, w, h = cv2.boundingRect(thresh12)
cropped_panorama12 = panorama12[y:y+h, x:x+w]

# ==============================  34  ==============================

img3 = cv.imread('augo/scene1/3.jpg')
img3 = resize_image(img3)
cv.namedWindow('img3')
cv.imshow('img3', img3)
cv.waitKey(0)
kp3 = sift.detect(img3)
desc3 = sift.compute(img3, kp3)

img4 = cv.imread('augo/scene1/4.jpg')
img4 = resize_image(img4)
cv.namedWindow('img4')
cv.imshow('img4', img4)
cv.waitKey(0)
kp4 = sift.detect(img4)
desc4 = sift.compute(img4, kp4)

matches34 = match_features(desc3[1], desc4[1])

panorama34 = create_panorama(img3, img4, kp3, kp4, desc3, desc4, matches34)
cv.namedWindow('panorama34')
cv.imshow('panorama34', panorama34)
cv2.imwrite('results/augo/scene1/panorama34.jpg', panorama34)
kp34 = sift.detect(panorama34)
desc34 = sift.compute(panorama34, kp34)
cv.waitKey(0)

dimg34 = cv.drawMatches(img3, desc3[0], img4, desc4[0], matches34, None)
cv.namedWindow('main3')
cv.imshow('main3', dimg34)
cv2.imwrite('results/augo/scene1/dimg34.jpg', dimg34)

gray34 = cv2.cvtColor(panorama34, cv2.COLOR_BGR2GRAY)
_, thresh12 = cv2.threshold(gray34, 1, 255, cv2.THRESH_BINARY)
x, y, w, h = cv2.boundingRect(thresh12)
cropped_panorama34 = panorama34[y:y+h, x:x+w]

# ==============================  final  ==============================

matches_final = match_features(desc12[1], desc34[1])

final_dimg = cv.drawMatches(cropped_panorama12, desc12[0], panorama34, desc34[0], matches_final, None)
cv.namedWindow('final_dimg')
cv.imshow('final_dimg', final_dimg)
cv2.imwrite('results/augo/scene1/final_dimg.jpg', final_dimg)
cv.waitKey(0)

final_panorama = create_panorama(cropped_panorama12, panorama34, kp12, kp34, desc12, desc34, matches_final)
cv.namedWindow('final_panorama')
cv.imshow('final_panorama', final_panorama)
cv2.imwrite('results/augo/scene1/final_panorama.jpg', final_panorama)
cv.waitKey(0)

pass


