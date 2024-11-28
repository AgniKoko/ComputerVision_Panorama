import numpy as np
import cv2 as cv

sift = cv.xfeatures2d_SIFT.create(400)
img1 = cv.imread('results/NISwGSP/final_panorama123.jpg')
# cv.namedWindow('main1')
# cv.imshow('main1', img1)
# cv.waitKey(0)
kp1 = sift.detect(img1)
desc1 = sift.compute(img1, kp1)

img2 = cv.imread('results/NISwGSP/final_panorama234.jpg')
# cv.namedWindow('main2')
# cv.imshow('main2', img2)
# cv.waitKey(0)
kp2 = sift.detect(img2)
desc2 = sift.compute(img2, kp2)

bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
matches = bf.match(desc1[1], desc2[1])

dimg = cv.drawMatches(img1, desc1[0], img2, desc2[0], matches, None)
cv.namedWindow('main3')
cv.imshow('main3', dimg)
cv.imwrite('results/NISwGSP/bf/final_dimg_bf.jpg', dimg)
cv.waitKey(0)

img_pt1 = []
img_pt2 = []
for x in matches:
    img_pt1.append(kp1[x.queryIdx].pt)
    img_pt2.append(kp2[x.trainIdx].pt)
img_pt1 = np.array(img_pt1)
img_pt2 = np.array(img_pt2)
# img_pt1 = np.array([kp1[x.queryIdx].pt for x in matches])
# img_pt2 = np.array([kp2[x.trainIdx].pt for x in matches])

M, mask = cv.findHomography(img_pt2, img_pt1, cv.RANSAC)
# Βρίσκει πώς πρέπει να μετατραπει η πρώτη για να "ταιριάξει" με τη δευτερη

img3 = cv.warpPerspective(img2, M, (img1.shape[1]+3000, img1.shape[0]+3000))
img3[0: img1.shape[0], 0: img1.shape[1]] = img1

print(f"Number of matches between final_panorama123 and final_panorama234: {len(matches)}")

cv.namedWindow('main')
cv.imshow('main', img3)
cv.imwrite('results/NISwGSP/bf/final_panorama_bf.jpg', img3)
cv.waitKey(0)

pass


