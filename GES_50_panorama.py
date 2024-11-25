import cv2
import numpy as np
import cv2 as cv

sift = cv.xfeatures2d_SIFT.create(400)
cv.namedWindow('main1')

def resize_image(img, width=500):
    h, w = img.shape[:2]
    aspect_ratio = h / w
    new_height = int(width * aspect_ratio)
    return cv2.resize(img, (width, new_height))

img1 = cv.imread('img/GES-50/0.jpg')
img1 = resize_image(img1)
cv.namedWindow('main1')
cv.imshow('main1', img1)
cv.waitKey(0)
kp1 = sift.detect(img1)
desc1 = sift.compute(img1, kp1)

img2 = cv.imread('img/GES-50/1.jpg')
img2 = resize_image(img2)
cv.namedWindow('main2')
cv.imshow('main2', img2)
cv.waitKey(0)
kp2 = sift.detect(img2)
desc2 = sift.compute(img2, kp2)

img3 = cv.imread('img/GES-50/2.jpg')
img3 = resize_image(img3)
cv.namedWindow('main2')
cv.imshow('main2', img3)
cv.waitKey(0)
kp3 = sift.detect(img3)
desc3 = sift.compute(img3, kp3)

img4 = cv.imread('img/GES-50/3.jpg')
img4 = resize_image(img4)
cv.namedWindow('main2')
cv.imshow('main2', img4)
cv.waitKey(0)
kp4 = sift.detect(img4)
desc4 = sift.compute(img4, kp4)

def match_features(d1, d2, method='match1'):
    n1 = d1.shape[0]
    n2 = d2.shape[0]

    matches = []
    if method == 'match1':
        for i in range(n1):
            fv = d1[i, :]
            diff = d2 - fv
            diff = np.abs(diff)
            distances = np.sum(diff, axis=1)

            i2 = np.argmin(distances)
            mindist2 = distances[i2]

            matches.append(cv.DMatch(i, i2, mindist2))
    elif method == 'match2':
        for i in range(n1):
            fv = d1[i, :]
            diff = d2 - fv
            diff = np.abs(diff)
            distances = np.sum(diff, axis=1)

            i2 = np.argmin(distances)
            mindist2 = distances[i2]

            distances[i2] = np.inf

            i3 = np.argmin(distances)
            mindist3 = distances[i3]

            if mindist2 / mindist3 < 0.5:
                matches.append(cv.DMatch(i, i2, mindist2))

    return matches

matches = match_features(desc1[1], desc2[1], method='match1')
matches = match_features(desc1[1], desc2[1], method='match2')

dimg = cv.drawMatches(img1, desc1[0], img2, desc2[0], matches, None)
cv.namedWindow('main3')
cv.imshow('main3', dimg)
cv2.imwrite('results/GES-50/dimg.jpg', dimg)
cv.waitKey(0)

img_pt1 = []
img_pt2 = []

def create_panorama(img1, img2, kp1, kp2, desc1, desc2, matches):
    img_pt1 = np.array([kp1[x.queryIdx].pt for x in matches])
    img_pt2 = np.array([kp2[x.trainIdx].pt for x in matches])

    M, mask = cv.findHomography(img_pt2, img_pt1, cv.RANSAC)

    panorama = cv.warpPerspective(img2, M, (img1.shape[1] + 1000, img1.shape[0] + 1000))
    panorama[0:img1.shape[0], 0:img1.shape[1]] = img1

    return panorama

panorama12 = create_panorama(img1, img2, kp1, kp2, desc1, desc2, matches)
cv.namedWindow('panorama12')
cv.imshow('panorama12', panorama12)
cv2.imwrite('results/GES-50/panorama12.jpg', panorama12)
cv.waitKey(0)

matches = match_features(desc3[1], desc4[1], method='match1')
matches = match_features(desc3[1], desc4[1], method='match2')
panorama34 = create_panorama(img3, img4, kp3, kp4, desc3, desc4, matches)
cv.namedWindow('panorama34')
cv.imshow('panorama34', panorama34)
cv2.imwrite('results/GES-50/panorama34.jpg', panorama34)
cv.waitKey(0)

pass


