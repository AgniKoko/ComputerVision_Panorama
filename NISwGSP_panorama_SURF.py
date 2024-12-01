import cv2
import numpy as np
import cv2 as cv

surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)

def resize_image(img, width=1000):
    h, w = img.shape[:2]
    aspect_ratio = h / w
    new_height = int(width * aspect_ratio)
    return cv2.resize(img, (width, new_height))

# ==============================  12  ==============================

img1 = cv.imread('img/NISwGSP/01.jpg')
img1 = resize_image(img1)
cv.namedWindow('img1')
cv.imshow('img1', img1)
cv.waitKey(0)
kp1 = surf.detect(img1)
desc1 = surf.compute(img1, kp1)

img2 = cv.imread('img/NISwGSP/02.jpg')
img2 = resize_image(img2)
cv.namedWindow('img2')
cv.imshow('img2', img2)
cv.waitKey(0)
kp2 = surf.detect(img2)
desc2 = surf.compute(img2, kp2)

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

matches12 = match_features(desc1[1], desc2[1], method='match1')
matches12 = match_features(desc1[1], desc2[1], method='match2')

dimg12 = cv.drawMatches(img1, desc1[0], img2, desc2[0], matches12, None)
cv.namedWindow('dimg12')
cv.imshow('dimg12', dimg12)
cv2.imwrite('results/NISwGSP/SURF/dimg12.jpg', dimg12)
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

panorama12 = create_panorama(img1, img2, kp1, kp2, desc1, desc2, matches12)
cv.namedWindow('panorama12')
cv.imshow('panorama12', panorama12)
cv2.imwrite('results/NISwGSP/SURF/panorama12.jpg', panorama12)
kp12 = surf.detect(panorama12)
desc12 = surf.compute(panorama12, kp12)
cv.waitKey(0)

gray12 = cv2.cvtColor(panorama12, cv2.COLOR_BGR2GRAY)
_, thresh12 = cv2.threshold(gray12, 1, 255, cv2.THRESH_BINARY)
x, y, w, h = cv2.boundingRect(thresh12)
cropped_panorama12 = panorama12[y:y+h, x:x+w]

# ==============================  34  ==============================

img3 = cv.imread('img/NISwGSP/03.jpg')
img3 = resize_image(img3)
cv.namedWindow('img3')
cv.imshow('img3', img3)
cv.waitKey(0)
kp3 = surf.detect(img3)
desc3 = surf.compute(img3, kp3)

img4 = cv.imread('img/NISwGSP/04.jpg')
img4 = resize_image(img4)
cv.namedWindow('img4')
cv.imshow('img4', img4)
cv.waitKey(0)
kp4 = surf.detect(img4)
desc4 = surf.compute(img4, kp4)

matches34 = match_features(desc3[1], desc4[1], method='match1')
matches34 = match_features(desc3[1], desc4[1], method='match2')

dimg34 = cv.drawMatches(img3, desc3[0], img4, desc4[0], matches34, None)
cv.namedWindow('main3')
cv.imshow('main3', dimg34)
cv2.imwrite('results/NISwGSP/SURF/dimg34.jpg', dimg34)

panorama34 = create_panorama(img3, img4, kp3, kp4, desc3, desc4, matches34)
cv.namedWindow('panorama34')
cv.imshow('panorama34', panorama34)
cv2.imwrite('results/NISwGSP/SURF/panorama34.jpg', panorama34)
kp34 = surf.detect(panorama34)
desc34 = surf.compute(panorama34, kp34)
cv.waitKey(0)

gray34 = cv2.cvtColor(panorama34, cv2.COLOR_BGR2GRAY)
_, thresh12 = cv2.threshold(gray34, 1, 255, cv2.THRESH_BINARY)
x, y, w, h = cv2.boundingRect(thresh12)
cropped_panorama34 = panorama34[y:y+h, x:x+w]

# ==============================  final  ==============================

matches_final = match_features(desc12[1], desc34[1], method='match1')
matches_final = match_features(desc12[1], desc34[1], method='match2')
final_dimg = cv.drawMatches(cropped_panorama12, desc12[0], panorama34, desc34[0], matches_final, None)
cv.namedWindow('final_dimg')
cv.imshow('final_dimg', final_dimg)
cv2.imwrite('results/NISwGSP/SURF/final_dimg.jpg', final_dimg)
cv.waitKey(0)

final_panorama = create_panorama(cropped_panorama12, panorama34, kp12, kp34, desc12, desc34, matches_final)
cv.namedWindow('final_panorama')
cv.imshow('final_panorama', final_panorama)
cv2.imwrite('results/NISwGSP/SURF/final_panorama.jpg', final_panorama)
cv.waitKey(0)

pass


