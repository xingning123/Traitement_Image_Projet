import cv2
import matplotlib.pyplot as plt
from skimage import morphology
import numpy as np
import math


def hist(img):
    img_ravel = img.ravel()
    hist = [0 for i in range(256)]
    for p in img_ravel:
        hist[int(p)] += 1
    for i in range(256):
        hist[i] /= img_ravel.shape[0]

    return hist


def binary(img, t):
    index = 0
    flag = False
    hist_sum = 0
    img_ravel = img.ravel()
    hist = [0 for i in range(256)]
    for p in img_ravel:
        hist[int(p)] += 1
    for i in range(256):
        hist[i] /= img_ravel.shape[0]
        hist_sum += hist[i]
        if not flag:
            if hist_sum >= t:
                index = i
                flag = True
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if img[x][y] <= index:
                img[x][y] = 0
            else:
                img[x][y] = 255
    return img


def thinning(binary):
    binary = cv2.bitwise_not(binary)
    binary[binary == 255] = 1
    skeleton = morphology.skeletonize(binary)
    skeleton = skeleton.astype(np.uint8) * 255

    return skeleton


def compute_crossing_number(values):
    return np.count_nonzero(values < np.roll(values, -1))


# Create a filter that converts any 8-neighborhood into the corresponding byte value [0,255]
cn_filter = np.array([[1, 2, 4],
                      [128, 0, 8],
                      [64, 32, 16]
                      ])

def CrosssingNumber(skeleton):

    # Create a lookup table that maps each byte value to the corresponding crossing number
    all_8_neighborhoods = [np.array([int(d) for d in f'{x:08b}'])[::-1] for x in range(256)]
    cn_lut = np.array([compute_crossing_number(x) for x in all_8_neighborhoods]).astype(np.uint8)


    # Skeleton: from 0/255 to 0/1 values
    skeleton01 = np.where(skeleton!=0, 1, 0).astype(np.uint8)
    # Apply the filter to encode the 8-neighborhood of each pixel into a byte [0,255]
    cn_values = cv2.filter2D(skeleton01, -1, cn_filter, borderType = cv2.BORDER_CONSTANT)
    # Apply the lookup table to obtain the crossing number of each pixel
    cn = cv2.LUT(cn_values, cn_lut)
    # Keep only crossing numbers on the skeleton
    cn[skeleton==0] = 0
    # crossing number == 1 --> Termination, crossing number == 3 --> Bifurcation
    minutiae = [(x, y, cn[y, x] == 1) for y, x in zip(*np.where(np.isin(cn, [1, 3])))]



    return minutiae


# Utility function to draw a set of minutiae over an image
def draw_minutiae(fingerprint, minutiae, termination_color=(0, 0, 255), bifurcation_color=(255, 0, 0)):
    res = cv2.cvtColor(fingerprint, cv2.COLOR_GRAY2BGR)

    for x, y, t, *d in minutiae:
        color = termination_color if t else bifurcation_color
        if len(d) == 0:
            cv2.drawMarker(res, (x, y), color, cv2.MARKER_CROSS, 8)
        else:
            d = d[0]
            ox = int(round(math.cos(d) * 7))
            oy = int(round(math.sin(d) * 7))
            cv2.circle(res, (x, y), 3, color, 1, cv2.LINE_AA)
            cv2.line(res, (x, y), (x + ox, y - oy), color, 1, cv2.LINE_AA)
    return res

def carte_minuties(minutiae,feature_image_direction):
    termination_color = (0, 0, 255)
    bifurcation_color = (255, 0, 0)
    print("cara", feature_image_direction.shape)
    row, col, num = feature_image_direction.shape
    carte = np.ones((row, col, num), dtype=np.uint8)  # créer une gray image
    carte *= 255

    for x, y, t, *d in minutiae:
        color = termination_color if t else bifurcation_color
        if len(d) == 0:
            cv2.drawMarker(carte, (x, y), color, cv2.MARKER_CROSS, 8)
        else:
            d = d[0]
            ox = int(round(math.cos(d) * 7))
            oy = int(round(math.sin(d) * 7))
            cv2.circle(carte, (x, y), 3, color, 1, cv2.LINE_AA)
            cv2.line(carte, (x, y), (x + ox, y - oy), color, 1, cv2.LINE_AA)
    return carte



if __name__ == '__main__':
    image = cv2.imread('Empreinte1.bmp')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.subplot(321)
    plt.imshow(image)
    plt.subplot(322)
    plt.plot(hist(gray))
    plt.subplot(323)
    plt.imshow(binary(gray, 0.4), cmap='gray')
    plt.subplot(324)
    plt.imshow(thinning(binary(gray, 0.4)), cmap='gray')
    plt.subplot(325)
    plt.imshow(draw_minutiae(thinning(binary(gray, 0.4)),CrosssingNumber(thinning(binary(gray, 0.4)))))
    plt.subplot(326)
    plt.imshow(carte_minuties(CrosssingNumber(thinning(binary(gray, 0.4))),image))
    plt.tight_layout()
    plt.show()
