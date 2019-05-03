from modules.const import *
from modules.utility import show

import cv2
import numpy as np

import os
from glob import glob

IMAGE_MAX = 1000
MIN_FEATURE_SIZE = IMAGE_MAX // 12
CUTOUT_RATIO = 2 / 5
PROCESSED_CORNER_SIZE = 100

def is_convex(polygon):
    num_sides = len(polygon)
    products = []
    for i in range(num_sides):
        p0 = polygon[i]
        p1 = polygon[(i+1) % num_sides]
        p2 = polygon[(i+2) % num_sides]

        v1 = p1 - p0
        v2 = p2 - p1

        products.append(np.cross(v1, v2))

    return all([p < 0 for p in products]) or all([p > 0 for p in products])

        
def sharpen(image, kernel_size):
    blurred = cv2.blur(image, (kernel_size, kernel_size))
    return cv2.addWeighted(image, 2.0, blurred, -2.0, 0)    

def norm_rotation(image):
    if image.shape[0] > image.shape[1]:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image


def detect_cards(image):          
    sharpened = sharpen(image, 259)
    grey = cv2.cvtColor(sharpened, cv2.COLOR_RGB2GRAY)
    
    shape = grey.shape
    ratio = IMAGE_MAX / (shape[0] if shape[0] > shape[1] else shape[1])
    resized = cv2.resize(grey, (0, 0), fx=ratio, fy=ratio)
    
    blurred = cv2.GaussianBlur(resized, (19, 19), 0)     
    edges = cv2.Canny(blurred, 25, 65)

    dilation_kernel = np.ones((9,9),np.uint8)
    dilated = cv2.dilate(edges, dilation_kernel)
    erosion_kernel = np.ones((7,7),np.uint8)
    edges_closed = cv2.erode(dilated, erosion_kernel)
                
    # find contours in the thresholded image and initialize the
    # shape detector
    cnts = cv2.findContours(edges_closed.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)    
    cnts = cnts[0]

    cards = []    
    
    for cnt in cnts:
        epsilon = 0.08 * cv2.arcLength(cnt, True)
        card_rectangle = cv2.approxPolyDP(cnt, epsilon, True)    
        
        area = cv2.contourArea(card_rectangle)

        rect_scaled = card_rectangle.astype("float")
        rect_scaled /= ratio
        rect_scaled = rect_scaled.astype("int")
        
        if area < MIN_FEATURE_SIZE**2:
            continue
        if len(card_rectangle) != 4:
            continue    
        if not is_convex(card_rectangle):
            continue
        
        v1_norm = np.linalg.norm(card_rectangle[0] - card_rectangle[3])
        v2_norm = np.linalg.norm(card_rectangle[0] - card_rectangle[1])
        if v1_norm > v2_norm:
            v1_norm, v2_norm = v2_norm, v1_norm
        rect_ratio = v1_norm / v2_norm        
        if rect_ratio < 0.4 or rect_ratio > 1.0:
            continue

        cards.append(rect_scaled)        
    
    return cards

def cut_corners(image, card, corner_num = None, cutout_ratio=1/4):
    if corner_num is None:
        corner_num = 4
    corner_indexes = []
    
    corner_indexes = [0, 1, 2, 3]    
    if corner_num == 2:
        measure_func = lambda x: card[x][0][0] + card[x][0][1]
        topleft = min(corner_indexes, key=measure_func)
        botright = max(corner_indexes, key=measure_func)
        corner_indexes = [topleft, botright]

    corners = []
    
    for i in corner_indexes:
        middle_corner = card[i][0]
        previous_corner = card[(i - 1) % 4][0]
        next_corner = card[(i + 1) % 4][0]

        v1 = previous_corner - middle_corner
        v2 = next_corner - middle_corner

        if np.linalg.norm(v1) > np.linalg.norm(v2):
            v1, v2 = v2, v1

        v1_cut = (v1 * cutout_ratio)
        v2_cut = (v2 * cutout_ratio)

        cutout_previous = middle_corner + v1_cut
        cutout_next = middle_corner + v2_cut
        cutout_opposite = middle_corner + v1_cut + v2_cut

        corner_rect = np.array([
            [cutout_previous],
            [middle_corner],
            [cutout_next],
            [cutout_opposite]
        ], dtype=np.int)

        x,y,w,h = cv2.boundingRect(corner_rect)

        # cut into a square
        if w > h:
            h = w
        else:
            w = h

        corner_image = image[y:y+h, x:x+w]

        corners.append(corner_image)

    return corners

def for_classification(filename):
    image = cv2.imread(filename)
    if image is None:
        raise ValueError("Cannot read file")
    image = norm_rotation(image)

    cards = detect_cards(image)  
    extracted_corners = []

    for card in cards:
        corner_images = cut_corners(image, card, NUM_CORNERS) 
        resized = [cv2.resize(c, (PROCESSED_CORNER_SIZE, PROCESSED_CORNER_SIZE)) for c in corner_images]
        extracted_corners.append(resized)

    for card in cards:
        cv2.drawContours(image, [card], -1, (0, 255, 0), 20)
 
    return { 'image': image, 'corners': extracted_corners }

def test_detection(folder):
    import random
    paths = glob("{}\\*.jpg".format(folder))
    paths = sorted(paths, key=lambda x: random.random())
    
    for path in paths:
        image = cv2.imread(path)
        image = norm_rotation(image)
        card_outlines = detect_cards(image)             
        for outline in card_outlines:
            cv2.drawContours(image, [outline], -1, (0, 255, 0), 15)
    
    extracted_corners = []
    for card in card_outlines:
        corner_images = cut_corners(image, card, 2) 
        resized = [cv2.resize(c, (PROCESSED_CORNER_SIZE, PROCESSED_CORNER_SIZE)) for c in corner_images]
        extracted_corners.append(resized)
    
    for group in extracted_corners:
        for corner in group:
            show(corner, 4)
    