import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

def get_kps_des(img_path):

    img = cv2.imread(img_path)
    img_h, img_w, _ = img.shape

    max_resolution = 960
    if (img_h > max_resolution):
        scale_factor = float(max_resolution) / float(img_h)
    else:
        scale_factor = 1.0
    img = cv2.resize(img, (int(img_w * scale_factor), int(img_h * scale_factor)),
                     interpolation=cv2.INTER_CUBIC)

    descriptor = cv2.AKAZE_create()
    # descriptor = cv2.SIFT_create()

    kps = descriptor.detect(img)
    # vector_size = 1000
    # kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
    # kps, des = descriptor.detectAndCompute(img, kps)
    kps, des = descriptor.compute(img, kps)

    return img, kps, des, img_h, img_w

def compare_imgs(img1_path, img2_path, img1_w, img2_w, kp1, des1, kp2, des2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  
    # matches = bf.knnMatch(np.asarray(des1, np.float32), np.asarray(des2, np.float32), k=2)
    matches = bf.knnMatch(des1, des2, k=1)

    good = []
    for match in matches:
        if len(match) == 1:
            good.append(match)

    good = sorted(good, key=lambda x: x[0].distance)
    good = good[:40]

    src_pts = []
    dst_pts = []

    reprojThresh = 5

    for each_point in good:
        source_point = kp1[each_point[0].queryIdx].pt
        source_point_x = source_point[0]
        source_point_y = source_point[1]
        source_point = (source_point_x, source_point_y)
        src_pts.append(source_point)

        target_point = kp2[each_point[0].trainIdx].pt
        target_point_x = target_point[0]
        target_point_y = target_point[1]
        target_point = (target_point_x, target_point_y)
        dst_pts.append(target_point)

    H = cv2.findHomography(np.array(src_pts), np.array(dst_pts), cv2.RHO, reprojThresh)

    sim_account = 0

    for index, each_element in enumerate(H[1]):
        if each_element == [1]:
            sim_account = sim_account + 1

    sim_percent = float(sim_account) / float(len(H[1]))

    total_src_x = 0
    for x, y in src_pts:
        total_src_x += x

    total_dst_x = 0
    for x, y in dst_pts:
        total_dst_x += x

    mean_src_x = total_src_x / len(src_pts)
    middle_src_w = img1_w / 2

    if sim_percent > 0.5:
        if mean_src_x > middle_src_w:
            return 'Left'
        elif mean_src_x < middle_src_w:
            return 'Right'
    else:
        return 'UnConnect'

def sim_img_insertion_sort(images_list, little_pussy):
    """
        get the spatial sort image sequence.
    Args:
        images_list(list): List of random image sequence
        little_pussy(dict): Kps and Des info of images_list
    Returns:
        images_list(list): List of img sequence, after spatial sorting from left to right
    """

    length = len(images_list)
    split_index = [0]
    repp = []
    for i in range(length - 1):
        for j in range(i + 1, length):
            img1 = images_list[i]
            img2 = images_list[j]
            if (img1, img2) not in repp:
                img1_w = little_pussy[img1]['img_w']
                img2_w = little_pussy[img2]['img_w']
                kps1 = little_pussy[img1]['kps']
                des1 = little_pussy[img1]['des']
                kps2 = little_pussy[img2]['kps']
                des2 = little_pussy[img2]['des']

                compare_result = compare_imgs(img1, img2, img1_w, img2_w, kps1, des1, kps2, des2)
                # print('img1, img2', img1, img2, compare_result)
                if compare_result == 'Right':
                    repp.append((img1, img2))
                    repp.append((img2, img1))
                    if j == i + 1:
                        images_list[i], images_list[j] = images_list[j], images_list[i]
                    else:
                        images_list[i], images_list[i + 1], images_list[j] = images_list[j], images_list[i], images_list[
                            i + 1]
                    break
                elif compare_result == 'Left':
                    repp.append((img1, img2))
                    repp.append((img2, img1))
                    if j > i + 1:
                        images_list[i + 1], images_list[j] = images_list[j], images_list[i + 1]
                    break
                elif compare_result == 'UnConnect':
                    repp.append((img1, img2))
                    repp.append((img2, img1))

    return images_list, split_index
