import numpy as np
import cv2

def inliers_filter(mask, src_pts, dst_pts):
    inliers_count = 0
    res_src_pts = []
    res_dst_pts = []

    for index, each_element in enumerate(mask):
        if each_element == [1]:
            inliers_count += 1
        else:
            res_src_pts.append(src_pts[index])
            res_dst_pts.append(dst_pts[index])

    res_src_pts = np.array(res_src_pts)
    res_dst_pts = np.array(res_dst_pts)

    return res_src_pts, res_dst_pts, inliers_count

def calculate_homo_inliers(src_pts, dst_pts, filter_mode=cv2.RANSAC, reprojThresh=5):
    inliers_count = 0
    res_src_pts = []
    res_dst_pts = []
    try:
        homo, mask = cv2.findHomography(src_pts, dst_pts, filter_mode, reprojThresh)
        for index, each_element in enumerate(mask):
            if each_element == [1]:
                inliers_count += 1
            else:
                res_src_pts.append(src_pts[index])
                res_dst_pts.append(dst_pts[index])
    except:
        return np.array([]), np.array([]), 0

    res_src_pts = np.array(res_src_pts)
    res_dst_pts = np.array(res_dst_pts)

    return homo, res_src_pts, res_dst_pts, inliers_count