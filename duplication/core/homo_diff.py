import numpy as np
import cv2

def compute_homo_diff(homograph_1, homograph_2):
    pts = comp_pts = [[100, 100], [50, 100], [100, 50], [50, 50]]
    src_pts = np.float32(pts).reshape(-1, 1, 2)

    src_pts_1 = cv2.perspectiveTransform(src_pts, homograph_1)
    src_pts_2 = cv2.perspectiveTransform(src_pts, homograph_2)

    ttr = np.squeeze(src_pts_1)
    ggt = np.squeeze(src_pts_2)
    diff = ttr - ggt

    total_cdist = 0

    for each_point in diff:
        x = each_point[0]
        y = each_point[1]
        dist = x**2 + y **2
        total_cdist += dist ** 0.5

    diff = total_cdist / len(diff)
    return diff