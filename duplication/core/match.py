import cv2
import numpy as np


from core.calculate_homo import find_homo
from core.calculate_iou import calculate_iou
from core.homo_diff import compute_homo_diff
from core.filter import calculate_homo_inliers

def match_point(img1_path, kp1, kp2, des1, des2, homo_diff, iou):
    bf = cv2.DescriptorMatcher_create("BruteForce")
    matches = bf.knnMatch(np.asarray(des1, np.float32), np.asarray(des2, np.float32), k=2)
    good = []

    empty_box = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])

    try:
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])

    except:
        return 0, 0, empty_box, 0

    src_pts = []
    dst_pts = []
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

    src_pts = np.array(src_pts)
    dst_pts = np.array(dst_pts)

    try:
        # sim_account = 0
        # H_0, Mask_0 = find_homo(src_pts, dst_pts)
        # res_src_pts, res_dst_pts, homo_inliers_1 = inliers_filter(Mask_0, src_pts, dst_pts)
        # sim_account += homo_inliers_1
        # try:
        #     H_1, Mask_1 = find_homo(res_src_pts, res_dst_pts)
        #     _, _, homo_inliers_res = inliers_filter(Mask_1, res_src_pts, res_dst_pts)
        # except:
        #     homo_inliers_res = 0
        # sim_account += homo_inliers_res
        sim_account = 0

        homo1, res_src_pts, res_dst_pts, inliers_count = calculate_homo_inliers(src_pts, dst_pts)
        sim_account += inliers_count

        homo2, _, _, res_inliers_count = calculate_homo_inliers(res_src_pts, res_dst_pts)
        sim_account += res_inliers_count
        
        sim_percent = float(sim_account) / float(len(good))

        # IOU between mask and image
        IOU = 0
        if iou == True:
            try:
                IOU = calculate_iou(img1_path, homo1)
            except:
                IOU = 0

        # compute homo diff
        l2t_homograph_measurement = 1000
        if homo_diff == True:
            try:
                l2t_homograph_measurement = compute_homo_diff(homo1, homo2)
            except:
                l2t_homograph_measurement = 1000

        return sim_account, IOU, sim_percent, l2t_homograph_measurement

    except Exception as e:
        print('homo compute error')
        return 0, 0, 0, 0
