import cv2
import numpy as np

from core.point import calculatecorners

def calculate_iou(img1_path, homograph):
    im = cv2.imread(img1_path)
    orig_h, orig_w, _ = im.shape

    max_resolution = 960
    if (orig_h > max_resolution):
        scale_factor = float(max_resolution) / float(orig_h)
    else:
        scale_factor = 1.0

    w, h = int(orig_w * scale_factor), int(orig_h * scale_factor)
    img1 = cv2.resize(im, (w, h),
                     interpolation=cv2.INTER_CUBIC)

    corner = calculatecorners(img1, homograph)
    ltop_x = int(corner.ltop.x)
    ltop_y = int(corner.ltop.y)
    lbottom_x = int(corner.lbottom.x)
    lbottom_y = int(corner.lbottom.y)
    rtop_x = int(corner.rtop.x)
    rtop_y = int(corner.rtop.y)
    rbottom_x = int(corner.rbottom.x)
    rbottom_y = int(corner.rbottom.y)

    # determine if the contor is OK
    top_w = rtop_x - ltop_x
    buttom_w = rbottom_x - lbottom_x
    # if not 0.2<float(top_w)/float(buttom_w)<1.8:
    #     return 0, 0, empty_box, 0
    #
    # left_h = lbottom_y - ltop_y
    # right_h = rbottom_y - rtop_y
    # if not 0.2 < float(left_h) / float(right_h) < 1.8:
    #     return 0, 0, empty_box, 0

    color = (1, 0, 0)
    color2 = (0, 1, 0)

    img = np.zeros([h, w, 2])
    triangle1 = np.array([[ltop_x, ltop_y], [lbottom_x, lbottom_y], [rbottom_x, rbottom_y],
                          [rtop_x, rtop_y]])
    cv2.fillConvexPoly(img, triangle1, color)
    area1 = img.sum()
    img = np.zeros([h, w, 3])
    triangle2 = np.array([[0, 0], [0, h], [w, h], [w, 0]])
    cv2.fillConvexPoly(img, triangle2, color2)
    area2 = img.sum()

    # if area2 > 8 * area1:
    #     return 0

    cv2.fillConvexPoly(img, triangle1, color)
    union_area = img.sum() + area2

    inter_area = area1 + area2
    IOU = inter_area / union_area

    return IOU