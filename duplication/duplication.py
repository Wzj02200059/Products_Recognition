# ---------------------------------------------
# File        : duplication.py
# Description : search similiar images
# Author      : Wang Zhen Jie
# Version     : 0.4
# ---------------------------------------------
import os
import argparse

from core.extractor import get_img_interest_point
from core.match import match_point


def get_args():
    parser = argparse.ArgumentParser("Duplicate")
    parser.add_argument('--image_dir', type=str, help='image path')
    parser.add_argument('--homo_diff', type=bool, default=False ,help='calculate the diff between homos from two planes')
    parser.add_argument('--iou', type=bool, default=False, help='add iou threshold to determine duplicate')
    parser.add_argument('--gpu', type=bool, default=False, help='use gpu upspeed')
    # TODO
    # Add more params mode
    args = parser.parse_args()
    return args

def main(args):
    images = os.listdir(args.image_dir)
    des_list = []
    kps_list = []
    img_list = []
    duplicate_img_path_list = []
    duplicate_confidence = []
    length = len(images)

    # Step 1 : 提取每一张图片的特征
    for i in images:
        im_path = os.path.join(args.image_dir, i)
        kps, des = get_img_interest_point(im_path)
        des_list.append(des)
        kps_list.append(kps)
        img_list.append(im_path)

    # Step 2 : 两两比对
    for i in range(length - 1):
        for j in range(i + 1, length):
            each_kps = kps_list[i]
            each_kps2 = kps_list[j]
            each_des = des_list[i]
            each_des2 = des_list[j]
            img1_real_path = img_list[i]
            each_path2 = img_list[j]
            
            match_points, match_account, sim_percent, l2t_homograph_measurement = match_point(
                img1_real_path, each_kps, each_kps2, each_des, each_des2, args.homo_diff, args.iou)

            if sim_percent > 0.5:
                duplicate_img_path_list.append(each_path2)
                duplicate_confidence.append(sim_percent)
                continue

    return duplicate_img_path_list, duplicate_confidence


if __name__ == '__main__':
    args = get_args()
    duplicate_img_path_list, duplicate_confidence = main(args)
    print(duplicate_img_path_list, duplicate_confidence)


