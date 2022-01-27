# ---------------------------------------------
# File        : deep_duplication.py
# Description : search similiar images
# Author      : Wang Zhen Jie
# Version     : 0.3
# ---------------------------------------------
import os
import argparse

from core.extractor import Detector
from core.deep_match import Matching

"""
TODO:
boxes_embedding: Using high-level feature to find dup img
portrait: find the img with same sku|posm in differnt background
"""

def get_args():
    parser = argparse.ArgumentParser("Duplicate")
    parser.add_argument('--image_dir', type=str, help='image path')
    parser.add_argument('--homo_diff', type=bool, default=False ,help='calculate the diff between homos from two planes')
    parser.add_argument('--iou', type=bool, default=False, help='add iou threshold to determine duplicate')
    parser.add_argument('--det_model_name', type=str, default='', help='choce which det model')
    parser.add_argument('--det_model', type=str, default='', help='choce det model’s checkpoint ')
    parser.add_argument('--match_model_config', type=str, default='', help='choce match model’s config ')
    parser.add_argument('--portrait', type=bool, default=False, help='recognition portrait duplicate')
    parser.add_argument('--gpu', type=bool, default=False, help='use gpu upspeed')
    # TODO
    # Add more params mode
    args = parser.parse_args()
    return args

def main(args):
    images = os.listdir(args.image_dir)
    img_list = []
    bbox_details = []
    duplicate_img_path_list = []
    duplicate_confidence = []
    length = len(images)

    matching = Matching.eval().to(device)
    detecting = Detector.eval().to(device)
    # Step 1 : 提取每一张图片的特征
    for i in images:
        im_path = os.path.join(args.image_dir, i)
        img_list.append(im_path)
        det_result = detecting(im_path)
        bbox_details.append(det_result)

    # Step 2 : 两两比对
    for i in range(length - 1):
        for j in range(i + 1, length):
            data_0 = {}
            data_1 = {}
            data_0['keypoints0'] = bbox_details[i]['bbox_center']
            data_1['keypoints1'] = bbox_details[j]['bbox_center']
            data_0['descriptors0'] = bbox_details[i]['embeding']
            data_1['descriptors1'] = bbox_details[j]['embeding']
            data_0['scores0'] = bbox_details[i]['score']
            data_1['scores1'] = bbox_details[j]['score']
            data_0['image0'] = img_list[i]
            data_1['image1'] = img_list[j]

            pred = matching(data_0, data_1)
            # match_points, match_account, sim_percent, l2t_homograph_measurement = match_box(
            #     img1_real_path, each_kps, each_kps2, each_des, each_des2, each_scores, each_scores2, args.homo_diff, args.iou)
            
            if pred['sim_percent'] > 0.5:
                duplicate_img_path_list.append(data['image1'])
                duplicate_confidence.append(pred['sim_percent'])
                continue

    return duplicate_img_path_list, duplicate_confidence


if __name__ == '__main__':
    args = get_args()
    duplicate_img_path_list, duplicate_confidence = main(args)
    print(duplicate_img_path_list, duplicate_confidence)


