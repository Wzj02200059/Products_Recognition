import os
import cv2
import random
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='split image to train and test in triplet type')
parser.add_argument("--folder_path", nargs="+",
                    help='origin data dir, it contains all images grouped by category')
parser.add_argument("--target_dir", type=str, default='',
                    help='target dir to save the generated train.txt test.txt and synset.txt')
parser.add_argument("--test_partion", type=float, default=0.1,
                    help='how many images will splitted to test, defaults is 0.05')
args = parser.parse_args()


IMG_SUFFIX_LIST = ['.jpg', '.JPG', '.png', '.PNG', '.jpeg', '.JPEG']

if not os.path.exists(args.target_dir):
    os.makedirs(args.target_dir)

f_train = open(os.path.join(args.target_dir, 'train.txt'),
               'w', encoding='utf-8')
f_test = open(os.path.join(args.target_dir, 'test.txt'),
              'w', encoding='utf-8')
f_synset = open(os.path.join(args.target_dir, 'synset.txt'),
                'w', encoding='utf-8')

total_sample_list = []
labels_list = []
labels = []
for data_dir in args.folder_path:
    print('reading images from {}'.format(data_dir))
    files = os.listdir(data_dir)
    files.sort()
    for cls in files:
        cls_folder = os.path.join(data_dir, cls)
        if os.path.isdir(cls_folder):
            if cls not in labels:
                labels.append(cls)
                f_synset.write(cls + '\n')
        for im in os.listdir(cls_folder):
            base_name, suffix = os.path.splitext(im)
            im_path = os.path.join(cls_folder, im)
            if suffix not in IMG_SUFFIX_LIST:
                continue
            total_sample_list.append(im_path)
            labels_list.append(cls)

# labels分组、label:该label对应的index
label_to_indices = {label: np.where(np.asarray(labels_list) == label)[0] for label in labels}

for index in range(len(total_sample_list)):
    anchor_im, anchor_label = total_sample_list[index], labels_list[index]

    positive_index = index

    while positive_index == index:
        positive_index = np.random.choice(label_to_indices[anchor_label])

    negative_label = np.random.choice(list(set(labels) - set([anchor_label])))
    negative_index = np.random.choice(label_to_indices[negative_label])

    positive_im = total_sample_list[positive_index]
    negative_im = total_sample_list[negative_index]

    if random.random() > args.test_partion:
        f_train.write(total_sample_list[index] + ', ' + positive_im+', ' + negative_im + ', ' + anchor_label + '\n')
    else:
        f_test.write(total_sample_list[index] + ', ' + positive_im+', ' + negative_im + ', ' + anchor_label + '\n')

f_train.close()
f_test.close()
f_synset.close()
print('DataSet Prepare Is Done!')
