import cv2

def get_img_interest_point(img_path):
    img = cv2.imread(img_path)
    h, w, _ = img.shape

    # print("the h is : {}".format(str(h)))
    max_resolution = 960
    if (h > max_resolution):
        scale_factor = float(max_resolution) / float(h)
    else:
        scale_factor = 1.0
    img = cv2.resize(img, (int(w * scale_factor), int(h * scale_factor)),
                     interpolation=cv2.INTER_CUBIC)
    descriptor = cv2.AKAZE_create()
    # masks = np.zeros((h, w, 1), np.uint8)
    # masks[:h//3*2,:w//3*2]=255
    # kps = descriptor.detect(img,mask=masks)
    kps = descriptor.detect(img)
    vector_size = 2000
    kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
    # kps, des = descriptor.detectAndCompute(img, kps)
    kps, des = descriptor.compute(img, kps)

    return kps, des