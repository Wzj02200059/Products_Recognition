import cv2

# TODO
# Add more homo calculate mode
def find_homo(src_pts, dst_pts, filter_mode = cv2.RANSAC, reprojThresh = 5):

    homo, mask = cv2.findHomography(src_pts, dst_pts, filter_mode, reprojThresh)

    return homo, mask

