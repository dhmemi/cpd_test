import cv2
import numpy as np

IMAGE_DIR= "img/"
DATA_DIR= "data/"

def extract_contours(img_path, point_txt):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 10, maxval=255, type=cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    # draw_img = np.zeros(img.shape, np.uint8)
    # cv2.drawContours(draw_img, contours, -1, [255])
    # cv2.imshow('draw', draw_img)
    # cv2.waitKey(0)
    # return
    with open(point_txt, 'w') as f:
        f.writelines([f'{pt[0][0]} {pt[0][1]}\n' for pt in contours[0]])

def rotate_img(img_path, to_file):
    img = cv2.imread(img_path)
    m = cv2.getRotationMatrix2D([196, 181.5], 5, 0.9)
    out = cv2.warpAffine(img, m, img.shape[:2])
    cv2.imshow('draw', out)
    cv2.waitKey(0)
    cv2.imwrite(to_file, out)


if __name__ == '__main__':
    file_name = "cpd-source-3"
    to_name = "cpd-source-4"
    rotate_img(IMAGE_DIR +  file_name + ".png", IMAGE_DIR + to_name + ".png")
    extract_contours(IMAGE_DIR + to_name + ".png", DATA_DIR + to_name + ".txt")
    extract_contours(IMAGE_DIR + file_name + ".png", DATA_DIR + file_name + ".txt")