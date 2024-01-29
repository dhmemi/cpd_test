import cv2
import numpy as np
import test_cpd

def show_img(img):
    cv2.namedWindow("img", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("img", img)
    cv2.waitKey(0)

def extract_template_contour(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = cv2.resize(gray, None, fx = 0.3, fy = 0.3)
    _, binary = cv2.threshold(gray, 160, maxval=255, type=cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    draw_img = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(draw_img, [contours[0]], -1, [255], 2)
    # show_img(draw_img)
    return contours[0], binary.shape

def extract_target_contour(img_path, template):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.resize(img, None, fx = 0.3, fy = 0.3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    _, img = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY_INV)
    # img = cv2.GaussianBlur(img, (3, 3), 0)
    # img = cv2.Canny(img, 20, 200)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # draw_img = np.zeros(img.shape, dtype=np.uint8)
    # cv2.drawContours(draw_img, contours, -1, [255], 2)
    # show_img(draw_img)

    min_score = 0x7fffffff
    target_contour = None
    for contour in contours:
        similarity = abs(cv2.contourArea(template) - cv2.contourArea(contour))
        # similarity = cv2.matchShapes(template, contour, cv2.CONTOURS_MATCH_I3, 0)
        if similarity < min_score:
            min_score = similarity
            target_contour = contour

    # draw_img = np.zeros(img.shape, dtype=np.uint8)
    # cv2.drawContours(draw_img, [target_contour], -1, [255], 2)
    # show_img(draw_img)

    return target_contour



if __name__ == '__main__':
    template_contour, img_shape = extract_template_contour(f"cm01/{0}.png")
    for idx in range(20, 28):
        image_path = f"cm01/{idx}.png"
        target_contour = extract_target_contour(image_path, template_contour)
        x = template_contour.reshape(-1, 2)
        y = target_contour.reshape(-1, 2)
        y_out = test_cpd.cpd(x, y)

        x_img = test_cpd.draw_binary(x, img_shape)
        y_img = test_cpd.draw_binary(y_out, img_shape)
        diff = cv2.bitwise_xor(x_img, y_img)
        cv2.namedWindow("diff", cv2.WINDOW_FREERATIO)
        cv2.imshow("diff", test_cpd.erode_and_dilate(diff, 2))
        cv2.waitKey(0)
