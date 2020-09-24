import cv2
import numpy as np


def crop_images(im):
    s = np.sum(im, axis=1)
    # max_s = s.max()
    max_s = np.max(s)

    black = s[s < max_s]
    black_min = black[0]
    black_max = black[-1]

    first = np.where(s == black_min)[0][0]
    last = np.where(s == black_max)[0][0]

    return im[max(first - 30, 0):last + 30, 0:im.shape[1]], 0


# ======================================================================================================================


def segment_lines(width, height, avg, lines):
    segments = []
    cr = br = 0

    while cr < height - 5:
        segment = np.average(lines[cr: cr + 10])

        if segment > avg:
            segments.append((br, cr, "blank"))

            lr = cr
            while lr < height - 5:
                lr += 1
                segment = np.average(lines[lr: lr + 10])
                if segment < avg:
                    break

            segments.append((cr, lr, "full"))
            cr = br = lr + 1

        cr += 1

    segments.append((br, cr, "blank"))
    return segments


def refine_lines(segments, im):
    if len(segments) == 3:
        return [(segments[0][0], segments[2][1])]

    lines = []
    after = None

    for c, segment in enumerate(segments):
        if segment[2] == "blank":
            continue

        before = segments[c - 1]
        if c + 1 < len(segments):
            after = segments[c + 1]

        lines.append((((before[1] + before[0]) // 2), ((after[1] + after[0]) // 2)))

    return lines


def split_lines(im):
    line_images = []

    hist = horizontal_histogram(im)
    avg = np.average(hist[hist > 0])

    height, width = im.shape
    lines = refine_lines(segment_lines(width, height, avg, hist), im)

    for line in lines:
        im_c = im[line[0]:line[1], 0:im.shape[1]]
        line_images.append(im_c)

    return line_images, 0


def split_lines_dummy(im):
    return [im], 0


# ======================================================================================================================


def highest_n_sum(lines, num):
    lines[::-1].sort()
    h_sum = 0

    if np.size(lines, 0) < num:
        num = np.size(lines, 0)

    for x in range(num):
        h_sum += lines[x]

    return h_sum


def extract_baseline(image):
    goodness = []

    for angle in range(-20, 21, 5):
        im_rot = rotate_image(image, angle)

        lines = horizontal_histogram(im_rot)
        goodness.append((angle, highest_n_sum(lines, 15)))

    goodness.sort(reverse=True, key=lambda x: x[1])
    first = goodness[0][0]
    second = goodness[1][0]

    best_angle = best_goodness = -50
    for angle in range(min(first, second), max(first + 1, second + 1)):
        im_rot = rotate_image(image, angle)
        lines = horizontal_histogram(im_rot)

        g = highest_n_sum(lines, 15)
        if g > best_goodness:
            best_goodness = g
            best_angle = angle

    return best_angle


def flatten_baseline(line_images):
    images = []
    angles = []

    for line_im in line_images:
        angle = extract_baseline(line_im)
        angles.append(angle)
        images.append(rotate_image(line_im, angle))

    return images, np.average(angles)


# ======================================================================================================================


def horizontal_histogram(im):
    _, image = cv2.threshold(im, 254, 1, cv2.THRESH_BINARY)
    image = np.invert(image.astype(bool))

    return np.sum(image, axis=1)


def rotate_image(image, angle):
    h, w = image.shape[:2]
    cX, cY = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return cv2.warpAffine(image, M, (nW, nH), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


# ======================================================================================================================


preprocessors_list = [crop_images, split_lines, flatten_baseline]


# MAIN
def preprocess_image(im):
    baseline = 0

    for preprocessor in preprocessors_list:
        im, baseline = preprocessor(im)

    return baseline, im
