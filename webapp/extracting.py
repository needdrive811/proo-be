import cv2
import math
import peakutils
import numpy as np
import pandas as pd


SAMPLE_EXTENSION = ".json"
IMAGE_EXTENSION = ".png"

# COLUMNS = ["pressure", "tilt_x", "tilt_y", "b", "pointer_id", "timestamp", "x", "y"]
COLUMNS = ["timestamp", "pointer_id", "x", "y", "pressure", "tilt_x", "tilt_y", "b"]

TIME_TOTAL = "t_total"
TIME_PAPER = "t_paper"
TIME_AIR = "t_air"

SPEED_AVG_X = "sp_x"
SPEED_STROKES_AVG = "sp_strokes_avg"
SPEED_STROKES_SD = "sp_strokes_sd"

STROKES_AIR = "st_air"
STROKES_PAPER = "st_paper"

PRESSURE_AVG = "p_avg"
PRESSURE_DIF = "p_dif"
PRESSURE_SD = "p_sd"

TILT_X_AVG = "tilt_x_avg"
TILT_Y_AVG = "tilt_y_avg"
TILT_X_SD = "tilt_x_sd"
TILT_Y_SD = "tilt_y_sd"

IMG_BASELINE = "i_bl"
IMG_WORD_SLANT = "i_wslant"
IMG_WORD_DIST = "i_wdis"
IMG_WORD_SIZE = "i_wsize"
IMG_MARGINS_TOP = "i_m_top"
IMG_MARGINS_LEFT = "i_m_left"

FEATURE_ORDER = [TIME_TOTAL, TIME_PAPER, TIME_AIR, SPEED_AVG_X, SPEED_STROKES_AVG, SPEED_STROKES_SD,
                 STROKES_AIR, STROKES_PAPER, PRESSURE_AVG, PRESSURE_DIF, PRESSURE_SD, TILT_X_AVG,
                 TILT_Y_AVG, TILT_X_SD, TILT_Y_SD, IMG_BASELINE, IMG_WORD_SLANT, IMG_WORD_DIST,
                 IMG_WORD_SIZE, IMG_MARGINS_TOP, IMG_MARGINS_LEFT]


# DEVICE FEATURES EXTRACTORS
# ======================================================================================================================


def extract_pressure(data):
    p = data[data > 0]["pressure"]
    return {PRESSURE_AVG: p.mean(), PRESSURE_DIF: p.max() - p.min(), PRESSURE_SD: p.std()}


def extract_tilt(data):
    t = data[["tilt_x", "tilt_y"]]
    return {
            TILT_X_AVG: t["tilt_x"].mean(), TILT_Y_AVG: t["tilt_y"].mean(),
            TILT_X_SD: t["tilt_x"].std(), TILT_Y_SD: t["tilt_y"].std()
           }


def extract_strokes(data):
    stroke_air = np.size(data[data["ShiftStatus"] == -1], 0)
    stroke_paper = stroke_air
    if data.at[data.index[0], "b"] == 1:
        stroke_paper = stroke_paper + 1
    else:
        stroke_air = stroke_air + 1

    return {STROKES_AIR: stroke_air, STROKES_PAPER: stroke_paper}


def extract_time(data):
    StatusChangeData = data[data["ShiftStatus"] != 0].copy()
    StatusChangeData["TimeDiff"] = StatusChangeData["timestamp"].shift(1)
    StatusChangeData.at[StatusChangeData.index[0], "TimeDiff"] = data.at[data.index[0], "timestamp"]
    StatusChangeData["TimeDiff"] = StatusChangeData["timestamp"] - StatusChangeData["TimeDiff"]

    status_one = StatusChangeData[StatusChangeData["b"] == 1]
    _sum = status_one["TimeDiff"].sum()
    status_zero = StatusChangeData[StatusChangeData["b"] == 0]
    sum_zero = status_zero["TimeDiff"].sum()

    if data.at[data.index[-1], "b"] == 1:
        _sum = _sum + data.at[data.index[-1], "timestamp"] - StatusChangeData.at[
            StatusChangeData.index[-1], "timestamp"]
    else:
        sum_zero = sum_zero + data.at[data.index[-1], "timestamp"] - StatusChangeData.at[
            StatusChangeData.index[-1], "timestamp"]

    time_total = data.at[data.index[-1], "timestamp"] - data.at[data.index[0], "timestamp"]

    return {TIME_TOTAL: time_total / 1000, TIME_PAPER: _sum / time_total, TIME_AIR: sum_zero / time_total}


def extract_speed(data):
    help_df = data[data["ShiftStatus"] != 0].copy()
    help_df["ShiftX"] = help_df["x"].shift(-1)
    help_df.at[help_df.index[-1], "ShiftX"] = help_df.at[help_df.index[-1], "x"]
    # u ShiftX je razlika između te i sljedece koord x
    help_df["ShiftX"] = help_df["ShiftX"] - help_df["x"]

    help_df["endLine"] = np.zeros(help_df.shape[0])

    help_df.index = range(help_df.shape[0])

    help_index = help_df[help_df["ShiftX"] < -max(help_df["x"]) / 2].index
    help_df.at[help_df.index[help_index], "endLine"] = 1
    # u endLine je označen kraj linije, točka prije prelaska u novi red
    help_df.at[help_df.index[-1], "endLine"] = 1

    help_df["ShiftEndLine"] = help_df["endLine"].shift(1)
    help_df.at[help_df.index[0], "ShiftEndLine"] = 1
    help_df["ShiftEndLine"] = help_df["ShiftEndLine"] - help_df["endLine"]

    # tocke izmedu kojih je doslo do prelaska u novi red
    newLinePoints = help_df[help_df["ShiftEndLine"] != 0].copy()

    newLinePoints["ShiftEndX"] = newLinePoints["x"].shift(1)
    newLinePoints.at[newLinePoints.index[0], "ShiftEndX"] = newLinePoints.at[newLinePoints.index[0], "x"]
    newLinePoints["ShiftEndX"] = newLinePoints["ShiftEndX"] - newLinePoints["x"]

    newLinePoints["ShiftStamp"] = newLinePoints["timestamp"].shift(1)
    newLinePoints.at[newLinePoints.index[0], "ShiftStamp"] = newLinePoints.at[newLinePoints.index[0], "timestamp"]
    newLinePoints["ShiftStamp"] = newLinePoints["ShiftStamp"] - newLinePoints["timestamp"]

    newLinePoints["Speed"] = newLinePoints["ShiftEndX"] / newLinePoints["ShiftStamp"]
    avg_speed = newLinePoints[newLinePoints["Speed"] > 0]["Speed"].sum() * 2000 / newLinePoints.shape[0]

    return {SPEED_AVG_X: avg_speed}


def extract_stroke_speed(data):
    data["next_x"] = data["x"].shift(-1).copy()
    data["next_y"] = data["y"].shift(-1).copy()

    pen_down = data.index[data["ShiftStatus"] == 1].tolist()
    pen_up = data.index[data["ShiftStatus"] == -1].tolist()

    if data.at[data.index[0], "b"] == 1:
        pen_down.insert(0, -1)

    if data.at[data.index[-1], "b"] == 1:
        pen_up.append(data.shape[0] - 1)

    stroke_speeds = []
    for i in range(len(pen_down)):
        data_subset = data.iloc[pen_down[i] + 1: pen_up[i]]

        dis_x = data_subset["x"] - data_subset["next_x"]
        dis_y = data_subset["y"] - data_subset["next_y"]

        dis = np.sum(np.sqrt(dis_x * dis_x + dis_y * dis_y))
        start = data.at[pen_down[i] + 1, "timestamp"]
        end = data.at[pen_up[i], "timestamp"]

        if start != end:
            stroke_speeds.append(dis / ((end - start) / 1000))

    return {SPEED_STROKES_AVG: np.average(stroke_speeds), SPEED_STROKES_SD: np.std(stroke_speeds)}


# IMAGE FEATURE EXTRACTORS
# ======================================================================================================================


# MARGINS
# ------------------------------------------------------------------------
def extract_margins(line_images):
    top_margins = []
    left_margins = []

    for im in line_images:
        top, left = extract_top_left(im)

        top_margins.append(top)
        left_margins.append(left)

    return {IMG_MARGINS_TOP: np.average(top_margins), IMG_MARGINS_LEFT: np.average(left_margins)}


def extract_top_left(im):
    _, im = cv2.threshold(im, 254, 1, cv2.THRESH_BINARY)
    im = np.invert(im.astype(bool))

    hor = np.sum(im, axis=1)
    ver = np.sum(im, axis=0)

    top = np.where(hor == hor[hor > hor.mean()][0])[0][0]
    left = np.where(ver == ver[ver > ver.mean()][0])[0][0]

    return top, left


# WORD SLANT, SIZE AND DISTANCE
# ------------------------------------------------------------------------
def extract_word_features(line_images):
    distances = []
    slants = []
    sizes = []

    for im in line_images:
        ret2, th2 = cv2.threshold(im, 254, 255, cv2.THRESH_BINARY)
        segmentation = word_segmentation(th2, kernel_size=25, sigma=11, theta=5, min_area=400)

        mean_value = []
        last_one = 0
        words_list = []
        cnt = 0
        distance_line = 0

        for j, w in enumerate(segmentation):
            word_box, word_img = w
            x, y, w, h = word_box

            current = x - last_one
            if x + w > last_one:
                last_one = x + w
                if cnt == 0:
                    cnt = 1
                elif current > 0:
                    mean_value.append(current)

                words_list.append(word_img)

        if len(mean_value) > 0:
            distance_line = np.mean(mean_value)
        distances.append(distance_line)

        slant, size = extract_slant_and_size(words_list)
        slants.append(slant)
        sizes.append(size)

    return {IMG_WORD_DIST: np.average(distances), IMG_WORD_SLANT: np.average(slants), IMG_WORD_SIZE: np.average(sizes)}


def extract_slant_and_size(words):
    angle_sum = 0
    letter_sizes = np.zeros(0)
    cnt = 0

    # provodi algoritam za sve riječi dobivene iz segmentacije
    for w_img in words:
        cnt += 1
        rows, cols = w_img.shape
        w_img = w_img[1:rows, 1:cols]
        rows, cols = w_img.shape

        ret2, w_img = cv2.threshold(w_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # nagib riječi (slant)
        max_angle = find_max_angle(w_img)

        angle = math.atan(max_angle / rows) * 180 / math.pi

        # rotacija slike za pronađeni kut
        w_img = rotate_word_image(w_img, max_angle, rows, cols)

        angle_sum += angle

        w_img = 255 - w_img
        cols_t = np.sum(w_img, 1)
        cols_t = np.flip(cols_t, axis=0)

        # veličina slova
        letter_sizes = np.append(letter_sizes, [find_letter_size(cols_t)])

    slant = angle_sum / cnt

    # micanje malih veličina slova (u takvima je došlo do greške)
    old_avg_letter_size = sum(letter_sizes) / np.size(letter_sizes, 0)
    new_letters = letter_sizes[np.where(letter_sizes > 0.5 * old_avg_letter_size)]
    avg_letter_size = sum(new_letters) / np.size(new_letters, 0)

    return slant, avg_letter_size


def find_max_angle(image):
    rows, cols = image.shape
    peak_sums = []

    for deg in range(-rows, rows + 1, max(1, rows // 10)):
        peak_sums.append((deg, calculate_peak_sum(rows, cols, deg, image)))

    peak_sums.sort(reverse=True, key=lambda x: x[1])

    first = peak_sums[0][0]
    second = peak_sums[1][0]

    max_angle = -rows
    max_sum = 0

    for deg in range(min(first, second), max(first + 1, second + 1)):
        peak_sum = calculate_peak_sum(rows, cols, deg, image)

        if peak_sum > max_sum:
            max_sum = peak_sum
            max_angle = deg

    return max_angle


def find_letter_size(cols_t):
    cb = cols_t.astype(int)
    proj_list = cb.tolist()
    # Izračunava se maksimalna vrijednost i prosjek vrijednosti histograma.
    max_ind = proj_list.index(max(proj_list))
    mean_height = cb.mean()

    # 1. Promatranje piksela koji se nalaze "iznad" piksela s maksimalnom vrijednošću
    after_max = cb[max_ind:]
    min_ind = np.where(after_max == min(after_max[np.where(after_max > 0)]))
    min_ind = min_ind[0][0]

    large_index = np.where(after_max >= mean_height)
    large_upper = after_max[np.where(after_max >= mean_height)]
    value_max = large_upper[max(np.where(large_index <= min_ind)[1])]

    equalMaxList = np.where(cb == value_max)[0]
    less_mini = equalMaxList[np.where(equalMaxList <= min_ind+max_ind)[0]]

    upper_line = max(less_mini)

    # 2. Promatranje piksela koji se nalaze "ispod" piksela s maksimalnom vrijednošću
    before_max = cb[:max_ind+1]
    min_bef_ind = np.where(before_max == min(before_max[np.where(before_max > 0)]))
    min_bef_ind = min_bef_ind[0][0]

    large_below_index = np.where(before_max>=mean_height)
    large_below = before_max[np.where(before_max >= mean_height)]
    value_min = large_below[min(np.where(large_below_index >= min_bef_ind)[1])]

    equal_mini_list = np.where(cb == value_min)[0]
    less_mini = equal_mini_list[np.where(equal_mini_list >= min_bef_ind)[0]]

    avg_line = min(less_mini)

    return upper_line - avg_line


# WORD SEGMENTATION
# ------------------------------------------------------------------------
def word_segmentation(image, kernel_size=25, sigma=11, theta=7, min_area=0):
    kernel = create_kernel(kernel_size, sigma, theta)
    img_filtered = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REPLICATE).astype(np.uint8)

    _, img_thr = cv2.threshold(img_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_thr = 255 - img_thr

    _, components, _ = cv2.findContours(img_thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    res = []
    for c in components:
        # skip small word candidates
        if cv2.contourArea(c) < min_area:
            continue

        # append bounding box and image of word to result list
        curr_box = cv2.boundingRect(c)  # returns (x, y, w, h)
        (x, y, w, h) = curr_box
        curr_img = image[y: y + h, x: x + w]
        res.append((curr_box, curr_img))

    # return list of words, sorted by x-coordinate
    return sorted(res, key=lambda entry: entry[0][0])


def create_kernel(kernel_size, sigma, theta):
    assert kernel_size % 2 # must be odd size
    half_size = kernel_size // 2

    kernel = np.zeros([kernel_size, kernel_size])
    sigma_x = sigma
    sigma_y = sigma * theta

    for i in range(kernel_size):
        for j in range(kernel_size):

            x = i - half_size
            y = j - half_size

            exp_term = np.exp(-x**2 / (2 * sigma_x) - y**2 / (2 * sigma_y))
            x_term = (x**2 - sigma_x**2) / (2 * math.pi * sigma_x**5 * sigma_y)
            y_term = (y**2 - sigma_y**2) / (2 * math.pi * sigma_y**5 * sigma_x)

            kernel[i, j] = (x_term + y_term) * exp_term

    kernel = kernel / np.sum(kernel)
    return kernel


# UTIL
# ------------------------------------------------------------------------
def highest_n_sum_or_less(lines, num):
    lines[::-1].sort()
    h_sum = 0
    if num > np.size(lines):
        num = np.size(lines)
    for x in range(num):
        h_sum += lines[x]

    if num != 0:
        h_sum = h_sum / num
    return h_sum


def calculate_peak_sum(rows, cols, deg, image):
    # Transformacija
    # pts1 = np.float32([[0, rows], [cols, rows], [0, 0]])
    # pts2 = np.float32([[0, rows], [cols, rows], [deg, 0]])
    #
    # M = cv2.getAffineTransform(pts1, pts2)
    # crop_img_t = cv2.warpAffine(image, M, (cols, rows), borderValue=(255, 255, 255))
    crop_img_t = rotate_word_image(image, deg, rows, cols)
    crop_img_rev = 255 - crop_img_t

    # Calculate horizontal projection
    proj = np.sum(crop_img_rev, 0)
    cb = proj.astype(int)

    # Izracunavanje vrhova
    indexes = peakutils.indexes(cb, thres=0.1, min_dist=5)
    proj_list = proj.tolist()
    res_list = list(map(proj_list.__getitem__, indexes))

    return highest_n_sum_or_less(res_list, 10)


def rotate_word_image(image, max_angle, rows, cols):
    pts1 = np.float32([[0, rows], [cols, rows], [0, 0]])
    pts2 = np.float32([[0, rows], [cols, rows], [max_angle, 0]])

    M = cv2.getAffineTransform(pts1, pts2)
    dst_fn = cv2.warpAffine(image, M, (cols, rows), borderValue=(255, 255, 255))
    return dst_fn


# MAIN
# ======================================================================================================================


DEVICE_EXTRACTORS = [extract_pressure, extract_tilt, extract_strokes, extract_time,
                     extract_speed, extract_stroke_speed]

IMAGE_EXTRACTORS = [extract_word_features, extract_margins]


def to_array(features):
    feat_df = pd.DataFrame([features])

    cols = FEATURE_ORDER
    feat_df = feat_df[cols]

    np_feat = feat_df.values

    return np_feat


def preprocess_data(data):
    data.drop_duplicates(inplace=True)
    data.reset_index(drop=True, inplace=True)
    data.columns = COLUMNS

    data["b"] = data["b"].replace(False, 0)

    data["ShiftStatus"] = data["b"].shift(-1)
    data.at[data.index[-1], "ShiftStatus"] = data.at[data.index[-1], "b"]
    data["ShiftStatus"] = data["ShiftStatus"] - data["b"]


def extract_written_features(data):
    data = pd.read_json(data)
    preprocess_data(data)

    features = {}

    for extractor in DEVICE_EXTRACTORS:
        features.update(extractor(data))

    return features


def extract_image_features(image_files, baseline):
    images = []

    for im in image_files:
        images.append(im)

    features = {IMG_BASELINE: baseline}

    for extractor in IMAGE_EXTRACTORS:
        features.update(extractor(images))

    return features


# MAIN
def extract_features(images, device_data, baseline):
    features = extract_written_features(device_data)
    features.update(extract_image_features(images, baseline))

    return to_array(features)
