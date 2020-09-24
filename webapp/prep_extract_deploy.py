from preprocessing import preprocess_image
from extracting import extract_features


def preprocess_and_extract_feat(filename_image, filename_device):
    baseline, images = preprocess_image(filename_image)
    features = extract_features(images, filename_device, baseline)

    return features
