from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import cv2

from prep_extract_deploy import preprocess_and_extract_feat as get_sample

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

loaded_model = pickle.load(open('tuzno_sretno_ljuto_KP_SVC_model.sav', 'rb'))

COLUMN_NUM = 22

SCALER = None

FEATURES_FILE = "features_KP.txt"

emotions = {
    0: "ljuto",
    1: "sretno",
    2: "tuzno"
}


@app.route('/')
def hello():
    return '<div>Hello</div>'


@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)

    sample = np.array(data).reshape(1, -1)

    # Make prediction using model loaded from disk as per the data.
    prediction = loaded_model.predict(sample)

    # Take the first value of prediction
    output = emotions[int(prediction[0])]
    return jsonify(output)


@app.route('/extract', methods=['POST'])
def extract_predict():
    # Get the data from the POST request.
    global SCALER
    if SCALER is None:
        fit_scaler()

    image_stream = request.files['image'].read()
    device_file = request.files['device'].read()

    bytes_as_np_array = np.frombuffer(image_stream, dtype=np.uint8)
    image_file = cv2.imdecode(bytes_as_np_array, 0)

    sample = get_sample(image_file, device_file)
    sample = SCALER.transform(sample)

    prediction = loaded_model.predict(sample)

    # Take the first value of prediction
    result = emotions[int(prediction[0])]
    output = jsonify({'prediction': result})

    return output


def fit_scaler():
    global SCALER
    SCALER = StandardScaler()
    x = np.loadtxt(FEATURES_FILE, delimiter=' ', usecols=range(1, COLUMN_NUM))
    SCALER.fit(x)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
