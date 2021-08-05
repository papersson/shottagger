from joblib import load
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__, template_folder='.')

model = load('model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    location = request.get_json()
    x = to_features(location)
    x = np.array(x).reshape(1, -1)
    xg = model.predict_proba(x)[:, 1][0]
    print(xg)
    return {'xg': xg}

def to_features(location):
    destination = np.array([105, 34])
    distance = np.sqrt(np.sum((location - destination) ** 2))

    p0 = np.array((105, 34 - 7.32 / 2))
    p1 = np.array(location, dtype=np.float)
    p2 = np.array((105, 34 + 7.32 / 2))

    v0 = p0 - p1
    v1 = p2 - p1

    angle = np.abs(np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1)))

    return [*location, distance, angle]


@app.route('/')
def home():
    return render_template('index.html')

app.run(debug=True)

#x_new = np.array([88, 32]).reshape(1, -1)
#prediction = model.predict_proba(x_new)[:, 1][0]
