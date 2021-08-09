from joblib import load
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__, template_folder='.')

model = load('model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    shot_location = request.get_json()
    x = to_features(shot_location)
    x = np.array(x).reshape(1, -1)
    xg = round(model.predict_proba(x)[:, 1][0], 2)
    print(xg)
    return {'xg': xg}

def to_features(shot_location):
    goal_location = np.array([105, 34])
    distance = np.sqrt(np.sum((shot_location - goal_location) ** 2))

    goal_width = 7.32
    dx = shot_location[0] - goal_location[0]
    dy1 = shot_location[1] - (goal_location[1] + goal_width / 2)
    dy2 = shot_location[1] - (goal_location[1] - goal_width / 2)
    u = np.array([dx, dy1])
    v = np.array([dx, dy2])
    angle = np.arccos((u @ v) / (np.linalg.norm(u) * np.linalg.norm(v)))

    return [*shot_location, distance, angle]


@app.route('/')
def home():
    return render_template('index.html')

app.run(debug=True)

#x_new = np.array([88, 32]).reshape(1, -1)
#prediction = model.predict_proba(x_new)[:, 1][0]
