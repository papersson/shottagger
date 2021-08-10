from joblib import load
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__, template_folder='.')

model = load('model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    x = to_features(data)
    x = np.array(x).reshape(1, -1)
    xg = round(model.predict_proba(x)[:, 1][0], 2)
    return {'xg': xg}

def to_features(data):
    shot_location = data['shotLocation']
    goal_location = np.array([105, 34])
    distance = np.sqrt(np.sum((shot_location - goal_location) ** 2))

    goal_width = 7.32
    dx = shot_location[0] - goal_location[0]
    dy1 = shot_location[1] - (goal_location[1] + goal_width / 2)
    dy2 = shot_location[1] - (goal_location[1] - goal_width / 2)
    u = np.array([dx, dy1])
    v = np.array([dx, dy2])
    angle = np.arccos((u @ v) / (np.linalg.norm(u) * np.linalg.norm(v)))

    defenders = data['defenders']
    n_defenders = defenders_in_fov(defenders, shot_location)
    return [*shot_location, distance, angle, n_defenders]

def point_in_triangle(p0, p1, p2, p):
    # https://stackoverflow.com/questions/2049582/how-to-determine-if-a-point-is-in-a-2d-triangle
    area = 0.5 *(-p1[1]*p2[0] + p0[1]*(-p1[0] + p2[0]) + p0[0]*(p1[1] - p2[1]) + p1[0]*p2[1]);
    s = 1/(2*area)*(p0[1]*p2[0] - p0[0]*p2[1] + (p2[1]- p0[1])*p[0] + (p0[0] - p2[0])*p[1]);
    t = 1/(2*area)*(p0[0]*p1[1] - p0[1]*p1[0] + (p0[1] - p1[1])*p[0] + (p1[0] - p0[0])*p[1]);
    return s > 0 and t > 0 and 1 - s - t > 0

def defenders_in_fov(defenders, shot_location):
    goal_width = 7.32

    p1 = np.array([105, 34 + goal_width / 2])
    p2 = np.array([105, 34 - goal_width / 2])
    count = 0
    for defender in defenders:
        if point_in_triangle(shot_location, p1, p2, defender):
            count += 1
    return count


@app.route('/')
def home():
    return render_template('index.html')

app.run(debug=True)
