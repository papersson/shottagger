from flask import Flask, render_template, request
from joblib import load
import numpy as np
import json

app = Flask(__name__)

model = load('model.joblib')
score_total = 0
xg_total = 0.0

@app.route('/')
def index():
    return render_template('index.html', score_total=score_total, xg_total=xg_total)

@app.route('/predict', methods=['POST'])
def predict():
    global score_total, xg_total
    data = request.get_json(force=True)
    shot_location = np.array(data['shotLocation'])
    defenders = data['defenders']
    outcome = data['outcome']  # 0 (miss) or 1 (goal)

    # Compute features
    goal_location = np.array([105, 34])
    distance = np.sqrt(np.sum((shot_location - goal_location) ** 2))

    goal_width = 7.32
    dx = shot_location[0] - goal_location[0]
    dy1 = shot_location[1] - (goal_location[1] + goal_width / 2)
    dy2 = shot_location[1] - (goal_location[1] - goal_width / 2)
    u = np.array([dx, dy1])
    v = np.array([dx, dy2])
    angle = np.arccos((u @ v) / (np.linalg.norm(u) * np.linalg.norm(v)))

    n_defenders = defenders_in_fov(defenders, shot_location)
    open_net = 1 if n_defenders == 0 else 0

    X = np.array([shot_location[0], shot_location[1], distance, angle, n_defenders, open_net]).reshape(1, -1)
    xg = float(model.predict_proba(X)[:, 1][0])

    score_total += outcome
    xg_total += xg

    response = {
        'score_total': score_total,
        'xg_total': xg_total,
        'xg': xg
    }
    return json.dumps(response), 200, {'Content-Type': 'application/json'}

@app.route('/reset', methods=['POST'])
def reset_scores():
    global score_total, xg_total
    score_total = 0
    xg_total = 0.0
    response = {
        'score_total': score_total,
        'xg_total': xg_total
    }
    return json.dumps(response), 200, {'Content-Type': 'application/json'}

def defenders_in_fov(defenders, shot_location):
    goal_width = 7.32
    p1 = np.array([105, 34 + goal_width / 2])
    p2 = np.array([105, 34 - goal_width / 2])
    count = 0
    for d in defenders:
        d = np.array(d)
        if point_in_triangle(shot_location, p1, p2, d):
            count += 1
    return count

def point_in_triangle(p0, p1, p2, p):
    area = 0.5 *(-p1[1]*p2[0] + p0[1]*(-p1[0]+p2[0]) + p0[0]*(p1[1]-p2[1]) + p1[0]*p2[1])
    s = 1/(2*area)*(p0[1]*p2[0] - p0[0]*p2[1] + (p2[1]- p0[1])*p[0] + (p0[0]-p2[0])*p[1])
    t = 1/(2*area)*(p0[0]*p1[1] - p0[1]*p1[0] + (p0[1]-p1[1])*p[0] + (p1[0]-p0[0])*p[1])
    return s > 0 and t > 0 and 1 - s - t > 0

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
