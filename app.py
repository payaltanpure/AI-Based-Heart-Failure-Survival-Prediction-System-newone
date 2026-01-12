from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form values
        data = [
            float(request.form['age']),
            int(request.form['anaemia']),
            int(request.form['creatinine_phosphokinase']),
            int(request.form['diabetes']),
            int(request.form['ejection_fraction']),
            int(request.form['high_blood_pressure']),
            float(request.form['platelets']),
            float(request.form['serum_creatinine']),
            int(request.form['serum_sodium']),
            int(request.form['sex']),
            int(request.form['smoking']),
            int(request.form['time'])
        ]

        # Scale input
        scaled_data = scaler.transform([data])
        prediction = model.predict(scaled_data)[0]

        result = "Patient likely to survive" if prediction == 0 else "Patient likely to die"
        return render_template('index.html', result=result)

    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)

