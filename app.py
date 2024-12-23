from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import statistics

app = Flask(__name__)

# Load the models and data
data = pd.read_csv("dataset/Training.csv").dropna(axis=1)
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

final_svm_model = SVC()
final_nb_model = GaussianNB()
final_rf_model = RandomForestClassifier(random_state=18)
final_svm_model.fit(X, y)
final_nb_model.fit(X, y)
final_rf_model.fit(X, y)

symptoms = X.columns.values
symptom_index = { " ".join([i.capitalize() for i in value.split("_")]): index for index, value in enumerate(symptoms) }
data_dict = { "symptom_index": symptom_index, "predictions_classes": encoder.classes_ }

def predictDisease(symptoms):
    symptoms = symptoms.split(",")
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        index = data_dict["symptom_index"].get(symptom.strip(), None)
        if index is not None:
            input_data[index] = 1
    input_data = np.array(input_data).reshape(1, -1)
    
    # Debugging statements
    print(f"Symptoms: {symptoms}")
    print(f"Input Data: {input_data}")
    
    rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
    nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
    svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]
    
    # Debugging statements
    print(f"RF Prediction: {rf_prediction}")
    print(f"NB Prediction: {nb_prediction}")
    print(f"SVM Prediction: {svm_prediction}")
    
    final_prediction = statistics.mode([rf_prediction, nb_prediction, svm_prediction])
    return {
        "rf_model_prediction": rf_prediction,
        "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": svm_prediction,
        "final_prediction": nb_prediction
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        symptoms = request.form['symptoms']
        prediction = predictDisease(symptoms)
        return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(port=5001,debug=True)