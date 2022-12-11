import flask
from flask import Flask, request, render_template
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np

app = flask.Flask(__name__, template_folder='templates')

loaded_model = pickle.load(open("Random_Forest_for_Depth.pkl", 'rb'))
loaded_model_for_Width = pickle.load(open("Random_Forest_for_Width.pkl", 'rb'))
scaler_X = pickle.load(open("scaler_X2.pkl", 'rb'))

@app.route('/', methods = ['POST', 'GET'])

def main():
    if request.method == 'GET':
        return render_template('main.html')
    
    if request.method == 'POST':
        X_list_for_predict = []
        print('Done')

        IW = request.form['iw']
        IF = request.form['if']
        VW = request.form['vw']
        FP = request.form['fp']
        X_list_for_predict.append(float(IW))
        X_list_for_predict.append(float(IF))
        X_list_for_predict.append(float(VW))
        X_list_for_predict.append(float(FP))

        print(X_list_for_predict)

        X_list_for_predict_scaled = scaler_X.transform([X_list_for_predict])
        y_pred_Depth = loaded_model.predict(X_list_for_predict_scaled)
        y_pred_Width = loaded_model_for_Width.predict(X_list_for_predict_scaled)
        return render_template('main.html', result = [round(float(np.exp(y_pred_Depth)), 4) , round(float(np.exp(y_pred_Width)), 4)])

if __name__ == '__main__':
    app.run(debug = True)

