import flask
from flask import Flask, request, render_template
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

app = flask.Flask(__name__, template_folder='templates')

loaded_model = pickle.load(open("Random_Forest.pkl", 'rb'))
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
        y_pred = loaded_model.predict(X_list_for_predict_scaled)


        return render_template('main.html', result = y_pred)

if __name__ == '__main__':
    app.run(debug = True)

