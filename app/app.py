
from flask import Flask, request

import numpy as np
from sklearn import linear_model
from sklearn.externals import joblib
from sklearn import datasets
from sklearn.model_selection import train_test_split

# data
diabetes = datasets.load_diabetes()

# select one feature
X = diabetes.data[:, np.newaxis, 2]
y = diabetes.target

# fit model
model = linear_model.LinearRegression()
model.fit(X, y)

# save
model_filename = 'model.pkl'
joblib.dump(model, model_filename) 

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello"

@app.route('/predict', methods=['GET'])
def predict():
    x = float(request.args.get('x'))
    clf = joblib.load(model_filename) 
    prediction = clf.predict(x)
    return str(prediction)
   
if __name__ == '__main__':
    app.run(port=5000, host='0.0.0.0')        
