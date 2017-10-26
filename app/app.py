
from flask import Flask, request, render_template, jsonify

import numpy as np

from sklearn import linear_model
from sklearn.externals import joblib
from sklearn import datasets

import bokeh
import bokeh.plotting as plt
from bokeh.embed import components

# toy data
data = datasets.load_boston()
data_id = 'Boston Housing Dataset'
target_name = 'Price'

# select one feature
feature_index = 0
feature_name = data.feature_names[feature_index]
X = data.data
y = data.target

# fit model
model = linear_model.LinearRegression()
model.fit(X, y)

# save model
model_filename = 'model.pkl'
joblib.dump(model, model_filename) 

app = Flask(__name__)

@app.route('/')
def index():
	return "Hello, from a Machine Learning Web Tool"

@app.route('/predict', methods=['GET'])
def predict():
	loaded_model = joblib.load(model_filename)
	Xp = np.empty((1, X.shape[1]))
	for i, feat in enumerate(data.feature_names):
		Xp[0, i] = request.args.get(feat, default=X.mean(axis=0)[i], type=np.float)
	yp = loaded_model.predict(Xp)
	return jsonify(
		data=dict(zip(data.feature_names, Xp.T.tolist())), 
		prediction={target_name: yp.tolist()})

@app.route('/data')
def show_data():
	return jsonify(
		data=dict(zip(data.feature_names, X.tolist())), 
		prediction={target_name: y.tolist()})

@app.route('/chart')
def chart():
	loaded_model = joblib.load(model_filename)

	plot = scatter(X, y, 
		model=loaded_model,
		title=data_id, 
		xlabel=feature_name, 
		ylabel=target_name)

	script, div = components(plot)
	head = '''
	<link href="http://cdn.pydata.org/bokeh/release/bokeh-{}.min.css" rel="stylesheet" type="text/css">
  	<script src="http://cdn.pydata.org/bokeh/release/bokeh-{}.min.js"></script>
  	'''.format(bokeh.__version__, bokeh.__version__)

	return render_template('chart.html', 
		page_title='Basic Machine Learning',
		chart_title=str(loaded_model).split('(')[0],
		chart_head=head,
		chart_div=div, 
		chart_script=script)	

def scatter(X, y, model=None, title=None, xlabel=None, ylabel=None):
	'''bokeh plot'''
	p = plt.figure(title=title, 
			x_axis_label=xlabel, 
			y_axis_label=ylabel)
	p.circle(X[:,feature_index], y,
		fill_color='blue', 
		fill_alpha=0.8,
		size=8)
	if not model is None:
		N = 100
		Xp = np.empty((N, X.shape[1]))
		Xp[:,:] = X.mean(axis=0)
		Xp[:,feature_index] = np.linspace(X[:,feature_index].min(), 
			X[:,feature_index].max(), N)
		yp = model.predict(Xp)
		p.line(Xp[:,feature_index], yp, 
			line_color='red', 
			line_width=2)
	return p

if __name__ == '__main__':
	app.run(port=5000, host='0.0.0.0', debug=False)        
