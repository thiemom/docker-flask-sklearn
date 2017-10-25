
from flask import Flask, request, render_template, jsonify
import numpy as np
from sklearn import linear_model
from sklearn.externals import joblib
from sklearn import datasets
import bokeh
import bokeh.plotting as plt
from bokeh.embed import components

# toy data
diabetes = datasets.load_diabetes()

# select one feature
feature_index = 2
feature_name = diabetes.feature_names[feature_index]
X = diabetes.data[:, np.newaxis, feature_index]
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
	loaded_model = joblib.load(model_filename)
	prediction = loaded_model.predict(x)
	return str(np.asscalar(prediction))

@app.route('/data')
def show_data():
	return jsonify(X=X.tolist(), y=y.tolist())

@app.route('/chart')
def chart():
	loaded_model = joblib.load(model_filename)

	plot = scatter(X, y, 
		model=loaded_model,
		title='Diabetes Dataset', 
		xlabel=feature_name, 
		ylabel='Disease progression after 1 year')

	script, div = components(plot)
	head = '''
	<link href="http://cdn.pydata.org/bokeh/release/bokeh-{}.min.css" rel="stylesheet" type="text/css">
  	<script src="http://cdn.pydata.org/bokeh/release/bokeh-{}.min.js"></script>
  	'''.format(bokeh.__version__, bokeh.__version__)

	return render_template('chart.html', 
		page_title='Basic Machine Learning',
		chart_title='Linear Regression Model',
		chart_head=head,
		chart_div=div, 
		chart_script=script)	

def scatter(X, y, model=None, title=None, xlabel=None, ylabel=None):
	'''bokeh plot'''
	p = plt.figure(title=title, 
			x_axis_label=xlabel, 
			y_axis_label=ylabel)
	p.circle(X.ravel(), y,
		fill_color='blue', 
		fill_alpha=0.8,
		size=8)
	if not model is None:
		Xp = np.linspace(X.min(), X.max(), 100).reshape(-1,1)
		yp = model.predict(Xp)
		p.line(Xp.ravel(), yp, 
			line_color='red', 
			line_width=2)
	return p

if __name__ == '__main__':
	app.run(port=5000, host='0.0.0.0', debug=True)        
