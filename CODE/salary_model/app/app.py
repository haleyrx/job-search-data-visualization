import pandas as pd
import os
from flask import Flask, render_template, request, redirect, Response, url_for
import pickle

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# constants
bin_dict = {'<60K':'Below $60K', '60K - 80K':'$60K-$80K', 
			'80K - 100K':'$80K-100K','100K - 140K':'$100K-$140K', 
			'>140K': 'Above $140K'}

# load model
with open('model.pkl','rb') as f:
	model = pickle.load(f)

# app
app = Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html', error_msg='', 
							salary_range='', salary_prediction_text=''
							)

# routes
@app.route('/predict', methods=['GET', 'POST'])
def predict():
	title = request.form.get('title')
	level = request.form.get('level')
	company = request.form.get('company')
	living_index = request.form.get('living_index')
	description = request.form.get('description')

	try:
		# convert data to dataframe
		column_names = ['company', 'title_parsed', 'level', 'description', 'living_index']
		df = pd.DataFrame(columns = column_names)
		df['company'] = [company]
		df['title_parsed'] = [title]
		df['level'] = [level]
		df['description'] = [description]
		df['living_index'] = [float(living_index)]

		print(df)
		result = model.predict(df)
		print(result)
		error = ''
		prediction = bin_dict[result[0]]
		text = 'Salary Range of Job: '

	except:
		error = 'Please fill in the missing fields below'
		prediction = ''
		text = ''

	# return data
	return render_template('index.html', error_msg=error, 
							salary_range=prediction, 
							salary_prediction_text=text, title=title, level=level)


if __name__ == '__main__':
	app.run(debug=True)

