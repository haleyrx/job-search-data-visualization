# Job Search Visualization Dashboard



## Overview
Our project is meant to provide meaningful visualizations of data job 
postings around the world. The final product is Tableau dashboard with 
multiple views. 

We've spent considerable time pulling data from various sources and quite 
a bit of time cleaning and parsing the data to procure meaningful 
information. The world view of the dashboard, USA dashboard, and topics 
modeling dashboards use postings from a large Kaggle dataset sourced from 
Glassdoor:

	https://www.kaggle.com/andresionek/data-jobs-listings-glassdoor

The code used to clean the data can be found in the data_cleaning
directory (you can follow along with the steps in the Jupyter notebooks).

Data for the salary model is a combination of job listings pulled from a 
variety of data sources online and data scraped using the BeautifulSoup 
and Selenium Python libraries. All data is sourced from Indeed. The final 
data set is viewable here:

	https://gtvault-my.sharepoint.com/:x:/g/personal/hxue42_gatech_edu/EZ_P93gSgQ9JqjE6gRwiV4kB2XIPNLaCRkhJpPj-aSHgWg?e=O67xHY
	
While most of the visualizations you see in the dashboard were implemented 
within Tableau, our analytical portions are implemented outside of Tableau 
and are embedded as webpages. Notably, the predictive salary widget is a 
presented as a hosted app coded in Flask. It allows users to input their
own job parameters into a pretrained model (TFIDF + Random Forest)to obtain 
a salary prediction. The app is hosted on Heroku here:

	https://indeed-salary-predictor.herokuapp.com/

The LDA model and analysis was implemented in Python and visualizations 
were produced using pyLDAvis


## Installation
Our dashboard is meant to be accessible out of the box--only installation 
of Tableau Desktop is neccessary to open the workbook. All associated 
data used to generate the visualizations is saved along with the workbook
in the packaged Tableau workbook file. 

If you'd like to play around with or view the data cleaning and source 
code, installation of Python with Jupyter Notebooks is necessary. 
A list of libraries used are viewable in /CODE/requirements.txt
Python dependencies can be viewed in the respective Jupyter notebooks 
and/or modules found in CODE. 



## Execution
Only an installation of Tableau is necessary to view the final dashboard. 

If you would like to view the source code, relevant files are listed below.
The jupyter notebooks have further details/comments. 

Data Sourcing: ```/CODE/data_sourcing```
* indeed.py - Indeed scraper using BeautifulSoup
* selenium_scrape.ipynb - Indeed scraper using Selenium

Data Cleaning: ```/CODE/data_cleaning``` 
* columns.txt - columns used from glassdoor.csv sourced from Kaggle dataset 
* glassdoor.ipynb, skills.ipynb - code to clean and generate data for Tableau

Salary Model: ```/CODE/salary_model```
* /app - files needed to build Flask app
* salary_pred.ipynb - data cleaning and model building for app
* X.csv - input values for BERT model, taken from salary_pred.ipynb
* y.csv - input truth for BERT model, taken from salary_pred.ipynb
* BERT_salary_prediction.ipynb - model building for BERT prediction, originally run on colab

Topics Model: ```/CODE/topics_modeling```
* Topics_Modeling.ipynb - standalone notebooks that utilized developed pre-processing methods and run the LDA model
* tm_helper.py - python file containing all the pre-processing methods
* lda_model.pkl - final LDA model, it can be loaded in Topics_Modeling.ipynb
* processed_text.pkl - output text after applying pre-processing
* vectorizer.pkl - sklearn.feature_extraction.text.TfidfVectorizer
* glassdoor_us_only_ldavis.csv - final data set with job descriptions and model

