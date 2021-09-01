import csv
import datetime
from bs4 import BeautifulSoup
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm

url = "http://www.indeed.com/jobs?q=data+scientist&sort=date&start={}&l={}"


d = {"Title":[], "Company":[], "Rate":[], "Location":[], "Salary":[], "Summary":[], "Url":[]}

city="Atlanta, Georgia"
city = city.replace(" ", "+")

for page in tqdm(range(0,20,10)):

	response = requests.get(url.format(str(page), city))


	soup = BeautifulSoup(response.text, "html.parser")

	cards = soup.find_all("div", "jobsearch-SerpJobCard")



	for i in cards:

	#Title
		atag = i.h2.a
		job_title = atag.get('title')
	#Salary
		salary = i.find("span", "salaryText")
		if salary is not None:
			salary = salary.text.strip()
	#Company
		company = i.find("span", "company").text.strip()
	#Rating
		g = str(i).find("Company rating ")
		rate = str(i)[g+15:][:3]
	#Summary
		summary = i.find("div", "summary").text.strip()
	#Location
		# loc = i.find("div", "recJobLoc")
		# if loc is not None:
		# 	loc = loc.text.strip()
		loc = str(i.find("div", "recJobLoc"))[36:]
		g = loc.find('"')
		loc = loc[:g]
	#d
		d["Title"].append(job_title)
		d["Company"].append(company)
		d["Location"].append(loc)
		d["Salary"].append(salary)
		if rate != "bse":
			d["Rate"].append(rate)
		else:
			d["Rate"].append("")

		gg = str(i).find("data-jk=")
		card_url_text = str(i)[gg+9:][:30]
		ggg = card_url_text.find('"')
		card_url = card_url_text[:ggg]

		summary_url = "http://www.indeed.com/viewjob?jk=" + card_url
		d["Url"].append(summary_url)
		summary_parse = requests.get(summary_url)
		sum_soup = BeautifulSoup(summary_parse.text, "html.parser")

		try:
			temp_sum = sum_soup.find("div", attrs={"class":"jobsearch-jobDescriptionText"}).text.strip()
			d["Summary"].append(temp_sum)
		except:
			d["Summary"].append(summary.strip())


df = pd.DataFrame(data=d)
print(df.head())
df.to_csv('indeed_scrap.csv', index=False)

print("DONE OwO")










