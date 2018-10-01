#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print('Total data points = ', len(enron_data))
print('Features per person = ', len(enron_data['TAYLOR MITCHELL S']))
print(enron_data['TAYLOR MITCHELL S'].keys())

count_poi = 0
for person in enron_data:
    if enron_data[person]['poi']==1:
        count_poi += 1

print('Count of Persons of Interest = ', count_poi)

print('Stocks of James Prentice = ', enron_data['PRENTICE JAMES']['total_stock_value'])
print('Mails from Wesley Colwell to POIs = ', enron_data['COLWELL WESLEY']['from_this_person_to_poi'])
print('Exercised stock options of Jeffrey K Skilling = ', enron_data['SKILLING JEFFREY K']['exercised_stock_options'])

print('Total payments of Lay = ', enron_data['LAY KENNETH L']['total_payments'])
print('Total payments of Skilling = ', enron_data['SKILLING JEFFREY K']['total_payments'])
print('Total payments of Fastow = ', enron_data['FASTOW ANDREW S']['total_payments'])

quantified_salaries = 0
known_email_addresses = 0
NaN_total_payments = 0
POI_with_NaN_total_payments = 0
for k, v in enron_data.iteritems():
    if v['salary'] != 'NaN':
        quantified_salaries += 1
    if v['email_address'] != 'NaN':
        known_email_addresses += 1
    if v['total_payments'] == 'NaN':
        NaN_total_payments += 1
    if v['poi']==1 and v['total_payments'] == 'NaN':
        POI_with_NaN_total_payments += 1

print('Total quantified salaries = ', quantified_salaries)
print('Known Email addresses = ', known_email_addresses)
print('Count of people with NaN as total payments = ', NaN_total_payments)
print('Count of POI with NaN as total payments = ', POI_with_NaN_total_payments)
