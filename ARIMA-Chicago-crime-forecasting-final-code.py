# Imports
import pandas
import numpy
import seaborn
import matplotlib.colors as colors
import matplotlib.pyplot as pyplot
import folium
import pyflux
import IPython
import warnings
from math import sqrt, ceil
from geopy.geocoders import Nominatim
from sklearn.metrics import mean_squared_error, mean_absolute_error
print('Import done')

# Seed numpy
numpy.random.seed(1234)

# turn off warnings
warnings.filterwarnings("ignore")
print('Init done')

# DATA IMPORTATION
crimes2007 = pandas.read_csv('./Chicago_Crimes_2005_to_2007.csv', error_bad_lines=False)
crimes2011 = pandas.read_csv('./Chicago_Crimes_2008_to_2011.csv', error_bad_lines=False)
crimes2017 = pandas.read_csv('./Chicago_Crimes_2012_to_2017.csv', error_bad_lines=False)
data = pandas.concat([crimes2007, crimes2011, crimes2017], ignore_index=False, axis=0)
print('Head: \n' + str(data.head()))
print('Types: \n' + str(data.dtypes))
print('Rows: ' + str(len(data)))
print('Importation done')

# PRE-PROCESSING
# 1. missing values
pyplot.figure(figsize=(12, 6))
seaborn.heatmap(data.isnull(), cbar=False, cmap='viridis')
pyplot.show()
print('Missing values: \n' + str(data.isna().mean().round(5) * 100))
# drop rows with null values
data = data.dropna()

# 2. duplicate data
pyplot.figure(figsize=(12, 6))
seaborn.heatmap(data.duplicated(subset=['ID'], keep='first').to_frame(), cbar=False, cmap='viridis')
pyplot.show()
# drop duplicates
data.drop_duplicates(keep='first', inplace=True)
print('Pre-processing done')

# Gather chicago location data
chicago = Nominatim(user_agent='my-application').geocode("Chicago Illinois")
print('Chicago location GET done')

# Generate map function
def count_crime_for_year_and_map(ward_data, year, calculate=False):
    # calculate incidents if needed
    if (calculate):
        d_year = ward_data[ward_data["Year"] == int(year)]
        ward_data = pandas.DataFrame(d_year['Ward'].value_counts().astype(float))
        ward_data.to_json('Ward_Map.json')
        ward_data = ward_data.reset_index()
        ward_data.columns = ['ward', 'count']
    
    #  make sure were using string keys and float results
    ward_data['ward'] = ward_data['ward'].astype('int').astype('string')
    ward_data['count'] = ward_data['count'].astype('Float64')

    # create choropleth
    m = folium.Map(location=[chicago.latitude, chicago.longitude], zoom_start=10, tiles='Stamen Toner')
    m.choropleth(geo_data='./ward-boundaries.geojson',
                 data=ward_data,
                 columns=['ward', 'count'],
                 key_on='feature.properties.ward',
                 fill_color='Reds',
                 fill_opacity=0.7,
                 line_opacity=0.2,
                 threshold_scale=[0, 4000, 8000, 12000, 16000, 20000],
                 legend_name='Reports per police ward ' + year
                 )
    m.save(year + '-map.html')
    return m
print('Mapping function done')

# Generate maps for analysis
count_crime_for_year_and_map(data, '2013', calculate=True)
count_crime_for_year_and_map(data, '2014', calculate=True)
count_crime_for_year_and_map(data, '2015', calculate=True)
count_crime_for_year_and_map(data, '2016', calculate=True)
print('Mapping done')

# tuning and evaluation model function setups
# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model_for_error_average(input_data, p, d, q):
    # set up index
    input_data.Date = pandas.to_datetime(data.Date, format='%m/%d/%Y %I:%M:%S %p')
    input_data.index = pandas.DatetimeIndex(data.Date)
        
    # create error results list
    errors = list()
    
    # iterate
    for i, ward in enumerate(input_data.Ward.unique()):
        # feature engineering: get police ward specfic data
        ward_data = input_data[input_data["Ward"] == ward]
        ward_data_by_month = ward_data.resample('M').size().reset_index()
        ward_data_by_month.columns = ['Date', str(ward)]
        ward_data_by_month_frame = pandas.DataFrame(ward_data_by_month)
        
        # split into training and testing
        train_split = int(len(ward_data_by_month_frame) * 0.70)
        train_data, test_data = ward_data_by_month_frame[0:train_split], ward_data_by_month_frame[train_split:]
        
        # create arima and fit model
        model = pyflux.ARIMA(data=train_data, ar=p, integ=d, ma=q, target=str(ward), family=pyflux.Normal())
        x = model.fit("MLE")
#         x.summary()
        prediction = model.predict(h=len(test_data.index), intervals=False)
#         prediction[str(float(ward))] = prediction[str(float(ward))].round()
#         print(prediction)
#         print(test_data.drop('Date', axis=1))
        
        # calculate MSE, RMSE, and add error to errors average list
        MSE = mean_squared_error(test_data.drop('Date', axis=1), prediction)
        RMSE = numpy.sqrt(MSE)
        MAE = mean_absolute_error(test_data.drop('Date', axis=1), prediction)
        print(MAE)
        errors.append(RMSE)
    
    # create average MSE from all 50 wards
    error_average = sum(errors) / float(len(errors))
    return error_average 
    
# method testing
print(evaluate_arima_model_for_error_average(data, p=4, d=0, q=4))

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(data, p_values, d_values, q_values):
    best_error, best_config = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                average_mse = evaluate_arima_model_for_error_average(data, p, d, q)
                if average_mse < best_error: best_error, best_config = average_mse, [p, d, q]
                print('ARIMA config: %s with average MSE of %.3f' % ([p, d, q], average_mse))
    print('[BEST] ARIMA config: %s with average MSE of %.3f' % (best_config, best_error))
    
# evaluate models with varying arima values to facilitate grid search tuning
evaluate_models(data, p_values=range(1, 20), d_values=range(0, 4), q_values=range(1, 20))
print('evaluation done')

def extract_wards_and_forecast_all_wards(input_data, months_to_forecast, p=4, d=0, q=4):
    # set up index
    input_data.Date = pandas.to_datetime(data.Date, format='%m/%d/%Y %I:%M:%S %p')
    input_data.index = pandas.DatetimeIndex(data.Date)
    
    # create results output frame
    results = pandas.DataFrame(index=range(0, 49), columns=['ward', 'count'])
    
    # iterate
    for i, ward in enumerate(input_data.Ward.unique()):
        # feature engineering: get police ward specfic data
        wd = input_data[input_data["Ward"] == ward]
        wdm = wd.resample('M').size().reset_index()
        wdm.columns = ['Date', str(ward)]
        frame = pandas.DataFrame(wdm)
        
        # create arima model
        model = pyflux.ARIMA(data=frame, ar=p, integ=d, ma=q, target=str(ward), family=pyflux.Normal())
        x = model.fit("MLE")
        x.summary()
        
        # plot predict
        model.plot_fit(figsize=(15,10))
        model.plot_predict(h=months_to_forecast, figsize=(15, 5), past_values=190)
        prediction = model.predict(months_to_forecast, intervals=False)
        next_year_predicted_sum = prediction.tail(12)[ward].sum()        
        print('[OUTPUT] WARD ' + str(ward) + ': ' + str(next_year_predicted_sum) + ' crimes')
        
        # set ward year prediction to results
        results.loc[i] = [int(ward), float(next_year_predicted_sum)]
#         results.append({'ward': str(w), 'count': int(predicted_year[str(w)].sum())}, ignore_index=True)
        break
#     print(results)
    return results
    
forecast = extract_wards_and_forecast_all_wards(data, months_to_forecast=24, p=1, d=0, q=4)
print('Forecasting done')


count_crime_for_year_and_map(forecast, '2017', calculate=False)
print('Done mapping')