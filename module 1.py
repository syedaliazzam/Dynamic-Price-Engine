# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 12:07:14 2022

@author: Ali Azzam
"""  

import h3
from shapely.geometry import Polygon, Point
import shapely.wkt
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import folium
from geojson import Feature, Point, FeatureCollection
import json
import h3
import numpy as np

df = pd.read_csv('H30DayBikeTaxi.csv',sep=',', converters = {'pickup_lat':float(),'pickup_lng':float()}, encoding = 'latin-1' )

resolution = 8

hex_ids = df.apply(lambda row: h3.geo_to_h3(row["pickup_lat"], row["pickup_lng"], resolution), axis = 1)

df = df.assign(hex_id=hex_ids.values)

# df.to_csv('H30DayDeliveryWithHexes.csv', sep=',', index=False)

df2 = df[(df['TripDate'] == '2023-01-14')]
# df3 = df2[(df2['TripHour'] == 15)]

# df2 = df[(df['status'] == 'Cancelled')]


dfbyhexid = (df2
            .groupby(['TripHour'])
            .trip_id
            .agg(list)
            .to_frame("ids")
            .reset_index())
# Let's count each points inside the hexagon

dfbyhexid['count'] =(dfbyhexid['ids']
                      .apply(lambda trip_id:len(trip_id)))

dfbyhexid.drop('ids', axis=1, inplace=True)
# dfbyhexid.to_csv('H30DayDeliveryGrossBookingsByHours14th.csv', sep=',', index=False)




dfbyhexidf = (df2
            .groupby(['TripHour'])
            .StatusEncoded
            .agg(list)
            .to_frame("ids")
            .reset_index())
# Let's count each points inside the hexagon

# Let's count each points inside the hexagon

dfbyhexidf.dtypes
dfbyhexidf['NetTrips'] =(dfbyhexidf['ids']
                        .apply(lambda StatusEncoded: sum(StatusEncoded)))

# Let's count each points inside the hexagon
dfbyhexidf['GrossTrips'] =(dfbyhexidf['ids']
                      .apply(lambda trip_id:len(trip_id)))


dfbyhexidf['Fulfillment%'] = (dfbyhexidf['NetTrips']/dfbyhexidf['GrossTrips'])*100
                                                                          

dfbyhexidf.drop('ids', axis=1, inplace=True)


--------

df4 = df2[(df2['status'] == 'cancelled')]


dfbyhexidC = (df4
            .groupby(['TripHour'])
            .trip_id
            .agg(list)
            .to_frame("ids")
            .reset_index())
# Let's count each points inside the hexagon

dfbyhexidC['CancelledTrips'] =(dfbyhexidC['ids']
                      .apply(lambda trip_id:len(trip_id)))

dfbyhexidC.drop('ids', axis=1, inplace=True)
# dfbyhexid.to_csv('H30DayDeliveryGrossBookingsByHours14th.csv', sep=',', index=False)


Rankings = pd.merge(dfbyhexid,
                    dfbyhexidf,
                    left_on='TripHour',
                    right_on='TripHour',
                    how='left')


Rankingss = pd.merge(Rankings,
                    dfbyhexidC,
                    left_on='TripHour',
                    right_on='TripHour',
                    how='left')



Rankingss.to_csv('SawariHourly14thJan.csv', sep=',', index=False)


##################################Demand Map####################################################


df2 = df[(df['TripDate'] == '2023-01-14')]
# df3 = df2[(df2['TripHour'] == 15)]

dfbyhexid = (df
            .groupby(['TripDate'])
            .trip_id
            .agg(list)
            .to_frame("ids")
            .reset_index())
# Let's count each points inside the hexagon

dfbyhexid['count'] =(dfbyhexid['ids']
                      .apply(lambda trip_id:len(trip_id)))

dfbyhexid.drop('ids', axis=1, inplace=True)

#dfbyhexid.to_csv('H30DayDeliveryGrossBookingsByHours14th.csv', sep=',', index=False)

Rankings.to_csv('14thJanBookingsAndFulfillment.csv', sep=',', index=False)





###

from shapely.geometry import Polygon
def add_geometry(row):
  points = h3.h3_to_geo_boundary(row['hex_id'], True)
  return Polygon(points)
#Apply function into our dataframe
dfbyhexid['geometry'] = (dfbyhexid
                                .apply(add_geometry,axis=1))


from geojson import Feature, Point, FeatureCollection, Polygon

def hexagons_dataframe_to_geojson(df_hex, hex_id_field,geometry_field, value_field,file_output = None):

    list_features = []

    for i, row in df_hex.iterrows():
        feature = Feature(geometry = row[geometry_field],
                          id = row[hex_id_field],
                          properties = {"value": row[value_field]})
        list_features.append(feature)

    feat_collection = FeatureCollection(list_features)

    if file_output is not None:
        with open(file_output, "w") as f:
            json.dump(feat_collection, f)

    else :
      return feat_collection
  
    
  
geojson_obj = (hexagons_dataframe_to_geojson
                (dfbyhexid,
                 hex_id_field='hex_id',
                 value_field='count',
                 geometry_field='geometry'))

import plotly.express as px

import plotly.graph_objects as go
import plotly.io as pio
import plotly


fig = px.choropleth_mapbox(
                    dfbyhexid, 
                    geojson=geojson_obj, 
                    locations='hex_id', 
                    color='count',
                    color_continuous_scale=px.colors.sequential.Plasma,
                    range_color=(0,dfbyhexid['count'].max()),                  
                    mapbox_style='carto-positron',
                    zoom=12,
                    center = {"lat": 24.8615, "lon": 67.0099},
                    opacity=0.7,
                    labels={'count':'# of GrossBookings 5pm '})
    
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
plotly.offline.plot(fig, filename='5pmGB.html')



##################################Fulfillment Map####################################################

df2 = df[(df['TripDate'] == '2023-01-14')]
df3 = df2[(df2['TripHour'] == 17)]

dfbyhexid = (df3
            .groupby(['hex_id'])
            .StatusEncoded
            .agg(list)
            .to_frame("ids")
            .reset_index())
# Let's count each points inside the hexagon

# Let's count each points inside the hexagon

dfbyhexid.dtypes
dfbyhexid['NetTrips'] =(dfbyhexid['ids']
                        .apply(lambda StatusEncoded: sum(StatusEncoded)))

# Let's count each points inside the hexagon
dfbyhexid['GrossTrips'] =(dfbyhexid['ids']
                      .apply(lambda trip_id:len(trip_id)))


dfbyhexid['count'] = (dfbyhexid['NetTrips']/dfbyhexid['GrossTrips'])*100
                                                                          

dfbyhexid.drop('ids', axis=1, inplace=True)
# dfbyhexid.to_csv('H30DayDeliveryGrossBookingsByPickup_zone.csv', sep=',', index=False)




from shapely.geometry import Polygon
def add_geometry(row):
  points = h3.h3_to_geo_boundary(row['hex_id'], True)
  return Polygon(points)
#Apply function into our dataframe
dfbyhexid['geometry'] = (dfbyhexid
                                .apply(add_geometry,axis=1))


from geojson import Feature, Point, FeatureCollection, Polygon

def hexagons_dataframe_to_geojson(df_hex, hex_id_field,geometry_field, value_field,file_output = None):

    list_features = []

    for i, row in df_hex.iterrows():
        feature = Feature(geometry = row[geometry_field],
                          id = row[hex_id_field],
                          properties = {"value": row[value_field]})
        list_features.append(feature)

    feat_collection = FeatureCollection(list_features)

    if file_output is not None:
        with open(file_output, "w") as f:
            json.dump(feat_collection, f)

    else :
      return feat_collection
  
    
  
geojson_obj = (hexagons_dataframe_to_geojson
                (dfbyhexid,
                 hex_id_field='hex_id',
                 value_field='count',
                 geometry_field='geometry'))

import plotly.express as px

import plotly.graph_objects as go
import plotly.io as pio
import plotly


fig = px.choropleth_mapbox(
                    dfbyhexid, 
                    geojson=geojson_obj, 
                    locations='hex_id', 
                    color='count',
                    color_continuous_scale=px.colors.sequential.Plasma,
                    range_color=(0,dfbyhexid['count'].max()),                  
                    mapbox_style='carto-positron',
                    zoom=12,
                    center = {"lat": 24.8615, "lon": 67.0099},
                    opacity=0.7,
                    labels={'count':'% of FF 5pm '})
    
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
plotly.offline.plot(fig, filename='5pmFF.html')













-------------------------------------------------------

from datetime import datetime


def extract_hour(date_string):
    date_object = datetime.strptime(date_string, '%H:%M:%S.%M')
    return date_object.hour

# Apply the function to the date column
df['hour'] = df['TripTime'].apply(extract_hour)


df['TripTime'] = pd.to_datetime(df['TripTime'], format='%H:%M:%S.%f')

# Extract the hour from the datetime column
df['hour'] = df['time'].dt.hour

# Print the resulting DataFrame
print(df)








rides_by_hex_and_hour = df.groupby(['hex_id', 'TripHour']).size().reset_index(name='counts')

# Plot the hexagons on a map
for hour in range(24):
    # Get the hexagons for this hour
    hexagons = rides_by_hex_and_hour[rides_by_hex_and_hour['TripHour'] == hour]
    # Plot the hexagons
    plt.scatter(hexagons['hex_id'], hexagons['counts'], label=f'Hour {hour}')

# Add a legend and show the plot
plt.legend()
plt.show()

























##################################Supply Map####################################################

df = df.dropna('driver_id', axis = 1)

df = df.dropna(subset=['driver_id'])



dfbyhexid = (df
            .groupby('hex_id')
            .driver_id
            .agg(list)
            .to_frame("ids")
            .reset_index())

# Let's count each points inside the hexagon
dfbyhexid['count'] =(dfbyhexid['ids']
                      .apply(lambda driver_id:len(driver_id)))


from shapely.geometry import Polygon
def add_geometry(row):
  points = h3.h3_to_geo_boundary(row['hex_id'], True)
  return Polygon(points)
#Apply function into our dataframe
dfbyhexid['geometry'] = (dfbyhexid
                                .apply(add_geometry,axis=1))


from geojson import Feature, Point, FeatureCollection, Polygon

def hexagons_dataframe_to_geojson(df_hex, hex_id_field,geometry_field, value_field,file_output = None):

    list_features = []

    for i, row in df_hex.iterrows():
        feature = Feature(geometry = row[geometry_field],
                          id = row[hex_id_field],
                          properties = {"value": row[value_field]})
        list_features.append(feature)

    feat_collection = FeatureCollection(list_features)

    if file_output is not None:
        with open(file_output, "w") as f:
            json.dump(feat_collection, f)

    else :
      return feat_collection
  
    
  
geojson_obj = (hexagons_dataframe_to_geojson
                (dfbyhexid,
                 hex_id_field='hex_id',
                 value_field='count',
                 geometry_field='geometry'))

import plotly.express as px

import plotly.graph_objects as go
import plotly.io as pio
import plotly


fig = px.choropleth_mapbox(
                    dfbyhexid, 
                    geojson=geojson_obj, 
                    locations='hex_id', 
                    color='count',
                    color_continuous_scale=px.colors.sequential.Plasma,
                    range_color=(0,dfbyhexid['count'].mean()),                  
                    mapbox_style='carto-positron',
                    zoom=12,
                    center = {"lat": 24.8615, "lon": 67.0099},
                    opacity=0.7,
                    labels={'count':'# of Drivers '})
    
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
plotly.offline.plot(fig, filename='Evening6_7SupplyMap.html')




##################################Fulfillment Map####################################################


dfbyhexid = (df
            .groupby('hex_id')
            .StatusEncoded
            .agg(list)
            .to_frame("ids")
            .reset_index())

# Let's count each points inside the hexagon


dfbyhexid['NetTrips'] =(dfbyhexid['ids']
                      .apply(lambda StatusEncoded: sum(StatusEncoded)))

# Let's count each points inside the hexagon
dfbyhexid['GrossTrips'] =(dfbyhexid['ids']
                      .apply(lambda passenger_id:len(passenger_id)))


dfbyhexid['count'] = (dfbyhexid['NetTrips']/dfbyhexid['GrossTrips'])*100
                                                                          

from shapely.geometry import Polygon
def add_geometry(row):
  points = h3.h3_to_geo_boundary(row['hex_id'], True)
  return Polygon(points)
#Apply function into our dataframe
dfbyhexid['geometry'] = (dfbyhexid
                                .apply(add_geometry,axis=1))


from geojson import Feature, Point, FeatureCollection, Polygon

def hexagons_dataframe_to_geojson(df_hex, hex_id_field,geometry_field, value_field,file_output = None):

    list_features = []

    for i, row in df_hex.iterrows():
        feature = Feature(geometry = row[geometry_field],
                          id = row[hex_id_field],
                          properties = {"value": row[value_field]})
        list_features.append(feature)

    feat_collection = FeatureCollection(list_features)

    if file_output is not None:
        with open(file_output, "w") as f:
            json.dump(feat_collection, f)

    else :
      return feat_collection
  
    
  
geojson_obj = (hexagons_dataframe_to_geojson
                (dfbyhexid,
                 hex_id_field='hex_id',
                 value_field='count',
                 geometry_field='geometry'))

import plotly.express as px

import plotly.graph_objects as go
import plotly.io as pio
import plotly


fig = px.choropleth_mapbox(
                    dfbyhexid, 
                    geojson=geojson_obj, 
                    locations='hex_id', 
                    color='count',
                    color_continuous_scale=px.colors.sequential.Plasma,
                    range_color=(0,dfbyhexid['count'].max()),                  
                    mapbox_style='carto-positron',
                    zoom=12,
                    center = {"lat": 24.8615, "lon": 67.0099},
                    opacity=0.7,
                    labels={'Fulfillment %'})
    
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
plotly.offline.plot(fig, filename='Evening15Day6_7FulfillmentMap.html')

dfbyhexid.to_csv('Evening1Day6_7FulfillmentMap.csv', sep=',', index=False)







-------------------------------------------------------------------------------

import h3
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from pandas.io.json import json_normalize



# Load the trip data into a Pandas DataFrame
# df = pd.read_csv('trips.csv')

df = pd.read_csv('Delivery10Day.csv',sep=',', converters = {'pickup_lat':float(),'pickup_lng':float()}, encoding = 'latin-1' )


# Add a new column to the DataFrame with the H3 index for each trip
df['h3_index'] = df.apply(lambda row: h3.geo_to_h3(row['pickup_lat'], row['pickup_lng'], 8), axis=1)

# Group the data by H3 index and hour
# grouped = df.groupby(['h3_index', 'TripHour']).size().reset_index(name='count')

df['TripHour'] = df['TripHour'].astype(str)

grouped = df.groupby(['h3_index', 'TripDate', 'TripHour']).size().reset_index(name='count')
# grouped = df.groupby(['h3_index', 'TripDate', 'TripHour'], as_index=False).size().reset_index(name='count')


# grouped = df.groupby(['h3_index', 'TripDate', 'TripHour'], as_index=False).size()
# grouped = grouped.reset_index().rename(columns={0:'count'})





# Create an empty figure
fig = go.Figure()


for hour in grouped['TripHour'].unique():
    hex_data = grouped[grouped['TripHour'] == hour]
    hex_data = hex_data.groupby('h3_index').sum()
    hex_data = hex_data.reset_index()
    hex_data['coordinates'] = hex_data['h3_index'].apply(lambda x: h3.h3_to_geo_boundary(x))
    hex_data = hex_data.explode('coordinates')
    hex_data = hex_data.rename(columns={'h3_index': 'index'})
    hex_data = hex_data.groupby('coordinates').sum()
    hex_data = hex_data.reset_index()
    hex_data = hex_data.rename(columns={'coordinates': 'latlng'})
    hex_data['lat'] = hex_data['latlng'].apply(lambda x: x[0])
    hex_data['lon'] = hex_data['latlng'].apply(lambda x: x[1])
    hex_data = hex_data.drop(columns=['latlng'])
    fig.add_trace(go.Scattermapbox(
        lon=hex_data['lon'],
        lat=hex_data['lat'],
        mode='markers',
        # fill='toself',
        marker=go.scattermapbox.Marker(
            size=8, color='blue'
        ),
        text=hex_data['count'],
        # text=hex_data['h3_index'] + ": " + hex_data['count'].astype(str),
        name=str(hour)
    ))



# Update the layout of the figure
fig.update_layout(
    mapbox=go.layout.Mapbox(
        accesstoken='pk.eyJ1IjoiZ2VvcmdlYXZlciIsImEiOiJjazlzYjg3eW8wNnJlM2RvYXl4dW5hNjJjIn0.2QKjSZW8SZGv-LbXjK9XaA',
        style='carto-positron',
        zoom=6,
        center=go.layout.mapbox.Center(
            lat=24.8615,
            lon=-67.0099
        )
    ),
    updatemenus=[
        go.layout.Updatemenu(
            type='buttons',
            buttons=[go.layout.updatemenu.Button(
                label=str(i),
                method='update',
                args=[{'visible': [i==j for j in grouped['TripHour'].unique()]},
                      {'title': 'Hour '+str(i)}]) for i in grouped['TripHour'].unique()])
    ]
    )


fig.show(renderer="notebook")
fig.write_html("30dayplot.html")



from plotly.offline import plot
plot(fig, auto_open=True)




# Loop through the hours in the data and add a trace for each hour
for hour in grouped['TripHour'].unique():
    hex_data = grouped[grouped['TripHour'] == hour]
    hex_data['h3_index'] = hex_data['h3_index'].apply(lambda x: h3.h3_to_geo_boundary(x))
    hex_data['pickup_lat'] = hex_data['h3_index'].apply(lambda x: x[0][0])
    hex_data['pickup_lng'] = hex_data['h3_index'].apply(lambda x: x[0][1])
    fig.add_trace(go.Scattermapbox(
        lon=hex_data['pickup_lng'],
        lat=hex_data['pickup_lat'],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=10, color='blue'
        ),
        text=hex_data['count'],
        name=str(hour)
    ))
    
    
---------------    
    
for hour in grouped['TripHour'].unique():
    hex_data = grouped[grouped['TripHour'] == hour]
    # Group data by TripDate
    for date in hex_data['TripDate'].unique():
        hex_data_by_date = hex_data[hex_data['TripDate'] == date]
        hex_data_by_date = hex_data_by_date.groupby('h3_index').sum()
        hex_data_by_date = hex_data_by_date.reset_index()
        hex_data_by_date['coordinates'] = hex_data_by_date['h3_index'].apply(lambda x: h3.h3_to_geo_boundary(x))
        hex_data_by_date = hex_data_by_date.explode('coordinates')
        hex_data_by_date = hex_data_by_date.rename(columns={'h3_index': 'index'})
        hex_data_by_date = hex_data_by_date.groupby('coordinates').sum()
        hex_data_by_date = hex_data_by_date.reset_index()
        hex_data_by_date = hex_data_by_date.rename(columns={'coordinates': 'latlng'})
        hex_data_by_date['lat'] = hex_data_by_date['latlng'].apply(lambda x: x[0])
        hex_data_by_date['lon'] = hex_data_by_date['latlng'].apply(lambda x: x[1])
        hex_data_by_date = hex_data_by_date.drop(columns=['latlng'])
        fig.add_trace(go.Scattermapbox(
            lon=hex_data_by_date['lon'],
            lat=hex_data_by_date['lat'],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=8, color='blue'
            ),
            text=hex_data_by_date['count'],
            name=str(date) + '-' + str(hour)
        ))
        
        
        
--------------------
for hour in grouped['TripHour'].unique():
    hex_data = grouped[grouped['TripHour'] == hour]
    df['count'] = 1
    hex_data = df.groupby(['h3_index','TripDate','TripHour'], as_index=False)['count'].sum()
    hex_data = hex_data.groupby('h3_index').sum().reset_index()
    hex_data = hex_data[['h3_index', 'count']]
    hex_data['coordinates'] = hex_data['h3_index'].apply(lambda x: h3.h3_to_geo_boundary(x))
    hex_data = hex_data.explode('coordinates')
    hex_data = hex_data.rename(columns={'h3_index': 'index'})
    hex_data = hex_data.groupby('coordinates').sum().reset_index()
    hex_data = hex_data.rename(columns={'coordinates': 'latlng'})
    hex_data['lat'] = hex_data['latlng'].apply(lambda x: x[0])
    hex_data['lon'] = hex_data['latlng'].apply(lambda x: x[1])
    hex_data = hex_data.drop(columns=['latlng'])
    fig.add_trace(go.Scattermapbox(
        lon=hex_data['lon'],
        lat=hex_data['lat'],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=8, color='blue'
        ),
        text=hex_data[['h3_index', 'count']].apply(lambda x: f"{x[0]}: {x[1]}", axis=1),
        name=str(hour)
    ))

fig.update_layout(
            mapbox=go.layout.Mapbox(
            accesstoken='pk.eyJ1IjoiZ2VvcmdlYXZlciIsImEiOiJjazlzYjg3eW8wNnJlM2RvYXl4dW5hNjJjIn0.2QKjSZW8SZGv-LbXjK9XaA',
            style='carto-positron',
            zoom=6,
            center=go.layout.mapbox.Center(
                lat=24.8615,
                lon=-67.0099
                )
            ),
            updatemenus=[
                go.layout.Updatemenu(
                    type='dropdown',
                    buttons= [dict(label=str(date),
                                   method='update',
                                   args=[{'visible': [date == j for j in grouped['TripDate'].unique()]},
                                         {'title': 'Date: '+str(date)}]) for date in grouped['TripDate'].unique()],
                    ),
                go.layout.Updatemenu(
                    type='dropdown',
                    buttons= [dict(label=str(hour),
                                   method='update',
                                   args=[{'visible': [hour == j for j in grouped['TripHour'].unique()]},
                                         {'title': 'Hour: '+str(hour)}]) for hour in grouped['TripHour'].unique()],
                    ),
                ]
            
            )   



fig.show(renderer="notebook")
fig.write_html("plot.html")

from plotly.offline import plot
plot(fig, auto_open=True)




###Predictions


# import necessary libraries
import pandas as pd
from prophet import Prophet
import h3



# load data into a DataFrame
df = pd.read_csv('1Day.csv')

df =df.head(100)

# Add a new column to the DataFrame with the H3 index for each trip
df['h3_index'] = df.apply(lambda row: h3.geo_to_h3(row['pickup_lat'], row['pickup_lng'], 8), axis=1)

df['TripHour'] = df['TripHour'].astype(str)

df = df.groupby(['h3_index', 'created_at']).size().reset_index(name='count')


# convert 'created_at' column to datetime and set it as the index
df['created_at'] = pd.to_datetime(df['created_at'])
df = df.set_index('created_at')

# group dataframe by h3_index and resample to hourly frequency
df = df.groupby('h3_index').resample('H').sum()

df = df.groupby('h3_index').filter(lambda x: x['count'].count() >= 2)


# create a new dataframe with columns 'ds' and 'y' for prophet
df_prophet = df.reset_index()
df_prophet = df_prophet.rename(columns={'created_at': 'ds', 'count': 'y'})
df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)


# loop through each h3_index
for h3_index in df_prophet['h3_index'].unique():
    # filter dataframe by h3_index
    df_h3 = df_prophet[df_prophet['h3_index'] == h3_index]
    
    # df_h3['y'] = np.log(df_h3['y'])

    
    # initialize and fit prophet model
    m = Prophet()
    m.fit(df_h3)
    
    # make hourly predictions for the next 7 days
    future = m.make_future_dataframe(periods=9*24, freq='H')
    forecast = m.predict(future)
    
    # print the predictions for h3_index
    print("Predictions for h3_index: ", h3_index)
    print(forecast[['ds', 'yhat']][9*24:])




predictions = forecast[['ds', 'yhat']]
predictions['h3_index'] = h3_index
predictions = predictions.rename(columns={'yhat': 'count'})

predictions_df = pd.DataFrame()



if 'predictions' not in locals():
    predictions_df = predictions.copy()
else:
    predictions_df = predictions_df.append(predictions)




predictions_df.to_csv('MobilityDemandPredictions.csv', index=False)






########Prediction2

from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# load your data
df = pd.read_csv('1Day.csv')

# handling missing data and outliers
df.fillna(df.mean(), inplace=True)

# Add a new column to the DataFrame with the H3 index for each trip
df['h3_index'] = df.apply(lambda row: h3.geo_to_h3(row['pickup_lat'], row['pickup_lng'], 8), axis=1)

df['TripHour'] = df['TripHour'].astype(str)

df = df.groupby(['h3_index', 'created_at', 'TripHour']).size().reset_index(name='count')

# convert 'created_at' column to datetime and set it as the index
df['created_at'] = pd.to_datetime(df['created_at'])
df = df.set_index(['created_at','TripHour'])

# loop through each h3_index
for h3_index in df['h3_index'].unique():
    # filter dataframe by h3_index
    df_h3 = df[df['h3_index'] == h3_index]
    df_h3 = df_h3.groupby(['created_at', 'TripHour']).sum()
    df_h3 = df_h3.reset_index()
    df_h3 = df_h3.set_index('created_at')
    df_h3 = df_h3['count']
    # fit the ARIMA model
    model = ARIMA(df_h3, order=(1,1,1))
    model_fit = model.fit()
    print(f'Predictions for h3_index: {h3_index}')
    print(model_fit.summary())

                           
                          
# make predictions
future_predictions = model_fit.forecast(steps=9*24)
forecast = future_predictions[0]
print(forecast)

                           

                           

                           


#########Plott

predictions_df = pd.read_csv("10Days.csv")

import plotly.express as px
import h3

# Convert h3_index to latitude and longitude coordinates
lat_long = predictions_df['h3_index'].apply(h3.h3_to_geo_boundary)
predictions_df['latitude'] = [point[0][1] for point in lat_long]
predictions_df['longitude'] = [point[0][0] for point in lat_long]


predictions_df['count'] = predictions_df['count'].clip(lower=0)

import folium
from folium.plugins import HeatMap

lat=24.8615
lon=-67.0099

# Create a map centered on the mean of the longitude and latitude values

map = folium.Map(location=[lat, lon], zoom_start=13)

# Create the heatmap layer
heatmap = HeatMap(data=predictions_df[['latitude', 'longitude', 'count']].values.tolist(),
                  name='Predicted Demand',
                  overlay=True,
                  control=False)
heatmap.add_to(map)

# Add a layer control to allow toggling of the heatmap
folium.LayerControl().add_to(map)

# Display the map
map.save("predicted_demand_map.html")



predictions_df.dtypes

print(f'Number of data points: {len(predictions_df)}')
print(f'Number of data points within range: {len(predictions_df[(predictions_df["longitude"] >= -180) & (predictions_df["longitude"] <= 180) & (predictions_df["latitude"] >= -90) & (predictions_df["latitude"] <= 90)])}')

print(predictions_df['longitude'].min(), predictions_df['longitude'].max())
print(predictions_df['latitude'].min(), predictions_df['latitude'].max())

predictions_df.isnull().sum()

-------------------------------

from shapely.geometry import Polygon
def add_geometry(row):
  points = h3.h3_to_geo_boundary(row['h3_index'], True)
  return Polygon(points)
#Apply function into our dataframe
predictions_df['geometry'] = (predictions_df
                                .apply(add_geometry,axis=1))


from geojson import Feature, Point, FeatureCollection, Polygon

def hexagons_dataframe_to_geojson(df_hex, hex_id_field,geometry_field, value_field,file_output = None):

    list_features = []

    for i, row in df_hex.iterrows():
        feature = Feature(geometry = row[geometry_field],
                          id = row[hex_id_field],
                          properties = {"value": row[value_field]})
        list_features.append(feature)

    feat_collection = FeatureCollection(list_features)

    if file_output is not None:
        with open(file_output, "w") as f:
            json.dump(feat_collection, f)

    else :
      return feat_collection
  
    
  
geojson_obj = (hexagons_dataframe_to_geojson
                (predictions_df,
                 hex_id_field='h3_index',
                 value_field='count',
                 geometry_field='geometry'))

import plotly.express as px

import plotly.graph_objects as go
import plotly.io as pio
import plotly


fig = px.choropleth_mapbox(
                    predictions_df, 
                    geojson=geojson_obj, 
                    locations='h3_index', 
                    color='count',
                    color_continuous_scale=px.colors.sequential.Plasma,
                    range_color=(0,predictions_df['count'].max()),                  
                    mapbox_style='carto-positron',
                    zoom=12,
                    center = {"lat": 24.8615, "lon": 67.0099},
                    opacity=0.7,
                    labels={'count':'# of passengers '})
    
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})


from plotly.offline import plot
plot(fig, auto_open=True)


---
from datetime import datetime

predictions_df['ds'] = pd.to_datetime(predictions_df['ds'])
predictions_df.dtypes
fig = px.choropleth_mapbox(
    predictions_df, 
    geojson=geojson_obj, 
    locations='h3_index', 
    color='count',
    color_continuous_scale=px.colors.sequential.Plasma,
    range_color=(0,predictions_df['count'].max()),                  
    mapbox_style='carto-positron',
    zoom=12,
    center = {"lat": 24.8615, "lon": 67.0099},
    opacity=0.7,
    labels={'count':'# of passengers '}
)
fig.update_layout(
    updatemenus=[
        dict(
            type = "buttons",
            buttons=[
                dict(
                    args=[{"visible": [True, False, False]}],
                    label="All",
                    method="update"
                ),
                dict(
                    args=[{"visible": [False, True, False]}],
                    label="Morning",
                    method="update"
                ),
                dict(
                    args=[{"visible": [False, False, True]}],
                    label="Evening",
                    method="update"
                )
            ],
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.1,
            xanchor="left",
            y=1.1,
            yanchor="top"
        ),
    ]
)



plotly.offline.plot(fig, filename='map.html')


from plotly.offline import plot
plot(fig, auto_open=True)


morning_mask = (predictions_df['ds'].dt.hour >= 6) & (predictions_df['ds'].dt.hour < 12)
evening_mask = (predictions_df['ds'].dt.hour >= 18) & (predictions_df['ds'].dt.hour < 24)

fig.data[0].visible = True
fig.data[1].visible = morning_mask
fig.data[2].visible = evening_mask





