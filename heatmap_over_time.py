#!/usr/bin/env python3

import os
import folium
import numpy as np
import pandas as pd
from datetime import date
import random
from folium.plugins import HeatMapWithTime, MiniMap, Fullscreen, MarkerCluster
from folium.map import LayerControl
import warnings
warnings.filterwarnings("ignore")
# import seaborn as sns; sns.set()
import datetime
import mgrs
import jinja2
os.chdir(rf'{os.getcwd()[0]}:\Users\brend\Documents\Coding Projects\CEMA')
from main import get_coords_from_LOBS,get_line,organize_polygon_coords,get_polygon_area,create_lob_cluster,create_marker,add_copiable_markers
from main import convert_meters_to_coordinates, convert_coords_to_mgrs

def create_map(center_coordinate):
    m = folium.Map(
        location=center_coordinate,
        tiles = None,
        zoom_start=15,
        min_zoom = 0,
        max_zoom = 18,
        control_scale = False)
    return m

def add_tilelayers(m):
    folium.TileLayer(
        tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr = 'Esri',
        name = 'Satellite',
        overlay = False,
        control = True
        ).add_to(m)
    # folium.TileLayer(
    #     tiles = 'http://10.0.0.46/tile/{z}/{x}/{y}.png',
    #     attr = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    #     name = 'Offline',
    #     overlay = False,
    #     control = True
    #     ).add_to(m)    
    return m  

def marker_multiple_coordinates(m,coords,series_name,marker_color,marker_icon,marker_prefix='fa'):
    for index,item in enumerate(coords):
        marker = create_marker(item,f'{series_name} {index+1}',marker_color,marker_icon,marker_prefix='fa')
        marker.add_to(m)
    return m

def create_synthetic_ems_data(center_points,possible_dates = [str(date.today())],variances_in_m=100,num_datapoints = 100):
    data = []
    avg_received_power = -70; received_power_variance = 15 # dBm
    datapoint_index = 1
    elements = list(range(0,24))
    probabilities = [0.02,0.02,0.02,0.02,0.02,0.03,0.05,0.05,0.06,0.07,0.07,0.07]
    probabilities += probabilities[::-1]
    hours = np.random.choice(elements,num_datapoints,p=probabilities)
    # Datetime format: [YYYY-MM-DD HH:MM:SS.SSSSSS]
    p_days = [int(dt.split('-')[1]) for dt in possible_dates]
    for d in range(num_datapoints):
        dt = f'{datetime.datetime.today().year}-07-{random.choice(p_days):01d} {int(hours[d]):02d}:00:00.000000'
        center_point = random.choice(center_points)
        lat = np.random.normal(center_point[0],convert_meters_to_coordinates(variances_in_m))
        long = np.random.normal(center_point[1],convert_meters_to_coordinates(variances_in_m))
        pwr = np.random.normal(avg_received_power,received_power_variance)
        while pwr > 0:
            pwr = np.random.normal(avg_received_power,received_power_variance)
        row = [datapoint_index,dt,[lat,long,pwr]]
        data.append(row)
        datapoint_index += 1
    data_df = pd.DataFrame(data,columns=['ID','Datetime','Heatmap Data'])
    return data_df

def normalize_weights(df_heatmap):
    max_power = max([data[-1] for data in df_heatmap['Heatmap Data']])
    for index, row in df_heatmap.iterrows():
        row['Heatmap Data'][-1] = max_power/row['Heatmap Data'][-1]
    return df_heatmap

def create_heatmap_over_time(center_points,possible_dates,name,color_gradient,overlay_bool=True):
    variance_in_m = 150
    num_datapoints = 100*len(possible_dates)*24
    df_heatmap = create_synthetic_ems_data(center_points,possible_dates,variance_in_m,num_datapoints)
    df_heatmap_normalized = normalize_weights(df_heatmap)
    lat_long_list = []
    for i in df_heatmap_normalized['Datetime'].unique():
        temp=[]
        for index, instance in df_heatmap_normalized[df_heatmap_normalized['Datetime'] == i].iterrows():
            temp.append([instance['Heatmap Data'][0],instance['Heatmap Data'][1],instance['Heatmap Data'][2]])
        lat_long_list.append(temp)
    df_heatmap_normalized['Datetime'] = pd.to_datetime(df_heatmap_normalized['Datetime'])
    time_index = []
    df_heatmap_normalized['Datetime'] = pd.to_datetime(df_heatmap_normalized['Datetime'])
    for i in df_heatmap_normalized['Datetime'].unique():
        time_index.append(i)
    date_strings = [pd.to_datetime(str(d)).strftime('%d/%m/%Y, %H:%M:%S') for d in time_index]
    date_strings.sort()
    return HeatMapWithTime(lat_long_list,index=date_strings,radius=30,name=name,gradient=color_gradient,blur=1,min_opacity=0.1,max_opacity=0.7,auto_play=True)

### GENERAL VALUES
possible_dates = [str(date.today())]
### TRUE EMITTER VALUES
true_emitter_center_points = [[32.01153936023982, -81.82645560554637],[32.01279443341273, -81.83110523326633],[32.01556074243233, -81.82412615561911]]
true_emitter_name = 'BN CP MILBAND (30-88MHz) Emissions'
true_emitter_color_gradient = {.10:'paleturquoise',.25:'aqua',.5:'blue',.75:'navy',.9:'midnightblue'}
true_emitter_hmot = create_heatmap_over_time(true_emitter_center_points,possible_dates,true_emitter_name,true_emitter_color_gradient)
### DECOY EMITTER VALUES
decoy_emitter_center_points = [[32.0206561880565, -81.82431545886503],[32.01578477054998, -81.81931777761724],[32.00877895689758, -81.82294613432661]]
decoy_emitter_name = 'Decoy MILBAND (30-88MHz) Emissions'
decoy_emitter_color_gradient = {.10:'lightyellow',.25:'yellow',.5:'orange',.75:'red',.9:'darkred'}
decoy_emitter_hmot = create_heatmap_over_time(decoy_emitter_center_points,possible_dates,decoy_emitter_name,decoy_emitter_color_gradient,overlay_bool=False) 
### ENY EW TEAMS
ewt_1_coords = [32.026822828794764, -81.81664437372622]; ewt_1_azimuth = 215; ewt_1_sensor_error = 3.5
ewt_2_coords = [32.021410278965014, -81.81082500495434]; ewt_2_azimuth = 245; ewt_2_sensor_error = 3.5
min_distance_m = 250
max_distance_m = 3000
lob1_center, lob1_near_right_coord, lob1_near_left_coord, lob1_far_right_coord, lob1_far_left_coord, lob1_far_middle_coord = get_coords_from_LOBS(ewt_1_coords,ewt_1_azimuth,ewt_1_sensor_error,min_distance_m,max_distance_m)
lob2_center, lob2_near_right_coord, lob2_near_left_coord, lob2_far_right_coord, lob2_far_left_coord, lob2_far_middle_coord = get_coords_from_LOBS(ewt_2_coords,ewt_2_azimuth,ewt_2_sensor_error,min_distance_m,max_distance_m)
lob1_center = get_line(ewt_1_coords, lob1_far_middle_coord)
lob1_right_bound = get_line(lob1_near_right_coord, lob1_far_right_coord)
lob1_left_bound = get_line(lob1_near_left_coord, lob1_far_left_coord)
lob2_center = get_line(ewt_2_coords, lob2_far_middle_coord)
lob2_right_bound = get_line(lob2_near_right_coord, lob2_far_right_coord)
lob2_left_bound = get_line(lob2_near_left_coord, lob2_far_left_coord)
points_lob1_polygon = [lob1_near_right_coord,lob1_far_right_coord,lob1_far_left_coord,lob1_near_left_coord]
points_lob1_polygon = organize_polygon_coords(points_lob1_polygon)
points_lob2_polygon = [lob2_near_right_coord,lob2_far_right_coord,lob2_far_left_coord,lob2_near_left_coord] 
points_lob2_polygon = organize_polygon_coords(points_lob2_polygon)
lob1_area = get_polygon_area(points_lob1_polygon)
lob2_area = get_polygon_area(points_lob2_polygon)
points_lob_polygons = [points_lob1_polygon,points_lob2_polygon]
azimuths = [ewt_1_azimuth,ewt_2_azimuth]
### MAP
center_points = true_emitter_center_points + decoy_emitter_center_points
center_coordinate = [np.average([x[0] for x in center_points]),np.average([x[1] for x in center_points])]
m = create_map(center_coordinate)
m = add_tilelayers(m)
true_emitter_hmot.add_to(m)
# decoy_emitter_hmot.add_to(m)
# m = marker_multiple_coordinates(m,decoy_emitter_center_points,'Decoy Emitter','purple','tower-broadcast',marker_prefix='fa')
m = marker_multiple_coordinates(m,true_emitter_center_points,'BN Emitters','blue','walkie-talkie',marker_prefix='fa')
m = marker_multiple_coordinates(m,[ewt_1_coords,ewt_2_coords],'ENY EWT','red','binoculars',marker_prefix='fa')
m = add_copiable_markers(m)
lob_cluster = MarkerCluster(name='ENY EW LOBs',show=False)
lob_cluster.add_to(m)
lob_cluster = create_lob_cluster(lob_cluster,points_lob_polygons,azimuths,ewt_1_sensor_error,min_distance_m,max_distance_m)
LayerControl(position='topright').add_to(m)
Fullscreen(position='topleft',title='Full Screen').add_to(m)
m.add_child(folium.LatLngPopup())
m.save("C:\\Users\\brend\\Documents\\Coding Projects\\CEMA\\Maps\\heatmap_over_time.html")
