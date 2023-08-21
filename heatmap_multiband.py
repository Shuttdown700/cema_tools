#!/usr/bin/env python3

import os 
import folium
import branca.colormap as cm
from collections import defaultdict
from folium import plugins
# folium icons: https://fontawesome.com/icons?d=gallery
# folium icon colors: ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']
# folium tiles: OpenStreetMap , Stamen Terrain, Stamen Toner
import jinja2
import earthpy as et
import earthpy.spatial as es
import numpy as np
import pandas as pd
from datetime import date
import random
from folium.plugins import HeatMap, HeatMapWithTime, MiniMap, Fullscreen, LocateControl
from folium.map import LayerControl
import string
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns; sns.set()
import csv
import mgrs
from math import sin, cos, tan
import haversine as hs
from haversine import Unit


# https://www.kaggle.com/code/daveianhickey/how-to-folium-for-maps-heatmaps-time-data

def create_map(center_coordinate):
    'You can pass a custom tileset to Folium by passing a xyzservices.TileProvider or a Leaflet-style URL to the tiles parameter: http://{s}.yourtiles.com/{z}/{x}/{y}.png.'
    m = folium.Map(
        location=center_coordinate,
        tiles = None, # Different "tiles" options: Stamen Toner
        zoom_start=13,
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
    folium.TileLayer(
        tiles = 'openstreetmap',
        name = 'Street Map',
        overlay = False,
        control = True
        ).add_to(m)
    folium.TileLayer(
        tiles = 'Stamen Toner',
        name = 'Black & White',
        overlay = False,
        control = True
        ).add_to(m)
    # from pymbtiles import MBtiles
    # with MBtiles(r"C:\Users\brend\Documents\Coding Projects\SDR\Maps\mbutil-master\maptiler-osm-2020-02-10-v3.11-planet.mbtiles") as src:
    #     tile_data = src.read_tile(z=0, x=0, y=0)
    # folium.TileLayer(
    #     tiles = tile_data,
    #     attr="<a href=https://endless-sky.github.io/>Endless Sky</a>",
    #     name = 'Local File',
    #     overlay = False,
    #     control = True
    #     ).add_to(m)
    return m

# def add_marker(m,marker_coords,marker_name,marker_color,marker_icon,marker_prefix='fa'):
#     folium.Marker(marker_coords,
#                   # popup = f'<input type="text" value="{location[0]}, {location[1]}" id="myInput"><button onclick="myFunction()">Copy location</button>',
#                   tooltip=marker_name,
#                   icon=folium.Icon(color=marker_color,
#                                    icon_color='White',
#                                    icon=marker_icon,
#                                    prefix=marker_prefix)
#                   ).add_to(m)
#     return m

def add_marker_with_copiable_coordinates(m,marker_coords,marker_name,marker_color,marker_icon,marker_prefix='fa'):
    folium.Marker(marker_coords,
                  # popup = f'<input type="text" value="{marker_coords[0]}, {marker_coords[1]}" id="myInput"><button onclick="myFunction()">Copy location</button>',
                  popup = f'<input type="text" value="{convert_coords_to_mgrs(marker_coords)}" id="myInput"><button onclick="myFunction()">Copy MGRS Grid</button>',
                  tooltip=marker_name,
                  icon=folium.Icon(color=marker_color,
                                   icon_color='White',
                                   icon=marker_icon,
                                   prefix=marker_prefix)
                  ).add_to(m)
    el = folium.MacroElement().add_to(m)
    el._template = jinja2.Template("""
        {% macro script(this, kwargs) %}
        function myFunction() {
          /* Get the text field */
          var copyText = document.getElementById("myInput");

          /* Select the text field */
          copyText.select();
          copyText.setSelectionRange(0, 99999); /* For mobile devices */

          /* Copy the text inside the text field */
          document.execCommand("copy");
        }
        {% endmacro %}
    """)
    return m    

def create_gradient_map(min_power,max_power,colors=['lightyellow','yellow','orange','red','darkred'],steps=20):
    # colormap = cm.linear.YlOrRd_09.scale(min_power,max_power).to_step(steps)
    # colormap.caption = "Power (dBm)"
    colormap = cm.LinearColormap(
        colors = colors,
        index = [min_power,
                 np.average([min_power,np.average([min_power,max_power])]),
                 np.average([min_power,max_power]),
                 np.average([np.average([min_power,max_power]),max_power]),
                 max_power],
        vmin = int(min_power),
        vmax = int(max_power),
        caption='Received Signal Power (dBm)',
        tick_labels = [min_power,
                 np.average([min_power,np.average([min_power,max_power])]),
                 np.average([min_power,max_power]),
                 np.average([np.average([min_power,max_power]),max_power]),
                 max_power]
        )
    gradient_map=defaultdict(dict)
    for i in range(steps):
        gradient_map[1/steps*i] = colormap.rgb_hex_str(min_power + ((max_power-min_power)/steps*i))
    return colormap, gradient_map

def create_synthetic_ems_data(center_points_comprehension,possible_bands = ['MILBAND'],possible_datetimes = [str(date.today())],variances_in_m=10000,num_datapoints = 100):
    data = []
    avg_received_power = -60; received_power_variance = 10 # dBm
    # Data format: [lat, lng] or [lat, lng, weight]
    datapoint_index = 1
    for i_cp,center_point_list in enumerate(center_points_comprehension):
        num_datapoints *= len(center_point_list)
        for d in range(num_datapoints):
            center_point = random.choice(center_point_list)
            lat = np.random.normal(center_point[0],convert_meters_to_coordinates(variances_in_m[i_cp]))
            long = np.random.normal(center_point[1],convert_meters_to_coordinates(variances_in_m[i_cp]))
            pwr = np.random.normal(avg_received_power,received_power_variance)
            band = possible_bands[i_cp % len(possible_bands)]
            datetime = random.choice(possible_datetimes)
            row = [datapoint_index,[lat,long,pwr],band,datetime,lat,long,pwr]
            data.append(row)
            datapoint_index += 1
        data_df = pd.DataFrame(data,columns=['ID','Heatmap Data','Frequency Band','Datetime','Latitude','Longitude','Received Signal Power'])
    return data_df

def convert_meters_to_coordinates(meters):
    return meters / 111139

def convert_coords_to_mgrs(coords,precision=5):
    return mgrs.MGRS().toMGRS(coords[0], coords[1],MGRSPrecision=precision)

def covert_degrees_to_radians(degrees):
    return degrees * np.pi/180

def get_distance_between_coords(coord1,coord2):
    return hs.haversine(coord1,coord2,unit=Unit.METERS)

def get_center_coord(coord_list):
    return [np.average([c[0] for c in coord_list]),np.average([c[1] for c in coord_list])]

def get_line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def get_intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return [x,y]
    else:
        return False

def get_emissions_centerpoints(df_heatmap,frequency_band,min_improvement=1/3):
    df_heatmap = df_heatmap[df_heatmap['Frequency Band']==frequency_band]
    # frequency_band = 'MILBAND (30-88MHz)'
    def display_elbow_curve(K_clusters,scores,frequency_band):
        plt.plot(K_clusters, scores)
        plt.xlabel('Number of Clusters (aka Source Emitters)')
        plt.ylabel('Total Cluster Score')
        plt.title(f'Elbow Curve ({frequency_band})')
        plt.show()
    def determine_optimal_k(scores, min_improvement): # returns the optimal k value for the k-means
        optimal_k = 1
        previous_improvement = 0
        relative_improvement = 0
        for i,score in enumerate(scores[:-1]):
            absolute_improvement = scores[i+1] - score 
            relative_improvement = (scores[i+1] - score)/abs(score)
            if previous_improvement != 0:
                improvement_acceleration = 1 - (relative_improvement/previous_improvement)
            else:
                improvement_acceleration = relative_improvement
            if relative_improvement >= min_improvement and absolute_improvement >= 0.01 and (improvement_acceleration >= min_improvement or improvement_acceleration == 0):
                previous_improvement = relative_improvement
                optimal_k += 1
                continue
            else:
                break
        return optimal_k
    X = df_heatmap.loc[:,['ID','Latitude','Longitude']]
    K_clusters = range(1,10)
    kmeans = [KMeans(n_clusters=i) for i in K_clusters]
    Y_axis = df_heatmap[['Latitude']]
    X_axis = df_heatmap[['Longitude']]
    y_scores = [kmeans[i].fit(Y_axis).score(Y_axis) for i in range(len(kmeans))]
    x_scores = [kmeans[i].fit(X_axis).score(X_axis) for i in range(len(kmeans))]
    scores = [x_scores[i]+y_scores[i] for i in range(len(x_scores))]
    # display_elbow_curve(K_clusters,scores,frequency_band)
    optimal_k = determine_optimal_k(scores,min_improvement)
    kmeans = KMeans(n_clusters = optimal_k, init ='k-means++')
    kmeans.fit(X[X.columns[1:3]]) # Compute k-means clustering.
    # X['cluster_label'] = kmeans.fit_predict(X[X.columns[1:3]])
    centers = kmeans.cluster_centers_ # Coordinates of cluster centers.
    # labels = kmeans.predict(X[X.columns[1:3]]) # Labels of each point
    return centers

## INPUT DATA ##

# Defaults
ems_band_colors = {'MILBAND (30-88MHz)':'red',
                   'Low-UHF (300-500MHz)':'lightblue',
                   'WiFi (2.4GHz)':'lightgreen'}
ems_band_icons = {'MILBAND (30-88MHz)':'person-military-rifle',
                   'Low-UHF (300-500MHz)':'walkie-talkie',
                   'WiFi (2.4GHz)':'wifi'}
ems_heatmap_colors = {'MILBAND (30-88Mhz)':['lightyellow','yellow','orange','red','darkred'],
                      'Low-UHF (300-500MHz)':['lightyellow','limegreen','cyan','steelblue','midnightblue'],
                      'WiFi (2.4GHz)':['bisque','darkorange','orangered','darkred','maroon']}
scientific_multiplier = {'MHz':10**6,'GHz':10**9}
num_datapoints = 100
mark_clusters_bool = False
mark_true_emitter_sources_bool = False
mark_key_terrain = False
static_frequency_groups = True
mark_lob_centers = False
mark_sensor_locations = True
mark_cut_location = True
mark_cut_error_corners = False
plot_lob_fans = True
plot_center_lobs = True
# Synthetic Emissions Input
center_points_comprehension = [[[33.82020918727583, 42.462662094695744],[33.81280807914385, 42.42543296008796]],[[33.796017215219585, 42.44480198632105]],[[33.79447850651584, 42.436030339247395]]]
possible_datetimes = [str(date.today())]
possible_bands = ['MILBAND (30-88MHz)','Low-UHF (300-500MHz)','WiFi (2.4GHz)']
variances_in_m = [750,500,100]
df_heatmap = create_synthetic_ems_data(center_points_comprehension,possible_bands,possible_datetimes,variances_in_m,num_datapoints)
# Marker Input
marker_coords = [33.8081039282387, 42.443838016360324]
marker_name = 'Al Asad Airbase, Iraq'
marker_color = 'darkblue'
marker_icon = 'plane'

## INTERMEDIATE DATA ##

lats = []; longs = []
for center_point_list in center_points_comprehension:
    for center_point in center_point_list:
        lats.append(center_point[0])
        longs.append(center_point[1])
center_coordinate = [np.average(lats),np.average(longs)]
heatmap_data = list(df_heatmap['Heatmap Data'])
datetime_data = list(df_heatmap['Datetime'])
band_data = sorted(list(set(df_heatmap['Frequency Band'])), key = lambda x: (float(''.join([y for y in x.split('(')[-1].split('-')[0] if y in string.digits+'.']))*scientific_multiplier[x[-4:-1]]))
min_power = min([pwr[-1] for pwr in df_heatmap['Heatmap Data']])
max_power = max([pwr[-1] for pwr in df_heatmap['Heatmap Data']])
colors = ['lightyellow','yellow','orange','red','darkred']
colormap = create_gradient_map(min_power,max_power,colors)[0]

## CREATE MAP ##

# Generate map
m = create_map(center_coordinate)
# Add marker(s) to map
if mark_key_terrain:
    m = add_marker_with_copiable_coordinates(m,marker_coords,marker_name,marker_color,marker_icon)
if mark_true_emitter_sources_bool:
    for cpc_index, center_point_list in enumerate(center_points_comprehension):
        for center_point in center_point_list:
            add_marker_with_copiable_coordinates(m,[center_point[0],center_point[1]],f'{possible_bands[cpc_index]} Emitter Actual Location','black',ems_band_icons.get(possible_bands[cpc_index],'tower-broadcast'),marker_prefix='fa')
if mark_clusters_bool:
    for bd in band_data:
        emissions_sources = get_emissions_centerpoints(df_heatmap,bd,1/3)
        for i,emissions_source in enumerate(emissions_sources):
            add_marker_with_copiable_coordinates(m,list(emissions_source),f'{bd.split("(")[0].strip()} Emitter # {i+1} Estimated Location',ems_band_colors.get(bd,'Black'),ems_band_icons.get(bd,'tower-broadcast'),marker_prefix='fa')

def get_coords_from_LOBS(sensor_coord,azimuth,error):
    def middle_coords(coord1,coord2):
        return [np.average([coord1[0],coord2[0]]),np.average([coord1[1],coord2[1]]),1]
    '''
    add min and max distance based on power output
    create in-between data so the heatmaps are not as non-uniform
    plot the "area of error" for the target emitter
    '''
    right_error_azimuth = (azimuth+error) % 360
    left_error_azimuth = (azimuth-error) % 360
    running_coord_left = [list(sensor_coord)[0],list(sensor_coord)[1]]
    running_coord_center = [list(sensor_coord)[0],list(sensor_coord)[1]]
    running_coord_right = [list(sensor_coord)[0],list(sensor_coord)[1]]
    lob_heatmap_data = []
    # on_LOB_weight = 1
    # error_edge_weight = on_LOB_weight*0.1
    min_lob_length = 1000
    max_lob_length = 5000
    lob_length = 0
    interval_meters = 50
    near_right_coord = []; near_left_coord = []; far_right_coord = []; far_left_coord = []
    # point = [33.796017215219585, 42.44480198632105]
    radians_left = covert_degrees_to_radians(left_error_azimuth)
    radians_center = covert_degrees_to_radians(azimuth)
    radians_right = covert_degrees_to_radians(right_error_azimuth)
    easting_meters_left = sin(radians_left)*interval_meters
    northing_meters_left = cos(radians_left)*interval_meters
    easting_meters_center = sin(radians_center)*interval_meters
    northing_meters_center = cos(radians_center)*interval_meters
    easting_meters_right = sin(radians_right)*interval_meters
    northing_meters_right = cos(radians_right)*interval_meters
    easting_coord_left_correction = convert_meters_to_coordinates(easting_meters_left)
    northing_coord_left_correction = convert_meters_to_coordinates(northing_meters_left)
    easting_coord_center_correction = convert_meters_to_coordinates(easting_meters_center)
    northing_coord_center_correction = convert_meters_to_coordinates(northing_meters_center)
    easting_coord_right_correction = convert_meters_to_coordinates(easting_meters_right)
    northing_coord_right_correction = convert_meters_to_coordinates(northing_meters_right)
    while lob_length <= max_lob_length:
        running_coord_left[1] += easting_coord_left_correction
        running_coord_left[0] += northing_coord_left_correction
        running_coord_center[1] += easting_coord_center_correction
        running_coord_center[0] += northing_coord_center_correction       
        running_coord_right[1] += easting_coord_right_correction
        running_coord_right[0] += northing_coord_right_correction        
        if lob_length > min_lob_length:
            if near_right_coord == []: near_right_coord = list(running_coord_right)
            if near_left_coord == []: near_left_coord = list(running_coord_left)
            lob_heatmap_data.append([running_coord_center[0],running_coord_center[1],1])
            if running_coord_left != running_coord_center:
                lob_heatmap_data.append([running_coord_left[0],running_coord_left[1],1])
                lmcords = middle_coords(running_coord_center,running_coord_left)
                lob_heatmap_data.append(lmcords)
                lob_heatmap_data.append(middle_coords(running_coord_center,lmcords))
                lob_heatmap_data.append(middle_coords(running_coord_left,lmcords))
            if running_coord_right != running_coord_center:
                lob_heatmap_data.append([running_coord_right[0],running_coord_right[1],1])
                lob_heatmap_data.append(middle_coords(running_coord_center,running_coord_right))
                rmcords = middle_coords(running_coord_center,running_coord_right)                
                lob_heatmap_data.append(middle_coords(running_coord_center,rmcords))
                lob_heatmap_data.append(middle_coords(running_coord_right,rmcords))
        lob_length += interval_meters
    far_right_coord = running_coord_right
    far_left_coord = running_coord_left
    center_coord = get_center_coord([near_right_coord,near_left_coord,far_right_coord,far_left_coord])
    heatmap_weight = []
    for lhd in lob_heatmap_data:
        coords = lhd[:2]
        dst = get_distance_between_coords(center_coord,coords)
        if dst >= 1:
            heatmap_weight.append(1/dst)
        else:
            heatmap_weight.append(1)
    lob_heatmap_data_adjusted = []
    for i,lhd in enumerate(lob_heatmap_data):
        lob_heatmap_data_adjusted.append(lhd[:2]+[heatmap_weight[i]])
    return lob_heatmap_data_adjusted, center_coord, near_right_coord, near_left_coord, far_right_coord, far_left_coord, running_coord_center

def add_heatmap(m,heatmap_data,frequency_band = 'MILBAND (30-88MHz)',heatmap_colors=['lightyellow','yellow','orange','red','darkred']):
    min_power = min([weight[2] for weight in heatmap_data]); max_power = max([weight[2] for weight in heatmap_data])
    gradient_map = create_gradient_map(min_power,max_power,heatmap_colors)[1]
    HeatMap(data=heatmap_data,
            name=frequency_band,
            gradient=gradient_map,
            radius = 25,
            blur = 15,
            overlay = True,
            control = True,
            show=False
            ).add_to(m)
    return m

def add_polygon(m,points,line_color='black',shape_fill_color=None,line_weight=5):
    folium.Polygon(locations = points,
                   color=line_color,
                   weight=line_weight,
                   fill_color=shape_fill_color
                   ).add_to(m)
    return m

def add_line(m,points,line_color='black',line_weight=5,dash_weight=None):
    folium.PolyLine(locations = points,
                    color = line_color,
                    weight = line_weight,
                    dash_array = dash_weight
                    ).add_to(m)
    return m

# Add tiles to map
m = add_tilelayers(m)
# Add heat maps for defined frequency groups
if static_frequency_groups:
    for i,pb in enumerate(possible_bands):
        frequency_band_data = df_heatmap[df_heatmap['Frequency Band']==pb]['Heatmap Data']
        m = add_heatmap(m,frequency_band_data,pb,ems_heatmap_colors.get(pb,['lightyellow','yellow','orange','red','darkred']))
        
## ADD PLUGINS/WIDGETS ##

# Add colormap
# m.add_child(colormap)
# Add minimap
MiniMap(tile_layer="Stamen WaterColor", position="bottomleft").add_to(m)
# Add layer control
LayerControl(position='topright').add_to(m)
# Add fullscreen option
Fullscreen(position='topleft',title='Full Screen').add_to(m)
# Add locate control
LocateControl().add_to(m)
m.add_child(folium.LatLngPopup())


## SAVE MAP ##

m.save("C:\\Users\\brend\\Documents\\Coding Projects\\CEMA\\Maps\\heatmap_multiband.html")

'''
NOTES:
    MUST NORMALIZE HEATMAP WEIGHTS [0-1] 
'''






