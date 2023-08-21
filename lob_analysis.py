#!/usr/bin/env python3

import os 
import folium
# folium icons: https://fontawesome.com/icons?d=gallery
# folium icon colors: ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']
# folium tiles: OpenStreetMap , Stamen Terrain, Stamen Toner
# import earthpy as et
# import earthpy.spatial as es
from folium.plugins import HeatMap, HeatMapWithTime, MiniMap, Fullscreen
from folium.plugins import MarkerCluster, MeasureControl
from folium.map import LayerControl
import warnings
warnings.filterwarnings("ignore")

os.chdir(rf'{os.getcwd()[0]}:\Users\brend\Documents\Coding Projects\CEMA')
from main import plot_grid_lines,convert_mgrs_to_coords,convert_coords_to_mgrs,add_copiable_markers,create_marker,create_polygon,create_line,convert_meters_to_coordinates
from main import create_map,create_gradient_map,convert_watts_to_dBm,covert_degrees_to_radians,get_distance_between_coords
from main import get_center_coord,emission_distance,emission_optical_maximum_distance,emission_optical_maximum_distance_with_ducting
from main import get_emission_distance,organize_polygon_coords,get_line,get_intersection,get_polygon_area,get_accuracy_improvement_of_cut
from main import get_accuracy_improvement_of_fix,get_coords_from_LOBS,get_fix_coords,plot_LOB,create_lob_cluster
# https://www.kaggle.com/code/daveianhickey/how-to-folium-for-maps-heatmaps-time-data

def add_tilelayers(m):
    # folium.TileLayer(
    #     tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    #     attr = 'Esri',
    #     name = 'Satellite',
    #     overlay = False,
    #     control = True
    #     ).add_to(m)
    folium.TileLayer(
        tiles = 'http://10.0.0.40/tile/{z}/{x}/{y}.png',
        attr = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        name = 'Offline',
        overlay = False,
        control = True
        ).add_to(m)   
    return m

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

# Boolean Parameters
mark_clusters_bool = False
mark_true_emitter_sources_bool = False
mark_key_terrain = False
static_frequency_groups = False
mark_lob_centers = False
mark_sensor_locations = True
mark_cut_location = False
mark_cut_error_corners = False
plot_lob_fans = True
plot_center_lobs = True
plot_cut_area = True
plot_fix_area = True
plot_center_marker = True
optical_emission_distance = True
ducting_emission_distance = False

# Marker Input
marker_coords = [33.8081039282387, 42.443838016360324]
marker_name = 'Al Asad Airbase, Iraq'
marker_color = 'darkblue'
marker_icon = 'plane'

sensor_coord_1 = [32.01844315777277, -81.82981355491064]
sensor_coord_2 = [32.0132109964423, -81.83281150219979]
sensor_coord_3 = [32.01005667742662, -81.82774993986048]
sensor_r_p_1_dBm = -75
sensor_r_p_2_dBm = -75
sensor_r_p_3_dBm = -75
sensor_coords = [sensor_coord_1,sensor_coord_2,sensor_coord_3]
sensor_received_power = [sensor_r_p_1_dBm,sensor_r_p_2_dBm,sensor_r_p_3_dBm]
azimuth_sensor_1 = 115
azimuth_sensor_2 = 80
azimuth_sensor_3 = 30
error_sensor_1 = error_sensor_2 = error_sensor_3 = 6
min_wattage = 0.1
receiver_gain_dBi = 0
receiver_height_m = 1.5
transmitter_gain_dBi = 0
transmitter_height_m = 1.5
max_wattage = 5
frequency_MHz = 462
received_power_dBm = -85
temp_f = 80


min_distance_km = get_emission_distance(ducting_emission_distance,optical_emission_distance,min_wattage,frequency_MHz,transmitter_gain_dBi,receiver_gain_dBi,received_power_dBm,transmitter_height_m,receiver_height_m,temp_f,path_loss_coeff=3,weather_coeff=4/3)
min_distance_m = min_distance_km * 1000
max_distance_km = get_emission_distance(ducting_emission_distance,optical_emission_distance,max_wattage,frequency_MHz,transmitter_gain_dBi,receiver_gain_dBi,received_power_dBm,transmitter_height_m,receiver_height_m,temp_f,path_loss_coeff=3,weather_coeff=4/3)
max_distance_m = max_distance_km * 1000
# Find LOBs

lob1_center, lob1_near_right_coord, lob1_near_left_coord, lob1_far_right_coord, lob1_far_left_coord, lob1_far_middle_coord = get_coords_from_LOBS(sensor_coord_1,azimuth_sensor_1,error_sensor_1,min_distance_m,max_distance_m)
lob2_center, lob2_near_right_coord, lob2_near_left_coord, lob2_far_right_coord, lob2_far_left_coord, lob2_far_middle_coord = get_coords_from_LOBS(sensor_coord_2,azimuth_sensor_2,error_sensor_2,min_distance_m,max_distance_m)
lob3_center, lob3_near_right_coord, lob3_near_left_coord, lob3_far_right_coord, lob3_far_left_coord, lob3_far_middle_coord = get_coords_from_LOBS(sensor_coord_3,azimuth_sensor_3,error_sensor_3,min_distance_m,max_distance_m)

lob1_center = get_line(sensor_coord_1, lob1_far_middle_coord)
lob1_right_bound = get_line(lob1_near_right_coord, lob1_far_right_coord)
lob1_left_bound = get_line(lob1_near_left_coord, lob1_far_left_coord)

lob2_center = get_line(sensor_coord_2, lob2_far_middle_coord)
lob2_right_bound = get_line(lob2_near_right_coord, lob2_far_right_coord)
lob2_left_bound = get_line(lob2_near_left_coord, lob2_far_left_coord)

lob3_center = get_line(sensor_coord_3, lob3_far_middle_coord)
lob3_right_bound = get_line(lob3_near_right_coord, lob3_far_right_coord)
lob3_left_bound = get_line(lob3_near_left_coord, lob3_far_left_coord)
   
points_lob1_polygon = [lob1_near_right_coord,lob1_far_right_coord,lob1_far_left_coord,lob1_near_left_coord]
points_lob1_polygon = organize_polygon_coords(points_lob1_polygon)

points_lob2_polygon = [lob2_near_right_coord,lob2_far_right_coord,lob2_far_left_coord,lob2_near_left_coord] 
points_lob2_polygon = organize_polygon_coords(points_lob2_polygon)

points_lob3_polygon = [lob3_near_right_coord,lob3_far_right_coord,lob3_far_left_coord,lob3_near_left_coord] 
points_lob3_polygon = organize_polygon_coords(points_lob3_polygon)

lob1_area = get_polygon_area(points_lob1_polygon)
lob2_area = get_polygon_area(points_lob2_polygon)
lob3_area = get_polygon_area(points_lob3_polygon)

intersection_l1_l2 = get_intersection(lob1_center, lob2_center)
if intersection_l1_l2:
    center_coordinate = get_center_coord([intersection_l1_l2])
    intersection_l1r_l2r = get_intersection(lob1_right_bound, lob2_right_bound)
    intersection_l1r_l2l = get_intersection(lob1_right_bound, lob2_left_bound)
    intersection_l1l_l2r = get_intersection(lob1_left_bound, lob2_right_bound)
    intersection_l1l_l2l = get_intersection(lob1_left_bound, lob2_left_bound)
    intersection_l1_l3 = get_intersection(lob1_center, lob3_center)
    intersection_l2_l3 = get_intersection(lob2_center, lob3_center)
    if intersection_l1_l3:
        intersection_l1r_l3r = get_intersection(lob1_right_bound, lob3_right_bound)
        intersection_l1r_l3l = get_intersection(lob1_right_bound, lob3_left_bound)
        intersection_l1l_l3r = get_intersection(lob1_left_bound, lob3_right_bound)
        intersection_l1l_l3l = get_intersection(lob1_left_bound, lob3_left_bound)
    if intersection_l2_l3:
        intersection_l3r_l2r = get_intersection(lob3_right_bound, lob2_right_bound)
        intersection_l3r_l2l = get_intersection(lob3_right_bound, lob2_left_bound)
        intersection_l3l_l2r = get_intersection(lob3_left_bound, lob2_right_bound)
        intersection_l3l_l2l = get_intersection(lob3_left_bound, lob2_left_bound)
    if intersection_l1_l2 and intersection_l2_l3 and intersection_l1_l3:
        center_coordinate = get_center_coord([intersection_l1_l2,intersection_l2_l3,intersection_l1_l3])

## FIND CUT POLYGONS ##
points_cut_polygons = []
cut_titles = []
cut_areas = []
lob_to_cut_improvements = []
if intersection_l1r_l2r and intersection_l1r_l2l and intersection_l1l_l2l and intersection_l1l_l2r:
    points_cut_polygon = [intersection_l1r_l2r,intersection_l1r_l2l,intersection_l1l_l2l,intersection_l1l_l2r]
    points_cut_polygon = organize_polygon_coords(points_cut_polygon)
    points_cut_polygons.append(points_cut_polygon)
    cut_titles.append('EWT 1 & 2 Cut')
    cut_area = get_polygon_area(points_cut_polygon)
    cut_areas.append(cut_area)
    lob_to_cut_improvement = get_accuracy_improvement_of_cut(lob1_area,lob2_area,cut_area)
    lob_to_cut_improvements.append(lob_to_cut_improvement)
if intersection_l1r_l3r and intersection_l1r_l3l and intersection_l1l_l3r and intersection_l1l_l3l:
    points_cut_polygon = [intersection_l1r_l3r,intersection_l1r_l3l,intersection_l1l_l3r,intersection_l1l_l3l]
    points_cut_polygon = organize_polygon_coords(points_cut_polygon)
    points_cut_polygons.append(points_cut_polygon)
    cut_titles.append('EWT 1 & 3 Cut')
    cut_area = get_polygon_area(points_cut_polygon)
    cut_areas.append(cut_area)
    lob_to_cut_improvement = get_accuracy_improvement_of_cut(lob1_area,lob3_area,cut_area)
    lob_to_cut_improvements.append(lob_to_cut_improvement)
if intersection_l3r_l2r and intersection_l3r_l2l and intersection_l3l_l2r and intersection_l3l_l2l:
    points_cut_polygon = [intersection_l3r_l2r,intersection_l3r_l2l,intersection_l3l_l2r,intersection_l3l_l2l]
    points_cut_polygon = organize_polygon_coords(points_cut_polygon)
    points_cut_polygons.append(points_cut_polygon)
    cut_titles.append('EWT 2 & 3 Cut')
    cut_area = get_polygon_area(points_cut_polygon)
    cut_areas.append(cut_area)
    lob_to_cut_improvement = get_accuracy_improvement_of_cut(lob3_area,lob2_area,cut_area)
    lob_to_cut_improvements.append(lob_to_cut_improvement)

if plot_fix_area and intersection_l1_l2 and intersection_l2_l3 and intersection_l1_l3:
    points_fix_polygon = get_fix_coords(points_cut_polygons)
    points_fix_polygon = organize_polygon_coords(points_fix_polygon)
    fix_area = get_polygon_area(points_fix_polygon)
    

points_lob_polygons = [points_lob1_polygon,points_lob2_polygon,points_lob3_polygon]
azimuths = [azimuth_sensor_1,azimuth_sensor_2,azimuth_sensor_3]

## INTERMEDIATE DATA ##

colors = ['lightyellow','yellow','orange','red','darkred']

## CREATE MAP ##

# Generate map
m = create_map(center_coordinate)
ew_team_cluster = MarkerCluster(name='EW Teams',show=False)
ew_team_cluster.add_to(m)
lob_cluster = MarkerCluster(name='MILBAND LOBs',show=False)
lob_cluster.add_to(m)
cut_cluster = MarkerCluster(name='MILBAND Cuts',show=False)
cut_cluster.add_to(m)
fix_cluster = MarkerCluster(name='MILBAND Fixes',show=False)
fix_cluster.add_to(m)
centerpoint_cluster = MarkerCluster(name='MILBAND Centerpoint',show=False)
centerpoint_cluster.add_to(m)

# Add marker(s) to map
if mark_key_terrain:
    marker = create_marker(marker_coords,marker_name,marker_color,marker_icon)
    marker.add_to(m)

if mark_sensor_locations:
    marker_sensor_1 = create_marker(sensor_coord_1,"EWT 1",'blue','binoculars')
    # marker_sensor_1.add_to(m)
    ew_team_cluster.add_child(marker_sensor_1)
    marker_sensor_2 = create_marker(sensor_coord_2,"EWT 2",'blue','binoculars')
    # marker_sensor_2.add_to(m)
    ew_team_cluster.add_child(marker_sensor_2)
    marker_sensor_3 = create_marker(sensor_coord_3,"EWT 3",'blue','binoculars')
    # marker_sensor_3.add_to(m)
    ew_team_cluster.add_child(marker_sensor_3)

if mark_lob_centers:
    marker_lob_center_1 = create_marker(lob1_center,"LOB 1 Center",'black','bullseye')
    marker_lob_center_1.add_to(m)
    marker_lob_center_2 = create_marker(lob2_center,"LOB 2 Center",'black','bullseye')
    marker_lob_center_2.add_to(m)
    marker_lob_center_3 = create_marker(lob2_center,"LOB 3 Center",'black','bullseye')
    marker_lob_center_3.add_to(m)

if plot_lob_fans:
    lob_cluster = create_lob_cluster(lob_cluster,points_lob_polygons,azimuths,error_sensor_1,min_distance_m,max_distance_m)

# Add tiles to map
m = add_tilelayers(m)

if plot_center_lobs:
    line_lob1 = create_line([sensor_coord_1,lob1_far_middle_coord],'orange',2,f'{azimuth_sensor_1}° ({min_distance_m/1000:.2f}-{max_distance_m/1000:.2f}km)',8)
    lob_cluster.add_child(line_lob1)
    line_lob2 = create_line([sensor_coord_2,lob2_far_middle_coord],'orange',2,f'{azimuth_sensor_2}° ({min_distance_m/1000:.2f}-{max_distance_m/1000:.2f}km)',8)
    lob_cluster.add_child(line_lob2)
    line_lob3 = create_line([sensor_coord_3,lob3_far_middle_coord],'orange',2,f'{azimuth_sensor_2}° ({min_distance_m/1000:.2f}-{max_distance_m/1000:.2f}km)',8)
    lob_cluster.add_child(line_lob3)

if center_coordinate and plot_center_marker and len(points_lob_polygons) == 3:
    fix_polygon = create_polygon(points_fix_polygon,'yellow','yellow',5,f'Fix Error: {fix_area:,.2f} acres ({get_accuracy_improvement_of_fix(fix_area,cut_areas)*100:.1f}% improvement over Cuts)')
    fix_cluster.add_child(fix_polygon)
    marker_center = create_marker(center_coordinate,"Center",'black','bullseye')
    centerpoint_cluster.add_child(marker_center)
    if plot_cut_area:
        colors = ['Red','Black','White']
        for i,cp in enumerate(points_cut_polygons):
            cut_polygon = create_polygon(cp,colors[i],colors[i],5,f'{cut_titles[i]} w/ {cut_areas[i]:,.2f} acres of error ({lob_to_cut_improvements[i]*100:.1f}% improvement over LOBs)')
            cut_cluster.add_child(cut_polygon)
    
    
if center_coordinate and plot_cut_area and len(points_lob_polygons) == 2:
    colors = ['Red','Black','White']
    for i,cp in enumerate(points_cut_polygons):
        cut_polygon = create_polygon(cp,colors[i],colors[i],5,f'{cut_titles[i]} w/ {cut_areas[i]:,.2f} acres of error ({lob_to_cut_improvements[i]*100:.1f}% improvement over LOBs)')
        cut_cluster.add_child(cut_polygon)
    if plot_center_marker:
        marker_center = create_marker(center_coordinate,"Center",'black','bullseye')
        centerpoint_cluster.add_child(marker_center)
    # if mark_cut_error_corners:  
    #     m = add_marker_with_copiable_coordinates(m,intersection_l1r_l2r,"Cut Intersection",'black','bullseye')
    #     m = add_marker_with_copiable_coordinates(m,intersection_l1l_l2r,"Cut Intersection",'black','bullseye')
    #     m = add_marker_with_copiable_coordinates(m,intersection_l1r_l2l,"Cut Intersection",'black','bullseye')
    #     m = add_marker_with_copiable_coordinates(m,intersection_l1l_l2l,"Cut Intersection",'black','bullseye')

gridLines = plot_grid_lines(center_coordinate)
mgrs_gridzone = MarkerCluster(name='MGRS Gridlines',show=False)
mgrs_gridzone.add_to(m)
for gl in gridLines:
    mgrs_gridzone.add_child(gl)

# Add minimap
MiniMap(tile_layer="Stamen WaterColor", position="bottomleft").add_to(m)
# Add fullscreen option
Fullscreen(position='topleft',title='Full Screen').add_to(m)
# Add Copiable Markers
m = add_copiable_markers(m)
# Add layer control
LayerControl(position='topright').add_to(m)
m.add_child(folium.LatLngPopup())

## SAVE MAP ##
m.save("C:\\Users\\brend\\Documents\\Coding Projects\\CEMA\\Maps\\lob_analysis.html")

'''
IDEAS:
    Heatmap over time (X hour chunks) ** (can only do a single frequency group per product)
    Combining LOB analysis w/ heatmaping over time *** (can only do a single frequency group per product)
    Add polygon layer to mimic MGRS grids
    
    
    Datetime slider (only for a single frequency group)
    Frequency Band selector
    Collection site variable
    Received Power contribution to density function or min received power conditional
    Tile selector (satelite, streetview, MGRS, Stamen Toner?)
    
    Beyond EMS Surveys:
        plot LOBs on the map
    Program that distingishes emitters from a single location based on received power
'''

## RANDOM WORK ##


