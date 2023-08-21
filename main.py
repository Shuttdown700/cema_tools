#!/usr/bin/env python3

import folium
import branca.colormap as cm
from collections import defaultdict
import jinja2
import numpy as np
import math
from datetime import date
import warnings
import mgrs
from math import sin, cos, pi
import haversine as hs
from haversine import Unit
from sympy import Point, Polygon
warnings.filterwarnings("ignore")

def convert_meters_to_coordinates(meters):
    """
    Converts meters to equivalent coordinate distance.

    Parameters
    ----------
    meters : float
        Length in meters.

    Returns
    -------
    float
        Length in coordinates.

    """
    assert isinstance(meters,(float,int)), 'Input needs to be a float.'
    return meters / 111139

def convert_coordinates_to_meters(coord):
    """
    Converts coodinate distance to meters.

    Parameters
    ----------
    meters : float
        Length in meters.

    Returns
    -------
    float
        Length in coordinates.

    """
    assert isinstance(coord,(float,int)), 'Input needs to be a float.'
    return coord * 111139

def adjust_lat(center_coord,dist_km):
    """
    Adjusts the latitude of a coordinate by a distance (in km)

    Parameters
    ----------
    center_coord : list of length 2
        Grid coordinate. Example: [lat,long]
    dist_km : float
        Distance in km.

    Returns
    -------
    float
        Adjusted latitude of given coordinate.

    """
    assert isinstance(center_coord,list) and len(center_coord) == 2, 'Coordinate needs to be a list of length 2.'
    assert isinstance(dist_km,(float,int)), 'Adjustment needs to be a float.'
    return center_coord[0]  + (dist_km / 6378) * (180 / pi)

def adjust_long(center_coord,dist_km):
    """
    Adjusts the longitude of a coordinate by a distance (in km)

    Parameters
    ----------
    center_coord : list of length 2
        Grid coordinate. Example: [lat,long]
    dist_km : float
        Distance in km.

    Returns
    -------
    float
        Adjusted longitude of given coordinate.

    """
    assert isinstance(center_coord,list) and len(center_coord) == 2, 'Coordinate needs to be a list of length 2.'
    assert isinstance(dist_km,(float,int)), 'Adjustment needs to be a float.'
    return center_coord[1] + (dist_km / 6378) * (180 / pi) / cos(center_coord[1] * pi/180)

def convert_watts_to_dBm(p_watts):
    """
    Converts watts to dBm.

    Parameters
    ----------
    p_watts : float
        Power in watts (W).

    Returns
    -------
    float
        Power in dBm.

    """
    assert isinstance(p_watts,(float,int)) and p_watts >= 0, 'Wattage needs to be a float greater than zero.'
    return 10*np.log10(1000*p_watts)

def convert_coords_to_mgrs(coords,precision=5):
    """
    Convert location from coordinates to MGRS.

    Parameters
    ----------
    coords : list of length 2
        Grid coordinate. Example: [lat,long].
    precision : int, optional
        Significant figures per easting/northing value. The default is 5.

    Returns
    -------
    str
        Location in MGRS notation.

    """
    assert isinstance(coords,list), 'Coordinate input must be a list.'
    assert len(coords) == 2, 'Coordinate input must be of length 2.'
    return mgrs.MGRS().toMGRS(coords[0], coords[1],MGRSPrecision=precision)

def convert_mgrs_to_coords(milGrid):
    """
    Convert location from MGRS to coordinates.

    Parameters
    ----------
    milGrid : str
        Location in MGRS notation.

    Returns
    -------
    list
        Grid coordinate. Example: [lat,long].

    """
    assert isinstance(milGrid,str), 'MGRS must be a string'
    return list(mgrs.MGRS().toLatLon(milGrid))

def covert_degrees_to_radians(degrees):
    """
    Converts angle from degrees to radians.

    Parameters
    ----------
    degrees : float [0-360]
        Angle measured in degrees.

    Returns
    -------
    float
        Angle measured in radians [0-2π].

    """
    assert isinstance(degrees,(float,int)), 'Degrees must be a float [0-360]'
    return (degrees%360) * np.pi/180

def get_distance_between_coords(coord1,coord2):
    """
    Determines distance between two coordinates in meters

    Parameters
    ----------
    coord1 : list of length 2
        Grid coordinate. Example: [lat,long].
    coord2 : list of length 2
        Grid coordinate. Example: [lat,long].
    Returns
    -------
    float
        Distance between two coordinates in meters.

    """
    assert isinstance(coord1,list), 'Coordinate 1 must be a list.'
    assert len(coord1) == 2, 'Coordinate 1 must be of length 2.'
    assert isinstance(coord2,list), 'Coordinate 2 must be a list.'
    assert len(coord2) == 2, 'Coordinate 2 must be of length 2.'
    return hs.haversine(coord1,coord2,unit=Unit.METERS)

def get_center_coord(coord_list):
    """
    Returns the average coordinate from list of coordinates.

    Parameters
    ----------
    coord_list : list comprehension of length n
        List of coordinates. Example: [[lat1,long1],[lat2,long2]]

    Returns
    -------
    list
        Center coordiante.

    """
    return [np.average([c[0] for c in coord_list]),np.average([c[1] for c in coord_list])]

def emission_distance(P_t_watts,f_MHz,G_t,G_r,R_s,path_loss_coeff=3):
    """
    Returns theoretical maximum distance of emission.

    Parameters
    ----------
    P_t_watts : float
        Power output of transmitter in watts (W).
    f_MHz : float
        Operating frequency in MHz.
    G_t : float
        Transmitter antenna gain in dBi.
    G_r : float
        Receiver antenna gain in dBi.
    R_s : float
        Receiver sensitivity in dBm *OR* power received in dBm.
    path_loss_coeff : float, optional
        Coefficient that considers partial obstructions such as foliage. 
        The default is 3.

    Returns
    -------
    float
        Theoretical maximum distance in km.

    """
    return 10**((convert_watts_to_dBm(P_t_watts)+(G_t-2.15)-32.4-(10*path_loss_coeff*np.log10(f_MHz))+(G_r-2.15)-R_s)/(10*path_loss_coeff))

def emission_optical_maximum_distance(t_h,r_h):
    """
    Returns theoretical maximum line-of-sight between transceivers.

    Parameters
    ----------
    t_h : float
        Transmitter height in meters (m).
    r_h : float
        Receiver height in meters (m).

    Returns
    -------
    float
        Maximum line-of-sight due to Earth curvature in km.

    """
    return (np.sqrt(2*6371000*r_h+r_h**2)/1000)+(np.sqrt(2*6371000*t_h+t_h**2)/1000)

def emission_optical_maximum_distance_with_ducting(t_h,r_h,f_MHz,temp_f,weather_coeff=4/3):
    """
    Returns theoretical maximum line-of-sight between transceivers with ducting consideration.

    Parameters
    ----------
    t_h : float
        Transmitter height in meters (m).
    r_h : float
        Receiver height in meters (m).
    f_MHz : float
        Operating frequency in MHz.
    temp_f : float
        ENV Temperature in fahrenheit.
    weather_coeff : float, optional
        ENV Weather conditions coefficient. The default is 4/3.

    Returns
    -------
    float
        Maximum line-of-sight due to Earth curvature and ducting in km.

    """
    return (np.sqrt(2*weather_coeff*6371000*r_h+temp_f**2)/1000)+(np.sqrt(2*weather_coeff*6371000*t_h+f_MHz**2)/1000)

def get_emission_distance(ducting_emission_distance,optical_emission_distance,P_t_watts,f_MHz,G_t,G_r,R_s,t_h,r_h,temp_f,path_loss_coeff=3,weather_coeff=4/3):
    """
    Returns theoretical maximum line-of-sight between transceivers all pragmatic consideration.

    Parameters
    ----------
    ducting_emission_distance : float
        Maximum line-of-sight due to Earth curvature and ducting in km.
    optical_emission_distance : float
        Maximum line-of-sight due to Earth curvature.
    P_t_watts : float
        Power output of transmitter in watts (W).
    f_MHz : float
        Operating frequency in MHz.
    G_t : float
        Transmitter antenna gain in dBi.
    G_r : float
        Receiver antenna gain in dBi.
    R_s : float
        Receiver sensitivity in dBm *OR* power received in dBm.
    t_h : float
        Transmitter height in meters (m).
    r_h : float
        Receiver height in meters (m).
    temp_f : TYPE
        DESCRIPTION.
    temp_f : float
        ENV Temperature in fahrenheit.
    weather_coeff : float, optional
        ENV Weather conditions coefficient. The default is 4/3.

    Returns
    -------
    float
        Maximum line-of-sight due to path-loss, Earth curvature and ducting in km.

    """
    if ducting_emission_distance:
        return min([emission_optical_maximum_distance_with_ducting(t_h,r_h,f_MHz,temp_f,weather_coeff),emission_optical_maximum_distance(t_h,r_h),emission_distance(P_t_watts,f_MHz,G_t,G_r,R_s,path_loss_coeff)])
    elif optical_emission_distance:
        return min([emission_optical_maximum_distance(t_h,r_h),emission_distance(P_t_watts,f_MHz,G_t,G_r,R_s,path_loss_coeff)])
    else:
        return emission_distance(P_t_watts,f_MHz,G_t,G_r,R_s,path_loss_coeff)

def create_map(center_coordinate,zs=16):
    """
    Creates a folium basemap for map product development.

    Parameters
    ----------
    center_coordinate : list
        Grid coordinate. Example: [lat,long]
    zoom_start : int, optional
        Initial zoom level [0-18]. The default is 14.

    Returns
    -------
    m : Folium Map Obj
        Folium basemap with satellite tile.

    """
    assert isinstance(center_coordinate,list), 'Coordinate input must be a list.'
    assert len(center_coordinate) == 2, 'Coordinate input must be of length 2.'
    m = folium.Map(
        location=center_coordinate,
        tiles = folium.TileLayer(
            tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr = 'Esri',
            name = 'Satellite',
            overlay = False,
            control = True
            ),
        zoom_start=zs,
        min_zoom = 0,
        max_zoom = 18,
        control_scale = False)
    return m

def add_title(m,df_heatmap=False):
    if df_heatmap:
        min_date = min(list(df_heatmap['Datetime']))
        max_date = max(list(df_heatmap['Datetime']))
    else:
        min_date = max_date = str(date.today())
    location = 'AL ASAD AIRBASE'
    prefix = '***DEMO***'; suffix = '***DEMO***'
    if min_date == max_date and ':' in min_date:
        map_title = f'{prefix} ELECTROMAGNETIC SPECTRUM (EMS) SURVEY: {location.upper()}, IRAQ at {min_date} {suffix}'
    elif min_date == max_date:
        map_title = f'{prefix} ELECTROMAGNETIC SPECTRUM (EMS) SURVEY: {location.upper()}, IRAQ on {min_date} {suffix}'
    else:
        map_title = f'{prefix} ELECTROMAGNETIC SPECTRUM (EMS) SURVEY: {location.upper()}, IRAQ from {min_date} to {max_date} {suffix}'
    title_html = f'''
                 <h3 align="center" style="font-size:32px;background-color:yellow;color=black;font-family:arial"><b>{map_title}</b></h3>
                 ''' 
    m.get_root().html.add_child(folium.Element(title_html))
    return m

def create_marker(marker_coords,marker_name,marker_color,marker_icon,marker_prefix='fa'):
    marker = folium.Marker(marker_coords,
                  # popup = f'<input type="text" value="{marker_coords[0]}, {marker_coords[1]}" id="myInput"><button onclick="myFunction()">Copy location</button>',
                  popup = f'<input type="text" value="{convert_coords_to_mgrs(marker_coords)}" id="myInput"><button onclick="copyTextFunction()">Copy MGRS Grid</button>',
                  tooltip=marker_name,
                  icon=folium.Icon(color=marker_color,
                                   icon_color='White',
                                   icon=marker_icon,
                                   prefix=marker_prefix)
                  )
    return marker  

def add_copiable_markers(m):
    el = folium.MacroElement().add_to(m)
    el._template = jinja2.Template("""
        {% macro script(this, kwargs) %}
        function copyTextFunction() {
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

def create_polygon(points,line_color='black',shape_fill_color=None,line_weight=5,text=None):
    iframe = folium.IFrame(text, width=250, height=75)
    popup = folium.Popup(iframe, max_width=250)
    polygon = folium.Polygon(locations = points,
                   color=line_color,
                   weight=line_weight,
                   fill_color=shape_fill_color,
                   popup = popup,
                   name = 'test',
                   overlay = False,
                   control = True,
                   show=False
                   )
    return polygon

def organize_polygon_coords(coord_list):
    def clockwiseangle_and_distance(point,origin,refvec):
        vector = [point[0]-origin[0], point[1]-origin[1]]
        lenvector = math.hypot(vector[0], vector[1])
        if lenvector == 0:
            return -math.pi, 0
        normalized = [vector[0]/lenvector, vector[1]/lenvector]
        dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1]
        diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1]
        angle = math.atan2(diffprod, dotprod)
        if angle < 0:
            return 2*math.pi+angle, lenvector
        return angle, lenvector
    coord_list = sorted(coord_list, key = lambda x: (x[1],x[0]),reverse=True)
    origin = coord_list[0]; refvec = [0,1]
    # ordered_points = coord_list[0]; coord_list = coord_list[1:]
    ordered_points = sorted(coord_list, key = lambda x: clockwiseangle_and_distance(x,origin,refvec))
    return ordered_points
    
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
    
def get_polygon_area(shape_coords): # returns area in acres
    x = [convert_coordinates_to_meters(sc[0]) for sc in shape_coords]
    y = [convert_coordinates_to_meters(sc[1]) for sc in shape_coords]    
    return (0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1))))/4046.856422

def get_accuracy_improvement_of_cut(lob1_error,lob2_error,cut_error):
    return 1 - (cut_error/(lob1_error+lob2_error-cut_error))

def get_accuracy_improvement_of_fix(fix_area,cut_areas):
    return 1-(fix_area/(cut_areas[0] + cut_areas[1] + cut_areas[2] - 2*fix_area))

def get_coords_from_LOBS(sensor_coord,azimuth,error,min_lob_length,max_lob_length):
    right_error_azimuth = (azimuth+error) % 360
    left_error_azimuth = (azimuth-error) % 360
    running_coord_left = [list(sensor_coord)[0],list(sensor_coord)[1]]
    running_coord_center = [list(sensor_coord)[0],list(sensor_coord)[1]]
    running_coord_right = [list(sensor_coord)[0],list(sensor_coord)[1]]
    lob_length = 0
    interval_meters = 50
    near_right_coord = []; near_left_coord = []; far_right_coord = []; far_left_coord = []
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
        lob_length += interval_meters
    far_right_coord = running_coord_right
    far_left_coord = running_coord_left
    center_coord = get_center_coord([near_right_coord,near_left_coord,far_right_coord,far_left_coord])
    return center_coord, near_right_coord, near_left_coord, far_right_coord, far_left_coord, running_coord_center

def get_fix_coords(points_cut_polygons):
    def get_polygon(points):
        try:
            p1, p2, p3, p4 = map(Point, points)
            poly = Polygon(p1, p2, p3, p4)
        except ValueError:
            p1, p2, p3 = map(Point, points)
            poly = Polygon(p1, p2, p3)
        return poly
    def get_intersection_coords(intersection):
        x_coords = []; y_coords = []
        for index,ip in enumerate(intersection):
            try:
                x = round(float(intersection[index].x),13)
                y = round(float(intersection[index].y),13)
                if x not in x_coords and y not in y_coords:
                    x_coords.append(x)
                    y_coords.append(y)
                    # print("x:",x); print("y:",y)
            except AttributeError:
                x1 = float(intersection[index].p1.x)
                y1 = float(intersection[index].p1.y)
                if x1 not in x_coords and y1 not in y_coords:
                    x_coords.append(x1)
                    y_coords.append(y1)
                    # print("x1:",x); print("y1:",y)
                x2 = float(intersection[index].p2.x)
                y2 = float(intersection[index].p2.y)
                if x2 not in x_coords and y2 not in y_coords:
                    x_coords.append(x2)
                    y_coords.append(y2)
                    # print("x2:",x); print("y2:",y)
        coords = [[x,y_coords[i]] for i,x in enumerate(x_coords)]
        return coords
    poly1 = get_polygon(points_cut_polygons[0])
    poly2 = get_polygon(points_cut_polygons[1])
    poly3 = get_polygon(points_cut_polygons[2])
    intersection_1_2 = poly1.intersection(poly2)
    intersection_1_3 = poly1.intersection(poly3)
    coords_intersection_1_2 = get_intersection_coords(intersection_1_2)
    coords_intersection_1_3 = get_intersection_coords(intersection_1_3)
    int_poly1 = get_polygon(coords_intersection_1_2)
    int_poly2 = get_polygon(coords_intersection_1_3)
    fix_intersection = int_poly1.intersection(int_poly2)
    fix_coords = get_intersection_coords(fix_intersection)
    return fix_coords

def create_line(points,line_color='black',line_weight=5,line_popup=None,dash_weight=None):
    line = folium.PolyLine(locations = points,
                    color = line_color,
                    weight = line_weight,
                    dash_array = dash_weight,
                    tooltip = line_popup
                    )
    return line

def plot_LOB(points_lob_polygon,azimuth,sensor_error,min_distance_m,max_distance_m):
    lob_description = f'{azimuth}° with {sensor_error}° RMS error from {min_distance_m/1000:.1f} to {max_distance_m/1000:.1f}km ({get_polygon_area(points_lob_polygon):,.0f} acres of error)'
    lob_polygon = create_polygon(points_lob_polygon,'black','blue',2,lob_description)
    return lob_polygon

def create_lob_cluster(lob_cluster,points_lob_polygons,azimuths,sensor_error,min_distance_m,max_distance_m):
    for index,plp in enumerate(points_lob_polygons):
        lob_polygon = plot_LOB(plp,azimuths[index],sensor_error,min_distance_m,max_distance_m)
        lob_cluster.add_child(lob_polygon)
    return lob_cluster

def plot_grid_lines(center_coord):
    precision = 5; num_gridlines = 10
    def round_to_nearest_km(num):
        km = str(round(int(num)*10**(-1*(len(num)-2)),0)).replace('.','')
        while len(km) < len(num):
            km += '0'
        return km
    def round_coord_to_nearest_mgrs_km2(center_coord,precision=5):
        center_coord_preamble = convert_coords_to_mgrs(center_coord,precision)[:precision*-2]
        easting = convert_coords_to_mgrs(center_coord,precision)[precision*-2:precision*-1]
        northing = convert_coords_to_mgrs(center_coord,precision)[precision*-1:]
        return center_coord_preamble+round_to_nearest_km(easting)+round_to_nearest_km(northing)
    center_coord_preamble = convert_coords_to_mgrs(center_coord,precision)[:precision*-2]
    easting = convert_coords_to_mgrs(center_coord,precision)[precision*-2:precision*-1]
    northing = convert_coords_to_mgrs(center_coord,precision)[precision*-1:]
    center_mgrs = round_coord_to_nearest_mgrs_km2(center_coord)
    base_points_hl = [(center_coord,center_coord_preamble+round_to_nearest_km(easting)+round_to_nearest_km(northing))]
    base_points_vl = [(center_coord,center_coord_preamble+round_to_nearest_km(easting)+round_to_nearest_km(northing))]
    for i in range(num_gridlines):
        try:
            right_mgrs = center_mgrs[:precision*-2]+str(int(center_mgrs[precision*-2:precision*-1])+(1000*(i+1)))+center_mgrs[precision*-1:]
            base_points_hl.append((convert_mgrs_to_coords(right_mgrs),right_mgrs))
        except:
            old_right_mgrs = right_mgrs[precision*-2:]
            right_mgrs = round_coord_to_nearest_mgrs_km2([center_coord[0],adjust_long(center_coord,1*(i+2))])
            new_mgrs_preamble = right_mgrs[:precision*-2]
            base_points_vl.append((convert_mgrs_to_coords(new_mgrs_preamble+old_right_mgrs),new_mgrs_preamble+old_right_mgrs))
            base_points_vl.append((convert_mgrs_to_coords(right_mgrs),right_mgrs))  
        try:
            left_mgrs = center_mgrs[:precision*-2]+str(int(center_mgrs[precision*-2:precision*-1])-(1000*(i+1)))+center_mgrs[precision*-1:]
            base_points_hl.append((convert_mgrs_to_coords(left_mgrs),left_mgrs))
        except:
            old_left_mgrs = left_mgrs[precision*-2:]
            left_mgrs = round_coord_to_nearest_mgrs_km2([center_coord[0],adjust_long(center_coord,-1*(i+2))])
            new_mgrs_preamble = left_mgrs[:precision*-2]
            base_points_vl.append((convert_mgrs_to_coords(new_mgrs_preamble+old_left_mgrs),new_mgrs_preamble+old_left_mgrs))
            base_points_vl.append((convert_mgrs_to_coords(left_mgrs),left_mgrs))            
        try:
            up_mgrs = center_mgrs[:precision*-2]+center_mgrs[precision*-2:precision*-1]+str(int(center_mgrs[precision*-1:])+(1000*(i+1)))
            base_points_vl.append((convert_mgrs_to_coords(up_mgrs),up_mgrs))
        except:
            old_up_mgrs = up_mgrs[precision*-2:]
            up_mgrs = round_coord_to_nearest_mgrs_km2([adjust_lat(center_coord,1*(i+2)),center_coord[1]])
            new_mgrs_preamble = up_mgrs[:precision*-2]
            base_points_vl.append((convert_mgrs_to_coords(new_mgrs_preamble+old_up_mgrs),new_mgrs_preamble+old_up_mgrs))
            base_points_vl.append((convert_mgrs_to_coords(up_mgrs),up_mgrs))
        try:
            down_mgrs = center_mgrs[:precision*-2]+center_mgrs[precision*-2:precision*-1]+str(int(center_mgrs[precision*-1:])-(1000*(i+1)))
            base_points_vl.append((convert_mgrs_to_coords(down_mgrs),down_mgrs))
        except:
            old_down_mgrs = down_mgrs[precision*-2:]
            down_mgrs = round_coord_to_nearest_mgrs_km2([adjust_lat(center_coord,-1*(i+2)),center_coord[1]])
            new_mgrs_preamble = down_mgrs[:precision*-2]
            base_points_vl.append((convert_mgrs_to_coords(new_mgrs_preamble+old_down_mgrs),new_mgrs_preamble+old_down_mgrs))
            base_points_vl.append((convert_mgrs_to_coords(down_mgrs),down_mgrs))
    mgrs_grids = [item[1] for item in base_points_vl] + [item[1] for item in base_points_hl]
    eastings = sorted(list(set([vl[1][precision*-2:precision*-1] for vl in base_points_vl] + [hl[1][precision*-2:precision*-1] for hl in base_points_hl])))
    northings = sorted(list(set([vl[1][precision*-1:] for vl in base_points_vl] + [hl[1][precision*-1:] for hl in base_points_hl])))
    mgrs_preambles = sorted(list(set([vl[1][:precision*-2] for vl in base_points_vl] + [hl[1][:precision*-2] for hl in base_points_hl])))
    gridPoints = []
    for e in eastings:
        for n in northings:
            for pa in mgrs_preambles:
                if pa+e+n in mgrs_grids:
                    gridPoints.append(pa+e+n)
                    break
    lines = []
    for e in eastings:
        pa_index = 0
        while True:
            try:
                p1 = convert_mgrs_to_coords(mgrs_preambles[pa_index]+e+min(northings))
                break
            except:
                if pa_index < len(mgrs_preambles):
                    pa_index +=1 
                else:
                    break
        pa_index = 0
        while True:
            try:
                p2 = convert_mgrs_to_coords(mgrs_preambles[pa_index]+e+max(northings))
                break
            except:
                if pa_index < len(mgrs_preambles):
                    pa_index +=1 
                else:
                    break
        lines.append(create_line([p1,p2],'black',2,f'Easting: {e[:2]}'))
    for n in northings:
        pa_index = 0
        while True:
            try:
                p1 = convert_mgrs_to_coords(mgrs_preambles[pa_index]+min(eastings)+n)
                break
            except:
                if pa_index < len(mgrs_preambles):
                    pa_index +=1 
                else:
                    break
        pa_index = 0
        while True:
            try:
                p2 = convert_mgrs_to_coords(mgrs_preambles[pa_index]+max(eastings)+n)
                break
            except:
                if pa_index < len(mgrs_preambles):
                    pa_index +=1 
                else:
                    break 
        lines.append(create_line([p1,p2],'black',2,f'Northing: {n[:2]}'))
    return lines

## TEST ###
# if __name__ == '__main__':
#     coord = [31.976157948608897, -81.82428711467796]
#     down_mgrs = '17SMR2200040000'
#     center_coord_preamble = down_mgrs[:5*-2]
#     center_coord = [32.01516613971403, -81.82400529485693]
#     round_to_nearest_km2(center_coord_preamble,convert_coords_to_mgrs([center_coord[0],center_coord[1]-convert_meters_to_coordinates(1000*(2))]))
    
#     m = create_map(center_coord,zoom_start=14)
#     lines = plot_grid_lines(center_coord)
#     mgrs_gridzone = MarkerCluster(name='MGRS Gridlines',show=False)
#     mgrs_gridzone.add_to(m)
#     for l in lines:
#           mgrs_gridzone.add_child(l)
#     m.save("C:\\Users\\brend\\Documents\\Coding Projects\\CEMA\\Maps\\test.html")




