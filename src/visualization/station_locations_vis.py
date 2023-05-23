import folium
import matplotlib.pyplot as plt
import geopandas as gpd
import random
import branca.colormap as cm


def create_station_map(bike_locs, london_coords):
    """
    This function creates an interactive map showing the locations of bike stations in London. 
    It adds markers for each station and creates a heatmap layer based on the stations' geographical coordinates.

    Args:
        bike_locs (pd.DataFrame): A pandas dataframe containing the bike station locations. 
                                  It must have 'id', 'name', 'borough', 'lat', and 'lon' columns.
        london_coords (tuple): A tuple containing the latitude and longitude of London for centering the map.

    Returns:
        map (folium.folium.Map): A folium map object with added markers for each bike station and a heatmap layer.
    """

    map = folium.Map(location=london_coords, zoom_start=12, tiles='Stamen Toner')

    # add markers for each bike station location
    for index, row in bike_locs.iterrows():
        popup_text = f"{row['name']} (id: {row['id']}) (borough: {row['borough']})"
        marker = folium.CircleMarker(location=(row['lat'], row['lon']), popup=popup_text)
        marker.add_to(map)

    # add a heatmap layer
    heat_data = [[row['lat'], row['lon']] for index, row in bike_locs.iterrows()]
    heatmap = folium.FeatureGroup(heat_data)
    heatmap.add_to(map)

    return map



def bar_chart_stations_per_borough(color_dict, borough_counts):
    """
    This function creates a bar chart representing the number of bike stations per borough.

    Args:
        color_dict (dict): A dictionary mapping each borough to a specific color for the bar chart.
        borough_counts (pd.Series): A pandas series where the index represents the borough names
                                     and the values represent the count of bike stations.

    Returns:
        plt (matplotlib.pyplot): A matplotlib object with the created bar chart.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(borough_counts.index, borough_counts.values, color=[color_dict[borough] for borough in borough_counts.index], alpha=0.75)

    ax.set_title('Number of Bike Stations per Borough')
    ax.set_ylabel('Number of Stations')
    plt.xticks(rotation=45, ha='right')

    plt.subplots_adjust(bottom=0.3)

    return plt



def create_borough_map(london_coords, boroughs_geojson_path, color_dict):
    """
    Create a map of London boroughs with colored boundaries based on the provided density color dictionary.

    Args:
        color_dict_density (dict): Dictionary mapping borough names to color values.
        london_coords (tuple): Tuple of latitude and longitude specifying the center coordinates of London.
        boroughs_geojson_path (str): Path to the GeoJSON file containing London borough boundaries.
        save_path (str): Path to save the resulting HTML map file.

    Returns:
        None
    """
    
    default_color = '#999999'

    def style_function(feature):
        borough_name = feature['properties']['name']
        if borough_name in color_dict:
            fill_color = color_dict[borough_name]
        else:
            fill_color = default_color
        return {
            'fillColor': fill_color,
            'color': 'black',
            'weight': 2,
            'fillOpacity': 0.6,
        }

    map = folium.Map(location=london_coords, zoom_start=11, tiles='Stamen Toner')

    boroughs_geojson = gpd.read_file(boroughs_geojson_path)
    folium.GeoJson(boroughs_geojson, name='geojson', style_function=style_function).add_to(map)

    return map


def create_borough_station_map(bike_locs, base_map, markerColor):

    for index, row in bike_locs.iterrows():
        popup_text = f"{row['name']} (borough: {row['borough']}))"
        marker = folium.CircleMarker(location=(row['lat'], row['lon']), radius=1, popup=popup_text, color=markerColor, fill_opacity=0.7)
        marker.add_to(base_map)
    
    return base_map
    




def count_stations_per_borough(bike_locs):
    borough_counts = bike_locs.groupby('borough').size()
    borough_counts = borough_counts.sort_values(ascending=False)

    return borough_counts


def create_color_borough_mapping(colormap_density, borough_counts):
    max_count = borough_counts.max()

    def map_count_to_color_index(count, max_count):
        index = int(count / max_count * (len(colormap_density) - 1))
        return index

    color_dict_density = borough_counts.apply(lambda x: colormap_density[map_count_to_color_index(x, max_count)]).to_dict()
    
    return color_dict_density