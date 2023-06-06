import folium
import matplotlib.pyplot as plt
import geopandas as gpd
import branca.colormap as cm


def bar_chart_stations_per_borough(color_dict, borough_stat_counts, borough_docks_counts):
    """
    This function creates a bar chart representing the number of bike stations per borough.

    Args:
        color_dict (dict): A dictionary mapping each borough to a specific color for the bar chart.
        borough_counts (pd.Series): A pandas series where the index represents the borough names
                                     and the values represent the count of bike stations.

    Returns:
        plt (matplotlib.pyplot): A matplotlib object with the created bar chart.
    """

    fig = plt.figure(figsize=(18, 6))

    # First subplot
    ax1 = fig.add_subplot(1, 2, 1)
    bars = ax1.bar(borough_stat_counts.index, borough_stat_counts.values, color=[color_dict[borough] for borough in borough_stat_counts.index], alpha=0.75)
    ax1.set_title('Number of Bike Stations per Borough')
    ax1.set_ylabel('Number of Stations')
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.3, wspace=0.5)

    # Second subplot
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(borough_docks_counts, borough_stat_counts)
    ax2.set_title('Number of Bike Stations vs Docks per Borough')
    ax2.set_ylabel('Number of Stations')
    ax2.set_xlabel('Number of Docks')


    return plt



def create_borough_station_map(bike_locs, markerColor, color_dict, boroughs_geojson_path):
    """
    Create a map with markers representing bike stations colored based on the borough.

    Args:
        bike_locs (pandas.DataFrame): DataFrame containing bike station data.
        markerColor (str): Color of the markers.

    Returns:
        amap (folium.Map): Map with markers representing bike stations colored based on the borough.

    """

    # create borough map
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

    map = folium.Map(location=(51.5074, -0.1278), zoom_start=11, tiles='Stamen Toner')

    boroughs_geojson = gpd.read_file(boroughs_geojson_path)
    folium.GeoJson(boroughs_geojson, name='geojson', style_function=style_function).add_to(map)


    # add locations
    for index, row in bike_locs.iterrows():
        popup_text = f"{row['name']} (borough: {row['borough']}))"
        marker = folium.CircleMarker(location=(row['lat'], row['lon']), radius=1, popup=popup_text, color=markerColor, fill_opacity=0.7)
        marker.add_to(map)
    
    return map
    


def count_stations_per_borough(bike_locs):
    """
    Count the number of bike stations per borough.

    Args:
        bike_locs (pandas.DataFrame): DataFrame containing bike station data.

    Returns:
        pandas.Series: Series containing the number of bike stations per borough, sorted in descending order.

    """

    borough_counts = bike_locs.groupby('borough').size()
    borough_counts = borough_counts.sort_values(ascending=False)

    return borough_counts


def count_docks_per_borough(bike_locs):
    """
    Count the number of docks per borough.

    Args:
        bike_locs (pandas.DataFrame): DataFrame containing bike station data.

    Returns:
        pandas.Series: Series containing the number of docks per borough, sorted in descending order.

    """

    borough_counts = bike_locs.groupby('borough')['nr_of_docks'].sum()

    return borough_counts


def create_color_borough_mapping(colormap_density, borough_counts):
    """
    Create a color mapping dictionary for boroughs based on station counts.

    Args:
        colormap_density (list): List of colors representing the density scale.
        borough_counts (pandas.Series): Series containing the number of bike stations per borough.

    Returns:
        dict: Dictionary mapping borough names to color values.

    """
        
    max_count = borough_counts.max()

    def map_count_to_color_index(count, max_count):
        index = int(count / max_count * (len(colormap_density) - 1))
        return index

    color_dict_density = borough_counts.apply(lambda x: colormap_density[map_count_to_color_index(x, max_count)]).to_dict()
    
    return color_dict_density