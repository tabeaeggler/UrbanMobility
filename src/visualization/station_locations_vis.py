import folium


def vis_all_locations(bike_locs):

    # create a map centered on London
    london_coords = (51.5074, -0.1278)
    map = folium.Map(location=london_coords, zoom_start=12, tiles='Stamen Toner')

    # add markers for each bike station location
    for index, row in bike_locs.iterrows():
        popup_text = f"{row['name']} (id: {row['id']}) (terminal id: {row['terminalId']})"
        marker = folium.CircleMarker(location=(row['lat'], row['lon']), popup=popup_text)
        marker.add_to(map)

    # add a heatmap layer
    heat_data = [[row['lat'], row['lon']] for index, row in bike_locs.iterrows()]
    heatmap = folium.FeatureGroup(heat_data)
    heatmap.add_to(map)

    return map