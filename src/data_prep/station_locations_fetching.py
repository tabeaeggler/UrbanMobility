import pandas as pd
import requests
import json


def get_station_locations():
    """
    Retrieve bike station locations from the TFL (Transport for London) API.


    Returns:
        pandas.DataFrame: A DataFrame containing bike station locations with columns: 'id', 'name', 'lat', 'lon', 'docks'.

    """
    url = "https://api.tfl.gov.uk/BikePoint/"
    response = requests.get(url)
    root = json.loads(response.text)

    data = []
    for station in root:
        nb_docks = None
        for additional_prop in station['additionalProperties']:
            if additional_prop['key'] == 'NbDocks':
                nb_docks = additional_prop['value']
                break

        station_data = {
            "id": station['id'][11:],
            "name": station['commonName'],
            "lat": station['lat'],
            "lon": station['lon'],
            "nr_of_docks": nb_docks
        }
        data.append(station_data)

    return pd.DataFrame(data)



def get_borough(lat, lon):
    """
    Function to retrieve borough name using lat and lon coordinates.
    
    This function sends a GET request to the 'findthatpostcode' API, using 
    the provided lat and lon coordinates. If the request is successful, the 
    function extracts the borough name from the response data and returns it. 
    If the request is unsuccessful, the function returns 'no borough'.
    
    Parameters:
    lat (float): Latitude coordinate of the location.
    lon (float): Longitude coordinate of the location.

    Returns:
    str: Borough name or 'no borough' if the API request is unsuccessful.
    """
    url = f'https://findthatpostcode.uk/points/{lat},{lon}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        borough = data['included'][0]['attributes']['cty_name']
        return borough
    else:
        return 'no borough'