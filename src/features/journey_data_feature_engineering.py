import pandas as pd
from rapidfuzz import fuzz
from concurrent import futures


def get_part_of_day(hour):
    """
    Given an hour of the day (in a 24-hour format), this function 
    returns a string representing the general part of the day the 
    hour falls into. The categorizations used are:
    
    - Early Morning: 5:00 to 8:59
    - Morning: 9:00 to 12:59
    - Afternoon: 13:00 to 16:59
    - Evening: 17:00 to 20:59
    - Night: 21:00 to 4:59

    Input: 
    hour: integer (0-23)

    Returns: 
    part_of_day: string (Early Morning, Morning, Afternoon, Evening, Night)
    """
    if (hour > 4) and (hour <= 8):
        return '1'
    elif (hour > 8) and (hour <= 12 ):
        return '2'
    elif (hour > 12) and (hour <= 16):
        return'3'
    elif (hour > 16) and (hour <= 20) :
        return '4'
    elif (hour > 20) or (hour <=4):
        return'5'
    


def get_season(month):
    """
    This function classifies a given month into its corresponding season based on meteorological reckoning. 
    Here's the classification used:

    - Spring: March (3) through May (5)
    - Summer: June (6) through August (8)
    - Fall: September (9) through November (11)
    - Winter: December (12) through February (2)

    Input: 
    month: integer (1-12) representing the month of the year

    Returns: 
    season: string (spring, summer, fall, winter)
    """
    if month >= 3 and month <= 5:
        return '1'
    elif month >= 6 and month <= 8:
        return '2'
    elif month >= 9 and month <= 11:
        return '3'
    else:
        return '4'
    


def clean_enhance_weather_data(weather_df):
    """
    This function cleans and enhances the weather dataframe by adding a new 
    column for daylight hours and replacing NaN values with zero.

    Parameters:
    weather_df (pd.DataFrame): The original weather dataframe.

    Returns:
    weather_df (pd.DataFrame): The cleaned and enhanced weather dataframe.
    """

    # add daylight_hours
    weather_df['sunrise'] = pd.to_datetime(weather_df['sunrise'])
    weather_df['sunset'] = pd.to_datetime(weather_df['sunset'])
    weather_df['daylight_hours'] = (weather_df['sunset'] - weather_df['sunrise']).dt.total_seconds()/ 3600
    weather_df = weather_df.drop(columns=['sunrise', 'sunset'])

    # replace NaN by 0
    weather_df = weather_df.fillna(0)

    return weather_df



def merge_weather_journey_data(journey_df, weather_df):
    """
    This function merges the journey data and weather data on the date.

    Args:
        journey_df (pd.DataFrame): A pandas dataframe containing the journey data.
        weather_df (pd.DataFrame): A pandas dataframe containing the weather data.

    Returns:
        journey_df (pd.DataFrame): A pandas dataframe after merging journey and weather data.
    """

    # add a new column 'start_date_only' in journey data that only contains the date part of the 'start_date' column
    journey_df['start_date_only'] = journey_df['start_date'].dt.date

    # merge the journey data with the weather data on the date
    journey_df = pd.merge(journey_df, weather_df, left_on='start_date_only', right_index=True, how='left')
    
    # drop the 'start_date_only' column as it's not needed anymore
    journey_df.drop(columns=['start_date_only'], inplace=True)
    
    return journey_df




def direct_borough_mapping_by_stationname(bike_locs, journey_df):
    """
    This function maps boroughs to stations in the journey dataframe using direct mapping.
    It standardizes station names by stripping white space and converting to lower case,
    then performs a 1:1 mapping between station names and boroughs.

    Parameters:
    bike_locs (pd.DataFrame): Dataframe with bike location information. It includes 'name' and 'borough' columns.
    journey_df (pd.DataFrame): Dataframe with journey information. It includes 'start_station_name' and 'end_station_name' columns.

    Returns:
    journey_df (pd.DataFrame): The updated dataframe with 'start_borough' and 'end_borough' mapped from 'start_station_name' and 'end_station_name' respectively.
    """

    # standardize by stripping white space and converting to lower case
    bike_locs['name'] = bike_locs['name'].str.strip().str.lower()

    # create dictionary for mapping
    borough_mapping = bike_locs.set_index('name')['borough'].to_dict()

    # 1:1 mapping
    journey_df['start_borough'] = journey_df['start_station_name'].str.strip().str.lower().map(borough_mapping)
    journey_df['end_borough'] = journey_df['end_station_name'].str.strip().str.lower().map(borough_mapping)

    return journey_df



def fuzzy_borough_mapping_by_stationname(bike_locs, journey_df):
    """
    This function maps boroughs to stations in the journey dataframe using fuzzy matching.
    It uses multiprocessing to perform fuzzy matching in parallel, which can significantly speed up the process.

    Parameters:
    bike_locs (pd.DataFrame): Dataframe with bike location information. It includes 'name' and 'borough' columns.
    journey_df (pd.DataFrame): Dataframe with journey information. It includes 'start_station_name' and 'end_station_name' columns.

    Returns:
    journey_df (pd.DataFrame): The updated dataframe with 'start_borough' and 'end_borough'.
    """

    station_to_borough = {row['name']: row['borough'] for _, row in bike_locs.iterrows()}  
    empty_boroughs = journey_df[(journey_df['start_borough'].isna()) | (journey_df['end_borough'].isna())]

    # function to perform fuzzy matching in parallel
    def parallel_fuzzy_match(column):
        return column.apply(_fuzzy_match, args=(station_to_borough,))

    # split the DataFrame into chunks for parallel processing
    num_parallel_tasks = 6
    chunk_size = len(empty_boroughs) // num_parallel_tasks  
    chunks = [empty_boroughs[i:i+chunk_size] for i in range(0, len(empty_boroughs), chunk_size)]

    # update the StartBorough column, process chunks in parallel
    with futures.ThreadPoolExecutor() as executor: 
        results = list(executor.map(parallel_fuzzy_match, [chunk['start_station_name'] for chunk in chunks]))

    for i, result in enumerate(results):
        chunk = chunks[i]
        chunk.loc[:, 'start_borough'] = result

    # update the EndBorough column, process chunks in parallel
    with futures.ThreadPoolExecutor() as executor:  # Use ThreadPoolExecutor for threads or ProcessPoolExecutor for processes
        results = list(executor.map(parallel_fuzzy_match, [chunk['end_station_name'] for chunk in chunks]))

    for i, result in enumerate(results):
        chunk = chunks[i]
        chunk.loc[:, 'end_borough'] = result

    # replace the rows with missing borough data in the original dataframe with the processed rows
    updated_empty_boroughs = pd.concat(chunks)
    journey_df.update(updated_empty_boroughs)

    return journey_df



def _fuzzy_match(station_name, station_to_borough, min_score=70):  
    """
    Performs fuzzy matching between a given station name and a mapping of station names to boroughs.
    
    Args:
        station_name (str): The station name to be matched.
        station_to_borough (dict): The mapping of station names to boroughs.
        min_score (int): The minimum similarity score required for a match (default: 70).
    
    Returns:
        str or None: The borough corresponding to the best fuzzy match for the station name, 
        or None if no match is found below the minimum score threshold.
    """

    if station_name is None:
        return None

    best_match = None
    best_score = 0

    for name in station_to_borough.keys():
        score = fuzz.token_sort_ratio(station_name, name)
        if score > best_score:
            best_score = score
            best_match = name

    return station_to_borough[best_match] if best_match and best_score >= min_score else None



def former_station_borough_mapping_by_region(bike_locs, journey_df):
    """
    This function is used to map boroughs to journey data where borough information is missing.
    It uses the 'region' part of the station name (e.g. street name, region) and assigns the borough that most frequently
    appears in that region from the bike_locs data.

    Args:
    bike_locs (pd.DataFrame): The DataFrame containing information about bike locations.
    journey_df (pd.DataFrame): The DataFrame containing journey information.

    Returns:
    journey_df (pd.DataFrame): The DataFrame with updated boroughs.
    """

    # add a 'region' column to bike_locs
    bike_locs['region'] = bike_locs['name'].str.split(',').str[1].str.strip()

    # group by 'region' and get the borough with the maximum counts
    region_borough = bike_locs.groupby('region')['borough'].agg(lambda x: x.value_counts().index[0])

    # convert the Series to a dictionary
    location_borough_dict = region_borough.to_dict()

    # function to get borough from dictionary
    def _get_borough_from_dict(name):
        parts = name.split(',')
        if len(parts) > 1:
            region = parts[1].strip().lower()
            borough = location_borough_dict.get(region, None)
            return borough
        else:
            return None

    journey_df.loc[journey_df['start_borough'].isna(), 'start_borough'] = journey_df.loc[journey_df['start_borough'].isna(), 'start_station_name'].apply(_get_borough_from_dict)
    journey_df.loc[journey_df['end_borough'].isna(), 'end_borough'] = journey_df.loc[journey_df['end_borough'].isna(), 'end_station_name'].apply(_get_borough_from_dict)

    return journey_df



def manual_borough_mapping(journey_df):
    """
    This function manually assigns boroughs to specific station names in the journey dataframe. 
    If a station name doesn't have a corresponding borough, it will be manually filled with 
    known borough names. Stations with still unknown boroughs are then dropped as they are test and workshop stations.

    Args:
    journey_df (pd.DataFrame): The DataFrame containing journey information.

    Returns:
    journey_df (pd.DataFrame): The DataFrame with updated boroughs.
    """
        
    # fill values manually
    journey_df.loc[journey_df['start_station_name'] == 'Hansard Mews, Shepherds Bush', 'start_borough'] = 'Hammersmith and Fulham'
    journey_df.loc[journey_df['start_station_name'] == 'Columbia Road, Weavers', 'start_borough'] = 'Tower Hamlets'
    journey_df.loc[journey_df['start_station_name'] == 'Abingdon Green, Great College Street', 'start_borough'] = 'Westminster'
    journey_df.loc[journey_df['start_station_name'] == 'Oval Way, Lambeth', 'start_borough'] = 'Lambeth'
    journey_df.loc[journey_df['start_station_name'] == 'Monier Road', 'start_borough'] = 'Newham'
    journey_df.loc[journey_df['start_station_name'] == 'Victoria and Albert Museum, Cromwell Road', 'start_borough'] = 'Kensington and Chelsea'
    journey_df.loc[journey_df['start_station_name'] == 'Monier Road, Newham', 'start_borough'] = 'Newham'
    journey_df.loc[journey_df['start_station_name'] == 'Allington street, Off Victoria Street, Westminster', 'start_borough'] = 'Westminster'
    journey_df.loc[journey_df['start_station_name'] == 'Worship Street, Hackney', 'start_borough'] = 'Hackney'
    journey_df.loc[journey_df['start_station_name'] == 'York Way, Camden', 'start_borough'] = 'Camden'
    journey_df.loc[journey_df['start_station_name'] == 'Monier Road', 'start_borough'] = 'Hackney'
    journey_df.loc[journey_df['start_station_name'] == 'Canada Water Station', 'start_borough'] = 'Southwark'

    journey_df.loc[journey_df['end_station_name'] == 'Hansard Mews, Shepherds Bush', 'end_borough'] = 'Hammersmith and Fulham'
    journey_df.loc[journey_df['end_station_name'] == 'Columbia Road, Weavers', 'end_borough'] = 'Tower Hamlets'
    journey_df.loc[journey_df['end_station_name'] == 'Abingdon Green, Great College Street', 'end_borough'] = 'Westminster'
    journey_df.loc[journey_df['end_station_name'] == 'Oval Way, Lambeth', 'end_borough'] = 'Lambeth'
    journey_df.loc[journey_df['end_station_name'] == 'Monier Road', 'end_borough'] = 'Newham'
    journey_df.loc[journey_df['end_station_name'] == 'Victoria and Albert Museum, Cromwell Road', 'end_borough'] = 'Kensington and Chelsea'
    journey_df.loc[journey_df['end_station_name'] == 'Monier Road, Newham', 'end_borough'] = 'Newham'
    journey_df.loc[journey_df['end_station_name'] == 'Allington street, Off Victoria Street, Westminster', 'end_borough'] = 'Westminster'
    journey_df.loc[journey_df['end_station_name'] == 'Worship Street, Hackney', 'end_borough'] = 'Hackney'
    journey_df.loc[journey_df['end_station_name'] == 'York Way, Camden', 'end_borough'] = 'Camden'
    journey_df.loc[journey_df['end_station_name'] == 'Monier Road', 'end_borough'] = 'Hackney'
    journey_df.loc[journey_df['end_station_name'] == 'Canada Water Station', 'end_borough'] = 'Southwark'

    # drop irrelevant stations
    journey_df = journey_df.dropna(subset=['start_borough', 'end_borough'])

    return journey_df