import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from concurrent import futures
from visualization import station_locations_vis as vis_stations
from data import journey_data_preprocessing as preprocess


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



def add_borough_demographic_features(bike_locs, features_to_add):
    """
    This function reads various external data files to add demographic features like population density, age, gender, etc., based on the provided list of features.
    Each feature corresponds to a specific demographic indicator and is added to the DataFrame via merging.

    Args:
    bike_locs (DataFrame): DataFrame providing locations of bike stations.
    features_to_add (list): List of features to be added.

    Returns:
    DataFrame: DataFrame with demographic features for each borough.

    """
    # add bike station counts per borough
    borough_df = vis_stations.count_stations_per_borough(bike_locs)
    borough_df = borough_df.reset_index()
    borough_df.columns = ['borough', 'bike_station_counts']

    # add bike docks counts per borough
    docks = vis_stations.count_docks_per_borough(bike_locs)
    docks = docks.reset_index()
    docks.columns = ['borough', 'bike_docks_counts']
    borough_df = borough_df.merge(docks, on='borough', how='left')

    # add borough codes
    borough_codes = pd.read_csv('../data/external/borough_code_mapping.csv')
    borough_df = borough_df.merge(borough_codes, on='borough', how='left')


    # add additional demographic features
    if 'TS006' in features_to_add:
        TS006_df = pd.read_csv('../data/external/TS006_population_density.csv')
        borough_df = borough_df.merge(TS006_df, on='borough_code', how='left')

    if 'TS007' in features_to_add:
        TS007_df = pd.read_csv('../data/external/TS007_age.csv')
        borough_df = _melte_and_merge(borough_df, TS007_df, 'age_total', 'age', 'borough_code', _compute_age_stats)

    if 'TS008' in features_to_add:
        TS008_df = pd.read_csv('../data/external/TS008_gender.csv')
        TS008_df['female_ratio'] = TS008_df['female'] / TS008_df['all']
        borough_df = borough_df.merge(TS008_df[['borough_code', 'female_ratio']], on='borough_code', how='left')

    if 'TS017' in features_to_add:  
        TS017_df = pd.read_csv('../data/external/TS017_household_size.csv')
        TS017_df['householdsize_3-5'] = TS017_df[[f'householdsize_{i}' for i in range(3, 6)]].sum(axis=1)
        TS017_df['householdsize_6+'] = TS017_df[[f'householdsize_{i}' for i in range(6, 9)]].sum(axis=1)
        groups_TS017 = ['householdsize_1', 'householdsize_2', 'householdsize_3-5', 'householdsize_6+']
        borough_df = _compute_ratios_and_merge(borough_df, TS017_df, groups_TS017, 'all')
    
    if 'TS021' in features_to_add:
        TS021_df = pd.read_csv('../data/external/TS021_ethnic_group.csv')
        groups_TS021 = ['ethnic_asian', 'ethnic_african_caribbean', 'ethnic_mixed', 'ethnic_white', 'ethnic_arab_other']
        borough_df = _compute_ratios_and_merge(borough_df, TS021_df, groups_TS021, 'ethnic_all')

    if 'TS030' in features_to_add:
        TS030_df = pd.read_csv('../data/external/TS030_religion.csv')
        groups_TS030 = ['religion_no', 'religion_christian', 'religion_buddhist', 'religion_hindu', 'religion_jewish', 'religion_muslim', 'religion_sikh']
        borough_df = _compute_ratios_and_merge(borough_df, TS030_df, groups_TS030, 'religion_all')

    if 'TS067' in features_to_add:
        TS067_df = pd.read_csv('../data/external/TS067_education.csv')
        groups_TS067 = ['highes_education_no', 'highes_education_l1', 'highes_education_l2', 'highes_education_apprenticeship', 'highes_education_l3', 'highes_education_l4']
        borough_df = _compute_ratios_and_merge(borough_df, TS067_df, groups_TS067, 'highes_education_all')

    if 'TS037' in features_to_add:
        TS037_df = pd.read_csv('../data/external/TS037_general_health.csv')
        groups_TS037 = ['health_1', 'health_2', 'health_3', 'health_4', 'health_5']
        borough_df = _compute_ratios_and_merge(borough_df, TS037_df, groups_TS037, 'health_all')

    if 'ADD001' in features_to_add:
        ADD001_df = pd.read_csv('../data/external/AD001_green_blue_cover.csv')
        borough_df = borough_df.merge(ADD001_df[["green_cover_ratio",'blue_cover_ratio', "borough_code"]], on='borough_code', how='left')

    if 'ADD007' in features_to_add:
        ADD007_df = pd.read_csv('../data/external/ADD007_sports_participation_rates.csv')
        borough_df = borough_df.merge(ADD007_df[["sports_participation_ratio", 'borough']], on='borough', how='left')

    if 'ADD012' in features_to_add:
        ADD012_df = pd.read_csv('../data/external/ADD012_crime_offences_rate.csv')
        borough_df = borough_df.merge(ADD012_df[["crime_offences_rate", 'borough']], on='borough', how='left')

    if 'ADD003' in features_to_add:
        ADD003_df = pd.read_csv('../data/external/ADD003_business_density.csv')
        borough_df = borough_df.merge(ADD003_df[["business_density", 'borough_code']], on='borough_code', how='left')
    
    if 'TS058' in features_to_add:
        TS058_df = pd.read_csv('../data/external/TS058_travel_to_work.csv')
        TS058_df['distance_work_20km_more'] = TS058_df['distance_work_20km_30km'] + TS058_df['distance_work_30km_40km'] + TS058_df['distance_work_40km_60km'] + TS058_df['distance_work_60km_more']
        groups_TS058 = ['distance_work_less_2km', 'distance_work_2km_5km', 'distance_work_5km_10km', 'distance_work_10km_20km', 'distance_work_20km_more', 'distance_work_homeoffice', 'distance_work_no_fix_place']
        borough_df = _compute_ratios_and_merge(borough_df, TS058_df, groups_TS058, 'distance_work_all')

    if 'ADD008' in features_to_add:
        ADD008_df = pd.read_csv('../data/external/ADD008_road_traffic_area.csv')
        ADD008_1_df = pd.read_csv('../data/external/ADD008_road_traffic_volume.csv')
        ADD008_df = ADD008_df.merge(ADD008_1_df, on='borough_code', how='left')
        ADD008_df['road_traffic_ratio'] = ADD008_df['road_traffic_volume'] / ADD008_df['road_traffic_area'] 
        borough_df = borough_df.merge(ADD008_df[["road_traffic_ratio", 'borough_code']], on='borough_code', how='left')

    if 'ADD009' in features_to_add:
        ADD009_df = pd.read_csv('../data/external/ADD009_healthy_streets_score.csv')
        borough_df = borough_df.merge(ADD009_df[["street_health_score", 'borough']], on='borough', how='left')

    if 'ADD002' in features_to_add:
        ADD002_df = pd.read_csv('../data/external/ADD002_house_price_avg.csv')
        ADD002_df = ADD002_df.groupby('borough_code')['house_price_avg'].mean().reset_index()
        borough_df = borough_df.merge(ADD002_df, on='borough_code', how='left')

    if 'TS045' in features_to_add:
        TS045_df = pd.read_csv('../data/external/TS045_car.csv')
        TS045_df['car_household_ratio'] = (TS045_df['cars_1'] + TS045_df['cars_2'] + TS045_df['cars_3']) / TS045_df['cars_all']
        borough_df = borough_df.merge(TS045_df[["car_household_ratio", 'borough_code']], on='borough_code', how='left')

    if 'TS044' in features_to_add:
        ADD044_df = pd.read_csv('../data/external/TS044_accommodation_type.csv')
        ADD044_df['accommodation_house'] = ADD044_df['Accommodation type: Detached'] + ADD044_df['Accommodation type: Semi-detached'] + ADD044_df['Accommodation type: Terraced']
        ADD044_df['accommodation_flat'] = ADD044_df['Accommodation type: In a purpose-built block of flats or tenement'] + ADD044_df['Accommodation type: Part of a converted or shared house, including bedsits'] + ADD044_df['Accommodation type: Part of another converted building, for example, former school, church or warehouse'] + ADD044_df['Accommodation type: In a commercial building, for example, in an office building, hotel or over a shop']
        ADD044_df['accommodation_mobile'] = ADD044_df['Accommodation type: A caravan or other mobile or temporary structure']
        groups_TS044 = ['accommodation_house', 'accommodation_flat', 'accommodation_mobile']
        borough_df = _compute_ratios_and_merge(borough_df, ADD044_df, groups_TS044, 'accommodation_all')

    if 'TS054' in features_to_add:
        ADD054_df = pd.read_csv('../data/external/TS054_tenure.csv')
        ADD054_df['tenure_owned_sharedowned'] = ADD054_df['tenure_owned'] + ADD054_df['tenure_owned_outright'] + ADD054_df['tenure_owned_mortage'] + ADD054_df['tenure_owned_shared']
        groups_ADD054 = ['tenure_owned_sharedowned']
        borough_df = _compute_ratios_and_merge(borough_df, ADD054_df, groups_ADD054, 'tenure_all')

    if 'TS038' in features_to_add:
        ADD038_df = pd.read_csv('../data/external/TS038_disability.csv')
        groups_ADD038 = ['disability']
        borough_df = _compute_ratios_and_merge(borough_df, ADD038_df, groups_ADD038, 'disability_all')

    if 'TS016' in features_to_add:
        TS016_df = pd.read_csv('../data/external/TS016_length_residence.csv')
        TS016_df['residence_lengh_10yr_less'] = TS016_df['residence_lengh_2yr_less'] + TS016_df['residence_lengh_2yr_5yr'] + TS016_df['residence_lengh_5yr_10yr']
        groups_TS016 = ['residence_lengh_uk_born', 'residence_lengh_10yr_plus', 'residence_lengh_10yr_less']
        borough_df = _compute_ratios_and_merge(borough_df, TS016_df, groups_TS016, 'residence_length_all')

    if 'ADD006' in features_to_add:
        ADD006_df = pd.read_csv('../data/external/ADD006_personal_well_being.csv')
        borough_df = borough_df.merge(ADD006_df, on='borough_code', how='left')

    if 'ADD011' in features_to_add:
        ADD011_df = pd.read_csv('../data/external/ADD011_local_election_2018.csv')
        borough_df = borough_df.merge(ADD011_df, on='borough', how='left')

    if 'TS062' in features_to_add:
        TS062_df = pd.read_csv('../data/external/TS062_socio_economic_classification.csv')
        TS062_df['occupation_high_level_ratio'] = TS062_df['occupation_L1_L2_L3'] + TS062_df['occupation_L4_L5_L6']
        TS062_df['occupation_small_intermediate_ratio'] = TS062_df['occupation_L7'] + TS062_df['occupation_L8_L9']
        TS062_df['occupation_lower_level_ratio'] = TS062_df['occupation_L10_L11'] + TS062_df['occupation_L12'] + TS062_df['occupation_L13']
        TS062_df['occupation_unemployed_ratio'] = TS062_df['occupation_L14']
        TS062_df['occupation_student_ratio'] = TS062_df['occupation_L15']
        groups_TS062 = ['occupation_high_level_ratio', 'occupation_small_intermediate_ratio', 'occupation_lower_level_ratio', 'occupation_unemployed_ratio', 'occupation_student_ratio']
        borough_df = _compute_ratios_and_merge(borough_df, TS062_df, groups_TS062, 'occupation_all')

    if 'ADD004' in features_to_add:
            ADD004_df = pd.read_csv('../data/external/ADD004_earnings_workplace_borough.csv')
            borough_df = borough_df.merge(ADD004_df, on='borough_code', how='left')
            borough_df['earnings_workplace'] = pd.to_numeric(borough_df['earnings_workplace'], errors='coerce')

    
    return borough_df


def _compute_ratios_and_merge(original_df, new_df, groups, total_column_name):
    """
    Computes ratio for each group and merges the new data with the original DataFrame.

    Args:
        original_df (DataFrame): Original DataFrame.
        new_df (DataFrame): New DataFrame with group and total column data.
        groups (list): List of groups for which to compute ratios.
        total_column_name (str): The name of the total column.

    Returns:
        DataFrame: The merged DataFrame with computed ratio columns.
    """
    # calculate ratio for each group
    for group in groups:
        new_df[f'{group}_ratio'] = new_df[group] / new_df[total_column_name]

    # keep only the ratio columns
    ratio_columns = [f'{group}_ratio' for group in groups]
    new_df = new_df[['borough_code'] + ratio_columns]

    # merge with original dataframe
    merged_df = original_df.merge(new_df, on='borough_code', how='left')
    
    return merged_df



def _melte_and_merge(original_df, new_df, total_col, value_col, groupby_col, compute_stats):
    """
    Transforms and merges a new DataFrame with the original DataFrame.

    Args:
        original_df (DataFrame): Original DataFrame.
        new_df (DataFrame): New DataFrame with value and total columns.
        total_col (str): The name of the total column in new_df.
        value_col (str): The name of the value column to be created.
        groupby_col (str): The column to group by.
        compute_stats (func): Function to compute statistics on grouped data.

    Returns:
        DataFrame: The merged DataFrame with computed statistics.
    """
    value_cols = new_df.columns[2:].str.replace(f'{value_col}_', '').astype(int)
    new_df.columns = list(new_df.columns[:2]) + list(value_cols)

    # melt the dataframe to have values in one column and their frequencies in another
    melted_df = new_df.melt(id_vars=[groupby_col, total_col], var_name=value_col, value_name='frequency')
    melted_df[value_col] = melted_df[value_col].astype(int)

    # group by borough and apply function
    value_stats = melted_df.groupby(groupby_col).apply(compute_stats)

    # merge with original dataframe
    merged_df = original_df.merge(value_stats, left_on=groupby_col, right_index=True, how='left')
    
    return merged_df


def _compute_age_stats(df):
    """
    Computes specific age statistics - mean, 25th percentile and 75th percentile - from a DataFrame.
    """
    # Replicate each age value according to its frequency
    ages = np.repeat(df['age'], df['frequency'])

    # Calculate statistics
    age_mean = np.mean(ages)
    age_25_percentile = np.percentile(ages, 25)
    age_75_percentile = np.percentile(ages, 75)

    return pd.Series({'age_mean': age_mean, 
                      'age_25_percentile': age_25_percentile, 
                      'age_75_percentile': age_75_percentile})



def map_journey_borough_data(start_date, end_date, journey_df, borough_df, date_for_filename):
    """
    Maps borough data to filtered journey data by date, and saves the preprocessed and mapped data to csv files.

    Args:
        start_date (str): The start date filter.
        end_date (str): The end date filter.
        journey_df (DataFrame): DataFrame with journey data.
        borough_df (DataFrame): DataFrame with borough data.
        date_for_filename (str): The date to be included in the filename of the saved csv files.

    Returns:
        None
    """
    journey_df = preprocess.filter_date(journey_df, start_date, end_date)
    journey_df_mapped_start = journey_df.merge(borough_df, left_on='start_borough', right_on='borough', how='left')
    journey_df_mapped_start.to_csv(f'../data/processed/journey_data_{date_for_filename}.csv')
