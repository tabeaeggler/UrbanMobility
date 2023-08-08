import requests
import io
import urllib
import pandas as pd


def fetch_journey_data(filenames):
    """
    Fetch journey data from the given URLs and return a DataFrame.
    
    Parameters:
    filenames (list of str): A list of filenames to be appended to the base URL.

    Returns:
    df (DataFrame): The combined DataFrame of all journey data.
    """

    base_url = 'http://cycling.data.tfl.gov.uk/usage-stats/'
    url_list = (base_url + urllib.parse.quote(x) for x in filenames)
    unused_cols = ['Total duration (ms)', 'Total duration', 'Duration', 'Duration_Seconds', 'Bike Id', 'Bike number', 'Bike model']

    # loop through each URL to create requests and extract data
    temp_dfs = []
    for url in url_list:
        response = requests.get(url, verify=False, timeout=(3, 7))

        if url.endswith('.csv'):
            temp_df = pd.read_csv(io.StringIO(response.content.decode('utf-8')), usecols=lambda col: col not in unused_cols)

        elif url.endswith('.xlsx'):
            temp_df = pd.read_excel(io.BytesIO(response.content), usecols=lambda col: col not in unused_cols)

        temp_dfs.append(temp_df)

    # concatenate all temporary dataframes into a single dataframe
    df = pd.concat(temp_dfs, ignore_index=True)
    return df



def standardize_columns(df):
    """
    This function standardizes the column names in the provided dataframe 'df'
    according to a pre-defined mapping. It also changes the data type of some columns
    to facilitate further analysis. 

    Parameters:
    df (pd.DataFrame): The dataframe whose columns need to be standardized.

    Returns:
    df (pd.DataFrame): The dataframe with standardized column names and updated data types.
    """
     # remove any 'Unnamed' columns that might have been introduced during data fetching
    df = df.filter(regex='^(?!Unnamed)')

    # mapping of old column names to new standardized names
    column_names = {
        'EndStation Id': 'end_station_id',
        'End Station Id': 'end_station_id',
        'End station number': 'end_station_id',
        'StartStation Id': 'start_station_id',
        'Start Station Id': 'start_station_id',
        'Start station number': 'start_station_id',
        'EndStation Name': 'end_station_name',
        'End Station Name': 'end_station_name',
        'End station': 'end_station_name',
        'StartStation Name': 'start_station_name',
        'Start Station Name': 'start_station_name',
        'Start station': 'start_station_name',
        'Start Date': 'start_date',
        'Start date': 'start_date',
        'End Date': 'end_date',
        'End date': 'end_date',
        'Rental Id': 'rental_id',
        'Number': 'rental_id',
    }
    
    for old_name, new_name in column_names.items():
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})

    # convert data types
    id_columns = ['end_station_id', 'start_station_id', 'rental_id']
    date_columns = ['start_date', 'end_date']
    df.loc[:, id_columns] = df[id_columns].apply(pd.to_numeric, errors='coerce', downcast='integer')
    df.loc[:, date_columns] = df[date_columns].apply(pd.to_datetime, infer_datetime_format=True)

    return df



def drop_duplicates(df):
    """
    Drop duplicate rows from a DataFrame, prioritizing rows with more non-null values.

    Parameters:
        df (pandas.DataFrame): The DataFrame to remove duplicates from.

    Returns:
        pandas.DataFrame: The DataFrame with duplicate rows removed.

    """

    df = df.copy()

    # drop all samples with NaN only
    df = df.dropna(how='all')

    # drop all samples with duplicated rental_id, sort first to keep the row with the most non-null values
    df['nonnull_count'] = df.notnull().sum(axis=1)
    df = df.sort_values(by=['rental_id', 'nonnull_count'], ascending=[True, False])
    df = df.drop_duplicates(subset='rental_id', keep='first')
    df = df.drop(columns='nonnull_count')

    return df




def filter_date(df, start_date, end_date):
    """
    Filters the DataFrame to include only rows where the 'start_date'  and 'end_date' is within
    the provided range (inclusive).

    Parameters:
    df (pd.DataFrame): The DataFrame to be filtered.
    start_date (datetime): The start of the date range.
    end_date (datetime): The end of the date range.

    Returns:
    df (pd.DataFrame): The DataFrame filtered by date range.
    """
    mask = (df['start_date'] >= start_date) & (df['end_date'] <= end_date)
    df = df.loc[mask]

    return df