from geopy.geocoders import Nominatim
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def create_inbound_outbound_roundtrips_counts(journey_df):
    """
    Calculates the counts of outbound and inbound journeys for each borough.

    Parameters:
    journey_df (pandas.DataFrame): A DataFrame containing journey data, containing: 'start_borough', 'end_borough', and 'rental_id' columns.

    Returns:
    pandas.DataFrame: A DataFrame with columns 'borough', 'outbound', and 'inbound'. Each row represents a borough and includes the counts of outbound and inbound journeys.
    """

    # count of outbound journeys for each borough
    outbound_counts = journey_df.groupby('start_borough')['rental_id'].count().reset_index()
    outbound_counts.columns = ['borough', 'outbound']

    # count of inbound journeys for each borough
    inbound_counts = journey_df.groupby('end_borough')['rental_id'].count().reset_index()
    inbound_counts.columns = ['borough', 'inbound']

    # count of internal journeys for each borough
    roundtrip_borough_counts = journey_df[journey_df['start_borough'] == journey_df['end_borough']].groupby('start_borough')['rental_id'].count().reset_index()
    roundtrip_borough_counts.columns = ['borough', 'roundtrip by borough']

    # count of roundtrips for each borough
    roundtrip_station_counts = journey_df[journey_df['start_station_name'] == journey_df['end_station_name']].groupby('start_borough')['rental_id'].count().reset_index()
    roundtrip_station_counts.columns = ['borough', 'roundtrip by station']

    # merge the outbound and inbound counts
    merged_counts = pd.merge(outbound_counts, inbound_counts, on='borough')
    merged_counts = pd.merge(merged_counts, roundtrip_borough_counts, on='borough')
    merged_counts = pd.merge(merged_counts, roundtrip_station_counts, on='borough')

    return merged_counts



def create_inbound_outbound_roundtrips_plot(borough_counts):
    """
    Creates a plot of inbound, outbound, and roundtrip journey counts for each borough.

    Parameters:
    borough_counts (pandas.DataFrame): A DataFrame with columns 'borough', 'outbound', 'inbound', and 'roundtrips'.

    Returns:
    matplotlib.pyplot: A bar plot showing the counts of inbound, outbound, and roundtrip journeys for each borough.
    """

    # reshape the data
    borough_counts_melted = borough_counts.melt(id_vars='borough', var_name='type', value_name='count')
    borough_counts_melted = borough_counts_melted.sort_values('count', ascending=False)

    # create a grouped bar chart
    g = sns.catplot(data=borough_counts_melted, x='borough', y='count', hue='type', kind='bar', height=8, aspect=2, legend=False)
    plt.xlabel('Borough', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.title('Inbound, Outbound, and Roundtrips by Borough', fontsize=16)
    plt.xticks(rotation=70)
    plt.figure(figsize=(6, 4))

    g.add_legend(title="Type of Journeys", bbox_to_anchor=(0.85, 0.65), borderaxespad=0)
    
    return plt



def create_data_flowmap(journey_df):
    """
    Creates data for flowmap plot of inbound and outbound journey counts for each borough.

    Parameters:
    borough_counts (pandas.DataFrame): A DataFrame with columns 'borough', 'outbound', and 'inbound'. 

    Returns:
    matplotlib.pyplot: A bar plot showing the counts of inbound and outbound journeys for each borough.
    """

    # create and save count dataframe
    journey_df_borough= journey_df.copy()
    journey_df_borough = journey_df.groupby(['start_borough', 'end_borough']).size().reset_index(name='counts')
    journey_df_borough.to_csv('../data/interim/borough_flowmap.csv')

    # create and save borough locations dataframe
    def _get_location(borough_name):
        geolocator = Nominatim(user_agent="geoapiExercises")
        location = geolocator.geocode(borough_name + ', London', timeout=10000)
        if location is None:
            return 'NA', 'NA'
        else:
            return location.latitude, location.longitude

    def _borough_locations(boroughs):
        boroughs_df = pd.DataFrame(boroughs, columns=['Borough'])
        boroughs_df['Latitude'], boroughs_df['Longitude'] = zip(*boroughs_df['Borough'].apply(_get_location))
        return boroughs_df

    london_boroughs = ['City of London', 'Barking and Dagenham', 'Barnet', 'Bexley', 'Brent', 'Bromley', 'Camden', 'Croydon', 'Ealing', 'Enfield', 'Greenwich', 'Hackney', 'Hammersmith and Fulham', 'Haringey', 'Harrow', 'Havering', 'Hillingdon', 'Hounslow', 'Islington', 'Kensington and Chelsea', 'Kingston upon Thames', 'Lambeth', 'Lewisham', 'Merton', 'Newham', 'Redbridge', 'Richmond upon Thames', 'Southwark', 'Sutton', 'Tower Hamlets', 'Waltham Forest', 'Wandsworth', 'Westminster']
    boroughs_df = _borough_locations(london_boroughs)
    boroughs_df.to_csv('../data/interim/borough_locations.csv')

    return (journey_df_borough,boroughs_df)



