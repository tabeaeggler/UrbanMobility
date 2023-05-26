from geopy.geocoders import Nominatim
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def create_inbound_outbound_counts(journey_df):
    # Count of outbound journeys for each borough
    outbound_counts = journey_df.groupby('start_borough')['rental_id'].count().reset_index()
    outbound_counts.columns = ['borough', 'outbound']

    # Count of inbound journeys for each borough
    inbound_counts = journey_df.groupby('end_borough')['rental_id'].count().reset_index()
    inbound_counts.columns = ['borough', 'inbound']

    # Merge the outbound and inbound counts
    return pd.merge(outbound_counts, inbound_counts, on='borough')


def create_inbound_outbound_plot(borough_counts):
    # reshape the data
    borough_counts_melted = borough_counts.melt(id_vars='borough', var_name='type', value_name='count')
    borough_counts_melted = borough_counts_melted.sort_values('count', ascending=False)

    # create a grouped bar chart
    plt.figure(figsize=(8, 5))
    sns.catplot(data=borough_counts_melted, x='borough', y='count', hue='type', kind='bar', height=8, aspect=2)

    # add labels and a title
    plt.xlabel('Borough', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.title('Inbound and Outbound Counts by Borough', fontsize=16)
    plt.xticks(rotation=70)
    return plt



def create_data_flowmap(journey_df):
    # create count dataframe
    journey_df_borough= journey_df.copy()
    journey_df_borough = journey_df.groupby(['start_borough', 'end_borough']).size().reset_index(name='counts')
    journey_df_borough.to_csv('../data/interim/flow_count_per_borough.csv')

    # create borough locations dataframe
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



