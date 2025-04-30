import pandas as pd
from datetime import datetime
from meteostat import Daily, Point
import logging
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('weather_imputation.log'),
        logging.StreamHandler()
    ]
)

def fetch_meteostat_data():
    """Fetch weather data from Meteostat API for 2012-2016."""
    logging.info("Fetching Meteostat data...")
    
    # Define borough coordinates
    boroughs = {
        'Manhattan': (40.776676, -73.971321),
        'Brooklyn': (40.650002, -73.949997),
        'Queens': (40.742054, -73.769417),
        'Staten Island': (40.579021, -74.151535),
        'Bronx': (40.837048, -73.865433)
    }
    
    # Define date range
    start_date = datetime(2012, 1, 1)
    end_date = datetime(2016, 1, 1)
    
    all_boroughs_weather = []
    
    for borough, (lat, lon) in boroughs.items():
        logging.info(f"Fetching Meteostat data for {borough}...")
        location = Point(lat, lon)
        data = Daily(location, start_date, end_date)
        df = data.fetch().reset_index()
        df['borough'] = borough
        all_boroughs_weather.append(df)
    
    weather_df = pd.concat(all_boroughs_weather, ignore_index=True)
    
    # Rename columns to match our dataset
    weather_df = weather_df.rename(columns={
        'time': 'crash_date',
        'tavg': 'tavg',
        'tmin': 'tmin',
        'tmax': 'tmax',
        'prcp': 'prcp',
        'snow': 'snow',
        'wdir': 'wdir',
        'wspd': 'wspd',
        'pres': 'pres'
    })
    
    # Convert date to datetime
    weather_df['crash_date'] = pd.to_datetime(weather_df['crash_date'])
    
    return weather_df

def safe_date_convert(date_str):
    """Safely convert date string to datetime, handling different formats."""
    try:
        return pd.to_datetime(date_str, format='%Y-%m-%d %H:%M:%S')
    except ValueError:
        try:
            return pd.to_datetime(date_str, format='%Y-%m-%d')
        except ValueError:
            logging.warning(f"Could not parse date: {date_str}")
            return pd.NaT

def impute_weather_data():
    """Load the current dataset and impute missing weather values."""
    logging.info("Loading current dataset...")
    
    # Load the current dataset with low_memory=False to handle mixed types
    df = pd.read_csv('../static_data/processed/v4_balanced_traffic_weather_intersections.csv', low_memory=False)
    
    # Convert crash_date column safely
    df['crash_date'] = df['crash_date'].apply(safe_date_convert)
    
    # Drop rows with invalid dates
    invalid_dates = df['crash_date'].isna()
    if invalid_dates.any():
        logging.warning(f"Dropping {invalid_dates.sum()} rows with invalid dates")
        df = df[~invalid_dates]
    
    # Fetch Meteostat data
    weather_df = fetch_meteostat_data()
    
    # Create a mask for rows that need weather data (pre-2016)
    pre_2016_mask = df['crash_date'] < datetime(2016, 1, 1)
    
    # Create a mapping of weather data by date and borough
    weather_dict = {}
    for _, row in weather_df.iterrows():
        key = (row['crash_date'].date(), row['borough'])
        weather_dict[key] = {
            'tavg': row['tavg'],
            'tmin': row['tmin'],
            'tmax': row['tmax'],
            'prcp': row['prcp'],
            'snow': row['snow'],
            'wdir': row['wdir'],
            'wspd': row['wspd'],
            'pres': row['pres']
        }
    
    # Impute missing weather values
    logging.info("Imputing missing weather values...")
    imputed_count = 0
    missing_count = 0
    
    for idx, row in df[pre_2016_mask].iterrows():
        key = (row['crash_date'].date(), row['weather_borough'])
        if key in weather_dict:
            weather_data = weather_dict[key]
            for col in ['tavg', 'tmin', 'tmax', 'prcp', 'snow', 'wdir', 'wspd', 'pres']:
                df.at[idx, col] = weather_data[col]
            imputed_count += 1
        else:
            missing_count += 1
    
    # Save the updated dataset
    logging.info("Saving updated dataset...")
    df.to_csv('../static_data/processed/v5_balanced_traffic_weather_intersections.csv', index=False)
    
    # Log statistics
    total_rows = len(df)
    imputed_rows = pre_2016_mask.sum()
    logging.info(f"Total rows processed: {total_rows}")
    logging.info(f"Rows with imputed weather data: {imputed_count}")
    logging.info(f"Rows with missing weather data: {missing_count}")
    logging.info(f"Imputation complete!")

if __name__ == "__main__":
    impute_weather_data() 