import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import logging
from tqdm import tqdm
import glob
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('balanced_dataset_generation.log'),
        logging.StreamHandler()
    ]
)

def format_time(hour, minute):
    """Format time to ensure two digits for minutes."""
    return f"{hour}:{minute:02d}"

def load_and_preprocess_data():
    """Load and preprocess the input datasets efficiently."""
    logging.info("Reading and preprocessing input datasets...")
    
    # Load crash data
    crash_data = pd.read_csv('static_data/processed/traffic_weather_intersections.csv')
    crash_data['crash_date'] = pd.to_datetime(crash_data['crash_date'])
    crash_data['is_crash'] = 1
    
    # Add temporal features to positive samples
    crash_data['day_of_week'] = crash_data['crash_date'].dt.weekday
    crash_data['month'] = crash_data['crash_date'].dt.month
    crash_data['is_weekend'] = (crash_data['day_of_week'] >= 5).astype(int)
    
    # Load and preprocess weather data
    weather_data = pd.read_csv('static_data/raw/nyc_weather_meteostat.csv')
    weather_data['time'] = pd.to_datetime(weather_data['time'])
    
    # Create a lookup dictionary for weather data by date and borough
    weather_lookup = {}
    for _, row in weather_data.iterrows():
        date = row['time'].date()
        borough = row['borough']
        weather_lookup[(date, borough)] = {
            'tavg': row['tavg'],
            'tmin': row['tmin'],
            'tmax': row['tmax'],
            'prcp': row['prcp'],
            'snow': row['snow'],
            'wdir': row['wdir'],
            'wspd': row['wspd'],
            'pres': row['pres'],
            'borough': row['borough']
        }
    
    return crash_data, weather_lookup

def generate_negative_samples(crash_data, weather_lookup, chunk_size=1000):
    """Generate negative samples in chunks to manage memory usage."""
    logging.info("Starting negative sample generation...")
    
    # Group crash data by intersection for efficient processing
    intersection_groups = crash_data.groupby('nearest_intersection_id')
    total_intersections = len(intersection_groups)
    
    # Get all available dates and boroughs for random selection
    available_dates = set(date for date, _ in weather_lookup.keys())
    available_boroughs = set(borough for _, borough in weather_lookup.keys())
    
    # Process intersections in chunks
    for chunk_start in range(0, total_intersections, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_intersections)
        chunk_intersections = list(intersection_groups.groups.keys())[chunk_start:chunk_end]
        
        negative_samples_chunk = []
        for intersection_id in tqdm(chunk_intersections, desc=f"Processing intersections {chunk_start}-{chunk_end}"):
            try:
                # Get intersection data efficiently
                intersection_data = intersection_groups.get_group(intersection_id).iloc[0]
                num_positives = len(intersection_groups.get_group(intersection_id))
                
                # Generate negative samples for this intersection
                for _ in range(num_positives):
                    # Randomly select date and time
                    random_date = random.choice(list(available_dates))
                    random_hour = random.randint(0, 23)
                    random_minute = random.randint(0, 59)
                    random_time = format_time(random_hour, random_minute)
                    
                    # Calculate temporal features
                    day_of_week = random_date.weekday()  # Monday=0, Sunday=6
                    month = random_date.month
                    is_weekend = 1 if day_of_week >= 5 else 0  # 1 for Saturday or Sunday
                    
                    # Get weather data efficiently using lookup
                    weather_key = (random_date, intersection_data['weather_borough'])
                    if weather_key in weather_lookup:
                        weather_row = weather_lookup[weather_key]
                        
                        # Create negative sample with proper null values
                        negative_sample = {
                            'crash_date': random_date,
                            'crash_time': random_time,
                            'on_street_name': np.nan,
                            'off_street_name': np.nan,
                            'number_of_persons_injured': 0,
                            'number_of_persons_killed': 0,
                            'number_of_pedestrians_injured': 0,
                            'number_of_pedestrians_killed': 0,
                            'number_of_cyclist_injured': 0,
                            'number_of_cyclist_killed': 0,
                            'number_of_motorist_injured': 0,
                            'number_of_motorist_killed': 0,
                            'contributing_factor_vehicle_1': np.nan,
                            'contributing_factor_vehicle_2': np.nan,
                            'collision_id': np.nan,
                            'vehicle_type_code1': np.nan,
                            'vehicle_type_code2': np.nan,
                            'borough_traffic': np.nan,
                            'latitude': np.nan,
                            'longitude': np.nan,
                            'location': np.nan,
                            'weather_borough': intersection_data['weather_borough'],
                            'tavg': weather_row['tavg'],
                            'tmin': weather_row['tmin'],
                            'tmax': weather_row['tmax'],
                            'prcp': weather_row['prcp'],
                            'snow': weather_row['snow'],
                            'wdir': weather_row['wdir'],
                            'wspd': weather_row['wspd'],
                            'pres': weather_row['pres'],
                            'borough_weather': weather_row['borough'],
                            'x_ny_state_plane': np.nan,
                            'y_ny_state_plane': np.nan,
                            'nearest_intersection_id': intersection_data['nearest_intersection_id'],
                            'nearest_intersection_name': intersection_data['nearest_intersection_name'],
                            'nearest_intersection_lat': intersection_data['nearest_intersection_lat'],
                            'nearest_intersection_lon': intersection_data['nearest_intersection_lon'],
                            'nearest_intersection_x_ny': intersection_data['nearest_intersection_x_ny'],
                            'nearest_intersection_y_ny': intersection_data['nearest_intersection_y_ny'],
                            'distance_to_intersection_km': intersection_data['distance_to_intersection_km'],
                            'day_of_week': day_of_week,
                            'month': month,
                            'is_weekend': is_weekend,
                            'is_crash': 0
                        }
                        
                        negative_samples_chunk.append(negative_sample)
                    else:
                        logging.warning(f"No weather data found for {weather_key}")
            
            except Exception as e:
                logging.error(f"Error processing intersection {intersection_id}: {str(e)}")
                continue
        
        # Convert chunk to DataFrame and save to temporary file
        if negative_samples_chunk:
            chunk_df = pd.DataFrame(negative_samples_chunk)
            # Ensure consistent column order with original data
            chunk_df = chunk_df[crash_data.columns]
            chunk_file = f'static_data/processed/negative_samples_chunk_{chunk_start}_{chunk_end}.csv'
            chunk_df.to_csv(chunk_file, index=False)
            logging.info(f"Saved chunk {chunk_start}-{chunk_end} with {len(chunk_df)} samples")
    
    return total_intersections

def combine_and_save_final_dataset(crash_data):
    """Combine all chunks and save the final dataset."""
    logging.info("Combining chunks and saving final dataset...")
    
    # Read all chunk files
    chunk_files = sorted(glob.glob('static_data/processed/negative_samples_chunk_*.csv'))
    negative_dfs = [pd.read_csv(f) for f in chunk_files]
    
    # Combine all negative samples
    negative_df = pd.concat(negative_dfs, ignore_index=True)
    
    # Combine with positive samples
    final_dataset = pd.concat([crash_data, negative_df], ignore_index=True)
    
    # Ensure consistent column order
    final_dataset = final_dataset[crash_data.columns]
    
    # Save final dataset
    final_dataset.to_csv('static_data/processed/v3_balanced_traffic_weather_intersections.csv', index=False)
    
    # Clean up temporary files
    for f in chunk_files:
        os.remove(f)
    
    return len(final_dataset), len(crash_data), len(negative_df)

def main():
    try:
        # Load and preprocess data
        crash_data, weather_lookup = load_and_preprocess_data()
        
        # Generate negative samples in chunks
        total_intersections = generate_negative_samples(crash_data, weather_lookup)
        
        # Combine and save final dataset
        total_samples, positive_samples, negative_samples = combine_and_save_final_dataset(crash_data)
        
        # Log final statistics
        logging.info("Dataset generation complete!")
        logging.info(f"Final dataset statistics:")
        logging.info(f"Total samples: {total_samples}")
        logging.info(f"Positive samples: {positive_samples}")
        logging.info(f"Negative samples: {negative_samples}")
        logging.info(f"Balance ratio: {negative_samples/positive_samples:.2f}")
        
    except Exception as e:
        logging.error(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main() 