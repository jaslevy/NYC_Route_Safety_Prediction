import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import gc
import os
import logging
import psutil
from tqdm import tqdm

logger = logging.getLogger(__name__)

def get_memory_info():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    return f"{memory_mb:.1f} MB"

def log_dataframe_info(df: pd.DataFrame, name: str) -> None:
    """Log information about a DataFrame."""
    logger.info(f"\n{'='*50}")
    logger.info(f"DataFrame: {name}")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Memory usage: {df.memory_usage().sum() / 1024 / 1024:.2f} MB")
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"Data types:\n{df.dtypes}")
    logger.info(f"{'='*50}\n")

def generate_balanced_dataset(enhanced_df: pd.DataFrame, 
                            weather_df: pd.DataFrame, 
                            negative_to_positive_ratio: float = 1.0,
                            temp_dir: str = '../data/processed/temp') -> pd.DataFrame:
    """
    Generate a balanced dataset with negative examples proportional to positive examples
    at each intersection.
    
    Args:
        enhanced_df (pd.DataFrame): DataFrame containing crash data
        weather_df (pd.DataFrame): DataFrame containing weather data
        negative_to_positive_ratio (float): Ratio of negative to positive examples
        temp_dir (str): Directory for temporary files
        
    Returns:
        pd.DataFrame: Balanced dataset with both positive and negative examples
    """
    logger.info(f"Starting balanced dataset generation with ratio {negative_to_positive_ratio}")
    logger.info(f"Initial memory usage: {get_memory_info()}")
    
    # Log initial dataframe information
    log_dataframe_info(enhanced_df, "Enhanced DataFrame (Initial)")
    log_dataframe_info(weather_df, "Weather DataFrame (Initial)")
    
    # Create output directory for progress saving
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create fast weather lookup dictionary
    logger.info("Creating weather lookup dictionary...")
    weather_lookup = {
        (pd.Timestamp(row['time']).date(), row['time'].hour): row.drop('time').to_dict()
        for _, row in tqdm(weather_df.iterrows(), total=len(weather_df), desc="Building weather lookup")
    }
    logger.info(f"Weather lookup created. Memory usage: {get_memory_info()}")
    
    # Create set of positive events for collision prevention
    logger.info("Creating positive events set...")
    positive_keys = set(zip(enhanced_df['nearest_intersection_id'], 
                          enhanced_df['crash_date'].dt.date, 
                          enhanced_df['hour']))
    logger.info(f"Created {len(positive_keys)} positive event keys")
    
    # Get date range for random sampling
    start_date = enhanced_df['crash_date'].min()
    end_date = enhanced_df['crash_date'].max()
    all_dates = pd.date_range(start_date, end_date, freq='D')
    
    # Calculate number of positives per intersection
    positives_per_intersection = enhanced_df.groupby('nearest_intersection_id').size()
    intersections = positives_per_intersection.index
    
    # Process intersections in smaller batches
    batch_size = 100
    total_batches = (len(intersections) + batch_size - 1) // batch_size
    total_negatives = 0
    
    logger.info(f"Processing {len(intersections)} intersections in {total_batches} batches...")
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(intersections))
        batch_intersections = intersections[start_idx:end_idx]
        
        batch_negatives = []
        batch_pbar = tqdm(batch_intersections, 
                         desc=f"Batch {batch_idx + 1}/{total_batches}",
                         position=0)
        
        for intersection in batch_pbar:
            n_positives = positives_per_intersection[intersection]
            n_negatives = int(n_positives * negative_to_positive_ratio)
            
            samples = []
            attempts = 0
            max_attempts = n_negatives * 2
            
            while len(samples) < n_negatives and attempts < max_attempts:
                random_date = pd.Timestamp(np.random.choice(all_dates))
                random_hour = np.random.randint(0, 24)
                
                if (intersection, random_date.date(), random_hour) in positive_keys:
                    attempts += 1
                    continue
                
                weather_key = (random_date.date(), random_hour)
                if weather_key in weather_lookup:
                    weather_data = weather_lookup[weather_key].copy()
                    negative_entry = {
                        'nearest_intersection_id': intersection,
                        'crash_date': random_date,
                        'hour': random_hour,
                        'crash_count': 0,
                        'day_of_week': random_date.dayofweek,
                        'month': random_date.month,
                        'is_weekend': int(random_date.dayofweek in [5, 6])
                    }
                    negative_entry.update(weather_data)
                    samples.append(negative_entry)
                
                attempts += 1
            
            batch_negatives.extend(samples)
            batch_pbar.set_postfix({'samples': len(samples), 'attempts': attempts})
        
        if batch_negatives:
            batch_df = pd.DataFrame(batch_negatives)
            batch_file = os.path.join(temp_dir, f'negatives_batch_{batch_idx}.csv')
            batch_df.to_csv(batch_file, index=False)
            total_negatives += len(batch_negatives)
            logger.info(f"Saved batch {batch_idx + 1}: {len(batch_negatives)} samples to {batch_file}")
            logger.info(f"Current memory usage: {get_memory_info()}")
        
        del batch_negatives
        gc.collect()
    
    # Combine all batches
    logger.info("Combining all batches...")
    negative_files = [f for f in os.listdir(temp_dir) 
                     if f.startswith('negatives_batch_')]
    
    negatives_dfs = []
    for file in tqdm(negative_files, desc="Loading batches"):
        df = pd.read_csv(os.path.join(temp_dir, file))
        negatives_dfs.append(df)
        os.remove(os.path.join(temp_dir, file))
    
    negatives_df = pd.concat(negatives_dfs, ignore_index=True)
    logger.info(f"Combined negative examples. Shape: {negatives_df.shape}")
    del negatives_dfs
    gc.collect()
    
    # Prepare positive examples
    logger.info("Preparing positive examples...")
    group_cols = ['nearest_intersection_id', 'crash_date', 'hour', 
                 'day_of_week', 'month', 'is_weekend']
    positives_df = enhanced_df.groupby(group_cols, as_index=False).size()
    positives_df.columns = group_cols + ['crash_count']
    
    logger.info(f"Positive examples prepared. Shape: {positives_df.shape}")
    
    # Combine positive and negative examples
    logger.info("Combining positive and negative examples...")
    balanced_df = pd.concat([positives_df, negatives_df], ignore_index=True)
    balanced_df['had_crash'] = (balanced_df['crash_count'] > 0).astype(int)
    
    log_dataframe_info(balanced_df, "Final Balanced Dataset")
    logger.info(f"Final memory usage: {get_memory_info()}")
    
    return balanced_df 