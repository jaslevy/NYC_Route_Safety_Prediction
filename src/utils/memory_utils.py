import os
import psutil
import logging
import pandas as pd
from typing import Any, Dict

def setup_logging(log_file: str = '../data/processed/dataset_generation.log') -> None:
    """
    Set up logging configuration to output to both file and console.
    
    Args:
        log_file (str): Path to the log file
    """
    # Create the directory for the log file if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def get_memory_usage() -> Dict[str, Any]:
    """
    Get current memory usage of the process.
    
    Returns:
        Dict containing memory usage statistics
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        'rss': memory_info.rss / 1024 / 1024,  # RSS in MB
        'vms': memory_info.vms / 1024 / 1024,  # VMS in MB
        'percent': process.memory_percent()
    }

def log_dataframe_info(df: pd.DataFrame, name: str) -> None:
    """
    Log information about a DataFrame including its shape, memory usage, and columns.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze
        name (str): Name of the DataFrame for logging purposes
    """
    logging.info(f"\n{'='*50}")
    logging.info(f"DataFrame: {name}")
    logging.info(f"Shape: {df.shape}")
    logging.info(f"Memory usage: {df.memory_usage().sum() / 1024 / 1024:.2f} MB")
    logging.info(f"Columns: {df.columns.tolist()}")
    logging.info(f"Data types:\n{df.dtypes}")
    logging.info(f"{'='*50}\n") 