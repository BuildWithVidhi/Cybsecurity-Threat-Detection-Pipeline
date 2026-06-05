# services/data-producer/producer.py
# Updated to preserve original CSV field names instead of converting to snake_case

import pandas as pd
import json
import time
import logging
from kafka import KafkaProducer
from datetime import datetime
import numpy as np
import os
import glob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Expected CSV columns (keeping original names)
EXPECTED_CSV_FIELDS = [
    'Flow Duration',
    'Total Fwd Packets',
    'Total Backward Packets',
    'Total Length of Fwd Packets',
    'Total Length of Bwd Packets',
    'Fwd Packet Length Max',
    'Fwd Packet Length Min',
    'Fwd Packet Length Mean',
    'Fwd Packet Length Std',
    'Bwd Packet Length Max',
    'Bwd Packet Length Min',
    'Bwd Packet Length Mean',
    'Bwd Packet Length Std',
    'Flow Bytes/s',
    'Flow Packets/s',
    'Flow IAT Mean',
    'Flow IAT Std',
    'Flow IAT Max',
    'Flow IAT Min',
    'Fwd IAT Total',
    'Fwd IAT Mean',
    'Fwd IAT Std',
    'Fwd IAT Max',
    'Fwd IAT Min',
    'Bwd IAT Total',
    'Bwd IAT Mean',
    'Bwd IAT Std',
    'Bwd IAT Max',
    'Bwd IAT Min',
    'Fwd PSH Flags',
    'Bwd PSH Flags',
    'Fwd URG Flags',
    'Bwd URG Flags',
    'Fwd Header Length',
    'Bwd Header Length',
    'Fwd Packets/s',
    'Bwd Packets/s',
    'Min Packet Length',
    'Max Packet Length',
    'Packet Length Mean',
    'Packet Length Std',
    'Packet Length Variance',
    'FIN Flag Count',
    'SYN Flag Count',
    'RST Flag Count',
    'PSH Flag Count',
    'ACK Flag Count',
    'URG Flag Count',
    'CWE Flag Count',
    'ECE Flag Count',
    'Down/Up Ratio',
    'Average Packet Size',
    'Avg Fwd Segment Size',
    'Avg Bwd Segment Size',
    'Fwd Header Length.1',
    'Fwd Avg Bytes/Bulk',
    'Fwd Avg Packets/Bulk',
    'Fwd Avg Bulk Rate',
    'Bwd Avg Bytes/Bulk',
    'Bwd Avg Packets/Bulk',
    'Bwd Avg Bulk Rate',
    'Subflow Fwd Packets',
    'Subflow Fwd Bytes',
    'Subflow Bwd Packets',
    'Subflow Bwd Bytes',
    'Init_Win_bytes_forward',
    'Init_Win_bytes_backward',
    'act_data_pkt_fwd',
    'min_seg_size_forward',
    'Active Mean',
    'Active Std',
    'Active Max',
    'Active Min',
    'Idle Mean',
    'Idle Std',
    'Idle Max',
    'Idle Min'
]

def create_producer():
    """Create Kafka producer that works with your Kafka setup"""
    max_retries = 10
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            producer = KafkaProducer(
                bootstrap_servers=['kafka:29092'],
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                retries=5,
                max_in_flight_requests_per_connection=1,
                request_timeout_ms=30000,
                api_version=(2, 5, 0)
            )
            logger.info("Connected to Kafka successfully")
            return producer
        except Exception as e:
            logger.warning(f"Kafka connection attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error("Max retries reached. Could not connect to Kafka")
                return None

def find_csv_files():
    """Find CSV files in the mounted data directory"""
    data_path = '/app/data/GeneratedLabelledFlows/TrafficLabelling'
    
    logger.info(f"=== Searching for CSV files in {data_path} ===")
    
    if not os.path.exists(data_path):
        logger.error(f"Data directory does not exist: {data_path}")
        return []
    
    try:
        all_files = os.listdir(data_path)
        logger.info(f"Total files in {data_path}: {len(all_files)}")
        
        csv_files = [f for f in all_files if f.endswith('.csv')]
        logger.info(f"CSV files found: {len(csv_files)}")
        
        if csv_files:
            logger.info("All CSV files:")
            for i, csv_file in enumerate(csv_files):
                logger.info(f"  {i+1}. {csv_file}")
            
            full_paths = [os.path.join(data_path, f) for f in csv_files]
            return full_paths
        else:
            logger.warning("No CSV files found!")
            return []
            
    except Exception as e:
        logger.error(f"Error accessing {data_path}: {e}")
        return []

def find_label_column(df):
    """Find the label column with various naming conventions"""
    possible_names = [
        'Label', ' Label', 'label', ' label', 
        'Label ', 'label ', ' Label ', ' label ',
        'attack_type', 'Attack_Type', 'class', 'Class',
        'category', 'Category'
    ]
    
    for name in possible_names:
        if name in df.columns:
            logger.info(f"Found label column: '{name}'")
            return name
    
    # If not found by exact match, try partial matching
    for col in df.columns:
        if 'label' in col.lower():
            logger.info(f"Found label column by partial match: '{col}'")
            return col
    
    logger.warning("No label column found!")
    return None

def validate_and_clean_value(val):
    """Validate and clean individual values"""
    if pd.isna(val) or val is None:
        return 0.0
    
    if isinstance(val, str):
        try:
            val = float(val)
        except ValueError:
            return 0.0
    
    if np.isinf(val):
        return 0.0
    
    if isinstance(val, (int, float)):
        # Cap extremely large or small values
        if val > 1e10:
            return 1e10
        elif val < -1e10:
            return -1e10
        else:
            return float(val)
    
    return 0.0

def load_csv_file(csv_file):
    """Load and clean a single CSV file"""
    try:
        logger.info(f"Loading: {os.path.basename(csv_file)}")
        
        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                df = pd.read_csv(csv_file, encoding=encoding, low_memory=False)
                logger.info(f"Successfully loaded with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        else:
            logger.error(f"Could not load {csv_file} with any encoding")
            return None
        
        if df.empty:
            logger.warning(f"File {csv_file} is empty")
            return None
        
        # Clean the data
        df.columns = df.columns.str.strip()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        
        logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        logger.info(f"Sample columns: {list(df.columns[:10])}")
        
        # Check for label column
        label_col = find_label_column(df)
        if label_col:
            label_counts = df[label_col].value_counts()
            logger.info(f"Attack types found: {dict(label_counts.head())}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading {csv_file}: {e}")
        return None

def send_to_kafka(producer, df, csv_filename):
    """Send data to Kafka keeping original CSV field names"""
    topic = 'network-data'
    
    # Add a file boundary marker at the start
    boundary_message = {
        'message_type': 'FILE_START',
        'source_file': csv_filename,
        'timestamp': datetime.now().isoformat(),
        'total_records': len(df)
    }
    producer.send(topic, value=boundary_message)
    logger.info(f"📋 Sent FILE_START marker for {csv_filename}")
    
    # Find label column
    label_col = find_label_column(df)
    
    # FIXED: Sampling logic to create df_sample
    if label_col is None:
        sample_size = min(500, len(df))
        df_sample = df.sample(n=sample_size, random_state=42)
        logger.info(f"No label column found, using random sampling: {len(df_sample)} samples")
    else:
        # Stratified sampling to ensure we get both BENIGN and attack samples
        sample_size = min(500, len(df))
        
        label_counts = df[label_col].value_counts()
        samples_per_class = {}
        total_classes = len(label_counts)
        
        for label, count in label_counts.items():
            if str(label).upper() == 'BENIGN':
                samples_per_class[label] = min(int(sample_size * 0.6), count)
            else:
                remaining_sample = sample_size - samples_per_class.get('BENIGN', 0)
                attack_classes = total_classes - 1
                if attack_classes > 0:
                    samples_per_class[label] = min(
                        max(50, remaining_sample // attack_classes), 
                        count
                    )
        
        sampled_dfs = []
        for label, n_samples in samples_per_class.items():
            if n_samples > 0:
                class_data = df[df[label_col] == label]
                if len(class_data) >= n_samples:
                    sampled_dfs.append(class_data.sample(n=n_samples, random_state=42))
                else:
                    sampled_dfs.append(class_data)
        
        df_sample = pd.concat(sampled_dfs, ignore_index=True) if sampled_dfs else df.sample(n=min(500, len(df)), random_state=42)
        df_sample = df_sample.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Stratified sampling completed: {len(df_sample)} samples")
    
    logger.info(f"Sending {len(df_sample)} records from {csv_filename} to topic '{topic}'")
    
    if label_col:
        sample_labels = df_sample[label_col].value_counts()
        logger.info(f"Sample composition: {dict(sample_labels)}")
    
    success_count = 0
    error_count = 0
    
    for index, row in df_sample.iterrows():
        try:
            # Create message keeping original CSV field names
            message = {}
            
            # Add all available CSV fields with their original names
            for field in EXPECTED_CSV_FIELDS:
                if field in df.columns:
                    val = row.get(field, 0.0)
                    message[field] = validate_and_clean_value(val)
                else:
                    # If field doesn't exist in this CSV, set to 0.0
                    message[field] = 0.0
            
            # Add any additional fields that might be in the CSV but not in our expected list
            for col in df.columns:
                if col not in EXPECTED_CSV_FIELDS and col != label_col:
                    val = row.get(col, 0.0)
                    message[col] = validate_and_clean_value(val)
            
            # Add label if available (keep original label column name)
            if label_col and label_col in row:
                label_value = str(row[label_col]).strip()
                message[label_col] = label_value  # Keep original label column name
            
            # Add metadata including file identification
            message['message_type'] = 'DATA_RECORD'
            message['timestamp'] = datetime.now().isoformat()
            message['source_file'] = csv_filename
            message['record_id'] = int(index)
            
            # Send to Kafka
            producer.send(topic, value=message)
            success_count += 1
            
            # Log progress
            if success_count % 50 == 0:
                logger.info(f"📤 Sent {success_count}/{len(df_sample)} messages from {csv_filename}")
                if label_col:
                    logger.info(f"   Current traffic type: {message.get(label_col, 'UNKNOWN')}")
                logger.info(f"   Sample fields: {list(message.keys())[:10]}")
            
            # Send at 2 messages per second
            time.sleep(0.5)
            
        except Exception as e:
            error_count += 1
            logger.error(f"Error sending message {index}: {e}")
            
            if error_count > 20:
                logger.error("Too many errors, stopping")
                break
    
    # Send file completion marker
    completion_message = {
        'message_type': 'FILE_END',
        'source_file': csv_filename,
        'timestamp': datetime.now().isoformat(),
        'records_sent': success_count
    }
    producer.send(topic, value=completion_message)
    producer.flush()
    
    logger.info(f"🏁 Sent FILE_END marker for {csv_filename}")
    logger.info(f"Finished {csv_filename}: sent {success_count}, errors {error_count}")
    
    return success_count

def main():
    """Main producer function"""
    logger.info("=== STARTING DATA PRODUCER ===")
    logger.info(f"Will send data keeping original CSV field names")
    logger.info(f"Expected CSV fields: {len(EXPECTED_CSV_FIELDS)}")
    
    # Wait for Kafka to be ready
    logger.info("Waiting for Kafka to be ready...")
    time.sleep(45)
    
    # Connect to Kafka
    producer = create_producer()
    if not producer:
        logger.error("Cannot connect to Kafka - exiting")
        exit(1)
    
    # Find CSV files
    csv_files = find_csv_files()
    if not csv_files:
        logger.error("No CSV files found - check your volume mount")
        exit(1)
    
    logger.info(f"Found {len(csv_files)} CSV files to process")
    
    # SKIP FIRST 3 FILES - Process files starting from index 3
    files_to_skip = 3
    if len(csv_files) <= files_to_skip:
        logger.error(f"Not enough files to skip {files_to_skip}. Total files: {len(csv_files)}")
        exit(1)
    
    logger.info(f"⏭️  SKIPPING FIRST {files_to_skip} FILES:")
    for i in range(files_to_skip):
        logger.info(f"   Skipped: {os.path.basename(csv_files[i])}")
    
    # Process files starting from index 3 (4th file onwards)
    files_to_process = csv_files[files_to_skip:]
    logger.info(f"📋 PROCESSING {len(files_to_process)} FILES (after skipping first {files_to_skip}):")
    for i, csv_file in enumerate(files_to_process):
        logger.info(f"   {i+1}. {os.path.basename(csv_file)}")
    
    # Process files one by one with PROPER SEPARATION
    total_sent = 0
    
    # Process first 3 files after skipping (so files 4, 5, 6)
    files_to_process_now = files_to_process[:3]
    
    for i, csv_file in enumerate(files_to_process_now):
        filename = os.path.basename(csv_file)
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 PROCESSING FILE {i+1}/3: {filename}")
        logger.info(f"   (This is file #{files_to_skip + i + 1} in the original list)")
        logger.info(f"{'='*60}")
        
        # Load the file
        df = load_csv_file(csv_file)
        if df is None:
            logger.warning(f"Skipping {filename} - could not load")
            continue
        
        # Send to Kafka
        sent_count = send_to_kafka(producer, df, filename)
        total_sent += sent_count
        
        logger.info(f"✅ Completed {filename}: {sent_count} messages sent")
        
        # IMPORTANT: Wait for processing to complete before next file
        if i < len(files_to_process_now) - 1:  # Don't wait after the last file
            logger.info(f"⏳ Waiting for complete processing of {filename}...")
            logger.info("   This ensures no mixing between files")
            
            # Calculate wait time based on messages sent
            # Assuming ~2 messages/sec processing rate + buffer
            estimated_processing_time = (sent_count / 2) + 60  # +60 sec buffer
            wait_time = max(120, min(estimated_processing_time, 600))  # Between 2-10 minutes
            
            logger.info(f"   Waiting {wait_time:.0f} seconds for processing completion...")
            
            # Wait with progress indicators
            for remaining in range(int(wait_time), 0, -30):
                logger.info(f"   ⏱️  {remaining} seconds remaining...")
                time.sleep(30)
            
            logger.info(f"✨ Ready to process next file!\n")
    
    # Close producer
    producer.close()
    
    logger.info(f"=== PRODUCER FINISHED ===")
    logger.info(f"Total messages sent: {total_sent}")
    logger.info(f"🎉 Processed {len(files_to_process_now)} files (after skipping first {files_to_skip})!")

if __name__ == "__main__":
    main()

