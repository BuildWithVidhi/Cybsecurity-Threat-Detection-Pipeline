# services/ml-service/ml_api.py
# FIXED: Better attack detection with improved data handling

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import json

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from contextlib import asynccontextmanager
from collections import Counter

# ========================
# Configuration
# ========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ml_service.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("threat_detection")

# ========================
# FIXED: Custom JSON Encoder for Numpy Types
# ========================
def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

# ========================
# Data Models (unchanged)
# ========================
class NetworkFlow(BaseModel):
    """Complete network flow data matching CSV structure"""
    flow_duration: float = 0.0
    total_fwd_packets: float = 0.0
    total_backward_packets: float = 0.0
    total_length_fwd_packets: float = 0.0
    total_length_bwd_packets: float = 0.0
    fwd_packet_length_max: float = 0.0
    fwd_packet_length_min: float = 0.0
    fwd_packet_length_mean: float = 0.0
    fwd_packet_length_std: float = 0.0
    bwd_packet_length_max: float = 0.0
    bwd_packet_length_min: float = 0.0
    bwd_packet_length_mean: float = 0.0
    bwd_packet_length_std: float = 0.0
    flow_bytes_s: float = 0.0
    flow_packets_s: float = 0.0
    flow_iat_mean: float = 0.0
    flow_iat_std: float = 0.0
    flow_iat_max: float = 0.0
    flow_iat_min: float = 0.0
    fwd_iat_total: float = 0.0
    fwd_iat_mean: float = 0.0
    fwd_iat_std: float = 0.0
    fwd_iat_max: float = 0.0
    fwd_iat_min: float = 0.0
    bwd_iat_total: float = 0.0
    bwd_iat_mean: float = 0.0
    bwd_iat_std: float = 0.0
    bwd_iat_max: float = 0.0
    bwd_iat_min: float = 0.0
    fwd_psh_flags: float = 0.0
    bwd_psh_flags: float = 0.0
    fwd_urg_flags: float = 0.0
    bwd_urg_flags: float = 0.0
    fin_flag_count: float = 0.0
    syn_flag_count: float = 0.0
    rst_flag_count: float = 0.0
    psh_flag_count: float = 0.0
    ack_flag_count: float = 0.0
    urg_flag_count: float = 0.0
    cwe_flag_count: float = 0.0
    ece_flag_count: float = 0.0
    fwd_header_length: float = 0.0
    bwd_header_length: float = 0.0
    fwd_packets_s: float = 0.0
    bwd_packets_s: float = 0.0
    min_packet_length: float = 0.0
    max_packet_length: float = 0.0
    packet_length_mean: float = 0.0
    packet_length_std: float = 0.0
    packet_length_variance: float = 0.0
    down_up_ratio: float = 0.0
    average_packet_size: float = 0.0
    avg_fwd_segment_size: float = 0.0
    avg_bwd_segment_size: float = 0.0
    fwd_header_length_1: float = 0.0
    fwd_avg_bytes_bulk: float = 0.0
    fwd_avg_packets_bulk: float = 0.0
    fwd_avg_bulk_rate: float = 0.0
    bwd_avg_bytes_bulk: float = 0.0
    bwd_avg_packets_bulk: float = 0.0
    bwd_avg_bulk_rate: float = 0.0
    subflow_fwd_packets: float = 0.0
    subflow_fwd_bytes: float = 0.0
    subflow_bwd_packets: float = 0.0
    subflow_bwd_bytes: float = 0.0
    init_win_bytes_forward: float = 0.0
    init_win_bytes_backward: float = 0.0
    act_data_pkt_fwd: float = 0.0
    min_seg_size_forward: float = 0.0
    active_mean: float = 0.0
    active_std: float = 0.0
    active_max: float = 0.0
    active_min: float = 0.0
    idle_mean: float = 0.0
    idle_std: float = 0.0
    idle_max: float = 0.0
    idle_min: float = 0.0

class DirectFeaturePredictionRequest(BaseModel):
    features: List[Dict[str, float]]

class DirectFeaturePredictionResponse(BaseModel):
    predictions: List[str]
    processed_count: int
    timestamp: str

class ThreatPrediction(BaseModel):
    threat_type: str
    confidence: float
    is_malicious: bool
    timestamp: str
    all_probabilities: Dict[str, float] = {}

class BatchPredictionRequest(BaseModel):
    flows: List[NetworkFlow]

class BatchPredictionResponse(BaseModel):
    predictions: List[ThreatPrediction]
    processed_count: int
    timestamp: str

# ========================
# FIXED: Core Service with Better Training
# ========================
class ThreatDetector:
    def __init__(self):
        self.model: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.feature_columns: Optional[List[str]] = None
        self.class_distribution: Optional[Dict] = None
        self.model_metrics: Optional[Dict] = None
        self.is_ready = False
        self.initialization_error = None
        self.training_stats = {}

    async def initialize(self):
        """FIXED: Memory-efficient initialization"""
        try:
            logger.info("Starting memory-efficient model initialization...")
            
            # Load data with memory management
            df = await self._load_training_data_fixed()
            logger.info(f"Loaded data shape: {df.shape}")
            
            # Preprocess efficiently
            df = self._preprocess_data_fixed(df)
            logger.info(f"Preprocessed data shape: {df.shape}")
            
            # Prepare features
            X, y = self._prepare_features(df)
            logger.info(f"Features prepared: {X.shape}")
            
            # Train model
            self._train_model_fixed(X, y)
            
            # Clear training data from memory immediately
            del df, X, y
            
            # Force garbage collection
            import gc
            gc.collect()
            
            self.is_ready = True
            logger.info(" Memory-efficient model initialization complete!")
            
        except Exception as e:
            error_msg = f"Initialization failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.initialization_error = error_msg
            self.is_ready = False

    
    async def _load_training_data_fixed(self) -> pd.DataFrame:
        """FIXED: Memory-efficient loading with proper attack preservation"""
        data_path = Path("/app/data/GeneratedLabelledFlows/TrafficLabelling")
        if not data_path.exists():
            raise FileNotFoundError(f"Training data directory not found: {data_path}")

        csv_files = list(data_path.rglob("*.csv"))
        if not csv_files:
            raise FileNotFoundError("No training CSV files found")

        logger.info(f"FIXED: Found {len(csv_files)} CSV files")
        
        # MEMORY EFFICIENT: Process files one by one and collect strategically
        all_attacks = []
        benign_samples = []
        attack_type_counts = Counter()
        total_processed = 0
        
        # Limit total samples to prevent memory issues
        MAX_TOTAL_SAMPLES = 100000  # Reasonable limit
        MAX_ATTACKS_PER_TYPE = 5000  # Ensure good attack representation
        MAX_BENIGN_TOTAL = 50000    # Limit benign to prevent overwhelming
        
        for i, file in enumerate(csv_files):
            try:
                logger.info(f"Processing file {i+1}/{len(csv_files)}: {file.name}")
                
                # Read file in chunks to manage memory
                chunk_size = 10000
                file_attacks = []
                file_benign = []
                
                for chunk in pd.read_csv(file, encoding='latin-1', chunksize=chunk_size, low_memory=False):
                    chunk.columns = chunk.columns.str.strip()
                    
                    if 'Label' not in chunk.columns:
                        continue
                    
                    # Clean chunk immediately
                    chunk = chunk.dropna(subset=['Label'])
                    
                    # Separate attacks and benign
                    benign_labels = ['BENIGN', 'Benign', 'benign']
                    benign_mask = chunk['Label'].isin(benign_labels)
                    
                    chunk_attacks = chunk[~benign_mask].copy()
                    chunk_benign = chunk[benign_mask].copy()
                    
                    # Collect attacks (prioritize rare ones)
                    if len(chunk_attacks) > 0:
                        for attack_type in chunk_attacks['Label'].unique():
                            attack_data = chunk_attacks[chunk_attacks['Label'] == attack_type]
                            current_count = attack_type_counts[attack_type]
                            
                            if current_count < MAX_ATTACKS_PER_TYPE:
                                take_count = min(len(attack_data), MAX_ATTACKS_PER_TYPE - current_count)
                                if take_count > 0:
                                    sampled_attacks = attack_data.sample(n=take_count, random_state=42) if len(attack_data) > take_count else attack_data
                                    file_attacks.append(sampled_attacks)
                                    attack_type_counts[attack_type] += take_count
                                    logger.info(f"  Collected {take_count} {attack_type} samples")
                    
                    # Collect benign (limited)
                    if len(chunk_benign) > 0 and len(benign_samples) < MAX_BENIGN_TOTAL:
                        remaining_benign_quota = MAX_BENIGN_TOTAL - sum(len(df) for df in benign_samples)
                        if remaining_benign_quota > 0:
                            take_count = min(len(chunk_benign), remaining_benign_quota, 2000)  # Max 2000 per chunk
                            sampled_benign = chunk_benign.sample(n=take_count, random_state=42) if len(chunk_benign) > take_count else chunk_benign
                            file_benign.append(sampled_benign)
                            logger.info(f"  Collected {take_count} BENIGN samples")
                    
                    # Memory cleanup
                    del chunk, chunk_attacks, chunk_benign
                
                # Combine file data
                if file_attacks:
                    file_attack_df = pd.concat(file_attacks, ignore_index=True)
                    all_attacks.append(file_attack_df)
                    del file_attacks
                
                if file_benign:
                    file_benign_df = pd.concat(file_benign, ignore_index=True)
                    benign_samples.append(file_benign_df)
                    del file_benign
                
                total_processed += 1
                
                # Check if we have enough data
                total_attacks = sum(len(df) for df in all_attacks)
                total_benign = sum(len(df) for df in benign_samples)
                
                if total_attacks + total_benign >= MAX_TOTAL_SAMPLES:
                    logger.info(f"Reached sample limit ({MAX_TOTAL_SAMPLES}), stopping early")
                    break
                                
            except Exception as e:
                logger.warning(f"Failed to load {file.name}: {str(e)}")
                continue

        if not all_attacks:
            raise ValueError("No attack samples found in training data!")
        
        # Combine all data efficiently
        logger.info("Combining collected data...")
        
        final_dfs = []
        
        # Add all attacks
        if all_attacks:
            combined_attacks = pd.concat(all_attacks, ignore_index=True)
            final_dfs.append(combined_attacks)
            logger.info(f" Total attacks: {len(combined_attacks)}")
            del all_attacks, combined_attacks
        
        # Add balanced benign
        if benign_samples:
            combined_benign = pd.concat(benign_samples, ignore_index=True)
            final_dfs.append(combined_benign)
            logger.info(f"Total benign: {len(combined_benign)}")
            del benign_samples, combined_benign
        
        # Final combination
        final_df = pd.concat(final_dfs, ignore_index=True)
        final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Log final distribution
        final_distribution = final_df['Label'].value_counts()
        logger.info(f"🎯 FINAL training data distribution:")
        total_samples = len(final_df)
        for label, count in final_distribution.items():
            percentage = (count / total_samples) * 100
            symbol = "⚔️" if label.upper() != 'BENIGN' else "✅"
            logger.info(f"  {symbol} {label}: {count} ({percentage:.1f}%)")
        
        return final_df

    def _separate_attacks_and_benign(self, df: pd.DataFrame):
        """Separate attack and benign traffic"""
        benign_labels = ['BENIGN', 'Benign', 'benign']
        benign_mask = df['Label'].isin(benign_labels)
        
        benign_df = df[benign_mask].copy()
        attack_df = df[~benign_mask].copy()
        
        return attack_df, benign_df

    def _attack_focused_balancing(self, df: pd.DataFrame) -> pd.DataFrame:
        """FIXED: Much more conservative balancing to prevent false patterns"""
        label_counts = df['Label'].value_counts()
        
        # MAJOR FIX: Use original data distribution more closely
        total_samples = len(df)
        
        attack_labels = [label for label in label_counts.index if label.upper() != 'BENIGN']
        benign_labels = [label for label in label_counts.index if label.upper() == 'BENIGN']
        
        current_attack_samples = sum(label_counts[label] for label in attack_labels)
        current_benign_samples = sum(label_counts[label] for label in benign_labels)
        
        logger.info(f"Original ratio - Attacks: {current_attack_samples} ({current_attack_samples/total_samples*100:.1f}%), Benign: {current_benign_samples} ({current_benign_samples/total_samples*100:.1f}%)")
        
        balanced_dfs = []
        
        # FIXED: Only downsample, never upsample attacks to prevent artificial patterns
        for attack_label in attack_labels:
            attack_data = df[df['Label'] == attack_label]
            current_count = len(attack_data)
            
            if current_count > 0:
                # CRITICAL FIX: Use much smaller caps and never duplicate
                max_samples = min(current_count, 1000)  # Reduced from 2000 to 1000
                
                if current_count > max_samples:
                    # Only downsample, use stratified sampling to maintain patterns
                    sampled_data = attack_data.sample(n=max_samples, random_state=42)
                    balanced_dfs.append(sampled_data)
                    logger.info(f"  ⚔️  {attack_label}: {current_count} -> {max_samples} (downsampled)")
                else:
                    # Keep all original samples - NO DUPLICATION
                    balanced_dfs.append(attack_data)
                    logger.info(f"  ⚔️  {attack_label}: kept all {current_count} samples (original)")

        # FIXED: Keep benign samples at realistic ratio (85-90% of total)
        total_attack_samples = sum(len(df_part) for df_part in balanced_dfs)
        # Much higher benign ratio to match real-world scenarios
        target_benign_samples = int(total_attack_samples * 8)  # 8:1 benign to attack ratio
        
        for benign_label in benign_labels:
            benign_data = df[df['Label'] == benign_label]
            if len(benign_data) > 0:
                take_samples = min(len(benign_data), target_benign_samples)
                if take_samples > 0:
                    sampled_benign = benign_data.sample(n=take_samples, random_state=42)
                    balanced_dfs.append(sampled_benign)
                    logger.info(f"  ✅ {benign_label}: took {take_samples} samples")
                    target_benign_samples -= take_samples

        if balanced_dfs:
            result = pd.concat(balanced_dfs, ignore_index=True)
            result = result.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Log final distribution
            final_counts = result['Label'].value_counts()
            final_total = len(result)
            final_attack_count = sum(count for label, count in final_counts.items() if label.upper() != 'BENIGN')
            final_attack_ratio = final_attack_count / final_total
            
            logger.info(f"📊 FINAL balanced distribution: {final_attack_count}/{final_total} attacks ({final_attack_ratio:.1%})")
            
            return result
        
        return df

    def _preprocess_data_fixed(self, df: pd.DataFrame) -> pd.DataFrame:
        """FIXED: Minimal preprocessing to preserve real attack signatures"""
        logger.info(" MINIMAL preprocessing to preserve attack patterns...")
        original_count = len(df)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Remove rows with missing labels first
        df = df.dropna(subset=['Label'])
        
        # Get numeric columns (excluding Label)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Label' in numeric_cols:
            numeric_cols.remove('Label')
        
        # MINIMAL cleaning - only handle extreme cases
        logger.info("Minimal data cleaning...")
        for col in numeric_cols:
            # Only replace infinite values, don't touch NaN initially
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            
            # Only fill NaN if there are many (>10% of column)
            nan_ratio = df[col].isnull().sum() / len(df)
            if nan_ratio > 0.1:  # Only if more than 10% are NaN
                median_val = df[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                df[col] = df[col].fillna(median_val)
                logger.info(f"  {col}: filled {nan_ratio:.1%} NaN values")
            elif nan_ratio > 0:
                # For small amounts of NaN, fill with 0
                df[col] = df[col].fillna(0.0)
        
        # REMOVED: Outlier handling completely - let the model see real attack patterns
        logger.info(" SKIPPED outlier handling to preserve attack signatures")
        
        logger.info(f" Minimal preprocessing complete: {original_count} -> {len(df)} samples")
        return df

    def _train_model_fixed(self, X, y):
        """FIXED: Training with strict false positive prevention"""
        logger.info(" STRICT: Training with false positive prevention...")
        
        # Check class distribution
        class_counts = pd.Series(y).value_counts()
        logger.info(f"Training class distribution: {dict(class_counts)}")
        
        # Ensure we have attacks
        attack_classes = [cls for cls in class_counts.index if cls.upper() != 'BENIGN']
        if not attack_classes:
            raise ValueError("No attack classes found in training data!")
        
        # FIXED: Larger test set for better validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y  # Increased to 30%
        )
        
        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        
        # Initialize scalers
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Fit and transform training data
        logger.info("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        # CRITICAL FIX: Balanced class weights (NO overweighting attacks)
        classes = np.unique(y_train_encoded)
        class_weights = {}
        
        for class_idx in classes:
            class_name = self.label_encoder.inverse_transform([class_idx])[0]
            class_count = np.sum(y_train_encoded == class_idx)
            
            if class_name.upper() == 'BENIGN':
                class_weights[class_idx] = 1.0  # Baseline weight
            else:
                # CRITICAL: Much more conservative weights
                weight = len(y_train_encoded) / (len(classes) * class_count)
                class_weights[class_idx] = min(weight * 0.8, 1.5)  # REDUCED to 0.8x multiplier, max 1.5x
        
        logger.info("Conservative class weights:")
        for class_idx, weight in class_weights.items():
            class_name = self.label_encoder.inverse_transform([class_idx])[0]
            logger.info(f"  {class_name}: {weight:.2f}")
        
        # FIXED: Even more conservative model to reduce false positives
        self.model = RandomForestClassifier(
            n_estimators=200,              # Increased for more stability
            max_depth=12,                  # REDUCED to prevent overfitting
            max_features='sqrt',           
            min_samples_split=15,          # INCREASED to prevent overfitting
            min_samples_leaf=8,            # INCREASED to prevent overfitting
            class_weight=class_weights,    # Conservative weights
            random_state=42,
            n_jobs=-1,
            bootstrap=True,
            max_samples=0.6,               # REDUCED to 60% for more diversity
            criterion='gini'
        )
        
        logger.info("🚀 Training ultra-conservative model...")
        self.model.fit(X_train_scaled, y_train_encoded)
        
        # FIXED: Strict validation focusing on precision
        logger.info("Strict model validation...")
        X_test_scaled = self.scaler.transform(X_test)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_encoded, y_pred)
        
        # Detailed classification report
        target_names = self.label_encoder.classes_
        class_report = classification_report(
            y_test_encoded, y_pred, 
            target_names=target_names, 
            output_dict=True,
            zero_division=0
        )
        
        # CRITICAL: Check false positive rates
        conf_matrix = confusion_matrix(y_test_encoded, y_pred)
        
        # Calculate precision for each attack class
        attack_precisions = {}
        benign_idx = None
        
        for i, class_name in enumerate(target_names):
            if class_name.upper() == 'BENIGN':
                benign_idx = i
                continue
            
            # True positives for this attack
            tp = conf_matrix[i, i]
            # False positives (other classes predicted as this attack)
            fp = conf_matrix[:, i].sum() - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            attack_precisions[class_name] = precision
            
            if precision < 0.7:  # Less than 70% precision
                logger.warning(f"LOW PRECISION for {class_name}: {precision:.3f}")
        
        # Check if benign is being misclassified as attacks (major issue)
        if benign_idx is not None:
            benign_misclassified = conf_matrix[benign_idx, :].sum() - conf_matrix[benign_idx, benign_idx]
            benign_total = conf_matrix[benign_idx, :].sum()
            benign_error_rate = benign_misclassified / benign_total if benign_total > 0 else 0
            
            if benign_error_rate > 0.05:  # More than 5% of benign classified as attacks
                logger.error(f"HIGH FALSE POSITIVE RATE: {benign_error_rate:.1%} of benign traffic classified as attacks!")
        
        # Store metrics with proper type conversion
        self.model_metrics = convert_numpy_types({
            'accuracy': accuracy,
            'classification_report': class_report,
            'classes': [str(cls) for cls in target_names],
            'feature_count': len(self.feature_columns),
            'class_weights': {str(self.label_encoder.inverse_transform([k])[0]): float(v) for k, v in class_weights.items()},
            'attack_precisions': attack_precisions,
            'benign_error_rate': benign_error_rate if benign_idx is not None else 0
        })
        
        # Log performance
        logger.info(f" Training complete! Accuracy: {accuracy:.4f}")
        logger.info("PRECISION-FOCUSED Performance:")
        
        for class_name in target_names:
            if class_name in class_report:
                precision = class_report[class_name]['precision']
                recall = class_report[class_name]['recall']
                f1 = class_report[class_name]['f1-score']
                support = class_report[class_name]['support']
                
                symbol = "⚔️" if class_name.upper() != 'BENIGN' else "✅"
                
                # Flag low precision
                precision_flag = "⚠️ LOW" if precision < 0.7 else "✅"
                logger.info(f"  {symbol} {class_name}: P={precision:.3f} {precision_flag}, R={recall:.3f}, F1={f1:.3f}, N={support}")
        
        # Clear training data from memory
        del X_train, X_train_scaled, y_train, y_train_encoded
        del X_test, X_test_scaled, y_test, y_test_encoded, y_pred, y_pred_proba

    def _prepare_features(self, df: pd.DataFrame):
        """Enhanced feature preparation"""
        metadata_cols = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol', 'Timestamp']
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove metadata and label columns
        for col in metadata_cols + ['Label']:
            if col in numeric_cols:
                numeric_cols.remove(col)
        
        self.feature_columns = numeric_cols
        
        logger.info(f"🔍 Selected {len(self.feature_columns)} features for training")
        
        X = df[self.feature_columns]
        y = df['Label']
        
        # Store class distribution
        self.class_distribution = {str(k): int(v) for k, v in y.value_counts().items()}
        
        logger.info(f"📊 Feature matrix shape: {X.shape}")
        logger.info(f"🏷️  Class distribution: {self.class_distribution}")
        
        return X, y

    # Keep existing feature extraction method
    def _extract_features_from_flow(self, flow: NetworkFlow) -> pd.DataFrame:
        """FIXED: Extract features using direct column name mapping"""
        # Create mapping from NetworkFlow fields to exact CSV column names
        flow_to_csv_mapping = {
            'flow_duration': 'Flow Duration',
            'total_fwd_packets': 'Total Fwd Packets', 
            'total_backward_packets': 'Total Backward Packets',
            'total_length_fwd_packets': 'Total Length of Fwd Packets',
            'total_length_bwd_packets': 'Total Length of Bwd Packets',
            'fwd_packet_length_max': 'Fwd Packet Length Max',
            'fwd_packet_length_min': 'Fwd Packet Length Min', 
            'fwd_packet_length_mean': 'Fwd Packet Length Mean',
            'fwd_packet_length_std': 'Fwd Packet Length Std',
            'bwd_packet_length_max': 'Bwd Packet Length Max',
            'bwd_packet_length_min': 'Bwd Packet Length Min',
            'bwd_packet_length_mean': 'Bwd Packet Length Mean', 
            'bwd_packet_length_std': 'Bwd Packet Length Std',
            'flow_bytes_s': 'Flow Bytes/s',
            'flow_packets_s': 'Flow Packets/s',
            'flow_iat_mean': 'Flow IAT Mean',
            'flow_iat_std': 'Flow IAT Std',
            'flow_iat_max': 'Flow IAT Max',
            'flow_iat_min': 'Flow IAT Min',
            'fwd_iat_total': 'Fwd IAT Total',
            'fwd_iat_mean': 'Fwd IAT Mean',
            'fwd_iat_std': 'Fwd IAT Std', 
            'fwd_iat_max': 'Fwd IAT Max',
            'fwd_iat_min': 'Fwd IAT Min',
            'bwd_iat_total': 'Bwd IAT Total',
            'bwd_iat_mean': 'Bwd IAT Mean',
            'bwd_iat_std': 'Bwd IAT Std',
            'bwd_iat_max': 'Bwd IAT Max', 
            'bwd_iat_min': 'Bwd IAT Min',
            'fwd_psh_flags': 'Fwd PSH Flags',
            'bwd_psh_flags': 'Bwd PSH Flags',
            'fwd_urg_flags': 'Fwd URG Flags',
            'bwd_urg_flags': 'Bwd URG Flags',
            'fwd_header_length': 'Fwd Header Length',
            'bwd_header_length': 'Bwd Header Length',
            'fwd_packets_s': 'Fwd Packets/s',
            'bwd_packets_s': 'Bwd Packets/s',
            'min_packet_length': 'Min Packet Length',
            'max_packet_length': 'Max Packet Length',
            'packet_length_mean': 'Packet Length Mean',
            'packet_length_std': 'Packet Length Std',
            'packet_length_variance': 'Packet Length Variance',
            'fin_flag_count': 'FIN Flag Count',
            'syn_flag_count': 'SYN Flag Count',
            'rst_flag_count': 'RST Flag Count',
            'psh_flag_count': 'PSH Flag Count',
            'ack_flag_count': 'ACK Flag Count',
            'urg_flag_count': 'URG Flag Count',
            'cwe_flag_count': 'CWE Flag Count',
            'ece_flag_count': 'ECE Flag Count',
            'down_up_ratio': 'Down/Up Ratio',
            'average_packet_size': 'Average Packet Size',
            'avg_fwd_segment_size': 'Avg Fwd Segment Size',
            'avg_bwd_segment_size': 'Avg Bwd Segment Size',
            'fwd_header_length_1': 'Fwd Header Length.1',
            'fwd_avg_bytes_bulk': 'Fwd Avg Bytes/Bulk',
            'fwd_avg_packets_bulk': 'Fwd Avg Packets/Bulk', 
            'fwd_avg_bulk_rate': 'Fwd Avg Bulk Rate',
            'bwd_avg_bytes_bulk': 'Bwd Avg Bytes/Bulk',
            'bwd_avg_packets_bulk': 'Bwd Avg Packets/Bulk',
            'bwd_avg_bulk_rate': 'Bwd Avg Bulk Rate',
            'subflow_fwd_packets': 'Subflow Fwd Packets',
            'subflow_fwd_bytes': 'Subflow Fwd Bytes',
            'subflow_bwd_packets': 'Subflow Bwd Packets',
            'subflow_bwd_bytes': 'Subflow Bwd Bytes',
            'init_win_bytes_forward': 'Init_Win_bytes_forward',
            'init_win_bytes_backward': 'Init_Win_bytes_backward', 
            'act_data_pkt_fwd': 'act_data_pkt_fwd',
            'min_seg_size_forward': 'min_seg_size_forward',
            'active_mean': 'Active Mean',
            'active_std': 'Active Std',
            'active_max': 'Active Max',
            'active_min': 'Active Min',
            'idle_mean': 'Idle Mean',
            'idle_std': 'Idle Std',
            'idle_max': 'Idle Max',
            'idle_min': 'Idle Min'
        }

        # Convert flow to dictionary
        flow_dict = flow.dict()
        
        # Create feature row based on the actual feature columns used during training
        features = {}
        
        # Add debugging
        logger.info(f"DEBUG: Available flow fields: {list(flow_dict.keys())[:10]}...")
        logger.info(f"DEBUG: Required features: {self.feature_columns[:5]}...")
        
        for csv_column in self.feature_columns:
            # Find the corresponding flow field
            flow_field = None
            for field, csv_col in flow_to_csv_mapping.items():
                if csv_col == csv_column:
                    flow_field = field
                    break
            
            if flow_field and flow_field in flow_dict:
                features[csv_column] = flow_dict[flow_field]
            else:
                # Default value for unmapped features
                features[csv_column] = 0.0
                if csv_column not in ['Timestamp', 'Label']:  # Don't warn for these
                    logger.warning(f"Feature {csv_column} not found in flow input (field: {flow_field}), using default 0.0")

        logger.info(f"DEBUG: Extracted {len(features)} features, sample values: {dict(list(features.items())[:3])}")
        return pd.DataFrame([features])

    async def predict(self, flow: NetworkFlow) -> ThreatPrediction:
        """FIXED: Prediction with much stricter thresholding"""
        if not self.is_ready:
            raise HTTPException(503, detail="Model not initialized")
        
        try:
            features_df = self._extract_features_from_flow(flow)
            
            # Ensure column order matches training data
            features_df = features_df[self.feature_columns]
            
            # Handle any NaN or inf values
            features_df = features_df.fillna(0.0)
            features_df = features_df.replace([np.inf, -np.inf], 0.0)
            
            features_scaled = self.scaler.transform(features_df)
            prediction = self.model.predict(features_scaled)
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            threat_type = self.label_encoder.inverse_transform(prediction)[0]
            confidence = float(np.max(probabilities))
            
            # CRITICAL FIX: Much stricter thresholding
            if str(threat_type).upper() != "BENIGN":
                # MUCH higher threshold for attack detection
                if confidence < 0.85:  # Require 85% confidence (was 70%)
                    threat_type = "BENIGN"
                    # Find benign probability
                    benign_prob = 0.5
                    for i, cls in enumerate(self.label_encoder.classes_):
                        if str(cls).upper() == "BENIGN":
                            benign_prob = float(probabilities[i])
                            break
                    confidence = benign_prob
                    logger.info(f"Attack prediction below strict threshold, classified as BENIGN (original: {np.max(probabilities):.3f})")
                else:
                    # Additional check: ensure attack prediction is significantly higher than benign
                    benign_prob = 0
                    for i, cls in enumerate(self.label_encoder.classes_):
                        if str(cls).upper() == "BENIGN":
                            benign_prob = float(probabilities[i])
                            break
                    
                    # If attack probability is not at least 2x higher than benign, classify as benign
                    if confidence < benign_prob * 2:
                        threat_type = "BENIGN"
                        confidence = benign_prob
                        logger.info(f"Attack confidence not sufficiently higher than benign, classified as BENIGN")
            
            # Create probability dictionary with all classes
            all_probabilities = {}
            for i, cls in enumerate(self.label_encoder.classes_):
                all_probabilities[str(cls)] = float(probabilities[i])

            return ThreatPrediction(
                threat_type=str(threat_type),
                confidence=confidence,
                is_malicious=str(threat_type).upper() != "BENIGN",
                timestamp=datetime.now().isoformat(),
                all_probabilities=all_probabilities
            )
            
        except Exception as e:
            logger.error(f"Prediction error for single flow: {str(e)}", exc_info=True)
            raise HTTPException(500, detail=f"Prediction failed: {str(e)}")
    
    async def predict_batch(self, flows: List[NetworkFlow]) -> List[ThreatPrediction]:
        if not self.is_ready:
            raise HTTPException(503, detail="Model not initialized")
        
        try:
            logger.info(f"Processing batch of {len(flows)} flows")
            
            # Extract features for all flows
            feature_dfs = []
            for i, flow in enumerate(flows):
                try:
                    flow_features = self._extract_features_from_flow(flow)
                    feature_dfs.append(flow_features)
                except Exception as e:
                    logger.warning(f"Failed to extract features for flow {i}: {str(e)}")
                    # Create default feature row
                    default_features = {col: 0.0 for col in self.feature_columns}
                    feature_dfs.append(pd.DataFrame([default_features]))
            
            if not feature_dfs:
                raise ValueError("No valid features extracted from flows")
            
            # Combine all feature DataFrames
            batch_df = pd.concat(feature_dfs, ignore_index=True)
            logger.info(f"Combined batch DataFrame shape: {batch_df.shape}")
            
            # Ensure column order matches training data
            batch_df = batch_df[self.feature_columns]
            
            # Handle any NaN or inf values
            batch_df = batch_df.fillna(0.0)
            batch_df = batch_df.replace([np.inf, -np.inf], 0.0)
            
            # Scale features
            features_scaled = self.scaler.transform(batch_df)
            
            # Make predictions
            predictions = self.model.predict(features_scaled)
            probabilities = self.model.predict_proba(features_scaled)

            results = []
            for i, pred in enumerate(predictions):
                threat_type = self.label_encoder.inverse_transform([pred])[0]
                confidence = float(np.max(probabilities[i]))
                
                # Create probability dictionary
                all_probabilities = {}
                for j, cls in enumerate(self.label_encoder.classes_):
                    all_probabilities[str(cls)] = float(probabilities[i][j])
                
                results.append(ThreatPrediction(
                    threat_type=str(threat_type),
                    confidence=confidence,
                    is_malicious=str(threat_type).upper() != "BENIGN",
                    timestamp=datetime.now().isoformat(),
                    all_probabilities=all_probabilities
                ))
            
            logger.info(f"Batch prediction completed: {len(results)} predictions")
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction error: {str(e)}", exc_info=True)
            raise HTTPException(500, detail=f"Batch prediction failed: {str(e)}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and metrics"""
        if not self.is_ready:
            return {"status": "not_initialized", "error": self.initialization_error}
        
        return convert_numpy_types({
            "status": "ready",
            "feature_count": len(self.feature_columns) if self.feature_columns else 0,
            "classes": [str(cls) for cls in self.label_encoder.classes_] if self.label_encoder else [],
            "class_distribution": self.class_distribution,
            "model_metrics": self.model_metrics,
            "features": self.feature_columns[:10] if self.feature_columns else []  # Show first 10 features
        })

# ========================
# Global Model Instance
# ========================
threat_detector = ThreatDetector()

# ========================
# FastAPI Lifespan Management
# ========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - initialize model on startup"""
    logger.info("Starting ML Threat Detection Service...")
    
    # Startup
    await threat_detector.initialize()
    
    yield
    
    # Shutdown
    logger.info("Shutting down ML Threat Detection Service...")

# ========================
# FastAPI Application
# ========================
app = FastAPI(
    title="ML Threat Detection Service",
    description="Advanced ML-based cybersecurity threat detection pipeline",
    version="2.0.0",
    lifespan=lifespan
)

# ========================
# API Endpoints
# ========================
@app.get("/")
async def root():
    """Root endpoint with service info"""
    return {
        "service": "ML Threat Detection Service",
        "version": "2.0.0",
        "status": "ready" if threat_detector.is_ready else "initializing",
        "endpoints": [
            "/predict", 
            "/predict_batch",         # Direct features (for data processor)
            "/predict_batch_flows",   # Original flow format
            "/health", 
            "/model-info"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not threat_detector.is_ready:
        return {
            "status": "unhealthy",
            "error": threat_detector.initialization_error,
            "timestamp": datetime.now().isoformat()
        }
    
    return {
        "status": "healthy",
        "model_ready": threat_detector.is_ready,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model-info")
async def get_model_info():
    """Get detailed model information and performance metrics"""
    return threat_detector.get_model_info()

@app.post("/predict", response_model=ThreatPrediction)
async def predict_threat(flow: NetworkFlow):
    """Predict threat for a single network flow"""
    try:
        result = await threat_detector.predict(flow)
        logger.info(f"Prediction: {result.threat_type} (confidence: {result.confidence:.3f})")
        return result
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_batch", response_model=DirectFeaturePredictionResponse)
async def predict_batch_direct_features(request: DirectFeaturePredictionRequest):
    """FIXED: Ultra-conservative batch prediction"""
    try:
        if not threat_detector.is_ready:
            raise HTTPException(503, detail="Model not initialized")
        
        if not request.features:
            raise HTTPException(400, detail="No features provided")
        
        if len(request.features) > 500:
            raise HTTPException(400, detail="Batch size too large (max 500)")
        
        logger.info(f"🔍 Processing batch of {len(request.features)} feature sets")
        
        # Convert to DataFrame efficiently
        features_df = pd.DataFrame(request.features)
        
        # Add missing features with default values
        for required_feature in threat_detector.feature_columns:
            if required_feature not in features_df.columns:
                features_df[required_feature] = 0.0
        
        # Reorder columns to match training
        features_df = features_df[threat_detector.feature_columns]
        
        # Clean data
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(0.0)
        
        # Apply scaling
        features_scaled = threat_detector.scaler.transform(features_df)
        
        # Make predictions
        predictions = threat_detector.model.predict(features_scaled)
        probabilities = threat_detector.model.predict_proba(features_scaled)
        
        # Convert predictions with ULTRA-STRICT thresholding
        prediction_labels = threat_detector.label_encoder.inverse_transform(predictions)
        prediction_strings = []
        
        # Find benign class index
        benign_class_idx = None
        for i, cls in enumerate(threat_detector.label_encoder.classes_):
            if str(cls).upper() == 'BENIGN':
                benign_class_idx = i
                break
        
        for i, (label, prob_array) in enumerate(zip(prediction_labels, probabilities)):
            max_confidence = float(np.max(prob_array))
            
            # ULTRA-STRICT thresholding for attacks
            if str(label).upper() != 'BENIGN':
                # Multiple conditions for attack detection
                benign_prob = float(prob_array[benign_class_idx]) if benign_class_idx is not None else 0.5
                
                # Condition 1: Must have high confidence (85%+)
                # Condition 2: Must be at least 3x higher than benign probability
                # Condition 3: Benign probability must be low (<0.3)
                if (max_confidence < 0.85 or 
                    max_confidence < benign_prob * 3 or 
                    benign_prob > 0.3):
                    
                    final_label = 'BENIGN'
                    logger.debug(f"Sample {i}: Attack {label} rejected - conf:{max_confidence:.3f}, benign:{benign_prob:.3f}")
                else:
                    final_label = str(label)
                    logger.debug(f"Sample {i}: Attack {label} CONFIRMED - conf:{max_confidence:.3f}, benign:{benign_prob:.3f}")
            else:
                final_label = str(label)
            
            prediction_strings.append(final_label)
        
        # Log prediction summary
        prediction_counts = Counter(prediction_strings)
        attack_count = sum(count for label, count in prediction_counts.items() if label.upper() != 'BENIGN')
        attack_ratio = attack_count / len(prediction_strings) if prediction_strings else 0
        
        logger.info(f" ULTRA-STRICT Batch results: {len(prediction_strings)} predictions")
        logger.info(f"Attack detection: {attack_count} attacks ({attack_ratio:.1%}) [After ultra-strict filtering]")
        logger.info(f"Distribution: {dict(prediction_counts)}")
        
        # Additional warning if attack rate is still high
        if attack_ratio > 0.1:  # More than 10% attacks
            logger.warning(f"⚠️  Attack rate still high after filtering: {attack_ratio:.1%}")
        
        return DirectFeaturePredictionResponse(
            predictions=prediction_strings,
            processed_count=len(prediction_strings),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}", exc_info=True)
        raise HTTPException(500, detail=f"Prediction failed: {str(e)}")
    
        
@app.post("/predict_batch_flows", response_model=BatchPredictionResponse)
async def predict_threats_batch_flows(request: BatchPredictionRequest):
    """Predict threats for multiple network flows (original format)"""
    try:
        if not request.flows:
            raise HTTPException(400, detail="No flows provided")
        
        if len(request.flows) > 1000:
            raise HTTPException(400, detail="Batch s" \
            "size too large (max 1000)")
        
        predictions = await threat_detector.predict_batch(request.flows)
        
        logger.info(f"Batch prediction completed: {len(predictions)} flows processed")
        
        return BatchPredictionResponse(
            predictions=predictions,
            processed_count=len(predictions),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    """Get service metrics and statistics"""
    if not threat_detector.is_ready:
        raise HTTPException(503, detail="Model not ready")
    
    return convert_numpy_types({
        "service_status": "operational",
        "model_metrics": threat_detector.model_metrics,
        "feature_count": len(threat_detector.feature_columns),
        "supported_classes": list(threat_detector.label_encoder.classes_),
        "class_distribution": threat_detector.class_distribution,
        "timestamp": datetime.now().isoformat()
    })

# ========================
# Error Handlers
# ========================
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled error: {str(exc)}", exc_info=True)
    return {
        "error": "Internal server error",
        "detail": str(exc),
        "timestamp": datetime.now().isoformat()
    }

# ========================
# Main Entry Point
# ========================
if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting ML Threat Detection Service...")
    uvicorn.run(
        "ml_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
