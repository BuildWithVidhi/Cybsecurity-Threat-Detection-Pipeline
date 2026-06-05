# services/data-processor/processor.py
# FIXED VERSION - Ensures final summary is always generated

import json
import logging
import requests
import time
import numpy as np
from kafka import KafkaConsumer
from datetime import datetime
from collections import defaultdict
import signal
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def classify_threat_type(prediction: str) -> str:
    """
    ENHANCED: Improved threat classification with detailed logging
    """
    # Normalize prediction for case-insensitive matching
    pred_lower = prediction.lower().strip()
    original_prediction = prediction.strip()
   
    # Log every prediction for debugging
    logger.info(f"🔍 CLASSIFYING: '{original_prediction}' -> '{pred_lower}'")
   
    # Comprehensive threat type mapping - EXACT MATCHES ONLY to prevent confusion
    exact_mappings = {
        # Web attacks (common exact matches)
        'web attack': 'WEB_ATTACK',
        'web attack – brute force': 'WEB_ATTACK',
        'web attack \x96 brute force': 'WEB_ATTACK',
        'web attack - brute force': 'WEB_ATTACK',
        'web attack brute force': 'WEB_ATTACK',
        'sql injection': 'WEB_ATTACK',
        'xss': 'WEB_ATTACK',
        
        # DDoS variants (exact matches)
        'ddos': 'DDOS',
        'dos': 'DDOS',
        'dos slowloris': 'DDOS',
        'dos hulk': 'DDOS', 
        'dos goldeneye': 'DDOS',
        'slowloris': 'DDOS',
        'hulk': 'DDOS',
        'goldeneye': 'DDOS',
       
        # Port scanning (exact matches)
        'portscan': 'PORT_SCAN',
        'port scan': 'PORT_SCAN',
        'port_scan': 'PORT_SCAN',
        
        # Brute force attacks (exact matches)
        'ftp-patator': 'BRUTE_FORCE',
        'ssh-patator': 'BRUTE_FORCE',
        'patator': 'BRUTE_FORCE',
        'brute force': 'BRUTE_FORCE',
        'bruteforce': 'BRUTE_FORCE',
       
        # Infiltration
        'infiltration': 'INFILTRATION',
        'infilterate': 'INFILTRATION',
       
        # Botnet
        'botnet': 'BOTNET',
        'bot': 'BOTNET',
       
        # Heartbleed
        'heartbleed': 'HEARTBLEED',
       
        # Benign traffic
        'benign': 'BENIGN',
        'normal': 'BENIGN',
        'legitimate': 'BENIGN'
    }
   
    # Check for exact matches first
    if pred_lower in exact_mappings:
        result = exact_mappings[pred_lower]
        logger.info(f"✅ EXACT MATCH: '{original_prediction}' -> {result}")
        return result
   
    # If no exact match, try to identify the threat type more carefully
    logger.warning(f"⚠️  NO EXACT MATCH for: '{original_prediction}'")
    
    # Check what the actual prediction contains
    if 'web' in pred_lower and 'attack' in pred_lower:
        logger.info(f"🌐 WEB ATTACK pattern detected: '{original_prediction}' -> WEB_ATTACK")
        return 'WEB_ATTACK'
    elif 'dos' in pred_lower or 'ddos' in pred_lower:
        logger.info(f"💥 DDOS pattern detected: '{original_prediction}' -> DDOS")
        return 'DDOS'
    elif 'port' in pred_lower and 'scan' in pred_lower:
        logger.info(f"🔍 PORT_SCAN pattern detected: '{original_prediction}' -> PORT_SCAN")
        return 'PORT_SCAN'
    elif 'brute' in pred_lower and 'force' in pred_lower:
        logger.info(f"🔨 BRUTE_FORCE pattern detected: '{original_prediction}' -> BRUTE_FORCE")
        return 'BRUTE_FORCE'
    elif 'infiltrat' in pred_lower:
        logger.info(f"🕵️ INFILTRATION pattern detected: '{original_prediction}' -> INFILTRATION")
        return 'INFILTRATION'
    elif 'bot' in pred_lower:
        logger.info(f"🤖 BOTNET pattern detected: '{original_prediction}' -> BOTNET")
        return 'BOTNET'
    elif 'heartbleed' in pred_lower:
        logger.info(f"💔 HEARTBLEED pattern detected: '{original_prediction}' -> HEARTBLEED")
        return 'HEARTBLEED'
    elif 'benign' in pred_lower or 'normal' in pred_lower:
        logger.info(f"✅ BENIGN pattern detected: '{original_prediction}' -> BENIGN")
        return 'BENIGN'
   
    # If still no match, log it and return as-is
    logger.error(f"❌ UNKNOWN THREAT: '{original_prediction}' - add this to exact_mappings!")
    return original_prediction.upper()

def debug_predictions(predictions):
    """
    Enhanced debug function to show ALL predictions and their classifications
    """
    logger.info("🔍 DEBUGGING ALL PREDICTIONS:")
    
    # Count occurrences of each prediction
    prediction_counts = {}
    for pred in predictions:
        pred_clean = str(pred).strip()
        prediction_counts[pred_clean] = prediction_counts.get(pred_clean, 0) + 1
    
    # Show unique predictions and their classifications
    for i, (pred, count) in enumerate(prediction_counts.items()):
        threat_type = classify_threat_type(pred)
        is_threat = is_malicious_traffic(pred)
        logger.info(f"   {i+1}. '{pred}' (×{count}) -> '{threat_type}' (Threat: {is_threat})")
        
        # Stop after showing 10 unique predictions to avoid spam
        if i >= 9:
            logger.info(f"   ... and {len(prediction_counts) - 10} more unique predictions")
            break

def is_malicious_traffic(prediction: str) -> bool:
    """
    FIXED: Determine if traffic is malicious based on threat classification
    """
    threat_type = classify_threat_type(prediction)
    return threat_type != 'BENIGN'

class RealDataProcessor:
    def __init__(self):
        self.batch_size = 50
        self.expected_total_messages = 800
        self.messages_processed_count = 0
        self.final_summary_generated = False  # ADDED: Track if summary was generated
       
        # Simple stats tracking
        self.stats = {
            'total_processed': 0,
            'threats_detected': 0,
            'benign_detected': 0,
            'processing_errors': 0,
            'start_time': time.time(),
            'threat_types': defaultdict(int)
        }
       
        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
       
        # ML features (USING ORIGINAL CSV FORMAT - Title Case with spaces)
        self.ml_features = [
            "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
            "Total Length of Fwd Packets", "Total Length of Bwd Packets",
            "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
            "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std",
            "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
            "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
            "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
            "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
            "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
            "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
            "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
            "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio", "Average Packet Size",
            "Avg Fwd Segment Size", "Avg Bwd Segment Size", "Fwd Header Length.1",
            "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
            "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
            "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
            "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward",
            "Active Mean", "Active Std", "Active Max", "Active Min",
            "Idle Mean", "Idle Std", "Idle Max", "Idle Min"
        ]

    def signal_handler(self, sig, frame):
        logger.info("🛑 Received shutdown signal, generating final summary...")
        self.generate_final_summary()
        sys.exit(0)

    def wait_for_ml_service(self, max_wait=60):
        """Wait for ML service - SIMPLIFIED"""
        logger.info("Waiting for ML service...")
        start_time = time.time()
       
        while time.time() - start_time < max_wait:
            try:
                # Try the root endpoint instead of /status
                response = requests.get('http://ml-service-1:8000/', timeout=5)
                if response.status_code == 200:
                    logger.info("ML service is ready!")
                    return True
            except:
                pass
           
            logger.info("🔄 Still waiting for ML service...")
            time.sleep(5)
       
        logger.error("❌ ML service not ready")
        return False

    def create_consumer(self):
        """Create Kafka consumer"""
        try:
            consumer = KafkaConsumer(
                'network-data',
                bootstrap_servers=['kafka:29092'],
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',
                enable_auto_commit=True,
                max_poll_records=100,
                consumer_timeout_ms=30000  # ADDED: Timeout to prevent infinite waiting
            )
            logger.info("📡 Connected to Kafka")
            return consumer
        except Exception as e:
            logger.error(f"Failed to create consumer: {e}")
            return None

    def extract_features(self, message):
        """Extract features from message using original CSV format (Title Case)"""
        features = {}
       
        for feature_name in self.ml_features:
            if feature_name in message:
                try:
                    value = float(message[feature_name])
                    if np.isnan(value) or np.isinf(value):
                        value = 0.0
                    features[feature_name] = value
                except:
                    features[feature_name] = 0.0
            else:
                features[feature_name] = 0.0
       
        return features

    def process_batch(self, messages):
        """
        FIXED: Process batch with improved threat classification
        """
        if not messages:
            return []
       
        batch_results = []
       
        try:
            # Extract features for all messages
            feature_batch = []
            for message in messages:
                features = self.extract_features(message)
                feature_batch.append(features)
           
            # Send to ML service
            payload = {'features': feature_batch}
            response = requests.post(
                'http://ml-service-1:8000/predict_batch',
                json=payload,
                timeout=30
            )
           
            if response.status_code != 200:
                logger.error(f"❌ ML service error: {response.status_code}")
                self.stats['processing_errors'] += len(messages)
                return []
           
            predictions = response.json().get('predictions', [])
           
            # Debug predictions for first batch
            if self.stats['total_processed'] == 0:
                debug_predictions(predictions)
           
            # Process results with improved classification
            for i, (message, prediction) in enumerate(zip(messages, predictions)):
                # Use improved threat classification
                threat_type = classify_threat_type(str(prediction))
                is_threat = is_malicious_traffic(str(prediction))
               
                result = {
                    'message_id': message.get('Flow ID', f'msg_{i}'),
                    'source_ip': message.get('Source IP', 'unknown'),
                    'predicted_class': threat_type,
                    'is_malicious': is_threat,
                    'actual_label': message.get('Label', 'Unknown')
                }
               
                batch_results.append(result)
               
                # Update stats with improved classification
                if is_threat:
                    self.stats['threats_detected'] += 1
                    self.stats['threat_types'][threat_type] += 1
                    # Log individual detections for debugging
                    logger.info(f"⚠️ THREAT DETECTED: {prediction} -> {threat_type}")
                else:
                    self.stats['benign_detected'] += 1
           
            self.stats['total_processed'] += len(batch_results)
            logger.info(f"✅ Processed batch: {len(batch_results)} messages")
           
        except Exception as e:
            logger.error(f"❌ Batch processing error: {e}")
            self.stats['processing_errors'] += len(messages)
       
        return batch_results

    def generate_batch_summary(self, batch_results):
        """Show progress after each batch - FIXED logic"""
        if not batch_results:
            return False
       
        progress = (self.stats['total_processed'] / self.expected_total_messages) * 100
       
        logger.info(f"📊 Progress: {self.stats['total_processed']}/{self.expected_total_messages} ({progress:.1f}%)")
        logger.info(f"   🎯 Threats: {self.stats['threats_detected']} | ✅ Benign: {self.stats['benign_detected']}")
       
        # FIXED: Check completion condition properly
        if self.stats['total_processed'] >= self.expected_total_messages:
            logger.info("🎉 Reached expected total - generating final summary!")
            self.generate_final_summary()
            return True
       
        return False

    def generate_final_summary(self):
        """
        FIXED: Generate final summary with accurate threat classification and prevent duplicates
        """
        # FIXED: Prevent duplicate summaries
        if self.final_summary_generated:
            logger.info("ℹ️ Final summary already generated, skipping duplicate")
            return
        
        self.final_summary_generated = True
        
        total_time = time.time() - self.stats['start_time']
       
        logger.info("\n" + "="*60)
        logger.info("🏁 FINAL CYBERSECURITY ANALYSIS SUMMARY")
        logger.info("="*60)
        logger.info("📈 PROCESSING RESULTS:")
        logger.info(f"   Total Messages Analyzed: {self.stats['total_processed']}")
        logger.info(f"   Processing Time: {total_time:.1f} seconds")
        logger.info(f"   Messages/Second: {self.stats['total_processed']/max(total_time, 1):.1f}")
        logger.info("")
       
        logger.info("🔍 THREAT DETECTION:")
        logger.info(f"   Threats Detected: {self.stats['threats_detected']}")
        logger.info(f"   Benign Traffic: {self.stats['benign_detected']}")
       
        if self.stats['total_processed'] > 0:
            threat_rate = (self.stats['threats_detected'] / self.stats['total_processed']) * 100
            logger.info(f"   Threat Rate: {threat_rate:.1f}%")
       
        logger.info("")
       
        if self.stats['threat_types']:
            logger.info("🎯 ATTACK TYPES DETECTED:")
            # Sort by count (descending)
            sorted_threats = sorted(self.stats['threat_types'].items(), key=lambda x: x[1], reverse=True)
           
            for threat_type, count in sorted_threats:
                percentage = (count / self.stats['threats_detected']) * 100 if self.stats['threats_detected'] > 0 else 0
                logger.info(f"   - {threat_type}: {count} ({percentage:.1f}%)")
        else:
            logger.info("🎯 No threats detected in this analysis")
       
        if self.stats['processing_errors'] > 0:
            logger.info(f"\n❌ ERRORS: {self.stats['processing_errors']}")
       
        logger.info("="*60)
        logger.info("✅ ANALYSIS COMPLETE!")
        logger.info("="*60 + "\n")

    def run(self):
        """FIXED: Main processing loop with guaranteed final summary"""
        logger.info("🚀 Starting Cybersecurity Data Processor...")
       
        # Wait for ML service
        if not self.wait_for_ml_service():
            logger.error("❌ Cannot start without ML service")
            return
       
        # Create consumer
        consumer = self.create_consumer()
        if not consumer:
            logger.error("❌ Cannot start without Kafka")
            return
       
        logger.info(f"🎯 Ready to process up to {self.expected_total_messages} network traffic messages")
        logger.info("📝 Note: Final summary will be generated when target is reached OR when no more messages arrive")
       
        try:
            current_batch = []
            no_message_count = 0
            max_no_message_iterations = 10  # Exit after 10 empty polls
           
            for message in consumer:
                # Reset no-message counter when we get data
                no_message_count = 0
                
                # Skip non-data messages (FILE_START, FILE_END markers)
                if isinstance(message.value, dict) and message.value.get('message_type') in ['FILE_START', 'FILE_END']:
                    logger.info(f"📋 Received marker: {message.value.get('message_type')} for {message.value.get('source_file', 'unknown')}")
                    continue
                
                current_batch.append(message.value)
               
                # Process when batch is full
                if len(current_batch) >= self.batch_size:
                    batch_results = self.process_batch(current_batch)
                    completed = self.generate_batch_summary(batch_results)
                    current_batch = []
                   
                    if completed:
                        logger.info("🎉 Processing completed - target reached!")
                        break
               
                # Safety check - prevent infinite processing
                if self.stats['total_processed'] >= self.expected_total_messages:
                    logger.info("🛑 Target message count reached!")
                    break
            
            # FIXED: Handle timeout scenario
            else:
                # This executes if the for loop completes without breaking (timeout)
                logger.info("⏰ No more messages available (consumer timeout)")
                no_message_count += 1
           
            # FIXED: Always process remaining messages in batch
            if current_batch:
                logger.info(f"🔄 Processing final batch of {len(current_batch)} messages...")
                batch_results = self.process_batch(current_batch)
                self.generate_batch_summary(batch_results)
           
            # FIXED: Always generate final summary if not already generated
            if not self.final_summary_generated:
                logger.info("📊 Generating final summary for completed processing...")
                self.generate_final_summary()
               
        except KeyboardInterrupt:
            logger.info("🛑 Processing stopped by user")
            if not self.final_summary_generated:
                self.generate_final_summary()
        except Exception as e:
            logger.error(f"❌ Processing error: {e}")
            if not self.final_summary_generated:
                self.generate_final_summary()
        finally:
            logger.info("🔄 Closing consumer connection...")
            consumer.close()
            
            # FINAL SAFETY CHECK: Always generate summary
            if not self.final_summary_generated:
                logger.info("🆘 Emergency final summary generation...")
                self.generate_final_summary()

if __name__ == "__main__":
    processor = RealDataProcessor()
    processor.run()