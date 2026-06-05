# 🛡️ Cybersecurity Threat Detection Pipeline

A production-inspired, real-time cybersecurity threat detection system built using Apache Kafka, FastAPI, Docker, and Machine Learning.

This project simulates a modern Security Operations Center (SOC) workflow by streaming network traffic, processing flow data in real time, and classifying cyber threats using a trained machine learning model.

---

## 🚀 Overview

Traditional signature-based intrusion detection systems struggle to identify previously unseen attacks and evolving threat patterns.

This project addresses that challenge by building an end-to-end threat detection pipeline capable of:

- Streaming network traffic data in real time
- Processing and validating network flow records
- Detecting malicious traffic using Machine Learning
- Classifying attack categories
- Running as independently scalable microservices

The system was developed during my internship at Bharat Electronics Limited (BEL) under the Network Centric Systems (NCS) division.

---

## 🏗️ Architecture

```text
┌────────────────────┐
│  CICIDS2017 Data   │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│   Data Producer    │
│  CSV → JSON Stream │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│      Kafka         │
│  network-data      │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│   Data Processor   │
│ Feature Extraction │
│ Batch Processing   │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│     ML Service     │
│ FastAPI + RF Model │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ Threat Detection & │
│ Security Analytics │
└────────────────────┘
```

---

## ✨ Key Features

### Real-Time Data Streaming
- Apache Kafka based message pipeline
- Decoupled producer-consumer architecture
- Fault-tolerant event streaming

### Machine Learning Threat Detection
- Random Forest Classifier
- Multi-class attack classification
- Conservative false-positive reduction strategy

### Production-Oriented Design
- Dockerized microservices
- Service isolation
- Health monitoring endpoints
- Structured logging

### Cybersecurity Focus
Detects multiple attack categories including:

- DDoS
- DoS Hulk
- Port Scan
- Botnet Traffic
- FTP-Patator
- SSH-Patator
- Web Attacks
- Infiltration Attempts

---

## 📊 Dataset

The model is trained using the **CICIDS2017 Dataset**, one of the most widely used cybersecurity intrusion detection datasets.

Dataset characteristics:

- Realistic network traffic
- 70+ flow features
- Multiple attack categories
- Benign and malicious traffic
- Large-scale labeled data

---

## 🧠 Machine Learning Pipeline

### Data Preparation

- Multi-file CSV ingestion
- Missing value handling
- Feature cleaning
- Class balancing
- Attack preservation preprocessing

### Feature Engineering

Selected:

- Flow statistics
- Packet statistics
- TCP flag counts
- Inter-arrival times
- Active/Idle metrics

Total ML Features Used:

```text
77 Features
```

### Model

```text
Algorithm: Random Forest
Trees: 200
Max Depth: 12
Class Weighting: Custom Balanced
```

Special focus was placed on reducing false positives since excessive false alerts reduce the practical usability of security systems.

---

## 📈 Performance

### Results

| Metric | Value |
|----------|---------|
| Accuracy | 98.6% |
| Training Samples | 103,678 |
| Features Used | 77 |
| Architecture | Kafka + FastAPI + Docker |

High precision was achieved for:

- BENIGN
- DDoS
- PortScan
- Bot
- FTP-Patator

---

## 🛠️ Tech Stack

### Backend

- Python
- FastAPI

### Data Streaming

- Apache Kafka
- Zookeeper

### Machine Learning

- Scikit-Learn
- Pandas
- NumPy

### Infrastructure

- Docker
- Docker Compose

### Development

- VS Code
- Git

---

## 📂 Project Structure

```text
Cybersecurity-Threat-Detection-Pipeline
│
├── data producer service/
│   ├── producer.py
│
├── data preprocessor service/
│   ├── processor.py
│
├── ML service/
│   ├── ml_service.py
│
├── dataset/
│
├── docker-compose.yml
│
└── README.md
```

---

## ⚙️ Running the Project

### Clone Repository

```bash
git clone https://github.com/your-username/Cybersecurity-Threat-Detection-Pipeline.git
cd Cybersecurity-Threat-Detection-Pipeline
```

### Start Services

```bash
docker-compose up --build
```

Services launched:

- Kafka
- Zookeeper
- Data Producer
- Data Processor
- ML Service

---

## 🔍 Example Workflow

1. CSV network flow files are loaded.
2. Producer streams records to Kafka.
3. Processor consumes messages in batches.
4. Features are extracted.
5. ML Service performs inference.
6. Threats are detected and classified.
7. Logs and summaries are generated.

---

## 🎯 Future Improvements

- Real-time dashboard using Grafana
- Prometheus monitoring
- Kubernetes deployment
- Online learning models
- Threat intelligence integration
- SIEM integration
- Explainable AI (SHAP/LIME)
- Alert notification system

---

## 👩‍💻 Author

**Vidhi Agarwal**

Integrated M.Tech (ECE)
Jaypee Institute of Information Technology

Areas of Interest:

- Cybersecurity
- Cloud Infrastructure
- Distributed Systems
- DevOps
- Machine Learning Systems

---

## ⭐ Why This Project?

I wanted to move beyond training a machine learning model and build a system closer to how real-world threat detection pipelines operate.

This project combines streaming systems, distributed messaging, containerized microservices, and machine learning into a single end-to-end cybersecurity workflow, providing hands-on exposure to concepts used in modern security operations and cloud-native architectures.
