#!/bin/bash

# Kafka Topic Setup Script for Avatar Project
# Creates topics with 3 partitions distributed across 3 brokers

echo "Waiting for Kafka cluster to be ready..."
sleep 15

KAFKA_BROKER="kafka-broker-1:29092"

echo "Creating EEG data streaming topic..."
docker exec kafka-broker-1 kafka-topics --create \
  --topic eeg-brainwave-data \
  --bootstrap-server $KAFKA_BROKER \
  --partitions 3 \
  --replication-factor 3 \
  --config retention.ms=3600000 \
  --config segment.ms=300000

echo "Creating prediction results topic..."
docker exec kafka-broker-1 kafka-topics --create \
  --topic prediction-results \
  --bootstrap-server $KAFKA_BROKER \
  --partitions 3 \
  --replication-factor 3 \
  --config retention.ms=3600000

echo "Creating drone command topic..."
docker exec kafka-broker-1 kafka-topics --create \
  --topic drone-commands \
  --bootstrap-server $KAFKA_BROKER \
  --partitions 3 \
  --replication-factor 3 \
  --config retention.ms=1800000

echo -e "\nTopics created successfully!"
echo -e "\nListing all topics:"
docker exec kafka-broker-1 kafka-topics --list \
  --bootstrap-server $KAFKA_BROKER

echo -e "\nDescribing eeg-brainwave-data topic:"
docker exec kafka-broker-1 kafka-topics --describe \
  --topic eeg-brainwave-data \
  --bootstrap-server $KAFKA_BROKER

echo -e "\nKafka cluster is ready for Avatar data streaming!"
echo "Topics:"
echo "  - eeg-brainwave-data: Streams EEG data from GUI5.py"
echo "  - prediction-results: ML predictions from prediction service"
echo "  - drone-commands: Commands to execute on drones"