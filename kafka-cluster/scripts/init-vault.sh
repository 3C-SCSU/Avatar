#!/bin/bash

# Initialize Vault for Kafka Encryption
# This script sets up the KV secrets engine and creates encryption keys

echo "Configuring Vault for Kafka encryption..."

# Enable KV version 2 secrets engine
docker exec -e VAULT_ADDR='http://127.0.0.1:8200' -e VAULT_TOKEN='root-token' vault \
  vault secrets enable -version=2 -path=secret kv 2>/dev/null || echo "KV engine already enabled"

# Create encryption key for eeg-brainwave-data topic
docker exec -e VAULT_ADDR='http://127.0.0.1:8200' -e VAULT_TOKEN='root-token' vault \
  vault kv put secret/kafka/eeg-brainwave-data \
  key=$(openssl rand -base64 32) \
  description="Encryption key for EEG brainwave data topic"

# Create encryption key for prediction-results topic
docker exec -e VAULT_ADDR='http://127.0.0.1:8200' -e VAULT_TOKEN='root-token' vault \
  vault kv put secret/kafka/prediction-results \
  key=$(openssl rand -base64 32) \
  description="Encryption key for ML prediction results topic"

# Create encryption key for drone-commands topic
docker exec -e VAULT_ADDR='http://127.0.0.1:8200' -e VAULT_TOKEN='root-token' vault \
  vault kv put secret/kafka/drone-commands \
  key=$(openssl rand -base64 32) \
  description="Encryption key for drone commands topic"

echo "Vault configuration complete!"
echo "Verifying keys..."

# Verify keys were created
docker exec -e VAULT_ADDR='http://127.0.0.1:8200' -e VAULT_TOKEN='root-token' vault \
  vault kv list secret/kafka

echo ""
echo "Encryption keys stored in Vault at:"
echo "  - secret/kafka/eeg-brainwave-data"
echo "  - secret/kafka/prediction-results"
echo "  - secret/kafka/drone-commands"