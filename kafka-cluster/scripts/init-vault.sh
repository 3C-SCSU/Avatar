#!/bin/bash

# Initialize Vault for Kafka Encryption
echo "Configuring Vault for Kafka encryption..."

# Enable Transit secrets engine (for encryption as a service)
docker exec -e VAULT_ADDR='http://127.0.0.1:8200' -e VAULT_TOKEN='root-token' vault \
  vault secrets enable transit 2>/dev/null || echo "Transit engine already enabled"

# Create encryption key for Kafka
docker exec -e VAULT_ADDR='http://127.0.0.1:8200' -e VAULT_TOKEN='root-token' vault \
  vault write -f transit/keys/kafka-encryption-key

echo "Vault Transit engine configured!"
echo "Encryption key 'kafka-encryption-key' created"