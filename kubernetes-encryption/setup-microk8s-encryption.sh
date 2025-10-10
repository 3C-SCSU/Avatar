#!/bin/bash

# MicroK8s Secrets Encryption Setup Script
# Configures encryption at rest for Kubernetes secrets in MicroK8s

set -e

echo "MicroK8s Secrets Encryption Setup"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root (use sudo)"
    exit 1
fi

# Generate 32-byte encryption key
echo "[1/5] Generating encryption key..."
ENCRYPTION_KEY=$(head -c 32 /dev/urandom | base64)
echo "Key generated successfully"
echo ""

# Create encryption config
echo "[2/5] Creating encryption configuration..."
cat > /var/snap/microk8s/current/encryption-config.yaml << EOF
apiVersion: apiserver.config.k8s.io/v1
kind: EncryptionConfiguration
resources:
  - resources:
      - secrets
    providers:
      - secretbox:
          keys:
            - name: key1
              secret: ${ENCRYPTION_KEY}
      - identity: {}
EOF

chmod 600 /var/snap/microk8s/current/encryption-config.yaml
echo "Encryption config created at /var/snap/microk8s/current/encryption-config.yaml"
echo ""

# Backup kube-apiserver args
echo "[3/5] Backing up kube-apiserver configuration..."
cp /var/snap/microk8s/current/args/kube-apiserver /var/snap/microk8s/current/args/kube-apiserver.backup
echo "Backup created"
echo ""

# Add encryption flag
echo "[4/5] Adding encryption flag to kube-apiserver..."
if grep -q "encryption-provider-config" /var/snap/microk8s/current/args/kube-apiserver; then
    echo "Encryption flag already exists, skipping..."
else
    echo "--encryption-provider-config=/var/snap/microk8s/current/encryption-config.yaml" >> /var/snap/microk8s/current/args/kube-apiserver
    echo "Flag added successfully"
fi
echo ""

# Restart MicroK8s
echo "[5/5] Restarting MicroK8s services..."
systemctl restart snap.microk8s.daemon-kubelite
echo "MicroK8s restarted"
echo ""

echo "Setup Complete!"
echo ""
echo "Next steps:"
echo "1. Wait 30 seconds for services to stabilize"
echo "2. Verify encryption: sudo ./verify-encryption.sh"
echo "3. Encrypt existing secrets: microk8s kubectl get secrets --all-namespaces -o json | microk8s kubectl replace -f -"
echo ""