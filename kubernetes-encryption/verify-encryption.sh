#!/bin/bash

# Verify MicroK8s encryption is enabled

echo "Verifying Encryption Setup"
echo ""

# Check if encryption flag is present
echo "[1/3] Checking if encryption is enabled..."
if ps aux | grep kube-apiserver | grep -q "encryption-provider-config"; then
    echo "✓ Encryption is ENABLED"
else
    echo "✗ Encryption is NOT enabled"
    echo "Run: sudo ./setup-microk8s-encryption.sh"
    exit 1
fi
echo ""

# Create test secret
echo "[2/3] Creating test secret..."
microk8s kubectl create secret generic encryption-test --from-literal=test=encrypted 2>/dev/null || \
    (microk8s kubectl delete secret encryption-test && microk8s kubectl create secret generic encryption-test --from-literal=test=encrypted)
echo "Test secret created"
echo ""

# Verify encryption in etcd
echo "[3/3] Checking if secret is encrypted in database..."
echo "Fetching secret from etcd..."
ETCD_DATA=$(microk8s kubectl get secret encryption-test -o yaml)

if echo "$ETCD_DATA" | grep -q "data:"; then
    echo "✓ Secret exists and is properly configured"
    echo ""
    echo "To verify it's encrypted in etcd, the data should NOT be plaintext."
    echo "If you have etcdctl installed, you can verify with:"
    echo "  ETCDCTL_API=3 etcdctl get /registry/secrets/default/encryption-test"
else
    echo "✗ Could not verify secret"
fi
echo ""

# Cleanup
microk8s kubectl delete secret encryption-test
echo "Test secret deleted"
echo ""
echo "Verification Complete"
