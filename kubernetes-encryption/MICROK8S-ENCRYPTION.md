# MicroK8s Secrets Encryption Setup

Complete guide for enabling encryption at rest for Kubernetes secrets in MicroK8s.

## Overview

By default, Kubernetes stores secrets in etcd as base64-encoded data (not encrypted). This setup enables encryption at rest using the `secretbox` encryption provider, protecting secrets from etcd database compromise.

### Architecture

**Without Encryption:**
```
Secret → kube-apiserver → etcd (base64 plaintext)
```

**With Encryption:**
```
Secret → kube-apiserver → Encryption Layer → etcd (encrypted)
```

## Repository Contents

- `encryption-config.yaml` - Template encryption configuration
- `setup-microk8s-encryption.sh` - Automated setup script
- `verify-encryption.sh` - Verification script
- `MICROK8S-ENCRYPTION.md` - This documentation

## Prerequisites

- MicroK8s installed and running
- Root/sudo access to the server
- Basic kubectl knowledge

## Quick Start

### Step 1: Run Setup Script

```bash
sudo ./setup-microk8s-encryption.sh
```

This will:
- Generate a secure 32-byte encryption key
- Create encryption config at `/var/snap/microk8s/current/encryption-config.yaml`
- Update kube-apiserver arguments
- Restart MicroK8s services

### Step 2: Wait for Services

Wait 30 seconds for MicroK8s services to stabilize:

```bash
sleep 30
microk8s kubectl get nodes
```

### Step 3: Verify Encryption

```bash
sudo ./verify-encryption.sh
```

### Step 4: Encrypt Existing Secrets

All **NEW** secrets will be automatically encrypted. To encrypt existing secrets:

```bash
microk8s kubectl get secrets --all-namespaces -o json | microk8s kubectl replace -f -
```

## Manual Setup (Alternative)

If you prefer manual setup:

### 1. Generate Encryption Key

```bash
head -c 32 /dev/urandom | base64
```

### 2. Create Encryption Config

Create `/var/snap/microk8s/current/encryption-config.yaml`:

```yaml
apiVersion: apiserver.config.k8s.io/v1
kind: EncryptionConfiguration
resources:
  - resources:
      - secrets
    providers:
      - secretbox:
          keys:
            - name: key1
              secret: <YOUR_BASE64_KEY>
      - identity: {}
```

### 3. Update kube-apiserver

Add to `/var/snap/microk8s/current/args/kube-apiserver`:

```
--encryption-provider-config=/var/snap/microk8s/current/encryption-config.yaml
```

### 4. Restart MicroK8s

```bash
sudo systemctl restart snap.microk8s.daemon-kubelite
```

## Security Considerations

### What This Protects

 Protects against etcd database compromise  
 Encrypts secrets at rest  
 Secures sensitive configuration data

### What This Does NOT Protect

 Does not protect if attacker has root access to the server  
 Does not encrypt secrets in transit (TLS handles that)  
 Keys are stored locally on the server

### Recommendations

- **File Permissions:** Encryption config is set to 600 (owner read/write only)
- **Backup:** Keep secure backup of `/var/snap/microk8s/current/encryption-config.yaml`
- **Key Rotation:** Periodically rotate encryption keys
- **Production:** Consider external KMS (AWS KMS, Azure Key Vault) for production

##  Encryption Providers

This setup uses `secretbox`:

- Strong NaCl-based encryption
- Requires 32-byte base64 key
- Better than AES-CBC/AES-GCM
- Good for most use cases

## Troubleshooting

### MicroK8s Won't Start

```bash
# Check logs
sudo journalctl -u snap.microk8s.daemon-kubelite -f

# Restore backup
sudo cp /var/snap/microk8s/current/args/kube-apiserver.backup \
         /var/snap/microk8s/current/args/kube-apiserver
sudo systemctl restart snap.microk8s.daemon-kubelite
```

### Secrets Still Unencrypted

```bash
# Verify encryption is enabled
ps aux | grep kube-apiserver | grep encryption

# Re-encrypt all secrets
microk8s kubectl get secrets --all-namespaces -o json | microk8s kubectl replace -f -
```

### Permission Denied Errors

Ensure encryption config has correct permissions:

```bash
sudo chmod 600 /var/snap/microk8s/current/encryption-config.yaml
```

## Converting ConfigMaps to Secrets

If you have sensitive data in ConfigMaps, convert them to Secrets:

```bash
# Export ConfigMap
microk8s kubectl get configmap <name> -o yaml > configmap.yaml

# Edit and change:
# kind: ConfigMap  →  kind: Secret
# type: Opaque (add this line)

# Apply as Secret
microk8s kubectl apply -f configmap.yaml
```

## References

- [Kubernetes Encryption Documentation](https://kubernetes.io/docs/tasks/administer-cluster/encrypt-data/)
- [MicroK8s Documentation](https://microk8s.io/docs)
- [4sysops Encryption Tutorial](https://4sysops.com/archives/encrypt-kubernetes-secrets-at-rest/)

## Issue Reference

Resolves #340 - Setup encryption for the Kubernetes cluster

---

**Note:** This encryption setup significantly improves security posture but should be part of a comprehensive security strategy including network policies, RBAC, and regular security audits.