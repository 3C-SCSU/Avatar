#!/bin/bash

# Generate SSL certificates for Kafka broker encryption

echo "Generating SSL certificates for Kafka brokers..."

CERT_DIR="../ssl-certs"
mkdir -p $CERT_DIR

# Generate CA key and certificate
openssl req -new -x509 -keyout $CERT_DIR/ca-key -out $CERT_DIR/ca-cert -days 365 \
  -subj "/CN=Kafka-CA" -passout pass:kafka-ca-password

# Generate broker keystore and certificate for each broker
for i in 1 2 3; do
  echo "Generating certificate for broker $i..."
  
  # Create keystore
  keytool -keystore $CERT_DIR/broker$i.keystore.jks -alias broker$i -validity 365 \
    -genkey -keyalg RSA -storepass kafka-broker-password -keypass kafka-broker-password \
    -dname "CN=kafka-broker-$i,OU=Engineering,O=Avatar,L=Minneapolis,ST=MN,C=US"
  
  # Create certificate signing request
  keytool -keystore $CERT_DIR/broker$i.keystore.jks -alias broker$i -certreq \
    -file $CERT_DIR/broker$i.csr -storepass kafka-broker-password
  
  # Sign certificate with CA
  openssl x509 -req -CA $CERT_DIR/ca-cert -CAkey $CERT_DIR/ca-key -in $CERT_DIR/broker$i.csr \
    -out $CERT_DIR/broker$i-cert -days 365 -CAcreateserial -passin pass:kafka-ca-password
  
  # Import CA certificate into keystore
  keytool -keystore $CERT_DIR/broker$i.keystore.jks -alias CARoot -import -file $CERT_DIR/ca-cert \
    -storepass kafka-broker-password -noprompt
  
  # Import signed certificate into keystore
  keytool -keystore $CERT_DIR/broker$i.keystore.jks -alias broker$i -import \
    -file $CERT_DIR/broker$i-cert -storepass kafka-broker-password
  
  # Create truststore
  keytool -keystore $CERT_DIR/broker$i.truststore.jks -alias CARoot -import \
    -file $CERT_DIR/ca-cert -storepass kafka-broker-password -noprompt
done

echo "SSL certificates generated in $CERT_DIR/"
ls -la $CERT_DIR/