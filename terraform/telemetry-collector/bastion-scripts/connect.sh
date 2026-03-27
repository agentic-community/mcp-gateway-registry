#!/bin/bash
# Fetch credentials from Secrets Manager and connect to DocumentDB interactively
set -e

# Configuration (set by setup.sh)
source ~/bastion.env

SECRET=$(aws secretsmanager get-secret-value \
  --secret-id "$SECRET_ARN" \
  --region "$AWS_REGION" \
  --query SecretString --output text)

USERNAME=$(echo "$SECRET" | jq -r .username)
PASSWORD=$(echo "$SECRET" | jq -r .password)
DATABASE=$(echo "$SECRET" | jq -r .database)

echo "Connecting to DocumentDB as $USERNAME..."
export MONGOSH_PASSWORD="$PASSWORD"
mongosh "mongodb://$USERNAME@$DOCDB_ENDPOINT:27017/$DATABASE" \
  --tls \
  --tlsCAFile ~/global-bundle.pem \
  --retryWrites false \
  --authenticationMechanism SCRAM-SHA-1 \
  --password "$MONGOSH_PASSWORD"
unset MONGOSH_PASSWORD
