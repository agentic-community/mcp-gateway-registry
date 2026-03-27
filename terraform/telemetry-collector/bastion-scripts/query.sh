#!/bin/bash
# Run a quick summary query against telemetry collections
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

export MONGOSH_PASSWORD="$PASSWORD"
mongosh "mongodb://$USERNAME@$DOCDB_ENDPOINT:27017/$DATABASE" \
  --tls \
  --tlsCAFile ~/global-bundle.pem \
  --retryWrites false \
  --authenticationMechanism SCRAM-SHA-1 \
  --password "$MONGOSH_PASSWORD" \
  --quiet \
  --eval '
    print("=== Startup Events ===");
    print("Total:", db.startup_events.countDocuments());
    print("Last 5:");
    db.startup_events.find({}, {_id:0})
      .sort({_id:-1}).limit(5).forEach(printjson);

    print("\n=== Heartbeat Events ===");
    print("Total:", db.heartbeat_events.countDocuments());
    print("Last 5:");
    db.heartbeat_events.find({}, {_id:0})
      .sort({_id:-1}).limit(5).forEach(printjson);

    print("\n=== Storage Backend Breakdown ===");
    db.startup_events.aggregate([
      {$group: {_id: "$storage", count: {$sum: 1}}},
      {$sort: {count: -1}}
    ]).forEach(printjson);
  '
unset MONGOSH_PASSWORD
