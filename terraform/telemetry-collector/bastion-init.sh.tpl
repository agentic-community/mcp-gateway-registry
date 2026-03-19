#!/bin/bash
# Bastion host initialization — installs mongosh and sets up query helper
set -e

# Install mongosh
cat > /etc/yum.repos.d/mongodb-org-7.repo << 'EOF'
[mongodb-org-7]
name=MongoDB Repository
baseurl=https://repo.mongodb.org/yum/amazon/2023/mongodb-org/7.0/x86_64/
gpgcheck=1
enabled=1
gpgkey=https://pgp.mongodb.com/server-7.0.asc
EOF
dnf install -y mongodb-mongosh

# Install AWS CLI v2 (already present on AL2023, but ensure it's there)
dnf install -y aws-cli jq

# Download DocumentDB CA bundle
curl -sS https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem \
  -o /home/ec2-user/global-bundle.pem
chown ec2-user:ec2-user /home/ec2-user/global-bundle.pem

# Write the connect script
cat > /home/ec2-user/connect.sh << 'SCRIPT'
#!/bin/bash
# Fetch credentials from Secrets Manager and connect to DocumentDB
SECRET=$(aws secretsmanager get-secret-value \
  --secret-id "${secret_arn}" \
  --region "${aws_region}" \
  --query SecretString --output text)

USERNAME=$(echo "$SECRET" | jq -r .username)
PASSWORD=$(echo "$SECRET" | jq -r .password)
DATABASE=$(echo "$SECRET" | jq -r .database)

echo "Connecting to DocumentDB as $USERNAME..."
mongosh "mongodb://${docdb_endpoint}:27017/$DATABASE" \
  --tls \
  --tlsCAFile ~/global-bundle.pem \
  --retryWrites false \
  --authenticationMechanism SCRAM-SHA-1 \
  --username "$USERNAME" \
  --password "$PASSWORD"
SCRIPT

# Write a quick query script (non-interactive)
cat > /home/ec2-user/query.sh << 'SCRIPT'
#!/bin/bash
# Run a quick summary query against telemetry collections
SECRET=$(aws secretsmanager get-secret-value \
  --secret-id "${secret_arn}" \
  --region "${aws_region}" \
  --query SecretString --output text)

USERNAME=$(echo "$SECRET" | jq -r .username)
PASSWORD=$(echo "$SECRET" | jq -r .password)
DATABASE=$(echo "$SECRET" | jq -r .database)

mongosh "mongodb://${docdb_endpoint}:27017/$DATABASE" \
  --tls \
  --tlsCAFile ~/global-bundle.pem \
  --retryWrites false \
  --authenticationMechanism SCRAM-SHA-1 \
  --username "$USERNAME" \
  --password "$PASSWORD" \
  --quiet \
  --eval '
    print("=== Startup Events ===");
    print("Total:", db.startup_events.countDocuments());
    print("Last 5:");
    db.startup_events.find({}, {instance_id:1, v:1, os:1, storage:1, ts:1, _id:0})
      .sort({_id:-1}).limit(5).forEach(printjson);

    print("\n=== Heartbeat Events ===");
    print("Total:", db.heartbeat_events.countDocuments());
    print("Last 5:");
    db.heartbeat_events.find({}, {instance_id:1, v:1, uptime_hours:1, servers_count:1, ts:1, _id:0})
      .sort({_id:-1}).limit(5).forEach(printjson);

    print("\n=== Storage Backend Breakdown ===");
    db.startup_events.aggregate([
      {$group: {_id: "$storage", count: {$sum: 1}}},
      {$sort: {count: -1}}
    ]).forEach(printjson);
  '
SCRIPT

chmod +x /home/ec2-user/connect.sh /home/ec2-user/query.sh
chown ec2-user:ec2-user /home/ec2-user/connect.sh /home/ec2-user/query.sh

# Write README
cat > /home/ec2-user/README.md << 'EOF'
# Telemetry DocumentDB Bastion

## Connect interactively
./connect.sh

## Run summary query
./query.sh

## Common mongosh queries (after ./connect.sh)
```
use telemetry

# Count all events
db.startup_events.countDocuments()
db.heartbeat_events.countDocuments()

# Recent events
db.startup_events.find().sort({_id:-1}).limit(10).pretty()

# Events by storage backend
db.startup_events.aggregate([{$group:{_id:"$storage",count:{$sum:1}}}])

# Events by OS
db.startup_events.aggregate([{$group:{_id:"$os",count:{$sum:1}}}])

# Events by version
db.startup_events.aggregate([{$group:{_id:"$v",count:{$sum:1}}},{$sort:{count:-1}}])
```
EOF
chown ec2-user:ec2-user /home/ec2-user/README.md
