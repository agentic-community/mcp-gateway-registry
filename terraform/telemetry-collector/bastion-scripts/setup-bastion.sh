#!/bin/bash
# Post-deploy script: installs tools and copies helper scripts to bastion host
# Usage: ./bastion-scripts/setup-bastion.sh
# Run from terraform/telemetry-collector/ after terraform apply
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TF_DIR="$(dirname "$SCRIPT_DIR")"
SSH_KEY="${SSH_KEY:-~/.ssh/id_ed25519}"

cd "$TF_DIR"

# Get values from terraform outputs
BASTION_IP=$(terraform output -raw bastion_public_ip 2>/dev/null)
DOCDB_ENDPOINT=$(terraform output -raw documentdb_endpoint 2>/dev/null)
SECRET_ARN=$(terraform output -raw documentdb_secret_arn 2>/dev/null)
AWS_REGION=$(terraform output -raw aws_region 2>/dev/null || echo "us-east-1")

if [ -z "$BASTION_IP" ] || [ "$BASTION_IP" = "Bastion not enabled" ]; then
    echo "Error: Could not get bastion IP. Is bastion_enabled = true?"
    exit 1
fi

echo "Setting up bastion host at $BASTION_IP..."

# Step 1: Install mongosh, jq, and download CA bundle on bastion
echo "Installing mongosh and dependencies..."
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" ec2-user@"$BASTION_IP" 'bash -s' <<'REMOTE'
sudo bash -c '
cat > /etc/yum.repos.d/mongodb-org-7.repo << EOF
[mongodb-org-7]
name=MongoDB Repository
baseurl=https://repo.mongodb.org/yum/amazon/2023/mongodb-org/7.0/x86_64/
gpgcheck=1
enabled=1
gpgkey=https://pgp.mongodb.com/server-7.0.asc
EOF
dnf install -y mongodb-mongosh aws-cli jq
'
[ -f ~/global-bundle.pem ] || curl -sS https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem -o ~/global-bundle.pem
REMOTE

# Step 2: Create bastion.env with terraform output values
echo "Copying configuration and scripts..."
cat > /tmp/bastion.env <<EOF
DOCDB_ENDPOINT="$DOCDB_ENDPOINT"
SECRET_ARN="$SECRET_ARN"
AWS_REGION="$AWS_REGION"
EOF

# Step 3: SCP scripts and config to bastion
scp -o StrictHostKeyChecking=no -i "$SSH_KEY" \
    /tmp/bastion.env \
    "$SCRIPT_DIR/connect.sh" \
    "$SCRIPT_DIR/query.sh" \
    ec2-user@"$BASTION_IP":~/

# Step 4: Make scripts executable
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" ec2-user@"$BASTION_IP" \
    'chmod +x ~/connect.sh ~/query.sh'

rm /tmp/bastion.env

echo ""
echo "Bastion setup complete!"
echo "  ssh -i $SSH_KEY ec2-user@$BASTION_IP"
echo "  ./connect.sh   # interactive DocumentDB session"
echo "  ./query.sh     # quick telemetry summary"
