# Bastion host for DocumentDB access
# Free tier: t2.micro, Amazon Linux 2023, in public subnet

# IAM role for bastion to read Secrets Manager
resource "aws_iam_role" "bastion" {
  count = var.bastion_enabled ? 1 : 0
  name  = "telemetry-collector-bastion-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "bastion_secrets" {
  count = var.bastion_enabled ? 1 : 0
  name  = "bastion-read-secrets"
  role  = aws_iam_role.bastion[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = "secretsmanager:GetSecretValue"
      Resource = aws_secretsmanager_secret.documentdb_credentials.arn
    }]
  })
}

resource "aws_iam_instance_profile" "bastion" {
  count = var.bastion_enabled ? 1 : 0
  name  = "telemetry-collector-bastion-profile"
  role  = aws_iam_role.bastion[0].name
}

# Key pair for SSH access
resource "aws_key_pair" "bastion" {
  count      = var.bastion_enabled ? 1 : 0
  key_name   = "telemetry-collector-bastion"
  public_key = var.bastion_public_key

  tags = {
    Name = "telemetry-collector-bastion"
  }
}

# Security group for bastion
resource "aws_security_group" "bastion" {
  count       = var.bastion_enabled ? 1 : 0
  name        = "telemetry-collector-bastion-sg"
  description = "Bastion host for DocumentDB access"
  vpc_id      = aws_vpc.telemetry.id

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.bastion_allowed_cidrs
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "telemetry-collector-bastion-sg"
  }
}

# Allow bastion to reach DocumentDB
resource "aws_security_group_rule" "docdb_from_bastion" {
  count                    = var.bastion_enabled ? 1 : 0
  type                     = "ingress"
  from_port                = 27017
  to_port                  = 27017
  protocol                 = "tcp"
  security_group_id        = aws_security_group.documentdb.id
  source_security_group_id = aws_security_group.bastion[0].id
  description              = "MongoDB from bastion"
}

# Latest Amazon Linux 2023 AMI
data "aws_ami" "amazon_linux_2023" {
  count       = var.bastion_enabled ? 1 : 0
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# Bastion EC2 instance (t2.micro — free tier eligible)
resource "aws_instance" "bastion" {
  count = var.bastion_enabled ? 1 : 0

  ami                         = data.aws_ami.amazon_linux_2023[0].id
  instance_type               = "t2.micro"
  subnet_id                   = aws_subnet.public[0].id
  vpc_security_group_ids      = [aws_security_group.bastion[0].id]
  key_name                    = aws_key_pair.bastion[0].key_name
  associate_public_ip_address = true
  iam_instance_profile        = aws_iam_instance_profile.bastion[0].name

  tags = {
    Name = "telemetry-collector-bastion"
  }
}
