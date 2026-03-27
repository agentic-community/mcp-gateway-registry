# VPC for telemetry collector infrastructure
resource "aws_vpc" "telemetry" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "telemetry-collector-vpc"
  }
}

# Internet Gateway for NAT Gateway
resource "aws_internet_gateway" "telemetry" {
  vpc_id = aws_vpc.telemetry.id

  tags = {
    Name = "telemetry-collector-igw"
  }
}

# Public subnets for NAT Gateway (2 AZs for high availability)
resource "aws_subnet" "public" {
  count = 2

  vpc_id                  = aws_vpc.telemetry.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, count.index)
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name = "telemetry-collector-public-${count.index + 1}"
  }
}

# Private subnets for Lambda and DocumentDB (2 AZs for DocumentDB requirement)
resource "aws_subnet" "private" {
  count = 2

  vpc_id            = aws_vpc.telemetry.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 8, count.index + 10)
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = {
    Name = "telemetry-collector-private-${count.index + 1}"
  }
}

# Elastic IPs for NAT Gateways
resource "aws_eip" "nat" {
  count  = 2
  domain = "vpc"

  tags = {
    Name = "telemetry-collector-nat-eip-${count.index + 1}"
  }

  depends_on = [aws_internet_gateway.telemetry]
}

# NAT Gateways for Lambda internet access (2 for high availability)
resource "aws_nat_gateway" "telemetry" {
  count = 2

  allocation_id = aws_eip.nat[count.index].id
  subnet_id     = aws_subnet.public[count.index].id

  tags = {
    Name = "telemetry-collector-nat-${count.index + 1}"
  }

  depends_on = [aws_internet_gateway.telemetry]
}

# Route table for public subnets
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.telemetry.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.telemetry.id
  }

  tags = {
    Name = "telemetry-collector-public-rt"
  }
}

# Associate public subnets with public route table
resource "aws_route_table_association" "public" {
  count = 2

  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

# Route tables for private subnets (one per AZ for NAT Gateway routing)
resource "aws_route_table" "private" {
  count = 2

  vpc_id = aws_vpc.telemetry.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.telemetry[count.index].id
  }

  tags = {
    Name = "telemetry-collector-private-rt-${count.index + 1}"
  }
}

# Associate private subnets with private route tables
resource "aws_route_table_association" "private" {
  count = 2

  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private[count.index].id
}

# Security group for DocumentDB cluster (no inline rules to avoid cycle)
resource "aws_security_group" "documentdb" {
  name        = "telemetry-collector-documentdb-sg"
  description = "Security group for DocumentDB cluster - allow Lambda access on port 27017"
  vpc_id      = aws_vpc.telemetry.id

  egress {
    description = "Allow all outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "telemetry-collector-documentdb-sg"
  }
}

# Security group for Lambda function
# All rules are inline to prevent Terraform from removing standalone rules
resource "aws_security_group" "lambda" {
  name        = "telemetry-collector-lambda-sg"
  description = "Security group for Lambda function - allow outbound to DocumentDB and internet"
  vpc_id      = aws_vpc.telemetry.id

  tags = {
    Name = "telemetry-collector-lambda-sg"
  }
}

# Standalone rules to avoid inline/standalone conflict and break SG cycles
resource "aws_security_group_rule" "lambda_egress_https" {
  type              = "egress"
  description       = "HTTPS for AWS APIs (DynamoDB, Secrets Manager, CloudWatch)"
  from_port         = 443
  to_port           = 443
  protocol          = "tcp"
  cidr_blocks       = ["0.0.0.0/0"]
  security_group_id = aws_security_group.lambda.id
}

resource "aws_security_group_rule" "documentdb_ingress_from_lambda" {
  type                     = "ingress"
  description              = "MongoDB protocol from Lambda"
  from_port                = 27017
  to_port                  = 27017
  protocol                 = "tcp"
  security_group_id        = aws_security_group.documentdb.id
  source_security_group_id = aws_security_group.lambda.id
}

resource "aws_security_group_rule" "lambda_egress_to_documentdb" {
  type                     = "egress"
  description              = "DocumentDB access"
  from_port                = 27017
  to_port                  = 27017
  protocol                 = "tcp"
  security_group_id        = aws_security_group.lambda.id
  source_security_group_id = aws_security_group.documentdb.id
}

# Data source for available AZs
data "aws_availability_zones" "available" {
  state = "available"
}
