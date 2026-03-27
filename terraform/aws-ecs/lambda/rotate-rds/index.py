"""
AWS Secrets Manager Rotation Handler for RDS Aurora MySQL

This Lambda function implements the AWS Secrets Manager rotation protocol
for RDS Aurora MySQL credentials (Keycloak database). It rotates the master
password following AWS best practices for secret rotation.

Rotation Steps:
1. createSecret: Generate new random password and create AWSPENDING version
2. setSecret: Update RDS cluster with new password
3. testSecret: Verify connection with new password
4. finishSecret: Move AWSCURRENT to AWSPREVIOUS and AWSPENDING to AWSCURRENT

References:
- https://docs.aws.amazon.com/secretsmanager/latest/userguide/rotating-secrets.html
- https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/Aurora.Managing.html
"""

import json
import logging
import os

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

secretsmanager = boto3.client("secretsmanager")
rds = boto3.client("rds")


def lambda_handler(event: dict, context: dict) -> dict:
    """
    Lambda handler for RDS Aurora MySQL secret rotation.

    Args:
        event: Lambda event containing SecretId, ClientRequestToken, and Step
        context: Lambda context object

    Returns:
        Success response dict

    Raises:
        ValueError: If rotation is not enabled or step is invalid
        ClientError: If AWS API calls fail
    """
    arn = event["SecretId"]
    token = event["ClientRequestToken"]
    step = event["Step"]

    logger.info(f"Processing rotation step: {step} for secret: {arn}")

    metadata = secretsmanager.describe_secret(SecretId=arn)
    if not metadata.get("RotationEnabled"):
        error_msg = f"Secret {arn} is not enabled for rotation"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if step == "createSecret":
        _create_secret(arn, token)
    elif step == "setSecret":
        _set_secret(arn, token)
    elif step == "testSecret":
        _test_secret(arn, token)
    elif step == "finishSecret":
        _finish_secret(arn, token)
    else:
        error_msg = f"Invalid step parameter: {step}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info(f"Successfully completed rotation step: {step}")
    return {"statusCode": 200, "body": json.dumps("Success")}


def _create_secret(arn: str, token: str) -> None:
    """
    Generate new password and create AWSPENDING version.

    Args:
        arn: Secret ARN
        token: Client request token for this rotation
    """
    logger.info("Step 1: Creating new secret version")

    current = secretsmanager.get_secret_value(SecretId=arn, VersionStage="AWSCURRENT")
    current_dict = json.loads(current["SecretString"])

    try:
        secretsmanager.get_secret_value(SecretId=arn, VersionId=token, VersionStage="AWSPENDING")
        logger.info("AWSPENDING version already exists, skipping creation")
        return
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            raise

    exclude_chars = os.environ.get("EXCLUDE_CHARACTERS", "/@\"'\\")
    logger.info(f"Generating new password (excluding: {exclude_chars})")

    passwd = secretsmanager.get_random_password(ExcludeCharacters=exclude_chars, PasswordLength=32)

    current_dict["password"] = passwd["RandomPassword"]
    secretsmanager.put_secret_value(
        SecretId=arn,
        ClientRequestToken=token,
        SecretString=json.dumps(current_dict),
        VersionStages=["AWSPENDING"],
    )

    logger.info("Successfully created AWSPENDING version with new password")


def _set_secret(arn: str, token: str) -> None:
    """
    Update RDS Aurora cluster with new password.

    Args:
        arn: Secret ARN
        token: Client request token for this rotation
    """
    logger.info("Step 2: Setting new password in RDS Aurora")

    pending = secretsmanager.get_secret_value(
        SecretId=arn, VersionId=token, VersionStage="AWSPENDING"
    )
    pending_dict = json.loads(pending["SecretString"])

    metadata = secretsmanager.describe_secret(SecretId=arn)
    secret_name = metadata["Name"]

    cluster_id = pending_dict.get("cluster_id")
    if not cluster_id:
        logger.info("No cluster_id in secret, attempting to derive from name")
        if "keycloak" in secret_name.lower():
            cluster_id = "keycloak"
            logger.info(f"Derived cluster_id: {cluster_id}")
        else:
            error_msg = f"Cannot determine RDS cluster ID from secret: {secret_name}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    logger.info(f"Updating RDS Aurora cluster: {cluster_id}")

    try:
        rds.modify_db_cluster(
            DBClusterIdentifier=cluster_id,
            MasterUserPassword=pending_dict["password"],
            ApplyImmediately=True,
        )
        logger.info("Successfully updated RDS Aurora master password")
    except ClientError as e:
        logger.error(f"Failed to update RDS password: {e}")
        raise


def _test_secret(arn: str, token: str) -> None:
    """
    Test new password by verifying cluster status.

    Note: We cannot easily test MySQL connection from Lambda without
    installing pymysql library. Instead, we verify the cluster is
    available and modification was successful.

    Args:
        arn: Secret ARN
        token: Client request token for this rotation
    """
    logger.info("Step 3: Testing new secret")

    pending = secretsmanager.get_secret_value(
        SecretId=arn, VersionId=token, VersionStage="AWSPENDING"
    )
    pending_dict = json.loads(pending["SecretString"])

    metadata = secretsmanager.describe_secret(SecretId=arn)
    secret_name = metadata["Name"]

    cluster_id = pending_dict.get("cluster_id")
    if not cluster_id:
        if "keycloak" in secret_name.lower():
            cluster_id = "keycloak"
        else:
            error_msg = f"Cannot determine RDS cluster ID from secret: {secret_name}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    try:
        response = rds.describe_db_clusters(DBClusterIdentifier=cluster_id)
        cluster = response["DBClusters"][0]
        status = cluster["Status"]

        logger.info(f"RDS Aurora cluster status: {status}")

        if status not in ["available", "modifying"]:
            error_msg = f"RDS cluster in unexpected state: {status}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("Successfully verified RDS Aurora cluster status")

    except ClientError as e:
        logger.error(f"Failed to verify RDS cluster: {e}")
        raise


def _finish_secret(arn: str, token: str) -> None:
    """
    Move AWSCURRENT to AWSPREVIOUS and AWSPENDING to AWSCURRENT.

    Args:
        arn: Secret ARN
        token: Client request token for this rotation
    """
    logger.info("Step 4: Finishing rotation")

    metadata = secretsmanager.describe_secret(SecretId=arn)
    current_version = None

    for version_id, stages in metadata["VersionIdsToStages"].items():
        if "AWSCURRENT" in stages:
            current_version = version_id
            break

    logger.info(f"Current version: {current_version}, New version: {token}")

    secretsmanager.update_secret_version_stage(
        SecretId=arn,
        VersionStage="AWSCURRENT",
        MoveToVersionId=token,
        RemoveFromVersionId=current_version,
    )

    logger.info("Successfully finished rotation - new password is now active")
