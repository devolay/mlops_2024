import os
import uuid

from aws_cdk.aws_s3 import Bucket
from aws_cdk.aws_ecr_assets import DockerImageAsset, Platform
from aws_cdk import RemovalPolicy, Stack
from constructs import Construct


class InitialSetupStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs):
        super().__init__(scope, id, **kwargs)

        bucket_sufix = os.environ["BUCKET_NAME"].lower()
        bucket_name = f"{uuid.uuid4().hex.lower()}-{bucket_sufix}"

        # Setup an S3 bucket
        self.bucket = Bucket(
            self,
            bucket_name,
            bucket_name=bucket_name,
            versioned=False,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        # Build and Push Docker Image to ECR
        self.docker_image = DockerImageAsset(
            self,
            os.environ["DOCKER_IMAGE"],
            directory="./",
            platform=Platform.LINUX_AMD64,
            build_args={"bucket_name": bucket_name},
        )
