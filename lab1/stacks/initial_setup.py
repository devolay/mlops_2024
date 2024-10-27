import os

from aws_cdk.aws_s3 import Bucket
from aws_cdk.aws_ecr_assets import DockerImageAsset, DockerCacheOption
from aws_cdk.aws_ecr import Repository
from aws_cdk import RemovalPolicy, Stack
from constructs import Construct

class InitialSetupStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs):
        super().__init__(scope, id, **kwargs)

        # Setup an S3 bucket
        self.bucket = Bucket(
            self, os.environ["BUCKET_NAME"],
            versioned=False,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True
        )

        # Build and Push Docker Image to ECR
        self.docker_image = DockerImageAsset(self, os.environ["DOCKER_IMAGE"], directory="./", build_args={"platform": "linux/amd64"})

