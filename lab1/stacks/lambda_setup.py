from aws_cdk.aws_lambda import DockerImageFunction, DockerImageCode
from aws_cdk.aws_s3 import Bucket
from aws_cdk.aws_ecr_assets import DockerImageAsset
from aws_cdk import Stack, Duration
from constructs import Construct

from stacks.initial_setup import InitialSetupStack


class LambdaStack(Stack):
    def __init__(
        self,
        scope: Construct,
        id: str,
        bucket: Bucket,
        docker_image: DockerImageAsset,
        **kwargs
    ):
        super().__init__(scope, id, **kwargs)

        # Create a Lambda function from the Docker image
        self.lambda_function = DockerImageFunction(
            self,
            "InferenceLambda",
            code=DockerImageCode.from_ecr(
                repository=docker_image.repository, tag_or_digest=docker_image.image_tag
            ),
            memory_size=2048,
            timeout=Duration.seconds(120),
        )

        # Grant Permissions
        bucket.grant_read_write(self.lambda_function)
        docker_image.repository.grant_pull(self.lambda_function)
