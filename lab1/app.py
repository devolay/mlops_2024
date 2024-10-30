from dotenv import load_dotenv
from aws_cdk import App

from stacks.initial_setup import InitialSetupStack
from stacks.lambda_setup import LambdaStack

load_dotenv()


app = App()

initial_setup_stack = InitialSetupStack(app, "InitialSetupStack")

lambda_stack = LambdaStack(
    app,
    "LambdaStack",
    bucket=initial_setup_stack.bucket,
    docker_image=initial_setup_stack.docker_image,
)
lambda_stack.add_dependency(initial_setup_stack)

app.synth()
