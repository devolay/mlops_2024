from dotenv import load_dotenv
from aws_cdk import App

from stacks.initial_setup import InitialSetupStack
from stacks.lambda_setup import LambdaStack

load_dotenv()

app = App()

initial_setup_stack = InitialSetupStack(app, "InitialSetupStack")

lambda_stack = LambdaStack(app, "LambdaStack", initial_setup_stack=initial_setup_stack)
lambda_stack.add_dependency(initial_setup_stack)

app.synth()
