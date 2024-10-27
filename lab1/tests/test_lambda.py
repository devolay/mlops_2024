import boto3
import json

# Initialize Lambda client
lambda_client = boto3.client('lambda')

# Prepare the event payload
payload = {
    "input_key": "your-image.jpg"
}

# Invoke the Lambda function
response = lambda_client.invoke(
    FunctionName='InferenceLambda',
    InvocationType='RequestResponse',
    Payload=json.dumps(payload),
)

# Read and print the response
response_payload = json.loads(response['Payload'].read())
print(response_payload)
