import os
import boto3
import json
import torch
from PIL import Image
import io

model = torch.load(os.environ['MODEL_PATH'])
model.eval()

def lambda_handler(event, context):
    bucket_name = os.environ['BUCKET_NAME']
    input_key = f"input/{event['input_key']}"
    output_key = f"output/{event['input_key']}"
    
    # Initialize S3 client
    s3 = boto3.client('s3')
    
    # Download the image from S3
    response = s3.get_object(Bucket=bucket_name, Key=input_key)
    image_content = response['Body'].read()
    
    # Open the image
    image = Image.open(io.BytesIO(image_content)).convert('RGB')
    
    # Preprocess the image
    preprocess = torch.transforms.Compose([
        torch.transforms.Resize(256),
        torch.transforms.CenterCrop(224),
        torch.transforms.ToTensor(),
        torch.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch
    
    # Perform inference
    with torch.no_grad():
        output = model(input_batch)
    
    # Process the output as needed
    _, predicted = torch.max(output, 1)
    result = predicted.item()
    
    # Save the result back to S3
    s3.put_object(Bucket=bucket_name, Key=output_key, Body=json.dumps({'prediction': result}))
    
    return {
        'statusCode': 200,
        'body': json.dumps('Inference complete')
    }