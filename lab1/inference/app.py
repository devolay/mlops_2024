import os
import json
import boto3
import torch
import logging

from PIL import Image
from io import BytesIO
from torchvision import transforms

log_level = os.environ.get("LAMBDA_LOG_LEVEL", "INFO")
logger = logging.getLogger()
logger.setLevel(log_level)


def handler(event, context):
    logger.info("Initializing model")
    model = torch.load(f"{os.environ['MODEL_PATH']}/model.pth")
    model.eval()
    logger.info("Model initialized")

    input_key = event["input_key"]
    bucket_name = os.environ["BUCKET_NAME"]
    logger.info(
        f"Received request to process image at s3://{bucket_name}/input/{input_key}"
    )

    s3_client = boto3.client("s3")
    logger.info("S3 client initialized")

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    logger.info("Loading image")
    img_data = s3_client.get_object(Bucket=bucket_name, Key=f"input/{input_key}")
    img = Image.open(BytesIO(img_data["Body"].read())).convert("RGB")
    logger.info("Image loaded")

    logger.info("Preprocessing image")
    input_tensor = transform(img).unsqueeze(0)

    logger.info("Running inference")
    with torch.no_grad():
        output = model(input_tensor)
    logger.info("Inference complete")

    logger.info("Postprocessing output")
    _, predicted = torch.max(output, 1)
    predicted_class = predicted.item()

    # Save output to S3
    logger.info("Saving output")
    output_key = f"output/{input_key}_prediction.json"
    s3_client.put_object(
        Bucket=bucket_name,
        Key=output_key,
        Body=json.dumps({"predicted_class": predicted_class}),
    )
    logger.info(f"Output saved at s3://{bucket_name}/{output_key}")

    return {
        "statusCode": 200,
        "body": json.dumps(f"Inference complete. Result saved at {output_key}"),
    }
