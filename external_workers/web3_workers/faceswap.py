import asyncio
import os
import logging
import uuid

import httpx
from camunda.external_task.external_task_worker import ExternalTaskWorker
from camunda.external_task.external_task import ExternalTask
import requests
import replicate

from web3_workers.dall_e_blender import fetch_art
from web3_workers.utils import upload_file_to_s3_binary

# Set up logging
logging.basicConfig(level=logging.INFO)

# Environment variables
REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN')
CAMUNDA_URL = os.getenv('CAMUNDA_URL', 'http://demo:demo@localhost:8080/engine-rest')
REPLICATE_MODEL = "yan-ops/face_swap:d5900f9ebed33e7ae08a07f17e0d98b4ebc68ab9528a70462afc3899cfe23bab"
TOPIC_NAME = os.getenv('TOPIC_NAME', "FaceSwapWorker")
AWS_S3_BUCKET = os.getenv('AWS_S3_BUCKET', 'pixelpact')

if not REPLICATE_API_TOKEN:
    raise ValueError("REPLICATE_API_TOKEN is not defined in the environment variables")

# Configure headers for Replicate API
headers = {
    'Authorization': f'Token {REPLICATE_API_TOKEN}',
    'Content-Type': 'application/json'
}

default_config = {
    "maxTasks": 1,
    "lockDuration": 10000,
    "asyncResponseTimeout": 5000,
    "retries": 3,
    "retryTimeout": 15000,
    "sleepSeconds": 10
}

def run_replicate_model(source_image, target_image, replies: int = 1):
    """Run the replicate model with given images."""
    image = replicate.run(
        REPLICATE_MODEL,
        input={'source_image': source_image,
                'target_image':  target_image}
    )
    if image.get('status') != 'succeed':
        if replies > 0:
            return run_replicate_model(source_image, target_image, replies - 1)
        else:
            return {'error_code': 510, 'error': f"replicate model error: {image.get('msg')} in source: {source_image},"
                                                f"or target{target_image}"}
    if image:
        return image


def run_replicate_model_and_upload(source_image, target_image):
    """Run the replicate model with given images and upload the output to S3."""
    image = run_replicate_model(source_image, target_image)
    if image.get('error_code') and image['error_code'] == 510:
        logging.error(f"Error during face swap operation: {image['error']}")
        return image
    image_url = image.get('image')
    # Assume response contains image URLs directly
    if image_url:
        s3_file_name = f"fs/{uuid.uuid4()}.jpg"
        with httpx.stream("GET", image_url) as response:
            image_bytes = response.read()
        # Upload to S3
        return {'s3_url': upload_file_to_s3_binary(image_bytes, AWS_S3_BUCKET, s3_file_name),
                'status': 'succeed'}
    else:
        raise Exception("No images were returned from the Replicate API.")


def handle_task(task: ExternalTask):
    variables = task.get_variables()

    target_image = variables.get('blended_image_url')
    if variables.get('ingested_art_id'):
        loop = asyncio.get_event_loop()
        source_art = loop.run_until_complete(fetch_art(variables['ingested_art_id']))
        image_type = source_art.get('type')
        if image_type != 'person':
            return task.failure(
                "wrong_image_type",
                f"Expected image type 'person', but received '{image_type}'",
                max_retries=0,
                retry_timeout=1000
            )
    else:
        return task.failure(
            "missing_ingested_art_id",
            "No source image provided",
            max_retries=0,
            retry_timeout=1000
        )

    source_image = source_art.get('img_picture')
    try:
        result = run_replicate_model_and_upload(source_image, target_image)
        if result.get('error_code') and result['error_code'] == 510:
            return task.bpmn_error(
                "cannot_swap_faces",
                result['error'],
            )
        else:
            return task.complete({'image_url': result['s3_url']})
    except Exception as e:
        logging.error(f"Error during face swap operation: {str(e)}")
        return task.failure(
            "FaceSwapError",
            str(e),
            max_retries=1,
            retry_timeout=1000
        )



if __name__ == '__main__':
    # Worker initialization with logging
    logging.info("Initializing Camunda External Task Worker")
    ExternalTaskWorker(worker_id="1", base_url=CAMUNDA_URL,
                       config=default_config).subscribe([TOPIC_NAME], handle_task)
