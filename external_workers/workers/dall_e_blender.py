"""
const response = await openai.images.generate({
            model: "dall-e-3",
            prompt: descriptivePrompt,
            size: "1024x1024",
            quality: "standard",
            n: 1,
        });

        const generatedImageUrls = response.data.map(image => image.url);
        const savedImageUrls = await uploadImageFromUrl(generatedImageUrls, user_id, worker);

        const processVariables = new Variables();
        processVariables.set('blended_image_url', savedImageUrls)
        await taskService.complete(task, processVariables);
    } catch (error) {
        console.error('Error when trying to generate image with DALL-E:', error);
        // await taskService.handleBpmnError(task, 'DaleeFailure', error)
        await taskService.handleFailure(task, {
            errorMessage: error.message,
            errorDetails: error.stack,
            retries: 0,
            retryTimeout: 0,
        });
        throw error;
    }

"""


import asyncio
import os
import logging
import uuid
from typing import Optional

import httpx
from camunda.external_task.external_task import ExternalTask, TaskResult
from camunda.external_task.external_task_worker import ExternalTaskWorker
from openai import AsyncOpenAI
from openai.types import ImagesResponse

from workers.utils import upload_file_to_s3_binary

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Environment variables to configure the script
TOPIC_NAME = os.getenv('TOPIC_NAME', "DallEGenerateArtBlender")
CAMUNDA_URL = os.getenv('CAMUNDA_URL', 'http://demo:demo@localhost:8080/engine-rest')
ARTWORKS_URL = os.getenv('ARTWORKS_URL', 'http://localhost:5000/api/arts')
AWS_S3_BUCKET = os.getenv('AWS_S3_BUCKET', 'pixelpact')

# Logging the script startup
logging.info("Starting DallEGenerateArtBlender Prompt script")

# Configuration for the Camunda External Task Client
default_config = {
    "maxTasks": 1,
    "lockDuration": 10000,
    "asyncResponseTimeout": 5000,
    "retries": 3,
    "retryTimeout": 15000,
    "sleepSeconds": 10
}

# Initialize the AsyncOpenAI client with an API key from environment variables
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def generate_image(prompt):
    """Generate an image using OpenAI's DALL-E model."""
    try:
        response: ImagesResponse = await client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            n=1
        )
        if response.data:
            image_url = response.data[0].url
        s3_file_name = f"blender_worker/{uuid.uuid4()}.jpg"
        with httpx.stream("GET", image_url) as response:
            image_bytes = response.read()
        # Upload to S3
        return upload_file_to_s3_binary(image_bytes, AWS_S3_BUCKET, s3_file_name)

    except Exception as e:
        logging.error(f"Failed to generate image: {str(e)}")
        raise


async def fetch_art(art_id: str) -> Optional[dict]:
    async with httpx.AsyncClient() as _client:
        response = await _client.get(f"{ARTWORKS_URL}/{art_id}")
        if response.status_code == 200:
            return response.json()
        else:
            logging.error(f"Failed to fetch art with ID {art_id}")
            return None


async def generate_prompt(ingested_art_description, art_description):
    """Generate a prompt for an image combining descriptions of a person and art."""
    prompt_text = (
        f"Create a prompt for an image generation that will result in an artwork, valued at a million dollars and "
        f"suitable for the world's top galleries. The image should feature a person resembling {ingested_art_description} "
        f"in a setting that incorporates elements or styles from {art_description}. The face of the person should be "
        f"clearly visible and created with a sense of depth, as if the person is positioned at least 1 meter away from "
        f"the viewer. The resulting image should capture the essence of luxury and exclusivity, combining the individual's "
        f"features with the specified art styles in a harmonious and visually appealing manner. Max length 250 words."
    )
    try:
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            max_tokens=300,
            n=1,
            messages=[
                {"role": "system", "content": "You are a helpful assistant designed to generate prompts for image creation."},
                {"role": "user", "content": prompt_text},

            ]
        )
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        else:
            return "No valid description was generated."
    except Exception as e:
        print(f"Error in generating prompt: {str(e)}")
        return f"Failed to generate prompt due to an error: {str(e)}"


async def blend_art_and_ingested_art(art_id: str, ingested_art_id: str):
    original_art, ingested_art_id = await asyncio.gather(
        fetch_art(art_id),
        fetch_art(ingested_art_id)
    )
    if not original_art or not ingested_art_id:
        return False, "Failed to fetch art data."

    generated_prompt = await generate_prompt(ingested_art_id['description_prompt'], original_art['description_prompt'])

    image_url = await generate_image(generated_prompt)

    if image_url:
        return True, image_url
    else:
        return False, "Failed to generate image."


# Function to handle tasks from Camunda
def handle_task(task: ExternalTask) -> TaskResult:
    # Task handling code with added logging
    logging.info(f"Handling task")
    variables = task.get_variables()
    art_id = variables.get("art_id")
    ingested_art_id = variables.get("ingested_art_id")

    loop = asyncio.get_event_loop()
    try:
        status, blended_image = loop.run_until_complete(blend_art_and_ingested_art(art_id, ingested_art_id))
        if not status:
            return task.bpmn_error(
                "art_description_generation_failed",
                blended_image,
                variables
            )
        variables["blended_image_url"] = blended_image
        return task.complete(variables)
    except Exception as e:
        logging.error(f"Error during blending operation: {str(e)}")
        return task.failure(
            "BlendingFailure",
            f"Failed to blend images.{str(e)[:50]}",
            max_retries=1,
            retry_timeout=1000
        )


if __name__ == '__main__':
    # Worker initialization with logging
    logging.info("Initializing Camunda External Task Worker")
    ExternalTaskWorker(worker_id="1", base_url=CAMUNDA_URL,
                       config=default_config).subscribe([TOPIC_NAME], handle_task)
