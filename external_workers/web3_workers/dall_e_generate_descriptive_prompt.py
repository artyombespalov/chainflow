import asyncio
import os
import logging
from camunda.external_task.external_task import ExternalTask, TaskResult
from camunda.external_task.external_task_worker import ExternalTaskWorker
from openai import AsyncOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Environment variables to configure the script
TOPIC_NAME = os.getenv('TOPIC_NAME', "DallEGenerateDescriptivePrompt")
CAMUNDA_URL = os.getenv('CAMUNDA_URL', 'http://demo:demo@localhost:8080/engine-rest')

# Logging the script startup
logging.info("Starting Dali Generate Art Prompt script")

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


# Main function to describe images using OpenAI's model
async def describe_image_with_openai_vision(image_url, name, description, image_type) -> tuple[bool, str]:
    # Example of logging an operation
    logging.info(f"Generating description for {image_type} image: {image_url}")
    if image_type == 'person':
        prompt = "Generate a description of a person from a photograph. Focus on the individual's body type, facial features such as the shape of their face, the hair's length, style, and color, as well as any notable expressions or gestures they may be making. The description should serve to convey the person's likeness without revealing their personal identity. Maximum 250 words"
    else:
        prompt = "Provide a detailed analysis of the visual elements, colors, textures, and any distinctive stylistic features present in the image. Focus on describing the composition, any patterns or motifs, the use of light and shadow, and overall thematic presence. Maximum length 250 words."

    prompt = f"{prompt}\n\nArtwork Name: {name}\n\nArtwork Description: {description}"

    try:
        response = await client.chat.completions.create(model="gpt-4-vision-preview",
                                                        messages=[
                                                            {
                                                                "role": "user",
                                                                "content": [
                                                                    {"type": "text", "text": prompt},
                                                                    {"type": "image_url", "image_url": image_url}
                                                                ],
                                                            }
                                                        ])

        # Process the response
        if response.choices and len(response.choices) > 0:
            choice = response.choices[0]
            description_text = choice.message.content
            return True, description_text.strip()
        message = f"Error while generating description for image: {response}"
        logging.error(message)
        return False, message
    except Exception as e:
        message = f"Error while generating description for image: {str(e)}"
        logging.error(message)
        return False, message


# Function to handle tasks from Camunda
def handle_task(task: ExternalTask) -> TaskResult:
    # Task handling code with added logging
    logging.info(f"Handling task")
    variables = task.get_variables()
    img_art_thumbnail = variables.get("img_picture")
    art_name = variables.get("name")
    art_description = variables.get("description")
    image_type = variables.get("type")
    loop = asyncio.get_event_loop()
    status, result = loop.run_until_complete(describe_image_with_openai_vision(img_art_thumbnail, art_name,
                                                                            art_description, image_type))
    if not status:
        return task.bpmn_error(
            "art_description_generation_failed",
            result,
            variables
        )
    variables["description_prompt"] = result
    return task.complete(variables)


if __name__ == '__main__':
    # Worker initialization with logging
    logging.info("Initializing Camunda External Task Worker")
    ExternalTaskWorker(worker_id="1", base_url=CAMUNDA_URL, config=default_config).subscribe([TOPIC_NAME], handle_task)
