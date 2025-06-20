# stdlib
import os

# third party
import braintrust
from dotenv import load_dotenv

# first party
from src.server import prompt

load_dotenv()

project = braintrust.projects.create(name=os.getenv("BRAINTRUST_PROJECT_NAME"))


MODEL = "gpt-4o-mini"
NAME = "Semantic Layer Prompt (SDK)"
SLUG = f"semantic-layer-prompt-sdk-{MODEL}"


system_prompt = prompt.BASIC_INSTRUCTIONS

project.prompts.create(
    name=NAME,
    description="Semantic Layer Agent",
    model=MODEL,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "{{{input}}}"},
    ],
)
