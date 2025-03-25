#!/usr/bin/env python
import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from autoevals import Levenshtein

from braintrust import EvalAsync
from langchain_openai_voice import VoiceToTextReactAgent
from server.prompt import INSTRUCTIONS
from server.settings import settings
from server.tools import create_tools
from server.vectorstore import SemanticLayerVectorStore


async def create_input_stream(text: str) -> AsyncIterator[str]:
    """Creates an async iterator that simulates a WebSocket stream"""
    print("[DEBUG] Starting input stream")

    # First send initial session update like the websocket does
    session_update = {
        "type": "session.update",
        "session": {
            "input_audio_transcription": {"model": "whisper-1"},
        },
    }
    print("[DEBUG] Yielding session update")
    yield json.dumps(session_update)

    # Then send the conversation item
    conversation_item = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": text}],
        },
    }
    print("[DEBUG] Yielding conversation item")
    yield json.dumps(conversation_item)

    # Send response.create to trigger model response
    print("[DEBUG] Sending response.create")
    yield json.dumps({"type": "response.create", "response": {}})

    print("[DEBUG] Input stream sent messages, now keeping connection alive")

    # Keep connection alive while waiting for responses
    try:
        while True:
            await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
    except (asyncio.CancelledError, StopAsyncIteration):
        print("[DEBUG] Input stream cancelled or completed - closing connection")
    except Exception as e:
        print(f"[DEBUG] Unexpected error in input stream: {str(e)}")

    print("[DEBUG] Input stream complete")


class OutputCollector:
    """Collects output from the agent for evaluation"""

    def __init__(self):
        self.outputs: list[dict[str, Any]] = []
        self.response_received = asyncio.Event()
        self.final_response = ""
        print("[DEBUG] OutputCollector initialized")

    async def send_text(self, output_str: str):
        """Simulates WebSocket's send_text method"""
        if not output_str:  # Skip empty strings
            return

        print(f"[DEBUG] OutputCollector received: {output_str}")
        try:
            output = json.loads(output_str)
            self.outputs.append(output)

            # Check for different types of responses
            if output.get("type") == "assistant.response":
                print("[DEBUG] Received assistant.response")
                self.final_response = output.get("text", "")
                self.response_received.set()
            elif output.get("type") == "error":
                print(f"[DEBUG] Error from API: {output.get('error', 'Unknown error')}")
                self.response_received.set()
            elif output.get("type") == "function_call_output":
                print("[DEBUG] Received function_call_output")
                try:
                    # Parse the output string which contains the query result
                    result = json.loads(output.get("output", "{}"))
                    if result.get("type") == "query_result":
                        # Extract the query part and set as final response
                        query_data = result.get("query", {})
                        query_data = {k: v for k, v in query_data.items() if v}
                        self.final_response = json.dumps(query_data)
                        self.response_received.set()
                        print(
                            f"[DEBUG] Set final response from query: {self.final_response}"
                        )
                except json.JSONDecodeError:
                    print("[DEBUG] Error decoding function call output")
            elif output.get("type") == "conversation.item.create":
                print("[DEBUG] Received conversation.item.create")
                item = output.get("item", {})
                if item.get("role") == "assistant":
                    print("[DEBUG] Processing assistant response")
                    # Extract text from content parts
                    content_parts = item.get("content", [])
                    text_parts = []
                    for part in content_parts:
                        if part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                    if text_parts:
                        self.final_response = " ".join(text_parts)
                        self.response_received.set()
                        print(f"[DEBUG] Set final response: {self.final_response}")

        except json.JSONDecodeError:
            print(f"[DEBUG] Error decoding output: {output_str}")


async def run_agent_with_input(agent: VoiceToTextReactAgent, input_text: str) -> str:
    """Run the agent with a simulated input and return its response"""
    print(f"[DEBUG] Starting run_agent_with_input with text: {input_text}")
    collector = OutputCollector()
    connection_task = None

    # Create a task for the agent connection
    async def run_connection():
        async def input_stream_generator():
            """Create an async generator that yields from create_input_stream"""
            try:
                async for item in create_input_stream(input_text):
                    yield item
            except (asyncio.CancelledError, StopAsyncIteration):
                print("[DEBUG] Input stream generator cancelled or completed")
            except Exception as e:
                print(f"[DEBUG] Error in input stream generator: {str(e)}")

        try:
            await agent.aconnect(input_stream_generator(), collector.send_text)
        except (asyncio.CancelledError, StopAsyncIteration):
            print("[DEBUG] Agent connection cancelled or completed")
        except Exception as e:
            print(f"[DEBUG] Error in agent connection: {str(e)}")

    try:
        print("[DEBUG] Starting agent connection")
        # Start the connection task
        connection_task = asyncio.create_task(run_connection())

        print("[DEBUG] Waiting for response")
        # Wait for response with timeout
        try:
            await asyncio.wait_for(collector.response_received.wait(), timeout=30.0)
            print(f"[DEBUG] Received response: {collector.final_response}")
            return collector.final_response
        except asyncio.TimeoutError:
            print("[DEBUG] Timeout waiting for response")
            return ""

    except Exception as e:
        print(f"[DEBUG] Error in run_agent_with_input: {str(e)}")
        return ""
    finally:
        # Ensure we clean up the connection task
        if connection_task and not connection_task.done():
            connection_task.cancel()
            try:
                await connection_task
            except (asyncio.CancelledError, StopAsyncIteration):
                pass
            except Exception as e:
                print(f"[DEBUG] Error cleaning up connection task: {str(e)}")


async def aiter_to_list(ait):
    """Helper to consume an async iterator to completion"""
    return [x async for x in ait]


def load_examples(jsonl_path: str) -> list[dict[str, Any]]:
    """Load examples from a JSONL file"""
    examples = []
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                example = json.loads(line)
                examples.append(example)
    return examples


async def initialize_vector_store():
    """Initialize and refresh the vector store"""
    vector_store = SemanticLayerVectorStore()
    await vector_store.refresh_stores()
    return vector_store


def create_agent() -> VoiceToTextReactAgent:
    """Create a VoiceToTextReactAgent instance for evaluation"""
    tools = create_tools()  # Get the same tools used in the web app

    return VoiceToTextReactAgent(
        model="gpt-4o-realtime-preview",
        tools=tools,
        instructions=INSTRUCTIONS,
        openai_api_key=settings.openai_api_key,
    )


async def evaluate_agent(agent: VoiceToTextReactAgent, input_text: str) -> str:
    """Evaluate a single input using the agent"""
    result = await run_agent_with_input(agent, input_text)
    return result


def main():
    """Run the evaluation using Braintrust"""
    # Load examples
    all_examples = load_examples("datasets/examples.jsonl")

    async def initialize_resources():
        """Initialize resources and process one example"""
        print("[DEBUG] Initializing vector store and creating agent...")
        await initialize_vector_store()
        agent = create_agent()
        print("[DEBUG] Setup complete")
        return agent

    async def run_eval():
        """Run the evaluation in an async context"""
        agent = await initialize_resources()
        experiment_name = f"voice_agent_eval_{uuid.uuid4()}"
        print(f"[DEBUG] Running experiment: {experiment_name}")

        async def eval_task(example: str | dict[str, Any]) -> str:
            """Task function that handles both string and dict inputs"""
            if isinstance(example, dict):
                input_text = example.get("input", "")
            else:
                input_text = str(example)
            print(f"[DEBUG] Processing input: {input_text}")
            return await evaluate_agent(agent, input_text)

        try:
            await EvalAsync(
                "Voice Agent",
                experiment_name=experiment_name,
                data=lambda: all_examples,
                task=eval_task,
                scores=[Levenshtein],
            )
        except Exception as e:
            print(f"[DEBUG] Error during evaluation: {str(e)}")
        finally:
            # Clean up any remaining tasks
            tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    # Run the async evaluation
    asyncio.run(run_eval())


if __name__ == "__main__":
    main()
