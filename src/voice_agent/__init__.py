import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from typing import Any

from braintrust import init_logger, traced, wrap_openai
from langchain_core.tools import BaseTool
from langchain_core.utils import secret_from_env
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, PrivateAttr, SecretStr

from server.settings import settings
from voice_agent.utils import amerge

DEFAULT_MODEL = "gpt-4o-realtime-preview-2024-10-01"
DEFAULT_URL = "wss://api.openai.com/v1/realtime"

EVENTS_TO_IGNORE = {
    "response.function_call_arguments.delta",
    "rate_limits.updated",
    "response.audio_transcript.delta",
    "response.created",
    "response.content_part.added",
    "response.content_part.done",
    "conversation.item.created",
    "response.audio.done",
    "session.created",
    "session.updated",
    "response.output_item.done",
}

bt_logger = init_logger(project=settings.braintrust_project_name)
client = wrap_openai(AsyncOpenAI(api_key=settings.openai_api_key))


@asynccontextmanager
async def connect() -> AsyncGenerator[
    tuple[
        Callable[[dict[str, Any] | str], Coroutine[Any, Any, None]],
        AsyncIterator[dict[str, Any]],
    ],
    None,
]:
    """
    async with connect(model="gpt-4o-realtime-preview-2024-10-01") as websocket:
        await websocket.send("Hello, world!")
        async for message in websocket:
            print(message)
    """

    async with client.beta.realtime.connect(model=settings.openai_model) as connection:
        try:

            async def send_event(event: dict[str, Any] | str) -> None:
                await connection.send(event)

            async def event_stream() -> AsyncIterator[dict[str, Any]]:
                try:
                    async for raw_event in connection:
                        if isinstance(raw_event, BaseModel):
                            yield raw_event.model_dump()
                        else:
                            yield json.loads(raw_event)
                except StopAsyncIteration:
                    print("[DEBUG] Connection stream completed")
                    raise  # Re-raise to trigger context manager cleanup

            stream: AsyncIterator[dict[str, Any]] = event_stream()

            yield send_event, stream
        except StopAsyncIteration:
            print("[DEBUG] Stream completed, closing connection")
        finally:
            print("[DEBUG] Cleaning up connection")
            await connection.close()


class VoiceToolExecutor(BaseModel):
    """
    Can accept function calls and emits function call outputs to a stream.
    """

    tools_by_name: dict[str, BaseTool]
    _trigger_future: asyncio.Future = PrivateAttr(default_factory=asyncio.Future)
    _lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    async def _trigger_func(self) -> dict:  # returns a tool call
        return await self._trigger_future

    async def add_tool_call(self, tool_call: dict) -> None:
        # lock to avoid simultaneous tool calls racing and missing
        # _trigger_future being
        async with self._lock:
            if self._trigger_future.done():
                # TODO: handle simultaneous tool calls better
                raise ValueError("Tool call adding already in progress")

            self._trigger_future.set_result(tool_call)

    @traced(name="create_tool_call_task")
    async def _create_tool_call_task(self, tool_call: dict) -> asyncio.Task[dict]:
        tool = self.tools_by_name.get(tool_call["name"])
        if tool is None:
            # immediately yield error, do not add task
            raise ValueError(
                f"tool {tool_call['name']} not found. "
                f"Must be one of {list(self.tools_by_name.keys())}"
            )

        # try to parse args
        try:
            args = json.loads(tool_call["arguments"])
        except json.JSONDecodeError:
            raise ValueError(
                f"failed to parse arguments `{tool_call['arguments']}`. Must be valid JSON."
            )

        @traced(name="run_tool")
        async def run_tool() -> dict:
            result = await tool.ainvoke(args)

            # If tool specifies return_direct and result is not an error, pass through the result directly
            if getattr(tool, "return_direct", False) and not (
                isinstance(result, dict) and result.get("type") == "error"
            ):
                return result

            # Otherwise wrap in conversation.item.create as before
            try:
                result_str = json.dumps(result)
            except TypeError:
                # not json serializable, use str
                result_str = str(result)
            return {
                "type": "conversation.item.create",
                "item": {
                    "id": tool_call["call_id"],
                    "call_id": tool_call["call_id"],
                    "type": "function_call_output",
                    "output": result_str,
                },
            }

        task = asyncio.create_task(run_tool())
        return task

    async def output_iterator(self) -> AsyncIterator[dict]:  # yield events
        trigger_task = asyncio.create_task(self._trigger_func())
        tasks = {trigger_task}
        while True:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                tasks.remove(task)
                if task == trigger_task:
                    async with self._lock:
                        self._trigger_future = asyncio.Future()
                    trigger_task = asyncio.create_task(self._trigger_func())
                    tasks.add(trigger_task)
                    tool_call = task.result()
                    try:
                        new_task = await self._create_tool_call_task(tool_call)
                        tasks.add(new_task)
                    except ValueError as e:
                        yield {
                            "type": "conversation.item.create",
                            "item": {
                                "id": tool_call["call_id"],
                                "call_id": tool_call["call_id"],
                                "type": "function_call_output",
                                "output": (f"Error: {str(e)}"),
                            },
                        }
                else:
                    yield task.result()


class VoiceToTextReactAgent(BaseModel):
    """
    A React Agent that accepts voice input but responds with text.
    Uses the same voice input infrastructure as OpenAIVoiceReactAgent but
    processes the model's responses as text instead of audio.
    """

    model: str
    api_key: SecretStr = Field(
        alias="openai_api_key",
        default_factory=secret_from_env("OPENAI_API_KEY", default=""),
    )
    instructions: str | None = None
    tools: list[BaseTool] | None = None
    url: str = Field(default=DEFAULT_URL)

    @traced(name="Semantic Layer Voice Agent")
    async def aconnect(
        self,
        input_stream: AsyncIterator[str],
        send_output_chunk: Callable[[str], Coroutine[Any, Any, None]],
    ) -> None:
        """
        Connect to the OpenAI API and send and receive messages.

        input_stream: AsyncIterator[str]
            Stream of input events to send to the model. Usually transports input_audio_buffer.append events from the microphone.
        output: Callable[[str], None]
            Callback to receive text output events from the model.
        """
        tools_by_name = {tool.name: tool for tool in self.tools}
        tool_executor = VoiceToolExecutor(tools_by_name=tools_by_name)

        async with connect() as (
            model_send,
            model_receive_stream,
        ):
            # Send tools and instructions with initial chunk
            tool_defs = [
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {"type": "object", "properties": tool.args},
                }
                for tool in tools_by_name.values()
            ]
            await model_send(
                {
                    "type": "session.update",
                    "session": {
                        "instructions": self.instructions,
                        "input_audio_transcription": {
                            "model": "whisper-1",
                        },
                        "tools": tool_defs,
                        "temperature": 0.6,
                    },
                }
            )

            try:
                async for stream_key, data_raw in amerge(
                    input_mic=input_stream,
                    output_speaker=model_receive_stream,
                    tool_outputs=tool_executor.output_iterator(),
                ):
                    try:
                        data = (
                            json.loads(data_raw)
                            if isinstance(data_raw, str)
                            else data_raw
                        )
                    except json.JSONDecodeError:
                        print("error decoding data:", data_raw)
                        continue

                    if stream_key == "input_mic":
                        await model_send(data)
                    elif stream_key == "tool_outputs":
                        if data.get("type") == "conversation.item.create":
                            # Regular tool output - send to model
                            await model_send(data)
                            await model_send(
                                {"type": "response.create", "response": {}}
                            )
                        else:
                            # Direct tool output - send straight to client
                            await send_output_chunk(json.dumps(data))
                    elif stream_key == "output_speaker":
                        t = data["type"]
                        if t == "response.audio.delta":
                            # Ignore audio deltas, we don't need them
                            pass
                        elif t == "response.audio_transcript.done":
                            print("model:", data["transcript"])
                            # Send the transcript to the frontend
                            await send_output_chunk(
                                json.dumps(
                                    {
                                        "type": "assistant.response",
                                        "text": data["transcript"],
                                    }
                                )
                            )
                        elif t == "error":
                            print("error:", data)
                            await send_output_chunk(
                                json.dumps(
                                    {
                                        "type": "error",
                                        "content": str(
                                            data.get("error", "Unknown error")
                                        ),
                                    }
                                )
                            )
                        elif t == "response.function_call_arguments.done":
                            await tool_executor.add_tool_call(data)
                        elif (
                            t == "conversation.item.input_audio_transcription.completed"
                        ):
                            # Send the transcribed text to the client
                            await send_output_chunk(
                                json.dumps(
                                    {
                                        "type": "user.input",
                                        "text": data["transcript"],
                                    }
                                )
                            )
                        elif t == "response.done":
                            usage = data.get("response", {}).get("usage", {})
                            print("data is: ", data)
                            print("usage is: ", usage)
                        elif t in EVENTS_TO_IGNORE:
                            pass
                        else:
                            print(t)
            except (StopAsyncIteration, RuntimeError) as e:
                print(f"[DEBUG] Stream completed: {str(e)}")
            except Exception as e:
                print(f"[DEBUG] Error in stream processing: {str(e)}")
                raise


__all__ = ["OpenAIVoiceReactAgent", "VoiceToTextReactAgent"]
