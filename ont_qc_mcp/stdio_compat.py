from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import anyio
import anyio.lowlevel
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

import mcp.types as types
from mcp.shared.message import SessionMessage


@asynccontextmanager
async def stdio_server_compat(
    *,
    encoding: str = "utf-8",
    errors: str = "strict",
) -> AsyncIterator[
    tuple[
    MemoryObjectReceiveStream[SessionMessage | Exception],
    MemoryObjectSendStream[SessionMessage],
    ]
]:
    """
    Stdio transport that avoids anyio's AsyncFile wrappers.

    Some environments exhibit hangs when using anyio.wrap_file() for stdin/stdout.
    This compat transport uses an asyncio pipe reader for stdin and writes newline-delimited
    JSON to stdout (flushing per message).
    """

    read_stream_writer: MemoryObjectSendStream[SessionMessage | Exception]
    read_stream: MemoryObjectReceiveStream[SessionMessage | Exception]
    write_stream: MemoryObjectSendStream[SessionMessage]
    write_stream_reader: MemoryObjectReceiveStream[SessionMessage]

    read_stream_writer, read_stream = anyio.create_memory_object_stream(0)
    write_stream, write_stream_reader = anyio.create_memory_object_stream(0)

    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    reader_protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: reader_protocol, sys.stdin)

    async def stdin_reader() -> None:
        try:
            async with read_stream_writer:
                while True:
                    line_bytes = await reader.readline()
                    if not line_bytes:
                        break
                    try:
                        line = line_bytes.decode(encoding, errors=errors)
                        message = types.JSONRPCMessage.model_validate_json(line)
                    except Exception as exc:  # pragma: no cover
                        await read_stream_writer.send(exc)
                        continue

                    await read_stream_writer.send(SessionMessage(message))
        except anyio.ClosedResourceError:  # pragma: no cover
            await anyio.lowlevel.checkpoint()

    async def stdout_writer() -> None:
        try:
            async with write_stream_reader:
                async for session_message in write_stream_reader:
                    json_text = session_message.message.model_dump_json(by_alias=True, exclude_none=True)
                    try:
                        sys.stdout.write(json_text + "\n")
                        sys.stdout.flush()
                    except BrokenPipeError:  # pragma: no cover
                        return
        except anyio.ClosedResourceError:  # pragma: no cover
            await anyio.lowlevel.checkpoint()

    async with anyio.create_task_group() as tg:
        tg.start_soon(stdin_reader)
        tg.start_soon(stdout_writer)
        yield read_stream, write_stream

__all__ = ["stdio_server_compat"]
