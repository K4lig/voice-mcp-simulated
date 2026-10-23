"""Web app: a voice assistant in the browser that uses the tools of any MCP server.

    MCP_URL=http://127.0.0.1:8000/mcp MODEL_PROVIDER=groq python -m voice_mcp_sim.app
    open http://127.0.0.1:8080
"""

import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Route

from . import __version__
from .models import build_model, provider

STATIC = Path(__file__).parent / "static"

DEFAULT_PROMPT = """You are {name}, a voice assistant. Your reply is spoken aloud, so:
- Always call a tool to answer questions about live state; never answer from memory or from
  earlier turns, because the data changes. If a tool result has a 'speech' field, say exactly that text,
  once, even if it has several sentences.
- Otherwise reply with one or two short sentences. No lists, no markdown, no notes in parentheses.
- Do not offer extra help and do not add follow-up suggestions.
If a tool asks for confirmation before acting, read its preview to the user and stop.
Only call it again to confirm if the user's latest message clearly says yes. Never invent
confirmation codes."""


def system_prompt() -> str:
    path = os.getenv("SYSTEM_PROMPT_FILE")
    if path:
        return Path(path).read_text()
    return DEFAULT_PROMPT.format(name=os.getenv("ASSISTANT_NAME", "Assistant"))


# ---------------------------------------------------------------- reply cleanup
def clean_reply(text: str) -> str:
    """Some model/provider combinations return the answer twice ("X. X." or "X.X."). Say it once."""
    text = text.strip()
    half = len(text) // 2
    for cut in (half, half + 1, half - 1):
        a, b = text[:cut].strip(), text[cut:].strip()
        if a and a == b:
            return a
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) > 1 and len(set(lines)) == 1:
        return lines[0]
    return text


# ---------------------------------------------------------------- tool trace
def _text_of(blocks) -> str:
    return " ".join(b.get("text", "") for b in blocks if isinstance(b, dict) and "text" in b)


def tool_trace(messages: list) -> list[dict]:
    """Pull the tool calls (name, input, result) out of the agent's new messages."""
    calls, by_id = [], {}
    for message in messages:
        for block in message.get("content", []):
            if "toolUse" in block:
                use = block["toolUse"]
                call = {"tool": use.get("name"), "input": use.get("input", {}), "status": "pending", "result": None}
                by_id[use.get("toolUseId")] = call
                calls.append(call)
            elif "toolResult" in block:
                res = block["toolResult"]
                call = by_id.get(res.get("toolUseId"))
                if call is None:
                    continue
                raw = _text_of(res.get("content", []))
                try:
                    call["result"] = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    call["result"] = raw
                call["status"] = res.get("status", "success")
    return calls


# ---------------------------------------------------------------- speech-to-text fallback
AUDIO_EXT = {"audio/webm": "webm", "audio/ogg": "ogg", "audio/mp4": "mp4", "audio/mpeg": "mp3", "audio/wav": "wav"}
MAX_AUDIO = 10 * 1024 * 1024


def whisper_available() -> bool:
    return bool(os.getenv("GROQ_API_KEY"))


def groq_whisper(audio: bytes, content_type: str) -> str:
    """Transcribe with Whisper on Groq (OpenAI-compatible API). Used when the browser's
    own speech recognition is unavailable, e.g. Brave, Firefox, or a network that blocks Google."""
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1")
    ext = AUDIO_EXT.get(content_type.split(";")[0].strip(), "webm")
    result = client.audio.transcriptions.create(
        file=(f"speech.{ext}", audio),
        model=os.getenv("WHISPER_MODEL", "whisper-large-v3-turbo"),
        language=os.getenv("WHISPER_LANGUAGE", "en"),
    )
    return result.text.strip()


# ---------------------------------------------------------------- text-to-speech (Amazon Polly)
# Natural voices from AWS. "generative" is the most human-sounding engine; if a voice or region
# does not offer it, we fall back to "neural".
POLLY_VOICES = {
    "Ruth": ["generative", "neural"],
    "Matthew": ["generative", "neural"],
    "Danielle": ["generative", "neural"],
    "Joanna": ["generative", "neural"],
    "Stephen": ["generative", "neural"],
    "Gregory": ["neural"],
}
MAX_SPEECH_CHARS = 1500


def polly_available() -> bool:
    if os.getenv("POLLY", "on").lower() in {"off", "0", "false"}:
        return False
    try:
        import boto3

        return boto3.Session().get_credentials() is not None
    except Exception:
        return False


def polly_speak(text: str, voice: str) -> bytes:
    import boto3
    from botocore.exceptions import ClientError

    polly = boto3.client("polly", region_name=os.getenv("AWS_REGION", "us-east-1"))
    last_error = None
    for engine in POLLY_VOICES[voice]:
        try:
            out = polly.synthesize_speech(Text=text, VoiceId=voice, Engine=engine, OutputFormat="mp3")
            return out["AudioStream"].read()
        except ClientError as exc:          # engine not offered for this voice/region: try the next one
            last_error = exc
    raise last_error


# ---------------------------------------------------------------- the app
def create_app(agent_factory=None, tools_loader=None, transcriber=None, speaker=None) -> Starlette:
    """agent_factory(tools) -> callable agent; tools_loader() -> (tools, close);
    transcriber(audio_bytes, content_type) -> text; speaker(text, voice) -> mp3 bytes.
    All injectable for tests."""
    state: dict = {"sessions": {}, "tools": [], "close": None, "error": None}

    def default_tools_loader():
        from strands.tools.mcp import MCPClient

        headers = {"Authorization": f"Bearer {os.environ['MCP_TOKEN']}"} if os.getenv("MCP_TOKEN") else None
        client = MCPClient(url=os.getenv("MCP_URL", "http://127.0.0.1:8000/mcp"), headers=headers)
        client.start()
        return client.list_tools_sync(), client.stop

    def default_agent_factory(tools):
        from strands import Agent

        return Agent(model=build_model(), tools=tools, system_prompt=system_prompt(), callback_handler=None)

    load = tools_loader or default_tools_loader
    transcribe_audio = transcriber or groq_whisper
    synthesize = speaker or polly_speak
    polly_on = speaker is not None or polly_available()
    make_agent = agent_factory or default_agent_factory

    @asynccontextmanager
    async def lifespan(app):
        try:
            state["tools"], state["close"] = await run_in_threadpool(load)
        except Exception as exc:  # the page still opens and explains what is wrong
            state["error"] = f"Could not connect to the MCP server: {exc}"
        yield
        if state["close"]:
            try:
                state["close"](None, None, None)
            except TypeError:
                state["close"]()

    async def index(_: Request):
        return FileResponse(STATIC / "index.html")

    async def info(_: Request):
        return JSONResponse({
            "version": __version__,
            "mcp_url": os.getenv("MCP_URL", "http://127.0.0.1:8000/mcp"),
            "provider": provider(),
            "assistant": os.getenv("ASSISTANT_NAME", "Assistant"),
            "tools": [getattr(t, "tool_name", str(t)) for t in state["tools"]],
            "error": state["error"],
            "whisper": transcriber is not None or whisper_available(),
            "voices": list(POLLY_VOICES) if polly_on else [],
        })

    async def chat(request: Request):
        if state["error"]:
            return JSONResponse({"error": state["error"]}, status_code=503)
        body = await request.json()
        message = (body.get("message") or "").strip()
        if not message:
            return JSONResponse({"error": "Empty message."}, status_code=400)
        session_id = body.get("session_id") or uuid.uuid4().hex
        agent = state["sessions"].get(session_id)
        if agent is None:
            agent = state["sessions"][session_id] = make_agent(state["tools"])

        before = len(getattr(agent, "messages", []))
        started = time.time()
        try:
            reply = await run_in_threadpool(agent, message)
        except Exception as exc:
            return JSONResponse({"session_id": session_id, "error": f"{type(exc).__name__}: {exc}"}, status_code=502)
        new_messages = getattr(agent, "messages", [])[before:]
        if os.getenv("VMS_DEBUG"):
            print("---- new messages ----\n" + json.dumps(new_messages, indent=1, default=str)[:6000], flush=True)
        return JSONResponse({
            "session_id": session_id,
            "reply": clean_reply(str(reply)),
            "tools": tool_trace(new_messages),
            "seconds": round(time.time() - started, 2),
        })

    async def transcribe(request: Request):
        if transcriber is None and not whisper_available():
            return JSONResponse({"error": "Server-side transcription needs GROQ_API_KEY."}, status_code=501)
        audio = await request.body()
        if not audio:
            return JSONResponse({"error": "No audio received."}, status_code=400)
        if len(audio) > MAX_AUDIO:
            return JSONResponse({"error": "Recording too long."}, status_code=413)
        try:
            text = await run_in_threadpool(transcribe_audio, audio, request.headers.get("content-type", "audio/webm"))
        except Exception as exc:
            return JSONResponse({"error": f"Transcription failed: {type(exc).__name__}: {exc}"}, status_code=502)
        return JSONResponse({"text": text})

    async def speak(request: Request):
        if not polly_on:
            return JSONResponse({"error": "Amazon Polly needs AWS credentials on the server."}, status_code=501)
        body = await request.json()
        text = (body.get("text") or "").strip()[:MAX_SPEECH_CHARS]
        voice = body.get("voice") or "Ruth"
        if not text:
            return JSONResponse({"error": "Nothing to say."}, status_code=400)
        if voice not in POLLY_VOICES:
            return JSONResponse({"error": f"Unknown voice {voice}."}, status_code=400)
        try:
            audio = await run_in_threadpool(synthesize, text, voice)
        except Exception as exc:
            return JSONResponse({"error": f"Polly failed: {type(exc).__name__}: {exc}"}, status_code=502)
        return Response(audio, media_type="audio/mpeg")

    async def reset(request: Request):
        body = await request.json()
        state["sessions"].pop(body.get("session_id"), None)
        return JSONResponse({"ok": True})

    return Starlette(
        routes=[
            Route("/", index),
            Route("/api/info", info),
            Route("/api/chat", chat, methods=["POST"]),
            Route("/api/reset", reset, methods=["POST"]),
            Route("/api/transcribe", transcribe, methods=["POST"]),
            Route("/api/speak", speak, methods=["POST"]),
        ],
        lifespan=lifespan,
    )


def main() -> None:
    host, port = os.getenv("HOST", "127.0.0.1"), int(os.getenv("PORT", "8080"))
    print(f"voice-mcp-sim on http://{host}:{port}  (MCP: {os.getenv('MCP_URL', 'http://127.0.0.1:8000/mcp')}, model: {provider()})")
    uvicorn.run(create_app(), host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
