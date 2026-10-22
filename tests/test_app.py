"""Tests with a fake agent and fake tools: no model, no MCP server, no network."""

import json

from starlette.testclient import TestClient

from voice_mcp_sim.app import create_app, tool_trace


class FakeTool:
    def __init__(self, name):
        self.tool_name = name


class FakeAgent:
    """Behaves like a Strands Agent: callable, keeps .messages."""

    def __init__(self, tools):
        self.tools = tools
        self.messages = []

    def __call__(self, text):
        self.messages.append({"role": "user", "content": [{"text": text}]})
        self.messages.append({"role": "assistant", "content": [
            {"toolUse": {"toolUseId": "t1", "name": "diagnose_service", "input": {"service": "checkout"}}}]})
        result = {"speech": "checkout-api is degraded.", "details": {"status": "degraded"}}
        self.messages.append({"role": "user", "content": [
            {"toolResult": {"toolUseId": "t1", "status": "success", "content": [{"text": json.dumps(result)}]}}]})
        self.messages.append({"role": "assistant", "content": [{"text": "checkout-api is degraded."}]})
        return "checkout-api is degraded."


def client(**kw):
    loader = kw.pop("tools_loader", lambda: ([FakeTool("diagnose_service")], None))
    return TestClient(create_app(agent_factory=FakeAgent, tools_loader=loader))


def test_info_lists_tools():
    with client() as c:
        info = c.get("/api/info").json()
    assert info["tools"] == ["diagnose_service"]
    assert info["error"] is None


def test_index_page_served():
    with client() as c:
        page = c.get("/")
    assert page.status_code == 200
    assert "voice-mcp-sim" in page.text


def test_chat_returns_reply_and_tool_trace():
    with client() as c:
        data = c.post("/api/chat", json={"message": "Why is checkout slow?"}).json()
    assert data["reply"] == "checkout-api is degraded."
    assert data["session_id"]
    [call] = data["tools"]
    assert call["tool"] == "diagnose_service"
    assert call["input"] == {"service": "checkout"}
    assert call["result"]["details"]["status"] == "degraded"


def test_session_keeps_one_agent_and_trace_shows_only_new_calls():
    with client() as c:
        first = c.post("/api/chat", json={"message": "hi"}).json()
        second = c.post("/api/chat", json={"session_id": first["session_id"], "message": "again"}).json()
    assert second["session_id"] == first["session_id"]
    assert len(second["tools"]) == 1


def test_empty_message_rejected():
    with client() as c:
        assert c.post("/api/chat", json={"message": "  "}).status_code == 400


def test_mcp_down_is_explained():
    def broken():
        raise ConnectionError("refused")

    with client(tools_loader=broken) as c:
        assert "refused" in c.get("/api/info").json()["error"]
        assert c.post("/api/chat", json={"message": "hi"}).status_code == 503


def test_agent_failure_returns_502():
    class Boom(FakeAgent):
        def __call__(self, text):
            raise RuntimeError("model throttled")

    app = create_app(agent_factory=Boom, tools_loader=lambda: ([], None))
    with TestClient(app) as c:
        res = c.post("/api/chat", json={"message": "hi"})
    assert res.status_code == 502
    assert "throttled" in res.json()["error"]


def test_tool_trace_keeps_non_json_results_as_text():
    msgs = [
        {"content": [{"toolUse": {"toolUseId": "a", "name": "echo", "input": {}}}]},
        {"content": [{"toolResult": {"toolUseId": "a", "status": "error", "content": [{"text": "boom"}]}}]},
    ]
    [call] = tool_trace(msgs)
    assert call["result"] == "boom" and call["status"] == "error"


def test_transcribe_uses_injected_transcriber():
    seen = {}

    def fake_whisper(audio, content_type):
        seen["args"] = (audio, content_type)
        return "why is checkout slow"

    app = create_app(agent_factory=FakeAgent, tools_loader=lambda: ([], None), transcriber=fake_whisper)
    with TestClient(app) as c:
        assert c.get("/api/info").json()["whisper"] is True
        res = c.post("/api/transcribe", content=b"RIFF....", headers={"Content-Type": "audio/webm;codecs=opus"})
    assert res.json() == {"text": "why is checkout slow"}
    assert seen["args"] == (b"RIFF....", "audio/webm;codecs=opus")


def test_transcribe_without_key_is_explained(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with client() as c:
        assert c.get("/api/info").json()["whisper"] is False
        assert c.post("/api/transcribe", content=b"x").status_code == 501


def test_transcribe_rejects_empty_audio():
    app = create_app(agent_factory=FakeAgent, tools_loader=lambda: ([], None), transcriber=lambda a, t: "x")
    with TestClient(app) as c:
        assert c.post("/api/transcribe", content=b"").status_code == 400


def test_speak_returns_mp3_from_injected_speaker():
    calls = []

    def fake_polly(text, voice):
        calls.append((text, voice))
        return b"ID3fake-mp3"

    app = create_app(agent_factory=FakeAgent, tools_loader=lambda: ([], None), speaker=fake_polly)
    with TestClient(app) as c:
        assert "Ruth" in c.get("/api/info").json()["voices"]
        res = c.post("/api/speak", json={"text": "Hello", "voice": "Matthew"})
        assert c.post("/api/speak", json={"text": "Hello", "voice": "Nobody"}).status_code == 400
        assert c.post("/api/speak", json={"text": " ", "voice": "Ruth"}).status_code == 400
    assert res.headers["content-type"] == "audio/mpeg" and res.content == b"ID3fake-mp3"
    assert calls == [("Hello", "Matthew")]


def test_speak_without_aws_is_explained(monkeypatch):
    monkeypatch.setenv("POLLY", "off")
    with client() as c:
        assert c.get("/api/info").json()["voices"] == []
        assert c.post("/api/speak", json={"text": "hi"}).status_code == 501


def test_polly_falls_back_to_neural(monkeypatch):
    import boto3
    from botocore.exceptions import ClientError
    from voice_mcp_sim import app as app_module

    engines = []

    class FakeStream:
        def read(self):
            return b"mp3"

    class FakePolly:
        def synthesize_speech(self, **kw):
            engines.append(kw["Engine"])
            if kw["Engine"] == "generative":
                raise ClientError({"Error": {"Code": "EngineNotSupportedException", "Message": "no"}}, "SynthesizeSpeech")
            return {"AudioStream": FakeStream()}

    monkeypatch.setattr(boto3, "client", lambda *a, **k: FakePolly())
    assert app_module.polly_speak("hi", "Ruth") == b"mp3"
    assert engines == ["generative", "neural"]
