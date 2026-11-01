"""Tests for every MCP tool exposed by main.py.

No WhatsApp account, bridge, ffmpeg or whisper.cpp is needed: read tools run
against a temporary SQLite database with the bridge's schema, and everything
that would reach the network or the transcriber is replaced with fakes.
"""

import asyncio
import sqlite3

import pytest

import main
import whatsapp

EXPECTED_TOOLS = {
    "search_contacts", "list_messages", "list_chats", "get_chat",
    "get_direct_chat_by_contact", "get_contact_chats", "get_last_interaction",
    "get_message_context", "send_message", "send_file", "send_audio_message",
    "transcribe_audio", "download_media",
}
READ_TOOLS = {
    "search_contacts", "list_messages", "list_chats", "get_chat",
    "get_direct_chat_by_contact", "get_contact_chats", "get_last_interaction",
    "get_message_context",
}
SEND_TOOLS = {"send_message", "send_file", "send_audio_message"}
DOWNLOAD_TOOLS = {"transcribe_audio", "download_media"}

ALICE = "5511999990000@s.whatsapp.net"
BOB = "5511888880000@s.whatsapp.net"
GROUP = "120363000000000000@g.us"


def _tools():
    return {t.name: t for t in asyncio.run(main.mcp.list_tools())}


# --- Annotations -----------------------------------------------------------

def test_all_tools_are_registered():
    assert set(_tools()) == EXPECTED_TOOLS


@pytest.mark.parametrize("name", sorted(EXPECTED_TOOLS))
def test_every_tool_declares_all_four_hints(name):
    a = _tools()[name].annotations
    assert a is not None, f"{name} has no annotations"
    for hint in ("readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint"):
        assert isinstance(getattr(a, hint), bool), f"{name} missing {hint}"


def test_hints_match_behaviour():
    tools = _tools()
    for name in READ_TOOLS:
        a = tools[name].annotations
        assert a.readOnlyHint and not a.destructiveHint and a.idempotentHint
        assert not a.openWorldHint, f"{name} only reads the local database"
    for name in SEND_TOOLS:
        a = tools[name].annotations
        assert not a.readOnlyHint and not a.idempotentHint and a.openWorldHint
    for name in DOWNLOAD_TOOLS:
        a = tools[name].annotations
        assert not a.readOnlyHint and a.idempotentHint and a.openWorldHint
    assert not any(t.annotations.destructiveHint for t in tools.values())


# --- Read tools against a temporary database -------------------------------

@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "messages.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE chats (jid TEXT PRIMARY KEY, name TEXT, last_message_time TIMESTAMP);
        CREATE TABLE messages (
            id TEXT, chat_jid TEXT, sender TEXT, content TEXT, timestamp TIMESTAMP,
            is_from_me BOOLEAN, media_type TEXT, filename TEXT, url TEXT,
            media_key BLOB, file_sha256 BLOB, file_enc_sha256 BLOB, file_length INTEGER,
            PRIMARY KEY (id, chat_jid)
        );
        """
    )
    conn.executemany(
        "INSERT INTO chats VALUES (?, ?, ?)",
        [
            (ALICE, "Alice", "2026-10-30 10:02:00"),
            (BOB, "Bob", "2026-10-29 09:00:00"),
            (GROUP, "Family", "2026-10-28 08:00:00"),
        ],
    )
    conn.executemany(
        "INSERT INTO messages (id, chat_jid, sender, content, timestamp, is_from_me, media_type)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("m1", ALICE, "5511999990000", "oi, tudo bem?", "2026-10-30 10:00:00", 0, None),
            ("m2", ALICE, "me", "tudo sim", "2026-10-30 10:01:00", 1, None),
            ("m3", ALICE, "5511999990000", "", "2026-10-30 10:02:00", 0, "audio"),
            ("m4", BOB, "5511888880000", "reunião amanhã", "2026-10-29 09:00:00", 0, None),
            ("m5", GROUP, "5511999990000", "bom dia família", "2026-10-28 08:00:00", 0, None),
        ],
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(whatsapp, "MESSAGES_DB_PATH", str(path))
    return path


def test_search_contacts(db):
    result = main.search_contacts("Alice")
    assert [c.jid for c in result] == [ALICE]


def test_search_contacts_excludes_groups(db):
    assert main.search_contacts("Family") == []


def test_list_messages_filters_by_chat_and_query(db):
    out = main.list_messages(chat_jid=BOB, include_context=False)
    assert "reunião amanhã" in out and "oi, tudo bem?" not in out

    out = main.list_messages(query="bom dia", include_context=False)
    assert "bom dia família" in out and "reunião" not in out


def test_list_messages_date_window(db):
    out = main.list_messages(after="2026-10-29T12:00:00", include_context=False)
    assert "tudo sim" in out and "reunião amanhã" not in out


def test_list_chats_sorted_by_activity(db):
    chats = main.list_chats()
    assert [c.jid for c in chats] == [ALICE, BOB, GROUP]
    assert chats[2].is_group


def test_list_chats_query(db):
    assert [c.jid for c in main.list_chats(query="Bob")] == [BOB]


def test_get_chat(db):
    chat = main.get_chat(ALICE)
    assert chat.name == "Alice"
    assert main.get_chat("unknown@s.whatsapp.net") is None


def test_get_direct_chat_by_contact(db):
    assert main.get_direct_chat_by_contact("5511888880000").jid == BOB


def test_get_contact_chats(db):
    # The bridge stores the sender as the bare phone number (JID user part).
    jids = {c.jid for c in main.get_contact_chats("5511999990000")}
    assert jids == {ALICE, GROUP}


def test_get_last_interaction(db):
    out = main.get_last_interaction("5511888880000")
    assert "reunião amanhã" in out


def test_get_message_context(db):
    ctx = main.get_message_context("m2", before=1, after=1)
    assert ctx.message.id == "m2"
    assert [m.id for m in ctx.before] == ["m1"]
    assert [m.id for m in ctx.after] == ["m3"]


# --- Send tools: the bridge HTTP call is faked ------------------------------

class _FakeResponse:
    def __init__(self, status_code=200, body=None):
        self.status_code = status_code
        self._body = body or {"success": True, "message": "sent"}
        self.text = str(self._body)

    def json(self):
        return self._body


@pytest.fixture
def bridge(monkeypatch):
    calls = []

    def fake_post(url, json=None, **kwargs):
        calls.append((url, json))
        return _FakeResponse()

    monkeypatch.setattr(whatsapp.requests, "post", fake_post)
    return calls


def test_send_message(bridge):
    result = main.send_message(ALICE, "olá")
    assert result["success"] is True
    url, payload = bridge[0]
    assert url.endswith("/send")
    assert payload == {"recipient": ALICE, "message": "olá"}


def test_send_message_requires_recipient(bridge):
    result = main.send_message("", "olá")
    assert result["success"] is False
    assert bridge == []


def test_send_file(bridge, tmp_path):
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"%PDF-1.4")
    result = main.send_file(ALICE, str(f))
    assert result["success"] is True
    assert bridge[0][1]["media_path"] == str(f)


def test_send_file_missing_path(bridge, tmp_path):
    result = main.send_file(ALICE, str(tmp_path / "nope.pdf"))
    assert result["success"] is False
    assert bridge == []


def test_send_audio_message_ogg_skips_conversion(bridge, tmp_path):
    f = tmp_path / "voice.ogg"
    f.write_bytes(b"OggS")
    result = main.send_audio_message(ALICE, str(f))
    assert result["success"] is True
    assert bridge[0][1]["media_path"] == str(f)


# --- Download and transcription --------------------------------------------

def test_download_media(monkeypatch):
    monkeypatch.setattr(main, "whatsapp_download_media", lambda mid, jid: "/tmp/a.ogg")
    assert main.download_media("m3", ALICE) == {
        "success": True,
        "message": "Media downloaded successfully",
        "file_path": "/tmp/a.ogg",
    }


def test_download_media_failure(monkeypatch):
    monkeypatch.setattr(main, "whatsapp_download_media", lambda mid, jid: None)
    assert main.download_media("m3", ALICE)["success"] is False


def test_transcribe_audio(monkeypatch):
    monkeypatch.setattr(main, "whatsapp_download_media", lambda mid, jid: "/tmp/a.ogg")
    seen = {}

    def fake_transcribe(path, language):
        seen.update(path=path, language=language)
        return "oi, tudo bem?"

    monkeypatch.setattr(main, "transcribe_file", fake_transcribe)
    result = main.transcribe_audio("m3", ALICE, language="pt")
    assert result == {"success": True, "transcript": "oi, tudo bem?", "audio_path": "/tmp/a.ogg"}
    assert seen == {"path": "/tmp/a.ogg", "language": "pt"}


def test_transcribe_audio_download_fails(monkeypatch):
    monkeypatch.setattr(main, "whatsapp_download_media", lambda mid, jid: None)
    assert main.transcribe_audio("m3", ALICE)["success"] is False


def test_transcribe_audio_transcriber_error(monkeypatch):
    monkeypatch.setattr(main, "whatsapp_download_media", lambda mid, jid: "/tmp/a.ogg")

    def boom(path, language):
        raise RuntimeError("whisper-cli não encontrado")

    monkeypatch.setattr(main, "transcribe_file", boom)
    result = main.transcribe_audio("m3", ALICE)
    assert result["success"] is False and "whisper-cli" in result["message"]
