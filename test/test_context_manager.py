from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from context_management.config import ContextConfig
from context_management.manager import ContextBlock, ContextManager
from context_management.store import InMemoryConversationStore
from context_management.token_counter import HeuristicTokenCounter


def _manager() -> ContextManager:
    config = ContextConfig(
        max_input_tokens=220,
        reserved_output_tokens=40,
        max_history_tokens=80,
        max_external_context_tokens=80,
        default_session_id="test",
        store_backend="memory",
        session_ttl_seconds=0,
        redis_key_prefix="test:context",
        token_encoding="cl100k_base",
        chars_per_token=4.0,
        message_overhead_tokens=2,
    )
    return ContextManager(
        system_prompt="system",
        config=config,
        store=InMemoryConversationStore(),
        token_counter=HeuristicTokenCounter(4.0, 2),
    )


def test_sessions_are_isolated() -> None:
    manager = _manager()
    manager.record_turn(user_input="u1", assistant_output="a1", session_id="s1")
    manager.record_turn(user_input="u2", assistant_output="a2", session_id="s2")

    assert [m.content for m in manager.get_history("s1")] == ["u1", "a1"]
    assert [m.content for m in manager.get_history("s2")] == ["u2", "a2"]


def test_external_context_is_not_persisted() -> None:
    manager = _manager()
    assembly = manager.prepare(
        user_input="question",
        session_id="s1",
        context_blocks=[ContextBlock(name="kb", content="secret reference")],
    )
    assert "secret reference" in assembly.messages[-1].content

    manager.record_turn(user_input="question", assistant_output="answer", session_id="s1")
    persisted = manager.get_history("s1")
    assert all("secret reference" not in str(message.content) for message in persisted)


def test_recent_history_is_trimmed_by_budget() -> None:
    manager = _manager()
    for idx in range(20):
        manager.record_turn(
            user_input=f"user-{idx}-" + "x" * 24,
            assistant_output=f"assistant-{idx}-" + "y" * 24,
            session_id="s1",
        )

    assembly = manager.prepare(user_input="new question", session_id="s1")

    assert assembly.history_messages_dropped > 0
    assert assembly.estimated_input_tokens <= 180
    assert isinstance(assembly.messages[0].content, str)
    assert isinstance(assembly.messages[-1], HumanMessage)
