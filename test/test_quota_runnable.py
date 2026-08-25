from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from quota_management.config import QuotaLimit, QuotaSettings
from quota_management.manager import QuotaManager
from quota_management.runnable import QuotaRunnable
from quota_management.store import InMemoryQuotaStore


def test_runnable_records_real_usage_metadata():
    def fake(_):
        msg = AIMessage(content="ok")
        msg.usage_metadata = {
            "input_tokens": 7,
            "output_tokens": 3,
            "total_tokens": 10,
        }
        return msg

    settings = QuotaSettings(
        enabled=True,
        default_subject="default",
        default_limit=QuotaLimit(10, 100, 3600),
    )
    manager = QuotaManager(settings, InMemoryQuotaStore())
    wrapped = QuotaRunnable(
        RunnableLambda(fake),
        manager=manager,
        workload="dmp-chat",
    )

    assert wrapped.invoke("hello").content == "ok"
    snapshot = manager.snapshot(workload="dmp-chat")
    assert snapshot.requests == 1
    assert snapshot.tokens == 10
