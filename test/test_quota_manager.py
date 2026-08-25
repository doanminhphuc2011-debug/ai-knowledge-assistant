from quota_management.config import QuotaLimit, QuotaSettings
from quota_management.manager import QuotaExceededError, QuotaManager
from quota_management.store import InMemoryQuotaStore


def settings(*, requests=None, tokens=None):
    return QuotaSettings(
        enabled=True,
        default_subject="default",
        default_limit=QuotaLimit(
            max_requests=requests,
            max_tokens=tokens,
            window_seconds=3600,
        ),
    )


def test_request_quota_counts_actual_committed_calls():
    manager = QuotaManager(settings(requests=2), InMemoryQuotaStore())
    manager.reserve(workload="chat").commit(10)
    manager.reserve(workload="chat").commit(20)

    try:
        manager.reserve(workload="chat")
        assert False, "expected QuotaExceededError"
    except QuotaExceededError as exc:
        assert exc.resource == "requests"
        assert exc.used == 2


def test_cancelled_request_does_not_consume_quota():
    manager = QuotaManager(settings(requests=1), InMemoryQuotaStore())
    reservation = manager.reserve(workload="chat")
    reservation.cancel()
    manager.reserve(workload="chat").commit(5)
    assert manager.snapshot(workload="chat").requests == 1


def test_token_quota_uses_recorded_usage():
    manager = QuotaManager(settings(tokens=100), InMemoryQuotaStore())
    manager.reserve(workload="chat").commit(100)

    try:
        manager.reserve(workload="chat")
        assert False, "expected QuotaExceededError"
    except QuotaExceededError as exc:
        assert exc.resource == "tokens"
        assert exc.used == 100


def test_workloads_are_isolated():
    manager = QuotaManager(settings(requests=1), InMemoryQuotaStore())
    manager.reserve(workload="dmp-intent").commit(1)
    manager.reserve(workload="dmp-chat").commit(1)
    assert manager.snapshot(workload="dmp-intent").requests == 1
    assert manager.snapshot(workload="dmp-chat").requests == 1
