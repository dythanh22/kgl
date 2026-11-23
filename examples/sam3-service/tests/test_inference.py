"""Kiểm hàng đợi/luồng thực thi. Không cần Torch hay GPU."""

import asyncio
import threading
import time

import pytest

from sam3_service.inference import (
    EngineUnavailable,
    InferenceEngine,
    InferenceFailed,
    QueueFull,
    RequestTimeout,
)
from sam3_service.schemas import ApiError


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def engine():
    # 8 chỗ: các test dùng chung fixture này không nhằm kiểm hàng đợi đầy.
    eng = InferenceEngine(max_queue=8)
    eng.start()
    yield eng
    eng.stop(timeout=5)


def test_runs_and_reports_timing(engine):
    async def go():
        value, t = await engine.run(lambda: 42, timeout_s=5)
        assert value == 42 and t.queue_ms >= 0 and t.run_ms >= 0

    run(go())


def test_jobs_never_overlap_and_run_in_one_thread(engine):
    active = 0
    max_active = 0
    threads = set()
    lock = threading.Lock()

    def job():
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
            threads.add(threading.get_ident())
        time.sleep(0.05)
        with lock:
            active -= 1

    async def go():
        await asyncio.gather(*[engine.run(job, timeout_s=5) for _ in range(3)])

    run(go())
    assert max_active == 1
    assert len(threads) == 1


def test_queue_full_is_rejected_clearly():
    eng = InferenceEngine(max_queue=1)
    eng.start()
    gate = threading.Event()

    async def go():
        running = asyncio.create_task(eng.run(gate.wait, timeout_s=10))
        await asyncio.sleep(0.1)  # job 1 đang chạy, hàng đợi trống
        queued = asyncio.create_task(eng.run(lambda: "q", timeout_s=10))
        await asyncio.sleep(0.1)  # job 2 chiếm chỗ duy nhất
        with pytest.raises(QueueFull) as ei:
            await eng.run(lambda: "x", timeout_s=10)
        assert ei.value.status_code == 503 and "Retry-After" in ei.value.headers
        gate.set()
        await running
        assert (await queued)[0] == "q"

    try:
        run(go())
    finally:
        gate.set()
        eng.stop(timeout=5)


def test_timeout_while_running_keeps_gpu_slot_until_job_finishes():
    """Hết hạn HTTP không dừng tính toán: job sau không được chen vào."""
    eng = InferenceEngine(max_queue=2)
    eng.start()
    events = []

    def slow():
        events.append("slow-start")
        time.sleep(0.4)
        events.append("slow-end")

    def fast():
        events.append("fast-start")

    async def go():
        with pytest.raises(RequestTimeout) as ei:
            await eng.run(slow, timeout_s=0.1)
        assert "suy luận" in ei.value.message
        # Gửi job kế tiếp ngay khi job chậm vẫn đang chạy.
        await eng.run(fast, timeout_s=5)

    try:
        run(go())
    finally:
        eng.stop(timeout=5)
    assert events == ["slow-start", "slow-end", "fast-start"]


def test_job_cancelled_in_queue_is_skipped():
    eng = InferenceEngine(max_queue=4)
    eng.start()
    gate = threading.Event()
    ran = []

    async def go():
        blocker = asyncio.create_task(eng.run(gate.wait, timeout_s=10))
        await asyncio.sleep(0.1)
        victim = asyncio.create_task(eng.run(lambda: ran.append("victim"), timeout_s=10))
        await asyncio.sleep(0.1)
        victim.cancel()  # client ngắt kết nối khi còn chờ
        with pytest.raises(asyncio.CancelledError):
            await victim
        gate.set()
        await blocker
        await eng.run(lambda: ran.append("after"), timeout_s=5)

    try:
        run(go())
    finally:
        gate.set()
        eng.stop(timeout=5)
    assert ran == ["after"]


def test_job_expired_in_queue_does_not_run():
    eng = InferenceEngine(max_queue=4)
    eng.start()
    gate = threading.Event()
    ran = []

    async def go():
        blocker = asyncio.create_task(eng.run(gate.wait, timeout_s=10))
        await asyncio.sleep(0.1)
        with pytest.raises(RequestTimeout) as ei:
            await eng.run(lambda: ran.append("late"), timeout_s=0.1)
        assert "hàng đợi" in ei.value.message
        gate.set()
        await blocker
        await eng.run(lambda: None, timeout_s=5)

    try:
        run(go())
    finally:
        gate.set()
        eng.stop(timeout=5)
    assert ran == []


def test_unexpected_exception_does_not_kill_worker(engine):
    def boom():
        raise ValueError("secret internal detail")

    async def go():
        with pytest.raises(InferenceFailed) as ei:
            await engine.run(boom, timeout_s=5)
        assert "secret" not in ei.value.message  # không lộ chi tiết nội bộ
        assert (await engine.run(lambda: "ok", timeout_s=5))[0] == "ok"

    run(go())


def test_api_error_passes_through(engine):
    def oom():
        raise ApiError(503, "gpu_out_of_memory", "oom")

    async def go():
        with pytest.raises(ApiError) as ei:
            await engine.run(oom, timeout_s=5)
        assert ei.value.code == "gpu_out_of_memory"
        assert engine.fatal is None
        assert (await engine.run(lambda: 1, timeout_s=5))[0] == 1

    run(go())


def test_fatal_error_marks_engine_and_calls_hook():
    called = []
    eng = InferenceEngine(max_queue=2, on_fatal=lambda e: called.append(e.code))
    eng.start()

    def fatal():
        raise ApiError(500, "gpu_fatal_error", "cuda dead", fatal=True)

    async def go():
        with pytest.raises(ApiError):
            await eng.run(fatal, timeout_s=5)
        assert eng.fatal is not None
        with pytest.raises(EngineUnavailable):
            await eng.run(lambda: 1, timeout_s=5)

    try:
        run(go())
    finally:
        eng.stop(timeout=5)
    assert called == ["gpu_fatal_error"]


def test_stop_rejects_pending_and_new_jobs():
    eng = InferenceEngine(max_queue=4)
    eng.start()
    gate = threading.Event()

    async def go():
        running = asyncio.create_task(eng.run(gate.wait, timeout_s=10))
        await asyncio.sleep(0.1)
        pending = asyncio.create_task(eng.run(lambda: "p", timeout_s=10))
        await asyncio.sleep(0.1)
        stopper = threading.Thread(target=lambda: eng.stop(timeout=5))
        stopper.start()
        await asyncio.sleep(0.1)
        with pytest.raises(EngineUnavailable):
            await pending
        with pytest.raises(EngineUnavailable):
            await eng.run(lambda: 1, timeout_s=1)
        gate.set()
        await running
        stopper.join()

    try:
        run(go())
    finally:
        gate.set()
