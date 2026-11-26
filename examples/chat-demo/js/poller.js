// Single-flight readiness poller after Stop or 503 worker_busy.
// Never retries POST automatically; stops on ready, 401, connection_lost, wrong_instance, or 30s timeout.

export function createReadinessPoller({
  probe,
  onReady,
  onTerminal,
  onTick,
  onTimeout,
  intervalMs = 1800,
  maxDurationMs = 30000,
  now = () => performance.now(),
  setTimer = (fn, ms) => setTimeout(fn, ms),
  clearTimer = id => clearTimeout(id),
}) {
  let timerId = null;
  let activeToken = 0;

  function stop() {
    activeToken++;
    if (timerId !== null) {
      clearTimer(timerId);
      timerId = null;
    }
  }

  function isActive() {
    return timerId !== null;
  }

  function start(initialDelayMs = intervalMs) {
    stop();
    const token = ++activeToken;
    const deadline = now() + maxDurationMs;

    const schedule = delayMs => {
      const remaining = deadline - now();
      if (remaining <= 0) {
        timerId = null;
        if (onTimeout) onTimeout();
        return;
      }
      const wait = Math.min(Math.max(250, delayMs), remaining);
      timerId = setTimer(async () => {
        if (token !== activeToken) return;
        timerId = null;
        try {
          const result = await probe();
          if (token !== activeToken) return;
          if (result.kind === "ready") {
            stop();
            onReady(result);
            return;
          }
          if (result.kind === "terminal") {
            stop();
            onTerminal(result);
            return;
          }
          const retryAfterMs = Number.isFinite(result.retryAfterSec) && result.retryAfterSec > 0
            ? result.retryAfterSec * 1000 : 0;
          const nextDelay = Math.max(intervalMs, retryAfterMs);
          if (onTick) onTick(Math.max(0, Math.ceil((deadline - now()) / 1000)));
          schedule(nextDelay);
        } catch (error) {
          if (token !== activeToken) return;
          stop();
          onTerminal({ kind: "terminal", error });
        }
      }, wait);
    };

    schedule(Math.max(intervalMs, initialDelayMs));
  }

  return { start, stop, isActive };
}
