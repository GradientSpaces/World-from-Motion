from __future__ import annotations

import atexit
import subprocess
import threading
import time
from pathlib import Path

from .util import utc_now, write_json_atomic


class RuntimeMonitor:
    def __init__(self, output: Path, poll_seconds: float = 1.0) -> None:
        self.output = output
        self.poll_seconds = poll_seconds
        self.started_at = utc_now()
        self.start_time = time.monotonic()
        self.peak_gpu_memory_mib = 0
        self.samples = 0
        self._stop = threading.Event()
        self._finished = False
        self._thread = threading.Thread(target=self._poll, name="wfm-gpu-monitor", daemon=True)

    def start(self) -> None:
        self._thread.start()
        atexit.register(self.finish)

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                )
                values = [int(line.strip()) for line in output.splitlines() if line.strip()]
                if values:
                    self.peak_gpu_memory_mib = max(self.peak_gpu_memory_mib, max(values))
                    self.samples += 1
            except (FileNotFoundError, subprocess.SubprocessError, ValueError):
                pass
            self._stop.wait(self.poll_seconds)

    def _output_bytes(self) -> int:
        total = 0
        for path in self.output.rglob("*"):
            try:
                if path.is_file():
                    total += path.stat().st_size
            except OSError:
                pass
        return total

    def _fallback_events(self) -> list[dict[str, str]]:
        events = []
        logs = self.output / "logs"
        for path in sorted(logs.glob("*.log")):
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for line in lines:
                if "fallback" in line.lower() or "falling back" in line.lower():
                    events.append({"log": path.name, "message": line[-1000:]})
        return events

    def finish(self) -> dict[str, object]:
        if self._finished:
            path = self.output / "diagnostics" / "runtime_metrics.json"
            if path.is_file():
                import json

                return json.loads(path.read_text(encoding="utf-8"))
            return {}
        self._finished = True
        self._stop.set()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=max(2.0, self.poll_seconds + 1.0))
        payload: dict[str, object] = {
            "started_at": self.started_at,
            "finished_at": utc_now(),
            "wall_seconds": time.monotonic() - self.start_time,
            "peak_gpu_memory_mib": self.peak_gpu_memory_mib or None,
            "gpu_samples": self.samples,
            "output_disk_bytes": self._output_bytes(),
            "fallback_events": self._fallback_events(),
        }
        write_json_atomic(self.output / "diagnostics" / "runtime_metrics.json", payload)
        return payload
