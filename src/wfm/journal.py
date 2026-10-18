from __future__ import annotations

import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from .errors import ConfigurationError
from .util import utc_now, write_json_atomic


class StageJournal:
    def __init__(
        self,
        path: Path,
        fingerprint: str,
        *,
        resume: bool = True,
        provenance: dict[str, Any] | None = None,
    ) -> None:
        self.path = path
        self.fingerprint = fingerprint
        self.payload: dict[str, Any]
        if path.exists():
            import json

            self.payload = json.loads(path.read_text(encoding="utf-8"))
            previous = self.payload.get("run_fingerprint")
            if previous != fingerprint:
                raise ConfigurationError(
                    f"Output belongs to a different input/configuration ({previous}); "
                    "choose a new output directory"
                )
            if not resume and self.payload.get("stages"):
                raise ConfigurationError("Output already contains a run; resume is disabled")
        else:
            self.payload = {
                "schema_version": 1,
                "run_fingerprint": fingerprint,
                "created_at": utc_now(),
                "updated_at": utc_now(),
                "state": "created",
                "provenance": provenance or {},
                "stages": {},
            }
            self._save()

    def _invalidate_from(self, name: str, reason: str) -> None:
        stages = self.payload.setdefault("stages", {})
        names = list(stages)
        start = names.index(name) if name in names else len(names)
        for stage_name in names[start:]:
            record = stages[stage_name]
            if record.get("state") == "completed":
                record.update(
                    {
                        "state": "invalidated",
                        "invalidated_at": utc_now(),
                        "invalidation_reason": reason,
                    }
                )
        self.payload["state"] = "running"
        self._save()

    def _save(self) -> None:
        self.payload["updated_at"] = utc_now()
        write_json_atomic(self.path, self.payload)

    def completed(self, name: str) -> bool:
        return self.payload.get("stages", {}).get(name, {}).get("state") == "completed"

    @contextmanager
    def stage(
        self, name: str, *, validator: Callable[[], bool] | None = None
    ) -> Iterator[bool]:
        if self.completed(name):
            valid = True
            reason = "stage output validation returned false"
            if validator is not None:
                try:
                    valid = bool(validator())
                except Exception as exc:
                    valid = False
                    reason = f"stage output validation failed: {exc}"
            if valid:
                # Older/failed attempts may have left diagnostic fields on a
                # stage that later completed successfully. Self-heal these on
                # resume so machine consumers never see a successful stage
                # carrying a stale failure traceback.
                record = self.payload["stages"][name]
                changed = False
                for key in ("traceback", "invalidated_at", "invalidation_reason"):
                    if key in record:
                        record.pop(key)
                        changed = True
                if record.get("error") is not None:
                    record["error"] = None
                    changed = True
                if changed:
                    self._save()
                yield False
                return
            self._invalidate_from(name, reason)
        record = self.payload.setdefault("stages", {}).setdefault(name, {})
        for key in ("traceback", "invalidated_at", "invalidation_reason"):
            record.pop(key, None)
        record.update({"state": "running", "started_at": utc_now(), "error": None})
        self.payload["state"] = "running"
        self._save()
        try:
            yield True
        except Exception as exc:
            record.update(
                {
                    "state": "failed",
                    "finished_at": utc_now(),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            self.payload["state"] = "failed"
            self._save()
            raise
        else:
            record.pop("traceback", None)
            record.update({"state": "completed", "finished_at": utc_now(), "error": None})
            self._save()

    def finish(self) -> None:
        self.payload["state"] = "completed"
        self.payload["completed_at"] = utc_now()
        self._save()
