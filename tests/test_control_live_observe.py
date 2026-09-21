"""Live Control observe must expose the same freshness contract as a pinned read."""

from __future__ import annotations

from typing import Any

from embodirun.robots.sensors.cameras import CameraFrame
from embodirun.services.control.application import (
    ApplicationStaleObservation,
    ControlApplication,
)
from embodirun.services.control.observation_store import ObservationStore
from embodirun.services.control.observation_values import ObservationSnapshot

NOW_NS = 100_000_000


class _LiveService:
    """Minimal owner whose observe reports the latest shared snapshot ID."""

    def __init__(self, store: ObservationStore | None, *, declared: str | None = None) -> None:
        self.store = store
        self.declared = declared
        self.calls = 0

    def describe(self) -> dict[str, Any]:
        return {"status": "ok", "robot_id": "fake"}

    def observe(self, *, runtime_id: str | None, include_robot: bool) -> dict[str, Any]:
        self.calls += 1
        declared = self.declared
        if declared is None and self.store is not None:
            latest = self.store.latest()
            declared = None if latest is None else latest.observation_id
        return {
            "status": "ok",
            "runtime_id": runtime_id,
            "snapshot_id": declared,
            "robot": {"joint": 3} if include_robot else None,
        }


def _publish(store: ObservationStore, published_ns: int) -> ObservationSnapshot:
    observation_id, generation, sequence = store.next_observation_id()
    return store.publish(
        ObservationSnapshot(
            observation_id=observation_id,
            service_instance_id=store.service_instance_id,
            generation=generation,
            sequence=sequence,
            state={"joint": 3},
            cameras=(
                CameraFrame(
                    "front",
                    "image/jpeg",
                    b"jpeg",
                    captured_timestamp_ns=published_ns - 100,
                    received_timestamp_ns=published_ns - 50,
                    clock_domain="host_monotonic_ns",
                ),
            ),
            metadata={"clock_domain": "host_monotonic_ns", "source": "fake"},
            captured_timestamp_ns=published_ns - 100,
            received_timestamp_ns=published_ns - 50,
            published_timestamp_ns=published_ns,
            source_timestamps_ns={"front": published_ns - 100},
            source_received_timestamps_ns={"front": published_ns - 50},
            clock_domains={"front": "host_monotonic_ns"},
            skew_ns=0,
        )
    )


def _observe(app: ControlApplication, **kwargs: Any) -> dict[str, Any]:
    return app.observe(caller_id="caller", session_id="session", **kwargs)


def test_live_observe_reports_identity_age_and_fresh() -> None:
    store = ObservationStore(service_instance_id="live-fresh")
    snapshot = _publish(store, NOW_NS - 1_000)
    app = ControlApplication(
        _LiveService(store),
        observation_store=store,
        clock_ns=lambda: NOW_NS,
        max_observation_age_ns=500_000_000,
    )
    try:
        payload = _observe(app, include_robot=True)
        assert payload["observation_id"] == snapshot.observation_id
        assert payload["age_ns"] == 1_100
        assert payload["fresh"] is True
        assert payload["robot"] == {"joint": 3}
    finally:
        app.close()


def test_live_observe_fails_closed_when_observation_is_stale() -> None:
    store = ObservationStore(service_instance_id="live-stale")
    _publish(store, NOW_NS - 10_000)
    app = ControlApplication(
        _LiveService(store),
        observation_store=store,
        clock_ns=lambda: NOW_NS,
        max_observation_age_ns=1_000,
    )
    try:
        payload = _observe(app)
        assert payload["fresh"] is False
        assert payload["age_ns"] == 10_100
    finally:
        app.close()


def test_live_observe_fails_closed_without_a_shared_store() -> None:
    app = ControlApplication(_LiveService(None, declared="owner-only"))
    try:
        payload = _observe(app)
        assert payload["observation_id"] == "owner-only"
        assert payload["age_ns"] is None
        assert payload["fresh"] is False
        assert payload["freshness"] == "unknown"
        assert "freshness_reason" in payload
    finally:
        app.close()


def test_live_observe_fails_closed_when_snapshot_is_unknown() -> None:
    store = ObservationStore(service_instance_id="live-unknown")
    _publish(store, NOW_NS)
    app = ControlApplication(
        _LiveService(store, declared="obs-does-not-exist"),
        observation_store=store,
        clock_ns=lambda: NOW_NS,
    )
    try:
        payload = _observe(app)
        assert payload["observation_id"] == "obs-does-not-exist"
        assert payload["fresh"] is False
        assert payload["freshness"] == "unknown"
        assert "freshness_reason" in payload
    finally:
        app.close()


def test_live_observe_never_reports_fresh_without_a_snapshot() -> None:
    store = ObservationStore(service_instance_id="live-empty")
    app = ControlApplication(
        _LiveService(store),
        observation_store=store,
        clock_ns=lambda: NOW_NS,
    )
    try:
        payload = _observe(app)
        assert payload["observation_id"] is None
        assert payload["fresh"] is False
        assert payload["age_ns"] is None
    finally:
        app.close()


def test_pinned_observation_freshness_semantics_are_preserved() -> None:
    store = ObservationStore(service_instance_id="pinned")
    fresh_snapshot = _publish(store, NOW_NS - 1_000)
    app = ControlApplication(
        _LiveService(store),
        observation_store=store,
        clock_ns=lambda: NOW_NS,
        max_observation_age_ns=500_000_000,
    )
    try:
        pinned = _observe(app, observation_id=fresh_snapshot.observation_id)
        assert pinned["observation_id"] == fresh_snapshot.observation_id
        assert pinned["age_ns"] == 1_100
        assert pinned["fresh"] is True

        stale_snapshot = _publish(store, NOW_NS - 50_000_000)
        stale = _observe(app, observation_id=stale_snapshot.observation_id, max_age_ns=1_000)
        assert stale["observation_id"] == stale_snapshot.observation_id
        assert stale["fresh"] is False

        try:
            _observe(app, observation_id="obs-not-in-store")
        except ApplicationStaleObservation:
            pass
        else:  # pragma: no cover - the pinned branch must stay strict
            raise AssertionError("pinned observe must reject an unknown observation")
    finally:
        app.close()


def test_live_missing_id_never_borrows_latest_snapshot():
    store = ObservationStore(service_instance_id="missing-id")
    _publish(store, NOW_NS - 100)
    service = _LiveService(store)
    service.observe = lambda **kwargs: {"robot": {"joint": 999}}
    app = ControlApplication(service, observation_store=store, clock_ns=lambda: NOW_NS)
    try:
        payload = _observe(app)
        assert payload["fresh"] is False
        assert payload["observation_id"] is None
        assert payload["robot"] == {"joint": 999}
    finally:
        app.close()
