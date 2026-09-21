"""Actual UCX transfer/ownership checks; optional dependency, no devices opened."""

import asyncio

import pytest

from embodirun.services.rollout.nixl_tensors import NixlTensorTransport

torch = pytest.importorskip("torch")
pytest.importorskip("nixl")


def equal_bytes(left, right):
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and torch.equal(
            left.contiguous().reshape(-1).view(torch.uint8),
            right.contiguous().reshape(-1).view(torch.uint8),
        )
    )


async def connected(device, *, timeout=5):
    a = NixlTensorTransport(device, chunk_bytes=37, timeout_s=timeout)
    b = NixlTensorTransport(device, chunk_bytes=31, timeout_s=timeout)
    ab, ba = asyncio.Queue(), asyncio.Queue()
    await asyncio.gather(
        a.connect(ab.put, ba.get, initiator=True),
        b.connect(ba.put, ab.get, initiator=False),
    )
    return a, b, ab, ba


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_registered_chunks_preserve_bytes_ownership_and_reverse_direction(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    async def run():
        a, b, _, _ = await connected(device)
        retained = []
        try:
            for version in range(3):
                values = [
                    (torch.arange(48, dtype=torch.float32, device=device) + version).reshape(4, 12)[:, ::2],
                    torch.tensor(-0.0, device=device),
                    torch.tensor([1.25, 2.5], dtype=torch.bfloat16, device=device),
                    torch.empty(0, dtype=torch.int64, device=device),
                ]
                assert not values[0].is_contiguous()
                _, (received, metadata) = await asyncio.gather(a.send(values, {"version": version}), b.recv())
                assert metadata == {"version": version}
                assert all(equal_bytes(x, y) for x, y in zip(values, received))
                for actual, previous in retained:
                    assert all(equal_bytes(x, y) for x, y in zip(actual, previous))
                retained.append((received, [tensor.clone() for tensor in received]))
                assert a.last_metrics["chunks"] == b.last_metrics["chunks"] == 4
            _, (received, metadata) = await asyncio.gather(b.send(values, {"reverse": True}), a.recv())
            assert metadata == {"reverse": True}
            assert all(equal_bytes(x, y) for x, y in zip(values, received))
            _, (empty, _) = await asyncio.gather(a.send([values[-1]], {}), b.recv())
            assert empty[0].numel() == 0
            assert a.last_metrics["chunks"] == 0 and a.sequence == b.sequence == 5
        finally:
            a.close()
            b.close()
        assert a.closed and b.closed and a.arena is b.arena is None

    asyncio.run(run())


def test_timeout_quarantines_registered_memory_and_forbids_connection_reuse():
    async def run():
        a, b, _, _ = await connected("cpu")
        a.timeout_s = b.timeout_s = 0.05
        original = b._send

        async def stop_credit(message):
            if message.get("kind") == "copied":
                await asyncio.Event().wait()
            await original(message)

        b._send = stop_credit
        result = await asyncio.gather(a.send([torch.arange(50)], {}), b.recv(), return_exceptions=True)
        assert all(isinstance(error, TimeoutError) for error in result)
        assert a.failed and b.failed
        assert a.arena is not None and b.arena is not None
        with pytest.raises(RuntimeError, match="restart the worker"):
            await a.send([torch.zeros(1)], {})
        a.close()
        b.close()
        assert not a.closed and not b.closed  # Still retained until process exit.

    asyncio.run(run())


def test_foreign_sequence_is_rejected_before_receiving_any_payload():
    async def run():
        a, b, _, _ = await connected("cpu")
        a.timeout_s = b.timeout_s = 0.1
        original = a._send

        async def corrupt(message):
            if message.get("kind") == "offer":
                message = dict(message, sequence=99)
            await original(message)

        a._send = corrupt
        result = await asyncio.gather(a.send([torch.arange(50)], {}), b.recv(), return_exceptions=True)
        assert isinstance(result[0], TimeoutError)
        assert isinstance(result[1], ValueError) and "sequence" in str(result[1])
        assert a.sequence == b.sequence == 0
        assert not a._handles  # No READY credit, so no UCX WRITE was posted.

    asyncio.run(run())
