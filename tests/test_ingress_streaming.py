import asyncio
import pytest

from loom.config import Settings
from loom.observability import IngressGuard, new_metrics


def test_large_upload_spools_to_disk_and_replays_in_chunks(monkeypatch):
    async def scenario():
        import loom.observability as observability

        created = []
        original = observability.tempfile.SpooledTemporaryFile

        def tracked(*args, **kwargs):
            stream = original(*args, **kwargs)
            created.append(stream)
            return stream

        monkeypatch.setattr(observability.tempfile, "SpooledTemporaryFile", tracked)
        received = []

        async def application(scope, receive, send):
            while True:
                message = await receive()
                received.append(message.get("body", b""))
                if not message.get("more_body", False):
                    break
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        parts = iter([
            {"type": "http.request", "body": b"a" * 700_000, "more_body": True},
            {"type": "http.request", "body": b"b" * 700_000, "more_body": False},
        ])

        async def receive():
            return next(parts)

        async def send(message):
            pass

        guard = IngressGuard(application, Settings(), new_metrics())
        await guard({"type": "http", "path": "/api/papers", "method": "POST", "headers": []}, receive, send)
        assert sum(map(len, received)) == 1_400_000
        assert max(map(len, received)) <= 64 * 1024
        assert created[0]._rolled is True
        assert created[0].closed is True

    asyncio.run(scenario())


@pytest.mark.parametrize('failure', ['overflow','disconnect','downstream'])
def test_spool_is_closed_and_capacity_released_on_failure(monkeypatch, failure):
    async def scenario():
        import loom.observability as observability
        streams = []
        original = observability.tempfile.SpooledTemporaryFile
        def track(*args, **kwargs):
            result = original(*args, **kwargs); streams.append(result); return result
        monkeypatch.setattr(observability.tempfile, 'SpooledTemporaryFile', track)
        called = []; sent = []
        async def app(scope, receive, send):
            called.append(True)
            raise RuntimeError('downstream')
        parts = iter([{'type':'http.request','body':b'a'*1_100_000,'more_body':True},
                      {'type':'http.disconnect'} if failure == 'disconnect' else
                      {'type':'http.request','body':b'b'*(2_000_000 if failure == 'overflow' else 0),'more_body':False}])
        async def receive(): return next(parts)
        async def send(message): sent.append(message)
        guard = IngressGuard(app, Settings(max_upload_bytes=1_200_000), new_metrics())
        scope = {'type':'http','path':'/api/papers','method':'POST','headers':[]}
        if failure == 'downstream':
            with pytest.raises(RuntimeError): await guard(scope, receive, send)
        else:
            await guard(scope, receive, send)
            assert not called
        if failure == 'overflow': assert sent[0]['status'] == 413
        assert streams[0]._rolled and streams[0].closed
        assert guard.active == guard.uploads == 0
    asyncio.run(scenario())
