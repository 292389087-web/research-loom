"""同主机协作式 GPU 互斥，进程退出由 OS 释放；不是显存虚拟化。"""

from contextlib import contextmanager
import os
from pathlib import Path


class ResourceBusy(RuntimeError):
    pass


@contextmanager
def gpu_slot(root):
    path = Path(root) / "gpu.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = open(path, "a+b")
    if path.stat().st_size == 0:
        stream.write(b"0")
        stream.flush()
    stream.seek(0)
    acquired = False
    try:
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError as exc:
            raise ResourceBusy("GPU 被另一个本地任务占用，等待资源") from exc
        yield
    finally:
        if acquired:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def gpu_busy(root):
    try:
        with gpu_slot(root):
            return False
    except ResourceBusy:
        return True
