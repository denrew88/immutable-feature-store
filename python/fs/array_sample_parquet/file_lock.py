"""재개 가능한 Python builder에서 사용하는 owner-token 파일락."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
import uuid
from typing import Optional


RELEASE_RETRY_DELAYS_SECONDS = (0.025, 0.05, 0.1, 0.2, 0.4)
STALE_LOCK_MILLIS_ENV = "FS_FILE_LOCK_STALE_MILLIS"


class FileLockOwnerMismatch(RuntimeError):
    """현재 프로세스가 만든 lock이 아니면 release하지 않도록 알리는 예외."""

    pass


class FilePathLock:
    """파일 생성 원자성으로 잡는 lock.

    lock 파일에는 token, pid, host를 기록합니다. release 시에는 token이
    일치할 때만 삭제하므로, 다른 프로세스가 새로 잡은 lock을 실수로 지우지
    않습니다. Windows에서 백신/IDE가 lock 파일 핸들을 잠깐 잡는 경우를
    대비해 삭제는 짧게 재시도합니다.
    """

    def __init__(self, path: str, *, timeout_seconds: float = 30.0, stale_lock_seconds: Optional[float] = None):
        self.path = os.path.abspath(path)
        self.timeout_seconds = float(timeout_seconds)
        self.stale_lock_seconds = _default_stale_lock_seconds() if stale_lock_seconds is None else max(0.0, float(stale_lock_seconds))
        self.token = uuid.uuid4().hex
        self.owner_text = _owner_text(self.path, self.token)
        self._fd: Optional[int] = None
        self._acquired = False

    @property
    def acquired(self) -> bool:
        return self._acquired

    def acquire(self) -> None:
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        deadline = time.monotonic() + self.timeout_seconds
        last_error: Optional[BaseException] = None
        while True:
            try:
                # O_CREAT | O_EXCL은 "없을 때만 생성"을 원자적으로 보장합니다.
                # 이 파일 생성 성공이 lock 소유권을 얻었다는 유일한 기준입니다.
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                success = False
                try:
                    # lock 파일 본문은 사람이 장애 원인을 추적하기 위한 metadata입니다.
                    # 실제 release 권한 검증에는 token 값을 사용합니다.
                    os.write(fd, self.owner_text.encode("utf-8"))
                    self._fd = fd
                    self._acquired = True
                    success = True
                    return
                finally:
                    if not success:
                        try:
                            os.close(fd)
                        finally:
                            self._delete_with_retries()
            except FileExistsError:
                pass
            except OSError as exc:
                last_error = exc

            # stale lock 회수는 명시적으로 켠 경우에만 동작합니다. 판단이 조금이라도
            # 애매하면 False가 반환되어 기존 lock을 유지하고 timeout으로 실패합니다.
            if self._maybe_delete_stale_lock():
                continue
            if time.monotonic() >= deadline:
                detail = self._describe_existing_lock()
                raise TimeoutError(f"timed out acquiring file lock: {self.path} existing_lock={detail}") from last_error
            time.sleep(0.05)

    def release(self) -> None:
        if not self._acquired:
            return

        last_error: Optional[BaseException] = None
        for attempt in range(len(RELEASE_RETRY_DELAYS_SECONDS) + 1):
            try:
                # release_once는 token 검증과 실제 삭제를 한 번 시도합니다.
                # Windows에서 외부 프로세스가 파일 handle을 잠깐 잡으면 OSError가
                # 날 수 있으므로 여기에서 짧게 retry합니다.
                self._release_once()
                self._acquired = False
                return
            except FileLockOwnerMismatch:
                raise
            except OSError as exc:
                last_error = exc
            if attempt < len(RELEASE_RETRY_DELAYS_SECONDS):
                time.sleep(RELEASE_RETRY_DELAYS_SECONDS[attempt])
        raise OSError(f"failed to release file lock after retries: {self.path}") from last_error

    def __enter__(self) -> "FilePathLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def _release_once(self) -> None:
        self._close_fd()
        try:
            existing = _read_text(self.path)
        except FileNotFoundError:
            # 이미 lock 파일이 없다면 release가 한 번 더 호출된 상황으로 보고 성공 처리합니다.
            # PermissionError/OSError는 여기서 삼키지 않습니다. 실제 lock 파일이 남았는데
            # 접근만 실패한 상황을 "없음"으로 착각하면 stale lock을 남길 수 있습니다.
            return
        existing_token = _field(existing, "token")
        if existing_token != self.token:
            # 같은 path라도 token이 다르면 다른 프로세스가 새로 잡은 lock일 수 있습니다.
            # 이 경우 삭제하면 동시 write 보호를 깨뜨리므로 acquired 상태를 유지한 채 실패합니다.
            raise FileLockOwnerMismatch(
                f"refusing to release file lock owned by another token: {self.path} "
                f"expected_token={self.token} existing_lock={_summarize(existing)}"
            )
        try:
            os.remove(self.path)
        except FileNotFoundError:
            return

    def _close_fd(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def _delete_with_retries(self) -> bool:
        last_error: Optional[BaseException] = None
        for attempt in range(len(RELEASE_RETRY_DELAYS_SECONDS) + 1):
            try:
                # delete retry는 백신/인덱서/IDE가 lock 파일을 짧게 열고 있는 Windows
                # 환경을 방어하기 위한 것입니다.
                os.remove(self.path)
                return True
            except FileNotFoundError:
                return True
            except OSError as exc:
                last_error = exc
            if attempt < len(RELEASE_RETRY_DELAYS_SECONDS):
                time.sleep(RELEASE_RETRY_DELAYS_SECONDS[attempt])
        if last_error is not None:
            raise OSError(f"failed to delete file lock after retries: {self.path}") from last_error
        return False

    def _maybe_delete_stale_lock(self) -> bool:
        # 기본값은 0초이므로 자동 stale 회수는 꺼져 있습니다. 운영자가 명시적으로
        # 시간을 지정했을 때만 "죽은 프로세스의 오래된 lock" 후보를 검사합니다.
        if self.stale_lock_seconds <= 0.0 or not _safe_is_file(self.path):
            return False
        try:
            modified = _safe_getmtime(self.path)
            if modified is None:
                return False
            age_seconds = time.time() - modified
            if age_seconds < self.stale_lock_seconds:
                return False
            existing = _read_text(self.path)
            token = _field(existing, "token")
            host = _field(existing, "host")
            pid = _field(existing, "pid")
            current_host = _current_host()
            if not token or not pid or current_host == "unknown" or host != current_host:
                return False
            alive = _is_process_alive(pid)
            # None은 "판단 불가"입니다. 판단 불가나 alive=True는 모두 삭제하지 않습니다.
            if alive is None or alive:
                return False
            before_delete = _read_text(self.path)
            # 삭제 직전에 token을 다시 확인해서, 첫 read 이후 다른 프로세스가
            # 같은 path에 새 lock을 만든 경우에는 삭제하지 않습니다.
            if _field(before_delete, "token") != token:
                return False
            return self._delete_with_retries()
        except OSError:
            return False

    def _describe_existing_lock(self) -> str:
        try:
            return _summarize(_read_text(self.path))
        except FileNotFoundError:
            return "missing"
        except OSError as exc:
            return f"unreadable path={self.path} error={exc}"


def _owner_text(path: str, token: str) -> str:
    # token은 소유권 검증용이고, pid/thread/host/path는 장애 분석용 metadata입니다.
    return "\n".join(
        [
            f"token={token}",
            f"pid={os.getpid()}",
            f"thread={threading.current_thread().name}",
            f"host={_current_host()}",
            f"created_at_ms={int(time.time() * 1000)}",
            f"path={path}",
            "",
        ]
    )


def _default_stale_lock_seconds() -> float:
    # 환경 변수는 millisecond 단위로 받습니다. 잘못된 값이면 stale 회수를 끄는
    # 쪽으로 실패해서 lock을 잘못 지우지 않게 합니다.
    raw = os.environ.get(STALE_LOCK_MILLIS_ENV, "").strip()
    if not raw:
        return 0.0
    try:
        return max(0.0, float(raw) / 1000.0)
    except ValueError:
        return 0.0


def _current_host() -> str:
    try:
        return socket.gethostname()
    except OSError:
        return "unknown"


def _safe_is_file(path: str) -> bool:
    try:
        return os.path.isfile(path)
    except OSError:
        return False


def _safe_getmtime(path: str) -> Optional[float]:
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _field(text: str, name: str) -> Optional[str]:
    prefix = f"{name}="
    for line in text.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :]
    return None


def _summarize(text: str) -> str:
    return " ".join(text.split())


def _is_process_alive(pid_text: str) -> Optional[bool]:
    # 반환값 의미:
    # True  = 살아있다고 판단됨
    # False = 죽었다고 판단됨
    # None  = 권한/플랫폼 문제로 판단 불가, stale lock 삭제 금지
    try:
        pid = int(pid_text)
    except ValueError:
        return None
    if pid <= 0:
        return None
    if pid == os.getpid():
        return True
    if sys.platform.startswith("win"):
        return _is_windows_process_alive(pid)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return None


def _is_windows_process_alive(pid: int) -> Optional[bool]:
    # Python 표준 라이브러리만 쓰기 위해 Windows에서는 tasklist를 호출합니다.
    # 실행 실패, timeout, 권한 문제는 모두 None으로 보고 stale 삭제를 포기합니다.
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except Exception:
        return None
    return str(pid) in result.stdout
