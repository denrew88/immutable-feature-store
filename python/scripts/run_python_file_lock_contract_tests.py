from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Type


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (
    REPO_ROOT / "packages" / "scalar_feature_shard" / "src",
    REPO_ROOT / "packages" / "array_sample_parquet" / "src",
    REPO_ROOT / "python",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from array_sample_parquet._impl.file_lock import FilePathLock as ArrayPackageFilePathLock
from fs.array_sample_parquet.file_lock import FilePathLock as ArrayLegacyFilePathLock
from scalar_feature_shard._impl.file_lock import FilePathLock as ScalarFilePathLock


# Python FilePathLock 구현의 owner-token/stale/retry 계약을 검증하는 실행형 테스트입니다.


def _field(text: str, name: str) -> str:
    prefix = f"{name}="
    for line in text.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :]
    raise AssertionError(f"missing owner field: {name}")


def _replace_field(text: str, name: str, value: str) -> str:
    prefix = f"{name}="
    out = []
    replaced = False
    for line in text.splitlines():
        if line.startswith(prefix):
            out.append(prefix + value)
            replaced = True
        else:
            out.append(line)
    if not replaced:
        raise AssertionError(f"missing owner field: {name}")
    return "\n".join(out) + "\n"


def _write_stale_lock(path: Path, *, pid: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "token=stale-token",
                f"pid={pid}",
                "thread=stale-test",
                f"host={__import__('socket').gethostname()}",
                f"created_at_ms={int(time.time() * 1000) - 10_000}",
                f"path={path}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    old = time.time() - 10.0
    os.utime(path, (old, old))


def _test_release_deletes_owned_lock(lock_cls: Type[ScalarFilePathLock], path: Path) -> None:
    """정상 acquire/release가 lock 파일과 acquired 상태를 모두 정리하는지 확인합니다."""

    lock = lock_cls(str(path), timeout_seconds=1.0)
    lock.acquire()
    text = path.read_text(encoding="utf-8")
    assert _field(text, "token") == lock.token
    assert _field(text, "pid") == str(os.getpid())
    lock.release()
    assert not path.exists()
    assert not lock.acquired


def _test_release_refuses_other_token(lock_cls: Type[ScalarFilePathLock], path: Path) -> None:
    """다른 owner token으로 바뀐 lock 파일을 release가 삭제하지 않는지 확인합니다."""

    lock = lock_cls(str(path), timeout_seconds=1.0)
    lock.acquire()
    original = path.read_text(encoding="utf-8")
    path.write_text(_replace_field(original, "token", "somebody-else"), encoding="utf-8")
    try:
        lock.release()
    except RuntimeError as exc:
        assert exc.__class__.__name__ == "FileLockOwnerMismatch", exc
    else:
        raise AssertionError("release must refuse a lock file with another owner token")
    assert path.exists()
    assert lock.acquired
    path.write_text(original, encoding="utf-8")
    lock.release()
    assert not path.exists()


def _test_fresh_lock_is_not_stolen(lock_cls: Type[ScalarFilePathLock], path: Path) -> None:
    """아직 stale 기준에 도달하지 않은 lock은 pid가 죽어 보여도 훔치지 않습니다."""

    _write_stale_lock(path, pid=999999)
    now = time.time()
    os.utime(path, (now, now))
    lock = lock_cls(str(path), timeout_seconds=0.2, stale_lock_seconds=60.0)
    try:
        lock.acquire()
    except TimeoutError:
        pass
    else:
        try:
            raise AssertionError("fresh existing lock must not be stolen")
        finally:
            if lock.acquired:
                lock.release()
    assert path.exists()
    path.unlink()


def _test_dead_stale_lock_can_be_recovered(lock_cls: Type[ScalarFilePathLock], path: Path) -> None:
    """stale 회수가 명시적으로 켜진 경우 죽은 pid의 오래된 lock만 회수합니다."""

    _write_stale_lock(path, pid=999999)
    lock = lock_cls(str(path), timeout_seconds=2.0, stale_lock_seconds=0.001)
    lock.acquire()
    try:
        text = path.read_text(encoding="utf-8")
        assert _field(text, "token") == lock.token
        assert _field(text, "pid") == str(os.getpid())
    finally:
        lock.release()
    assert not path.exists()


def _test_release_retries_external_file_handle(lock_cls: Type[ScalarFilePathLock], path: Path) -> None:
    """외부 프로세스가 lock 파일 handle을 잠깐 잡아도 release retry가 성공하는지 봅니다."""

    lock = lock_cls(str(path), timeout_seconds=1.0)
    lock.acquire()
    proc = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--hold-lock-file",
            str(path),
            "--hold-seconds",
            "0.30",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        ready = proc.stdout.readline().strip() if proc.stdout is not None else ""
        assert ready == "ready", ready
        started = time.perf_counter()
        lock.release()
        elapsed = time.perf_counter() - started
        if sys.platform.startswith("win"):
            assert elapsed >= 0.15, f"release did not appear to wait for the external handle: {elapsed:.3f}s"
        assert not path.exists()
        assert proc.wait(timeout=5.0) == 0
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5.0)


def _test_release_does_not_trust_exists_false(lock_cls: Type[ScalarFilePathLock], path: Path) -> None:
    """exists()가 거짓을 반환해도 실제 파일을 읽고 삭제하는지 확인합니다."""

    lock = lock_cls(str(path), timeout_seconds=1.0)
    module = sys.modules[lock_cls.__module__]
    original_exists = module.os.path.exists
    lock.acquire()
    try:
        # 일부 Windows/권한 상황에서는 exists()가 실제 파일 상태를 안정적으로 표현하지 못할 수 있습니다.
        # release가 exists()만 믿으면 lock 파일을 남기고도 성공 처리하므로, 강제로 False를 반환시킵니다.
        module.os.path.exists = lambda _path: False
        lock.release()
    finally:
        module.os.path.exists = original_exists
        if lock.acquired:
            lock.release()
    assert not path.exists()
    assert not lock.acquired


def _run_for_lock_class(label: str, lock_cls: Type[ScalarFilePathLock], root: Path) -> None:
    # scalar package, array package, legacy python/fs 경로가 같은 계약을 지키는지
    # 동일한 테스트 묶음을 재사용합니다.
    _test_release_deletes_owned_lock(lock_cls, root / f"{label}_owned.lock")
    _test_release_refuses_other_token(lock_cls, root / f"{label}_mismatch.lock")
    _test_fresh_lock_is_not_stolen(lock_cls, root / f"{label}_fresh.lock")
    _test_dead_stale_lock_can_be_recovered(lock_cls, root / f"{label}_stale.lock")
    _test_release_retries_external_file_handle(lock_cls, root / f"{label}_external_handle.lock")
    _test_release_does_not_trust_exists_false(lock_cls, root / f"{label}_exists_false.lock")


def _hold_lock_file(path: Path, hold_seconds: float) -> None:
    with open(path, "rb"):
        print("ready", flush=True)
        time.sleep(float(hold_seconds))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--hold-lock-file", default="")
    ap.add_argument("--hold-seconds", type=float, default=0.30)
    args = ap.parse_args(argv)

    if args.hold_lock_file:
        _hold_lock_file(Path(args.hold_lock_file), args.hold_seconds)
        return

    root = Path(args.out_dir).resolve() if args.out_dir else REPO_ROOT / "data" / "tmp_python_file_lock_contract_tests"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    _run_for_lock_class("scalar_package", ScalarFilePathLock, root)
    _run_for_lock_class("array_package", ArrayPackageFilePathLock, root)
    _run_for_lock_class("array_legacy", ArrayLegacyFilePathLock, root)

    print("python file lock contract tests passed")


if __name__ == "__main__":
    main(sys.argv[1:])
