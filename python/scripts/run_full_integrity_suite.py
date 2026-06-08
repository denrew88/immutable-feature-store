from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(sys.executable)


@dataclass(frozen=True)
class CommandSpec:
    command_id: str
    title: str
    argv: Callable[["SuiteContext"], list[str]]
    enabled_by_default: bool = True


@dataclass(frozen=True)
class CheckSpec:
    check_id: str
    title: str
    command_ids: tuple[str, ...]


@dataclass
class SuiteContext:
    repo_root: Path
    out_dir: Path
    java_exe: str
    javac_exe: str
    include_packaging: bool
    include_api_examples: bool


def _tool_from_java_home(tool_name: str) -> str:
    java_home = os.environ.get("JAVA_HOME")
    exe_name = tool_name + (".exe" if os.name == "nt" else "")
    if java_home:
        candidate = Path(java_home) / "bin" / exe_name
        if candidate.exists():
            return str(candidate)
    default = Path("C:/Program Files/Java/jdk-1.8/bin") / exe_name
    if default.exists():
        return str(default)
    return exe_name


def _cp(*parts: Path) -> str:
    return os.pathsep.join(str(part) for part in parts)


def _java_runtime_cp(repo_root: Path) -> str:
    return _cp(repo_root / "java" / "lib" / "*", repo_root / "java" / "out")


def _package_example_cp(repo_root: Path, out_dir: Path, jar_path: Path) -> str:
    return _cp(out_dir, jar_path, repo_root / "java" / "lib" / "*")


def _python_script(name: str, *extra: str) -> Callable[[SuiteContext], list[str]]:
    def make(ctx: SuiteContext) -> list[str]:
        return [str(PYTHON), str(ctx.repo_root / "python" / "scripts" / name), *extra]

    return make


def _java_main(class_name: str, *extra: str) -> Callable[[SuiteContext], list[str]]:
    def make(ctx: SuiteContext) -> list[str]:
        return [ctx.java_exe, "-cp", _java_runtime_cp(ctx.repo_root), class_name, *extra]

    return make


def _build_python_package(package_dir: str) -> Callable[[SuiteContext], list[str]]:
    def make(ctx: SuiteContext) -> list[str]:
        return [str(PYTHON), "-m", "build", "--no-isolation", str(ctx.repo_root / package_dir)]

    return make


def _build_java_package(script_path: str) -> Callable[[SuiteContext], list[str]]:
    def make(ctx: SuiteContext) -> list[str]:
        return ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(ctx.repo_root / script_path)]

    return make


def _compile_java(ctx: SuiteContext) -> list[str]:
    sources = sorted((ctx.repo_root / "java" / "src").rglob("*.java"))
    arg_file = ctx.out_dir / "java_sources.txt"
    arg_file.write_text("\n".join(str(path) for path in sources), encoding="utf-8")
    return [
        ctx.javac_exe,
        "-encoding",
        "UTF-8",
        "-cp",
        str(ctx.repo_root / "java" / "lib" / "*"),
        "-d",
        str(ctx.repo_root / "java" / "out"),
        "@" + str(arg_file),
    ]


def _compile_scalar_jar_example(ctx: SuiteContext) -> list[str]:
    out_dir = ctx.repo_root / "packages" / "scalar_feature_shard_java" / "examples" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        ctx.javac_exe,
        "-encoding",
        "UTF-8",
        "-cp",
        _cp(
            ctx.repo_root / "packages" / "scalar_feature_shard_java" / "dist" / "scalar-feature-shard-java-0.1.0.jar",
            ctx.repo_root / "java" / "lib" / "*",
        ),
        "-d",
        str(out_dir),
        str(
            ctx.repo_root
            / "packages"
            / "scalar_feature_shard_java"
            / "examples"
            / "BuildScalarFeatureShardWithJarExample.java"
        ),
        str(
            ctx.repo_root
            / "packages"
            / "scalar_feature_shard_java"
            / "examples"
            / "BuildScalarDenseLongFromValueApiWithJarExample.java"
        ),
    ]


def _run_scalar_jar_example(ctx: SuiteContext) -> list[str]:
    return [
        ctx.java_exe,
        "-cp",
        _package_example_cp(
            ctx.repo_root,
            ctx.repo_root / "packages" / "scalar_feature_shard_java" / "examples" / "out",
            ctx.repo_root / "packages" / "scalar_feature_shard_java" / "dist" / "scalar-feature-shard-java-0.1.0.jar",
        ),
        "BuildScalarFeatureShardWithJarExample",
    ]


def _compile_array_sample_jar_example(ctx: SuiteContext) -> list[str]:
    out_dir = ctx.repo_root / "packages" / "array_sample_parquet_java" / "examples" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        ctx.javac_exe,
        "-encoding",
        "UTF-8",
        "-cp",
        _cp(
            ctx.repo_root / "packages" / "array_sample_parquet_java" / "dist" / "array-sample-parquet-java-0.1.0.jar",
            ctx.repo_root / "java" / "lib" / "*",
        ),
        "-d",
        str(out_dir),
        str(
            ctx.repo_root
            / "packages"
            / "array_sample_parquet_java"
            / "examples"
            / "BuildArraySampleParquetWithJarExample.java"
        ),
        str(
            ctx.repo_root
            / "packages"
            / "array_sample_parquet_java"
            / "examples"
            / "BuildArraySampleParquetFromValueApiWithJarExample.java"
        ),
    ]


def _run_array_sample_jar_example(ctx: SuiteContext) -> list[str]:
    return [
        ctx.java_exe,
        "-cp",
        _package_example_cp(
            ctx.repo_root,
            ctx.repo_root / "packages" / "array_sample_parquet_java" / "examples" / "out",
            ctx.repo_root / "packages" / "array_sample_parquet_java" / "dist" / "array-sample-parquet-java-0.1.0.jar",
        ),
        "BuildArraySampleParquetWithJarExample",
    ]


def _run_api_jar_examples(ctx: SuiteContext) -> list[str]:
    scalar_out = ctx.out_dir / "tmp_scalar_feature_shard_api_jar_example"
    array_out = ctx.out_dir / "tmp_array_sample_parquet_api_jar_example"
    script = (
        "$ErrorActionPreference = 'Stop'; "
        f"$py = '{PYTHON}'; "
        f"$scalarOut = '{scalar_out}'; "
        f"$arrayOut = '{array_out}'; "
        "Remove-Item -LiteralPath $scalarOut,$arrayOut -Recurse -Force -ErrorAction SilentlyContinue; "
        "$proc = Start-Process -FilePath $py "
        "-ArgumentList @('python\\scripts\\serve_synthetic_value_api.py','--host','127.0.0.1','--port','8010') "
        "-WorkingDirectory (Get-Location) -PassThru -WindowStyle Hidden; "
        "try { "
        "Start-Sleep -Seconds 3; "
        "& '" + ctx.java_exe + "' -cp 'packages\\scalar_feature_shard_java\\examples\\out;packages\\scalar_feature_shard_java\\dist\\scalar-feature-shard-java-0.1.0.jar;java\\lib\\*' "
        "BuildScalarDenseLongFromValueApiWithJarExample --base-url http://127.0.0.1:8010 "
        "--sample-meta data\\tmp_scalar_feature_shard_jar_example\\sample_meta.parquet "
        "--feature-meta data\\tmp_scalar_feature_shard_jar_example\\feature_meta.parquet "
        "--out-dir $scalarOut --feature-chunk-size 2 --seed 11; "
        "if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; "
        "& '" + ctx.java_exe + "' -cp 'packages\\array_sample_parquet_java\\examples\\out;packages\\array_sample_parquet_java\\dist\\array-sample-parquet-java-0.1.0.jar;java\\lib\\*' "
        "BuildArraySampleParquetFromValueApiWithJarExample --base-url http://127.0.0.1:8010 "
        "--sample-meta data\\tmp_array_sample_parquet_jar_example\\sample_meta.parquet "
        "--feature-meta data\\tmp_array_sample_parquet_jar_example\\feature_meta.parquet "
        "--out-dir $arrayOut --feature-chunk-size 2 --seed 11 --compression none; "
        "if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } "
        "} finally { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }"
    )
    return ["powershell", "-Command", script]


def _scalar_api_jar_reader_smoke(ctx: SuiteContext) -> list[str]:
    manifest = ctx.out_dir / "tmp_scalar_feature_shard_api_jar_example" / "scalar_shard" / "scalar_shard_manifest.json"
    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, str(Path('packages/scalar_feature_shard/src').resolve()))\n"
        "from scalar_feature_shard import open_dense_long_shard\n"
        f"manifest={str(manifest)!r}\n"
        "with open_dense_long_shard(manifest) as ds:\n"
        "    values, valid = ds.load_feature_by_id(0)\n"
        "    sample_values, sample_valid = ds.load_sample_by_id(0)\n"
        "    assert len(values) > 0 and len(sample_values) > 0\n"
        "print('scalar api jar example reader smoke passed')\n"
    )
    return [str(PYTHON), "-c", code]


def _array_api_jar_validate(ctx: SuiteContext) -> list[str]:
    array_out = ctx.out_dir / "tmp_array_sample_parquet_api_jar_example"
    return [
        str(PYTHON),
        str(ctx.repo_root / "python" / "scripts" / "validate_array_sample_parquet_exhaustive.py"),
        str(array_out / "array_sample_parquet_manifest.json"),
        "--stage-dir",
        str(array_out),
    ]


COMMANDS: tuple[CommandSpec, ...] = (
    CommandSpec("java.compile", "Java source compile", _compile_java),
    CommandSpec("scalar.python.package", "Scalar Python package/direct integrity", _python_script("run_scalar_package_tests.py")),
    CommandSpec("scalar.python.api", "Scalar Python reader/API", _python_script("run_scalar_api_tests.py")),
    CommandSpec("scalar.python.selection_api", "Scalar selection API", _python_script("run_selection_api_tests.py")),
    CommandSpec("scalar.python_java.parity", "Scalar Python/Java parity", _python_script("run_scalar_python_java_parity_tests.py")),
    CommandSpec(
        "scalar.python.concurrency",
        "Scalar Python concurrency",
        _python_script("run_scalar_concurrent_builder_tests.py", "--n-samples", "24", "--n-features", "128", "--n-workers", "6"),
    ),
    CommandSpec(
        "scalar.python.raw_recovery",
        "Scalar Python raw-stage recovery/finalize locks",
        _python_script("run_scalar_raw_stage_recovery_contract_tests.py"),
    ),
    CommandSpec("scalar.java.builder", "Scalar Java builder", _java_main("scripts.RunScalarBuilderTestsMain")),
    CommandSpec(
        "scalar.java.concurrency",
        "Scalar Java concurrency",
        _java_main("scripts.RunScalarConcurrentBuilderTestsMain", "--n-samples", "24", "--n-features", "128", "--n-workers", "6"),
    ),
    CommandSpec(
        "scalar.java.raw_recovery",
        "Scalar Java raw-stage recovery/finalize locks",
        _java_main("scripts.RunScalarRawStageRecoveryContractTestsMain"),
    ),
    CommandSpec("array.parquet.python", "Array sample parquet Python direct/API", _python_script("run_array_sample_parquet_tests.py")),
    CommandSpec(
        "array.parquet.python.concurrency",
        "Array sample parquet Python concurrency",
        _python_script("run_array_sample_parquet_concurrent_builder_tests.py", "--n-samples", "24", "--n-features", "12", "--n-workers", "6"),
    ),
    CommandSpec("array.binary.python.package", "Array binary Python package", _python_script("run_array_binary_package_tests.py")),
    CommandSpec("array.binary.python.storage", "Array binary storage integrity", _python_script("run_array_binary_storage_tests.py")),
    CommandSpec("array.binary.python.api", "Array binary API tests", _python_script("run_array_api_tests.py")),
    CommandSpec("array.parquet_binary.parity", "Array parquet/binary parity", _python_script("run_array_parquet_binary_parity_tests.py")),
    CommandSpec("array.parquet.java", "Array sample parquet Java", _java_main("scripts.RunArraySampleParquetTestsMain")),
    CommandSpec("array.binary.java.builder", "Array binary Java builder", _java_main("scripts.RunArrayBuilderTestsMain")),
    CommandSpec("array.binary.java.v3", "Array binary Java v3", _java_main("scripts.RunArrayV3TestsMain")),
    CommandSpec(
        "package.python.array_binary",
        "Build array_binary_shard wheel/sdist",
        _build_python_package("packages/array_binary_shard"),
    ),
    CommandSpec(
        "package.python.array_parquet",
        "Build array_sample_parquet wheel/sdist",
        _build_python_package("packages/array_sample_parquet"),
    ),
    CommandSpec(
        "package.python.scalar",
        "Build scalar_feature_shard wheel/sdist",
        _build_python_package("packages/scalar_feature_shard"),
    ),
    CommandSpec(
        "package.java.array_binary",
        "Build array-binary-shard Java jar/source/javadoc",
        _build_java_package("packages/array_binary_shard_java/build.ps1"),
    ),
    CommandSpec(
        "package.java.array_parquet",
        "Build array-sample-parquet Java jar/source/javadoc",
        _build_java_package("packages/array_sample_parquet_java/build.ps1"),
    ),
    CommandSpec(
        "package.java.scalar",
        "Build scalar-feature-shard Java jar/source/javadoc",
        _build_java_package("packages/scalar_feature_shard_java/build.ps1"),
    ),
    CommandSpec(
        "package.python.array_parquet.smoke",
        "Array sample parquet package smoke",
        lambda ctx: [str(PYTHON), str(ctx.repo_root / "packages" / "array_sample_parquet" / "tests" / "run_package_smoke.py")],
    ),
    CommandSpec(
        "example.python.scalar",
        "Scalar Python example",
        lambda ctx: [str(PYTHON), str(ctx.repo_root / "packages" / "scalar_feature_shard" / "examples" / "build_scalar_dense_long_example.py")],
    ),
    CommandSpec(
        "example.python.array_parquet",
        "Array sample parquet Python example",
        lambda ctx: [str(PYTHON), str(ctx.repo_root / "packages" / "array_sample_parquet" / "examples" / "build_array_sample_parquet_example.py")],
    ),
    CommandSpec("example.java.scalar.compile", "Compile scalar jar examples", _compile_scalar_jar_example),
    CommandSpec("example.java.scalar.run", "Run scalar jar example", _run_scalar_jar_example),
    CommandSpec("example.java.array_parquet.compile", "Compile array sample parquet jar examples", _compile_array_sample_jar_example),
    CommandSpec("example.java.array_parquet.run", "Run array sample parquet jar example", _run_array_sample_jar_example),
    CommandSpec("example.java.api.run", "Run Java value API examples against Python server", _run_api_jar_examples),
    CommandSpec("example.java.api.scalar_reader", "Validate scalar Java API example output with reader", _scalar_api_jar_reader_smoke),
    CommandSpec("example.java.api.array_validate", "Validate array Java API example output exhaustively", _array_api_jar_validate),
)


CHECKS: tuple[CheckSpec, ...] = (
    CheckSpec("1.1", "Scalar raw sample parquet 기준 final parts 전수 비교", ("scalar.python.package", "scalar.python_java.parity")),
    CheckSpec("1.2", "Scalar raw에 없는 cell은 mask=0, value=NaN", ("scalar.python.package", "scalar.python.api", "scalar.java.builder")),
    CheckSpec("1.3", "Scalar feature_locator part/offset/first_row 검증", ("scalar.python.package", "scalar.python_java.parity")),
    CheckSpec("1.4", "Scalar manifest range/row count/coverage gap-overlap 검증", ("scalar.python.package", "scalar.python_java.parity")),
    CheckSpec("2.1", "n_y_overlap == count(mask=1 AND finite y)", ("scalar.python_java.parity", "scalar.java.builder")),
    CheckSpec("2.2", "r2y를 final shard value/mask와 sample_meta Y로 재계산", ("scalar.python_java.parity", "scalar.java.builder")),
    CheckSpec("2.3", "Y null/NaN/constant/partial missing 케이스", ("scalar.python_java.parity", "scalar.python.selection_api")),
    CheckSpec("2.4", "invalid r2 case는 0.0", ("scalar.python_java.parity", "scalar.java.builder")),
    CheckSpec("2.5", "/scalar/top-features와 reader topFeaturesFromStats 일치", ("scalar.python.api", "scalar.python.selection_api")),
    CheckSpec("3.1", "같은 random sparse 입력으로 Python/Java builder 실행", ("scalar.python_java.parity",)),
    CheckSpec("3.2", "각 builder 산출물을 자기 raw parquet와 직접 비교", ("scalar.python_java.parity",)),
    CheckSpec("3.3", "Python/Java final parts, locator, selection_stats 비교", ("scalar.python_java.parity",)),
    CheckSpec("3.4", "None/NaN/missing 처리 기준 Python/Java 일치", ("scalar.python_java.parity", "scalar.java.builder")),
    CheckSpec("4.1", "Scalar reader load_feature/load_sample 결과와 final parquet 일치", ("scalar.python.api",)),
    CheckSpec("4.2", "/scalar/features, /scalar/sample, /scalar/top-features와 reader 일치", ("scalar.python.api", "scalar.python.selection_api")),
    CheckSpec("4.3", "Scalar id/key 중 하나만 허용, 둘 다 주면 에러", ("scalar.python.api",)),
    CheckSpec("4.4", "Scalar 없는 key, 범위 밖 id, max_cells, max_features 제한", ("scalar.python.api",)),
    CheckSpec("5.1", "Scalar 순차/랜덤 sample write 후 final shard가 raw와 일치", ("scalar.python.package", "scalar.python.concurrency")),
    CheckSpec("5.2", "Scalar 여러 worker 동시 write 후 commit/state/manifest 보존", ("scalar.python.concurrency", "scalar.java.concurrency")),
    CheckSpec("5.3", "Scalar 중단 후 재개 completed/pending 확인", ("scalar.python.package", "scalar.python.concurrency", "scalar.java.concurrency")),
    CheckSpec("5.4", "Scalar skip_if_completed=True overwrite 방지", ("scalar.python.package", "scalar.java.builder")),
    CheckSpec("5.5", "Scalar final parquet 존재/commit log 누락 시 reconcile 복구", ("scalar.python.raw_recovery", "scalar.java.raw_recovery")),
    CheckSpec("5.6", "Scalar finish/build stage lock과 active sample lock 경계 검증", ("scalar.python.raw_recovery", "scalar.java.raw_recovery")),
    CheckSpec("6.1", "Array sample parquet raw sample files 기준 compact final 전수 비교", ("array.parquet.python", "array.parquet.python.concurrency")),
    CheckSpec("6.2", "Array trace_index_parts row range 검증", ("array.parquet.python",)),
    CheckSpec("6.3", "Array part row count, trace count, sort order 검증", ("array.parquet.python", "array.parquet.python.concurrency")),
    CheckSpec("6.4", "Array missing/empty/NaN/categorical point column 케이스", ("array.parquet.python",)),
    CheckSpec("6.5", "Array sample/feature metadata dense id-key mapping", ("array.parquet.python", "array.parquet.java")),
    CheckSpec("7.1", "Array reader get_traces 결과와 final parquet 일치", ("array.parquet.python", "array.parquet.java")),
    CheckSpec("7.2", "/array-sample-parquet/traces 응답과 reader 직접 호출 일치", ("array.parquet.python",)),
    CheckSpec("7.3", "Array id/key 중 하나만 허용, 둘 다 주면 에러", ("array.parquet.python",)),
    CheckSpec("7.4", "include_missing/decode_categorical/layout/max_traces 동작", ("array.parquet.python",)),
    CheckSpec("7.5", "Array multi-sample/multi-feature trace 수와 point 값", ("array.parquet.python",)),
    CheckSpec("8.1", "Array raw sample 순차/랜덤 생성 후 compact 결과가 raw와 일치", ("array.parquet.python", "array.parquet.python.concurrency")),
    CheckSpec("8.2", "Array 여러 worker 동시 write 후 commit/state/manifest 보존", ("array.parquet.python.concurrency",)),
    CheckSpec("8.3", "Array 중단 후 재개 completed/pending 확인", ("array.parquet.python", "array.parquet.python.concurrency")),
    CheckSpec("8.4", "Array worker별 categorical 값 보존", ("array.parquet.python.concurrency",)),
    CheckSpec("9.1", "Array binary bundle parquet 기준 reader 결과 직접 비교", ("array.binary.python.storage",)),
    CheckSpec("9.2", "Array binary time/value/ch_step/기타 point columns 비교", ("array.binary.python.storage", "array.binary.java.builder")),
    CheckSpec("9.3", "Array binary trace flags 검증", ("array.binary.python.storage", "array.binary.java.v3")),
    CheckSpec("9.4", "Array binary empty/missing/NaN/categorical trace 검증", ("array.binary.python.storage", "array.binary.java.v3")),
    CheckSpec("9.5", "Array binary blocks.idx/bin header/offset/length/checksum 검증", ("array.binary.python.storage",)),
    CheckSpec("9.6", "Array binary shard/bucket/feature partition coverage", ("array.binary.python.storage",)),
    CheckSpec("10.1", "Array binary reader 결과가 bundle 기준 데이터와 일치", ("array.binary.python.storage", "array.binary.java.builder")),
    CheckSpec("10.2", "Array binary/API 경로와 reader 직접 호출 비교", ("array.binary.python.api",)),
    CheckSpec("11.1", "Array sample parquet와 custom binary의 missing/empty 의미 일관", ("array.parquet_binary.parity",)),
    CheckSpec("11.2", "Array sample parquet와 custom binary point 값/NaN/ch_step 비교", ("array.parquet_binary.parity",)),
    CheckSpec(
        "12.1",
        "Python wheel/sdist 최신 코드 포함",
        ("package.python.array_binary", "package.python.array_parquet", "package.python.scalar"),
    ),
    CheckSpec(
        "12.2",
        "Java jar/sources/javadoc 최신 코드 포함",
        ("package.java.array_binary", "package.java.array_parquet", "package.java.scalar"),
    ),
    CheckSpec(
        "12.3",
        "Python/Java 예제가 현재 API와 산출물 경로로 실행",
        (
            "package.python.array_parquet.smoke",
            "example.python.scalar",
            "example.python.array_parquet",
            "example.java.scalar.compile",
            "example.java.scalar.run",
            "example.java.array_parquet.compile",
            "example.java.array_parquet.run",
            "example.java.api.run",
            "example.java.api.scalar_reader",
            "example.java.api.array_validate",
        ),
    ),
    CheckSpec("12.4", "README/docs 포맷/config/API 설명과 실제 동작 일치", ("scalar.python.api", "array.parquet.python", "array.binary.python.api")),
    CheckSpec("12.5", "생성 산출물 이름과 표준 경로 일치", ("scalar.python.package", "array.parquet.python", "array.binary.python.storage")),
)


def _command_enabled(spec: CommandSpec, ctx: SuiteContext) -> bool:
    if spec.command_id.startswith("package.") and not ctx.include_packaging:
        return False
    if spec.command_id.startswith("example.") and not ctx.include_packaging:
        return False
    if spec.command_id.startswith("example.java.api.") and not ctx.include_api_examples:
        return False
    return spec.enabled_by_default


def _run_command(spec: CommandSpec, ctx: SuiteContext) -> dict:
    argv = spec.argv(ctx)
    started = time.perf_counter()
    completed = subprocess.run(argv, cwd=str(ctx.repo_root), text=True, capture_output=True)
    elapsed = time.perf_counter() - started
    return {
        "id": spec.command_id,
        "title": spec.title,
        "argv": argv,
        "returncode": completed.returncode,
        "status": "pass" if completed.returncode == 0 else "fail",
        "elapsed_sec": round(elapsed, 3),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _status_for_check(check: CheckSpec, command_results: dict[str, dict], skipped: set[str]) -> str:
    if any(command_id in skipped for command_id in check.command_ids):
        return "skip"
    if all(command_results.get(command_id, {}).get("status") == "pass" for command_id in check.command_ids):
        return "pass"
    return "fail"


def _tail(text: str, max_lines: int = 8) -> str:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return "\n".join(lines[-max_lines:])


def _write_reports(ctx: SuiteContext, payload: dict) -> tuple[Path, Path]:
    json_path = ctx.out_dir / "full_integrity_report.json"
    md_path = ctx.out_dir / "full_integrity_report.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Full Integrity Suite Report")
    lines.append("")
    lines.append(f"- generated_at: `{payload['generated_at']}`")
    lines.append(f"- overall_status: `{payload['overall_status']}`")
    lines.append(f"- elapsed_sec: `{payload['elapsed_sec']}`")
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append("| status | id | check | evidence commands |")
    lines.append("| --- | --- | --- | --- |")
    for check in payload["checks"]:
        lines.append(
            f"| `{check['status']}` | `{check['id']}` | {check['title']} | "
            f"{', '.join('`' + cid + '`' for cid in check['commands'])} |"
        )
    lines.append("")
    lines.append("## Commands")
    lines.append("")
    lines.append("| status | id | sec | command |")
    lines.append("| --- | --- | ---: | --- |")
    for command in payload["commands"]:
        command_text = " ".join(command["argv"]) if command.get("argv") else ""
        command_text = command_text.replace("|", "\\|")
        lines.append(
            f"| `{command['status']}` | `{command['id']}` | {command.get('elapsed_sec', '')} | "
            f"`{command_text}` |"
        )
    lines.append("")
    lines.append("## Command Output Tails")
    for command in payload["commands"]:
        lines.append("")
        lines.append(f"### `{command['id']}`")
        if command["status"] == "skip":
            lines.append("")
            lines.append("Skipped.")
            continue
        stdout_tail = _tail(command.get("stdout", ""))
        stderr_tail = _tail(command.get("stderr", ""))
        if stdout_tail:
            lines.append("")
            lines.append("stdout:")
            lines.append("```text")
            lines.append(stdout_tail)
            lines.append("```")
        if stderr_tail:
            lines.append("")
            lines.append("stderr:")
            lines.append("```text")
            lines.append(stderr_tail)
            lines.append("```")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def run_suite(args: argparse.Namespace) -> int:
    out_dir = (REPO_ROOT / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    if args.clean_report_dir and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = SuiteContext(
        repo_root=REPO_ROOT,
        out_dir=out_dir,
        java_exe=args.java_exe or _tool_from_java_home("java"),
        javac_exe=args.javac_exe or _tool_from_java_home("javac"),
        include_packaging=not args.skip_packaging,
        include_api_examples=not args.skip_api_examples,
    )

    command_results: dict[str, dict] = {}
    skipped: set[str] = set()
    suite_started = time.perf_counter()
    for spec in COMMANDS:
        if not _command_enabled(spec, ctx):
            skipped.add(spec.command_id)
            command_results[spec.command_id] = {
                "id": spec.command_id,
                "title": spec.title,
                "argv": [],
                "returncode": None,
                "status": "skip",
                "elapsed_sec": 0.0,
                "stdout": "",
                "stderr": "",
            }
            continue
        print(f"[RUN] {spec.command_id}: {spec.title}", flush=True)
        result = _run_command(spec, ctx)
        command_results[spec.command_id] = result
        print(f"[{result['status'].upper()}] {spec.command_id} ({result['elapsed_sec']:.3f}s)", flush=True)
        if result["status"] != "pass" and args.fail_fast:
            break

    checks = []
    for check in CHECKS:
        status = _status_for_check(check, command_results, skipped)
        checks.append(
            {
                "id": check.check_id,
                "title": check.title,
                "commands": list(check.command_ids),
                "status": status,
            }
        )

    command_list = [command_results[spec.command_id] for spec in COMMANDS if spec.command_id in command_results]
    overall_status = "pass"
    if any(command["status"] == "fail" for command in command_list) or any(check["status"] == "fail" for check in checks):
        overall_status = "fail"
    elif any(check["status"] == "skip" for check in checks):
        overall_status = "partial"

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "overall_status": overall_status,
        "elapsed_sec": round(time.perf_counter() - suite_started, 3),
        "include_packaging": ctx.include_packaging,
        "include_api_examples": ctx.include_api_examples,
        "commands": command_list,
        "checks": checks,
    }
    json_path, md_path = _write_reports(ctx, payload)
    print(f"report_json={json_path}")
    print(f"report_markdown={md_path}")
    print(f"overall_status={overall_status}")
    return 0 if overall_status == "pass" else 1


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full scalar/array integrity checklist and write evidence reports.")
    parser.add_argument("--out-dir", default="data/full_integrity_suite", help="Report output directory.")
    parser.add_argument("--clean-report-dir", action="store_true", help="Delete the report directory before running.")
    parser.add_argument("--skip-packaging", action="store_true", help="Skip wheel/jar rebuild and package examples.")
    parser.add_argument("--skip-api-examples", action="store_true", help="Skip Java examples that start the Python value API server.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop at the first failed command.")
    parser.add_argument("--java-exe", default="", help="Override java executable path.")
    parser.add_argument("--javac-exe", default="", help="Override javac executable path.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    return run_suite(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
