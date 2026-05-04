$ErrorActionPreference = "Stop"

$packageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent (Split-Path -Parent $packageRoot)
$version = "0.3.0"
$artifactName = "array-binary-shard-java-$version.jar"

$jdkBin = "C:\Program Files\Java\jdk-1.8\bin"
$javac = Join-Path $jdkBin "javac.exe"
$jar = Join-Path $jdkBin "jar.exe"

$duckdbJar = Join-Path $repoRoot "java\lib\duckdb_jdbc-1.1.3.jar"
if (-not (Test-Path $javac)) {
    throw "javac.exe not found at $javac"
}
if (-not (Test-Path $jar)) {
    throw "jar.exe not found at $jar"
}
if (-not (Test-Path $duckdbJar)) {
    throw "duckdb jdbc jar not found at $duckdbJar. Run: powershell -ExecutionPolicy Bypass -File java\\download_duckdb_jdbc.ps1"
}

$buildDir = Join-Path $packageRoot "build"
$classesDir = Join-Path $buildDir "classes"
$distDir = Join-Path $packageRoot "dist"
$manifestFile = Join-Path $buildDir "jar-manifest.mf"

Remove-Item -Recurse -Force $buildDir -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force $distDir -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path $classesDir | Out-Null
New-Item -ItemType Directory -Path $distDir | Out-Null

$sources = @(
    "java\src\fs\config\ArrayBundleConfig.java",
    "java\src\fs\config\ArrayShardConfig.java",
    "java\src\fs\io\array\ArrayBinaryFormat.java",
    "java\src\fs\io\ArrayBinaryShardReader.java",
    "java\src\fs\io\ArrayBinaryShards.java",
    "java\src\fs\io\array\ArrayBundleManifestIO.java",
    "java\src\fs\io\ArrayDatasetBuilder.java",
    "java\src\fs\io\array\ArrayFeatureFlags.java",
    "java\src\fs\io\array\ArrayFeatureIdIndex.java",
    "java\src\fs\io\array\ArrayFeatureLocatorIndex.java",
    "java\src\fs\io\common\JsonUtils.java",
    "java\src\fs\io\common\ArrayMetadataWriter.java",
    "java\src\fs\io\array\ArraySampleBundleWriter.java",
    "java\src\fs\io\array\ArraySampleIdIndex.java",
    "java\src\fs\io\array\ArrayShardBuilder.java",
    "java\src\fs\io\array\ArrayShardManifestIO.java",
    "java\src\fs\io\common\ArrayUtils.java",
    "java\src\fs\io\common\DuckDBUtils.java",
    "java\src\fs\model\array\ArrayBinaryShardInfo.java",
    "java\src\fs\model\array\ArrayBlockLocation.java",
    "java\src\fs\model\array\ArrayBundleManifest.java",
    "java\src\fs\model\array\ArrayFeatureBlock.java",
    "java\src\fs\model\array\ArrayShardManifest.java",
    "java\src\fs\model\array\ArrayTrace.java",
    "java\src\fs\model\common\LogicalType.java",
    "java\src\fs\model\common\PointColumnSpec.java",
    "java\src\fs\model\common\StorageType.java"
) | ForEach-Object { Join-Path $repoRoot $_ }

& $javac -encoding UTF-8 -cp $duckdbJar -d $classesDir $sources
if ($LASTEXITCODE -ne 0) {
    throw "javac failed with exit code $LASTEXITCODE"
}

@(
    "Manifest-Version: 1.0"
    "Automatic-Module-Name: array.binary.shard"
    "Implementation-Title: array-binary-shard-java"
    "Implementation-Version: $version"
    ""
) | Set-Content -Path $manifestFile -Encoding Ascii

$jarPath = Join-Path $distDir $artifactName
& $jar cfm $jarPath $manifestFile -C $classesDir .
if ($LASTEXITCODE -ne 0) {
    throw "jar packaging failed with exit code $LASTEXITCODE"
}

Write-Output $jarPath
