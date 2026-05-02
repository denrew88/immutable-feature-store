$ErrorActionPreference = "Stop"

$packageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent (Split-Path -Parent $packageRoot)
$version = "0.1.0"
$artifactName = "scalar-feature-shard-java-$version.jar"

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
    "java\src\fs\config\BuildShardConfig.java",
    "java\src\fs\config\SelectionConfig.java",
    "java\src\fs\io\ArrayFeatureIdIndex.java",
    "java\src\fs\io\ArrayMetadataWriter.java",
    "java\src\fs\io\ArraySampleIdIndex.java",
    "java\src\fs\io\ArrayUtils.java",
    "java\src\fs\io\DuckDBShardReader.java",
    "java\src\fs\io\DuckDBUtils.java",
    "java\src\fs\io\FeatureIdIndex.java",
    "java\src\fs\io\FeatureLocatorIndex.java",
    "java\src\fs\io\ManifestIO.java",
    "java\src\fs\io\SampleIdIndex.java",
    "java\src\fs\io\SampleMetaLoader.java",
    "java\src\fs\io\ScalarDatasetBuilder.java",
    "java\src\fs\io\ScalarFeatureShards.java",
    "java\src\fs\io\ScalarMetadataWriter.java",
    "java\src\fs\io\ScalarSampleBundleManifestIO.java",
    "java\src\fs\io\ScalarSampleBundleWriter.java",
    "java\src\fs\io\ScalarShardDataset.java",
    "java\src\fs\io\ShardBuilder.java",
    "java\src\fs\io\ShardReader.java",
    "java\src\fs\math\Pearson.java",
    "java\src\fs\model\Candidate.java",
    "java\src\fs\model\Feature.java",
    "java\src\fs\model\FeatureLocation.java",
    "java\src\fs\model\LogicalType.java",
    "java\src\fs\model\PointColumnSpec.java",
    "java\src\fs\model\RowBatch.java",
    "java\src\fs\model\SampleMeta.java",
    "java\src\fs\model\ScalarFeatureValues.java",
    "java\src\fs\model\ScalarSampleBundleManifest.java",
    "java\src\fs\model\ScalarValue.java",
    "java\src\fs\model\ShardManifest.java",
    "java\src\fs\model\StorageType.java",
    "java\src\fs\pipeline\CandidateBuilder.java",
    "java\src\fs\pipeline\Selector.java"
) | ForEach-Object { Join-Path $repoRoot $_ }

& $javac -cp $duckdbJar -d $classesDir $sources
if ($LASTEXITCODE -ne 0) {
    throw "javac failed with exit code $LASTEXITCODE"
}

@(
    "Manifest-Version: 1.0"
    "Automatic-Module-Name: scalar.feature.shard"
    "Implementation-Title: scalar-feature-shard-java"
    "Implementation-Version: $version"
    ""
) | Set-Content -Path $manifestFile -Encoding Ascii

$jarPath = Join-Path $distDir $artifactName
& $jar cfm $jarPath $manifestFile -C $classesDir .
if ($LASTEXITCODE -ne 0) {
    throw "jar packaging failed with exit code $LASTEXITCODE"
}

Write-Output $jarPath
