$ErrorActionPreference = "Stop"

$packageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent (Split-Path -Parent $packageRoot)
$version = "0.1.0"
$artifactName = "scalar-feature-shard-java-$version.jar"
$sourcesArtifactName = "scalar-feature-shard-java-$version-sources.jar"
$javadocArtifactName = "scalar-feature-shard-java-$version-javadoc.jar"

$jdkBin = "C:\Program Files\Java\jdk-1.8\bin"
$javac = Join-Path $jdkBin "javac.exe"
$jar = Join-Path $jdkBin "jar.exe"
$javadoc = Join-Path $jdkBin "javadoc.exe"

$duckdbJar = Join-Path $repoRoot "java\lib\duckdb_jdbc-1.1.3.jar"
$jacksonCoreJar = Join-Path $repoRoot "java\lib\jackson-core-2.20.0.jar"
$jacksonDatabindJar = Join-Path $repoRoot "java\lib\jackson-databind-2.20.0.jar"
$jacksonAnnotationsJar = Join-Path $repoRoot "java\lib\jackson-annotations-2.20.jar"
if (-not (Test-Path $javac)) {
    throw "javac.exe not found at $javac"
}
if (-not (Test-Path $jar)) {
    throw "jar.exe not found at $jar"
}
if (-not (Test-Path $javadoc)) {
    throw "javadoc.exe not found at $javadoc"
}
if (-not (Test-Path $duckdbJar)) {
    throw "duckdb jdbc jar not found at $duckdbJar. Run: powershell -ExecutionPolicy Bypass -File java\\download_java_libs.ps1"
}
if (-not (Test-Path $jacksonCoreJar)) {
    throw "jackson-core jar not found at $jacksonCoreJar. Run: powershell -ExecutionPolicy Bypass -File java\\download_java_libs.ps1"
}
if (-not (Test-Path $jacksonDatabindJar)) {
    throw "jackson-databind jar not found at $jacksonDatabindJar. Run: powershell -ExecutionPolicy Bypass -File java\\download_java_libs.ps1"
}
if (-not (Test-Path $jacksonAnnotationsJar)) {
    throw "jackson-annotations jar not found at $jacksonAnnotationsJar. Run: powershell -ExecutionPolicy Bypass -File java\\download_java_libs.ps1"
}

$classpath = @(
    $duckdbJar
    $jacksonCoreJar
    $jacksonDatabindJar
    $jacksonAnnotationsJar
) -join ";"

$buildDir = Join-Path $packageRoot "build"
$classesDir = Join-Path $buildDir "classes"
$sourcesDir = Join-Path $buildDir "sources"
$javadocDir = Join-Path $buildDir "javadoc"
$distDir = Join-Path $packageRoot "dist"
$manifestFile = Join-Path $buildDir "jar-manifest.mf"

Remove-Item -Recurse -Force $buildDir -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force $distDir -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path $classesDir | Out-Null
New-Item -ItemType Directory -Path $sourcesDir | Out-Null
New-Item -ItemType Directory -Path $javadocDir | Out-Null
New-Item -ItemType Directory -Path $distDir | Out-Null

$sourceSpecs = @(
    "java\src\fs\config\BuildShardConfig.java",
    "java\src\fs\config\SelectionConfig.java",
    "java\src\fs\io\array\ArrayFeatureIdIndex.java",
    "java\src\fs\io\common\ArrayMetadataWriter.java",
    "java\src\fs\io\array\ArraySampleIdIndex.java",
    "java\src\fs\io\common\ArrayUtils.java",
    "java\src\fs\io\common\JsonUtils.java",
    "java\src\fs\io\common\DuckDBUtils.java",
    "java\src\fs\io\scalar\FeatureIdIndex.java",
    "java\src\fs\io\scalar\SampleIdIndex.java",
    "java\src\fs\io\scalar\SampleMetaLoader.java",
    "java\src\fs\io\ScalarDatasetBuilder.java",
    "java\src\fs\io\ScalarDenseLongDataset.java",
    "java\src\fs\io\ScalarFeatureShards.java",
    "java\src\fs\io\scalar\ScalarDenseLongManifestIO.java",
    "java\src\fs\io\scalar\ScalarDenseLongShardBuilder.java",
    "java\src\fs\io\scalar\ScalarFileLock.java",
    "java\src\fs\io\scalar\ScalarMetadataWriter.java",
    "java\src\fs\io\scalar\ScalarRawSampleWriter.java",
    "java\src\fs\io\scalar\ScalarSampleMajorManifestIO.java",
    "java\src\fs\io\scalar\ShardReader.java",
    "java\src\fs\math\Pearson.java",
    "java\src\fs\model\selection\Candidate.java",
    "java\src\fs\model\common\Feature.java",
    "java\src\fs\model\common\FeatureLocation.java",
    "java\src\fs\model\common\LogicalType.java",
    "java\src\fs\model\common\PointColumnSpec.java",
    "java\src\fs\model\scalar\RowBatch.java",
    "java\src\fs\model\common\SampleMeta.java",
    "java\src\fs\model\scalar\ScalarBuildSessionStatus.java",
    "java\src\fs\model\scalar\ScalarDenseLongManifest.java",
    "java\src\fs\model\scalar\ScalarDenseLongPart.java",
    "java\src\fs\model\scalar\ScalarFeatureValues.java",
    "java\src\fs\model\scalar\ScalarSampleMajorManifest.java",
    "java\src\fs\model\scalar\ScalarValue.java",
    "java\src\fs\model\common\StorageType.java",
    "java\src\fs\pipeline\CandidateBuilder.java",
    "java\src\fs\pipeline\Selector.java"
)

$sources = $sourceSpecs | ForEach-Object { Join-Path $repoRoot $_ }

& $javac -encoding UTF-8 -cp $classpath -d $classesDir $sources
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

foreach ($spec in $sourceSpecs) {
    $relativeSourcePath = $spec.Substring("java\src\".Length)
    $targetSourcePath = Join-Path $sourcesDir $relativeSourcePath
    $targetSourceParent = Split-Path -Parent $targetSourcePath
    if (-not (Test-Path $targetSourceParent)) {
        New-Item -ItemType Directory -Path $targetSourceParent -Force | Out-Null
    }
    Copy-Item -Path (Join-Path $repoRoot $spec) -Destination $targetSourcePath -Force
}

$sourcesJarPath = Join-Path $distDir $sourcesArtifactName
& $jar cf $sourcesJarPath -C $sourcesDir .
if ($LASTEXITCODE -ne 0) {
    throw "sources jar packaging failed with exit code $LASTEXITCODE"
}

& $javadoc `
    -encoding UTF-8 `
    -docencoding UTF-8 `
    -charset UTF-8 `
    -Xdoclint:none `
    -quiet `
    -cp $classpath `
    -d $javadocDir `
    $sources
if ($LASTEXITCODE -ne 0) {
    throw "javadoc generation failed with exit code $LASTEXITCODE"
}

$javadocJarPath = Join-Path $distDir $javadocArtifactName
& $jar cf $javadocJarPath -C $javadocDir .
if ($LASTEXITCODE -ne 0) {
    throw "javadoc jar packaging failed with exit code $LASTEXITCODE"
}

Write-Output $jarPath
Write-Output $sourcesJarPath
Write-Output $javadocJarPath
