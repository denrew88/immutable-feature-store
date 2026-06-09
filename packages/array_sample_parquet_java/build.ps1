$ErrorActionPreference = "Stop"

$packageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent (Split-Path -Parent $packageRoot)
$version = "0.1.0"
$artifactName = "array-sample-parquet-java-$version.jar"
$sourcesArtifactName = "array-sample-parquet-java-$version-sources.jar"
$javadocArtifactName = "array-sample-parquet-java-$version-javadoc.jar"

$jdkBin = "C:\Program Files\Java\jdk-1.8\bin"
$javac = Join-Path $jdkBin "javac.exe"
$jar = Join-Path $jdkBin "jar.exe"
$javadoc = Join-Path $jdkBin "javadoc.exe"

if (-not (Test-Path $javac)) { throw "javac.exe not found at $javac" }
if (-not (Test-Path $jar)) { throw "jar.exe not found at $jar" }
if (-not (Test-Path $javadoc)) { throw "javadoc.exe not found at $javadoc" }

$requiredJarNames = @(
    "duckdb_jdbc-1.1.3.jar",
    "jackson-core-2.20.0.jar",
    "jackson-databind-2.20.0.jar",
    "jackson-annotations-2.20.jar",
    "arrow-c-data-14.0.2.jar",
    "arrow-memory-core-14.0.2.jar",
    "arrow-memory-unsafe-14.0.2.jar",
    "arrow-vector-14.0.2-shade-format-flatbuffers.jar",
    "netty-common-4.1.96.Final.jar",
    "slf4j-api-1.7.36.jar"
)

$libJars = @()
foreach ($jarName in $requiredJarNames) {
    $jarPath = Join-Path $repoRoot ("java\lib\" + $jarName)
    if (-not (Test-Path $jarPath)) {
        throw "$jarName not found at $jarPath. Run: powershell -ExecutionPolicy Bypass -File java\\download_java_libs.ps1"
    }
    $libJars += $jarPath
}
$classpath = $libJars -join ";"

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
    "java\src\fs\io\ArraySampleParquets.java",
    "java\src\fs\io\array_sample_parquet\ArraySampleParquetCompactor.java",
    "java\src\fs\io\array_sample_parquet\ArraySampleParquetDatasetBuilder.java",
    "java\src\fs\io\array_sample_parquet\ArraySampleParquetDuckDB.java",
    "java\src\fs\io\array_sample_parquet\ArraySampleParquetManifestIO.java",
    "java\src\fs\io\array_sample_parquet\ArraySampleParquetOrderChecks.java",
    "java\src\fs\io\array_sample_parquet\ArraySampleParquetRawSampleWriter.java",
    "java\src\fs\io\array_sample_parquet\ArraySampleParquetReader.java",
    "java\src\fs\io\array_sample_parquet\ArraySampleParquetSampleContext.java",
    "java\src\fs\io\common\ArrayMetadataWriter.java",
    "java\src\fs\io\common\ArrayUtils.java",
    "java\src\fs\io\common\DuckDBUtils.java",
    "java\src\fs\io\common\FilePathLock.java",
    "java\src\fs\io\common\JsonUtils.java",
    "java\src\fs\model\array_sample_parquet\ArraySampleParquetBuildOptions.java",
    "java\src\fs\model\array_sample_parquet\ArraySampleParquetBuildSessionStatus.java",
    "java\src\fs\model\array_sample_parquet\ArraySampleParquetManifest.java",
    "java\src\fs\model\array_sample_parquet\ArraySampleParquetPart.java",
    "java\src\fs\model\array_sample_parquet\ArraySampleParquetTrace.java",
    "java\src\fs\model\common\LogicalType.java",
    "java\src\fs\model\common\PointColumnSpec.java",
    "java\src\fs\model\common\StorageType.java"
)

$sources = $sourceSpecs | ForEach-Object { Join-Path $repoRoot $_ }

& $javac -encoding UTF-8 -cp $classpath -d $classesDir $sources
if ($LASTEXITCODE -ne 0) { throw "javac failed with exit code $LASTEXITCODE" }

@(
    "Manifest-Version: 1.0"
    "Automatic-Module-Name: array.sample.parquet"
    "Implementation-Title: array-sample-parquet-java"
    "Implementation-Version: $version"
    ""
) | Set-Content -Path $manifestFile -Encoding Ascii

$jarPath = Join-Path $distDir $artifactName
& $jar cfm $jarPath $manifestFile -C $classesDir .
if ($LASTEXITCODE -ne 0) { throw "jar packaging failed with exit code $LASTEXITCODE" }

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
if ($LASTEXITCODE -ne 0) { throw "sources jar packaging failed with exit code $LASTEXITCODE" }

& $javadoc `
    -encoding UTF-8 `
    -docencoding UTF-8 `
    -charset UTF-8 `
    -quiet `
    -cp $classpath `
    -d $javadocDir `
    $sources
if ($LASTEXITCODE -ne 0) { throw "javadoc generation failed with exit code $LASTEXITCODE" }

$javadocJarPath = Join-Path $distDir $javadocArtifactName
& $jar cf $javadocJarPath -C $javadocDir .
if ($LASTEXITCODE -ne 0) { throw "javadoc jar packaging failed with exit code $LASTEXITCODE" }

Write-Output $jarPath
Write-Output $sourcesJarPath
Write-Output $javadocJarPath
