$ErrorActionPreference = "Stop"

$version = "1.1.3"
$repoRoot = Split-Path -Parent $PSScriptRoot
$libDir = Join-Path $PSScriptRoot "lib"
$jarName = "duckdb_jdbc-$version.jar"
$jarPath = Join-Path $libDir $jarName
$url = "https://repo1.maven.org/maven2/org/duckdb/duckdb_jdbc/$version/$jarName"

New-Item -ItemType Directory -Force -Path $libDir | Out-Null

if (Test-Path $jarPath) {
    Write-Output "DuckDB JDBC jar already exists: $jarPath"
    exit 0
}

Write-Output "Downloading $url"
Invoke-WebRequest -Uri $url -OutFile $jarPath
Write-Output "Saved DuckDB JDBC jar to $jarPath"
