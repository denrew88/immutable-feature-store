$ErrorActionPreference = "Stop"

$scriptPath = Join-Path $PSScriptRoot "download_java_libs.ps1"
& $scriptPath
