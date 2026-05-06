$ErrorActionPreference = "Stop"

$libDir = Join-Path $PSScriptRoot "lib"
New-Item -ItemType Directory -Force -Path $libDir | Out-Null

$dependencies = @(
    @{
        Name = "duckdb_jdbc-1.1.3.jar"
        Url = "https://repo1.maven.org/maven2/org/duckdb/duckdb_jdbc/1.1.3/duckdb_jdbc-1.1.3.jar"
    },
    @{
        Name = "jackson-core-2.20.0.jar"
        Url = "https://repo1.maven.org/maven2/com/fasterxml/jackson/core/jackson-core/2.20.0/jackson-core-2.20.0.jar"
    },
    @{
        Name = "jackson-databind-2.20.0.jar"
        Url = "https://repo1.maven.org/maven2/com/fasterxml/jackson/core/jackson-databind/2.20.0/jackson-databind-2.20.0.jar"
    },
    @{
        Name = "jackson-annotations-2.20.jar"
        Url = "https://repo1.maven.org/maven2/com/fasterxml/jackson/core/jackson-annotations/2.20/jackson-annotations-2.20.jar"
    }
)

foreach ($dependency in $dependencies) {
    $jarPath = Join-Path $libDir $dependency.Name
    if (Test-Path $jarPath) {
        Write-Output "Java dependency already exists: $jarPath"
        continue
    }
    Write-Output "Downloading $($dependency.Url)"
    Invoke-WebRequest -Uri $dependency.Url -OutFile $jarPath
    Write-Output "Saved Java dependency to $jarPath"
}
