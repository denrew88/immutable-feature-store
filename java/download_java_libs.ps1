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
    },
    @{
        Name = "parquet-hadoop-bundle-1.13.1.jar"
        Url = "https://repo1.maven.org/maven2/org/apache/parquet/parquet-hadoop-bundle/1.13.1/parquet-hadoop-bundle-1.13.1.jar"
    },
    @{
        Name = "hadoop-common-3.3.6.jar"
        Url = "https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-common/3.3.6/hadoop-common-3.3.6.jar"
    },
    @{
        Name = "slf4j-api-1.7.36.jar"
        Url = "https://repo1.maven.org/maven2/org/slf4j/slf4j-api/1.7.36/slf4j-api-1.7.36.jar"
    },
    @{
        Name = "woodstox-core-6.5.1.jar"
        Url = "https://repo1.maven.org/maven2/com/fasterxml/woodstox/woodstox-core/6.5.1/woodstox-core-6.5.1.jar"
    },
    @{
        Name = "stax2-api-4.2.1.jar"
        Url = "https://repo1.maven.org/maven2/org/codehaus/woodstox/stax2-api/4.2.1/stax2-api-4.2.1.jar"
    },
    @{
        Name = "commons-collections-3.2.2.jar"
        Url = "https://repo1.maven.org/maven2/commons-collections/commons-collections/3.2.2/commons-collections-3.2.2.jar"
    },
    @{
        Name = "commons-lang3-3.12.0.jar"
        Url = "https://repo1.maven.org/maven2/org/apache/commons/commons-lang3/3.12.0/commons-lang3-3.12.0.jar"
    },
    @{
        Name = "hadoop-mapreduce-client-core-3.3.6.jar"
        Url = "https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-mapreduce-client-core/3.3.6/hadoop-mapreduce-client-core-3.3.6.jar"
    },
    @{
        Name = "arrow-c-data-14.0.2.jar"
        Url = "https://repo1.maven.org/maven2/org/apache/arrow/arrow-c-data/14.0.2/arrow-c-data-14.0.2.jar"
    },
    @{
        Name = "arrow-memory-core-14.0.2.jar"
        Url = "https://repo1.maven.org/maven2/org/apache/arrow/arrow-memory-core/14.0.2/arrow-memory-core-14.0.2.jar"
    },
    @{
        Name = "arrow-memory-unsafe-14.0.2.jar"
        Url = "https://repo1.maven.org/maven2/org/apache/arrow/arrow-memory-unsafe/14.0.2/arrow-memory-unsafe-14.0.2.jar"
    },
    @{
        Name = "arrow-vector-14.0.2-shade-format-flatbuffers.jar"
        Url = "https://repo1.maven.org/maven2/org/apache/arrow/arrow-vector/14.0.2/arrow-vector-14.0.2-shade-format-flatbuffers.jar"
    },
    @{
        Name = "netty-common-4.1.96.Final.jar"
        Url = "https://repo1.maven.org/maven2/io/netty/netty-common/4.1.96.Final/netty-common-4.1.96.Final.jar"
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
