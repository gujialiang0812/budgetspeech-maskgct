param(
    [string]$Root = "E:\BudgetSpeechDatasets",
    [ValidateSet("paper_core", "topconf_eval", "optional_scaling", "all")]
    [string]$Preset = "paper_core",
    [switch]$DownloadOnly,
    [switch]$Extract
)

$ErrorActionPreference = "Stop"

$manifestPath = Join-Path (Split-Path -Parent $PSScriptRoot) "data\datasets_manifest.json"
$manifest = Get-Content -Raw -Path $manifestPath | ConvertFrom-Json

if ($Preset -eq "all") {
    $items = @($manifest.paper_core) + @($manifest.topconf_eval) + @($manifest.optional_scaling)
} else {
    $items = @($manifest.$Preset)
}

$archiveDir = Join-Path $Root "archives"
$extractRoot = Join-Path $Root "extracted"
New-Item -ItemType Directory -Force -Path $archiveDir, $extractRoot | Out-Null

function Download-File {
    param(
        [string]$Url,
        [string]$OutFile,
        [Int64]$ExpectedBytes = 0
    )

    if (Test-Path $OutFile) {
        $size = (Get-Item $OutFile).Length
        if ($ExpectedBytes -gt 0 -and $size -ge $ExpectedBytes) {
            Write-Host "Archive already complete: $OutFile"
            return
        } elseif ($size -gt 0) {
            Write-Host "Found partial archive, resuming: $OutFile ($size bytes)"
        }
    }

    Write-Host "Downloading $Url"
    Write-Host " -> $OutFile"
    & curl.exe -L --fail --retry 5 --retry-delay 10 --continue-at - --output $OutFile $Url
    if ($LASTEXITCODE -ne 0) {
        throw "curl failed for $Url"
    }
}

function Extract-ArchiveFile {
    param(
        [string]$ArchivePath,
        [string]$Destination
    )

    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    if ($ArchivePath.EndsWith(".zip")) {
        Write-Host "Extracting ZIP $ArchivePath"
        Expand-Archive -Path $ArchivePath -DestinationPath $Destination -Force
    } elseif ($ArchivePath.EndsWith(".tar.gz") -or $ArchivePath.EndsWith(".tgz")) {
        Write-Host "Extracting TAR $ArchivePath"
        & tar -xzf $ArchivePath -C $Destination
        if ($LASTEXITCODE -ne 0) {
            throw "tar extraction failed for $ArchivePath"
        }
    } else {
        throw "Unknown archive format: $ArchivePath"
    }
}

function Get-DirectorySize {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        return 0
    }
    $total = 0
    Get-ChildItem -LiteralPath $Path -Recurse -File | ForEach-Object {
        $total += $_.Length
    }
    return $total
}

$summary = @()
foreach ($item in $items) {
    if ($item.PSObject.Properties.Name -contains "repo_id") {
        $localDir = Join-Path $Root $item.local_dir
        if (-not (Get-Command huggingface-cli -ErrorAction SilentlyContinue)) {
            throw "huggingface-cli is required for $($item.name). Install huggingface_hub or download $($item.repo_id) manually."
        }
        New-Item -ItemType Directory -Force -Path $localDir | Out-Null
        Write-Host "Downloading Hugging Face dataset $($item.repo_id)"
        Write-Host " -> $localDir"
        & huggingface-cli download $item.repo_id --repo-type dataset --local-dir $localDir
        if ($LASTEXITCODE -ne 0) {
            throw "huggingface-cli failed for $($item.repo_id)"
        }
        $summary += [PSCustomObject]@{
            id = $item.id
            archive = $localDir
            sizeGB = [math]::Round((Get-DirectorySize -Path $localDir) / 1GB, 3)
            source = $item.source
        }
        continue
    }

    $archivePath = Join-Path $archiveDir $item.archive
    $expectedBytes = 0
    if ($item.PSObject.Properties.Name -contains "bytes") {
        $expectedBytes = [Int64]$item.bytes
    }
    Download-File -Url $item.url -OutFile $archivePath -ExpectedBytes $expectedBytes

    if ($expectedBytes -gt 0) {
        $actualBytes = (Get-Item $archivePath).Length
        if ($actualBytes -lt $expectedBytes) {
            throw "Incomplete archive: $archivePath ($actualBytes / $expectedBytes bytes)"
        }
    }

    if ($Extract -and -not $DownloadOnly) {
        $dest = Join-Path $extractRoot $item.extract_dir
        Extract-ArchiveFile -ArchivePath $archivePath -Destination $dest
    }

    $summary += [PSCustomObject]@{
        id = $item.id
        archive = $archivePath
        sizeGB = [math]::Round((Get-Item $archivePath).Length / 1GB, 3)
        source = $item.source
    }
}

$summaryPath = Join-Path $Root "download_summary.json"
$summary | ConvertTo-Json -Depth 4 | Set-Content -Path $summaryPath -Encoding UTF8
Write-Host "Done. Summary written to $summaryPath"
$summary | Format-Table -AutoSize
