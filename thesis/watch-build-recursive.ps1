$ErrorActionPreference = 'Stop'

$root = $PSScriptRoot
$env:Path = "C:\Strawberry\perl\bin;" + $env:Path
Set-Location $root

$buildDir = Join-Path $root 'build'
New-Item -ItemType Directory -Force -Path $buildDir | Out-Null

$lastRun = [datetime]::UtcNow
$debounceMs = 400

function Invoke-LatexBuild {
    Write-Host "[$([DateTime]::Now.ToString('HH:mm:ss'))] Recompilando LaTeX..."
    latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build -auxdir=build main.tex
    Write-Host "[$([DateTime]::Now.ToString('HH:mm:ss'))] Compilación terminada."
}

$watcher = New-Object System.IO.FileSystemWatcher
$watcher.Path = $root
$watcher.Filter = '*.tex'
$watcher.IncludeSubdirectories = $true
$watcher.EnableRaisingEvents = $true

$eventAction = {
    $now = [DateTime]::UtcNow
    if (($now - $global:lastRun).TotalMilliseconds -lt $global:debounceMs) {
        return
    }
    $global:lastRun = $now
    Invoke-LatexBuild
}

Register-ObjectEvent -InputObject $watcher -EventName Changed -Action $eventAction | Out-Null
Register-ObjectEvent -InputObject $watcher -EventName Created -Action $eventAction | Out-Null
Register-ObjectEvent -InputObject $watcher -EventName Renamed -Action $eventAction | Out-Null

Write-Host "Observando cambios en $root ..."
Write-Host "LaTeX se recompilará al guardar cualquier archivo .tex dentro de la tesis."
Write-Host "Presiona Ctrl+C para detener."

while ($true) {
    Start-Sleep -Seconds 1
}
