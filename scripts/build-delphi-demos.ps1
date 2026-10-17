<#
  Baut die Demos dieser Extension mit Delphi (dcc64, Win64).
  Pendant zu scripts/build-demo.sh (FPC); dieselbe Demo-Liste.

  FPC ist Referenz-Compiler. Dieses Skript prueft, dass die Demos auch mit
  Delphi 13 (Compiler-Version 37.0) uebersetzen - die Tests deckt
  run-delphi-tests.ps1 ab.

  Beispiele:
    .\scripts\build-delphi-demos.ps1                       # alle Demos
    .\scripts\build-delphi-demos.ps1 -Demo infer\llm-bench.lpr
    .\scripts\build-delphi-demos.ps1 -ShowAll              # voller Compilerlog
    .\scripts\build-delphi-demos.ps1 -Dcc 'C:\...\dcc64.exe'   # expliziter Compiler, ohne Versionspruefung

  mORMot2: $env:MORMOT2_ROOT, sonst das Submodul-Layout des Konsumenten
  (<konsument>\shared\delphi\libs\_git_Synopse2, drei Ebenen ueber diesem Repo).
  Ausgabe: bin\delphi\demos\ (gitignored). Die .lpr-Programme werden dort als
  .dpr abgelegt - dcc akzeptiert nur eine .dpr als Projektdatei.
#>
[CmdletBinding()]
param(
  # eine Demo, Pfad relativ zu demos\ (z. B. infer\llm-bench.lpr); leer = alle
  [string]$Demo = '',
  # Pfad zu dcc64.exe; gesetzt = ohne Versionspruefung (bewusste Gegenprobe)
  [string]$Dcc = $env:DCC64,
  [switch]$ShowAll
)

$ErrorActionPreference = 'Stop'
$baseline = '37.0'   # Delphi 13

$lai = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$mormot2 = if ($env:MORMOT2_ROOT) { $env:MORMOT2_ROOT } else { Join-Path $lai '..\..\..\shared\delphi\libs\_git_Synopse2' }
if (-not (Test-Path (Join-Path $mormot2 'src\mormot.defines.inc'))) {
  throw "mORMot2-Quellbaum nicht gefunden unter '$mormot2' - MORMOT2_ROOT setzen."
}
$mormot2 = (Resolve-Path $mormot2).Path

# --- Compiler: explizit (ohne Pruefung) oder das installierte Delphi 13 ---
if ([string]::IsNullOrWhiteSpace($Dcc)) {
  $bds = if ($env:BDS) { $env:BDS } else { Join-Path ${env:ProgramFiles(x86)} "Embarcadero\Studio\$baseline" }
  $Dcc = Join-Path $bds 'bin\dcc64.exe'
  if (-not (Test-Path $Dcc)) {
    throw "dcc64.exe nicht gefunden unter '$Dcc' - Delphi 13 installieren oder -Dcc angeben."
  }
  $banner = (& $Dcc --version 2>&1 | Out-String).Trim()
  if ($banner -notmatch "version $([regex]::Escape($baseline))") {
    throw "'$Dcc' ist nicht Delphi 13 (Compiler $baseline): $banner. Bewusst uebersteuern mit -Dcc."
  }
} else {
  if (-not (Test-Path $Dcc)) { throw "dcc64.exe nicht gefunden: '$Dcc'" }
  $bds = Split-Path -Parent (Split-Path -Parent $Dcc)
}
$compilerVersion = if ((& $Dcc --version 2>&1 | Out-String) -match 'version (\d+\.\d+)') { $Matches[1] } else { '?' }

$src = Join-Path $mormot2 'src'
$units = @((Join-Path $bds 'lib\win64\release'), $src) +
  @('core', 'lib', 'crypt', 'net', 'db', 'orm', 'rest', 'soa', 'app', 'misc', 'script', 'tools\mget' |
    ForEach-Object { Join-Path $src $_ }) +
  @((Join-Path $lai 'src'), (Join-Path $lai 'src\infer'), (Join-Path $lai 'demos'), (Join-Path $lai 'vendor'))
$inc = @($src, (Join-Path $src 'core'), (Join-Path $src 'net'), (Join-Path $lai 'vendor'))
$out = Join-Path $lai 'bin\delphi\demos'
$dcu = Join-Path $out 'dcu'
New-Item -ItemType Directory -Force -Path $dcu | Out-Null

# WICHTIG: hier ALLE Demos auflisten (wie build-demo.sh), sonst meldet der
# Gesamtlauf faelschlich Erfolg fuer Demos, die nie gebaut wurden
$demos = if ($Demo) { @($Demo) } else {
  @('mcp.examples.lpr', 'stdio\demo.mcp.stdio.lpr', 'http\demo.mcp.http.lpr',
    'jsonrpc\demo.mcp.jsonrpc.lpr', 'streamable\demo.mcp.streamable.lpr',
    'llm\llm-chat.lpr', 'llm\llm-agent.lpr', 'llm\llm-structured.lpr',
    'llm\llm-embed.lpr', 'llm\llm-vision.lpr', 'llm\llm-anthropic.lpr',
    'rag\rag-spike.lpr', 'rag\rag-chat.lpr', 'rag\rag-agent.lpr',
    'rag\embed-reference.lpr', 'rag\vector-bench.lpr', 'infer\embed-bench.lpr',
    'infer\llm-bench.lpr')
}

$failed = 0
foreach ($d in $demos) {
  $lpr = Join-Path (Join-Path $lai 'demos') $d
  if (-not (Test-Path $lpr)) { throw "Demo nicht gefunden: '$lpr'" }
  $name = [IO.Path]::GetFileNameWithoutExtension($lpr)
  $dpr = Join-Path $out "$name.dpr"
  $buildLog = Join-Path $out "$name.buildlog"
  Copy-Item $lpr $dpr -Force

  Write-Host ">>> demo $name (x86_64-win64, Delphi $compilerVersion)"
  # -B: immer alles neu uebersetzen (kein veralteter Kompilat-Cache), -GD: Map-Datei
  # dazu das Verzeichnis der Demo: FPC findet Units neben dem Programm von selbst,
  # dcc nicht (die .dpr liegt in bin\delphi\demos)
  $demoUnits = @($units) + @(Split-Path -Parent $lpr)
  & $Dcc -B -Q -GD "-U$($demoUnits -join ';')" "-I$($inc -join ';')" "-R$src" `
    "-O$(Join-Path $mormot2 'static\delphi')" '-NSSystem;Winapi;System.Win;Data;Data.Win;Xml' `
    "-NU$dcu" "-E$out" $dpr *> $buildLog
  $rc = $LASTEXITCODE

  if ($ShowAll) {
    Get-Content $buildLog
  } else {
    # Meldungen sind lokalisiert (englisch/deutsch). Fehler immer zeigen, Warnungen
    # nur aus den eigenen Quellen - die aus mORMot2 selbst sind nicht unsere.
    Select-String -Path $buildLog -Pattern '(Error|Fatal|Fehler|Schwerwiegend):' | ForEach-Object Line
    Select-String -Path $buildLog -Pattern '(Warning|Warnung):' |
      Where-Object { $_.Line -notmatch '\\mormot\.(core|net|db|orm|rest|crypt|lib|soa|app|misc|script|tools)\.' } |
      ForEach-Object Line
  }
  if ($rc -ne 0) {
    Write-Host "FAIL(compile): $name (rc=$rc) - Details: $buildLog" -ForegroundColor Red
    Get-Content $buildLog -Tail 15
    $failed++
    continue
  }
  Write-Host "OK: $(Join-Path $out "$name.exe")"
}
if ($failed -gt 0) {
  Write-Host "SOME DEMOS FAILED ($failed)" -ForegroundColor Red
  exit 1
}
Write-Host 'ALL DEMOS OK'
exit 0
