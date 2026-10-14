<#
  Baut + laeuft die Testsuiten dieser Extension mit Delphi 7 (dcc32, Win32).
  Pendant zu run-delphi-tests.ps1 (Delphi 13, Win64).

  Delphi 7 ist die aelteste Delphi-Version, die mORMot2 unterstuetzt. Die Extension
  baut dort ohne Generics: Registries ueber TSynDictionary, typisierte Tools ueber
  TMcpRecordTool (TMcpToolBase<T> gibt es nur mit HASGENERICS). Der Lauf hier
  belegt das zur Laufzeit, nicht nur beim Uebersetzen - RTTI aus
  Rtti.RegisterFromText, AnsiString-Text-I/O und Win32 verhalten sich anders.

  Beispiele:
    .\scripts\run-delphi7-tests.ps1                 # MCP- + LLM-Suite
    .\scripts\run-delphi7-tests.ps1 -Suite mcp      # eine Suite
    .\scripts\run-delphi7-tests.ps1 -ShowAll        # voller Compiler- + Testlog
    .\scripts\run-delphi7-tests.ps1 -Delphi7 'C:\Borland\Delphi7' -FastMM4 'C:\lib\FastMM4'

  Delphi 7: -Delphi7 / $env:DELPHI7 (Installationsordner mit Bin\DCC32.EXE und
  Lib). Die installierte Bin\dcc32.cfg wird nicht gebraucht: alle Pfade kommen
  von hier (sie zeigt oft auf einen alten Installationsort).
  FastMM4: -FastMM4 / $env:FASTMM4 (Ordner mit FastMM4.pas, FastMM4Messages.pas,
  FastMM4Options.inc). mORMot2 bindet den Speichermanager unter Delphi 7 ueber
  mormot.uses.inc ein; er ist eine eigene Bibliothek und liegt nicht im Repo.
  mORMot2: $env:MORMOT2_ROOT, sonst das Submodul-Layout des Konsumenten.
  Ausgabe: bin\delphi7\ (gitignored).
#>
[CmdletBinding()]
param(
  [ValidateSet('all', 'mcp', 'llm')]
  [string]$Suite = 'all',
  [string]$Delphi7 = $env:DELPHI7,
  [string]$FastMM4 = $env:FASTMM4,
  [switch]$ShowAll
)

$ErrorActionPreference = 'Stop'

$lai = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$mormot2 = if ($env:MORMOT2_ROOT) { $env:MORMOT2_ROOT } else { Join-Path $lai '..\..\..\shared\delphi\libs\_git_Synopse2' }
if (-not (Test-Path (Join-Path $mormot2 'src\mormot.defines.inc'))) {
  throw "mORMot2-Quellbaum nicht gefunden unter '$mormot2' - MORMOT2_ROOT setzen."
}
$mormot2 = (Resolve-Path $mormot2).Path

if ([string]::IsNullOrWhiteSpace($Delphi7)) {
  # Ablageort auf dem Entwicklungsrechner; anderswo -Delphi7 oder DELPHI7 setzen
  $Delphi7 = Join-Path $lai '..\..\..\..\Borland\Delphi7'
}
$Dcc = Join-Path $Delphi7 'Bin\DCC32.EXE'
if (-not (Test-Path $Dcc)) {
  throw "DCC32.EXE nicht gefunden unter '$Dcc' - -Delphi7 oder DELPHI7 setzen."
}
$Delphi7 = (Resolve-Path $Delphi7).Path
$banner = (& $Dcc 2>&1 | Select-Object -First 1 | Out-String).Trim()
if ($banner -notmatch 'Version 15\.0') {
  throw "'$Dcc' ist nicht Delphi 7 (Compiler 15.0): $banner"
}

if ([string]::IsNullOrWhiteSpace($FastMM4) -or
    -not (Test-Path (Join-Path $FastMM4 'FastMM4.pas'))) {
  throw "FastMM4.pas nicht gefunden ('$FastMM4') - -FastMM4 oder FASTMM4 setzen."
}
$FastMM4 = (Resolve-Path $FastMM4).Path

$src = Join-Path $mormot2 'src'
$units = @((Join-Path $Delphi7 'Lib'), (Join-Path $Delphi7 'Lib\Obj'), $FastMM4, $src) +
  @('core', 'lib', 'crypt', 'net', 'db', 'orm', 'rest', 'soa', 'app', 'misc', 'script', 'tools\mget' |
    ForEach-Object { Join-Path $src $_ }) +
  @((Join-Path $lai 'src'), (Join-Path $lai 'tests'), (Join-Path $lai 'vendor'))
$inc = @($src, (Join-Path $src 'core'), (Join-Path $src 'net'), $FastMM4)
$out = Join-Path $lai 'bin\delphi7'
$dcu = Join-Path $out 'dcu'
New-Item -ItemType Directory -Force -Path $dcu | Out-Null

# vec0 gibt es nur als 64-Bit-Bibliothek: ein Win32-Prozess kann sie nicht laden.
# Ein (geerbtes) SQLITE_EXT_DIR gilt hier nur, wenn es auf ein i386-win32-Bundle
# zeigt; sonst ueberspringt sich der vec0-Test, und das wird gemeldet.
$extDirBefore = $env:SQLITE_EXT_DIR
if ($env:SQLITE_EXT_DIR -and ($env:SQLITE_EXT_DIR -notmatch 'i386-win32')) {
  Write-Host ">>> SQLITE_EXT_DIR=$($env:SQLITE_EXT_DIR) ist kein Win32-Bundle - fuer diesen Lauf ignoriert"
  $env:SQLITE_EXT_DIR = $null
}
$ext = Join-Path $lai 'vendor\sqlite-ext\i386-win32'
if (-not $env:SQLITE_EXT_DIR -and (Test-Path $ext)) {
  $env:SQLITE_EXT_DIR = $ext
  Write-Host ">>> SQLITE_EXT_DIR=$ext (vec0 gefunden)"
}
if (-not $env:SQLITE_EXT_DIR) {
  Write-Host '>>> kein Win32-vec0: der vec0-Test der LLM-Suite wird uebersprungen'
}

$suites = if ($Suite -eq 'all') { @('mcp', 'llm') } else { @($Suite) }
$failed = 0
foreach ($s in $suites) {
  $prog = "$s.tests"
  $dpr = Join-Path $out "$prog.dpr"
  $buildLog = Join-Path $out "$prog.buildlog"
  $testLog = Join-Path $out "$prog.log"
  Copy-Item (Join-Path $lai "tests\$prog.lpr") $dpr -Force
  # ein altes Programm darf nie fuer einen neuen Build laufen: beide moeglichen
  # Namen (Delphi 7 kuerzt "mcp.tests" auf mcp.exe) vorher entfernen
  Remove-Item (Join-Path $out "$s.exe"), (Join-Path $out "$prog.exe") -ErrorAction SilentlyContinue

  Write-Host ">>> $prog (i386-win32, Delphi 7)"
  # -B: immer alles neu uebersetzen (siehe run-delphi-tests.ps1). -N: Unit-Ausgabe
  # (Delphi 7 kennt -NU noch nicht). -GD: Map-Datei fuer mORMots Stacktraces.
  # dcc32 liest Pfade mit Leerzeichen nur in Anfuehrungszeichen.
  & $Dcc -B -Q -GD "-U$($units -join ';')" "-I$($inc -join ';')" "-R$($units -join ';')" `
    "-O$(Join-Path $mormot2 'static\delphi')" "-N$dcu" "-E$out" $dpr *> $buildLog
  $rc = $LASTEXITCODE

  if ($ShowAll) {
    Get-Content $buildLog
  } else {
    Select-String -Path $buildLog -Pattern '(Error|Fatal):' | ForEach-Object Line
    Select-String -Path $buildLog -Pattern 'Warning:' |
      Where-Object { $_.Line -notmatch '\\mormot\.(core|net|db|orm|rest|crypt|lib|soa|app|misc|script|tools)\.' } |
      ForEach-Object Line
  }
  if ($rc -ne 0) {
    Write-Host "FAIL(compile): $prog (rc=$rc) - Details: $buildLog" -ForegroundColor Red
    Get-Content $buildLog -Tail 15
    $failed++
    continue
  }

  # Delphi 7 kennt keine Punkte im Programmnamen: aus "mcp.tests" wird mcp.exe
  $exe = Join-Path $out "$s.exe"
  if (-not (Test-Path $exe)) { $exe = Join-Path $out "$prog.exe" }
  $runOut = & $exe 2>&1
  $runRc = $LASTEXITCODE
  $runOut | Set-Content -Path $testLog -Encoding utf8
  if ($ShowAll) { $runOut }
  $summary = $runOut | Select-String -Pattern 'Total assertions failed for all test suits' | Select-Object -Last 1
  if (($runRc -eq 0) -and $summary) {
    Write-Host "PASS: $prog - $($summary.Line.Trim()) (Log: $testLog)"
  } else {
    Write-Host "FAIL: $prog (rc=$runRc) - Voller Log: $testLog" -ForegroundColor Red
    $runOut | Select-String -Pattern '(^!|FAILED|Exception)' | Select-Object -First 30 | ForEach-Object Line
    $failed++
  }
}
$env:SQLITE_EXT_DIR = $extDirBefore
if ($failed -gt 0) { exit 1 }
exit 0
