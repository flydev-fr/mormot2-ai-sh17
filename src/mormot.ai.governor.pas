/// LandrixAI - a load governor for bulk embedding (or any bulk CPU work)
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - a first indexing of millions of texts keeps every core busy for hours;
//   TLoadGovernor paces such a job without touching fans or power limits (they
//   stay with firmware and BIOS):
//   - a duty cycle: after each unit of work, pause in proportion to its busy time
//     (DutyPercent = 80 keeps each worker busy about 80 % of the wall time)
//   - a temperature threshold where a sensor is readable (Linux hwmon, also
//     inside a container): at MaxTempC the job waits until ResumeTempC, at most
//     MaxCoolWaitMs per wait
//   - while no threshold is active (none set, or no sensor readable), a
//     heuristic on the throughput: work units per busy second, window by window,
//     against a reference that slowly ages; a lasting drop - throttling, or just
//     competing load or a different mix of texts - lowers the duty cycle, a
//     recovery raises it again. A heuristic, not a measurement
// - pauses happen outside the lock, so workers sharing a governor pause side by
//   side; Abort ends every pause at once (e.g. on shutdown)
// - TGovernedEmbedder wraps any IEmbedder: a bulk job uses the wrapped one,
//   interactive callers keep the plain one
// - the clock, the pause and the sensor are virtual methods: tests run the whole
//   logic on a fake clock
unit mormot.ai.governor;

{
  *****************************************************************************

    - TLoadGovernor: duty cycle, temperature threshold, throughput heuristic
    - TGovernedEmbedder: an IEmbedder paced by a TLoadGovernor
    - ReadCpuTemperature: the CPU temperature where the OS exposes it

  *****************************************************************************
}

interface

{$I mormot.defines.inc}

uses
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.text,
  mormot.ai.llm.types,
  mormot.ai.embeddings;

type
  /// how a TLoadGovernor paces a job
  TLoadGovernorSettings = record
    /// share of the wall time each worker may compute, 1..100 (100 = no pause)
    DutyPercent: integer;
    /// wait from this CPU temperature on (degrees Celsius), 0 = no threshold
    MaxTempC: integer;
    /// ... until the temperature is down to this one (0 = MaxTempC - 5)
    ResumeTempC: integer;
    /// longest single wait for the temperature, in ms (0 = never wait)
    MaxCoolWaitMs: integer;
    /// read the sensor at most this often, in ms (also the step of a wait)
    TempCheckMs: integer;
    /// lower the duty cycle on a lasting throughput drop - used only while no
    // temperature threshold is active (none set, or no sensor readable)
    ThroughputHeuristic: boolean;
    /// the length of one throughput window, in busy seconds
    WindowSeconds: integer;
  end;

  /// paces a bulk job: call Account after every unit of work
  // - thread-safe: several workers may share one governor; each pauses for its
  // own work, side by side
  TLoadGovernor = class
  protected
    fSettings: TLoadGovernorSettings;
    fLock: TOSLightLock;
    fAborted: boolean;
    fDuty: integer;
    fDebtUs: Int64;
    fTemp: integer;
    fTempAt: Int64;
    fSensor: integer; // -1 unknown yet, 0 none, 1 readable
    // too hot: set at MaxTempC, cleared at ResumeTempC - shared by all workers,
    // so a worker arriving between two sensor reads waits too
    fHot: boolean;
    fPausedUs: Int64;
    fCoolWaits: integer;
    fDutyChanges: integer;
    fWinBusyUs, fWinUnits: Int64;
    fRefRate: double;
    fSlowWindows: integer;
    procedure Heuristic(aBusyUs, aUnits: Int64);
    /// one sensor reading into the shared state (fLock held): the hysteresis
    // lives here only, so every read - fresh or during a wait, by any worker -
    // leaves the same verdict for all of them
    procedure ApplyTemp(aTemp: integer);
    procedure CoolDown;
    /// sleep aMs, in slices, until done or aborted
    procedure Wait(aMs: Int64);
    /// the clock in microseconds - virtual for tests
    function NowUs: Int64; virtual;
    /// pause the calling thread (at most a few seconds) - virtual for tests
    procedure Pause(aMs: integer); virtual;
    /// the CPU temperature in degrees Celsius, -1 if unknown - virtual for tests
    function ReadTempC: integer; virtual;
  public
    /// a governor with these settings
    constructor Create(const aSettings: TLoadGovernorSettings);
    destructor Destroy; override;
    /// one unit of work took aBusyUs microseconds and did aUnits units (texts,
    // bytes or tokens - only compared with each other): pause as required
    procedure Account(aBusyUs, aUnits: Int64);
    /// end every running and future pause at once (e.g. on shutdown)
    procedure Abort;
    /// the settings in force (after their normalization)
    property Settings: TLoadGovernorSettings
      read fSettings;
    /// the duty cycle in force, below DutyPercent after a throughput drop
    property EffectiveDuty: integer
      read fDuty;
    /// how often the heuristic changed the duty cycle
    property DutyChanges: integer
      read fDutyChanges;
    /// the last temperature read, -1 if none
    property Temperature: integer
      read fTemp;
    /// how often a worker waited for the temperature
    property CoolWaits: integer
      read fCoolWaits;
    /// true once Abort was called
    property Aborted: boolean
      read fAborted;
    /// the time paused so far, in ms (all workers together)
    function PausedMs: Int64;
  end;

  /// an IEmbedder whose work is paced by a TLoadGovernor
  // - the units are the bytes of the texts: the throughput heuristic compares
  // windows of many texts, in which the mix of lengths evens out
  // - EmbedBatch works in slices of BatchSlice texts, each one accounted for
  // (also when the inner embedder raises), so a large batch is paced as it goes
  TGovernedEmbedder = class(TInterfacedObject, IEmbedder)
  protected
    fInner: IEmbedder;
    fGovernor: TLoadGovernor;
    fOwnsGovernor: boolean;
    fBatchSlice: integer;
  public
    /// wrap aInner; aOwnsGovernor: free aGovernor with this instance (also
    // when this constructor raises)
    constructor Create(const aInner: IEmbedder; aGovernor: TLoadGovernor;
      aOwnsGovernor: boolean = true); reintroduce;
    destructor Destroy; override;
    function Embed(const aText: RawUtf8): TLlmEmbedding;
    function EmbedBatch(const aTexts: TRawUtf8DynArray): TLlmEmbeddingDynArray;
    function Model: RawUtf8;
    /// the governor in use
    property Governor: TLoadGovernor
      read fGovernor;
    /// texts per inner EmbedBatch call (default 16)
    property BatchSlice: integer
      read fBatchSlice write fBatchSlice;
  end;

/// the default settings: no pause, no threshold, the heuristic on
// - MaxCoolWaitMs 60000, TempCheckMs 2000, WindowSeconds 30
function LoadGovernorSettings(aDutyPercent: integer = 100;
  aMaxTempC: integer = 0): TLoadGovernorSettings;

/// the CPU temperature in degrees Celsius, -1 where the OS does not expose it
// - Linux: a CPU sensor under /sys/class/hwmon (coretemp, k10temp, zenpower,
// cpu_thermal) - its labelled package value (Package id, Tdie, then Tctl) or
// else its hottest input; without one, a CPU thermal zone. Readable without
// root, also inside a container
// - Windows, macOS: -1 - their sensors need a kernel driver or private APIs
function ReadCpuTemperature: integer;

/// ReadCpuTemperature below another root than '/sys' - for tests
function ReadCpuTemperatureIn(const aSysRoot: TFileName): integer;

/// the trimmed text of a small system file such as /sys/.../name
// - sysfs and procfs report a size of 4096 or 0, whatever they hold: read
// without trusting it (StringFromFile would return '' for them)
function SysFileText(const aFileName: TFileName): RawUtf8;


implementation

function LoadGovernorSettings(aDutyPercent, aMaxTempC: integer): TLoadGovernorSettings;
begin
  FillCharFast(result, SizeOf(result), 0);
  result.DutyPercent := aDutyPercent;
  result.MaxTempC := aMaxTempC;
  result.MaxCoolWaitMs := 60000;
  result.TempCheckMs := 2000;
  result.ThroughputHeuristic := true;
  result.WindowSeconds := 30;
end;

function SysFileText(const aFileName: TFileName): RawUtf8;
begin
  {$ifdef OSLINUXANDROID}
  result := TrimU(StringFromFileNoSize(aFileName));
  {$else}
  result := TrimU(StringFromFile(aFileName));
  {$endif OSLINUXANDROID}
end;

/// the integer content of a system file, -1 if unreadable
function SysInt(const aFileName: TFileName): Int64;
var
  s: RawUtf8;
begin
  result := -1;
  s := SysFileText(aFileName);
  if (s <> '') and (s[1] in ['0'..'9']) then
    result := GetInt64(pointer(s));
end;

function IsCpuSensor(const aName: RawUtf8): boolean;
begin
  result := (aName = 'coretemp') or (aName = 'k10temp') or
            (aName = 'zenpower') or (aName = 'cpu_thermal') or
            (aName = 'cpu-thermal');
end;

/// rank of a temperature label: 3 Package id (Intel), 2 Tdie (AMD, without the
// Tctl offset of some Zen parts), 1 Tctl, 0 anything else
function LabelRank(const aLabel: RawUtf8): integer;
begin
  if copy(aLabel, 1, 10) = 'Package id' then
    result := 3
  else if aLabel = 'Tdie' then
    result := 2
  else if aLabel = 'Tctl' then
    result := 1
  else
    result := 0;
end;

/// the subfolders of aDir whose name starts with aPrefix
function SubFolders(const aDir, aPrefix: TFileName): TFileNameDynArray;
var
  sr: TSearchRec;
  n: integer;
  mask: TFileName;
begin
  result := nil;
  n := 0;
  // through a TFileName variable: FPC 3.2 gives the bare concatenation the code
  // page CP_NONE, which its Windows FindFirst does not convert (path not found)
  mask := aDir + aPrefix + '*';
  if FindFirst(mask, faAnyFile, sr) = 0 then
  try
    repeat
      if (sr.Name <> '.') and (sr.Name <> '..') then
      begin
        SetLength(result, n + 1);
        result[n] := aDir + sr.Name + PathDelim;
        inc(n);
      end;
    until FindNext(sr) <> 0;
  finally
    FindClose(sr);
  end;
end;

function ReadCpuTemperatureIn(const aSysRoot: TFileName): integer;
var
  dirs: TFileNameDynArray;
  i, j, rank, bestRank: integer;
  v, best, labelled: Int64;
  root: TFileName;
begin
  root := IncludeTrailingPathDelimiter(aSysRoot);
  best := -1;
  labelled := -1;
  bestRank := 0;
  // hwmon: one folder per sensor chip, temp<n>_input in millidegrees
  dirs := SubFolders(root + 'class' + PathDelim + 'hwmon' + PathDelim, 'hwmon');
  for i := 0 to high(dirs) do
  begin
    if not IsCpuSensor(SysFileText(dirs[i] + 'name')) then
      continue;
    for j := 1 to 64 do
    begin
      v := SysInt(dirs[i] + 'temp' + IntToStr(j) + '_input');
      if v < 0 then
        continue;
      if v > best then
        best := v;
      rank := LabelRank(SysFileText(dirs[i] + 'temp' + IntToStr(j) + '_label'));
      if (rank > bestRank) or ((rank = bestRank) and (rank > 0) and (v > labelled)) then
      begin
        bestRank := rank;
        labelled := v;
      end;
    end;
  end;
  if labelled >= 0 then
    best := labelled; // the package value, not the hottest core or offset Tctl
  if best < 0 then
  begin
    // no hwmon driver: a CPU thermal zone, if any
    dirs := SubFolders(root + 'class' + PathDelim + 'thermal' + PathDelim,
      'thermal_zone');
    for i := 0 to high(dirs) do
      if (SysFileText(dirs[i] + 'type') = 'x86_pkg_temp') or
         IsCpuSensor(SysFileText(dirs[i] + 'type')) then
      begin
        v := SysInt(dirs[i] + 'temp');
        if v > best then
          best := v;
      end;
  end;
  if best < 0 then
    result := -1
  else
    result := (best + 500) div 1000;
end;

function ReadCpuTemperature: integer;
begin
  {$ifdef OSLINUX}
  result := ReadCpuTemperatureIn('/sys');
  {$else}
  result := -1;
  {$endif OSLINUX}
end;


{ TLoadGovernor }

constructor TLoadGovernor.Create(const aSettings: TLoadGovernorSettings);
begin
  inherited Create;
  fLock.Init;
  fSettings := aSettings;
  if fSettings.DutyPercent < 1 then
    fSettings.DutyPercent := 1
  else if fSettings.DutyPercent > 100 then
    fSettings.DutyPercent := 100;
  if fSettings.MaxTempC < 0 then
    fSettings.MaxTempC := 0;
  if fSettings.MaxTempC > 0 then
  begin
    if (fSettings.ResumeTempC <= 0) or (fSettings.ResumeTempC > fSettings.MaxTempC) then
      fSettings.ResumeTempC := fSettings.MaxTempC - 5;
    if fSettings.ResumeTempC < 1 then
      fSettings.ResumeTempC := 1;
  end;
  if fSettings.MaxCoolWaitMs < 0 then
    fSettings.MaxCoolWaitMs := 0;
  if fSettings.TempCheckMs < 100 then
    fSettings.TempCheckMs := 100;
  if fSettings.WindowSeconds < 1 then
    fSettings.WindowSeconds := 1;
  fDuty := fSettings.DutyPercent;
  fTemp := -1;
  fTempAt := -1;
  fSensor := -1;
end;

destructor TLoadGovernor.Destroy;
begin
  fLock.Done;
  inherited Destroy;
end;

function TLoadGovernor.NowUs: Int64;
begin
  QueryPerformanceMicroSeconds(result);
end;

procedure TLoadGovernor.Pause(aMs: integer);
begin
  if aMs > 0 then
    SleepHiRes(aMs);
end;

function TLoadGovernor.ReadTempC: integer;
begin
  result := ReadCpuTemperature;
end;

procedure TLoadGovernor.Abort;
begin
  fAborted := true;
end;

function TLoadGovernor.PausedMs: Int64;
begin
  fLock.Lock;
  result := fPausedUs div 1000;
  fLock.UnLock;
end;

procedure TLoadGovernor.Wait(aMs: Int64);
var
  step: integer;
begin
  // slices of at most 100 ms: Abort ends a long pause quickly
  while (aMs > 0) and not fAborted do
  begin
    if aMs > 100 then
      step := 100
    else
      step := aMs;
    Pause(step);
    dec(aMs, step);
    fLock.Lock;
    inc(fPausedUs, Int64(step) * 1000);
    fLock.UnLock;
  end;
end;

procedure TLoadGovernor.ApplyTemp(aTemp: integer);
begin
  fTemp := aTemp;
  fTempAt := NowUs;
  if aTemp < 0 then
  begin
    fSensor := 0;
    fHot := false; // no sensor: nothing to wait for
  end
  else
  begin
    fSensor := 1;
    if aTemp >= fSettings.MaxTempC then
      fHot := true
    else if aTemp <= fSettings.ResumeTempC then
      fHot := false; // between both: the verdict stays as it was
  end;
end;

procedure TLoadGovernor.CoolDown;
var
  waited, step: integer;
  hot: boolean;
begin
  // too hot: wait in steps of TempCheckMs until the shared state says cool,
  // the budget is spent or the job is aborted; the sensor is read and its
  // value applied together under the lock, at most every TempCheckMs for all
  // workers - no stale read can overwrite a newer one
  waited := 0;
  while (waited < fSettings.MaxCoolWaitMs) and not fAborted do
  begin
    step := fSettings.MaxCoolWaitMs - waited;
    if step > fSettings.TempCheckMs then
      step := fSettings.TempCheckMs;
    Wait(step);
    inc(waited, step);
    fLock.Lock;
    try
      if (fTempAt < 0) or
         (NowUs - fTempAt >= Int64(fSettings.TempCheckMs) * 1000) then
        ApplyTemp(ReadTempC);
      hot := fHot;
    finally
      fLock.UnLock;
    end;
    if not hot then
      break;
  end;
end;

procedure TLoadGovernor.Heuristic(aBusyUs, aUnits: Int64);
var
  rate: double;
  floor: integer;
begin
  inc(fWinBusyUs, aBusyUs);
  inc(fWinUnits, aUnits);
  if fWinBusyUs < Int64(fSettings.WindowSeconds) * 1000000 then
    exit;
  rate := fWinUnits / fWinBusyUs; // units per busy microsecond
  fWinBusyUs := 0;
  fWinUnits := 0;
  // the reference ages by 2 % per window: an outlier window or a lasting change
  // of the work mix becomes the new normal instead of a permanent "drop"
  fRefRate := fRefRate * 0.98;
  if rate > fRefRate then
    fRefRate := rate;
  floor := 25;
  if floor > fSettings.DutyPercent then
    floor := fSettings.DutyPercent;
  if rate < 0.75 * fRefRate then
  begin
    // two slow windows in a row: less duty, the CPU may get its clock back
    inc(fSlowWindows);
    if (fSlowWindows >= 2) and (fDuty > floor) then
    begin
      fSlowWindows := 0;
      dec(fDuty, 10);
      if fDuty < floor then
        fDuty := floor;
      inc(fDutyChanges);
    end;
  end
  else
  begin
    fSlowWindows := 0;
    if (rate >= 0.9 * fRefRate) and (fDuty < fSettings.DutyPercent) then
    begin
      inc(fDuty, 5);
      if fDuty > fSettings.DutyPercent then
        fDuty := fSettings.DutyPercent;
      inc(fDutyChanges);
    end;
  end;
end;

const
  /// one unit counts at most a day of busy time: keeps the pause arithmetic
  // far from any overflow
  MAX_BUSY_US = Int64(86400) * 1000000;

procedure TLoadGovernor.Account(aBusyUs, aUnits: Int64);
var
  pauseMs: Int64;
  hot: boolean;
  now: Int64;
begin
  if fAborted then
    exit;
  if aBusyUs < 0 then
    aBusyUs := 0
  else if aBusyUs > MAX_BUSY_US then
    aBusyUs := MAX_BUSY_US;
  if aUnits < 0 then
    aUnits := 0;
  hot := false;
  fLock.Lock;
  try
    // the temperature, at most every TempCheckMs
    if fSettings.MaxTempC > 0 then
    begin
      now := NowUs;
      if (fTempAt < 0) or (now - fTempAt >= Int64(fSettings.TempCheckMs) * 1000) then
      begin
        ApplyTemp(ReadTempC);
      end;
      // the shared state, not only this worker's read: between two reads a
      // known overheat holds every worker
      hot := fHot and (fSettings.MaxCoolWaitMs > 0);
      if hot then
        inc(fCoolWaits);
    end;
    if fSettings.ThroughputHeuristic and
       ((fSettings.MaxTempC <= 0) or (fSensor = 0)) then
      Heuristic(aBusyUs, aUnits);
    // duty cycle: busy * (100 - duty) / duty of pause, the fractions of a
    // millisecond carried over to the next unit
    pauseMs := 0;
    if fDuty < 100 then
    begin
      inc(fDebtUs, aBusyUs * (100 - fDuty) div fDuty);
      pauseMs := fDebtUs div 1000;
      dec(fDebtUs, pauseMs * 1000);
    end;
  finally
    fLock.UnLock;
  end;
  // the waits happen outside the lock: each worker pauses for its own work
  if hot then
    CoolDown;
  Wait(pauseMs);
end;


{ TGovernedEmbedder }

constructor TGovernedEmbedder.Create(const aInner: IEmbedder;
  aGovernor: TLoadGovernor; aOwnsGovernor: boolean);
begin
  inherited Create;
  if (aInner = nil) or (aGovernor = nil) then
  begin
    if aOwnsGovernor then
      aGovernor.Free; // ownership was handed over: no leak on the error path
    raise ESynException.CreateUtf8('%.Create: embedder and governor required', [self]);
  end;
  fInner := aInner;
  fGovernor := aGovernor;
  fOwnsGovernor := aOwnsGovernor;
  fBatchSlice := 16;
end;

destructor TGovernedEmbedder.Destroy;
begin
  if fOwnsGovernor then
    fGovernor.Free;
  inherited Destroy;
end;

function TGovernedEmbedder.Embed(const aText: RawUtf8): TLlmEmbedding;
var
  t0, t1: Int64;
begin
  QueryPerformanceMicroSeconds(t0);
  try
    result := fInner.Embed(aText);
  finally
    // the work was done, also when it failed: it counts
    QueryPerformanceMicroSeconds(t1);
    fGovernor.Account(t1 - t0, length(aText));
  end;
end;

function TGovernedEmbedder.EmbedBatch(
  const aTexts: TRawUtf8DynArray): TLlmEmbeddingDynArray;
var
  t0, t1, units: Int64;
  first, n, i, slice: PtrInt;
  part: TRawUtf8DynArray;
  got: TLlmEmbeddingDynArray;
begin
  SetLength(result, length(aTexts));
  slice := fBatchSlice;
  if slice < 1 then
    slice := 1;
  first := 0;
  while first < length(aTexts) do
  begin
    n := length(aTexts) - first;
    if n > slice then
      n := slice;
    SetLength(part, n);
    units := 0;
    for i := 0 to n - 1 do
    begin
      part[i] := aTexts[first + i];
      inc(units, length(part[i]));
    end;
    QueryPerformanceMicroSeconds(t0);
    try
      got := fInner.EmbedBatch(part);
    finally
      QueryPerformanceMicroSeconds(t1);
      fGovernor.Account(t1 - t0, units);
    end;
    if length(got) <> n then
      raise ESynException.CreateUtf8('%.EmbedBatch: % vectors for % texts',
        [self, length(got), n]);
    for i := 0 to n - 1 do
      result[first + i] := got[i];
    inc(first, n);
  end;
end;

function TGovernedEmbedder.Model: RawUtf8;
begin
  result := fInner.Model;
end;

end.
