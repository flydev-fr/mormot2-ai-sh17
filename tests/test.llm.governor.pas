/// LandrixAI - tests of the load governor (mormot.ai.governor)
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - the whole logic runs on a fake clock, a recorded pause and a scripted
//   sensor: no test waits for real
// - the sensor reading runs on a fake /sys tree (folders linked like sysfs on
//   POSIX) and, on Linux, against the real /sys: sysfs reports 4096 bytes for
//   every attribute, which a size-trusting reader turns into empty strings
unit test.llm.governor;

interface

{$I mormot.defines.inc}

uses
  sysutils,
  {$ifdef FPC}
  {$ifdef OSPOSIX}
  baseunix,
  {$endif OSPOSIX}
  {$endif FPC}
  mormot.core.base,
  mormot.core.os,
  mormot.core.text,
  mormot.core.test,
  mormot.ai.llm.types,
  mormot.ai.embeddings,
  mormot.ai.governor;

type
  /// a governor on a fake clock: Pause advances the clock, the sensor is a
  // script of temperatures (the last one repeats; empty = no sensor)
  TFakeGovernor = class(TLoadGovernor)
  protected
    fClock: Int64;
    fTemps: TIntegerDynArray;
    fTempIndex: integer;
    function NowUs: Int64; override;
    procedure Pause(aMs: integer); override;
    function ReadTempC: integer; override;
  public
    /// the pauses so far, in ms
    Paused: Int64;
    /// how often the sensor was read
    Reads: integer;
    /// one unit of work: the clock moves on by aBusyUs, then Account
    procedure Work(aBusyUs, aUnits: Int64);
    /// the sensor script
    procedure SetTemps(const aTemps: array of integer);
  end;

  /// a fixed vector per text, counting its calls; the text 'boom' raises
  TFakeEmbedder = class(TInterfacedObject, IEmbedder)
  public
    Calls, BatchCalls: integer;
    function Embed(const aText: RawUtf8): TLlmEmbedding;
    function EmbedBatch(const aTexts: TRawUtf8DynArray): TLlmEmbeddingDynArray;
    function Model: RawUtf8;
  end;

  /// mormot.ai.governor tests
  TTestLoadGovernor = class(TSynTestCase)
  protected
    procedure WriteSys(const aRoot, aPath: TFileName; const aText: RawUtf8);
  published
    procedure DutyCycle;
    procedure TemperatureThreshold;
    procedure ThroughputHeuristic;
    procedure AbortEndsPauses;
    procedure GovernedEmbedder;
    procedure SensorTree;
    procedure CpuTemperature;
  end;


implementation

{ TFakeGovernor }

function TFakeGovernor.NowUs: Int64;
begin
  result := fClock;
end;

procedure TFakeGovernor.Pause(aMs: integer);
begin
  inc(Paused, aMs);
  inc(fClock, Int64(aMs) * 1000);
end;

function TFakeGovernor.ReadTempC: integer;
begin
  inc(Reads);
  if fTemps = nil then
    result := -1
  else
  begin
    result := fTemps[fTempIndex];
    if fTempIndex < high(fTemps) then
      inc(fTempIndex);
  end;
end;

procedure TFakeGovernor.SetTemps(const aTemps: array of integer);
var
  i: PtrInt;
begin
  SetLength(fTemps, length(aTemps));
  for i := 0 to high(aTemps) do
    fTemps[i] := aTemps[i];
  fTempIndex := 0;
end;

procedure TFakeGovernor.Work(aBusyUs, aUnits: Int64);
begin
  inc(fClock, aBusyUs);
  Account(aBusyUs, aUnits);
end;


{ TFakeEmbedder }

function TFakeEmbedder.Embed(const aText: RawUtf8): TLlmEmbedding;
begin
  inc(Calls);
  if aText = 'boom' then
    raise ESynException.Create('boom');
  SetLength(result, 2);
  result[0] := length(aText);
  result[1] := 1;
end;

function TFakeEmbedder.EmbedBatch(const aTexts: TRawUtf8DynArray): TLlmEmbeddingDynArray;
var
  i: PtrInt;
begin
  inc(BatchCalls);
  SetLength(result, length(aTexts));
  for i := 0 to high(aTexts) do
    result[i] := Embed(aTexts[i]);
end;

function TFakeEmbedder.Model: RawUtf8;
begin
  result := 'fake';
end;


{ TTestLoadGovernor }

/// a sysfs-style path with this OS's delimiter
function Sys(const aPath: TFileName): TFileName;
var
  i: PtrInt;
begin
  result := aPath;
  for i := 1 to length(result) do
    if result[i] = '/' then
      result[i] := PathDelim;
end;

procedure TTestLoadGovernor.WriteSys(const aRoot, aPath: TFileName;
  const aText: RawUtf8);
var
  f: TFileName;
begin
  f := aRoot + Sys(aPath);
  ForceDirectories(ExtractFilePath(f));
  FileFromString(aText + #10, f);
end;

procedure TTestLoadGovernor.DutyCycle;
var
  g: TFakeGovernor;
  i: integer;
begin
  // 100 %: never a pause
  g := TFakeGovernor.Create(LoadGovernorSettings(100));
  try
    for i := 1 to 10 do
      g.Work(5000, 1);
    CheckEqual(g.Paused, 0, 'duty 100: no pause');
  finally
    g.Free;
  end;
  // 50 %: as long a pause as the work; 25 %: three times as long
  g := TFakeGovernor.Create(LoadGovernorSettings(50));
  try
    g.Work(10000, 1);
    CheckEqual(g.Paused, 10, 'duty 50: 10 ms of work, 10 ms of pause');
  finally
    g.Free;
  end;
  g := TFakeGovernor.Create(LoadGovernorSettings(25));
  try
    g.Work(3000, 1);
    CheckEqual(g.Paused, 9, 'duty 25: 3 ms of work, 9 ms of pause');
    CheckEqual(g.PausedMs, 9, 'PausedMs');
  finally
    g.Free;
  end;
  // short units: the fractions of a millisecond add up, nothing is lost
  g := TFakeGovernor.Create(LoadGovernorSettings(50));
  try
    for i := 1 to 10 do
      g.Work(300, 1);
    CheckEqual(g.Paused, 3, 'duty 50: 10 x 0.3 ms of work, 3 ms of pause');
  finally
    g.Free;
  end;
  // a unit of two days: counted as one day of work, no overflow, no negative
  g := TFakeGovernor.Create(LoadGovernorSettings(50));
  try
    g.Work(Int64(2) * 86400 * 1000000, 1);
    CheckEqual(g.Paused, Int64(86400) * 1000, 'duty 50: clamped to one day of work');
  finally
    g.Free;
  end;
  // out-of-range settings are clamped
  g := TFakeGovernor.Create(LoadGovernorSettings(0));
  try
    CheckEqual(g.EffectiveDuty, 1, 'duty 0 -> 1');
  finally
    g.Free;
  end;
  g := TFakeGovernor.Create(LoadGovernorSettings(250));
  try
    CheckEqual(g.EffectiveDuty, 100, 'duty 250 -> 100');
  finally
    g.Free;
  end;
end;

procedure TTestLoadGovernor.TemperatureThreshold;
var
  g: TFakeGovernor;
  s: TLoadGovernorSettings;
begin
  // 90 >= 85: wait in steps of TempCheckMs (2 s); 88 is still above the resume
  // point 80, 79 below it
  g := TFakeGovernor.Create(LoadGovernorSettings(100, 85));
  try
    g.SetTemps([90, 88, 79]);
    g.Work(1000, 1);
    CheckEqual(g.CoolWaits, 1, 'one cool-down');
    CheckEqual(g.Paused, 4000, 'two steps of two seconds');
    CheckEqual(g.Temperature, 79, 'temperature after the wait');
    // within TempCheckMs: no new read
    g.Work(1000, 1);
    CheckEqual(g.Reads, 3, 'sensor read at most every TempCheckMs');
    g.Work(2000000, 1);
    CheckEqual(g.Reads, 4, 'and again after it');
    CheckEqual(g.CoolWaits, 1, 'cool: no further wait');
  finally
    g.Free;
  end;
  // a sensor that stays hot: the wait stops exactly at MaxCoolWaitMs
  s := LoadGovernorSettings(100, 85);
  s.MaxCoolWaitMs := 5000;
  g := TFakeGovernor.Create(s);
  try
    g.SetTemps([95]);
    g.Work(1000, 1);
    CheckEqual(g.Paused, 5000, 'the wait stops at MaxCoolWaitMs');
  finally
    g.Free;
  end;
  // still hot after a capped wait: the next unit waits again, even without a
  // new read (another worker arriving between two reads must not run on)
  s := LoadGovernorSettings(100, 85);
  s.MaxCoolWaitMs := 2000;
  s.TempCheckMs := 10000;
  g := TFakeGovernor.Create(s);
  try
    g.SetTemps([95]);
    g.Work(1000, 1);
    CheckEqual(g.Paused, 2000, 'first unit: capped wait');
    g.Work(1000, 1);
    CheckEqual(g.Reads, 1, 'no read before TempCheckMs has passed, waits included');
    CheckEqual(g.Paused, 4000, 'second unit waits too: still hot');
    CheckEqual(g.CoolWaits, 2, 'two waits');
  finally
    g.Free;
  end;
  // two workers: A just read "cool" and cleared the shared state, B ends its
  // wait with a hot read - the shared state is hot again, the next unit waits
  s := LoadGovernorSettings(100, 85);
  s.MaxCoolWaitMs := 2000;
  g := TFakeGovernor.Create(s);
  try
    g.SetTemps([60]);
    g.Work(1000, 1); // a fresh read: cool
    CheckEqual(g.Paused, 0, 'cool: no wait');
    g.SetTemps([95]);
    g.CoolDown; // worker B, already waiting: its read is hot
    CheckEqual(g.Paused, 2000, 'B waited its budget');
    g.Work(1000, 1); // no fresh read (TempCheckMs): the shared state rules
    CheckEqual(g.Paused, 4000, 'the next unit waits: hot again for everyone');
  finally
    g.Free;
  end;
  // MaxCoolWaitMs = 0: never wait
  s := LoadGovernorSettings(100, 85);
  s.MaxCoolWaitMs := 0;
  g := TFakeGovernor.Create(s);
  try
    g.SetTemps([95]);
    g.Work(1000, 1);
    CheckEqual(g.Paused, 0, 'MaxCoolWaitMs 0: no wait');
    CheckEqual(g.CoolWaits, 0, 'and no cool-down counted');
  finally
    g.Free;
  end;
  // no sensor: the threshold cannot apply, the job runs on
  g := TFakeGovernor.Create(LoadGovernorSettings(100, 85));
  try
    g.Work(1000, 1);
    CheckEqual(g.Paused, 0, 'no sensor, no wait');
    CheckEqual(g.Temperature, -1, 'no temperature');
  finally
    g.Free;
  end;
  // an explicit resume point above the threshold is corrected to 80
  s := LoadGovernorSettings(100, 85);
  s.ResumeTempC := 90;
  g := TFakeGovernor.Create(s);
  try
    g.SetTemps([86, 84, 80]);
    g.Work(1000, 1);
    CheckEqual(g.Settings.ResumeTempC, 80, 'resume point corrected');
    CheckEqual(g.Paused, 4000, 'waited down to 80');
  finally
    g.Free;
  end;
  // a tiny threshold keeps a positive resume point
  g := TFakeGovernor.Create(LoadGovernorSettings(100, 3));
  try
    CheckEqual(g.Settings.ResumeTempC, 1, 'resume point at least 1');
  finally
    g.Free;
  end;
end;

procedure TTestLoadGovernor.ThroughputHeuristic;
var
  g: TFakeGovernor;
  s: TLoadGovernorSettings;
  i: integer;
begin
  s := LoadGovernorSettings(100);
  s.WindowSeconds := 1;
  g := TFakeGovernor.Create(s);
  try
    // fast windows: 1000 units per busy second (10 units of 0.1 s per window)
    for i := 1 to 20 do
      g.Work(100000, 100);
    CheckEqual(g.EffectiveDuty, 100, 'steady: full duty');
    // one slow window alone changes nothing
    for i := 1 to 10 do
      g.Work(100000, 50);
    CheckEqual(g.EffectiveDuty, 100, 'one slow window: no change');
    // a second one in a row lowers the duty by 10
    for i := 1 to 10 do
      g.Work(100000, 50);
    CheckEqual(g.EffectiveDuty, 90, 'two slow windows: duty 90');
    // recovered windows raise it again, 5 per window, up to the setting
    for i := 1 to 10 do
      g.Work(100000, 100);
    CheckEqual(g.EffectiveDuty, 95, 'recovering: 95');
    for i := 1 to 30 do
      g.Work(100000, 100);
    CheckEqual(g.EffectiveDuty, 100, 'recovered: 100');
    // a lasting drop goes down to 25 at most
    for i := 1 to 400 do
      g.Work(100000, 10);
    CheckEqual(g.EffectiveDuty, 25, 'floor of 25');
    Check(g.DutyChanges > 0, 'duty changes counted');
  finally
    g.Free;
  end;
  // the floor never raises a configured duty below it
  s := LoadGovernorSettings(10);
  s.WindowSeconds := 1;
  g := TFakeGovernor.Create(s);
  try
    for i := 1 to 20 do
      g.Work(100000, 100);
    for i := 1 to 100 do
      g.Work(100000, 10);
    CheckEqual(g.EffectiveDuty, 10, 'duty 10 stays 10');
  finally
    g.Free;
  end;
  // one fast outlier window, then a normal rate: the reference ages, the
  // duty comes back instead of staying throttled for good
  s := LoadGovernorSettings(100);
  s.WindowSeconds := 1;
  g := TFakeGovernor.Create(s);
  try
    for i := 1 to 10 do
      g.Work(100000, 200);
    for i := 1 to 800 do
      g.Work(100000, 100);
    CheckEqual(g.EffectiveDuty, 100, 'outlier forgotten');
  finally
    g.Free;
  end;
  // with a readable sensor and a threshold, the heuristic stays out
  s := LoadGovernorSettings(100, 85);
  s.WindowSeconds := 1;
  g := TFakeGovernor.Create(s);
  try
    g.SetTemps([60]);
    for i := 1 to 20 do
      g.Work(100000, 100);
    for i := 1 to 40 do
      g.Work(100000, 10);
    CheckEqual(g.EffectiveDuty, 100, 'sensor in charge: heuristic off');
  finally
    g.Free;
  end;
  // switched off
  s := LoadGovernorSettings(100);
  s.WindowSeconds := 1;
  s.ThroughputHeuristic := false;
  g := TFakeGovernor.Create(s);
  try
    for i := 1 to 20 do
      g.Work(100000, 100);
    for i := 1 to 40 do
      g.Work(100000, 10);
    CheckEqual(g.EffectiveDuty, 100, 'heuristic off');
  finally
    g.Free;
  end;
end;

procedure TTestLoadGovernor.AbortEndsPauses;
var
  g: TFakeGovernor;
begin
  g := TFakeGovernor.Create(LoadGovernorSettings(50));
  try
    g.Work(10000, 1);
    CheckEqual(g.Paused, 10, 'before Abort: paused');
    g.Abort;
    Check(g.Aborted, 'Aborted');
    g.Work(10000, 1);
    CheckEqual(g.Paused, 10, 'after Abort: no further pause');
  finally
    g.Free;
  end;
end;

procedure TTestLoadGovernor.GovernedEmbedder;
var
  fake: TFakeEmbedder;
  inner, emb: IEmbedder;
  gov: TFakeGovernor;
  wrapper: TGovernedEmbedder;
  v: TLlmEmbedding;
  vs: TLlmEmbeddingDynArray;
  texts: TRawUtf8DynArray;
  i: integer;
  raised: boolean;
begin
  fake := TFakeEmbedder.Create;
  inner := fake;
  gov := TFakeGovernor.Create(LoadGovernorSettings(100));
  wrapper := TGovernedEmbedder.Create(inner, gov); // owns gov
  emb := wrapper;
  v := emb.Embed('abc');
  CheckUtf8((length(v) = 2) and (v[0] = 3), 'Embed passes through', []);
  // a batch of 5 in slices of 2: three inner calls, results in order
  wrapper.BatchSlice := 2;
  SetLength(texts, 5);
  for i := 0 to 4 do
    texts[i] := RawUtf8(StringOfChar('x', i + 1));
  vs := emb.EmbedBatch(texts);
  CheckEqual(fake.BatchCalls, 3, 'three slices');
  CheckEqual(length(vs), 5, 'five vectors');
  for i := 0 to 4 do
    CheckUtf8(vs[i][0] = i + 1, 'vector % in order', [i]);
  CheckUtf8(emb.Model = 'fake', 'Model passes through', []);
  // a failing text is still accounted for
  raised := false;
  try
    emb.Embed('boom');
  except
    on E: ESynException do
      raised := true;
  end;
  Check(raised, 'inner exception passes through');
  CheckEqual(gov.fWinUnits, 3 + 15 + 4, 'every unit accounted, the failed one too');
  emb := nil; // frees the governor too
  // nil inner: refused, and the handed-over governor is not leaked
  raised := false;
  try
    TGovernedEmbedder.Create(nil, TLoadGovernor.Create(LoadGovernorSettings));
  except
    on E: ESynException do
      raised := true;
  end;
  Check(raised, 'nil inner refused');
end;

procedure TTestLoadGovernor.SensorTree;
var
  root, chip: TFileName;
  {$ifdef FPC}
  {$ifdef OSPOSIX}
  link: TFileName;
  {$endif OSPOSIX}
  {$endif FPC}
begin
  root := TemporaryFileName + '.sys' + PathDelim;
  try
    // nothing at all
    CheckEqual(ReadCpuTemperatureIn(root), -1, 'empty tree');
    // a foreign chip only: ignored
    WriteSys(root, 'class/hwmon/hwmon0/name', 'acpitz');
    WriteSys(root, 'class/hwmon/hwmon0/temp1_input', '90000');
    CheckEqual(ReadCpuTemperatureIn(root), -1, 'foreign chip ignored');
    // the thermal zone fallback
    WriteSys(root, 'class/thermal/thermal_zone3/type', 'x86_pkg_temp');
    WriteSys(root, 'class/thermal/thermal_zone3/temp', '61000');
    CheckEqual(ReadCpuTemperatureIn(root), 61, 'thermal zone');
    // coretemp beyond hwmon31: the package value, not the hottest core
    chip := 'class/hwmon/hwmon40/';
    WriteSys(root, chip + 'name', 'coretemp');
    WriteSys(root, chip + 'temp1_input', '52400');
    WriteSys(root, chip + 'temp1_label', 'Package id 0');
    WriteSys(root, chip + 'temp2_input', '60000');
    WriteSys(root, chip + 'temp2_label', 'Core 0');
    CheckEqual(ReadCpuTemperatureIn(root), 52, 'coretemp package');
    // k10temp: Tdie preferred over the offset Tctl
    DirectoryDeleteAll(root + Sys(chip));
    chip := 'class/hwmon/hwmon2/';
    WriteSys(root, chip + 'name', 'k10temp');
    WriteSys(root, chip + 'temp1_input', '75000');
    WriteSys(root, chip + 'temp1_label', 'Tctl');
    WriteSys(root, chip + 'temp2_input', '55000');
    WriteSys(root, chip + 'temp2_label', 'Tdie');
    CheckEqual(ReadCpuTemperatureIn(root), 55, 'k10temp Tdie');
    {$ifdef FPC}
    {$ifdef OSPOSIX}
    // sysfs links its hwmon folders: the scan follows them
    DirectoryDeleteAll(root + Sys(chip));
    WriteSys(root, 'devices/cpu0/hwmon/name', 'coretemp');
    WriteSys(root, 'devices/cpu0/hwmon/temp1_input', '47000');
    link := root + 'class/hwmon/hwmon5';
    Check(fpSymlink(pchar(root + 'devices/cpu0/hwmon'), pchar(link)) = 0, 'symlink');
    CheckEqual(ReadCpuTemperatureIn(root), 47, 'linked hwmon folder');
    {$endif OSPOSIX}
    {$endif FPC}
  finally
    DirectoryDeleteAll(root);
    RemoveDir(root);
  end;
end;

procedure TTestLoadGovernor.CpuTemperature;
var
  t, i, seen: integer;
  f: TFileName;
begin
  // whatever the machine exposes: -1 or a plausible value
  t := ReadCpuTemperature;
  CheckUtf8((t = -1) or ((t > 0) and (t < 130)), 'temperature % plausible', [t]);
  if t < 0 then
    AddConsole('no CPU temperature readable here')
  else
    AddConsole('CPU temperature: % C', [t]);
  // the real sysfs: every hwmon chip has a name - read past its 4096-byte size
  seen := 0;
  for i := 0 to 15 do
  begin
    f := '/sys/class/hwmon/hwmon' + IntToStr(i) + '/name';
    if not FileExists(f) then
      continue;
    inc(seen);
    CheckUtf8(SysFileText(f) <> '', 'sysfs % read', [f]);
  end;
  if seen = 0 then
    AddConsole('no /sys/class/hwmon here');
end;

end.
