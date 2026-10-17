// Shared demo helpers for MCP examples
unit demo.mcp.shared;

interface

{$I mormot.defines.inc}

uses
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.rtti,
  mormot.core.unicode,
  mormot.core.text,
  mormot.core.variants,
  mormot.core.json,
  mormot.net.client,
  mormot.ai.mcp;

type
  TAddParams = record
    a: integer;
    b: integer;
  end;

  TIpInfo = packed record
    Ip: RawUtf8;
    Date: TDateTime;
  end;

  // typed: the schema and the parsing come from the record's RTTI
{$ifdef HASGENERICS}
  TAddTool = class(TMcpToolBase<TAddParams>)
  protected
    function ExecuteTyped(const aParams: TAddParams;
      const aAuthCtx: TMcpAuthContext): variant; override;
  end;
{$else}
  // without generics (Delphi 7) IMcpTool by hand: a record of two integers has
  // no RTTI there, so TMcpRecordTool could not describe it
  TAddTool = class(TInterfacedObject, IMcpTool)
  protected
    fName: RawUtf8;
    fDescription: RawUtf8;
  public
    constructor Create(const aName, aDescription: RawUtf8);
    function GetName: RawUtf8;
    function GetDescription: RawUtf8;
    function GetInputSchema: variant;
    function Execute(const aArgs: variant; const aAuthCtx: TMcpAuthContext): variant;
  end;
{$endif HASGENERICS}

  TGetIpTool = class(TInterfacedObject, IMcpTool)
  protected
    function FetchIp(out aIp: RawUtf8): boolean;
  public
    function GetName: RawUtf8;
    function GetDescription: RawUtf8;
    function GetInputSchema: variant;
    function Execute(const aArgs: variant; const aAuthCtx: TMcpAuthContext): variant;
  end;

  TVersionResource = class(TMcpResourceBase)
  protected
    function GetContent: RawUtf8; override;
  end;

procedure EnsureDemoRtti;
procedure RegisterDemoServices(const aServer: TMcpServer);
function CreateDemoServer(const aName, aVersion: RawUtf8): TMcpServer;

implementation

procedure EnsureDemoRtti;
begin
  {$ifdef HASGENERICS} // only the typed TAddTool reads its RTTI
  if not RecordHasFields(TypeInfo(TAddParams)) then
    Rtti.RegisterFromText(TypeInfo(TAddParams), 'a,b:integer');
  {$endif HASGENERICS}
  if not RecordHasFields(TypeInfo(TIpInfo)) then
    Rtti.RegisterFromText(TypeInfo(TIpInfo), 'Ip:RawUtf8 Date:TDateTime');
end;

{ TAddTool }

// the work of the tool, the same for both forms of TAddTool
function AddResult(const aParams: TAddParams): variant;
var
  builder: TMcpResponseBuilder;
begin
  builder := TMcpResponseBuilder.Create;
  try
    builder.AddText(FormatUtf8('% + % = %',
      [aParams.a, aParams.b, Int64(aParams.a) + aParams.b]));
    result := builder.Build;
  finally
    builder.Free;
  end;
end;

{$ifdef HASGENERICS}

function TAddTool.ExecuteTyped(const aParams: TAddParams;
  const aAuthCtx: TMcpAuthContext): variant;
begin
  result := AddResult(aParams);
end;

{$else}

constructor TAddTool.Create(const aName, aDescription: RawUtf8);
begin
  inherited Create;
  fName := aName;
  fDescription := aDescription;
end;

function TAddTool.GetName: RawUtf8;
begin
  result := fName;
end;

function TAddTool.GetDescription: RawUtf8;
begin
  result := fDescription;
end;

function TAddTool.GetInputSchema: variant;
begin
  // what TMcpRecordTool publishes for two integer fields
  result := _JsonFast('{"type":"object","properties":{' +
    '"a":{"type":"integer","minimum":-2147483648,"maximum":2147483647},' +
    '"b":{"type":"integer","minimum":-2147483648,"maximum":2147483647}},' +
    '"required":["a","b"]}');
end;

// an argument that is present, an integer JSON number and in the range of
// integer - as strict as TMcpRecordTool: "5" or 0.5 are refused
function IntArg(aArgs: PDocVariantData; const aName: RawUtf8;
  out aValue: integer): boolean;
var
  i: PtrInt;
  v: Int64;
begin
  result := false;
  i := aArgs^.GetValueIndex(aName);
  if (i < 0) or
     not (TVarData(aArgs^.Values[i]).VType in [varByte, varShortInt, varWord,
       varSmallInt, varLongWord, varInteger, varInt64, varWord64]) or
     not VariantToInt64(aArgs^.Values[i], v) or
     (v < low(integer)) or
     (v > high(integer)) then
    exit;
  aValue := v;
  result := true;
end;

function TAddTool.Execute(const aArgs: variant;
  const aAuthCtx: TMcpAuthContext): variant;
var
  args: PDocVariantData;
  params: TAddParams;
  builder: TMcpResponseBuilder;
begin
  args := _Safe(aArgs);
  if IntArg(args, 'a', params.a) and
     IntArg(args, 'b', params.b) then
    result := AddResult(params)
  else
  begin
    builder := TMcpResponseBuilder.Create;
    try
      builder.AddText('Invalid arguments: a and b must be integers').SetError;
      result := builder.Build;
    finally
      builder.Free;
    end;
  end;
end;

{$endif HASGENERICS}

{ TGetIpTool }

function TGetIpTool.FetchIp(out aIp: RawUtf8): boolean;
var
  body: RawUtf8;
  doc: PDocVariantData;
begin
  result := false;
  aIp := '';
  body := HttpGet('https://api.ipify.org?format=json');
  if body = '' then
    exit;
  doc := _Safe(_JsonFast(body));
  result := doc^.GetAsRawUtf8('ip', aIp) and (aIp <> '');
end;

function TGetIpTool.GetName: RawUtf8;
begin
  result := 'public_ip';
end;

function TGetIpTool.GetDescription: RawUtf8;
begin
  result := 'Fetch public IP address and current UTC timestamp';
end;

function TGetIpTool.GetInputSchema: variant;
var
  doc, props, req: TDocVariantData;
begin
  doc.InitObject(['type', 'object'], JSON_FAST);
  props.InitObject([], JSON_FAST);
  req.InitArray([], JSON_FAST);
  doc.AddValue('properties', variant(props));
  doc.AddValue('required', variant(req));
  result := variant(doc);
end;

function TGetIpTool.Execute(const aArgs: variant; const aAuthCtx: TMcpAuthContext): variant;
var
  builder: TMcpResponseBuilder;
  info: TIpInfo;
  json: RawUtf8;
begin
  EnsureDemoRtti;
  builder := TMcpResponseBuilder.Create;
  try
    if FetchIp(info.Ip) then
    begin
      info.Date := NowUtc;
      json := RecordSaveJson(info, TypeInfo(TIpInfo));
      builder.AddText(json);
    end
    else
      builder.AddText('{"error":"Unable to fetch public IP"}');
    result := builder.Build;
  finally
    builder.Free;
  end;
end;

{ TVersionResource }

function TVersionResource.GetContent: RawUtf8;
begin
  result := '{"version":"1.0.0","protocol":"MCP"}';
end;

procedure RegisterDemoServices(const aServer: TMcpServer);
var
  addTool: IMcpTool;
  ipTool: IMcpTool;
  versionRes: IMcpResource;
begin
  EnsureDemoRtti;
  addTool := TAddTool.Create('add', 'Add two numbers');
  ipTool := TGetIpTool.Create;
  versionRes := TVersionResource.Create('version://info', 'Version',
    'Server version information', 'application/json');
  aServer.RegisterTool(addTool);
  aServer.RegisterTool(ipTool);
  aServer.RegisterResource(versionRes);
end;

function CreateDemoServer(const aName, aVersion: RawUtf8): TMcpServer;
begin
  result := TMcpServer.Create(aName, aVersion);
  RegisterDemoServices(result);
  result.Start;
end;

end.
