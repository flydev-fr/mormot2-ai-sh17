unit main;

interface

{$I mormot.defines.inc}

{$ifdef OSWINDOWS}
  {$apptype console}
{$endif OSWINDOWS}

{.$define WITH_LOGS}

uses
  {$I mormot.uses.inc}
  sysutils,
  math,
  {$ifdef OSWINDOWS}
  Windows,
  {$endif}
  Types,
  mormot.core.base,
  mormot.core.os,
  mormot.core.rtti,
  mormot.core.unicode,
  mormot.core.text,
  mormot.core.variants,
  mormot.core.json,
  mormot.core.log,
  mormot.net.client,
  mormot.ai.mcp,
  mormot.ai.mcp.server,
  mormot.ai.mcp.stdio,
  mormot.ai.mcp.tools;
  
  
type
  // Example tool parameter type
  TAddParams = record
    a: integer;
    b: integer;
  end;

  // Demo output DTO for public IP
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

  // Demo tool implementation fetching public IP
  // - fetch ip from https://api.ipify.org
  TGetIpTool = class(TInterfacedObject, IMcpTool)
  protected
    function FetchIp(out aIp: RawUtf8): boolean;
  public
    function GetName: RawUtf8;
    function GetDescription: RawUtf8;
    function GetInputSchema: variant;
    function Execute(const aArgs: variant; const aAuthCtx: TMcpAuthContext): variant;
  end;

  // Example resource implementation
  TVersionResource = class(TMcpResourceBase)
  protected
    function GetContent: RawUtf8; override;
  end;


procedure TestStdioTransport;
procedure TestHttpTransport;
procedure TestJsonRpc;
procedure StartExample;

implementation


procedure StartExample;
{$ifdef WITH_LOGS}
var
  family: TSynLogFamily;
{$endif WITH_LOGS}
begin
  // setup logs
  {$ifdef WITH_LOGS}
  family := TSynLog.Family;
  family.Level := LOG_VERBOSE;
  family.EchoToConsole := LOG_VERBOSE;
  family.EchoToConsoleBackground := True;
  family.HighResolutionTimestamp := true;
  family.PerThreadLog := ptIdentifiedInOneFile;
  family.NoFile := True;
  {$endif WITH_LOGS}

  try
    ConsoleWrite('mORMot MCP Server Example', ccLightCyan);
    ConsoleWrite('================================', ccLightGray);
    ConsoleWrite('Build: demo server', ccLightGray);
    ConsoleWrite('Modes: jsonrpc (default), http, stdio', ccLightGray);
    ConsoleWriteLn;

    if ParamCount > 0 then
    begin
      if ParamStr(1) = 'http' then
        TestHttpTransport
      else if ParamStr(1) = 'stdio' then
        TestStdioTransport
      else
        ConsoleWrite('Usage: % [http|stdio] (default: jsonrpc)', [ParamStr(0)], ccYellow);
    end
    else
      TestJsonRpc;

  except
    on E: Exception do
      ConsoleWrite('ERROR - %: %', [E.ClassName, E.Message], ccLightRed);
  end;
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

procedure EnsureIpInfoRtti;
begin
  Rtti.RegisterFromText(TypeInfo(TIpInfo), 'ip:RawUtf8 date:TDateTime');
end;

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
  EnsureIpInfoRtti;
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

procedure TestJsonRpc;
var
  server: TMcpServer;
  addTool: IMcpTool;
  ipTool: IMcpTool;
  paintTool: IMcpTool;
  versionRes: IMcpResource;
  request, response: RawUtf8;
begin
  ConsoleWrite('JSON-RPC demo (default)', ccLightCyan);
  ConsoleWrite('About: runs in-memory JSON-RPC requests to validate tools/resources.', ccLightGray);
  ConsoleWrite('Status: starting...', ccLightGray);
  ConsoleWriteLn;

  server := TMcpServer.Create('TestServer', '1.0');
  try
    // Register tool and resource
    addTool := TAddTool.Create('add', 'Add two numbers');
    server.RegisterTool(addTool);
    ipTool := TGetIpTool.Create;
    server.RegisterTool(ipTool);

    versionRes := TVersionResource.Create('version://info', 'Version',
      'Server version information', 'application/json');
    server.RegisterResource(versionRes);

    server.Start;

    // Test server/discover
    request := '{"jsonrpc":"2.0","id":1,"method":"server/discover","params":{"_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientCapabilities":{}}}}';
    response := server.ExecuteRequest(request);
    ConsoleWrite('Step 1/5 - Discover:'#13#10+'%', [response], ccLightMagenta);
    ConsoleWriteLn;

    // Test tools/list
    request := '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{"_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientCapabilities":{}}}}';
    response := server.ExecuteRequest(request);
    ConsoleWrite('Step 2/5 - Tools list:'#13#10+'%', [response], ccLightCyan);
    ConsoleWriteLn;

    // Test tools/call — like every request since 2026-07-28, it must carry its
    // own protocol version and client capabilities (there is no handshake)
    request := _Safe(_ObjFast(['jsonrpc', '2.0', 'id', 3, 'method', 'tools/call',
      'params', McpRequestParams(_ObjFast([
        'name', 'add', 'arguments', _ObjFast(['a', 5, 'b', 3])]))]))^.ToJson;
    response := server.ExecuteRequest(request);
    ConsoleWrite('Step 3/5 - Tool call:'#13#10+'%', [response], ccLightBlue);
    ConsoleWriteLn;

    // Test resources/list
    request := '{"jsonrpc":"2.0","id":4,"method":"resources/list","params":{"_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientCapabilities":{}}}}';
    response := server.ExecuteRequest(request);
    ConsoleWrite('Step 4/5 - Resources list:'#13#10+'%', [response], ccLightGreen);
    ConsoleWriteLn;

    // Test resources/read
    request := _Safe(_ObjFast(['jsonrpc', '2.0', 'id', 5,
      'method', 'resources/read',
      'params', McpRequestParams(_ObjFast(['uri', 'version://info']))]))^.ToJson;
    response := server.ExecuteRequest(request);
    ConsoleWrite('Step 5/5 - Resource read:'#13#10+'%', [response], ccGreen);
    ConsoleWriteLn;
    ConsoleWrite('SUCCESS - JSON-RPC demo completed.', ccLightGreen);
    
  finally
    server.Free;
  end;
end;

procedure TestHttpTransport;
var
  server: TMcpServer;
  transport: TMcpHttpTransport;
  addTool: IMcpTool;
  ipTool: IMcpTool;
  paintTool: IMcpTool;
begin
  ConsoleWrite('HTTP transport demo', ccLightCyan);
  ConsoleWrite('About: starts an HTTP /mcp endpoint for JSON-RPC.', ccLightGray);
  ConsoleWrite('Status: starting...', ccLightGray);

  server := TMcpServer.Create('HttpTestServer', '1.0');
  try
    addTool := TAddTool.Create('add', 'Add two numbers');
    server.RegisterTool(addTool);
    ipTool := TGetIpTool.Create;
    server.RegisterTool(ipTool);
    server.Start;
    
    transport := TMcpHttpTransport.Create(server);
    try
      transport.Port := 8080;
      transport.Start;
      
      ConsoleWrite('SUCCESS - listening on http://localhost:%/mcp', [transport.Port], ccLightGreen);
      ConsoleWrite('Quick test:', ccLightGray);
      ConsoleWrite('  curl -X POST http://localhost:%/mcp -H "Content-Type: application/json" -d ''{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{"_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientCapabilities":{}}}}''', [transport.Port], ccYellow);
      ConsoleWrite('Stop: press Enter to stop the server.', ccLightGray);
      ConsoleWaitForEnterKey;
      ConsoleWrite('Status: stopped.', ccLightGray);
      
    finally
      transport.Free;
    end;
  finally
    server.Free;
  end;
end;

procedure TestStdioTransport;
var
  server: TMcpServer;
  transport: TMcpStdioTransport;
  addTool: IMcpTool;
  ipTool: IMcpTool;
  paintTool: IMcpTool;
begin
  ConsoleWrite('STDIO transport demo', ccLightCyan);
  ConsoleWrite('About: JSON-RPC over stdin/stdout for piping.', ccLightGray);
  ConsoleWrite('Status: ready - stdin in, stdout out', ccLightGray);
  ConsoleWrite('Example request: {"jsonrpc":"2.0","id":1,"method":"tools/list","params":{"_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientCapabilities":{}}}}', ccYellow);
  
  server := TMcpServer.Create('StdioTestServer', '1.0');
  try
    addTool := TAddTool.Create('add', 'Add two numbers');
    server.RegisterTool(addTool);
    ipTool := TGetIpTool.Create;
    server.RegisterTool(ipTool);
    server.Start;

    transport := TMcpStdioTransport.Create(server);
    try
      transport.Start;
      ConsoleWrite('SUCCESS - waiting for requests (Ctrl+C to stop).', ccLightGreen);
      
      // Wait for Ctrl+C or EOF
      while transport.IsActive do
        Sleep(100);
        
    finally
      transport.Free;
    end;
  finally
    server.Free;
  end;
end;

initialization

EnsureIpInfoRtti;


end.
