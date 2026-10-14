/// MCP Stdio Transport - Standard Input/Output for CLI Integration
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - adopted into the mormot.ai.* namespace for landrix (LandrixAI) from
//   flydev-fr/mormot2-extensions
unit mormot.ai.mcp.stdio;

{
  *****************************************************************************

   MCP Stdio Transport Implementation
    - Line-based JSON-RPC over stdin/stdout
    - Worker thread for non-blocking input
    - Suitable for CLI tool integration

  *****************************************************************************
}

interface

{$I mormot.defines.inc}

uses
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.core.threads,
  mormot.ai.mcp,
  mormot.ai.mcp.legacy; // LEGACY-ERA


{ ************ Stdio Transport }

{$I-} // avoid io check in writeln()

/// prepare a text input for MCP stdio: the messages are UTF-8 on every platform
// - sets the UTF-8 code page where the RTL decodes Text I/O (FPC, Delphi 2009+);
//   pre-Unicode Delphi (7) reads bytes unchanged, nothing to set there
procedure McpPrepareInput(var aInput: Text);

/// read one line of an MCP stdio stream as UTF-8
// - false at end of input or on an I/O error
// - the decoding follows McpPrepareInput: the result is exactly the UTF-8 the
//   client wrote, on every compiler
function McpReadLine(var aInput: Text; out aLine: RawUtf8): boolean;

type
  /// Worker thread for reading from stdin
  // - runs in background to avoid blocking main thread
  TMcpStdioWorker = class(TSynThread)
  private
    fTransport: TObject;  // TMcpStdioTransport
    fInputClosed: boolean;
    procedure ProcessLine(const aLine: RawUtf8);
  protected
    procedure Execute; override;
  public
    constructor Create(aTransport: TObject); reintroduce;
  end;

  /// Stdio transport for MCP over standard input/output
  // - reads JSON-RPC requests line-by-line from stdin
  // - writes JSON-RPC responses to stdout
  // - suitable for CLI tools and process-based integrations
  // - the session ends when the client closes stdin: IsActive turns false
  TMcpStdioTransport = class(TObject)
  private
    fServer: TMcpServer;
    fWorker: TMcpStdioWorker;
    fActive: boolean;
    fOutputLock: TLightLock;
    fLegacy: TMcpLegacyStdioBridge;     // LEGACY-ERA
    fAcceptLegacyInitialize: boolean;   // LEGACY-ERA
    procedure WriteOutput(const aLine: RawUtf8);
  public
    /// initialize with MCP server instance
    constructor Create(aServer: TMcpServer);
    /// finalize and cleanup
    destructor Destroy; override;
    /// start stdio communication
    procedure Start;
    /// stop stdio communication
    // - the worker reads stdin blocking: while the client keeps stdin open and
    // silent, Stop waits for its next line (or for EOF)
    procedure Stop;
    /// check if transport is active
    // - false once stopped, and once the client closed stdin (EOF)
    function IsActive: boolean;
    /// process a single request (called by worker thread)
    procedure ProcessRequest(const aRequest: RawUtf8);
    /// LEGACY-ERA: also answer the pre-2026-07-28 "initialize" handshake
    // - true by default: clients that open stdio with "initialize" would
    //   otherwise never connect (see mormot.ai.mcp.legacy for why and when
    //   this goes away)
    // - false = modern only: "initialize" is rejected like any request
    //   without _meta
    property AcceptLegacyInitialize: boolean
      read fAcceptLegacyInitialize write fAcceptLegacyInitialize;
  end;


implementation

uses
  classes,
  sysutils,
  mormot.core.json;

{ ************ TMcpStdioWorker }

constructor TMcpStdioWorker.Create(aTransport: TObject);
begin
  inherited Create(false);  // Start immediately
  FreeOnTerminate := false;
  fTransport := aTransport;
end;

procedure TMcpStdioWorker.ProcessLine(const aLine: RawUtf8);
begin
  if (fTransport <> nil) and (aLine <> '') then
    TMcpStdioTransport(fTransport).ProcessRequest(aLine);
end;

procedure McpPrepareInput(var aInput: Text);
begin
  // Text I/O decodes with the system code page by default (ANSI on Windows, for
  // Delphi and FPC alike); MCP stdio messages are UTF-8 on every platform
  {$ifdef HASCODEPAGE}
  SetTextCodePage(aInput, CP_UTF8);
  {$endif HASCODEPAGE}
end;

function McpReadLine(var aInput: Text; out aLine: RawUtf8): boolean;
var
  line: string;
begin
  result := false;
  aLine := '';
  if Eof(aInput) then
    exit;
  ReadLn(aInput, line);
  if IOResult <> 0 then
    exit;
  {$ifdef HASCODEPAGE}
  // the RTL decoded UTF-8 into the native string type (code page set above)
  aLine := StringToUtf8(line);
  {$else}
  // pre-Unicode Delphi: string is an AnsiString holding the bytes as they
  // came - already the UTF-8 the client sent. StringToUtf8 would treat them
  // as ANSI and encode every non-ASCII character a second time.
  aLine := RawUtf8(line);
  {$endif HASCODEPAGE}
  result := true;
end;

procedure TMcpStdioWorker.Execute;
var
  line: RawUtf8;
begin
  // Blocking read loop. ReadLn waits for the next request by itself - the loop
  // used to sleep 500 ms BEFORE every read, adding half a second to each
  // request. It also never noticed EOF: {$I-} suppresses the I/O exception it
  // waited for, so a client closing stdin left the server process running.
  McpPrepareInput(Input);
  try
    while not Terminated do
    begin
      if not McpReadLine(Input, line) then
        break;            // the client closed stdin: the session is over
      ProcessLine(line);
    end;
  finally
    fInputClosed := true;
  end;
end;


{ ************ TMcpStdioTransport }

constructor TMcpStdioTransport.Create(aServer: TMcpServer);
begin
  inherited Create;
  fServer := aServer;
  fActive := false;
  fOutputLock.Init;
  fLegacy := TMcpLegacyStdioBridge.Create(aServer); // LEGACY-ERA
  fAcceptLegacyInitialize := true;                   // LEGACY-ERA
end;

destructor TMcpStdioTransport.Destroy;
begin
  Stop;
  fLegacy.Free; // LEGACY-ERA
  fOutputLock.Done;
  inherited;
end;

procedure TMcpStdioTransport.WriteOutput(const aLine: RawUtf8);
begin
  fOutputLock.Lock;
  try
    // MCP stdio messages are UTF-8 on every platform. Declaring the Output text
    // file as UTF-8 lets the RTL pass the RawUtf8 bytes through unchanged. The
    // Windows branch used to Utf8Decode into an AnsiString instead, which turned
    // every character outside the ANSI code page into '?'.
    // Still through Output, not the raw handle: a host (or a test) may redirect
    // Output with AssignFile. Set on every write, because that redirection
    // resets the code page of the text file.
    {$ifdef HASCODEPAGE}
    SetTextCodePage(Output, CP_UTF8);
    {$endif HASCODEPAGE}
    WriteLn(aLine);
    Flush(Output);  // Ensure immediate delivery
  finally
    fOutputLock.UnLock;
  end;
end;

procedure TMcpStdioTransport.ProcessRequest(const aRequest: RawUtf8);
var
  request, response: RawUtf8;
  route: TMcpLegacyRoute; // LEGACY-ERA
begin
  if not fActive then
    exit;

  try
    request := aRequest;
    // LEGACY-ERA: a client of the "initialize" era is answered or translated by
    // the bridge; everything else reaches the core exactly as it came in
    if fAcceptLegacyInitialize then
      route := fLegacy.Route(request, response)
    else
      route := lrPassThrough;
    if route <> lrAnswered then
    begin
      // Execute MCP request
      response := fServer.ExecuteRequest(request);
      if (route = lrRewritten) and
         (response <> '') then
        response := fLegacy.AdaptResponse(response); // LEGACY-ERA
    end;

    // Send response if not a notification
    if response <> '' then
      WriteOutput(response);

  except
    on E: Exception do
      // Any exception, not only ESynException: an uncaught one would end the
      // worker thread silently and leave a server that reads no more requests.
      // The message goes through QuotedStrJson - a quote or backslash in it
      // used to produce invalid JSON.
      WriteOutput('{"jsonrpc":"2.0","error":{"code":-32603,"message":' +
        QuotedStrJson(StringToUtf8(E.Message)) + '},"id":null}');
  end;
end;

procedure TMcpStdioTransport.Start;
begin
  if fActive then
    exit;

  fActive := true;
  if RunFromSynTests then
    exit;

  // Start worker thread
  fWorker := TMcpStdioWorker.Create(self);
end;

procedure TMcpStdioTransport.Stop;
begin
  if not fActive then
    exit;

  fActive := false;

  // Stop worker thread
  if fWorker <> nil then
  begin
    fWorker.Terminate;
    fWorker.WaitFor;
    FreeAndNilSafe(fWorker);
  end;
end;

function TMcpStdioTransport.IsActive: boolean;
begin
  // EOF on stdin ends the session, so a host loop "while IsActive" returns and
  // the process exits instead of lingering after the client went away
  result := fActive and
            ((fWorker = nil) or
             not fWorker.fInputClosed);
end;


end.
