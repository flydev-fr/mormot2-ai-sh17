/// tests for the native ILlmClient (phase 2d, mormot.ai.llm.native)
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - the output parser and the prompt run on a tiny byte-level tokenizer and
//   need no model: every byte is one token, so a multi-byte character always
//   arrives cut in pieces
// - with vendor/models/qwen3-0.6b the client runs the reference prompts of
//   tests/data/infer-ref/qwen3 (complete and streamed, a tool call, a think
//   block cut by the token cap) and one agent loop with a tool
unit test.infer.llm;

interface

{$I mormot.defines.inc}

uses
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.core.variants,
  mormot.core.json,
  mormot.core.perf,
  mormot.core.test,
  mormot.ai.llm.types,
  mormot.ai.llm,
  mormot.ai.agent,
  mormot.ai.llm.native,
  mormot.ai.infer.tensor,
  mormot.ai.infer.unicode,
  mormot.ai.infer.tokenizer.bpe,
  mormot.ai.infer.chat,
  mormot.ai.infer.backend.cpu,
  mormot.ai.infer.model.qwen3,
  test.infer.ref,
  test.infer.bpe;

type
  /// collects what a parser or client gives out as stream deltas
  TDeltaLog = class
  public
    Deltas, Dones, Usages, Invalid: integer;
    AfterDone: boolean;
    FirstRole, Content, Reasoning, Finish: RawUtf8;
    Usage: TLlmUsage;
    CallIndex: TIntegerDynArray;
    CallIds, CallNames, CallArgs: TRawUtf8DynArray;
    procedure OnDelta(const aDelta: TLlmStreamDelta);
  end;

  /// phase 2d: the native ILlmClient
  TTestInferLlmNative = class(TSynTestCase)
  protected
    fTokenizer: TBpeTokenizer;
    fTemplate: TQwen3ChatTemplate;
    fToolArgs: RawUtf8;
    fToolCalls: integer;
    procedure Setup; override;
    procedure CleanUp; override;
    /// feed the bytes of aText, one token each
    procedure Feed(aParser: TQwen3OutputParser; const aText: RawUtf8);
    /// the real model folder, '' (and SKIP or a failure) when absent
    function RealModel: TFileName;
    /// the tool of the agent test
    function Weather(const aArgumentsJson: RawUtf8): RawUtf8;
  published
    /// text in cut characters; newlines at the ends; role, finish, usage, done
    procedure ParserPlainText;
    /// a leading <think> block is reasoning, not content
    procedure ParserThinking;
    /// two tool calls with Python-formatted arguments
    procedure ParserToolCalls;
    /// malformed and cut tool calls stay text; a cut think block
    procedure ParserMalformedAndCut;
    /// invalid UTF-8, control tokens inside the text, wrong ids
    procedure ParserInvalidUtf8AndControls;
    /// ParseQwen3ToolCall accepts exactly {"name": string, "arguments": object}
    procedure ToolCallJson;
    /// NativeChatPrompt: the template, enable_thinking, refusals
    procedure ChatPrompt;
    /// NativeDecode on the tiny Qwen3: the reference tokens, the context limit
    procedure DecodeOnTinyModel;
    /// Qwen3-0.6B behind ILlmClient against the reference continuations
    procedure Qwen3ChatClient;
    /// TLlmAgent over the native client: a tool call and its answer
    procedure Qwen3AgentLoop;
  end;


implementation

const
  IM_END = 300;
  IM_START = 301;
  THINK = 302;
  THINK_END = 303;
  TOOL_CALL = 304;
  TOOL_CALL_END = 305;
  END_OF_TEXT = 308;

/// bytes from hexadecimal: non-ASCII test data without non-ASCII source
function Bin(const aHex: RawUtf8): RawUtf8;
begin
  result := '';
  FastSetString(result, length(aHex) shr 1);
  HexToBin(pointer(aHex), pointer(result), length(result));
end;


{ TDeltaLog }

procedure TDeltaLog.OnDelta(const aDelta: TLlmStreamDelta);
var
  r: RawUtf8;
  n: PtrInt;
begin
  inc(Deltas);
  if Deltas = 1 then
    FirstRole := aDelta.Role;
  if Dones > 0 then
    AfterDone := true;
  if not IsStrictUtf8(pointer(aDelta.ContentDelta), length(aDelta.ContentDelta)) then
    inc(Invalid);
  Content := Content + aDelta.ContentDelta;
  r := _Safe(aDelta.Raw)^.A['choices']^._[0]^.O['delta']^.U['reasoning_content'];
  if not IsStrictUtf8(pointer(r), length(r)) then
    inc(Invalid);
  Reasoning := Reasoning + r;
  if aDelta.HasToolCall then
  begin
    n := length(CallIds);
    SetLength(CallIndex, n + 1);
    SetLength(CallIds, n + 1);
    SetLength(CallNames, n + 1);
    SetLength(CallArgs, n + 1);
    CallIndex[n] := aDelta.ToolCallIndex;
    CallIds[n] := aDelta.ToolCallId;
    CallNames[n] := aDelta.ToolCallName;
    CallArgs[n] := aDelta.ToolCallArgsDelta;
  end;
  if aDelta.FinishReason <> '' then
    Finish := aDelta.FinishReason;
  if aDelta.HasUsage then
  begin
    inc(Usages);
    Usage := aDelta.Usage;
  end;
  if aDelta.Done then
    inc(Dones);
end;


{ TTestInferLlmNative }

procedure TTestInferLlmNative.Setup;
var
  json, err: RawUtf8;
begin
  json := StringReplaceAll(TinyBpeJson('{"type":"NFC"}', '"aa":256', '[["a","a"]]'),
    '"added_tokens":[{"id":300,"content":"<|im_end|>","special":true}]',
    '"added_tokens":[{"id":300,"content":"<|im_end|>","special":true},' +
    '{"id":301,"content":"<|im_start|>","special":true},' +
    '{"id":302,"content":"<think>","special":false},' +
    '{"id":303,"content":"</think>","special":false},' +
    '{"id":304,"content":"<tool_call>","special":false},' +
    '{"id":305,"content":"</tool_call>","special":false},' +
    '{"id":306,"content":"<tool_response>","special":false},' +
    '{"id":307,"content":"</tool_response>","special":false},' +
    '{"id":308,"content":"<|endoftext|>","special":true}]');
  fTokenizer := LoadBpe(json, err);
  if CheckFailed(fTokenizer <> nil, Utf8ToString(err)) then
    exit;
  fTemplate := TQwen3ChatTemplate.Create(fTokenizer);
end;

procedure TTestInferLlmNative.CleanUp;
begin
  FreeAndNil(fTemplate);
  FreeAndNil(fTokenizer);
end;

procedure TTestInferLlmNative.Feed(aParser: TQwen3OutputParser;
  const aText: RawUtf8);
var
  i: PtrInt;
begin
  for i := 1 to length(aText) do
    if aParser.Push(ord(aText[i])) then
      Check(false, 'a text byte is no stop token');
end;

procedure TTestInferLlmNative.ParserPlainText;
var
  log: TDeltaLog;
  p: TQwen3OutputParser;
  text: RawUtf8;
begin
  // "Gr" u-umlaut "sse " CJK, an emoji, "!" - the two-, three- and four-byte
  // characters arrive one byte per token
  text := 'Gr' + Bin('c3bc') + 'sse ' + Bin('e4b896') + Bin('f09f9880') + '!';
  log := TDeltaLog.Create;
  p := TQwen3OutputParser.Create(fTokenizer, log.OnDelta);
  try
    Feed(p, #10#10 + text + #10 + 'second line' + #10#10);
    Check(p.Push(IM_END), 'im_end stops');
    p.Finish(false, 7);
    CheckEqual(p.Response.Content, text + #10 + 'second line'#10#10,
      'leading newlines dropped, the others kept (rendered verbatim)');
    Check(p.Response.FinishReason = lfrStop, 'stop');
    Check(p.Response.ToolCalls = nil, 'no tool call');
    CheckEqual(p.Response.Usage.PromptTokens, 7, 'prompt tokens');
    CheckEqual(p.Response.Usage.CompletionTokens, length(text) + 17, 'every push counted');
    CheckEqual(p.Response.Usage.TotalTokens, length(text) + 24, 'total');
    Check(VarIsEmptyOrNull(p.Response.Raw), 'no reasoning, no Raw');
    CheckEqual(log.Content, p.Response.Content, 'stream = complete');
    CheckEqual(log.Invalid, 0, 'every delta valid UTF-8');
    CheckEqual(log.FirstRole, 'assistant', 'role on the first delta');
    CheckEqual(log.Finish, 'stop', 'finish delta');
    CheckEqual(log.Usages, 1, 'one usage');
    CheckEqual(log.Usage.CompletionTokens, p.Response.Usage.CompletionTokens,
      'stream usage');
    CheckEqual(log.Dones, 1, 'one Done');
    Check(not log.AfterDone, 'Done is the last delta');
    Check(log.Deltas > 4, 'given out piece by piece');
  finally
    p.Free;
    log.Free;
  end;
  // no callback: the same response
  p := TQwen3OutputParser.Create(fTokenizer);
  try
    Feed(p, #10#10 + text + #10 + 'second line' + #10#10);
    Check(p.Push(END_OF_TEXT), 'endoftext stops too');
    p.Finish(false, 7);
    CheckEqual(p.Response.Content, text + #10 + 'second line'#10#10,
      'without stream');
  finally
    p.Free;
  end;
end;

procedure TTestInferLlmNative.ParserThinking;
var
  log: TDeltaLog;
  p: TQwen3OutputParser;
begin
  log := TDeltaLog.Create;
  p := TQwen3OutputParser.Create(fTokenizer, log.OnDelta);
  try
    p.Push(THINK);
    Feed(p, #10'Let me think.'#10'Twice.'#10);
    p.Push(THINK_END);
    Feed(p, #10#10'The answer.');
    p.Push(IM_END);
    p.Finish(false, 3);
    CheckEqual(p.Reasoning, 'Let me think.'#10'Twice.', 'reasoning');
    CheckEqual(p.Response.Content, 'The answer.', 'content');
    CheckEqual(_Safe(p.Response.Raw)^.A['choices']^._[0]^.O['message']^.
      U['reasoning_content'], p.Reasoning, 'reasoning_content in Raw');
    CheckEqual(log.Reasoning, p.Reasoning, 'streamed reasoning');
    CheckEqual(log.Content, 'The answer.', 'streamed content');
    CheckEqual(log.Invalid, 0, 'valid UTF-8');
  finally
    p.Free;
    log.Free;
  end;
  // the empty block of enable_thinking = false comes from the prompt, so the
  // answer starts with text; a <think> later on is text
  p := TQwen3OutputParser.Create(fTokenizer);
  try
    Feed(p, 'a');
    p.Push(THINK);
    Feed(p, 'b');
    p.Push(IM_END);
    p.Finish(false, 0);
    CheckEqual(p.Response.Content, 'a<think>b', 'think token inside the text');
    CheckEqual(p.Reasoning, '', 'no reasoning');
  finally
    p.Free;
  end;
end;

procedure TTestInferLlmNative.ParserToolCalls;
var
  log: TDeltaLog;
  p: TQwen3OutputParser;
  r: TLlmChatResponse;
begin
  log := TDeltaLog.Create;
  p := TQwen3OutputParser.Create(fTokenizer, log.OnDelta);
  try
    Feed(p, 'Sure.'#10);
    p.Push(TOOL_CALL);
    Feed(p, #10'{"name": "get_weather", "arguments": {"city": "Berlin", ' +
      '"days": 2.50}}'#10);
    p.Push(TOOL_CALL_END);
    Feed(p, #10);
    p.Push(TOOL_CALL);
    Feed(p, '{"arguments":{},"name":"now"}');
    p.Push(TOOL_CALL_END);
    p.Push(IM_END);
    p.Finish(false, 10);
    r := p.Response;
    CheckEqual(r.Content, 'Sure.', 'content before the calls');
    Check(r.FinishReason = lfrToolCalls, 'finish tool_calls');
    if CheckFailed(length(r.ToolCalls) = 2, 'two calls') then
      exit;
    CheckEqual(r.ToolCalls[0].Name, 'get_weather', 'name 0');
    CheckEqual(r.ToolCalls[0].ArgumentsJson, '{"city": "Berlin", "days": 2.5}',
      'arguments as json.dumps writes them');
    CheckEqual(r.ToolCalls[1].Name, 'now', 'name 1');
    CheckEqual(r.ToolCalls[1].ArgumentsJson, '{}', 'empty arguments');
    Check(IdemPChar(pointer(r.ToolCalls[0].Id), 'CALL_'), 'id 0');
    Check(length(r.ToolCalls[0].Id) > 8, 'id 0 length');
    Check(r.ToolCalls[0].Id <> r.ToolCalls[1].Id, 'ids differ');
    if CheckFailed(length(log.CallIds) = 2, 'two streamed calls') then
      exit;
    CheckEqual(log.CallIndex[0], 0, 'index 0');
    CheckEqual(log.CallIndex[1], 1, 'index 1');
    CheckEqual(log.CallIds[1], r.ToolCalls[1].Id, 'streamed id');
    CheckEqual(log.CallNames[0], 'get_weather', 'streamed name');
    CheckEqual(log.CallArgs[0], r.ToolCalls[0].ArgumentsJson, 'streamed arguments');
    CheckEqual(log.Content, 'Sure.', 'streamed content');
    CheckEqual(log.Finish, 'tool_calls', 'streamed finish');
  finally
    p.Free;
    log.Free;
  end;
end;

procedure TTestInferLlmNative.ParserMalformedAndCut;
var
  p: TQwen3OutputParser;
begin
  // reasoning, then a call: no content in between
  p := TQwen3OutputParser.Create(fTokenizer);
  try
    p.Push(THINK);
    Feed(p, #10'I need the weather.'#10);
    p.Push(THINK_END);
    Feed(p, #10#10);
    p.Push(TOOL_CALL);
    Feed(p, #10'{"name": "w", "arguments": {}}'#10);
    p.Push(TOOL_CALL_END);
    p.Push(IM_END);
    p.Finish(false, 0);
    CheckEqual(p.Reasoning, 'I need the weather.', 'reasoning before the call');
    CheckEqual(p.Response.Content, '', 'no content');
    CheckEqual(length(p.Response.ToolCalls), 1, 'the call');
    Check(p.Response.FinishReason = lfrToolCalls, 'tool_calls');
  finally
    p.Free;
  end;
  // text after a call: the newline before the call is the template's
  p := TQwen3OutputParser.Create(fTokenizer);
  try
    Feed(p, 'A'#10);
    p.Push(TOOL_CALL);
    Feed(p, '{"name": "w", "arguments": {}}');
    p.Push(TOOL_CALL_END);
    Feed(p, #10'B'#10);
    p.Push(IM_END);
    p.Finish(false, 0);
    CheckEqual(p.Response.Content, 'A'#10'B', 'separators dropped');
  finally
    p.Free;
  end;
  // no arguments: kept as text, no call - with the newline before it
  p := TQwen3OutputParser.Create(fTokenizer);
  try
    Feed(p, 'Hi'#10);
    p.Push(TOOL_CALL);
    Feed(p, '{"name": "f"}');
    p.Push(TOOL_CALL_END);
    p.Push(IM_END);
    p.Finish(false, 0);
    CheckEqual(p.Response.Content, 'Hi'#10'<tool_call>{"name": "f"}</tool_call>',
      'malformed call is text');
    Check(p.Response.ToolCalls = nil, 'no call');
    Check(p.Response.FinishReason = lfrStop, 'stop');
  finally
    p.Free;
  end;
  // cut by the token cap inside a call
  p := TQwen3OutputParser.Create(fTokenizer);
  try
    Feed(p, 'x'#10);
    p.Push(TOOL_CALL);
    Feed(p, '{"name": "f", "argu');
    p.Finish(true, 0);
    CheckEqual(p.Response.Content, 'x'#10'<tool_call>{"name": "f", "argu',
      'open call is text');
    Check(p.Response.ToolCalls = nil, 'no call');
    Check(p.Response.FinishReason = lfrLength, 'length');
  finally
    p.Free;
  end;
  // cut inside the think block: all reasoning, no content
  p := TQwen3OutputParser.Create(fTokenizer);
  try
    p.Push(THINK);
    Feed(p, #10'abc'#10);
    p.Finish(true, 0);
    CheckEqual(p.Reasoning, 'abc', 'cut reasoning');
    CheckEqual(p.Response.Content, '', 'no content');
    Check(p.Response.FinishReason = lfrLength, 'length');
  finally
    p.Free;
  end;
  // a cut call does not turn finish into tool_calls when an earlier one exists
  p := TQwen3OutputParser.Create(fTokenizer);
  try
    p.Push(TOOL_CALL);
    Feed(p, '{"name": "f", "arguments": {}}');
    p.Push(TOOL_CALL_END);
    p.Finish(true, 0);
    CheckEqual(length(p.Response.ToolCalls), 1, 'complete call kept');
    Check(p.Response.FinishReason = lfrLength, 'length wins');
  finally
    p.Free;
  end;
end;

procedure TTestInferLlmNative.ParserInvalidUtf8AndControls;
var
  log: TDeltaLog;
  p: TQwen3OutputParser;
  raised: boolean;
begin
  log := TDeltaLog.Create;
  p := TQwen3OutputParser.Create(fTokenizer, log.OnDelta);
  try
    // a stray continuation byte, then a lead byte the answer never completes
    Feed(p, Bin('ff') + 'a' + Bin('c3'));
    p.Push(IM_END);
    p.Finish(false, 0);
    CheckEqual(p.Response.Content, Bin('efbfbd') + 'a' + Bin('efbfbd'),
      'U+FFFD per invalid byte');
    CheckEqual(log.Content, p.Response.Content, 'streamed the same');
    CheckEqual(log.Invalid, 0, 'valid deltas');
  finally
    p.Free;
    log.Free;
  end;
  p := TQwen3OutputParser.Create(fTokenizer);
  try
    Feed(p, 'x');
    p.Push(IM_START);
    p.Push(THINK_END);
    p.Push(TOOL_CALL_END);
    Feed(p, 'y');
    p.Push(IM_END);
    p.Finish(false, 0);
    CheckEqual(p.Response.Content, 'x<|im_start|></think></tool_call>y',
      'control tokens inside the text are text');
    raised := false;
    try
      p.Push(65);
    except
      on ELlmClient do
        raised := true;
    end;
    Check(raised, 'Push after Finish');
  finally
    p.Free;
  end;
  p := TQwen3OutputParser.Create(fTokenizer);
  try
    raised := false;
    try
      p.Push(fTokenizer.VocabSize);
    except
      on ELlmClient do
        raised := true;
    end;
    Check(raised, 'id outside the tokenizer');
    raised := false;
    try
      p.Push(-1);
    except
      on ELlmClient do
        raised := true;
    end;
    Check(raised, 'negative id');
  finally
    p.Free;
  end;
end;

procedure TTestInferLlmNative.ToolCallJson;
var
  name, args: RawUtf8;
begin
  Check(ParseQwen3ToolCall(' {"name": "a", "arguments": {"x": [1, 2.0, "s"]}} ',
    name, args), 'plain call');
  CheckEqual(name, 'a', 'name');
  CheckEqual(args, '{"x": [1, 2.0, "s"]}', 'arguments');
  Check(ParseQwen3ToolCall('{"name":"b","arguments":{"n":{"deep":true}}}',
    name, args), 'compact call');
  CheckEqual(args, '{"n": {"deep": true}}', 'nested arguments re-formatted');
  Check(not ParseQwen3ToolCall('{"name": "a"}', name, args), 'no arguments');
  Check(name = '', 'name cleared on failure');
  Check(not ParseQwen3ToolCall('{"name": "a", "arguments": {}, "x": 1}',
    name, args), 'extra key');
  Check(not ParseQwen3ToolCall('{"Name": "a", "arguments": {}}', name, args),
    'name in another case');
  Check(not ParseQwen3ToolCall('{"name": "a", "Arguments": {}}', name, args),
    'arguments in another case');
  Check(not ParseQwen3ToolCall('{"name": "a", "arguments": {}, "": 1}', name, args),
    'an extra key with an empty name');
  Check(not ParseQwen3ToolCall('{"name": "a", "arguments": {}} x', name, args),
    'text after the object');
  // more digits than json.dumps keeps: that float stays as the model wrote it
  Check(ParseQwen3ToolCall('{"name": "t", "arguments": {"v":0.1234567890123456, ' +
    '"w":1.50} }', name, args), 'a float of 16 digits');
  CheckEqual(args, '{"v": 0.1234567890123456, "w": 1.5}',
    'the long float verbatim, the rest formatted, no blank after');
  Check(ParseQwen3ToolCall('{"name":"t","\u0061rguments":{"v":1e999}}', name, args),
    'an escaped key, a float beyond the range');
  CheckEqual(args, '{"v": 1e999}', 'beyond the range verbatim');
  // the long float switches no other rule off
  Check(not ParseQwen3ToolCall('{"name":"f","arguments":{"a":1,"a":2,' +
    '"v":0.1234567890123456}}', name, args), 'duplicate argument key');
  Check(not ParseQwen3ToolCall('{"name":"f","arguments":{"s":"\ud800",' +
    '"v":0.1234567890123456}}', name, args), 'lone surrogate');
  Check(not ParseQwen3ToolCall('{"name":"f","arguments":{"s":"a'#10'b",' +
    '"v":0.1234567890123456}}', name, args), 'raw newline in a string');
  Check(not ParseQwen3ToolCall('{"name":"f","arguments":{"s":"' + Bin('ff') + '"}}',
    name, args), 'invalid UTF-8');
  Check(not ParseQwen3ToolCall('{"name": "", "arguments": {}}', name, args),
    'empty name');
  Check(not ParseQwen3ToolCall('{"name": 5, "arguments": {}}', name, args),
    'name no string');
  Check(not ParseQwen3ToolCall('{"name": 0.00001, "arguments": {}}', name, args),
    'name a float the document would keep as text');
  Check(not ParseQwen3ToolCall('{"name": null, "arguments": {}}', name, args),
    'name null');
  Check(not ParseQwen3ToolCall('{"name": "a", "arguments": "{}"}', name, args),
    'arguments as a string');
  Check(not ParseQwen3ToolCall('{"name": "a", "arguments": [1]}', name, args),
    'arguments as an array');
  Check(not ParseQwen3ToolCall('{"name": "a", "name": "b"}', name, args),
    'duplicate key');
  Check(not ParseQwen3ToolCall('{"name": "a", "arguments": {}', name, args),
    'invalid JSON');
  Check(not ParseQwen3ToolCall('', name, args), 'empty');
  Check(not ParseQwen3ToolCall('[1]', name, args), 'array');
end;

procedure TTestInferLlmNative.ChatPrompt;
var
  req: TLlmChatRequest;
  msgs: TLlmMessageDynArray;
  ids, expected: TIntegerDynArray;
  images: TLlmImageDynArray;
  raised: boolean;

  function Refused: boolean;
  begin
    result := false;
    try
      NativeChatPrompt(fTemplate, req, true);
    except
      on ELlmClient do
        result := true;
    end;
  end;

begin
  if fTemplate = nil then
    exit;
  SetLength(msgs, 1);
  msgs[0] := LlmMessage(lrUser, 'hi');
  req := LlmChatRequest('', msgs);
  ids := NativeChatPrompt(fTemplate, req, true);
  expected := fTemplate.Encode(msgs, nil, true, true);
  CheckEqual(length(ids), length(expected), 'thinking by default');
  Check(CompareMem(pointer(ids), pointer(expected), length(ids) * SizeOf(integer)),
    'the template ids');
  CheckEqual(ids[high(ids)], 10, 'ends with the opened assistant turn');
  ids := NativeChatPrompt(fTemplate, req, false);
  CheckEqual(ids[high(ids) - 2], THINK_END, 'default false: empty think block');
  req.Extra := _JsonFast('{"chat_template_kwargs":{"enable_thinking":true}}');
  Check(NativeEnableThinking(req, false), 'request overrides the default');
  req.Extra := _JsonFast('{"chat_template_kwargs":{"enable_thinking":false}}');
  Check(not NativeEnableThinking(req, true), 'request switches thinking off');
  ids := NativeChatPrompt(fTemplate, req, true);
  expected := fTemplate.Encode(msgs, nil, true, false);
  CheckEqual(length(ids), length(expected), 'enable_thinking false');
  Check(CompareMem(pointer(ids), pointer(expected), length(ids) * SizeOf(integer)),
    'the template ids without thinking');
  req.Extra := _JsonFast('{"chat_template_kwargs":{"enable_thinking":"no"}}');
  Check(Refused, 'enable_thinking no boolean');
  req.Extra := _JsonFast('{"other":1}');
  Check(NativeEnableThinking(req, true), 'other Extra keys ignored');
  req.ResponseFormat := '{"type":"json_object"}';
  Check(Refused, 'ResponseFormat refused');
  req.ResponseFormat := '';
  SetLength(images, 1); // no [] literal: Delphi 7 has no dynamic array constants
  images[0] := LlmImageUrl('http://x/y.png');
  msgs[0] := LlmImageMessage(lrUser, 'look', images);
  req.Messages := msgs;
  Check(Refused, 'images refused');
  req.Messages := nil;
  Check(Refused, 'no messages');
  SetLength(msgs, 1);
  msgs[0] := LlmMessage(lrUser, 'hi');
  req.Messages := msgs;
  SetLength(req.Tools, 1);
  req.Tools[0].Name := 'f';
  req.Tools[0].ParametersJson := '{"type":';
  raised := Refused;
  Check(raised, 'invalid tool JSON as ELlmClient');
end;

procedure TTestInferLlmNative.DecodeOnTinyModel;
var
  dir: TFileName;
  cpu: TInferCpuBackend;
  model: TQwen3Model;
  p: TQwen3OutputParser;
  prompt: TIntegerDynArray;
  raised: boolean;

  function Decode(aPromptLen, aMaxNew: PtrInt): boolean;
  var
    i: PtrInt;
  begin
    SetLength(prompt, aPromptLen);
    for i := 0 to aPromptLen - 1 do
      prompt[i] := 33 + i mod 60; // printable bytes, all inside the tiny vocab
    FreeAndNil(p);
    p := TQwen3OutputParser.Create(fTokenizer);
    try
      NativeDecode(model, prompt, aMaxNew, p);
      result := true;
    except
      on ELlmClient do
        result := false;
    end;
  end;

begin
  if fTokenizer = nil then
    exit;
  dir := FindInferRefDir('tiny-qwen3');
  if CheckFailed(dir <> '', 'tiny-qwen3 found') then
    exit;
  cpu := TInferCpuBackend.Create;
  model := nil;
  p := nil;
  try
    model := TQwen3Model.Create(dir, cpu);
    CheckEqual(model.Config.MaxPositions, 512, 'tiny context');
    // case 0 of the reference: [77 29 40 57 65] continues with 3 3 3 3 3 3 49
    SetLength(prompt, 5);
    prompt[0] := 77;
    prompt[1] := 29;
    prompt[2] := 40;
    prompt[3] := 57;
    prompt[4] := 65;
    p := TQwen3OutputParser.Create(fTokenizer);
    NativeDecode(model, prompt, 7, p);
    CheckEqual(p.Response.Content, #3#3#3#3#3#3'1', 'greedy ids of the reference');
    Check(p.Response.FinishReason = lfrLength, 'cut by aMaxNew');
    CheckEqual(p.Response.Usage.CompletionTokens, 7, 'seven tokens');
    CheckEqual(p.Response.Usage.PromptTokens, 5, 'prompt tokens');
    // the context limit: P + n - 1 positions are fed
    Check(Decode(511, 100), 'one free position');
    CheckEqual(p.Count, 2, 'gives two tokens');
    Check(p.Response.FinishReason = lfrLength, 'cut by the context');
    Check(Decode(512, 100), 'a full context');
    CheckEqual(p.Count, 1, 'still gives one token');
    Check(not Decode(513, 1), 'a prompt beyond the context');
    Check(not Decode(5, 0), 'no new tokens');
    raised := false;
    try
      FreeAndNil(p);
      p := TQwen3OutputParser.Create(fTokenizer);
      NativeDecode(model, nil, 1, p);
    except
      on ELlmClient do
        raised := true;
    end;
    Check(raised, 'empty prompt');
  finally
    p.Free;
    model.Free;
    cpu.Free;
  end;
end;

function TTestInferLlmNative.RealModel: TFileName;
begin
  result := '';
  {$ifdef CPU32}
  AddConsole('SKIP native client: 2.4 GB of FP32 weights need a 64-bit process');
  exit;
  {$endif CPU32}
  result := FindQwen3Folder('model.safetensors');
  if result <> '' then
    exit;
  CheckUtf8(GetEnvironmentVariable('LAI_REQUIRE_MODEL') <> '1',
    'LAI_REQUIRE_MODEL=1 but no vendor/models/qwen3-0.6b/model.safetensors');
  AddConsole('SKIP native client: no vendor/models/qwen3-0.6b (tools/infer-ref, ' +
    'make_reference_qwen3.py)');
end;

/// the messages and tools of a reference prompt
procedure RefRequest(p: PDocVariantData; var aRequest: TLlmChatRequest);
var
  m, t: PDocVariantData;
  i: PtrInt;
begin
  m := p^.A['messages'];
  SetLength(aRequest.Messages, m^.Count);
  for i := 0 to m^.Count - 1 do
    aRequest.Messages[i] := LlmMessage(ToLlmRole(_Safe(m^.Values[i])^.U['role']),
      _Safe(m^.Values[i])^.U['content']);
  t := p^.A['tools'];
  SetLength(aRequest.Tools, t^.Count);
  for i := 0 to t^.Count - 1 do
  begin
    aRequest.Tools[i].Name := _Safe(t^.Values[i])^.U['name'];
    aRequest.Tools[i].Description := _Safe(t^.Values[i])^.U['description'];
    aRequest.Tools[i].ParametersJson := _Safe(t^.Values[i])^.U['parameters'];
  end;
  if p^.GetValueIndex('enable_thinking') >= 0 then
    aRequest.Extra := _ObjFast(['chat_template_kwargs',
      _ObjFast(['enable_thinking', p^.B['enable_thinking']])]);
end;

function FindPrompt(aRef: TInferRefSet; const aId: RawUtf8): PDocVariantData;
var
  prompts: PDocVariantData;
  i: PtrInt;
begin
  prompts := aRef.Manifest^.A['prompts'];
  for i := 0 to prompts^.Count - 1 do
  begin
    result := _Safe(prompts^.Values[i]);
    if result^.U['id'] = aId then
      exit;
  end;
  result := nil;
end;

procedure TTestInferLlmNative.Qwen3ChatClient;
var
  dir, refdir: TFileName;
  ref: TInferRefSet;
  client: TNativeLlmClient;
  llm: ILlmClient;
  p: PDocVariantData;
  req: TLlmChatRequest;
  r: TLlmChatResponse;
  log: TDeltaLog;
  expected: RawUtf8;
  timer: TPrecisionTimer;
begin
  dir := RealModel;
  if dir = '' then
    exit;
  refdir := FindInferRefDir('qwen3');
  if CheckFailed(refdir <> '', 'qwen3 reference found') then
    exit;
  ref := TInferRefSet.CreateFromFolder(refdir);
  try
    timer.Start;
    client := TNativeLlmClient.Create(dir);
    llm := client; // reference counted from here
    AddConsole(FormatString('native client loaded in %', [timer.Stop]));
    CheckEqual(llm.Config.DefaultModel, 'qwen3-0.6b', 'model name from the folder');
    // a short answer: the stop token ends it
    p := FindPrompt(ref, 'chat_en');
    if CheckFailed(p <> nil, 'chat_en') then
      exit;
    req := LlmChatRequest('', nil);
    RefRequest(p, req);
    r := llm.ChatComplete(req);
    CheckEqual(r.Content, 'Paris', 'chat_en content');
    Check(r.FinishReason = lfrStop, 'chat_en stop');
    CheckEqual(r.Usage.PromptTokens, p^.I['tokens'], 'chat_en prompt tokens');
    CheckEqual(r.Usage.CompletionTokens, 2, 'Paris + im_end');
    CheckEqual(r.Model, 'qwen3-0.6b', 'response model');
    // the tool call of the reference continuation
    p := FindPrompt(ref, 'chat_tool');
    if CheckFailed(p <> nil, 'chat_tool') then
      exit;
    req := LlmChatRequest('', nil);
    RefRequest(p, req);
    timer.Start;
    r := llm.ChatComplete(req);
    AddConsole(FormatString('chat_tool: % + % tokens in %',
      [r.Usage.PromptTokens, r.Usage.CompletionTokens, timer.Stop]));
    CheckEqual(r.Usage.PromptTokens, p^.I['tokens'], 'chat_tool prompt tokens');
    CheckEqual(r.Content, '', 'only a call');
    Check(r.FinishReason = lfrToolCalls, 'chat_tool finish');
    if not CheckFailed(length(r.ToolCalls) = 1, 'one call') then
    begin
      CheckEqual(r.ToolCalls[0].Name, 'get_weather', 'tool name');
      CheckEqual(r.ToolCalls[0].ArgumentsJson, '{"city": "Berlin"}', 'tool arguments');
    end;
    // thinking by default; the cap cuts the think block
    p := FindPrompt(ref, 'chat_think');
    if CheckFailed(p <> nil, 'chat_think') then
      exit;
    req := LlmChatRequest('', nil);
    RefRequest(p, req);
    req.MaxTokens := p^.A['greedy']^.Count;
    r := llm.ChatComplete(req);
    expected := p^.U['greedy_text'];
    Check(IdemPChar(pointer(expected), '<THINK>'#10), 'reference starts thinking');
    delete(expected, 1, 8);
    CheckEqual(_Safe(r.Raw)^.A['choices']^._[0]^.O['message']^.U['reasoning_content'],
      expected, 'reasoning of the reference');
    CheckEqual(r.Content, '', 'no answer yet');
    Check(r.FinishReason = lfrLength, 'chat_think cut');
    // streamed: the deltas give the reference text
    p := FindPrompt(ref, 'chat_de');
    if CheckFailed(p <> nil, 'chat_de') then
      exit;
    req := LlmChatRequest('', nil);
    RefRequest(p, req);
    req.MaxTokens := p^.A['greedy']^.Count;
    log := TDeltaLog.Create;
    try
      llm.ChatStream(req, log.OnDelta);
      CheckEqual(log.Content, p^.U['greedy_text'], 'chat_de streamed');
      CheckEqual(log.Invalid, 0, 'valid UTF-8 deltas');
      CheckEqual(log.FirstRole, 'assistant', 'role');
      CheckEqual(log.Finish, 'length', 'cut by MaxTokens');
      CheckEqual(log.Usage.CompletionTokens, req.MaxTokens, 'completion tokens');
      CheckEqual(log.Dones, 1, 'one Done');
      Check(not log.AfterDone, 'Done last');
    finally
      log.Free;
    end;
  finally
    ref.Free;
  end;
end;

function TTestInferLlmNative.Weather(const aArgumentsJson: RawUtf8): RawUtf8;
begin
  inc(fToolCalls);
  fToolArgs := aArgumentsJson;
  result := '{"temperature": 21, "unit": "celsius", "sky": "sunny"}';
end;

procedure TTestInferLlmNative.Qwen3AgentLoop;
var
  dir: TFileName;
  client: TNativeLlmClient;
  llm: ILlmClient;
  toolbox: TLlmCallbackToolbox;
  tools: ILlmToolbox;
  agent: TLlmAgent;
  tool: TLlmTool;
  msgs: TLlmMessageDynArray;
  r: TLlmChatResponse;
  timer: TPrecisionTimer;
begin
  dir := RealModel;
  if dir = '' then
    exit;
  client := TNativeLlmClient.Create(dir);
  llm := client;
  client.EnableThinking := false;
  client.MaxTokens := 40;
  toolbox := TLlmCallbackToolbox.Create;
  tools := toolbox;
  tool.Name := 'get_weather';
  tool.Description := 'Current weather for a city';
  tool.ParametersJson := '{"type":"object","properties":{"city":{"type":"string"}},' +
    '"required":["city"]}';
  toolbox.Add(tool, Weather);
  SetLength(msgs, 1);
  msgs[0] := LlmMessage(lrUser, 'What is the weather in Berlin?');
  fToolCalls := 0;
  agent := TLlmAgent.Create(llm, tools, '');
  try
    agent.MaxIterations := 3;
    timer.Start;
    r := agent.Run(msgs);
    AddConsole(FormatString('agent loop in %: %', [timer.Stop, r.Content]));
  finally
    agent.Free;
  end;
  CheckEqual(fToolCalls, 1, 'the tool ran once');
  CheckEqual(fToolArgs, '{"city": "Berlin"}', 'with the city');
  Check(r.ToolCalls = nil, 'the final answer calls no tool');
  Check(r.Content <> '', 'and says something');
  Check(PosEx('21', r.Content) > 0, 'with the tool result in it');
end;

end.
