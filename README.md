# mormot2-ai — the `mormot.ai.*` extension for mORMot2 (FPC · Delphi 7 … 13)

A **mORMot-native** AI extension for **Free Pascal** and **Delphi 7 to 13** — the same
compiler range as mORMot2. It covers the Model Context Protocol, LLM clients, embeddings,
vector search and RAG, and it brings **its own inference engine in plain Object Pascal**:
no Python, no llama.cpp, no foreign `.so`/`.dll` at run time. mORMot2's HTTP stack, JSON,
RTTI, SQLite and crypto are the foundation, not a dependency we tolerate.

> **Status, 2026-10-14:** MCP, LLM clients, agent loop, RAG and the native vector store
> are done and in use. The native inference engine runs BGE-M3 embeddings from the same
> GGUF file as `lembed`, with the same quality and **1.6 to 2 times its speed on x86_64** (Linux / Windows);
> it passes all suites, its quality gate and a 90-minute soak test on Linux x86_64 too,
> and it is the recommended provider of the consuming backend. The Qwen3 decoder passes
> its gate: on a fixed prompt set, greedy output in FP32 is token-identical to
> `transformers`. Next: the decoder behind `ILlmClient`, then quantisation and CUDA. Three test suites, about 3600 assertions, green wherever they have run (table below).

## What it does

| Area | Units | In short |
|---|---|---|
| **MCP server** | `mormot.ai.mcp[.server/.stdio/.tools]` | protocol revision **2026-07-28** (stateless); tools from typed records, input schema generated from RTTI; stdio, HTTP, Streamable HTTP, in-process; OAuth 2.1 resource server (opt-in) |
| **LLM clients** | `mormot.ai.llm[.openai/.anthropic/.sse/.structured]` | OpenAI wire as lingua franca (OpenAI, LiteLLM, Ollama-compatible), native Anthropic driver; streaming, tool calls, structured output, vision |
| **Agent** | `mormot.ai.agent[.mcp]` | tool-calling loop; MCP tools usable by the agent |
| **Embeddings** | `mormot.ai.embeddings`, `.embed.{native,provider,lembed}` | one `IEmbedder` seam: the native engine, any OpenAI-compatible API, or sqlite-lembed |
| **Vector search** | `mormot.ai.vectorstore[.native/.sqlitevec]` | native store in mORMot's SQLite: float32 rows as the truth, 1-bit RAM index with exact rescoring, SQL key filters — measured at 2 million × 1024 |
| **RAG** | `mormot.ai.rag[.tool]` | strict, atomic ingestion; retrieval; RAG as an MCP/agent tool |
| **Load governor** | `mormot.ai.governor` | paces bulk jobs: duty cycle, CPU temperature threshold (Linux hwmon), throughput heuristic; `TGovernedEmbedder` wraps any `IEmbedder` |
| **Native inference** | `src/infer/mormot.ai.infer.*` | GGUF and safetensors loaders, Q8_0, SentencePiece/Unigram tokenizer, XLM-RoBERTa encoder, CPU backend with NEON and AVX2 kernels and its own thread pool |

## Where it runs

Number of assertions per suite, every one passing; each row with numbers ran on 2026-10-14 or 15:

| Platform | Compiler | MCP | LLM | Inference |
|---|---|---|---|---|
| Linux aarch64 (WSL2, Snapdragon X) | FPC 3.2.4 and 3.2.2 | 1257 | 721 | 1597 |
| Windows x64 (i9-14900K) | FPC 3.2.2 | 1257 | 717 | 821 ¹ ⁵ |
| Windows x64 | Delphi 13 | 1257 | 720 | 1587 |
| Windows x86 | **Delphi 7** | 1253 | 708 | 891 ⁶ |
| macOS aarch64 (M2) | FPC 3.2.4 | 1257 | 647 ⁴ | not yet run |
| Windows ARM64 | FPC 3.3.1 | 1257 | green ² ⁴ | not yet run |
| Linux x86_64 (WSL2, i9-14900K) | FPC 3.2.2 | 1257 | 719 | 828 ³ ⁵ |

¹ Without the converted FP32 reference model on that machine; those tests skip.
² Needs an ARM64 `sqlite3-64.dll` next to the binary — mORMot2 ships no static SQLite for
that target.
³ The production target (Docker uses the same FPC 3.2.2), with the BGE-M3 corpus gate
(`LAI_INFER_GATE=1`) included: nDCG@10 0.8994 on the AVX2 path.
⁴ Run before the load governor tests were added (they bring about 60 assertions).
⁵ Run before the Qwen3 tests were added (tokenizer, chat template, decoder: about 380
assertions with the model, about 170 without it).
⁶ The Qwen3-0.6B tests skip in a 32-bit process (2.4 GB of FP32 weights); the tiny
model, the tokenizer and the chat template run.

Counts differ by compiler because some tests only exist where a feature does (the `vec0`
extension has no 32-bit build, so Delphi 7 skips that test).

## Native inference

The engine replaces `lembed` (llama.cpp inside SQLite) and, later, every other foreign
library. It is written from published specs and MIT-licensed templates (llama2.c, ggml,
calm); Apache-2.0 sources (transformers, tokenizers) are only read, never copied. It
targets mORMot2's compiler range — Delphi 7 as a correctness tier without a speed goal;
the tested configurations are listed under [Where it runs](#where-it-runs).

```pascal
program embed;

{$I mormot.defines.inc}

uses
  {$I mormot.uses.inc}
  mormot.ai.llm.types, mormot.ai.embeddings, mormot.ai.embed.native;

var
  emb: IEmbedder;
  v: TLlmEmbedding;
begin
  // one shared embedder per logical model name: the first call decides the
  // model path and the thread count
  emb := SharedNativeEmbedder('vendor/models/bge-m3-q8_0.gguf', 'bge-m3');
  v := emb.Embed('Abdichtung eines Flachdachs');
  // v: 1024 floats, L2-normalised - ready for TNativeVectorStore
end.
```

**How it is checked.** Every layer is compared against a fixed Python reference
(`tools/infer-ref/`, data under `tests/data/infer-ref/`); token ids match lembed on all
1318 cases without special-token literals (those are deliberately treated as text and
checked against SentencePiece instead); on a German retrieval corpus the embeddings reach
nDCG@10 **0.9004** on the NEON path and **0.8994** on the AVX2 path (lembed 0.8963,
gate 0.8863), with a cosine of at least 0.9991 against lembed's vectors. The backend has two arithmetic contracts: `Exact = true` is bit-identical to the
scalar double reference on all cores; the default computes in float lanes like ggml and
is tested against the reference with stated tolerances.

### Measured throughput

BGE-M3 (`bge-m3-q8_0.gguf`, the file lembed uses) on the embed-de corpus: 320 texts,
16268 tokens, mean 51 tokens per text, one text at a time. `demos/infer/embed-bench`
runs both engines over the same loop (`embed-bench [gguf] all [threads|lembed]`); runs
alternate between the engines so that heat and background load hit both alike.

**x86_64, AVX2** — Intel Core i9-14900K (8 P-cores with HT + 16 E-cores, 32 threads),
Windows 11, FPC 3.2.2 win64, 2026-10-14:

| | lembed | native | native / lembed |
|---|---|---|---|
| default threads, run 1 / 2 | 524 / 525 tokens/s | 1039 / 1027 tokens/s (22 threads) | 1.98x / 1.96x |
| native with 8 / 16 / 24 / 32 threads | | 739 / 954 / 1040 / 959 tokens/s | |
| peak working set | 514 MB | 377 MB | 0.73x |

Started over SSH, both engines ran at about half that speed on this machine (lembed 187,
native 417 tokens/s): Windows power throttling slowed the windowless process down.
Switching off execution-speed throttling for the process
(`SetProcessInformation(ProcessPowerThrottling)`) restored the numbers above;
`embed-bench` does so unless `LAI_THROTTLE` is set. A Windows service doing bulk
embedding may want the same.

**x86_64 Linux, AVX2** — the same i9-14900K, Ubuntu 24.04 under WSL2, FPC 3.2.2,
2026-10-14:

| | lembed | native | native / lembed |
|---|---|---|---|
| default threads, run 1 / 2 | 544 / 545 tokens/s | 861 / 860 tokens/s (22 threads) | 1.58x / 1.58x |
| native with 8 / 16 / 24 / 32 threads | | 640 / 755 / 877 / 709 tokens/s | |
| peak RSS | 755 MB | 439 MB | 0.58x |

**aarch64, NEON dot product** — Snapdragon X (12 cores), WSL2, FPC 3.2.4, model on the
Linux file system, 2026-10-14:

| | lembed | native | native / lembed |
|---|---|---|---|
| 4 cores (taskset, 4 threads each), run 1 / 2 | 359 / 297 tokens/s | 361 / 341 tokens/s | 1.00x / 1.15x |
| whole machine, default threads, run 1 / 2 | 336 / 316 tokens/s | 385 / 315 tokens/s (8 threads) | 1.15x / 0.99x |
| peak RSS | 758 MB | 617 MB | 0.81x |

Peak RSS includes the pages of the mapped model file and moves with the page cache
(480 to 617 MB for native over several runs). The laptop throttles under sustained load;
the spread between runs is that, not noise in the engines. The default thread count is
two thirds of the logical cores.

Plan, phases, gates and the full record: [docs/native-inference.md](docs/native-inference.md)
(German).

## A server in thirty lines

A tool is a class over a typed record. The input schema is generated from the record's
RTTI — there is no hand-written JSON Schema, so it cannot drift from the code:

```pascal
uses
  mormot.core.base, mormot.core.rtti, mormot.core.variants,
  mormot.ai.mcp, mormot.ai.mcp.stdio;

type
  TAddParams = record
    a: integer;
    b: integer;
  end;

  TAddTool = class(TMcpToolBase<TAddParams>)
  protected
    function ExecuteTyped(const aParams: TAddParams;
      const aAuthCtx: TMcpAuthContext): variant; override;
  end;

function TAddTool.ExecuteTyped(const aParams: TAddParams;
  const aAuthCtx: TMcpAuthContext): variant;
begin
  result := _ObjFast(['sum', aParams.a + aParams.b]);
end;

var
  server: TMcpServer;
  transport: TMcpStdioTransport;
  tool: IMcpTool;
begin
  // FPC 3.2 and Delphi 7 have no RTTI for plain records — declare the fields once.
  if not RecordHasFields(TypeInfo(TAddParams)) then
    Rtti.RegisterFromText(TypeInfo(TAddParams), 'a,b:integer');

  server := TMcpServer.Create('DemoServer', '1.0');
  try
    tool := TAddTool.Create('add', 'Add two numbers');
    server.RegisterTool(tool);
    server.Start;

    transport := TMcpStdioTransport.Create(server);
    try
      transport.Start;
      while transport.IsActive do
        Sleep(100);
    finally
      transport.Free;
    end;
  finally
    server.Free;
  end;
end.
```

Without generics (Delphi 7, or by choice) a tool derives from `TMcpRecordTool`, names its
record once and casts it back — `TMcpToolBase<T>` does exactly this for you. On Delphi 7 a
record only has `TypeInfo` if it holds a managed field (a string, say), so this one does:

```pascal
type
  TGreetParams = record
    name: RawUtf8;
    times: integer;
  end;

  TGreetTool = class(TMcpRecordTool)
  protected
    function ExecuteRecord(const aParams;
      const aAuthCtx: TMcpAuthContext): variant; override;
  public
    constructor Create(const aName, aDescription: RawUtf8);
  end;

constructor TGreetTool.Create(const aName, aDescription: RawUtf8);
begin
  inherited CreateTyped(TypeInfo(TGreetParams), aName, aDescription);
end;

function TGreetTool.ExecuteRecord(const aParams;
  const aAuthCtx: TMcpAuthContext): variant;
var
  p: ^TGreetParams;
begin
  p := @aParams; // the parsed record, of the type given to CreateTyped
  result := _ObjFast(['greeting', 'Hello ' + p^.name, 'times', p^.times]);
end;

// once, before the first tool is created - FPC 3.2 and Delphi 7 have no field RTTI:
//   if not RecordHasFields(TypeInfo(TGreetParams)) then
//     Rtti.RegisterFromText(TypeInfo(TGreetParams), 'name:RawUtf8 times:integer');
```

That `Rtti.RegisterFromText` line is the one thing that catches everyone: under FPC a plain
record carries no field RTTI, so the schema generator has nothing to read and the tool
shows up with an empty input schema. Register the record once at startup and it works.

Swap `TMcpStdioTransport` for the HTTP or Streamable transport and nothing else changes —
the server and its tools do not know which transport they are behind. Full examples in
[`demos/`](demos/).

## MCP — revision 2026-07-28, and only that one

The 2026-07-28 revision is the largest change since MCP launched, and it is **stateless**:
`initialize`, protocol sessions (`Mcp-Session-Id`), request batching, SSE resumability and
the legacy HTTP+SSE transport are **gone**. Every request carries its own protocol version
and client capabilities in `params._meta`; `server/discover` replaces the handshake.

We deleted the removed machinery rather than keeping it behind a negotiation layer. There
is **no multi-version support** — a greenfield project has nothing to be compatible with,
and a second code path is a second thing to get wrong.

**One temporary exception, marked for removal: the stdio legacy bridge**
(`mormot.ai.mcp.legacy`, code marker `LEGACY-ERA`). Clients still open stdio servers with
the old `initialize` handshake — Claude Code does so by default, and its `server/discover`
probe gives up after 3 s, which a slowly starting process misses. On stdio only, the bridge
answers `initialize` (2025-11-25 lifecycle), `notifications/initialized` and `ping`, then
passes requests without `_meta` to the unchanged 2026-07-28 core with the metadata filled
in. The spec allows a server to serve both eras. Requests that carry `_meta` stay purely
modern; `TMcpStdioTransport.AcceptLegacyInitialize := false` switches the bridge off. It
goes away once clients probe by default and survive a slow start — see
[CONCEPT.md §6](CONCEPT.md).

What that buys, beyond simplicity: session expiry, the use-after-free surface around the
streamable transport and the legacy SSE session map all stopped being problems, because
the state they guarded no longer exists.

Implemented, each built against the spec text: `server/discover`, `_meta` transport of the
protocol version, tools/resources/prompts, resource templates and completion, pagination
with a deterministic `tools/list` order, subscriptions, MRTR, cacheable results
(`ttlMs`/`cacheScope`), progress notifications with server-enforced monotonicity,
extensions, OTel `_meta` keys, `x-mcp-header`, JSON Schema 2020-12 in `inputSchema`.

**Deliberately not implemented:** `notifications/message` (logging) is deprecated in
2026-07-28 — "new implementations SHOULD NOT adopt it".

Transports: stdio, HTTP, Streamable HTTP, in-process. Tools register via
`TMcpServer.RegisterTool(IMcpTool)`; the input schema is generated **from RTTI** out of a
typed record (`TMcpToolBase<T: record>`), so there is no hand-written JSON Schema to drift.

**Authorization is opt-in, and closed once it is on.** The core is an OAuth 2.1 resource
server (RFC 9728 metadata, token validation before dispatch, audience binding, 401/403
challenges); the seam is `IMcpTokenVerifier`. Authorization is `OPTIONAL` in the
specification and the core follows that: **with no verifier set the server is open**, so any
HTTP deployment has to set one (stdio SHOULD NOT use it — credentials come from the
environment there). What the core does guarantee is that authorization can never be half on:
`Start` refuses a verifier without an `AuthResource` or without `AuthorizationServers`, and a
request a configured verifier does not accept is refused before dispatch, with a zeroed
context (`IsAuthenticated=false`, no scopes) so a tool that gates on identity fails closed.
Found a path around a configured verifier? [SECURITY.md](SECURITY.md).

### The spec is vendored

A verbatim copy of the specification lives under
[`docs/specs/mcp-2026-07-28/`](docs/specs/mcp-2026-07-28/) — 31 files including the full
schema reference. It is there so reviews and audits can **cite the normative text instead
of recalling it**. In protocol work the most common defect is not in the code, it is the
reviewer inventing a requirement, or remembering an older revision — and both sound
plausible. See the [README](docs/specs/mcp-2026-07-28/README.md) there for provenance,
license and how to update.

## LLM client and provider drivers

A clean-room client over provider-neutral records (`mormot.ai.llm.types`). The **OpenAI
Chat Completions wire is the lingua franca**: one `TLlmClient` covers OpenAI, LiteLLM and
Ollama-compatible endpoints. A provider whose wire genuinely differs gets a **native
driver** behind the same `ILlmClient` seam:

- **Anthropic** (`mormot.ai.llm.anthropic`, `TAnthropicClient`): the native Messages API —
  top-level `system`, mandatory `max_tokens`, `input_schema`/`tool_use`/`tool_result`,
  `x-api-key` + `anthropic-version`, event-typed SSE. The agent loop, RAG and structured
  output stay provider-agnostic; only the `ILlmClient` instance changes.

**Vision:** `TLlmMessage.Images` carries attachments (`LlmImageMessage`/`LlmImageBase64`/
`LlmImageUrl`); both wires serialise them (OpenAI `image_url` including base64 `data:` URIs,
Anthropic an `image` block with a typed `source`; an empty `MediaType` falls back to
`image/png`).

**Structured output** and **embeddings** run through the same seam; RAG combines
`IEmbedder` with `IVectorStore` (native store or sqlite-vec today, pgvector only when a
deployment needs it).

### Tests

The Anthropic driver test (`tests/test.llm.anthropic.pas`) is **hermetic** — no network, no
API key. It checks request construction (system hoisting, `input_schema`, tool round-trip),
response parsing (text/tool/usage, stop-reason mapping) and SSE decoding against canned
payloads, and runs as part of the suite.

**Native libraries are handled defensively.** `sqlite-vec` and `lembed` are external
`.so`/`.dll` files; the suite must catch the dynamic load and **skip cleanly** when they are
missing, never crash (see the env-gated `VectorStoreKeyedOps` test). That constraint goes
away once the **native engine** ([docs/native-inference.md](docs/native-inference.md))
and the native vector store replace the foreign libraries.

### Running the demos live

Provider configuration comes from the environment (`demos/.env`, copied from
[`demos/.env.sample`](demos/.env.sample)) — **API keys only from the environment**, never
committed, never logged:

```bash
bash scripts/build-demo.sh llm/llm-anthropic.lpr
set -a; . demos/.env; set +a      # ANTHROPIC_API_KEY[/_MODEL]
bin/fpc/demos/llm-anthropic
```

`llm-anthropic` is a line-for-line rebuild of the OpenAI tool-loop demo (`llm-agent`): the
same `TLlmAgent` and toolbox, only `TAnthropicClient` instead of `TLlmClient` — which is
what makes the lingua-franca seam a claim you can check rather than one you have to believe.

## Build & test

```bash
# from this repo, on Linux/WSL:
bash scripts/run-fpc-tests.sh        # MCP suite
bash scripts/run-fpc-llm-tests.sh    # LLM suite
bash scripts/run-fpc-infer-tests.sh  # native inference suite (reference data, no model needed)
LAI_INFER_GATE=1 bash scripts/run-fpc-infer-tests.sh  # + BGE-M3 Q8 corpus gate (nDCG@10, ~1 h scalar)
bash scripts/build-demo.sh           # all demos (or pass one, e.g. llm/llm-agent.lpr)
# VERBOSE=1 for the full compiler/test log; logs land in bin/fpc/
```

```powershell
# Delphi 13, on Windows:
.\scripts\run-delphi-tests.ps1               # MCP, LLM and inference suites (-Suite mcp|llm|infer, -ShowAll)
# logs and binaries land in bin\delphi\
# Delphi 7 (Win32):
.\scripts\run-delphi7-tests.ps1 -FastMM4 <dir>   # same suites; bin\delphi7\
```

**Requirements:** a mORMot2 source tree, plus
- **FPC 3.2.4** (currently the release candidate) for the bash scripts. They pin it
  (`scripts/fpc-env.sh`: a `fpc324` wrapper if one is on the PATH, else `fpc`) and stop on
  any other version, because a machine with several FPCs otherwise builds with whichever one
  the shell finds first. Override deliberately with `FPC=<compiler>`; `FPC_EXPECT=<version>`
  still insists on a version. Unit caches live under `lib/<fpc version>`, so builds with
  different compilers never share `.ppu` files.
- **Delphi 13** (`dcc64`, compiler 37.0) for `run-delphi-tests.ps1`; override with
  `-Dcc <path>`. On ARM64 Windows the x64 test programs run under emulation.

- **Delphi 7** (`dcc32`) for `run-delphi7-tests.ps1`: `-Delphi7 <install dir>` (or
  `DELPHI7`) and `-FastMM4 <dir with FastMM4.pas>` (or `FASTMM4`) — mORMot2 uses FastMM4
  as memory manager on Delphi 7, and it is a separate library. The `vec0` test is skipped
  there (no 32-bit build of the extension); the script says so.

Which compiler ran which suite on which platform: [Where it runs](#where-it-runs). Lazarus
is optional; `.lpi` files exist for the test projects, the scripts do not need the IDE.

The two native extensions (`sqlite-vec`, `lembed`) are **optional**: the tests that need
them skip without them; everything else, the native vector store and engine included,
runs.

**Locating mORMot2:** the scripts expect a mORMot2 source tree (`src/`, `static/`). With
no configuration they assume the submodule case — `<consumer>/shared/delphi/libs/_git_Synopse2`,
three levels above this repo. For a **standalone clone**, point at it explicitly:

```bash
MORMOT2_ROOT=/path/to/mORMot2 bash scripts/run-fpc-tests.sh
```

## Using it in your own project

There is no package file to install — add this repository's `src` (and `src/infer` for
native inference) plus mORMot2's source directories, and use the units. This is what the build scripts do; the flags below are lifted from
[`scripts/run-fpc-tests.sh`](scripts/run-fpc-tests.sh):

```bash
MORMOT2=/path/to/mORMot2
LAI=/path/to/mormot2-ai

fpc -MDelphi -Sci -Ci -O2 \
  -Fi"$MORMOT2/src;$MORMOT2/src/core;$MORMOT2/src/net" \
  -Fu"$LAI/src;$LAI/src/infer;$MORMOT2/src/app;$MORMOT2/src/core;$MORMOT2/src/crypt;$MORMOT2/src/db;$MORMOT2/src/lib;$MORMOT2/src/net;$MORMOT2/src/orm;$MORMOT2/src/rest;$MORMOT2/src/soa" \
  -Fl"$MORMOT2/static/$(fpc -iTP)-linux" \
  your-program.lpr
```

`-MDelphi` is not optional — the sources use Delphi-mode syntax, as mORMot2 itself does.
Start the program as mORMot2 programs do, with `{$I mormot.defines.inc}` and
`uses {$I mormot.uses.inc} ...`: on POSIX that brings in `cthreads`, which the MCP
transports and the native engine's thread pool need.
`-Fl` points at mORMot2's static libraries (the bundled SQLite among them); drop it only if
you use no unit that needs them.

In Lazarus, add `$(LAI)/src` and `$(LAI)/src/infer` to the project's *Other unit files* and mORMot2 per its own
instructions. Consuming this repository as a **git submodule** is the arrangement the
default paths assume — see "Locating mORMot2" under Build & test.

## Layout

```
mormot2-ai/
  src/
    mormot.ai.mcp[.server|.stdio|.tools|.legacy]  MCP: core, transports, sample tools
    mormot.ai.llm[.types|.openai|.anthropic]      LLM clients (+ .sse, .structured)
    mormot.ai.agent[.mcp]                         agent / tool-calling loop
    mormot.ai.embeddings, .embed.{native,provider,lembed}
    mormot.ai.vectorstore[.native|.sqlitevec]
    mormot.ai.sqlite                              SQLite helpers, extension loader
    mormot.ai.rag[.tool]
    mormot.ai.governor                            load governor for bulk jobs
    infer/mormot.ai.infer.*                       the native inference engine:
      tensor, memory, loader.{gguf,safetensors}, tokenizer.ugm, model.xlmr,
      backend[.cpu], kernels (NEON, AVX2), threads, pipeline.embed
  tests/    mcp.tests · llm.tests · infer.tests (mORMot TSynTests), data/ (reference data)
  demos/    stdio · http · jsonrpc · streamable · llm/* · rag/* · infer/embed-bench
  tools/    infer-ref (Python reference for the engine, not needed to build)
  docs/     native-inference.md · native-embeddings.md · specs/mcp-2026-07-28/
  vendor/   mormot.ext.os (adopted) · models, sqlite-ext (gitignored, not in git)
  scripts/  run-fpc-{tests,llm-tests,infer-tests}.sh · build-demo.sh · fpc-env.sh
            run-delphi-tests.ps1 · run-delphi7-tests.ps1 · x64-ext-verify.sh
  CONCEPT.md  DESIGN.md  LICENSE  NOTICE  SECURITY.md  UPSTREAM_BASE
```

## Maturity

Not everything here carries the same weight. "Live" means it was run against a real
provider or runtime, not only against tests.

| Area | State | Notes |
|---|---|---|
| MCP core, JSON-RPC, tool/resource registry | **stable** | hardened over five review rounds; a real consumer registers its tools and supplies the token verifier |
| Transports: stdio, HTTP, Streamable HTTP, in-process | **stable** | concurrency covered (parallel clients, keep-alive reuse) |
| MCP authorization (resource server) | **stable** | opt-in per spec (`OPTIONAL`), closed once a verifier is set; the verifier itself is supplied by the consumer |
| LLM client, OpenAI wire | **stable, live** | also covers LiteLLM and Ollama-compatible endpoints |
| Anthropic driver | **stable, live** | streaming, tool loop and structured output verified against a real model |
| Agent / tool-calling loop, structured output, vision | **stable, live** | |
| Embeddings via provider API | **stable** | |
| Native vector store (`mormot.ai.vectorstore.native`) | **stable** | no C extension; persisted model contract, SQL key filters, 1-bit RAM index with exact rescoring — measured at 2M × 1024 ([docs/native-embeddings.md](docs/native-embeddings.md)) |
| RAG (ingestion, retrieval, RAG-as-tool) | **works** | ingestion is strict and atomic; the rollback test still needs the runtime |
| Native embeddings (`mormot.ai.embed.native`) | **stable** | BGE-M3 from GGUF Q8_0; passes its quality gates on aarch64 and x86_64; AVX2 and NEON kernels; 90-minute soak test passed (79,628 texts, every vector bit-identical to the first pass); the recommended provider of the consuming backend |
| Qwen3 decoder (`mormot.ai.infer.model.qwen3`, `.tokenizer.bpe`, `.chat`) | **works, not wired yet** | tokenizer: 2416 cases identical to Hugging Face, fuzzed against its pre-tokenizer; chat template: 21 conversations identical; Qwen3-0.6B FP32: greedy output token-identical for 7 prompts; correctness first, slow (scalar FP32); not yet behind `ILlmClient` |
| Load governor (`mormot.ai.governor`) | **works** | tested on a fake clock and a fake sysfs tree; the temperature path needs a Linux CPU sensor, which WSL2 does not expose - not yet seen on real sensor hardware |
| Embeddings via `lembed` (local GGUF) | **works, needs a native library** | shared per model name; calls are serialised (its context is not re-entrant); kept as fallback and as the oracle of the native engine |
| Vector store `sqlite-vec` | **works, needs a native library** | superseded by the native store; real-extension test is env-gated (`SQLITE_EXT_DIR`) |
| pgvector backend, Ollama embedder | **not built** | the `IVectorStore` / `IEmbedder` seams exist; built when a deployment needs them |

The "needs a native library" rows are **optional dependencies**: an application that
selects one of these adapters has to provide its library (or handle the error its
constructor raises without it); the tests that need one are skipped when it is missing.
The libraries are **not in git** — `vendor/models` and `vendor/sqlite-ext` are ignored
and have to be provided locally.

## Roadmap

Done:
- ✓ **MCP** — server adopted and renamed into `mormot.ai.*`, moved to revision 2026-07-28
  (stateless) with the removed machinery deleted, used by a real consumer.
- ✓ **LLM** — clean-room client: OpenAI wire and Anthropic, agent/tool loop, structured
  output, vision, embeddings, RAG.
- ✓ **Compiler range of mORMot2** — FPC and Delphi 7 to 13, non-generic core.
- ✓ **Native vector store** — replaces sqlite-vec.
- ✓ **Native inference, phases 0 to 1b and the speed work of 1c** — reference data,
  BGE-M3 in FP32 and from the Q8_0 GGUF, NEON and AVX2 kernels, measured against lembed.
- ✓ **Qwen3 decoder up to its gate** — byte-level BPE tokenizer and chat template
  identical to Hugging Face on every reference case; Qwen3-0.6B in FP32 with KV cache,
  greedy output token-identical to `transformers` for all seven reference prompts.

Next, in this order:
1. **Decoder LLMs behind `ILlmClient`** — the Qwen3 engine is in place and passes its
   gate (greedy output token-identical to `transformers` in FP32); next is
   `mormot.ai.llm.native` with streaming, stop tokens and tool calls.
2. **Quantisation and speed** — Q4_0, later K-quants, sampling (temperature, top-k,
   top-p), benchmarks per platform.
3. **GPU** — CUDA through the driver API, kernels as embedded PTX; optionally Vulkan.
4. **After that, on demand** — more architectures (Llama 3, Mistral, Gemma), a reranker,
   Qwen3-Embedding, BGE-M3 sparse/ColBERT output.

Alongside: the first real consumer of the load governor (a bulk indexing job in the
backend), agent loop limits (cancellation, deadlines, context budget),
token-based chunking, a Delphi build script for the demos. The German [CONCEPT.md](CONCEPT.md) §9 is
the single source of truth for open points.

## Why a standalone extension

- **Not an upstream mORMot PR** (decided 2026-07-29). mORMot2 stays the **foundation**,
  not the delivery target: an own repository keeps our own pace and platform goals
  (native inference engine, CUDA) and lets MCP follow the 2026-07-28 spec alone.
  Rationale: [CONCEPT.md §1](CONCEPT.md).
- **FPC and Delphi 7 to 13** — the same compiler range as mORMot2. FPC is the reference
  compiler and the gate; Delphi 13 (Win64) and Delphi 7 (Win32), the two ends of the
  range, build the library and run all test suites.
  Parts that only exist under FPC — the POSIX process launcher in `vendor/` — stay behind
  `{$ifdef FPC}` instead of being duplicated. The library needs no generics: the MCP
  registries are `TSynDictionary`, typed tools build on `TMcpRecordTool`, and the generic
  `TMcpToolBase<T>` is only a shorthand where the compiler has generics.
- **Written like mORMot2** — unit headers, `{ ****** section }` layout, `RawUtf8`,
  `TDocVariantData`, `ESynException`, `f` field prefix, `///` docs. Stricter than mORMot2
  where Delphi 7 or readability needs it: a non-generic core (generics only as an optional
  typed shell under `HASGENERICS`), no anonymous methods, no inline variables, no
  `exit(value)`, no `with`, and an `a` prefix on our own parameters. The full rule set
  is in [CONCEPT.md §1](CONCEPT.md).
- **Spec-driven, not diff-driven** — we follow the published specs (MCP, JSON-RPC,
  provider wires) rather than chasing another library's changes.
- **License MPL/GPL/LGPL** (the mORMot three-license, confirmed 2026-10-14 — also for the
  native inference engine) — the adopted MCP units require it, and it composes with
  MIT-licensed ported code and LGPL-with-linking-exception.

## Origin and license

The MCP server units were **adopted** from
[flydev-fr/mormot2-extensions](https://github.com/flydev-fr/mormot2-extensions)
(mORMot-licensed) and renamed into the `mormot.ai.*` namespace; the original per-unit
headers are retained. Details: [NOTICE](NOTICE), [LICENSE](LICENSE), [DESIGN.md](DESIGN.md).

The vendored MCP specification under `docs/specs/` is **not** part of this work and not
covered by the three-license above — it keeps its own upstream license alongside it.

### Relationship to flydev — code adoption, not a fork

Our adoption from flydev so far comes from `mormot2-extensions`: the MCP units, their
tests and demos, and `mormot.ext.os`. From
[flydev-fr/mormot.ai](https://github.com/flydev-fr/mormot.ai) we may adopt **code only**,
selectively (pgvector, if a deployment needs it): no fork, no shared repository, no
upstream PRs. Its evaluation baseline is **pinned** in [UPSTREAM_BASE](UPSTREAM_BASE) so
that a later adoption can work from a diff; nothing from it is in this repository yet.
Both projects use the `mormot.ai.*` namespace — fine while they are used separately; a
program combining both has to mind the overlapping unit names.

> **Rule: whoever ports from upstream bumps `UPSTREAM_BASE` in the same commit.** Without
> it the diff base is lost and the next round has to guess. The same file records the
> templates of the native inference engine, the model revision and file hashes behind its
> reference data, the Python environment that produced them, and the mORMot2 commit the
> suites last ran against.

## A note on the documentation language

This README is English. The two design documents behind it — [CONCEPT.md](CONCEPT.md)
(decisions, open points, licensing strategy) and [DESIGN.md](DESIGN.md) (architecture,
protocol details, review history) — are **written in German**, as are most in-code
comments.

That is a working-language artefact, not a statement about who the code is for. Both files
are dense and worth translating; until that happens, machine translation handles them well
because they are structured prose with code identifiers left in English. If you hit a
passage that matters and does not survive translation, open an issue and quote it — that is
a faster path to a fix than a full rewrite nobody has scheduled.

New documentation in this repository should be written in English.
