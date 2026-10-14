/// native inference engine - byte-level BPE tokenizer (Qwen2 / Qwen3)
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - clean-room implementation of the pipeline a Hugging Face tokenizer.json
//   describes; the reference is the tokenizers library (Apache-2.0, read only
//   for the documented behaviour) and the gate tests/data/infer-ref/qwen3-tokens:
//   NFC normalizer, the Qwen2 Split regex (isolated), ByteLevel without prefix
//   space, BPE merges by rank (leftmost first on equal rank)
// - the split regex is not run by a regex engine: TBpeTokenizer.Split is the
//   leftmost-first match of its seven alternatives, written out by hand
//     (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N} |
//      ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
//   with the code point classes of mormot.ai.infer.unicode (measured against
//   the regex of the tokenizers library, also the case folding of (?i))
// - behaviour = parse_special=false: Encode treats every text as text, also
//   "<|im_end|>" or "<think>"; control tokens come only from TokenId (the chat
//   template), never from user text
// - invalid UTF-8 bytes become U+FFFD before normalization
// - thread-safe after loading: Encode uses only local state
unit mormot.ai.infer.tokenizer.bpe;

{
  *****************************************************************************

    - TBpeTokenizer: load a byte-level BPE tokenizer.json (Qwen2 pipeline),
      Normalize, Split, Encode text to ids and Decode ids to bytes

  *****************************************************************************
}

interface

{$I mormot.defines.inc}

uses
  sysutils,
  mormot.core.base,
  mormot.core.os,
  mormot.core.unicode,
  mormot.core.text,
  mormot.core.variants,
  mormot.ai.infer.tensor,
  mormot.ai.infer.unicode;

type
  /// one symbol of a word being merged
  TBpeSymbol = record
    Id: integer;
    Prev, Next: integer; // -1 at the ends
    Len: integer;        // 0 = merged into its left neighbour
  end;
  TBpeSymbolDynArray = array of TBpeSymbol;

  /// a min-heap of candidate merges: rank first, then position
  TBpeQueue = object
    Key: TInt64DynArray; // rank shl 32 + position
    New: TIntegerDynArray;
    Count: PtrInt;
    procedure Push(aRank, aPos, aNew: integer);
    function Pop(out aPos, aNew: integer): boolean;
  end;

  /// a byte-level BPE tokenizer with the Qwen2 pre-tokenizer
  TBpeTokenizer = class
  protected
    fTokens: TRawByteStringDynArray; // id -> bytes ('' = no token)
    fVocabCount: integer;            // ids below are BPE tokens, above added ones
    fByteId: array[byte] of integer;
    fHashHead, fHashNext: TIntegerDynArray;
    fHashMask: PtrInt;
    fMergeKey: TInt64DynArray;       // open addressing, -1 = empty slot
    fMergeRank, fMergeNew: TIntegerDynArray;
    fMergeMask: PtrInt;
    fMergeCount: integer;
    fAddedText: TRawUtf8DynArray;
    fAddedId: TIntegerDynArray;
    procedure AddMerge(aLeft, aRight, aRank, aNew: integer);
    function FindMerge(aLeft, aRight: integer; out aRank, aNew: integer): boolean;
    function BytesId(p: pointer; aLen: PtrInt): integer;
    procedure Bpe(p: PByteArray; aLen: PtrInt; var aIds: TIntegerDynArray;
      var aCount: PtrInt; var aSym: TBpeSymbolDynArray; var aQueue: TBpeQueue);
  public
    /// load and validate a Hugging Face tokenizer.json of this pipeline:
    // NFC, Split(Qwen2 regex, Isolated) + ByteLevel(no prefix space, no regex),
    // BPE (no dropout, no unk, no prefix/suffix, no byte fallback), ByteLevel
    // decoder - raises EInferError otherwise
    constructor Create(const aFileName: TFileName);
    /// the NFC-normalized text (invalid UTF-8 replaced first)
    function Normalize(const aText: RawUtf8): RawUtf8;
    /// the pre-tokenized pieces of the normalized text, as UTF-8
    function Split(const aText: RawUtf8): TRawUtf8DynArray;
    /// token ids of aText - every text is text, no control token is matched
    function Encode(const aText: RawUtf8): TIntegerDynArray;
    /// append the token ids of aText to aIds[0..aCount-1]
    procedure EncodeAppend(const aText: RawUtf8; var aIds: TIntegerDynArray;
      var aCount: PtrInt);
    /// the id of an added token by its content (e.g. '<|im_end|>'), -1 if none
    function TokenId(const aContent: RawUtf8): integer;
    /// the bytes of one token: '' for an id inside the vocabulary that has no
    // token; a BPE token may be part of a UTF-8 sequence only
    function TokenBytes(aId: integer): RawByteString;
    /// the concatenated bytes of the ids - valid UTF-8 only when no multi-byte
    // character is cut at the ends of the list
    function Decode(const aIds: array of integer): RawUtf8;
    /// highest id + 1 (BPE tokens and added tokens)
    function VocabSize: integer;
    /// number of BPE tokens (ids 0..BpeCount-1)
    property BpeCount: integer
      read fVocabCount;
    /// number of merges
    property MergeCount: integer
      read fMergeCount;
  end;


const
  /// the Split regex of Qwen2 / Qwen3 (tokenizer.json pre_tokenizer), verbatim
  QWEN2_SPLIT_REGEX =
    '(?i:''s|''t|''re|''ve|''m|''ll|''d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|' +
    ' ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+';


implementation

var
  /// GPT-2 byte-level alphabet: byte -> code point, and back (-1 = none)
  ByteToCp: array[byte] of word;
  CpToByte: array[0..511] of smallint;

procedure InitByteLevel;
var
  b, n: PtrInt;
begin
  // printable bytes map to themselves, the others to 256.. in byte order
  for b := 0 to high(CpToByte) do
    CpToByte[b] := -1;
  n := 0;
  for b := 0 to 255 do
  begin
    if ((b >= ord('!')) and (b <= ord('~'))) or ((b >= $a1) and (b <= $ac)) or
       ((b >= $ae) and (b <= $ff)) then
      ByteToCp[b] := b
    else
    begin
      ByteToCp[b] := 256 + n;
      inc(n);
    end;
    CpToByte[ByteToCp[b]] := b;
  end;
end;

/// the raw bytes of a byte-level token text, false if it has a foreign char
function ByteLevelDecode(const aText: RawUtf8; out aBytes: RawByteString): boolean;
var
  cp: TCardinalDynArray;
  n, i: PtrInt;
begin
  result := false;
  n := Utf8ToCodePoints(pointer(aText), length(aText), cp);
  SetLength(aBytes, n);
  for i := 0 to n - 1 do
  begin
    if (cp[i] > high(CpToByte)) or (CpToByte[cp[i]] < 0) then
      exit;
    PByteArray(aBytes)^[i] := CpToByte[cp[i]];
  end;
  result := n > 0;
end;

function MergeHash(aKey: Int64): PtrUInt;
  {$ifdef HASINLINE}inline;{$endif}
begin
  // Fibonacci hashing of the (left, right) pair - the low bits are masked
  result := PtrUInt(QWord(aKey) * QWord($9e3779b97f4a7c15) shr 32);
end;


{ TBpeTokenizer }

constructor TBpeTokenizer.Create(const aFileName: TFileName);
var
  doc: TDocVariantData;
  norm, pre, list, split, level, model, vocab, merges, entry, added,
  decoder: PDocVariantData;
  pair: PDocVariantData;
  i, n, id, maxId, h, sp: PtrInt;
  bytes, left, right: RawByteString;
  l, r, m, oldRank, oldNew: integer;
  text, a, b: RawUtf8;
  one: byte;
begin
  inherited Create;
  if not doc.InitJsonFromFile(aFileName, JSON_FAST_FLOAT) then
    EInferError.RaiseUtf8('%: cannot read %', [self, aFileName]);
  // the pipeline this class implements, and nothing else
  if not doc.GetAsObject('normalizer', norm) or (norm^.U['type'] <> 'NFC') then
    EInferError.RaiseUtf8('%: normalizer is not NFC', [self]);
  if not doc.GetAsObject('pre_tokenizer', pre) or (pre^.U['type'] <> 'Sequence') or
     not pre^.GetAsArray('pretokenizers', list) or (list^.Count <> 2) then
    EInferError.RaiseUtf8('%: pre_tokenizer is not Sequence[Split, ByteLevel]', [self]);
  split := _Safe(list^.Values[0]);
  if (split^.U['type'] <> 'Split') or (split^.U['behavior'] <> 'Isolated') or
     split^.B['invert'] or (split^.O['pattern']^.U['Regex'] <> QWEN2_SPLIT_REGEX) then
    EInferError.RaiseUtf8('%: first pre-tokenizer is not the Qwen2 Split', [self]);
  level := _Safe(list^.Values[1]);
  if (level^.U['type'] <> 'ByteLevel') or level^.B['add_prefix_space'] or
     level^.B['use_regex'] then
    EInferError.RaiseUtf8('%: second pre-tokenizer is not ByteLevel(no prefix, ' +
      'no regex)', [self]);
  if not doc.GetAsObject('decoder', decoder) or (decoder^.U['type'] <> 'ByteLevel') then
    EInferError.RaiseUtf8('%: decoder is not ByteLevel', [self]);
  if not doc.GetAsObject('model', model) or (model^.U['type'] <> 'BPE') or
     not VarIsEmptyOrNull(model^.Value['dropout']) or
     not VarIsEmptyOrNull(model^.Value['unk_token']) or
     (model^.U['continuing_subword_prefix'] <> '') or
     (model^.U['end_of_word_suffix'] <> '') or model^.B['fuse_unk'] or
     model^.B['byte_fallback'] or model^.B['ignore_merges'] then
    EInferError.RaiseUtf8('%: model is not a plain BPE', [self]);
  if not model^.GetAsObject('vocab', vocab) or (vocab^.Count = 0) or
     not model^.GetAsArray('merges', merges) then
    EInferError.RaiseUtf8('%: BPE without vocab or merges', [self]);
  // the vocabulary: ids 0..n-1, each exactly once, as raw bytes
  n := vocab^.Count;
  fVocabCount := n;
  maxId := n - 1;
  // added tokens: absent, or an array - anything else is no file of ours
  // (GetAsArray is false for an empty array too, hence GetAsDocVariant)
  added := nil;
  if doc.GetValueIndex('added_tokens') >= 0 then
    if not doc.GetAsDocVariant('added_tokens', added) or not added^.IsArray then
      EInferError.RaiseUtf8('%: added_tokens is not an array', [self]);
  if added <> nil then
    for i := 0 to added^.Count - 1 do
    begin
      id := _Safe(added^.Values[i])^.I['id'];
      if (id < 0) or (id > n + 1 shl 20) then
        EInferError.RaiseUtf8('%: added token id % out of range', [self, id]);
      if id > maxId then
        maxId := id;
    end;
  SetLength(fTokens, maxId + 1);
  for i := 0 to n - 1 do
  begin
    id := VariantToIntegerDef(vocab^.Values[i], -1);
    if (id < 0) or (id >= n) or (fTokens[id] <> '') then
      EInferError.RaiseUtf8('%: vocab id % of "%" invalid or used twice',
        [self, id, vocab^.Names[i]]);
    if not ByteLevelDecode(vocab^.Names[i], bytes) then
      EInferError.RaiseUtf8('%: vocab entry "%" is not byte-level text',
        [self, vocab^.Names[i]]);
    fTokens[id] := bytes;
  end;
  // hash of the byte strings, chained by id: lookups allocate nothing
  fHashMask := 1;
  while fHashMask < n * 2 do
    fHashMask := fHashMask shl 1;
  dec(fHashMask);
  SetLength(fHashHead, fHashMask + 1);
  FillCharFast(pointer(fHashHead)^, length(fHashHead) * SizeOf(integer), $ff); // -1
  SetLength(fHashNext, n);
  for i := 0 to n - 1 do
  begin
    h := crc32c(0, pointer(fTokens[i]), length(fTokens[i])) and fHashMask;
    fHashNext[i] := fHashHead[h];
    fHashHead[h] := i;
  end;
  for i := 0 to 255 do
  begin
    one := i;
    fByteId[i] := BytesId(@one, 1);
    if fByteId[i] < 0 then
      EInferError.RaiseUtf8('%: no token for byte %', [self, i]);
  end;
  // the merges: (left, right) -> rank and merged id
  fMergeCount := merges^.Count;
  fMergeMask := 1;
  while fMergeMask < fMergeCount * 2 + 2 do
    fMergeMask := fMergeMask shl 1;
  dec(fMergeMask);
  SetLength(fMergeKey, fMergeMask + 1);
  FillCharFast(pointer(fMergeKey)^, length(fMergeKey) * SizeOf(Int64), $ff); // -1
  SetLength(fMergeRank, fMergeMask + 1);
  SetLength(fMergeNew, fMergeMask + 1);
  for i := 0 to fMergeCount - 1 do
  begin
    // ["a", "b"] (current format) or "a b" (older files)
    if _Safe(merges^.Values[i])^.IsArray then
    begin
      pair := _Safe(merges^.Values[i]);
      if pair^.Count <> 2 then
        EInferError.RaiseUtf8('%: merge % is not a pair', [self, i]);
      VariantToUtf8(pair^.Values[0], a);
      VariantToUtf8(pair^.Values[1], b);
    end
    else
    begin
      VariantToUtf8(merges^.Values[i], text);
      sp := PosEx(' ', text);
      if (sp <= 1) or (sp = length(text)) or (PosEx(' ', text, sp + 1) > 0) then
        EInferError.RaiseUtf8('%: merge % "%" is not "a b"', [self, i, text]);
      a := copy(text, 1, sp - 1);
      b := copy(text, sp + 1, maxInt);
    end;
    if not ByteLevelDecode(a, left) or not ByteLevelDecode(b, right) then
      EInferError.RaiseUtf8('%: merge % is not byte-level text', [self, i]);
    l := BytesId(pointer(left), length(left));
    r := BytesId(pointer(right), length(right));
    bytes := left + right;
    m := BytesId(pointer(bytes), length(bytes));
    if (l < 0) or (r < 0) or (m < 0) then
      EInferError.RaiseUtf8('%: merge % "% %" outside the vocabulary', [self, i, a, b]);
    if FindMerge(l, r, oldRank, oldNew) then
      // the tokenizers library would keep the last rank: refuse instead
      EInferError.RaiseUtf8('%: merge % "% %" repeats merge %',
        [self, i, a, b, oldRank]);
    AddMerge(l, r, i, m);
  end;
  // added tokens: never matched in text, reachable by TokenId
  if added <> nil then
  begin
    SetLength(fAddedText, added^.Count);
    SetLength(fAddedId, added^.Count);
    for i := 0 to added^.Count - 1 do
    begin
      entry := _Safe(added^.Values[i]);
      id := entry^.I['id'];
      text := entry^.U['content'];
      if (id < 0) or (id > high(fTokens)) or (text = '') or
         ((id < n) and (fTokens[id] <> text)) or ((id >= n) and (fTokens[id] <> '')) then
        EInferError.RaiseUtf8('%: added token % "%" clashes with the vocabulary',
          [self, id, text]);
      fTokens[id] := text;
      fAddedText[i] := text;
      fAddedId[i] := id;
    end;
  end;
end;

procedure TBpeTokenizer.AddMerge(aLeft, aRight, aRank, aNew: integer);
var
  key: Int64;
  h: PtrInt;
begin
  key := (Int64(aLeft) shl 32) or cardinal(aRight);
  h := MergeHash(key) and fMergeMask;
  while fMergeKey[h] <> -1 do
    h := (h + 1) and fMergeMask;
  fMergeKey[h] := key;
  fMergeRank[h] := aRank;
  fMergeNew[h] := aNew;
end;

function TBpeTokenizer.FindMerge(aLeft, aRight: integer;
  out aRank, aNew: integer): boolean;
var
  key: Int64;
  h: PtrInt;
begin
  key := (Int64(aLeft) shl 32) or cardinal(aRight);
  h := MergeHash(key) and fMergeMask;
  repeat
    if fMergeKey[h] = key then
    begin
      aRank := fMergeRank[h];
      aNew := fMergeNew[h];
      result := true;
      exit;
    end;
    if fMergeKey[h] = -1 then
    begin
      result := false;
      exit;
    end;
    h := (h + 1) and fMergeMask;
  until false;
end;

function TBpeTokenizer.BytesId(p: pointer; aLen: PtrInt): integer;
begin
  result := -1;
  if aLen <= 0 then
    exit;
  result := fHashHead[crc32c(0, p, aLen) and fHashMask];
  while result >= 0 do
  begin
    if (length(fTokens[result]) = aLen) and
       CompareMem(pointer(fTokens[result]), p, aLen) then
      exit;
    result := fHashNext[result];
  end;
end;

procedure TBpeQueue.Push(aRank, aPos, aNew: integer);
var
  i, parent: PtrInt;
  k: Int64;
begin
  if Count = length(Key) then
  begin
    SetLength(Key, Count * 2 + 16);
    SetLength(New, Count * 2 + 16);
  end;
  k := (Int64(aRank) shl 32) or cardinal(aPos);
  i := Count;
  inc(Count);
  while i > 0 do
  begin
    parent := (i - 1) shr 1;
    if Key[parent] <= k then
      break;
    Key[i] := Key[parent];
    New[i] := New[parent];
    i := parent;
  end;
  Key[i] := k;
  New[i] := aNew;
end;

function TBpeQueue.Pop(out aPos, aNew: integer): boolean;
var
  i, child: PtrInt;
  k: Int64;
  n: integer;
begin
  result := Count > 0;
  if not result then
    exit;
  aPos := integer(Key[0] and $ffffffff);
  aNew := New[0];
  dec(Count);
  if Count = 0 then
    exit;
  k := Key[Count];
  n := New[Count];
  i := 0;
  repeat
    child := i * 2 + 1;
    if child >= Count then
      break;
    if (child + 1 < Count) and (Key[child + 1] < Key[child]) then
      inc(child);
    if Key[child] >= k then
      break;
    Key[i] := Key[child];
    New[i] := New[child];
    i := child;
  until false;
  Key[i] := k;
  New[i] := n;
end;

procedure TBpeTokenizer.Bpe(p: PByteArray; aLen: PtrInt;
  var aIds: TIntegerDynArray; var aCount: PtrInt; var aSym: TBpeSymbolDynArray;
  var aQueue: TBpeQueue);
var
  i, pos, next, prev, rank, newId, cur: integer;
begin
  if aLen <= 0 then
    exit;
  // one symbol per byte, then the merges of the tokenizers library: lowest
  // rank first, leftmost first on equal rank; a queue entry whose pair no
  // longer produces its merged id is stale and skipped
  if length(aSym) < aLen then
    SetLength(aSym, aLen);
  for i := 0 to aLen - 1 do
  begin
    aSym[i].Id := fByteId[p^[i]];
    aSym[i].Prev := i - 1;
    aSym[i].Next := i + 1;
    aSym[i].Len := 1;
  end;
  aSym[aLen - 1].Next := -1;
  aQueue.Count := 0; // the arrays are kept between the pieces of one text
  for i := 0 to aLen - 2 do
    if FindMerge(aSym[i].Id, aSym[i + 1].Id, rank, newId) then
      aQueue.Push(rank, i, newId);
  while aQueue.Pop(pos, newId) do
  begin
    if (aSym[pos].Len = 0) or (aSym[pos].Next < 0) then
      continue;
    next := aSym[pos].Next;
    if not FindMerge(aSym[pos].Id, aSym[next].Id, rank, cur) or (cur <> newId) then
      continue;
    // merge the right symbol into this one
    aSym[pos].Id := newId;
    inc(aSym[pos].Len, aSym[next].Len);
    aSym[pos].Next := aSym[next].Next;
    aSym[next].Len := 0;
    if aSym[pos].Next >= 0 then
      aSym[aSym[pos].Next].Prev := pos;
    prev := aSym[pos].Prev;
    if (prev >= 0) and FindMerge(aSym[prev].Id, aSym[pos].Id, rank, cur) then
      aQueue.Push(rank, prev, cur);
    next := aSym[pos].Next;
    if (next >= 0) and FindMerge(aSym[pos].Id, aSym[next].Id, rank, cur) then
      aQueue.Push(rank, pos, cur);
  end;
  i := 0;
  while i >= 0 do
  begin
    if aCount = length(aIds) then
      SetLength(aIds, aCount * 2 + 16);
    aIds[aCount] := aSym[i].Id;
    inc(aCount);
    i := aSym[i].Next;
  end;
end;

/// ASCII-only case folding of the contraction letters, plus U+017F -> s
// (measured: (?i) matches exactly these, see make_reference_qwen3.py)
function FoldContraction(aCp: cardinal): cardinal;
  {$ifdef HASINLINE}inline;{$endif}
begin
  if (aCp >= ord('A')) and (aCp <= ord('Z')) then
    result := aCp + 32
  else if aCp = $17f then
    result := ord('s')
  else
    result := aCp;
end;

/// end of the regex match starting at aPos (exclusive) - see the unit header
function SplitEnd(cp: PCardinalArray; aPos, aCount: PtrInt): PtrInt;
var
  c0, c1, c2: cardinal;
  k0: TUnicodeClass;
  j, k: PtrInt;

  function IsCrLf(c: cardinal): boolean;
  begin
    result := (c = 10) or (c = 13);
  end;

  function IsOther(i: PtrInt): boolean; // [^\s\p{L}\p{N}]
  begin
    result := UnicodeClassOf(cp^[i]) = ucOther;
  end;

begin
  c0 := cp^[aPos];
  k0 := UnicodeClassOf(c0);
  // 1. (?i:'s|'t|'re|'ve|'m|'ll|'d)
  if (c0 = ord('''')) and (aPos + 1 < aCount) then
  begin
    c1 := FoldContraction(cp^[aPos + 1]);
    if (c1 = ord('s')) or (c1 = ord('t')) or (c1 = ord('m')) or (c1 = ord('d')) then
    begin
      result := aPos + 2;
      exit;
    end;
    if aPos + 2 < aCount then
    begin
      c2 := FoldContraction(cp^[aPos + 2]);
      if (((c1 = ord('r')) or (c1 = ord('v'))) and (c2 = ord('e'))) or
         ((c1 = ord('l')) and (c2 = ord('l'))) then
      begin
        result := aPos + 3;
        exit;
      end;
    end;
  end;
  // 2. [^\r\n\p{L}\p{N}]?\p{L}+
  j := -1;
  if k0 = ucLetter then
    j := aPos
  else if (k0 <> ucNumber) and not IsCrLf(c0) and (aPos + 1 < aCount) and
          (UnicodeClassOf(cp^[aPos + 1]) = ucLetter) then
    j := aPos + 1;
  if j >= 0 then
  begin
    while (j < aCount) and (UnicodeClassOf(cp^[j]) = ucLetter) do
      inc(j);
    result := j;
    exit;
  end;
  // 3. \p{N}
  if k0 = ucNumber then
  begin
    result := aPos + 1;
    exit;
  end;
  // 4. ' '?[^\s\p{L}\p{N}]+[\r\n]*
  k := aPos;
  if (c0 = 32) and (aPos + 1 < aCount) and IsOther(aPos + 1) then
    k := aPos + 1;
  if IsOther(k) then
  begin
    j := k + 1;
    while (j < aCount) and IsOther(j) do
      inc(j);
    while (j < aCount) and IsCrLf(cp^[j]) do
      inc(j);
    result := j;
    exit;
  end;
  // only white space is left: its run is aPos..j-1
  j := aPos + 1;
  while (j < aCount) and (UnicodeClassOf(cp^[j]) = ucSpace) do
    inc(j);
  // 5. \s*[\r\n]+ - up to the last CR/LF of the run
  k := j - 1;
  while (k >= aPos) and not IsCrLf(cp^[k]) do
    dec(k);
  if k >= aPos then
    result := k + 1
  // 6. \s+(?!\S) - the run up to the end, or all but its last char
  else if j = aCount then
    result := j
  else if j - aPos >= 2 then
    result := j - 1
  // 7. \s+ - a single white space before a non-space
  else
    result := j;
end;

function TBpeTokenizer.Normalize(const aText: RawUtf8): RawUtf8;
begin
  result := NfcNormalize(aText);
end;

function TBpeTokenizer.Split(const aText: RawUtf8): TRawUtf8DynArray;
var
  cp: TCardinalDynArray;
  n, pos, e, count: PtrInt;
begin
  result := nil;
  n := Utf8ToCodePoints(pointer(aText), length(aText), cp);
  NfcCodePoints(cp, n);
  count := 0;
  pos := 0;
  while pos < n do
  begin
    e := SplitEnd(pointer(cp), pos, n);
    if count = length(result) then
      SetLength(result, count * 2 + 8);
    result[count] := CodePointsToUtf8(@cp[pos], e - pos);
    inc(count);
    pos := e;
  end;
  SetLength(result, count);
end;

procedure TBpeTokenizer.EncodeAppend(const aText: RawUtf8;
  var aIds: TIntegerDynArray; var aCount: PtrInt);
var
  cp: TCardinalDynArray;
  n, pos, e, len: PtrInt;
  bytes: TByteDynArray;
  sym: TBpeSymbolDynArray;
  queue: TBpeQueue;
begin
  // one set of buffers for all pieces of the text: no allocation per piece
  queue.Count := 0;
  n := Utf8ToCodePoints(pointer(aText), length(aText), cp);
  NfcCodePoints(cp, n);
  SetLength(bytes, n * 4 + 4);
  pos := 0;
  while pos < n do
  begin
    e := SplitEnd(pointer(cp), pos, n);
    len := CodePointsToUtf8Buffer(@cp[pos], e - pos, pointer(bytes));
    Bpe(pointer(bytes), len, aIds, aCount, sym, queue);
    pos := e;
  end;
end;

function TBpeTokenizer.Encode(const aText: RawUtf8): TIntegerDynArray;
var
  n: PtrInt;
begin
  result := nil;
  n := 0;
  EncodeAppend(aText, result, n);
  SetLength(result, n);
end;

function TBpeTokenizer.TokenId(const aContent: RawUtf8): integer;
var
  i: PtrInt;
begin
  for i := 0 to high(fAddedText) do
    if fAddedText[i] = aContent then
    begin
      result := fAddedId[i];
      exit;
    end;
  result := -1;
end;

function TBpeTokenizer.TokenBytes(aId: integer): RawByteString;
begin
  if (aId < 0) or (aId > high(fTokens)) then
    EInferError.RaiseUtf8('%.TokenBytes: id % outside 0..%',
      [self, aId, high(fTokens)]);
  result := fTokens[aId];
end;

function TBpeTokenizer.Decode(const aIds: array of integer): RawUtf8;
var
  i, n: PtrInt;
begin
  n := 0;
  for i := 0 to high(aIds) do
    inc(n, length(TokenBytes(aIds[i])));
  FastSetString(result, n);
  n := 0;
  for i := 0 to high(aIds) do
  begin
    MoveFast(pointer(fTokens[aIds[i]])^, PByteArray(result)^[n], length(fTokens[aIds[i]]));
    inc(n, length(fTokens[aIds[i]]));
  end;
end;

function TBpeTokenizer.VocabSize: integer;
begin
  result := length(fTokens);
end;


initialization
  InitByteLevel;

end.
