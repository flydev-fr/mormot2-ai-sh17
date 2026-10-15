/// LandrixAI - shared SQLite helpers for the mormot.ai.* SQLite backends
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - RollbackIfOpen: the transaction-recovery rule every SQLite-backed store needs
//   (roll back whatever is open, never raise from an except branch)
// - run-time extension loading (sqlite-vec / sqlite-lembed) is mORMot's own
//   TSqlDataBase.LoadExtension (mORMot2 6302d231e, 2026-10-15): it enables the C
//   loading API only for the call and never the SQL load_extension() function; the
//   extension shared libs live next to their llama/ggml deps
// - its TFileName goes through the ANSI code page where string is not Unicode
//   (Delphi 7, FPC on Windows): an extension folder with characters outside it
//   cannot be loaded there - there are no 32-bit builds of vec0 and lembed anyway
// - lives apart from the concrete stores so that a store without extensions (the
//   native one) never depends on the sqlite-vec unit
unit mormot.ai.sqlite;

{
  *****************************************************************************

    - RollbackIfOpen: transaction recovery shared by the SQLite backends
    - SqliteExtSuffix: the shared-library suffix of sqlite-vec and sqlite-lembed

  *****************************************************************************
}

interface

{$I mormot.defines.inc}

uses
  sysutils,
  mormot.core.base,
  mormot.core.unicode, // Utf8ToString (mORMot's, as before the extraction)
  mormot.core.text,
  mormot.db.raw.sqlite3;

/// roll back the open transaction of aDB, if SQLite says there is one
// - TransactionActive and RollBack follow SQLite's own state since mORMot2
// f08998d82 (2026-10-15) - before, a failed Commit (e.g. SQLITE_BUSY) cleared
// the flag and left the transaction open, so this unit asked get_autocommit
// itself; that is mORMot's job now
// - never raises: a failed rollback must not hide the original exception
procedure RollbackIfOpen(aDB: TSqlDataBase);

/// the platform shared-library suffix for a SQLite extension
function SqliteExtSuffix: RawUtf8;



implementation

procedure RollbackIfOpen(aDB: TSqlDataBase);
begin
  if (aDB = nil) or (aDB.DB = 0) then
    exit;
  try
    aDB.RollBack; // nothing when SQLite has no transaction open
  except
    // keep the original exception
  end;
end;

function SqliteExtSuffix: RawUtf8;
begin
  {$ifdef OSWINDOWS}
  result := '.dll';
  {$else}
  {$ifdef OSDARWIN}
  result := '.dylib';
  {$else}
  result := '.so';
  {$endif OSDARWIN}
  {$endif OSWINDOWS}
end;

end.
