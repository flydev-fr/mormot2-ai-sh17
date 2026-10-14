/// LandrixAI - shared SQLite helpers for the mormot.ai.* SQLite backends
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - RollbackIfOpen: the transaction-recovery rule every SQLite-backed store needs
//   (mORMot's Commit clears its flag even when COMMIT fails)
// - run-time extension loading (sqlite-vec / sqlite-lembed): mORMot does not bind
//   sqlite3_enable_load_extension, so loading is enabled via
//   db_config(SQLITE_DBCONFIG_ENABLE_LOAD_EXTENSION) - proven cross-platform (FPC,
//   Linux+Windows); the extension shared libs live next to their llama/ggml deps
// - lives apart from the concrete stores so that a store without extensions (the
//   native one) never depends on the sqlite-vec unit
unit mormot.ai.sqlite;

{
  *****************************************************************************

    - RollbackIfOpen: transaction recovery shared by the SQLite backends
    - SqliteExtSuffix / EnableSqliteExtensions / LoadSqliteExtension:
      run-time loading of sqlite-vec and sqlite-lembed

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
// - TSqlDataBase.Commit clears TransactionActive even when COMMIT itself fails
// (e.g. SQLITE_BUSY), and RollBack checks exactly that flag - so an
// `except RollBack` after a failed Commit is a no-op and the transaction stays
// open: every later TransactionBegin fails and single writes silently join it.
// This decides on SQLite's own state (get_autocommit) instead.
// - never raises: a failed rollback must not hide the original exception
procedure RollbackIfOpen(aDB: TSqlDataBase);

/// the platform shared-library suffix for a SQLite extension
function SqliteExtSuffix: RawUtf8;

/// enable run-time extension loading on a connection (mORMot omits the dedicated
/// enable API, so SQLITE_DBCONFIG_ENABLE_LOAD_EXTENSION via db_config is used)
procedure EnableSqliteExtensions(aDB: TSqlite3DB);

/// load a SQLite extension shared library by absolute path
procedure LoadSqliteExtension(aDB: TSqlite3DB; const aPath: RawUtf8);


implementation

procedure RollbackIfOpen(aDB: TSqlDataBase);
begin
  if (aDB = nil) or (aDB.DB = 0) then
    exit;
  try
    if sqlite3.get_autocommit(aDB.DB) = 0 then
      if aDB.TransactionActive then
        aDB.RollBack // also resets mORMot's flag
      else
        aDB.Execute('ROLLBACK TRANSACTION;'); // flag already gone (failed Commit)
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

procedure EnableSqliteExtensions(aDB: TSqlite3DB);
var
  enabled: integer;
begin
  enabled := 0;
  if not Assigned(sqlite3.db_config) then
    ESqlite3Exception.RaiseUtf8('EnableSqliteExtensions: db_config unavailable', []);
  if sqlite3.db_config(aDB, SQLITE_DBCONFIG_ENABLE_LOAD_EXTENSION, 1, @enabled) <> SQLITE_OK then
    ESqlite3Exception.RaiseUtf8('EnableSqliteExtensions failed: %',
      [Utf8ToString(sqlite3.errmsg(aDB))]);
end;

procedure LoadSqliteExtension(aDB: TSqlite3DB; const aPath: RawUtf8);
var
  msg: PUtf8Char;
begin
  msg := nil;
  if sqlite3.load_extension(aDB, pointer(aPath), nil, msg) <> SQLITE_OK then
    ESqlite3Exception.RaiseUtf8('LoadSqliteExtension % failed: %',
      [aPath, Utf8ToString(msg)]);
end;

end.
