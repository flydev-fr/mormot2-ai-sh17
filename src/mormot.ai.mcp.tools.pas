/// MCP example tool parameter records
// - part of the mormot.ai.* extension (LandrixAI)
// - licensed under a MPL/GPL/LGPL three license - see LICENSE
// - adopted into the mormot.ai.* namespace for landrix (LandrixAI) from
//   flydev-fr/mormot2-extensions
unit mormot.ai.mcp.tools;

{
  *****************************************************************************

    - TRunExecutableParams: sample parameter record for a typed MCP tool

  *****************************************************************************
}

interface

{$I mormot.defines.inc}

uses
  mormot.core.base;

type
  /// sample input of a tool that would start an executable
  // - shows the record shape a typed tool consumes: each field becomes a
  // property of the generated input schema (see TMcpRecordTool)
  // - no tool in this extension implements it: starting processes on behalf
  // of a client is a decision for the application, not for a library
  TRunExecutableParams = record
    /// the executable to start
    Path: RawUtf8;
    /// its command line arguments
    Args: RawUtf8;
    /// true to wait for the process to finish
    Wait: boolean;
  end;

implementation

end.
