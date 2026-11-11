// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Failures must come from the SQL stage in the real pipeline, before export.
// RUN: not %S/../reactant-pipeline-driver cpu %S/Inputs/pipeline_malformed.ll 2>&1 | FileCheck %s --check-prefix=MALFORMED --implicit-check-not='failed to translate' --implicit-check-not='failed to parse pass pipeline'
// RUN: not %S/../reactant-pipeline-driver xla %S/Inputs/pipeline_unsupported.ll 2>&1 | FileCheck %s --check-prefix=SQL-ERROR --implicit-check-not='failed to translate' --implicit-check-not='failed to parse pass pipeline'
// RUN: not %S/../reactant-pipeline-driver cpu %S/Inputs/pipeline_unsupported_abi.ll 2>&1 | FileCheck %s --check-prefix=ABI --implicit-check-not='failed to translate' --implicit-check-not='failed to parse pass pipeline'

// MALFORMED: SQL parse error at byte
// MALFORMED-SAME: expected an identifier
// SQL-ERROR: SQL parse error at byte
// SQL-ERROR-SAME: SELECT modifiers are not supported
// ABI: sql-lower: unsupported attribute on a rematerialized query argument
