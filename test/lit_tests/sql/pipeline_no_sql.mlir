// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A whole input module without SQL must still traverse both default backends.
// RUN: %S/../reactant-pipeline-driver cpu %S/Inputs/pipeline_no_sql.ll | FileCheck %s --implicit-check-not=PQexec --implicit-check-not=sql. --implicit-check-not=__sql_query
// RUN: %S/../reactant-pipeline-driver xla %S/Inputs/pipeline_no_sql.ll | FileCheck %s --implicit-check-not=PQexec --implicit-check-not=sql. --implicit-check-not=__sql_query

// CHECK-LABEL: define i64 @non_sql(
// CHECK-SAME: i64 %[[INPUT:[^ ,)]+]]
// CHECK: %[[RESULT:[^ ,=]+]] = add i64 %[[INPUT]], 17
// CHECK: ret i64 %[[RESULT]]
