// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: enzymexlamlir-opt %s --sql-lower --split-input-file --verify-diagnostics

func.func @missing_declaration(%handle: !llvm.ptr) -> i32 {
  // expected-error @+1 {{requires an existing PostgreSQL declaration}}
  %count = "sql.num_results"(%handle) : (!llvm.ptr) -> i32
  return %count : i32
}

// -----

llvm.func @PQntuples(!llvm.ptr) -> i32
llvm.func @argument_mismatch(%handle: !llvm.ptr<1>) -> i32 {
  // expected-error @+1 {{PostgreSQL argument type mismatch}}
  %count = "sql.num_results"(%handle) : (!llvm.ptr<1>) -> i32
  llvm.return %count : i32
}

// -----

llvm.func @PQntuples(!llvm.ptr) -> i32
llvm.func @result_mismatch(%handle: !llvm.ptr) -> i64 {
  // expected-error @+1 {{PostgreSQL result type mismatch}}
  %count = "sql.num_results"(%handle) : (!llvm.ptr) -> i64
  llvm.return %count : i64
}

// -----

func.func private @PQntuples(!llvm.ptr) -> i64
func.func @malformed_metadata(%handle: !llvm.ptr) -> i64 {
  // expected-error @+1 {{sql.source_call must be a dictionary}}
  %count = "sql.num_results"(%handle) {sql.source_call = "invalid"} : (!llvm.ptr) -> i64
  return %count : i64
}

// -----

func.func private @PQntuples(!llvm.ptr) -> i64
func.func @metadata_type_mismatch(%handle: !llvm.ptr) -> i64 {
  // expected-error @+1 {{sql.source_call disagrees with the PostgreSQL declaration}}
  %count = "sql.num_results"(%handle) {sql.source_call = {kind = "func", type = (!llvm.ptr) -> i32, attributes = {callee = @PQntuples}}} : (!llvm.ptr) -> i64
  return %count : i64
}

// -----

func.func private @PQntuples(!llvm.ptr) -> i64
func.func @metadata_callee_mismatch(%handle: !llvm.ptr) -> i64 {
  // expected-error @+1 {{sql.source_call disagrees with the PostgreSQL declaration}}
  %count = "sql.num_results"(%handle) {sql.source_call = {kind = "func", type = (!llvm.ptr) -> i64, attributes = {callee = @PQexec}}} : (!llvm.ptr) -> i64
  return %count : i64
}

// -----

llvm.func fastcc @PQntuples(!llvm.ptr) -> i32
llvm.func @convention_mismatch(%handle: !llvm.ptr) -> i32 {
  // expected-error @+1 {{sql.source_call calling convention mismatch}}
  %count = "sql.num_results"(%handle) {sql.source_call = {kind = "llvm", type = (!llvm.ptr) -> i32, attributes = {callee = @PQntuples}, calling_convention = #llvm.cconv<ccc>}} : (!llvm.ptr) -> i32
  llvm.return %count : i32
}

// -----

func.func private @PQgetvalue(!llvm.ptr, i32, i32) -> i32
func.func @numeric_extraction(%handle: !llvm.ptr, %row: i32, %col: i32) -> i32 {
  // expected-error @+1 {{PQgetvalue requires a text result}}
  %value = "sql.get_value"(%handle, %col, %row) : (!llvm.ptr, i32, i32) -> i32
  return %value : i32
}

// -----

func.func @dynamic_concat(%left: !llvm.ptr, %right: !llvm.ptr) -> !llvm.ptr {
  // expected-error @+1 {{dynamic string concatenation is not supported}}
  %text = "sql.string_concat"(%left, %right) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  return %text : !llvm.ptr
}

// -----

func.func @dynamic_string_type(%source: !llvm.ptr<1>) -> !llvm.ptr {
  %query = "sql.unparsed"(%source) : (!llvm.ptr<1>) -> !sql.expr
  // expected-error @+1 {{dynamic string type mismatch}}
  %text = "sql.to_string"(%query) : (!sql.expr) -> !llvm.ptr
  return %text : !llvm.ptr
}

// -----

func.func @unsupported_arithmetic() -> !llvm.ptr {
  %a = "sql.int"() <{expr = "1"}> : () -> !sql.expr
  %b = "sql.int"() <{expr = "2"}> : () -> !sql.expr
  %expr = "sql.arith"(%a, %b) <{op = "^"}> : (!sql.expr, !sql.expr) -> !sql.expr
  // expected-error @+1 {{unsupported arithmetic operator}}
  %text = "sql.to_string"(%expr) : (!sql.expr) -> !llvm.ptr
  return %text : !llvm.ptr
}

// -----

func.func @small_buffer() -> memref<1xi8> {
  // expected-error @+1 {{serialized query does not fit the declared static memref}}
  %text = "sql.str_constant"() <{input = "SELECT 1"}> : () -> memref<1xi8>
  return %text : memref<1xi8>
}

// -----

func.func @non_text_string() -> i64 {
  // expected-error @+1 {{string materialization requires a pointer or identity byte memref}}
  %text = "sql.str_constant"() <{input = "SELECT 1"}> : () -> i64
  return %text : i64
}

// -----

// expected-error @+1 {{SQL types cannot escape}}
func.func @escaping_expression() -> !sql.expr {
  %value = "sql.int"() <{expr = "1"}> : () -> !sql.expr
  return %value : !sql.expr
}

// -----

llvm.func @PQexec(!llvm.ptr, !llvm.ptr) -> !llvm.ptr
llvm.func @empty_projection(%conn: !llvm.ptr) -> !llvm.ptr {
  %table = "sql.table"() <{expr = "t"}> : () -> !sql.expr
  %true = "sql.constant_bool"() : () -> !sql.bool
  %where = "sql.where"(%true) : (!sql.bool) -> !sql.expr
  %query = "sql.select"(%table, %where) <{limit = -1 : si64}> : (!sql.expr, !sql.expr) -> !sql.expr
  // expected-error @+1 {{SELECT requires at least one projection}}
  %result = "sql.execute"(%conn, %query) : (!llvm.ptr, !sql.expr) -> !llvm.ptr
  llvm.return %result : !llvm.ptr
}

// -----

func.func @malformed_text_leaf() -> !llvm.ptr {
  %column = "sql.column"() <{expr = "x; SELECT y"}> : () -> !sql.expr
  // expected-error @+1 {{SQL parse error}}
  %text = "sql.to_string"(%column) : (!sql.expr) -> !llvm.ptr
  return %text : !llvm.ptr
}

// -----

llvm.func @PQntuples(!llvm.ptr) -> i32
llvm.func @invalid_saved_property(%handle: !llvm.ptr) -> i32 {
  // expected-error @+1 {{invalid source call properties}}
  %count = "sql.num_results"(%handle) {sql.source_call = {kind = "llvm", type = (!llvm.ptr) -> i32, attributes = {callee = @PQntuples, operandSegmentSizes = "bad"}, calling_convention = #llvm.cconv<ccc>}} : (!llvm.ptr) -> i32
  llvm.return %count : i32
}
