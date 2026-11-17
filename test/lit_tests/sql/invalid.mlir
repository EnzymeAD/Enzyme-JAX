// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: enzymexlamlir-opt %s --split-input-file --verify-diagnostics

func.func @invalid_arith(%bad: i32, %expr: !sql.expr) -> !sql.expr {
  // expected-error @+1 {{'sql.arith' op operand #0 must be}}
  %0 = "sql.arith"(%bad, %expr) <{op = "+"}> : (i32, !sql.expr) -> !sql.expr
  return %0 : !sql.expr
}

// -----

func.func @invalid_and(%bad: !sql.expr, %pred: !sql.bool) -> !sql.bool {
  // expected-error @+1 {{'sql.and' op operand #0 must be}}
  %0 = "sql.and"(%bad, %pred) : (!sql.expr, !sql.bool) -> !sql.bool
  return %0 : !sql.bool
}

// -----

func.func @invalid_or(%pred: !sql.bool, %bad: i1) -> !sql.bool {
  // expected-error @+1 {{'sql.or' op operand #1 must be}}
  %0 = "sql.or"(%pred, %bad) : (!sql.bool, i1) -> !sql.bool
  return %0 : !sql.bool
}

// -----

func.func @invalid_comparison(%bad: !sql.bool, %expr: !sql.expr) -> !sql.bool {
  // expected-error @+1 {{'sql.bool_arith' op operand #0 must be}}
  %0 = "sql.bool_arith"(%bad, %expr) <{op = "="}> : (!sql.bool, !sql.expr) -> !sql.bool
  return %0 : !sql.bool
}

// -----

func.func @invalid_where(%bad: !sql.expr) -> !sql.expr {
  // expected-error @+1 {{'sql.where' op operand #0 must be}}
  %0 = "sql.where"(%bad) : (!sql.expr) -> !sql.expr
  return %0 : !sql.expr
}

// -----

func.func @invalid_select_column(%bad: i32, %table: !sql.expr, %where: !sql.expr) -> !sql.expr {
  // expected-error @+1 {{'sql.select' op operand #0 must be}}
  %0 = "sql.select"(%bad, %table, %where) <{limit = 1 : si64}> : (i32, !sql.expr, !sql.expr) -> !sql.expr
  return %0 : !sql.expr
}

// -----

func.func @invalid_select_where(%col: !sql.expr, %table: !sql.expr, %bad: !sql.bool) -> !sql.expr {
  // expected-error @+1 {{'sql.select' op operand #2 must be}}
  %0 = "sql.select"(%col, %table, %bad) <{limit = 1 : si64}> : (!sql.expr, !sql.expr, !sql.bool) -> !sql.expr
  return %0 : !sql.expr
}

// -----

func.func @invalid_connection(%bad: i64, %query: !sql.expr) -> index {
  // expected-error @+1 {{'sql.execute' op operand #0 must be}}
  %0 = "sql.execute"(%bad, %query) : (i64, !sql.expr) -> index
  return %0 : index
}

// -----

func.func @invalid_int_result() -> i32 {
  // expected-error @+1 {{'sql.int' op result #0 must be}}
  %0 = "sql.int"() <{expr = "42"}> : () -> i32
  return %0 : i32
}
