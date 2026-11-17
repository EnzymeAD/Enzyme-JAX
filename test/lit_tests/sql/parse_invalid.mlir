// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: enzymexlamlir-opt %s --sql-raising --split-input-file --verify-diagnostics

func.func @empty() -> !sql.expr {
  %str = "sql.str_constant"() <{input = ""}> : () -> memref<?xi8>
  // expected-error @+1 {{only SELECT statements are supported}}
  %query = "sql.unparsed"(%str) : (memref<?xi8>) -> !sql.expr
  return %query : !sql.expr
}

// -----

func.func @missing_projection() -> !sql.expr {
  %str = "sql.str_constant"() <{input = "SELECT FROM t"}> : () -> memref<?xi8>
  // expected-error @+1 {{expected an identifier}}
  %query = "sql.unparsed"(%str) : (memref<?xi8>) -> !sql.expr
  return %query : !sql.expr
}

// -----

func.func @multiple_statements() -> !sql.expr {
  %str = "sql.str_constant"() <{input = "SELECT 1; SELECT 2"}> : () -> memref<?xi8>
  // expected-error @+1 {{unsupported clause or trailing input}}
  %query = "sql.unparsed"(%str) : (memref<?xi8>) -> !sql.expr
  return %query : !sql.expr
}

// -----

func.func @distinct() -> !sql.expr {
  %str = "sql.str_constant"() <{input = "SELECT DISTINCT a FROM t"}> : () -> memref<?xi8>
  // expected-error @+1 {{SELECT modifiers are not supported}}
  %query = "sql.unparsed"(%str) : (memref<?xi8>) -> !sql.expr
  return %query : !sql.expr
}

// -----

func.func @join() -> !sql.expr {
  %str = "sql.str_constant"() <{input = "SELECT a FROM t JOIN u ON t.x=u.x"}> : () -> memref<?xi8>
  // expected-error @+1 {{unsupported clause or trailing input}}
  %query = "sql.unparsed"(%str) : (memref<?xi8>) -> !sql.expr
  return %query : !sql.expr
}

// -----

func.func @alias() -> !sql.expr {
  %str = "sql.str_constant"() <{input = "SELECT a AS b FROM t"}> : () -> memref<?xi8>
  // expected-error @+1 {{unsupported clause or trailing input}}
  %query = "sql.unparsed"(%str) : (memref<?xi8>) -> !sql.expr
  return %query : !sql.expr
}

// -----

func.func @order() -> !sql.expr {
  %str = "sql.str_constant"() <{input = "SELECT a FROM t ORDER BY a"}> : () -> memref<?xi8>
  // expected-error @+1 {{unsupported clause or trailing input}}
  %query = "sql.unparsed"(%str) : (memref<?xi8>) -> !sql.expr
  return %query : !sql.expr
}

// -----

func.func @not_a_predicate() -> !sql.expr {
  %str = "sql.str_constant"() <{input = "SELECT a FROM t WHERE a"}> : () -> memref<?xi8>
  // expected-error @+1 {{WHERE requires a predicate}}
  %query = "sql.unparsed"(%str) : (memref<?xi8>) -> !sql.expr
  return %query : !sql.expr
}

// -----

func.func @missing_parenthesis() -> !sql.expr {
  %str = "sql.str_constant"() <{input = "SELECT a FROM t WHERE (a=1"}> : () -> memref<?xi8>
  // expected-error @+1 {{after expression}}
  %query = "sql.unparsed"(%str) : (memref<?xi8>) -> !sql.expr
  return %query : !sql.expr
}

// -----

func.func @negative_limit() -> !sql.expr {
  %str = "sql.str_constant"() <{input = "SELECT a LIMIT -1"}> : () -> memref<?xi8>
  // expected-error @+1 {{LIMIT requires a nonnegative signed 64-bit integer or ALL}}
  %query = "sql.unparsed"(%str) : (memref<?xi8>) -> !sql.expr
  return %query : !sql.expr
}

// -----

func.func @limit_overflow() -> !sql.expr {
  %str = "sql.str_constant"() <{input = "SELECT a LIMIT 9223372036854775808"}> : () -> memref<?xi8>
  // expected-error @+1 {{LIMIT requires a nonnegative signed 64-bit integer or ALL}}
  %query = "sql.unparsed"(%str) : (memref<?xi8>) -> !sql.expr
  return %query : !sql.expr
}

// -----

func.func @integer_overflow() -> !sql.expr {
  %str = "sql.str_constant"() <{input = "SELECT 9223372036854775808"}> : () -> memref<?xi8>
  // expected-error @+1 {{integer literal is outside the signed 64-bit range}}
  %query = "sql.unparsed"(%str) : (memref<?xi8>) -> !sql.expr
  return %query : !sql.expr
}

// -----

func.func @insert() -> !sql.expr {
  %str = "sql.str_constant"() <{input = "INSERT INTO t VALUES (1)"}> : () -> memref<?xi8>
  // expected-error @+1 {{only SELECT statements are supported}}
  %query = "sql.unparsed"(%str) : (memref<?xi8>) -> !sql.expr
  return %query : !sql.expr
}

// -----

func.func @string_literal() -> !sql.expr {
  %str = "sql.str_constant"() <{input = "SELECT 'text'"}> : () -> memref<?xi8>
  // expected-error @+1 {{unsupported character or literal}}
  %query = "sql.unparsed"(%str) : (memref<?xi8>) -> !sql.expr
  return %query : !sql.expr
}

// -----

func.func @unterminated_comment() -> !sql.expr {
  %str = "sql.str_constant"() <{input = "SELECT 1 /* comment"}> : () -> memref<?xi8>
  // expected-error @+1 {{unterminated comment}}
  %query = "sql.unparsed"(%str) : (memref<?xi8>) -> !sql.expr
  return %query : !sql.expr
}

// -----

func.func @nested_comment() -> !sql.expr {
  %str = "sql.str_constant"() <{input = "SELECT /* outer /* inner */ end */ 1"}> : () -> memref<?xi8>
  // expected-error @+1 {{nested comments are not supported}}
  %query = "sql.unparsed"(%str) : (memref<?xi8>) -> !sql.expr
  return %query : !sql.expr
}

// -----

func.func @boolean_projection() -> !sql.expr {
  %str = "sql.str_constant"() <{input = "SELECT TRUE"}> : () -> memref<?xi8>
  // expected-error @+1 {{predicate projections are not supported}}
  %query = "sql.unparsed"(%str) : (memref<?xi8>) -> !sql.expr
  return %query : !sql.expr
}

// -----

func.func @bad_boolean() -> !sql.expr {
  %str = "sql.str_constant"() <{input = "SELECT 1 WHERE a AND b"}> : () -> memref<?xi8>
  // expected-error @+1 {{AND/OR require predicates}}
  %query = "sql.unparsed"(%str) : (memref<?xi8>) -> !sql.expr
  return %query : !sql.expr
}

// -----

func.func @function() -> !sql.expr {
  %str = "sql.str_constant"() <{input = "SELECT count(a) FROM t"}> : () -> memref<?xi8>
  // expected-error @+1 {{unsupported clause or trailing input}}
  %query = "sql.unparsed"(%str) : (memref<?xi8>) -> !sql.expr
  return %query : !sql.expr
}

// -----

func.func @case_expression() -> !sql.expr {
  %str = "sql.str_constant"() <{input = "SELECT CASE WHEN x=1 THEN 2 END FROM t"}> : () -> memref<?xi8>
  // expected-error @+1 {{expected an identifier}}
  %query = "sql.unparsed"(%str) : (memref<?xi8>) -> !sql.expr
  return %query : !sql.expr
}

// -----

func.func @unary_expression() -> !sql.expr {
  %str = "sql.str_constant"() <{input = "SELECT -a FROM t"}> : () -> memref<?xi8>
  // expected-error @+1 {{unary '-' is supported only on integer literals}}
  %query = "sql.unparsed"(%str) : (memref<?xi8>) -> !sql.expr
  return %query : !sql.expr
}

// -----

func.func @float_literal() -> !sql.expr {
  %str = "sql.str_constant"() <{input = "SELECT 1.5"}> : () -> memref<?xi8>
  // expected-error @+1 {{unsupported clause or trailing input}}
  %query = "sql.unparsed"(%str) : (memref<?xi8>) -> !sql.expr
  return %query : !sql.expr
}

// -----

func.func @offset() -> !sql.expr {
  %str = "sql.str_constant"() <{input = "SELECT a FROM t LIMIT 1 OFFSET 1"}> : () -> memref<?xi8>
  // expected-error @+1 {{unsupported clause or trailing input}}
  %query = "sql.unparsed"(%str) : (memref<?xi8>) -> !sql.expr
  return %query : !sql.expr
}
