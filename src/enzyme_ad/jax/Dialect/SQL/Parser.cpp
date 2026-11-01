//===- Parser.cpp - SQL text to SQL dialect ------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Adapted from Polygeist's dependency-free Parser.cpp. Parse a deliberately
// bounded SELECT grammar; never silently discard an unsupported SQL clause.

#include "src/enzyme_ad/jax/Dialect/SQL/Parser.h"
#include "src/enzyme_ad/jax/Dialect/SQL/Dialect.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include <cstdint>
#include <string>

using namespace mlir;
using namespace mlir::sql;

namespace {
enum class Kind { End, Word, Quoted, Integer, Symbol };
struct Token {
  Kind kind;
  StringRef text;
  size_t offset;
};

class SQLParser {
public:
  SQLParser(Location loc, OpBuilder &builder, StringRef input)
      : loc(loc), builder(builder), input(input) {}

  FailureOr<Value> parse() {
    if (failed(lex()))
      return failure();
    if (!consume("SELECT"))
      return error("only SELECT statements are supported");
    if (at("DISTINCT") || at("ALL"))
      return error("SELECT modifiers are not supported");

    SmallVector<Value> columns;
    do {
      if (consume("*")) {
        columns.push_back(builder.create<AllColumnsOp>(loc, exprType()));
      } else {
        auto column = expression();
        if (failed(column))
          return failure();
        if (!isa<ExprType>(column->getType()))
          return error("predicate projections are not supported");
        columns.push_back(*column);
      }
    } while (consume(","));

    std::string table;
    if (consume("FROM")) {
      auto name = identifier();
      if (failed(name))
        return failure();
      table = *name;
    }
    Value tableValue = builder.create<TableOp>(loc, exprType(), table);

    Value predicate;
    if (consume("WHERE")) {
      auto parsed = expression();
      if (failed(parsed))
        return failure();
      if (!isa<BoolType>(parsed->getType()))
        return error("WHERE requires a predicate");
      predicate = *parsed;
    } else {
      predicate = builder.create<ConstantBoolOp>(loc, boolType());
    }
    Value where = builder.create<WhereOp>(loc, exprType(), predicate);

    int64_t limit = -1;
    if (consume("LIMIT") && !consume("ALL")) {
      if (peek().kind != Kind::Integer || peek().text.getAsInteger(10, limit))
        return error(
            "LIMIT requires a nonnegative signed 64-bit integer or ALL");
      ++position;
    }
    consume(";");
    if (peek().kind != Kind::End)
      return error("unsupported clause or trailing input");
    return builder.create<SelectOp>(loc, exprType(), columns, tableValue, where,
                                    limit).getResult();
  }

private:
  Location loc;
  OpBuilder &builder;
  StringRef input;
  SmallVector<Token> tokens;
  size_t position = 0;
  unsigned depth = 0;

  ExprType exprType() { return ExprType::get(builder.getContext()); }
  BoolType boolType() { return BoolType::get(builder.getContext()); }
  const Token &peek() const { return tokens[position]; }
  bool at(StringRef text) const {
    return peek().kind != Kind::Quoted && peek().text.equals_insensitive(text);
  }
  bool consume(StringRef text) {
    if (!at(text))
      return false;
    ++position;
    return true;
  }
  LogicalResult diagnose(size_t offset, StringRef message) {
    emitError(loc) << "SQL parse error at byte " << offset << ": " << message;
    return failure();
  }
  FailureOr<Value> error(StringRef message) {
    (void)diagnose(peek().offset, message);
    return failure();
  }

  LogicalResult lex() {
    size_t i = 0;
    while (i < input.size()) {
      if (llvm::isSpace(input[i])) {
        ++i;
        continue;
      }
      if (input.substr(i).starts_with("--")) {
        while (i < input.size() && input[i] != '\n')
          ++i;
        continue;
      }
      if (input.substr(i).starts_with("/*")) {
        size_t end = input.find("*/", i + 2);
        if (end == StringRef::npos)
          return diagnose(i, "unterminated comment");
        // PostgreSQL allows nesting; reject it rather than misparse it.
        if (input.slice(i + 2, end).contains("/*"))
          return diagnose(i, "nested comments are not supported");
        i = end + 2;
        continue;
      }
      size_t start = i++;
      Kind kind = Kind::Symbol;
      if (llvm::isAlpha(input[start]) || input[start] == '_') {
        kind = Kind::Word;
        while (i < input.size() &&
               (llvm::isAlnum(input[i]) || input[i] == '_' || input[i] == '$'))
          ++i;
      } else if (llvm::isDigit(input[start])) {
        kind = Kind::Integer;
        while (i < input.size() && llvm::isDigit(input[i]))
          ++i;
      } else if (input[start] == '"') {
        kind = Kind::Quoted;
        bool closed = false;
        while (i < input.size()) {
          if (input[i++] != '"')
            continue;
          if (i < input.size() && input[i] == '"') {
            ++i;
            continue;
          }
          closed = true;
          break;
        }
        if (!closed)
          return diagnose(start, "unterminated quoted identifier");
        if (i == start + 2)
          return diagnose(start, "empty quoted identifier");
      } else {
        if (!StringRef(",.();+-*/%=<>!").contains(input[start]))
          return diagnose(start, "unsupported character or literal");
        if (i < input.size() &&
            (input.slice(start, i + 1) == "<=" ||
             input.slice(start, i + 1) == ">=" ||
             input.slice(start, i + 1) == "!=" ||
             input.slice(start, i + 1) == "<>"))
          ++i;
      }
      tokens.push_back({kind, input.slice(start, i), start});
    }
    tokens.push_back({Kind::End, {}, input.size()});
    return success();
  }

  bool reserved() const {
    static const char *words[] = {
        "SELECT", "FROM", "WHERE", "LIMIT", "ALL", "DISTINCT", "AS", "JOIN",
        "LEFT", "RIGHT", "INNER", "OUTER", "CROSS", "ON", "GROUP", "HAVING",
        "ORDER", "BY", "OFFSET", "UNION", "INTERSECT", "EXCEPT", "AND", "OR",
        "NOT", "NULL", "IS", "IN", "LIKE", "BETWEEN", "ASC", "DESC", "FOR",
        "FETCH", "WINDOW", "WITH", "INSERT", "UPDATE", "DELETE", "CASE",
        "WHEN", "THEN", "ELSE", "END", "CAST", "EXISTS", "TRUE", "FALSE",
        "CURRENT_DATE", "CURRENT_TIME", "CURRENT_TIMESTAMP", "CURRENT_USER",
        "SESSION_USER", "USER", "DEFAULT", "VALUES", "INTO", "LATERAL",
        "ONLY", "NATURAL", "USING", "ARRAY", "ROW"};
    for (const char *word : words)
      if (at(word))
        return true;
    return false;
  }

  FailureOr<std::string> identifier() {
    std::string name;
    do {
      if ((peek().kind != Kind::Word && peek().kind != Kind::Quoted) ||
          reserved()) {
        (void)diagnose(peek().offset, "expected an identifier");
        return failure();
      }
      if (!name.empty())
        name += '.';
      name += peek().text.str();
      ++position;
    } while (consume("."));
    return name;
  }

  int precedence() const {
    if (at("OR"))
      return 1;
    if (at("AND"))
      return 2;
    if (at("=") || at("!=") || at("<>") || at("<") || at("<=") ||
        at(">") || at(">="))
      return 3;
    if (at("+") || at("-"))
      return 4;
    if (at("*") || at("/") || at("%"))
      return 5;
    return 0;
  }

  FailureOr<Value> expression(int minimum = 1) {
    if (++depth > 128) {
      --depth;
      return error("expression nesting limit exceeded");
    }
    auto result = expressionImpl(minimum);
    --depth;
    return result;
  }

  FailureOr<Value> expressionImpl(int minimum) {
    auto lhs = primary();
    if (failed(lhs))
      return failure();
    while (precedence() >= minimum) {
      int level = precedence();
      std::string op = peek().text.upper();
      ++position;
      auto rhs = expression(level + 1);
      if (failed(rhs))
        return failure();
      if (level <= 2) {
        if (!isa<BoolType>(lhs->getType()) || !isa<BoolType>(rhs->getType()))
          return error("AND/OR require predicates");
        if (level == 1)
          lhs = builder.create<OrOp>(loc, boolType(), *lhs, *rhs).getResult();
        else
          lhs = builder.create<AndOp>(loc, boolType(), *lhs, *rhs).getResult();
      } else {
        if (!isa<ExprType>(lhs->getType()) || !isa<ExprType>(rhs->getType()))
          return error("arithmetic/comparison requires SQL expressions");
        if (level == 3)
          lhs = builder.create<BoolArithOp>(loc, boolType(), *lhs, *rhs, op)
                    .getResult();
        else
          lhs = builder.create<ArithOp>(loc, exprType(), *lhs, *rhs, op)
                    .getResult();
      }
    }
    return lhs;
  }

  FailureOr<Value> primary() {
    if (consume("(")) {
      auto value = expression();
      if (failed(value))
        return failure();
      if (!consume(")"))
        return error("expected ')' after expression");
      return value;
    }
    bool negative = consume("-");
    if (peek().kind == Kind::Integer) {
      std::string text = (negative ? "-" : "") + peek().text.str();
      int64_t value;
      if (StringRef(text).getAsInteger(10, value))
        return error("integer literal is outside the signed 64-bit range");
      ++position;
      return builder.create<IntOp>(loc, exprType(), text).getResult();
    }
    if (negative)
      return error("unary '-' is supported only on integer literals");
    if (at("TRUE") || at("FALSE")) {
      std::string text = peek().text.upper();
      ++position;
      return builder.create<CalcBoolOp>(loc, boolType(), text).getResult();
    }
    auto name = identifier();
    if (failed(name))
      return failure();
    return builder.create<ColumnOp>(loc, exprType(), *name).getResult();
  }
};
} // namespace

FailureOr<Value> mlir::sql::parseSQL(Location loc, OpBuilder &builder,
                                    StringRef query) {
  if (!builder.getInsertionBlock()) {
    emitError(loc) << "SQL parser requires an insertion point";
    return failure();
  }
  Block staging;
  OpBuilder scratch(builder.getContext());
  scratch.setInsertionPointToEnd(&staging);
  auto result = SQLParser(loc, scratch, query).parse();
  if (failed(result))
    return failure();
  builder.getInsertionBlock()->getOperations().splice(
      builder.getInsertionPoint(), staging.getOperations());
  return result;
}
