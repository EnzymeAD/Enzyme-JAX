//===----------------------------------------------------------------------===//
//
// This file implements a custom lexer and parser to parse the tessera
// optimization rewrite rules defined by the user and then creates a PDL
// pattern from each rule.
//
//===----------------------------------------------------------------------===//

#include "Passes/Passes.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDL/IR/PDLOps.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "src/enzyme_ad/jax/Dialect/Tessera/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Tessera/Passes.h"
#include <optional>
#include <utility>
#include <variant>

template <class... Ts> struct overloaded : Ts... {
  using Ts::operator()...;
};

template <class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

namespace mlir {
namespace enzyme {
namespace tessera {
#define GEN_PASS_DEF_PARSEOPTIMIZATIONRULESPASS
#include "src/enzyme_ad/jax/Passes/Tessera/Passes.h.inc"
} // namespace tessera
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::enzyme::tessera;

namespace {

enum class TokenType {
  Ident,
  Number,
  LParen,
  RParen,
  Dot,
  Comma,
  Arrow,
  End,
  Error
};

struct Token {
  TokenType type;
  std::string value;
};

bool isAlpha(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

bool isNum(char c) { return c >= '0' && c <= '9'; }

bool isAlphaNum(char c) { return isAlpha(c) || isNum(c); }

bool isWhitespace(char c) { return c == ' ' || c == '\t' || c == '\n'; }

struct Lexer {
  std::string input;
  size_t pos = 0;

  char peek() { return pos < input.size() ? input[pos] : '\0'; }
  char advance() { return input[pos++]; }

  Token nextToken() {
    if (peek() == '\0') {
      return Token{TokenType::End, ""};
    }

    while (isWhitespace(peek()))
      advance();

    if (isAlpha(peek()) || peek() == '_') {
      std::string s;
      while (isAlphaNum(peek()) || peek() == '_') {
        s += advance();
      }
      return Token{TokenType::Ident, s};
    }

    if (isNum(peek())) {
      std::string num;
      while (isNum(peek())) {
        num += advance();
      }
      return Token{TokenType::Number, num};
    }

    if (peek() == '(') {
      advance();
      return Token{TokenType::LParen, ""};
    }

    if (peek() == ')') {
      advance();
      return Token{TokenType::RParen, ""};
    }

    if (peek() == '.') {
      advance();
      return Token{TokenType::Dot, ""};
    }

    if (peek() == ',') {
      advance();
      return Token{TokenType::Comma, ""};
    }

    if (peek() == '-' && input[pos + 1] == '>') {
      advance();
      advance();
      return Token{TokenType::Arrow, ""};
    }

    return Token{TokenType::Error, std::string(1, peek())};
  }
};

struct Var {
  std::string name;
};

struct Num {
  int value;
};

struct Expr;

struct Call {
  std::string dialect, opname;
  std::vector<Expr> args;
};

struct Expr {
  std::variant<Var, Num, Call> data;

  Expr() = default; // default constructor
  Expr(Var v) : data(v) {}
  Expr(Num n) : data(n) {}
  Expr(Call c) : data(std::move(c)) {} // move because Call has a vector
};

struct Rule {
  Expr lhs;
  Expr rhs;
};

struct Parser {
  Lexer lexer;
  Token current;
  Location loc;
  bool failed = false;

  Parser(std::string input, Location location) : lexer{input}, loc{location} {
    advance();
  }

  void advance() {
    current = lexer.nextToken();
    if (current.type == TokenType::Error) {
      emitError(loc, "unrecognized character '")
          << current.value << "' in optimization rule";
      failed = true;
    }
  }

  std::optional<Call> parseCall(std::string dialect_name);
  std::optional<Expr> parseExpr();
  std::optional<Rule> parseRule();
};

std::optional<Call> Parser::parseCall(std::string dialect_name) {
  if (current.type != TokenType::Dot) {
    emitError(loc, "expected '.' in optimization rule, got '")
        << current.value << "'";
    return std::nullopt;
  }
  advance();
  if (current.type != TokenType::Ident) {
    emitError(loc, "expected identifier in optimization rule, got '")
        << current.value << "'";
    return std::nullopt;
  }
  std::string op_name = current.value;
  advance();
  if (current.type != TokenType::LParen) {
    emitError(loc, "expected '(' in optimization rule, got '")
        << current.value << "'";
    return std::nullopt;
  }
  std::vector<Expr> args;
  advance();
  while (current.type != TokenType::RParen) {
    auto expr = parseExpr();
    if (!expr)
      return std::nullopt;
    args.push_back(std::move(*expr));
    if (current.type == TokenType::Comma)
      advance();
  }
  advance(); // consume ')'
  return Call{dialect_name, op_name, std::move(args)};
}

std::optional<Expr> Parser::parseExpr() {
  if (current.type == TokenType::Ident) {
    std::string s = current.value;
    advance();
    if (current.type == TokenType::Dot) {
      auto call = parseCall(s);
      if (!call)
        return std::nullopt;
      return Expr(std::move(*call));
    }
    return Var{s};
  }
  if (current.type == TokenType::Number) {
    int n = std::stoi(current.value);
    advance();
    return Expr(Num{n});
  }
  emitError(loc, "invalid optimization rule expression");
  return std::nullopt;
}

std::optional<Rule> Parser::parseRule() {
  auto lhs = parseExpr();
  if (!lhs)
    return std::nullopt;
  if (current.type != TokenType::Arrow) {
    emitError(loc, "expected '->' in optimization rule, got '")
        << current.value << "'";
    return std::nullopt;
  }
  advance();
  auto rhs = parseExpr();
  if (!rhs)
    return std::nullopt;
  return Rule{std::move(*lhs), std::move(*rhs)};
}

std::pair<mlir::Value, mlir::Value>
emitMatchPDL(const Expr &expr, OpBuilder &builder, Location loc,
             llvm::StringMap<mlir::Value> &boundVars) {
  return std::visit(
      overloaded{
          [&](const Var &v) -> std::pair<mlir::Value, mlir::Value> {
            if (!boundVars.count(v.name)) {
              boundVars[v.name] = pdl::OperandOp::create(
                  builder, loc, builder.getType<pdl::ValueType>(),
                  /*type=*/mlir::Value());
            }
            return {boundVars[v.name], mlir::Value()};
          },
          [&](const Num &n) -> std::pair<mlir::Value, mlir::Value> {
            auto attrVal = pdl::AttributeOp::create(
                builder, loc, builder.getI64IntegerAttr(n.value));
            auto constOp = pdl::OperationOp::create(
                builder, loc, "arith.constant",
                /*operands=*/ValueRange{},
                /*attrNames=*/ArrayRef<StringRef>{"value"},
                /*attrs=*/ValueRange{attrVal}, /*types=*/ValueRange{});
            return {pdl::ResultOp::create(
                        builder, loc, builder.getType<pdl::ValueType>(),
                        constOp, builder.getI32IntegerAttr(0)),
                    mlir::Value()};
          },
          [&](const Call &c) -> std::pair<mlir::Value, mlir::Value> {
            SmallVector<mlir::Value> argValues;
            for (int i = 0; i < c.args.size(); i++) {
              auto argPDL = emitMatchPDL(c.args[i], builder, loc, boundVars);
              argValues.push_back(argPDL.first);
            }
            auto calleeAttr = pdl::AttributeOp::create(
                builder, loc,
                FlatSymbolRefAttr::get(builder.getContext(),
                                       c.dialect + "." + c.opname));
            auto resultTypeOp = pdl::TypeOp::create(
                builder, loc, builder.getType<pdl::TypeType>(),
                /*type=*/TypeAttr());
            auto callOp = pdl::OperationOp::create(
                builder, loc, "tessera.call",
                /*operands=*/argValues,
                /*attrNames=*/ArrayRef<StringRef>{"callee"},
                /*attrs=*/ValueRange{calleeAttr},
                /*types=*/ValueRange{resultTypeOp});
            return {pdl::ResultOp::create(builder, loc,
                                          builder.getType<pdl::ValueType>(),
                                          callOp, builder.getI32IntegerAttr(0)),
                    callOp};
          },
      },
      expr.data);
}

std::pair<mlir::Value, mlir::Value>
emitRewritePDL(const Expr &expr, OpBuilder &builder, Location loc,
               llvm::StringMap<mlir::Value> &boundVars) {
  return std::visit(
      overloaded{
          [&](const Var &v) -> std::pair<mlir::Value, mlir::Value> {
            return {boundVars[v.name], mlir::Value()};
          },
          [&](const Num &n) -> std::pair<mlir::Value, mlir::Value> {
            auto attrVal = pdl::AttributeOp::create(
                builder, loc, builder.getI64IntegerAttr(n.value));
            auto constOp = pdl::OperationOp::create(
                builder, loc, "arith.constant",
                /*operands=*/ValueRange{},
                /*attrNames=*/ArrayRef<StringRef>{"value"},
                /*attrs=*/ValueRange{attrVal}, /*types=*/ValueRange{});
            return {pdl::ResultOp::create(
                        builder, loc, builder.getType<pdl::ValueType>(),
                        constOp, builder.getI32IntegerAttr(0)),
                    mlir::Value()};
          },
          [&](const Call &c) -> std::pair<mlir::Value, mlir::Value> {
            SmallVector<mlir::Value> argValues;
            for (int i = 0; i < c.args.size(); i++) {
              auto argPDL = emitRewritePDL(c.args[i], builder, loc, boundVars);
              argValues.push_back(argPDL.first);
            }
            auto calleeAttr = pdl::AttributeOp::create(
                builder, loc,
                FlatSymbolRefAttr::get(builder.getContext(),
                                       c.dialect + "." + c.opname));
            auto resultTypeOp = pdl::TypeOp::create(
                builder, loc, builder.getType<pdl::TypeType>(),
                /*type=*/TypeAttr());
            auto callOp = pdl::OperationOp::create(
                builder, loc, "tessera.call",
                /*operands=*/argValues,
                /*attrNames=*/ArrayRef<StringRef>{"callee"},
                /*attrs=*/ValueRange{calleeAttr},
                /*types=*/ValueRange{resultTypeOp});
            return {pdl::ResultOp::create(builder, loc,
                                          builder.getType<pdl::ValueType>(),
                                          callOp, builder.getI32IntegerAttr(0)),
                    callOp};
          },
      },
      expr.data);
}

struct ParseOptimizationRulesPass
    : public enzyme::tessera::impl::ParseOptimizationRulesPassBase<
          ParseOptimizationRulesPass> {
  using ParseOptimizationRulesPassBase::ParseOptimizationRulesPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();
    OpBuilder builder(ctx);
    builder.setInsertionPointToEnd(module.getBody());

    // Create nested module to store PDL patterns
    Location location = builder.getUnknownLoc();
    ModuleOp patternsModule =
        mlir::ModuleOp::create(builder, location, "patterns");

    // Find optimization ops and parse rewrite rules
    for (auto optimizations_op : module.getOps<tessera::OptimizationsOp>()) {
      for (auto optimization_op :
           optimizations_op.getBody().getOps<tessera::OptimizationOp>()) {
        Location loc = optimization_op.getLoc();
        Parser parser = Parser(optimization_op.getRule().str(), loc);
        auto rule = parser.parseRule();
        if (!rule) {
          signalPassFailure();
          llvm::errs() << "Pass failure\n";
          return;
        }

        // Create pdl.pattern op that will store PDL for parsed rewrite rule
        builder.setInsertionPointToStart(patternsModule.getBody());
        auto pattern = pdl::PatternOp::create(builder, loc, /*benefit=*/1,
                                              /*sym_name=*/nullptr);
        Block *patternBlock = builder.createBlock(&pattern.getBodyRegion());
        builder.setInsertionPointToStart(patternBlock);
        llvm::StringMap<mlir::Value> boundVars;

        // Emit PDL for the left hand side of the rewrite rule (the pattern to
        // match)
        auto root = emitMatchPDL(rule->lhs, builder, loc, boundVars);
        if (!root.second) {
          signalPassFailure();
          llvm::errs()
              << "Left hand side of optimization rule must be a call\n";
          return;
        }
        auto rewrite = pdl::RewriteOp::create(builder, loc, root.second,
                                              /*name=*/StringAttr(),
                                              /*externalArgs=*/ValueRange{});
        Block *rewriteBlock = builder.createBlock(&rewrite.getBodyRegion());
        builder.setInsertionPointToStart(rewriteBlock);

        // Emit PDL for the right hand side of the rewrite rule (the
        // replacement)
        auto replacement = emitRewritePDL(rule->rhs, builder, loc, boundVars);
        if (replacement.second) {
          pdl::ReplaceOp::create(builder, loc, root.second, replacement.second,
                                 ValueRange{});
        } else if (replacement.first) {
          pdl::ReplaceOp::create(builder, loc, root.second, mlir::Value(),
                                 ValueRange{replacement.first});
        } else {
          signalPassFailure();
          llvm::errs()
              << "Left hand side of optimization rule must be a call\n";
          return;
        }
      }
    }
  }
};
} // namespace
