//===----------------------------------------------------------------------===//
//
// Token, AST and parser definitions for the tessera optimization rewrite rules
// defined by the user.
//
// These live in a header rather than inside ParseOptimizationRules.cpp because
// two passes need them. parse-optimization-rules turns a rule into a PDL
// pattern, and tessera-apply-pdl re-parses the rule string carried on the
// generated pattern in order to build the guarded rewrite. The two run far
// apart in the pipeline -- and, in lit tests, in separate enzymexlamlir-opt
// invocations -- so the rule text is the only thing passed between them.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_AD_JAX_PASSES_TESSERA_RULEAST_H
#define ENZYME_AD_JAX_PASSES_TESSERA_RULEAST_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace mlir {
namespace enzyme {
namespace tessera {

//===----------------------------------------------------------------------===//
// Tokens
//===----------------------------------------------------------------------===//

enum class TokenType {
  Ident,
  Integer,
  Float,
  LParen,
  RParen,
  Dot,
  Comma,
  Arrow,
  // Condition operators.
  Bang,
  AndAnd,
  OrOr,
  EqEq,
  BangEq,
  Less,
  LessEq,
  Greater,
  GreaterEq,
  End,
  Error
};

struct Token {
  TokenType type;
  std::string value;
};

struct Lexer {
  std::string input;
  size_t pos = 0;

  char peek() { return pos < input.size() ? input[pos] : '\0'; }
  char peekNext() { return pos + 1 < input.size() ? input[pos + 1] : '\0'; }
  char advance() { return input[pos++]; }

  Token nextToken();
};

//===----------------------------------------------------------------------===//
// Expressions
//===----------------------------------------------------------------------===//

struct Var {
  std::string name;
};

struct IntLit {
  int64_t value;
};

struct FloatLit {
  double value;
};

struct Expr;

struct Call {
  std::string dialect, opname;
  std::vector<Expr> args;
};

struct Expr {
  std::variant<Var, IntLit, FloatLit, Call> data;

  Expr() = default; // default constructor
  Expr(Var v) : data(v) {}
  Expr(IntLit n) : data(n) {}
  Expr(FloatLit n) : data(n) {}
  Expr(Call c) : data(std::move(c)) {} // move because Call has a vector
};

//===----------------------------------------------------------------------===//
// Conditions
//===----------------------------------------------------------------------===//

enum class CmpOp { Eq, Ne, Lt, Le, Gt, Ge };

/// Spelling of a comparison operator, for diagnostics and for round-tripping a
/// condition back into its textual form.
llvm::StringRef getCmpOpSpelling(CmpOp op);

struct Cond;

/// A named predicate over matched values, e.g. `symmetric(x)`.
struct Pred {
  std::string name;
  std::vector<Expr> args;
};

/// A relational test between two expressions, e.g. `n > 64`.
struct Compare {
  CmpOp op;
  Expr lhs, rhs;
};

// The boolean connectives recurse through Cond, so they hold it indirectly.
// This makes Cond move-only, which is all the parser and the consumers need.
struct NotCond {
  std::unique_ptr<Cond> operand;
};

struct AndCond {
  std::unique_ptr<Cond> lhs, rhs;
};

struct OrCond {
  std::unique_ptr<Cond> lhs, rhs;
};

struct Cond {
  std::variant<Pred, Compare, NotCond, AndCond, OrCond> data;

  Cond() = default;
  Cond(Pred p) : data(std::move(p)) {}
  Cond(Compare c) : data(std::move(c)) {}
  Cond(NotCond c) : data(std::move(c)) {}
  Cond(AndCond c) : data(std::move(c)) {}
  Cond(OrCond c) : data(std::move(c)) {}
};

/// Wrap `c` in a heap allocation, for storing into a connective.
inline std::unique_ptr<Cond> box(Cond c) {
  return std::make_unique<Cond>(std::move(c));
}

//===----------------------------------------------------------------------===//
// Rules
//===----------------------------------------------------------------------===//

/// A rewrite rule: `[ 'if' cond ',' ] lhs '->' rhs`. An absent condition means
/// the rewrite is unconditional.
struct Rule {
  std::optional<Cond> cond;
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

  void advance();

  /// Report a parse error against the rule's location, and mark the parse as
  /// failed. Once a diagnostic has been reported this returns an inert one, so
  /// only the first error a rule produces is shown: everything after it is
  /// fallout from a token stream the parser has already lost track of.
  InFlightDiagnostic error();

  /// Return the token after `current` without consuming anything. Used to tell
  /// a predicate call apart from the left operand of a comparison, both of
  /// which start with an identifier.
  Token peekAhead();

  std::optional<Call> parseCall(std::string dialect_name);
  std::optional<Expr> parseExpr();
  std::optional<Cond> parseCond();
  std::optional<Cond> parseConj();
  std::optional<Cond> parseUnary();
  std::optional<Rule> parseRule();
};

/// Parse a bare condition, as carried on a tessera.guard, rather than a whole
/// rule. Returns nullopt and emits a diagnostic against `loc` if the text does
/// not parse or has trailing junk.
std::optional<Cond> parseConditionText(llvm::StringRef text, Location loc);

/// Render an expression or a condition back to its textual form. A rendered
/// condition is what a tessera.guard carries, so it has to parse back to an
/// equivalent tree: parentheses are inserted wherever precedence would
/// otherwise regroup the operands.
std::string renderExpr(const Expr &expr);
std::string renderCond(const Cond &cond);

//===----------------------------------------------------------------------===//
// Literal materialization
//===----------------------------------------------------------------------===//

/// Pick the narrowest standard integer width that can hold a literal from a
/// rule annotation. Literals are parsed as int64_t, so anything that does not
/// round-trip through int32_t needs an i64 attribute; asking for an i32
/// attribute in that case would silently truncate the value.
IntegerAttr getIntegerAttrForLiteral(OpBuilder &builder, int64_t value);

/// Same idea for float literals: f32 when the value survives the round trip
/// through it, otherwise f64. Values like 0.5 are exact in both, so they
/// narrow to f32; floats like pi and e are not, so they stay f64.
FloatAttr getFloatAttrForLiteral(OpBuilder &builder, double value);

} // namespace tessera
} // namespace enzyme
} // namespace mlir

#endif // ENZYME_AD_JAX_PASSES_TESSERA_RULEAST_H
