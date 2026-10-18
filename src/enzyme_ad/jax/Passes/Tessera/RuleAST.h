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
  String,
  Global,
  LParen,
  RParen,
  Dot,
  Comma,
  Semicolon,
  Arrow,
  // Arithmetic operators.
  Plus,
  Minus,
  Star,
  Slash,
  Percent,
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

/// A string literal, `'cg'`. Only allowed on the right-hand side of a rule,
/// where it is passed as a pointer to a NUL-terminated constant, as C passes
/// one: it is how a library that names things by string, as PETSc names its
/// solvers, is told which one to use.
struct StrLit {
  std::string value;
};

/// The address of a global variable, `@_ZN4mfem8IntRulesE`, named by its
/// symbol. Only allowed on the right-hand side of a rule, where it passes a
/// library's own global object, as MFEM's rules live in the global
/// `mfem::IntRules`. The clang plugin writes it for a C++ name the rule uses,
/// so a rule says `mfem::IntRules` and never spells a mangled name.
struct GlobalRef {
  std::string symbol;
};

struct Expr;

/// An arithmetic operator. Integer division and remainder are signed, as C's
/// are on the ints a library takes.
enum class ArithOp { Add, Sub, Mul, Div, Rem };

/// Spelling of an arithmetic operator, for diagnostics and rendering.
llvm::StringRef getArithOpSpelling(ArithOp op);

/// `lhs op rhs`, computing a value from others, as `2 * p + 1` computes a
/// quadrature order from a polynomial order. Allowed on the right-hand side of
/// a rule and in the operands of a comparison; there is nothing for it to
/// match on the left, or to test in a predicate. The operands are shared, not
/// owned, so that an Expr stays copyable: the tree is never changed once
/// parsed.
struct Arith {
  ArithOp op;
  std::shared_ptr<const Expr> lhs, rhs;
};

/// A call to a library op, `blas.gemm(...)`, or, with no dialect, to one of
/// the functions the right-hand side can compute an argument with, `log2(n)`.
struct Call {
  std::string dialect, opname;
  std::vector<Expr> args;

  bool isBuiltin() const { return dialect.empty(); }
};

struct Expr {
  std::variant<Var, IntLit, FloatLit, StrLit, GlobalRef, Call, Arith> data;

  Expr() = default; // default constructor
  Expr(Var v) : data(v) {}
  Expr(IntLit n) : data(n) {}
  Expr(FloatLit n) : data(n) {}
  Expr(StrLit s) : data(std::move(s)) {}
  Expr(GlobalRef g) : data(std::move(g)) {}
  Expr(Call c) : data(std::move(c)) {} // move because Call has a vector
  Expr(Arith a) : data(std::move(a)) {}
};

/// Whether `expr` contains arithmetic anywhere, for the places it is not
/// allowed.
bool containsArith(const Expr &expr);

/// Whether `expr` is made of literals and arithmetic only, so that its type
/// is whatever the context gives it.
bool isLiteralOnly(const Expr &expr);

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

/// A rewrite rule: `[ 'if' cond ',' ] lhs '->' rhs { ';' rhs }`. An absent
/// condition means the rewrite is unconditional.
///
/// The right-hand side runs in order, like C statements, and the last
/// expression replaces the matched call. Every expression before it is a call
/// made for its effect, whose results are discarded: `f(x) -> g(x); f(x)` adds
/// a call to g ahead of f.
struct Rule {
  std::optional<Cond> cond;
  Expr lhs;
  std::vector<Expr> rhs;

  /// The expression whose value replaces the matched call.
  const Expr &result() const { return rhs.back(); }
};

struct Parser {
  Lexer lexer;
  Token current;
  Location loc;
  bool failed = false;
  /// String literals, globals and built-in functions are only allowed on the
  /// right-hand side of a rule.
  bool allowStrings = false;

  Parser(std::string input, Location location) : lexer{input}, loc{location} {
    advance();
  }

  void advance();

  /// Report why the rule does not parse against its location, and mark the
  /// parse as failed. This is a warning: a rule that does not parse is left
  /// out, not a reason to fail the compile. Once a diagnostic has been
  /// reported this returns an inert one, so
  /// only the first error a rule produces is shown: everything after it is
  /// fallout from a token stream the parser has already lost track of.
  InFlightDiagnostic error();

  /// Return the token after `current` without consuming anything. Used to tell
  /// a predicate call apart from the left operand of a comparison, both of
  /// which start with an identifier.
  Token peekAhead();

  std::optional<Call> parseCall(std::string dialect_name);
  /// expr ::= term { ('+' | '-') term }
  std::optional<Expr> parseExpr();
  /// term ::= primary { ('*' | '/' | '%') primary }
  std::optional<Expr> parseTerm();
  /// primary ::= call | var | int | float | string | builtin | '(' expr ')'
  std::optional<Expr> parsePrimary();
  /// With `current` a '(' in a condition: whether it opens an arithmetic
  /// expression, `(p + 1) * 2 == q`, rather than a parenthesized condition.
  /// It does exactly when an arithmetic or comparison operator follows the
  /// matching ')', which can never follow a condition.
  bool parenOpensExpr();
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
