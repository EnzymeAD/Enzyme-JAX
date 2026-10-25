//===----------------------------------------------------------------------===//
//
// Tokens, AST and parser for tessera's rewrite rules. In a header because two
// passes use them: parse-optimization-rules turns each rule into a PDL
// pattern, and tessera-apply-pdl re-parses the rule text carried on that
// pattern to build the rewrite.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_AD_JAX_PASSES_TESSERA_RULEAST_H
#define ENZYME_AD_JAX_PASSES_TESSERA_RULEAST_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/StringMap.h"
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

// A string literal, `'cg'`. Right-hand side only, passed as a pointer to a
// NUL-terminated constant, for libraries that pick things by name (such as
// PETSc solvers).
struct StrLit {
  std::string value;
};

// The address of a global, `@_ZN4mfem8IntRulesE`. Right-hand side only, for
// passing a library's global object. A rule can write the C++ name,
// `mfem::IntRules`, and lift-tessera-annotations swaps in the symbol.
struct GlobalRef {
  std::string symbol;
};

struct Expr;

// An arithmetic operator. Integer division and remainder are signed, as in C.
enum class ArithOp { Add, Sub, Mul, Div, Rem };

// Spelling of an arithmetic operator, for diagnostics and rendering.
llvm::StringRef getArithOpSpelling(ArithOp op);

// `lhs op rhs`, such as `2 * p + 1`. Allowed on the right-hand side and in
// comparisons, not in what a rule matches. Operands are shared_ptrs so Expr
// stays copyable; trees never change after parsing.
struct Arith {
  ArithOp op;
  std::shared_ptr<const Expr> lhs, rhs;
};

// A call to a library op, `blas.gemm(...)`, or, with no dialect, a builtin
// function such as `log2(n)`.
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

// Whether `expr` contains arithmetic anywhere.
bool containsArith(const Expr &expr);

// Whether `expr` is only literals and arithmetic, so its type comes from
// context.
bool isLiteralOnly(const Expr &expr);

//===----------------------------------------------------------------------===//
// Conditions
//===----------------------------------------------------------------------===//

enum class CmpOp { Eq, Ne, Lt, Le, Gt, Ge };

// Spelling of a comparison operator, for diagnostics and rendering.
llvm::StringRef getCmpOpSpelling(CmpOp op);

struct Cond;

// A named predicate over matched values, e.g. `symmetric(x)`.
struct Pred {
  std::string name;
  std::vector<Expr> args;
};

// A relational test between two expressions, e.g. `n > 64`.
struct Compare {
  CmpOp op;
  Expr lhs, rhs;
};

// Cond is recursive, so connectives hold it through unique_ptr. That makes
// Cond move-only, which is all that's needed.
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

// Wrap `c` in a heap allocation, for storing into a connective.
inline std::unique_ptr<Cond> box(Cond c) {
  return std::make_unique<Cond>(std::move(c));
}

//===----------------------------------------------------------------------===//
// Rules
//===----------------------------------------------------------------------===//

// A rewrite rule: `[ 'if' cond ',' ] lhs '->' rhs { ';' rhs }`. With no
// condition it always applies. The right-hand side runs in order; the last
// expression replaces the matched call, and earlier ones are calls made for
// their effects: `f(x) -> g(x); f(x)` adds a call to g before f.
struct Rule {
  std::optional<Cond> cond;
  Expr lhs;
  std::vector<Expr> rhs;

  // The expression whose value replaces the matched call.
  const Expr &result() const { return rhs.back(); }
};

struct Parser {
  Lexer lexer;
  Token current;
  Location loc;
  bool failed = false;
  // String literals, globals and built-in functions are only allowed on the
  // right-hand side of a rule.
  bool allowStrings = false;

  Parser(std::string input, Location location) : lexer{input}, loc{location} {
    advance();
  }

  void advance();

  // Warn that the rule doesn't parse, and mark the parse failed. A bad rule is
  // skipped, not a compile error. Only the first error is shown; later ones
  // are just fallout from it.
  InFlightDiagnostic error();

  // The token after `current`, without consuming it. Tells a predicate call
  // apart from a comparison's left operand.
  Token peekAhead();

  std::optional<Call> parseCall(std::string dialect_name);
  // expr ::= term { ('+' | '-') term }
  std::optional<Expr> parseExpr();
  // term ::= primary { ('*' | '/' | '%') primary }
  std::optional<Expr> parseTerm();
  // primary ::= call | var | int | float | string | builtin | '(' expr ')'
  std::optional<Expr> parsePrimary();
  // With `current` a '(' in a condition: whether it starts an arithmetic
  // expression, `(p + 1) * 2 == q`, rather than a parenthesized condition.
  // True when an arithmetic or comparison operator follows the matching ')'.
  bool parenOpensExpr();
  std::optional<Cond> parseCond();
  std::optional<Cond> parseConj();
  std::optional<Cond> parseUnary();
  std::optional<Rule> parseRule();
};

// Parse a bare condition, as a tessera.guard carries. Warns and returns
// nullopt if it doesn't parse.
std::optional<Cond> parseConditionText(llvm::StringRef text, Location loc);

// Render an expression or condition as text, with parentheses where needed so
// it parses back to the same tree (a tessera.guard stores its condition so).
std::string renderExpr(const Expr &expr);
std::string renderCond(const Cond &cond);

// `rule` with each name its left-hand side doesn't bind replaced by its entry
// in `values`, so a C name like `MAT_SPD` gets the value the frontend found.
// Values are in rule syntax: an integer, a 'string' or an @symbol. A rule
// that doesn't parse is returned unchanged.
std::string substituteRuleNames(llvm::StringRef rule,
                                const llvm::StringMap<std::string> &values,
                                Location loc);

//===----------------------------------------------------------------------===//
// Literal materialization
//===----------------------------------------------------------------------===//

// An i32 attribute for `value` if it fits, else i64, so nothing is truncated.
IntegerAttr getIntegerAttrForLiteral(OpBuilder &builder, int64_t value);

} // namespace tessera
} // namespace enzyme
} // namespace mlir

#endif // ENZYME_AD_JAX_PASSES_TESSERA_RULEAST_H
