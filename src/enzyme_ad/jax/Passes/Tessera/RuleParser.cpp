//===----------------------------------------------------------------------===//
//
// Lexer and recursive-descent parser for tessera optimization rewrite rules.
//
// Grammar:
//
//   rule  ::= [ 'if' cond ',' ] expr '->' expr
//   cond  ::= conj { '||' conj }
//   conj  ::= unary { '&&' unary }
//   unary ::= '!' unary | '(' cond ')' | pred
//   pred  ::= ident '(' [ expr { ',' expr } ] ')'
//           | expr relop expr
//   relop ::= '==' | '!=' | '<' | '<=' | '>' | '>='
//   expr  ::= call | var | int | float
//   call  ::= ident '.' ident '(' [ expr { ',' expr } ] ')'
//   var   ::= ident
//
// A rule with no condition is unconditional and rewrites exactly as before.
//
//===----------------------------------------------------------------------===//

#include "src/enzyme_ad/jax/Passes/Tessera/RuleAST.h"
#include "mlir/IR/Diagnostics.h"

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::enzyme::tessera;

namespace {

bool isAlpha(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

bool isDigit(char c) { return c >= '0' && c <= '9'; }

bool isAlphaNum(char c) { return isAlpha(c) || isDigit(c); }

bool isWhitespace(char c) { return c == ' ' || c == '\n' || c == '\t'; }

/// The relational operators, in the order the grammar lists them. Returns
/// nullopt for any token that does not open a comparison.
std::optional<CmpOp> tokenToCmpOp(TokenType type) {
  switch (type) {
  case TokenType::EqEq:
    return CmpOp::Eq;
  case TokenType::BangEq:
    return CmpOp::Ne;
  case TokenType::Less:
    return CmpOp::Lt;
  case TokenType::LessEq:
    return CmpOp::Le;
  case TokenType::Greater:
    return CmpOp::Gt;
  case TokenType::GreaterEq:
    return CmpOp::Ge;
  default:
    return std::nullopt;
  }
}

} // namespace

namespace mlir {
namespace enzyme {
namespace tessera {

llvm::StringRef getCmpOpSpelling(CmpOp op) {
  switch (op) {
  case CmpOp::Eq:
    return "==";
  case CmpOp::Ne:
    return "!=";
  case CmpOp::Lt:
    return "<";
  case CmpOp::Le:
    return "<=";
  case CmpOp::Gt:
    return ">";
  case CmpOp::Ge:
    return ">=";
  }
  return "<invalid>";
}

//===----------------------------------------------------------------------===//
// Lexer
//===----------------------------------------------------------------------===//

Token Lexer::nextToken() {
  // Whitespace is skipped before the end-of-input test, not after: checking
  // for '\0' first would mislex any rule with trailing whitespace.
  while (isWhitespace(peek()))
    advance();

  if (peek() == '\0')
    return Token{TokenType::End, ""};

  if (isAlpha(peek()) || peek() == '_') {
    std::string s;
    while (isAlphaNum(peek()) || peek() == '_') {
      s += advance();
    }
    return Token{TokenType::Ident, s};
  }

  // Two lookahead rules decide whether a leading character starts a number.
  // A '-' does only when a digit follows, otherwise it is the head of the
  // "->" arrow handled below. A '.' does only when a digit follows.
  if (isDigit(peek()) ||
      ((peek() == '-' || peek() == '.') && isDigit(peekNext()))) {
    std::string num;
    if (peek() == '-') {
      num += advance();
    }
    while (isDigit(peek())) {
      num += advance();
    }
    if (peek() == '.') {
      num += advance();
      while (isDigit(peek())) {
        num += advance();
      }
      return Token{TokenType::Float, num};
    }
    return Token{TokenType::Integer, num};
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

  if (peek() == '-' && peekNext() == '>') {
    advance();
    advance();
    return Token{TokenType::Arrow, ""};
  }

  // Condition operators. Each two-character form must be tested before the
  // one-character form that prefixes it, so that '!' does not swallow '!='.
  if (peek() == '!') {
    advance();
    if (peek() == '=') {
      advance();
      return Token{TokenType::BangEq, ""};
    }
    return Token{TokenType::Bang, ""};
  }

  if (peek() == '<') {
    advance();
    if (peek() == '=') {
      advance();
      return Token{TokenType::LessEq, ""};
    }
    return Token{TokenType::Less, ""};
  }

  if (peek() == '>') {
    advance();
    if (peek() == '=') {
      advance();
      return Token{TokenType::GreaterEq, ""};
    }
    return Token{TokenType::Greater, ""};
  }

  // '=', '&' and '|' are only ever valid doubled, so a lone one is an error
  // reported against that character rather than silently accepted.
  if (peek() == '=' && peekNext() == '=') {
    advance();
    advance();
    return Token{TokenType::EqEq, ""};
  }

  if (peek() == '&' && peekNext() == '&') {
    advance();
    advance();
    return Token{TokenType::AndAnd, ""};
  }

  if (peek() == '|' && peekNext() == '|') {
    advance();
    advance();
    return Token{TokenType::OrOr, ""};
  }

  return Token{TokenType::Error, std::string(1, peek())};
}

//===----------------------------------------------------------------------===//
// Parser
//===----------------------------------------------------------------------===//

InFlightDiagnostic Parser::error() {
  // Only the first error in a rule is real. After it the token stream is out
  // of sync, so every later complaint is fallout -- an inert diagnostic keeps
  // those from reaching the user, and keeps -verify-diagnostics tests from
  // having to spell out a cascade.
  if (failed)
    return InFlightDiagnostic();
  failed = true;
  return emitError(loc);
}

void Parser::advance() {
  current = lexer.nextToken();
  if (current.type == TokenType::Error) {
    error() << "unrecognized character '" << current.value
            << "' in optimization rule";
  }
}

Token Parser::peekAhead() {
  // The lexer is a plain {input, pos} pair, so a copy is a cheap save point.
  // This deliberately does not go through advance(): looking ahead must not
  // report an error for a token the parser may never consume.
  Lexer saved = lexer;
  Token token = lexer.nextToken();
  lexer = saved;
  return token;
}

std::optional<Call> Parser::parseCall(std::string dialect_name) {
  if (current.type != TokenType::Dot) {
    error() << "expected '.' in optimization rule, got '"
        << current.value << "'";
    return std::nullopt;
  }
  advance();
  if (current.type != TokenType::Ident) {
    error() << "expected identifier in optimization rule, got '"
        << current.value << "'";
    return std::nullopt;
  }
  std::string op_name = current.value;
  advance();
  if (current.type != TokenType::LParen) {
    error() << "expected '(' in optimization rule, got '"
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
  if (current.type == TokenType::Integer) {
    int64_t n;
    if (StringRef(current.value).getAsInteger(10, n)) {
      error() << "integer literal out of range: " << current.value;
      return std::nullopt;
    }
    advance();
    return Expr(IntLit{n});
  }
  if (current.type == TokenType::Float) {
    double n;
    if (StringRef(current.value).getAsDouble(n)) {
      error() << "invalid floating point literal: " << current.value;
      return std::nullopt;
    }
    advance();
    return Expr(FloatLit{n});
  }
  error() << "invalid optimization rule expression";
  return std::nullopt;
}

std::optional<Cond> Parser::parseCond() {
  auto lhs = parseConj();
  if (!lhs)
    return std::nullopt;
  while (current.type == TokenType::OrOr) {
    advance();
    auto rhs = parseConj();
    if (!rhs)
      return std::nullopt;
    lhs = Cond(OrCond{box(std::move(*lhs)), box(std::move(*rhs))});
  }
  return lhs;
}

std::optional<Cond> Parser::parseConj() {
  auto lhs = parseUnary();
  if (!lhs)
    return std::nullopt;
  while (current.type == TokenType::AndAnd) {
    advance();
    auto rhs = parseUnary();
    if (!rhs)
      return std::nullopt;
    lhs = Cond(AndCond{box(std::move(*lhs)), box(std::move(*rhs))});
  }
  return lhs;
}

std::optional<Cond> Parser::parseUnary() {
  if (current.type == TokenType::Bang) {
    advance();
    auto operand = parseUnary();
    if (!operand)
      return std::nullopt;
    return Cond(NotCond{box(std::move(*operand))});
  }

  if (current.type == TokenType::LParen) {
    advance();
    auto inner = parseCond();
    if (!inner)
      return std::nullopt;
    if (current.type != TokenType::RParen) {
      error() << "expected ')' in optimization rule condition, got '"
          << current.value << "'";
      return std::nullopt;
    }
    advance();
    return inner;
  }

  // A predicate call and the left operand of a comparison both start with an
  // identifier, so one token of lookahead decides between them: only a
  // predicate is followed directly by '('. A qualified call such as
  // `eigen.inv(x)` has a '.' next and falls through to the comparison path,
  // where parseExpr handles it.
  if (current.type == TokenType::Ident &&
      peekAhead().type == TokenType::LParen) {
    std::string name = current.value;
    advance(); // consume the predicate name
    advance(); // consume '('
    std::vector<Expr> args;
    while (current.type != TokenType::RParen) {
      auto expr = parseExpr();
      if (!expr)
        return std::nullopt;
      args.push_back(std::move(*expr));
      if (current.type == TokenType::Comma) {
        advance();
      } else if (current.type != TokenType::RParen) {
        error() << "expected ',' or ')' in predicate '"
            << name << "', got '" << current.value << "'";
        return std::nullopt;
      }
    }
    advance(); // consume ')'
    return Cond(Pred{name, std::move(args)});
  }

  auto lhs = parseExpr();
  if (!lhs)
    return std::nullopt;
  std::optional<CmpOp> op = tokenToCmpOp(current.type);
  if (!op) {
    error() << "expected a comparison operator in optimization rule condition";
    return std::nullopt;
  }
  advance();
  auto rhs = parseExpr();
  if (!rhs)
    return std::nullopt;
  return Cond(Compare{*op, std::move(*lhs), std::move(*rhs)});
}

std::optional<Rule> Parser::parseRule() {
  // 'if' is only a keyword as the very first token of a rule, so it needs no
  // contextual handling and stays usable as a variable name everywhere else.
  std::optional<Cond> cond;
  if (current.type == TokenType::Ident && current.value == "if") {
    advance();
    auto parsed = parseCond();
    if (!parsed)
      return std::nullopt;
    if (current.type != TokenType::Comma) {
      error() << "expected ',' after the condition of an optimization rule, "
                 "got '"
              << current.value << "'";
      return std::nullopt;
    }
    advance();
    cond = std::move(*parsed);
  }

  auto lhs = parseExpr();
  if (!lhs)
    return std::nullopt;
  if (current.type != TokenType::Arrow) {
    error() << "expected '->' in optimization rule, got '"
        << current.value << "'";
    return std::nullopt;
  }
  advance();
  auto rhs = parseExpr();
  if (!rhs)
    return std::nullopt;
  return Rule{std::move(cond), std::move(*lhs), std::move(*rhs)};
}

} // namespace tessera
} // namespace enzyme
} // namespace mlir
