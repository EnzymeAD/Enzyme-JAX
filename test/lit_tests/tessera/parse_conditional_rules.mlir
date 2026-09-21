// RUN: enzymexlamlir-opt %s -parse-optimization-rules -split-input-file -verify-diagnostics

// Syntax of the optional condition clause on an optimization rule:
//
//   rule ::= [ 'if' cond ',' ] expr '->' expr
//
// These cases pin the shapes the grammar accepts and the diagnostics it gives
// for the ones it does not. Conditions are parsed but not yet lowered into a
// guarded rewrite, so every well-formed condition below currently stops at
// "not supported yet" -- that check is what proves the rule parsed rather than
// failing somewhere in the condition.

module {
  tessera.optimizations {
    // expected-error @+1 {{conditional optimization rules are not supported yet}}
    tessera.optimization "if symmetric(x), eigen.inv(x) -> eigen.inv_sym(x)"
  }
}

// -----

// Multiple arguments, both in the predicate and in the matched call.
module {
  tessera.optimizations {
    // expected-error @+1 {{conditional optimization rules are not supported yet}}
    tessera.optimization "if triangular_upper(a), eigen.solve(a, b) -> eigen.trsv(a, b)"
  }
}

// -----

// Conjunction and negation.
module {
  tessera.optimizations {
    // expected-error @+1 {{conditional optimization rules are not supported yet}}
    tessera.optimization "if symmetric(a) && !diagonal(a), eigen.matmul(a, b) -> eigen.symm(a, b)"
  }
}

// -----

// A relational test on a scalar operand.
module {
  tessera.optimizations {
    // expected-error @+1 {{conditional optimization rules are not supported yet}}
    tessera.optimization "if n > 64, eigen.blocked(x, n) -> eigen.blocked_tiled(x, n)"
  }
}

// -----

// Parenthesised disjunction binding tighter than the outer conjunction, and
// a two-character relational operator.
module {
  tessera.optimizations {
    // expected-error @+1 {{conditional optimization rules are not supported yet}}
    tessera.optimization "if (symmetric(x) || diagonal(x)) && n >= 2, eigen.inv(x) -> eigen.inv_sym(x)"
  }
}

// -----

// Negation applied to a parenthesised comparison, and a negative literal --
// the '-' here must lex as part of the number, not as the head of an arrow.
module {
  tessera.optimizations {
    // expected-error @+1 {{conditional optimization rules are not supported yet}}
    tessera.optimization "if !(a == b) || n <= -3, eigen.f(a, b) -> eigen.g(a, b)"
  }
}

// -----

// A predicate may take a call, not just a bare variable.
module {
  tessera.optimizations {
    // expected-error @+1 {{conditional optimization rules are not supported yet}}
    tessera.optimization "if symmetric(eigen.t(x)), eigen.inv(x) -> eigen.inv_sym(x)"
  }
}

// -----

// A rule with no condition is untouched by any of the above and still lowers
// to a pattern, so it produces no diagnostic at all.
module {
  tessera.optimizations {
    tessera.optimization "eigen.inv(eigen.inv(x)) -> x"
    tessera.optimization "tessera.pow(x, 2) -> tessera.mul(x, x)"
  }
}

// -----

// Trailing whitespace must not mislex: the end-of-input test runs after
// whitespace is skipped, not before.
module {
  tessera.optimizations {
    tessera.optimization "eigen.inv(eigen.inv(x)) -> x   "
  }
}

// -----

// 'if' is a keyword only as the first token of a rule, so it stays usable as
// an ordinary variable name.
module {
  tessera.optimizations {
    tessera.optimization "eigen.f(x, if) -> eigen.g(if, x)"
  }
}

// -----

// The comma after the condition is required.
module {
  tessera.optimizations {
    // expected-error @+1 {{expected ',' after the condition of an optimization rule, got 'eigen'}}
    tessera.optimization "if symmetric(x) eigen.inv(x) -> eigen.inv_sym(x)"
  }
}

// -----

// An unclosed predicate argument list.
module {
  tessera.optimizations {
    // expected-error @+1 {{expected ',' or ')' in predicate 'symmetric', got ''}}
    tessera.optimization "if symmetric(x, eigen.inv(x) -> eigen.inv_sym(x)"
  }
}

// -----

// A single '&' is never valid; only the doubled form is an operator.
module {
  tessera.optimizations {
    // expected-error @+1 {{unrecognized character '&' in optimization rule}}
    tessera.optimization "if n & 3, eigen.f(n) -> eigen.g(n)"
  }
}

// -----

// An expression in condition position with no relational operator after it.
module {
  tessera.optimizations {
    // expected-error @+1 {{expected a comparison operator in optimization rule condition}}
    tessera.optimization "if n 64, eigen.f(n) -> eigen.g(n)"
  }
}

// -----

// An unclosed parenthesised condition.
module {
  tessera.optimizations {
    // expected-error @+1 {{expected ')' in optimization rule condition, got ''}}
    tessera.optimization "if (symmetric(x), eigen.inv(x) -> eigen.inv_sym(x)"
  }
}
