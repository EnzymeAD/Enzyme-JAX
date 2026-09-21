// RUN: enzymexlamlir-opt %s -parse-optimization-rules -split-input-file -verify-diagnostics

// Syntax of the optional condition clause on an optimization rule:
//
//   rule ::= [ 'if' cond ',' ] expr '->' expr
//
// These cases pin the shapes the grammar accepts and the diagnostics it gives
// for the ones it does not. A well-formed condition produces no diagnostic at
// all here; what the resulting pattern looks like is checked separately, in
// conditional_pattern_gen.mlir.

module {
  tessera.optimizations {
    tessera.optimization "if symmetric(x), lib.foo(x) -> lib.symmetric_foo(x)"
  }
}

// -----

// Multiple arguments, both in the predicate and in the matched call.
module {
  tessera.optimizations {
    tessera.optimization "if triangular_upper(a), lib.bar(a, b) -> lib.triangular_bar(a, b)"
  }
}

// -----

// Conjunction and negation.
module {
  tessera.optimizations {
    tessera.optimization "if symmetric(a) && !diagonal(a), lib.baz(a, b) -> lib.symmetric_baz(a, b)"
  }
}

// -----

// A relational test on a scalar operand.
module {
  tessera.optimizations {
    tessera.optimization "if n > 64, lib.qux(x, n) -> lib.tiled_qux(x, n)"
  }
}

// -----

// Parenthesised disjunction binding tighter than the outer conjunction, and
// a two-character relational operator.
module {
  tessera.optimizations {
    tessera.optimization "if (symmetric(x) || diagonal(x)) && n >= 2, lib.foo(x) -> lib.symmetric_foo(x)"
  }
}

// -----

// Negation applied to a parenthesised comparison, and a negative literal --
// the '-' here must lex as part of the number, not as the head of an arrow.
module {
  tessera.optimizations {
    tessera.optimization "if !(a == b) || n <= -3, lib.f(a, b) -> lib.g(a, b)"
  }
}

// -----

// A predicate may take a call, not just a bare variable.
module {
  tessera.optimizations {
    tessera.optimization "if symmetric(lib.transpose(x)), lib.foo(x) -> lib.symmetric_foo(x)"
  }
}

// -----

// A rule with no condition is untouched by any of the above and still lowers
// to a pattern, so it produces no diagnostic at all.
module {
  tessera.optimizations {
    tessera.optimization "lib.foo(lib.foo(x)) -> x"
    tessera.optimization "tessera.pow(x, 2) -> tessera.mul(x, x)"
  }
}

// -----

// Trailing whitespace must not mislex: the end-of-input test runs after
// whitespace is skipped, not before.
module {
  tessera.optimizations {
    tessera.optimization "lib.foo(lib.foo(x)) -> x   "
  }
}

// -----

// 'if' is a keyword only as the first token of a rule, so it stays usable as
// an ordinary variable name.
module {
  tessera.optimizations {
    tessera.optimization "lib.f(x, if) -> lib.g(if, x)"
  }
}

// -----

// The comma after the condition is required.
module {
  tessera.optimizations {
    // expected-error @+1 {{expected ',' after the condition of an optimization rule, got 'lib'}}
    tessera.optimization "if symmetric(x) lib.foo(x) -> lib.symmetric_foo(x)"
  }
}

// -----

// An unclosed predicate argument list.
module {
  tessera.optimizations {
    // expected-error @+1 {{expected ',' or ')' in predicate 'symmetric', got ''}}
    tessera.optimization "if symmetric(x, lib.foo(x) -> lib.symmetric_foo(x)"
  }
}

// -----

// A single '&' is never valid; only the doubled form is an operator.
module {
  tessera.optimizations {
    // expected-error @+1 {{unrecognized character '&' in optimization rule}}
    tessera.optimization "if n & 3, lib.f(n) -> lib.g(n)"
  }
}

// -----

// An expression in condition position with no relational operator after it.
module {
  tessera.optimizations {
    // expected-error @+1 {{expected a comparison operator in optimization rule condition}}
    tessera.optimization "if n 64, lib.f(n) -> lib.g(n)"
  }
}

// -----

// An unclosed parenthesised condition.
module {
  tessera.optimizations {
    // expected-error @+1 {{expected ')' in optimization rule condition, got ''}}
    tessera.optimization "if (symmetric(x), lib.foo(x) -> lib.symmetric_foo(x)"
  }
}
