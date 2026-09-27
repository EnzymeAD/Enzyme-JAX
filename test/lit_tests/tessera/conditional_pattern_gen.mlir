// RUN: enzymexlamlir-opt %s -parse-optimization-rules -split-input-file | FileCheck %s

// Every rule hands its rewrite to a native function rather than spelling it
// out declaratively. For a conditional rule that is unavoidable: whether it
// rewrites directly or becomes a guard depends on what can be proven about the
// matched values, which is not knowable here.

module {
  tessera.optimizations {
    tessera.optimization "if n > 64, lib.qux(x, n) -> lib.tiled_qux(x, n)"
  }
}

// CHECK: module @patterns
// CHECK: pdl.pattern : benefit(1) {
// CHECK-NEXT:   %[[X:.*]] = operand
// CHECK-NEXT:   %[[N:.*]] = operand
// CHECK-NEXT:   %[[QUX:.*]] = attribute = @lib.qux
// CHECK-NEXT:   %[[T:.*]] = types
// CHECK-NEXT:   %[[CALL:.*]] = operation "tessera.call"(%[[X]], %[[N]] : !pdl.value, !pdl.value)  {"callee" = %[[QUX]]} -> (%[[T]] : !pdl.range<type>)
// CHECK-NEXT:   %{{.*}} = result 0 of %[[CALL]]

// The whole rule travels to the rewrite as text: the apply pass re-parses it,
// which is what keeps the two passes from needing to share state.
// CHECK-NEXT:   %[[RULE:.*]] = attribute = "if n > 64, lib.qux(x, n) -> lib.tiled_qux(x, n)"

// The names the rule uses, in the order the values follow.
// CHECK-NEXT:   %[[NAMES:.*]] = attribute = ["x", "n"]

// Without this the pattern would match the copy of the call that the guard
// keeps in its else region, and nest guards without end.
// CHECK-NEXT:   apply_native_constraint "tesseraRuleNotApplied"(%[[CALL]], %[[RULE]] : !pdl.operation, !pdl.attribute)

// Whether the rewrite can be built and is safe here is decided in the match,
// so a rule that declines leaves the IR untouched.
// CHECK-NEXT:   apply_native_constraint "tesseraRuleApplicable"(%[[CALL]], %[[RULE]] : !pdl.operation, !pdl.attribute)

// The matched root is passed to the rewrite function by PDL itself, so it is
// deliberately absent from this argument list.
// CHECK-NEXT:   rewrite %[[CALL]] with "tesseraRewrite"(%[[RULE]], %[[NAMES]], %[[X]], %[[N]] : !pdl.attribute, !pdl.attribute, !pdl.value, !pdl.value)
// CHECK-NEXT: }

// -----

// An unconditional rule takes the same native rewrite, but never builds a
// guard, so it has no need of the constraint that keeps one from matching its
// own else region.
module {
  tessera.optimizations {
    tessera.optimization "lib.foo(lib.foo(x)) -> x"
  }
}

// CHECK: pdl.pattern : benefit(1) {
// CHECK-NOT: tesseraRuleNotApplied
// CHECK: apply_native_constraint "tesseraRuleApplicable"(%[[ROOT:.*]], %[[RULE:.*]] : !pdl.operation, !pdl.attribute)
// CHECK-NEXT: rewrite %[[ROOT]] with "tesseraRewrite"(%[[RULE]]
// CHECK-NEXT: }
