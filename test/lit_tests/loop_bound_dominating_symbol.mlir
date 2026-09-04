// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(simplify-affine-exprs)" %s | FileCheck %s
// RUN: enzymexlamlir-opt --affine-cfg %s | FileCheck %s

// The outer parallel bound %n is a symbol of the gpu wrapper scope only by
// dominating it, being defined inside an scf.if rather than at the top level
// of a scope. The domain of the inner loop must still be derivable, so that
// under t >= 0, max(t + 2, 2) folds to t + 2.
func.func private @use(i32)
func.func @dominating(%ni: i32, %cond: i1) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c2_i32 = arith.constant 2 : i32
  scf.if %cond {
    %n = arith.index_cast %ni : i32 to index
    %w = "enzymexla.gpu_wrapper"(%n, %c1, %c1, %c2, %c1, %c1) ({
      affine.parallel (%b) = (0) to (symbol(%n)) {
        affine.parallel (%t) = (0) to (2) {
          %t2 = arith.addi %t, %c2 : index
          %lb = arith.index_castui %t2 : index to i32
          %m = arith.maxsi %lb, %c2_i32 : i32
          func.call @use(%m) : (i32) -> ()
        }
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
  }
  return
}

// CHECK-LABEL: func.func @dominating(
// CHECK-NOT: arith.maxsi
// CHECK: %[[lb:.+]] = arith.index_castui
// CHECK-NOT: arith.maxsi
// CHECK: func.call @use(%[[lb]])
