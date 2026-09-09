// RUN: enzymexlamlir-opt --affine-cfg %s | FileCheck %s

module {
  func.func @iter_args_symbol(%n: index, %ny: index, %buf: memref<?xi64>,
                              %val: i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %r = affine.for %i = 0 to 12 iter_args(%acc = %c0_i32) -> (i32) {
      %sym = arith.index_cast %acc : i32 to index
      %w = "enzymexla.gpu_wrapper"(%n, %ny, %c1, %c32, %c1, %c1) ({
        %flat = arith.muli %n, %c32 overflow<nsw> : index
        scf.parallel (%a5, %a6) = (%c0, %c0) to (%ny, %flat) step (%c1, %c1) {
          %idx = arith.addi %sym, %a6 : index
          memref.store %val, %buf[%idx] : memref<?xi64>
          scf.reduce
        }
        "enzymexla.polygeist_yield"() : () -> ()
      }) : (index, index, index, index, index, index) -> index
      %next = arith.addi %acc, %c1_i32 : i32
      affine.yield %next : i32
    }
    return
  }
}

// CHECK-LABEL:   func.func @iter_args_symbol(
// CHECK-SAME:      %[[n:.*]]: index, %[[ny:.*]]: index, %[[buf:.*]]: memref<?xi64>,
// CHECK-SAME:      %[[val:.*]]: i64) {
// CHECK:           affine.for %{{.*}} = 0 to 12 iter_args(%[[acc:.*]] = %{{.*}}) -> (i32) {
// CHECK:             %[[sym:.*]] = arith.index_cast %[[acc]] : i32 to index
// CHECK:             "enzymexla.gpu_wrapper"
// CHECK:               affine.parallel (%{{.*}}, %[[iv:.*]]) = (0, 0) to (symbol(%[[ny]]), symbol(%[[n]]) * 32) {
// The induction variable is the dim and the loop-carried value stays a symbol.
// CHECK:                 affine.store %[[val]], %[[buf]]{{\[}}%[[iv]] + symbol(%[[sym]])] : memref<?xi64>
