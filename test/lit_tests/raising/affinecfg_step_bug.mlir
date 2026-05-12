// RUN: enzymexlamlir-opt -affine-cfg -allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: func.func @repro
// CHECK:           %[[C2:.*]] = arith.constant 2 : i64
// CHECK:           affine.for %[[IV:.*]] = 0 to 4 {
// CHECK-NEXT:        %[[CAST:.*]] = arith.index_cast %[[IV]] : index to i64
// CHECK-NEXT:        %[[MUL:.*]] = arith.muli %[[CAST]], %[[C2]] : i64
// CHECK-NEXT:        %[[ADD:.*]] = arith.addi %[[MUL]], %[[C2]] : i64
// CHECK-NEXT:        "test.use"(%[[ADD]]) : (i64) -> ()
// CHECK-NEXT:      }
func.func @repro() {
  %c2_i64 = arith.constant 2 : i64
  %c10_i64 = arith.constant 10 : i64
  
  scf.for %arg2 = %c2_i64 to %c10_i64 step %c2_i64 : i64 {
    "test.use"(%arg2) : (i64) -> ()
  }
  return
}
