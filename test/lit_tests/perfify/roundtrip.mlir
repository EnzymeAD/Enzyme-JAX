// RUN: enzymexlamlir-opt %s | FileCheck %s
module {
   func.func @foo() {func.return}
    perfify.assumptions { // operation in the dialect
     perfify.cost "arith.mul" 3 // op
     perfify.cost "func.return" 0
     perfify.cost "scf.yield" 0
    

     perfify.conditions @foo true pre { 
        %b0 = perfify.arg 0 // op
        %c0 = arith.constant 0 
        %cmp = arith.cmpi eq, %c0, %b0 : i64
        perfify.assume %cmp
     } post {
        // %cost = perfify.fn_cost : perfify.cost
        // %c9 = perfify.constant_cost 9 : perfify.cost // then our cost is 9
        // %cmp = arith.cmpi eq, %cost, %c9
        %b0 = perfify.arg 0 // op
        %c0 = arith.constant 0 
        %cmp = arith.cmpi eq, %c0, %b0 : i64  
        perfify.assume %cmp
     }
    }
}

// CHECK: module {
// CHECK-NEXT:  func.func @foo() {
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK-NEXT:  perfify.assumptions {
// CHECK-NEXT:    perfify.cost "arith.mul" 3 : i64
// CHECK-NEXT:    perfify.cost "func.return" 0 : i64
// CHECK-NEXT:    perfify.cost "scf.yield" 0 : i64
// CHECK-NEXT:    perfify.conditions @foo true pre {
// CHECK-NEXT:      %0 = perfify.arg 0
// CHECK-NEXT:      %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT:      %1 = arith.cmpi eq, %c0_i64, %0 : i64
// CHECK-NEXT:      perfify.assume %1
// CHECK-NEXT:    } post {
// CHECK-NEXT:      %0 = perfify.arg 0
// CHECK-NEXT:      %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT:      %1 = arith.cmpi eq, %c0_i64, %0 : i64
// CHECK-NEXT:      perfify.assume %1
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT: }