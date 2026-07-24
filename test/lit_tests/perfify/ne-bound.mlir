// RUN: enzymexlamlir-opt %s | FileCheck %s
module {
   func.func @foo(%b0 : i64, %a0: i64) -> i64 {
      %cond = arith.cmpi eq, %b0, %a0 : i64
      %res = scf.if %cond -> (i64) { // take this branch, and we assume no extra overhead for taking this if statement
         %a1 = arith.muli %a0, %a0 : i64 // take 3 ops in our cost model
         %a2 = arith.muli %a1, %a1 : i64 // 3 ops
         %a3 = arith.muli %a2, %a2 : i64
         scf.yield %a3 : i64 // total of 9 ops by this point
      } else {
         scf.yield %a0 : i64
      }
      return %res : i64
   }

    perfify.assumptions { // operation in the dialect
     perfify.cost "arith.muli" 3 // op
     perfify.cost "func.return" 0
     perfify.cost "scf.yield" 0
     perfify.cost "arith.cmpi" 0
     perfify.cost "scf.if" 0
   
     perfify.conditions @foo true pre { // true here meaning verification is enabled
        %b0 = perfify.arg 0 : !perfify.ccost
        %c0 = perfify.constant_cost 0 : !perfify.ccost
        %cmp = perfify.cmp eq, %c0, %b0 : !perfify.ccost, !perfify.ccost
        perfify.assume %cmp
     } post {
        %cost = perfify.fn_cost : !perfify.ccost // compute the value of the defined operation (func.return)
        %c9 = perfify.constant_cost 9 : !perfify.ccost // set up our cost as 9
        %cmp = perfify.cmp ne, %cost, %c9 : !perfify.ccost, !perfify.ccost
        perfify.assume %cmp
     }
    }
}

// CHECK: module {
// CHECK-NEXT:  func.func @foo(%arg0: i64, %arg1: i64) -> i64 {
// CHECK-NEXT:    %0 = arith.cmpi eq, %arg0, %arg1 : i64
// CHECK-NEXT:    %1 = scf.if %0 -> (i64) {
// CHECK-NEXT:      %2 = arith.muli %arg1, %arg1 : i64
// CHECK-NEXT:      %3 = arith.muli %2, %2 : i64
// CHECK-NEXT:      %4 = arith.muli %3, %3 : i64
// CHECK-NEXT:      scf.yield %4 : i64
// CHECK-NEXT:    } else {
// CHECK-NEXT:      scf.yield %arg1 : i64
// CHECK-NEXT:    }
// CHECK-NEXT:    return %1 : i64
// CHECK-NEXT:  }
// CHECK-NEXT:  perfify.assumptions {
// CHECK-NEXT:    perfify.cost "arith.muli" 3 : i64
// CHECK-NEXT:    perfify.cost "func.return" 0 : i64
// CHECK-NEXT:    perfify.cost "scf.yield" 0 : i64
// CHECK-NEXT:    perfify.cost "arith.cmpi" 0 : i64
// CHECK-NEXT:    perfify.cost "scf.if" 0 : i64
// CHECK-NEXT:    perfify.conditions @foo true pre {
// CHECK-NEXT:      %0 = perfify.arg 0
// CHECK-NEXT:      %1 = perfify.constant_cost 0 : !perfify.cost
// CHECK-NEXT:      %2 = perfify.cmp eq, %1, %0
// CHECK-NEXT:      perfify.assume %2 {satres = true}
// CHECK-NEXT:    } post {
// CHECK-NEXT:      %0 = perfify.fn_cost : !perfify.cost
// CHECK-NEXT:      %1 = perfify.constant_cost 9 : !perfify.cost
// CHECK-NEXT:      %2 = perfify.cmp ne, %0, %1
// CHECK-NEXT:      perfify.assume %2 {satres = false}
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT: }