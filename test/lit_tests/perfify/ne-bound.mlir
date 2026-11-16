// RUN: enzymexlamlir-opt %s -simple-cycle-analysis | FileCheck %s
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
     perfify.cost "arith.muli" {
      arith.constant 3 // op
      perfify.yield
     }
     perfify.cost "func.return" {
      arith.constant 0
      perfify.yield
     }
     perfify.cost "scf.yield" {
      arith.constant 0
      perfify.yield
     }
     perfify.cost "arith.cmpi" {
      arith.constant 0
      perfify.yield
     }
     perfify.cost "scf.if" {
      arith.constant 0
      perfify.yield
     }
   
     perfify.conditions @foo true pre { // true here meaning verification is enabled
        %b0 = perfify.arg 0
        perfify.estimated_cost @pre_cost {
         arith.constant 0
         perfify.yield
        }
        %cmp = perfify.cmp eq, %b0, @pre_cost
        %h = perfify.hypothesis %cmp pre
        perfify.assume %h
     } post {
        %cost = perfify.fn_cost : !perfify.cost // compute the value of the defined operation (func.return)
        perfify.estimated_cost @post_cost { // set up our cost as 9
         arith.constant 9
         perfify.yield
        }
        %cmp = perfify.cmp ne, %cost, @post_cost
        %h = perfify.hypothesis %cmp post
        perfify.assume %h
     }
    }
}

// CHECK: module {
// CHECK-NEXT:   func.func @foo(%arg0: i64, %arg1: i64) -> i64 {
// CHECK-NEXT:     %0 = arith.cmpi eq, %arg0, %arg1 : i64
// CHECK-NEXT:     %1 = scf.if %0 -> (i64) {
// CHECK-NEXT:       %2 = arith.muli %arg1, %arg1 : i64
// CHECK-NEXT:       %3 = arith.muli %2, %2 : i64
// CHECK-NEXT:       %4 = arith.muli %3, %3 : i64
// CHECK-NEXT:       scf.yield %4 : i64
// CHECK-NEXT:     } else {
// CHECK-NEXT:       scf.yield %arg1 : i64
// CHECK-NEXT:     }
// CHECK-NEXT:     return %1 : i64
// CHECK-NEXT:   }
// CHECK-NEXT:   perfify.assumptions {
// CHECK-NEXT:     perfify.cost "arith.muli" {
// CHECK-NEXT:       %c3_i64 = arith.constant 3 : i64
// CHECK-NEXT:       perfify.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     perfify.cost "func.return" {
// CHECK-NEXT:       %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT:       perfify.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     perfify.cost "scf.yield" {
// CHECK-NEXT:       %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT:       perfify.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     perfify.cost "arith.cmpi" {
// CHECK-NEXT:       %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT:       perfify.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     perfify.cost "scf.if" {
// CHECK-NEXT:       %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT:       perfify.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     perfify.conditions @foo true pre {
// CHECK-NEXT:       %0 = perfify.arg 0
// CHECK-NEXT:       perfify.estimated_cost @pre_cost {
// CHECK-NEXT:         %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT:         perfify.yield
// CHECK-NEXT:       }
// CHECK-NEXT:       %1 = perfify.cmp eq, %0, @pre_cost
// CHECK-NEXT:       %2 = perfify.hypothesis %1 pre
// CHECK-NEXT:       perfify.assume %2 {satres = true}
// CHECK-NEXT:     } post {
// CHECK-NEXT:       %0 = perfify.fn_cost : !perfify.cost
// CHECK-NEXT:       perfify.estimated_cost @post_cost {
// CHECK-NEXT:         %c9_i64 = arith.constant 9 : i64
// CHECK-NEXT:         perfify.yield
// CHECK-NEXT:       }
// CHECK-NEXT:       %1 = perfify.cmp ne, %0, @post_cost
// CHECK-NEXT:       %2 = perfify.hypothesis %1 post
// CHECK-NEXT:       perfify.assume %2 {satres = false}
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
