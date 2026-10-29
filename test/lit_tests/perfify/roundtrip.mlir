// RUN: enzymexlamlir-opt %s -simple-cycle-analysis | FileCheck %s
module {
   func.func @bar(%a0: i64) -> i64 {
      %a1 = arith.muli %a0, %a0 : i64
      return %a1 : i64
   }

   func.func @foo(%b0 : i64, %a0: i64) -> i64 {
      %cond = arith.cmpi eq, %b0, %a0 : i64
      %res = scf.if %cond -> (i64) { // take this branch, and we assume no extra overhead for taking this if statement
         %a1 = arith.muli %a0, %a0 : i64
         %a2 = arith.muli %a1, %a1 : i64
         %a3 = arith.muli %a2, %a2 : i64
         %a4 = func.call @bar(%a3) : (i64) -> i64
         scf.yield %a3 : i64
      } else {
         scf.yield %a0 : i64
      }
      return %res : i64
   }

   perfify.define_symbol @C : i64

    perfify.assumptions { // operation in the dialect
     perfify.cost "arith.muli" {
      %c = perfify.sym_cost @C : i64
      %c2 = arith.constant 2
      %sq = arith.muli %c, %c : i64
      %ad = arith.addi %sq, %c2 : i64
      perfify.yield %ad : i64
     }
     perfify.cost "func.return" {
      %c2 = arith.constant 0
      perfify.yield
     }
     perfify.cost "scf.yield" {
      %c2 = arith.constant 0
      perfify.yield
     }
     perfify.cost "arith.cmpi" {
      %c2 = arith.constant 0
      perfify.yield
     }
     perfify.cost "scf.if" {
      %c2 = arith.constant 0
      perfify.yield
     }

     perfify.conditions @bar true pre {
      %b0 = perfify.arg 0
        perfify.estimated_cost @pre_cost {
         arith.constant 0
         perfify.yield
        }
        %cmp = perfify.cmp eq, %b0, @pre_cost
        %h = perfify.hypothesis %cmp pre
        perfify.assume %h
     } post {
        perfify.estimated_cost @post_cost {
         %c = perfify.sym_cost @C : i64
         %c2 = arith.constant 2
         %sq = arith.muli %c, %c : i64
         %ad = arith.addi %sq, %c2 : i64
         perfify.yield %ad : i64
        }
        %b0 = perfify.fn_cost : !perfify.cost
        %cmp = perfify.cmp eq, %b0, @post_cost
        %h = perfify.hypothesis %cmp post
        perfify.assume %h
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
        perfify.estimated_cost @post_cost {
            %c = perfify.sym_cost @C : i64
            %c2 = arith.constant 2
            %sq = arith.muli %c, %c : i64
            %ad = arith.addi %sq, %c2 : i64
            %c3 = arith.constant 3
            %ae = arith.muli %c3, %ad : i64
            %af = arith.addi %ae, %ad : i64 // 3 muli in @foo + call to @bar
            perfify.yield %af : i64
        }
        %cmp = perfify.cmp eq, %cost, @post_cost
        %h = perfify.hypothesis %cmp post
        perfify.assume %h
     }
    }
}

// CHECK: module {
// CHECK-NEXT:   func.func @bar(%arg0: i64) -> i64 {
// CHECK-NEXT:     %0 = arith.muli %arg0, %arg0 : i64
// CHECK-NEXT:     return %0 : i64
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @foo(%arg0: i64, %arg1: i64) -> i64 {
// CHECK-NEXT:     %0 = arith.cmpi eq, %arg0, %arg1 : i64
// CHECK-NEXT:     %1 = scf.if %0 -> (i64) {
// CHECK-NEXT:       %2 = arith.muli %arg1, %arg1 : i64
// CHECK-NEXT:       %3 = arith.muli %2, %2 : i64
// CHECK-NEXT:       %4 = arith.muli %3, %3 : i64
// CHECK-NEXT:       %5 = func.call @bar(%4) : (i64) -> i64
// CHECK-NEXT:       scf.yield %4 : i64
// CHECK-NEXT:     } else {
// CHECK-NEXT:       scf.yield %arg1 : i64
// CHECK-NEXT:     }
// CHECK-NEXT:     return %1 : i64
// CHECK-NEXT:   }
// CHECK-NEXT:   perfify.define_symbol @C : i64
// CHECK-NEXT:   perfify.assumptions {
// CHECK-NEXT:     perfify.cost "arith.muli" {
// CHECK-NEXT:       %0 = perfify.sym_cost @C : i64
// CHECK-NEXT:       %c2_i64 = arith.constant 2 : i64
// CHECK-NEXT:       %1 = arith.muli %0, %0 : i64
// CHECK-NEXT:       %2 = arith.addi %1, %c2_i64 : i64
// CHECK-NEXT:       perfify.yield %2 : i64
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
// CHECK-NEXT:     perfify.conditions @bar true pre {
// CHECK-NEXT:       %0 = perfify.arg 0
// CHECK-NEXT:       perfify.estimated_cost @pre_cost {
// CHECK-NEXT:         %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT:         perfify.yield
// CHECK-NEXT:       }
// CHECK-NEXT:       %1 = perfify.cmp eq, %0, @pre_cost
// CHECK-NEXT:       %2 = perfify.hypothesis %1 pre
// CHECK-NEXT:       perfify.assume %2 {satres = true}
// CHECK-NEXT:     } post {
// CHECK-NEXT:       perfify.estimated_cost @post_cost {
// CHECK-NEXT:         %3 = perfify.sym_cost @C : i64
// CHECK-NEXT:         %c2_i64 = arith.constant 2 : i64
// CHECK-NEXT:         %4 = arith.muli %3, %3 : i64
// CHECK-NEXT:         %5 = arith.addi %4, %c2_i64 : i64
// CHECK-NEXT:         perfify.yield %5 : i64
// CHECK-NEXT:       }
// CHECK-NEXT:       %0 = perfify.fn_cost : !perfify.cost
// CHECK-NEXT:       %1 = perfify.cmp eq, %0, @post_cost
// CHECK-NEXT:       %2 = perfify.hypothesis %1 post
// CHECK-NEXT:       perfify.assume %2 {satres = true}
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
// CHECK-NEXT:         %3 = perfify.sym_cost @C : i64
// CHECK-NEXT:         %c2_i64 = arith.constant 2 : i64
// CHECK-NEXT:         %4 = arith.muli %3, %3 : i64
// CHECK-NEXT:         %5 = arith.addi %4, %c2_i64 : i64
// CHECK-NEXT:         %c3_i64 = arith.constant 3 : i64
// CHECK-NEXT:         %6 = arith.muli %c3_i64, %5 : i64
// CHECK-NEXT:         %7 = arith.addi %6, %5 : i64
// CHECK-NEXT:         perfify.yield %7 : i64
// CHECK-NEXT:       }
// CHECK-NEXT:       %1 = perfify.cmp eq, %0, @post_cost
// CHECK-NEXT:       %2 = perfify.hypothesis %1 post
// CHECK-NEXT:       perfify.assume %2 {satres = true}
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
