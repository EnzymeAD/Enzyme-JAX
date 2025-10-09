// RUN: polygeist-opt --cpuify="method=distribute" --split-input-file %s | FileCheck %s

module {
  func.func private @use(i1) -> ()
  func.func private @something() -> ()
   func.func @_Z9calc_pathi(%arg0: i32, %c : i1) attributes {llvm.linkage = #llvm.linkage<external>} {
     %c0 = arith.constant 0 : index
     %c0_i32 = arith.constant 0 : i32
     %c1 = arith.constant 1 : index
     %false = arith.constant false
     %c9 = arith.constant 9 : index
     %true = arith.constant true
       %23 = memref.alloca() : memref<256xi32>
       scf.parallel (%arg4) = (%c0) to (%c9) step (%c1) {
           %26 = scf.if %c -> i1 {
             memref.store %c0_i32, %23[%c0] : memref<256xi32>
             "polygeist.barrier"(%arg4) : (index) -> ()
             func.call @something() : () -> ()
             scf.yield %true : i1
           } else {
             scf.yield %false : i1
           }
           func.call @use(%26) : (i1) -> ()
           scf.yield
       }
     return
   }
  func.func @fast(%arg0: i32, %c : i1, %25 : memref<9x9xi1>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c1 = arith.constant 1 : index
    %false = arith.constant false
    %c9 = arith.constant 9 : index
    %true = arith.constant true
      %23 = memref.alloca() : memref<256xi32>
      scf.parallel (%arg4) = (%c0) to (%c9) step (%c1) {
          %26 = scf.if %c -> i1 {
            memref.store %c0_i32, %23[%c0] : memref<256xi32>
            "polygeist.barrier"(%arg4) : (index) -> ()
            func.call @something() : () -> ()
            scf.yield %true : i1
          } else {
            scf.yield %false : i1
          }
          %s = "polygeist.subindex"(%25, %arg4) : (memref<9x9xi1>, index) -> memref<9xi1>
          memref.store %26, %s[%arg4] : memref<9xi1>
          scf.yield
      }
    return
  }
}

// CHECK:   func.func @_Z9calc_pathi(%[[arg0:.+]]: i32, %[[arg1:.+]]: i1)
// CHECK-NEXT:     %[[c0:.+]] = arith.constant 0 : index
// CHECK-NEXT:     %[[c0_i32:.+]] = arith.constant 0 : i32
// CHECK-NEXT:     %[[c1:.+]] = arith.constant 1 : index
// CHECK-NEXT:     %[[false:.+]] = arith.constant false
// CHECK-NEXT:     %[[c9:.+]] = arith.constant 9 : index
// CHECK-NEXT:     %[[true:.+]] = arith.constant true
// CHECK-NEXT:     %[[V0:.+]] = memref.alloca() : memref<256xi32>
// CHECK-NEXT:     memref.alloca_scope  {
// CHECK-NEXT:       %[[V1:.+]] = memref.alloca(%[[c9]]) : memref<?xi1>
// CHECK-NEXT:         memref.alloca_scope  {
// CHECK-NEXT:           scf.if %[[arg1]] {
// CHECK-NEXT:             memref.alloca_scope  {
// CHECK-NEXT:               scf.parallel (%[[arg2:.+]]) = (%[[c0]]) to (%[[c9]]) step (%[[c1]]) {
// CHECK-NEXT:                 memref.store %[[c0_i32]], %[[V0]][%[[c0]]] : memref<256xi32>
// CHECK-NEXT:                 scf.yield
// CHECK-NEXT:               }
// CHECK-NEXT:               scf.parallel (%[[arg2:.+]]) = (%[[c0]]) to (%[[c9]]) step (%[[c1]]) {
// CHECK-NEXT:                 func.call @something() : () -> ()
// CHECK-NEXT:                 %[[V2:.+]] = "polygeist.subindex"(%[[V1]], %[[arg2]]) : (memref<?xi1>, index) -> memref<i1>
// CHECK-NEXT:                 memref.store %[[true]], %[[V2]][] : memref<i1>
// CHECK-NEXT:                 scf.yield
// CHECK-NEXT:               }
// CHECK-NEXT:             }
// CHECK-NEXT:           } else {
// CHECK-NEXT:             scf.parallel (%[[arg2:.+]]) = (%[[c0]]) to (%[[c9]]) step (%[[c1]]) {
// CHECK-NEXT:               %[[V2:.+]] = "polygeist.subindex"(%[[V1]], %[[arg2]]) : (memref<?xi1>, index) -> memref<i1>
// CHECK-NEXT:               memref.store %[[false]], %[[V2]][] : memref<i1>
// CHECK-NEXT:               scf.yield
// CHECK-NEXT:             }
// CHECK-NEXT:           }
// CHECK-NEXT:           scf.parallel (%[[arg2:.+]]) = (%[[c0]]) to (%[[c9]]) step (%[[c1]]) {
// CHECK-NEXT:             %[[V2:.+]] = "polygeist.subindex"(%[[V1]], %[[arg2]]) : (memref<?xi1>, index) -> memref<i1>
// CHECK-NEXT:             %[[V3:.+]] = memref.load %[[V2]][] : memref<i1>
// CHECK-NEXT:             func.call @use(%[[V3]]) : (i1) -> ()
// CHECK-NEXT:             scf.yield
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK-NEXT:   func.func @fast(%[[arg0:.+]]: i32, %[[arg1:.+]]: i1, %[[arg2:.+]]: memref<9x9xi1>) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %[[c0:.+]] = arith.constant 0 : index
// CHECK-NEXT:     %[[c0_i32:.+]] = arith.constant 0 : i32
// CHECK-NEXT:     %[[c1:.+]] = arith.constant 1 : index
// CHECK-NEXT:     %[[false:.+]] = arith.constant false
// CHECK-NEXT:     %[[c9:.+]] = arith.constant 9 : index
// CHECK-NEXT:     %[[true:.+]] = arith.constant true
// CHECK-NEXT:     %[[V0:.+]] = memref.alloca() : memref<256xi32>
// CHECK-NEXT:     scf.if %[[arg1]] {
// CHECK-NEXT:       memref.alloca_scope  {
// CHECK-NEXT:         scf.parallel (%[[arg3:.+]]) = (%[[c0]]) to (%[[c9]]) step (%[[c1]]) {
// CHECK-NEXT:           memref.store %[[c0_i32]], %[[V0]][%[[c0]]] : memref<256xi32>
// CHECK-NEXT:           scf.yield
// CHECK-NEXT:         }
// CHECK-NEXT:         scf.parallel (%[[arg3:.+]]) = (%[[c0]]) to (%[[c9]]) step (%[[c1]]) {
// CHECK-NEXT:           func.call @something() : () -> ()
// CHECK-NEXT:           %[[V1:.+]] = "polygeist.subindex"(%[[arg2]], %[[arg3]]) : (memref<9x9xi1>, index) -> memref<9xi1>
// CHECK-NEXT:           memref.store %[[true]], %[[V1]][%[[arg3]]] : memref<9xi1>
// CHECK-NEXT:           scf.yield
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     } else {
// CHECK-NEXT:       scf.parallel (%[[arg3:.+]]) = (%[[c0]]) to (%[[c9]]) step (%[[c1]]) {
// CHECK-NEXT:         %[[V1:.+]] = "polygeist.subindex"(%[[arg2]], %[[arg3]]) : (memref<9x9xi1>, index) -> memref<9xi1>
// CHECK-NEXT:         memref.store %[[false]], %[[V1]][%[[arg3]]] : memref<9xi1>
// CHECK-NEXT:         scf.yield
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
