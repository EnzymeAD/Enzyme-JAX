// RUN: enzymexlamlir-opt --affine-cfg --simplify-affine-exprs %s | FileCheck %s
#set = affine_set<(d0, d1, d2, d3) : (d1 + d2 * 16 >= 0, -d1 - d2 * 16 + 61 >= 0, d0 + d3 * 16 >= 0, -d0 - d3 * 16 + 61 >= 0)>
#set1 = affine_set<(d0, d1, d2) : (-d1 - d2 * 16 + 61 >= 0, d0 - 1 >= 0, -d1 - d2 * 16 + 61 >= 0)>
module {
  func.func private @foo(%arg0: memref<31x78x78xf32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 754416 : i64, llvm.noalias}) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %cst = arith.constant 0.000000e+00 : f32
    affine.parallel (%arg13, %arg14, %arg15, %arg16, %arg17) = (0, 0, 0, 0, 0) to (15, 16, 16, 4, 4) {
      affine.if #set(%arg14, %arg15, %arg16, %arg17) {
        affine.if #set1(%arg13, %arg14, %arg17) {
          affine.store %cst, %arg0[0, 0, 0] : memref<31x78x78xf32, 1>
        } else {
          affine.store %cst, %arg0[%arg13 + (%arg14 + %arg17 * 16 + (%arg15 + %arg16 * 16 + 8) floordiv 78 + 8) floordiv 78 + 8, %arg14 + %arg17 * 16 + (%arg15 + %arg16 * 16 + 8) floordiv 78 - ((%arg14 + %arg17 * 16 + (%arg15 + %arg16 * 16 + 8) floordiv 78 + 8) floordiv 78) * 78 + 8, %arg15 + %arg16 * 16 - ((%arg15 + %arg16 * 16 + 8) floordiv 78) * 78 + 8] : memref<31x78x78xf32, 1>
        }
      }
    }
    return
  }
}

// CHECK: #[[$ATTR_0:.+]] = affine_set<(d0, d1) : (-d1 + 61 >= 0, d0 - 1 >= 0, -d1 + 61 >= 0)>

// CHECK-LABEL:   func.func private @foo(
// CHECK-SAME:                           %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<31x78x78xf32, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 754416 : i64, llvm.noalias}) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           affine.parallel (%[[VAL_2:.*]], %[[VAL_3:.*]], %[[VAL_4:.*]]) = (0, 0, 0) to (15, 62, 62) {
// CHECK:             affine.if #[[$ATTR_0]](%[[VAL_2]], %[[VAL_3]]) {
// CHECK:               affine.store %[[VAL_1]], %[[VAL_0]][0, 0, 0] : memref<31x78x78xf32, 1>
// CHECK:             } else {
// CHECK:               affine.store %[[VAL_1]], %[[VAL_0]]{{\[}}%[[VAL_2]] + 8, %[[VAL_3]] + 8, %[[VAL_4]] + 8] : memref<31x78x78xf32, 1>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }

