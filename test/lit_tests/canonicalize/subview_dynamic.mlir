// RUN: enzymexlamlir-opt --canonicalize %s | FileCheck %s

func.func @subview_dyn(%iv: index, %jv: index, %ptr: !llvm.ptr) -> f64 {
  %mem = "enzymexla.pointer2memref"(%ptr) : (!llvm.ptr) -> memref<4x4xf64>
  %subview = memref.subview %mem[%iv, 0] [1, 4] [1, 1] : memref<4x4xf64> to memref<4xf64, strided<[1], offset: ?>>
  %load = memref.load %subview[%jv] : memref<4xf64, strided<[1], offset: ?>>
  return %load : f64
}

// CHECK-LABEL:   func.func @subview_dyn(
// CHECK-SAME:      %[[ARG0:.*]]: index, %[[ARG1:.*]]: index,
// CHECK-SAME:      %[[ARG2:.*]]: !llvm.ptr) -> f64 {
// CHECK:           %[[VAL_0:.*]] = "enzymexla.pointer2memref"(%[[ARG2]]) : (!llvm.ptr) -> memref<4x4xf64>
// CHECK:           %[[LOAD_0:.*]] = memref.load %[[VAL_0]]{{\[}}%[[ARG0]], %[[ARG1]]] : memref<4x4xf64>
// CHECK:           return %[[LOAD_0]] : f64
// CHECK:         }
