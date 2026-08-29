// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math | FileCheck %s

func.func private @gpu_wrapper(%ptr: !llvm.ptr) {
  %c1 = arith.constant 1 : index
  %c120 = arith.constant 120 : index
  %code = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c120, %c120, %c1) ({
    affine.for %iv = 0 to 120 {
      %mem = "enzymexla.pointer2memref"(%ptr) : (!llvm.ptr) -> memref<?xf32>
      %val = affine.load %mem[%iv] : memref<?xf32>
      %sq = arith.mulf %val, %val : f32
      affine.store %sq, %mem[%iv] : memref<?xf32>
    }
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  return
}

func.func @dgpu_wrapper(%ptr: !llvm.ptr, %dptr: !llvm.ptr) {
  enzyme.fwddiff @gpu_wrapper(%ptr, %dptr) {
    activity = [#enzyme<activity enzyme_dup>],
    ret_activity = []
  } : (!llvm.ptr, !llvm.ptr) -> ()
  return
}

// CHECK-LABEL:   func.func private @fwddiffegpu_wrapper(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr, %[[ARG1:.*]]: !llvm.ptr) {
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[C120:.*]] = arith.constant 120 : index
// CHECK:           %{{.*}} = "enzymexla.gpu_wrapper"(%[[C1]], %[[C1]], %[[C1]], %[[C120]], %[[C120]], %[[C1]]) ({
// CHECK:             %[[DMEM:.*]] = "enzymexla.pointer2memref"(%[[ARG1]]) : (!llvm.ptr) -> memref<?xf32>
// CHECK:             %[[MEM:.*]] = "enzymexla.pointer2memref"(%[[ARG0]]) : (!llvm.ptr) -> memref<?xf32>
// CHECK:             affine.for %[[IV:.*]] = 0 to 120 {
// CHECK:               %[[DVAL:.*]] = affine.load %[[DMEM]]{{\[}}%[[IV]]] : memref<?xf32>
// CHECK:               %[[VAL:.*]] = affine.load %[[MEM]]{{\[}}%[[IV]]] : memref<?xf32>
// CHECK:               %[[L:.*]] = arith.mulf %[[DVAL]], %[[VAL]] fastmath<fast> : f32
// CHECK:               %[[R:.*]] = arith.mulf %[[DVAL]], %[[VAL]] fastmath<fast> : f32
// CHECK:               %[[DSQ:.*]] = arith.addf %[[L]], %[[R]] fastmath<fast> : f32
// CHECK:               %[[SQ:.*]] = arith.mulf %[[VAL]], %[[VAL]] : f32
// CHECK:               affine.store %[[DSQ]], %[[DMEM]]{{\[}}%[[IV]]] : memref<?xf32>
// CHECK:               affine.store %[[SQ]], %[[MEM]]{{\[}}%[[IV]]] : memref<?xf32>
// CHECK:             }
// CHECK:             "enzymexla.polygeist_yield"() : () -> ()
// CHECK:           }) : (index, index, index, index, index, index) -> index
// CHECK:           return
// CHECK:         }
