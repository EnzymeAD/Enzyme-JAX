// RUN: enzymexlamlir-opt %s --xla-megakernelize --symbol-dce | FileCheck %s --match-full-lines

// Minimal LBM case: A has a float view and a byte view. Both are read-only.
// The first call copies A to B and sets B[0] from A's first byte.
// The second call does the same from B to A.
//
// After fusion, keep only A and B as inputs. The second call must read both
// views from the updated B. Its byte view must not read the original B.
module {
  llvm.func @ping_pong(%a: !llvm.ptr, %b: !llvm.ptr) {
    %af = "enzymexla.pointer2memref"(%a) : (!llvm.ptr) -> memref<?xf32>
    %ab = "enzymexla.pointer2memref"(%a) : (!llvm.ptr) -> memref<?xi8>
    %bf = "enzymexla.pointer2memref"(%b) : (!llvm.ptr) -> memref<?xf32>
    %bb = "enzymexla.pointer2memref"(%b) : (!llvm.ptr) -> memref<?xi8>
    enzymexla.xla_wrapper @step (%af, %ab, %bf) : (memref<?xf32>, memref<?xi8>, memref<?xf32>) -> ()
    enzymexla.xla_wrapper @step (%bf, %bb, %af) : (memref<?xf32>, memref<?xi8>, memref<?xf32>) -> ()
    llvm.return
  }
  func.func private @step(%source: tensor<?xf32>, %bytes: tensor<?xi8>, %dest: tensor<?xf32>)
      -> (tensor<?xf32>, tensor<?xi8>, tensor<?xf32>) {
    %zero = stablehlo.constant dense<0> : tensor<i32>
    %flag = stablehlo.slice %bytes [0:1] : (tensor<?xi8>) -> tensor<1xi8>
    %flag_float = stablehlo.convert %flag : (tensor<1xi8>) -> tensor<1xf32>
    %updated = stablehlo.dynamic_update_slice %source, %flag_float, %zero : (tensor<?xf32>, tensor<1xf32>, tensor<i32>) -> tensor<?xf32>
    return %source, %bytes, %updated : tensor<?xf32>, tensor<?xi8>, tensor<?xf32>
  }
}

// CHECK-LABEL: module {
// CHECK-NEXT:   llvm.func @ping_pong(%[[HOST_A_PTR:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[HOST_B_PTR:[a-zA-Z0-9_]+]]: !llvm.ptr) {
// CHECK-NEXT:     %[[HOST_A:[a-zA-Z0-9_]+]] = "enzymexla.pointer2memref"(%[[HOST_A_PTR]]) : (!llvm.ptr) -> memref<?xf32>
// CHECK-NEXT:     %[[HOST_B:[a-zA-Z0-9_]+]] = "enzymexla.pointer2memref"(%[[HOST_B_PTR]]) : (!llvm.ptr) -> memref<?xf32>
// CHECK-NEXT:     enzymexla.xla_wrapper @step (%[[HOST_A]], %[[HOST_B]]) : (memref<?xf32>, memref<?xf32>) -> ()
// CHECK-NEXT:     llvm.return
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func private @step(%[[A:[a-zA-Z0-9_]+]]: tensor<?xf32>, %[[B:[a-zA-Z0-9_]+]]: tensor<?xf32>) -> (tensor<?xf32>, tensor<?xf32>) {
// CHECK-NEXT:     %[[ZERO:[a-zA-Z0-9_]+]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:     %[[FOUR:[a-zA-Z0-9_]+]] = stablehlo.constant dense<4> : tensor<1xi32>
// CHECK-NEXT:     %[[N_A:[a-zA-Z0-9_]+]] = stablehlo.get_dimension_size %[[A]], dim = 0 : (tensor<?xf32>) -> tensor<i32>
// CHECK-NEXT:     %[[SHAPE_A:[a-zA-Z0-9_]+]] = stablehlo.reshape %[[N_A]] : (tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT:     %[[BYTE_COUNT_A:[a-zA-Z0-9_]+]] = stablehlo.multiply %[[SHAPE_A]], %[[FOUR]] : tensor<1xi32>
// CHECK-NEXT:     %[[BYTES_A:[a-zA-Z0-9_]+]] = stablehlo.bitcast_convert %[[A]] : (tensor<?xf32>) -> tensor<?x4xi8>
// CHECK-NEXT:     %[[FLAT_A:[a-zA-Z0-9_]+]] = stablehlo.dynamic_reshape %[[BYTES_A]], %[[BYTE_COUNT_A]] : (tensor<?x4xi8>, tensor<1xi32>) -> tensor<?xi8>
// CHECK-NEXT:     %[[FLAG_A:[a-zA-Z0-9_]+]] = stablehlo.slice %[[FLAT_A]] [0:1] : (tensor<?xi8>) -> tensor<1xi8>
// CHECK-NEXT:     %[[FLOAT_FLAG_A:[a-zA-Z0-9_]+]] = stablehlo.convert %[[FLAG_A]] : (tensor<1xi8>) -> tensor<1xf32>
// CHECK-NEXT:     %[[UPDATED_B:[a-zA-Z0-9_]+]] = stablehlo.dynamic_update_slice %[[A]], %[[FLOAT_FLAG_A]], %[[ZERO]] : (tensor<?xf32>, tensor<1xf32>, tensor<i32>) -> tensor<?xf32>
// CHECK-NEXT:     %[[N_B:[a-zA-Z0-9_]+]] = stablehlo.get_dimension_size %[[UPDATED_B]], dim = 0 : (tensor<?xf32>) -> tensor<i32>
// CHECK-NEXT:     %[[SHAPE_B:[a-zA-Z0-9_]+]] = stablehlo.reshape %[[N_B]] : (tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT:     %[[BYTE_COUNT_B:[a-zA-Z0-9_]+]] = stablehlo.multiply %[[SHAPE_B]], %[[FOUR]] : tensor<1xi32>
// CHECK-NEXT:     %[[BYTES_B:[a-zA-Z0-9_]+]] = stablehlo.bitcast_convert %[[UPDATED_B]] : (tensor<?xf32>) -> tensor<?x4xi8>
// CHECK-NEXT:     %[[FLAT_B:[a-zA-Z0-9_]+]] = stablehlo.dynamic_reshape %[[BYTES_B]], %[[BYTE_COUNT_B]] : (tensor<?x4xi8>, tensor<1xi32>) -> tensor<?xi8>
// CHECK-NEXT:     %[[FLAG_B:[a-zA-Z0-9_]+]] = stablehlo.slice %[[FLAT_B]] [0:1] : (tensor<?xi8>) -> tensor<1xi8>
// CHECK-NEXT:     %[[FLOAT_FLAG_B:[a-zA-Z0-9_]+]] = stablehlo.convert %[[FLAG_B]] : (tensor<1xi8>) -> tensor<1xf32>
// CHECK-NEXT:     %[[UPDATED_A:[a-zA-Z0-9_]+]] = stablehlo.dynamic_update_slice %[[UPDATED_B]], %[[FLOAT_FLAG_B]], %[[ZERO]] : (tensor<?xf32>, tensor<1xf32>, tensor<i32>) -> tensor<?xf32>
// CHECK-NEXT:     return %[[UPDATED_A]], %[[UPDATED_B]] : tensor<?xf32>, tensor<?xf32>
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NOT: {{.}}
