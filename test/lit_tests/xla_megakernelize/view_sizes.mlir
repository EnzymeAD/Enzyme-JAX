// RUN: enzymexlamlir-opt %s --split-input-file --xla-megakernelize --symbol-dce | FileCheck %s --match-full-lines

// Both views cover 24 bytes: 2*3 floats and 3*8 bytes. Negate the floats,
// then invert the updated bytes. Return the final bits in the float shape.
module {
  llvm.func @static_sizes(%p: !llvm.ptr) {
    %floats = "enzymexla.pointer2memref"(%p) : (!llvm.ptr) -> memref<2x3xf32>
    %bytes = "enzymexla.pointer2memref"(%p) : (!llvm.ptr) -> memref<3x8xi8>
    enzymexla.xla_wrapper @negate (%floats) : (memref<2x3xf32>) -> ()
    enzymexla.xla_wrapper @invert (%bytes) : (memref<3x8xi8>) -> ()
    llvm.return
  }
  func.func private @negate(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %y = stablehlo.negate %x : tensor<2x3xf32>
    return %y : tensor<2x3xf32>
  }
  func.func private @invert(%x: tensor<3x8xi8>) -> tensor<3x8xi8> {
    %y = stablehlo.not %x : tensor<3x8xi8>
    return %y : tensor<3x8xi8>
  }
}

// CHECK-LABEL: module {
// CHECK-NEXT:   llvm.func @static_sizes(%[[HOST_ARG0:[a-zA-Z0-9_]+]]: !llvm.ptr) {
// CHECK-NEXT:     %[[HOST_0:[a-zA-Z0-9_]+]] = "enzymexla.pointer2memref"(%[[HOST_ARG0]]) : (!llvm.ptr) -> memref<2x3xf32>
// CHECK-NEXT:     enzymexla.xla_wrapper @negate (%[[HOST_0]]) : (memref<2x3xf32>) -> ()
// CHECK-NEXT:     llvm.return
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func private @negate(%[[BODY_ARG0:[a-zA-Z0-9_]+]]: tensor<2x3xf32>) -> tensor<2x3xf32> {
// CHECK-NEXT:     %[[BODY_0:[a-zA-Z0-9_]+]] = stablehlo.negate %[[BODY_ARG0]] : tensor<2x3xf32>
// CHECK-NEXT:     %[[BODY_1:[a-zA-Z0-9_]+]] = stablehlo.reshape %[[BODY_0]] : (tensor<2x3xf32>) -> tensor<6xf32>
// CHECK-NEXT:     %[[BODY_2:[a-zA-Z0-9_]+]] = stablehlo.bitcast_convert %[[BODY_1]] : (tensor<6xf32>) -> tensor<6x4xi8>
// CHECK-NEXT:     %[[BODY_3:[a-zA-Z0-9_]+]] = stablehlo.reshape %[[BODY_2]] : (tensor<6x4xi8>) -> tensor<24xi8>
// CHECK-NEXT:     %[[BODY_4:[a-zA-Z0-9_]+]] = stablehlo.reshape %[[BODY_3]] : (tensor<24xi8>) -> tensor<3x8xi8>
// CHECK-NEXT:     %[[BODY_5:[a-zA-Z0-9_]+]] = stablehlo.not %[[BODY_4]] : tensor<3x8xi8>
// CHECK-NEXT:     %[[BODY_6:[a-zA-Z0-9_]+]] = stablehlo.reshape %[[BODY_5]] : (tensor<3x8xi8>) -> tensor<24xi8>
// CHECK-NEXT:     %[[BODY_7:[a-zA-Z0-9_]+]] = stablehlo.reshape %[[BODY_6]] : (tensor<24xi8>) -> tensor<2x3x4xi8>
// CHECK-NEXT:     %[[BODY_8:[a-zA-Z0-9_]+]] = stablehlo.bitcast_convert %[[BODY_7]] : (tensor<2x3x4xi8>) -> tensor<2x3xf32>
// CHECK-NEXT:     return %[[BODY_8]] : tensor<2x3xf32>
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK-NEXT: // {{-----}}

// -----


// Both sizes come from the same runtime buffer. The dynamic dimensions have
// different positions and scales: N*3 floats correspond to 2*(6*N) bytes.
// Keep the scale factors when converting to bytes and back to floats.
module {
  llvm.func @dynamic_sizes(%p: !llvm.ptr) {
    %floats = "enzymexla.pointer2memref"(%p) : (!llvm.ptr) -> memref<?x3xf32>
    %bytes = "enzymexla.pointer2memref"(%p) : (!llvm.ptr) -> memref<2x?xi8>
    enzymexla.xla_wrapper @negate (%floats) : (memref<?x3xf32>) -> ()
    enzymexla.xla_wrapper @invert (%bytes) : (memref<2x?xi8>) -> ()
    llvm.return
  }
  func.func private @negate(%x: tensor<?x3xf32>) -> tensor<?x3xf32> {
    %y = stablehlo.negate %x : tensor<?x3xf32>
    return %y : tensor<?x3xf32>
  }
  func.func private @invert(%x: tensor<2x?xi8>) -> tensor<2x?xi8> {
    %y = stablehlo.not %x : tensor<2x?xi8>
    return %y : tensor<2x?xi8>
  }
}

// CHECK-LABEL: module {
// CHECK-NEXT:   llvm.func @dynamic_sizes(%[[HOST_ARG0:[a-zA-Z0-9_]+]]: !llvm.ptr) {
// CHECK-NEXT:     %[[HOST_0:[a-zA-Z0-9_]+]] = "enzymexla.pointer2memref"(%[[HOST_ARG0]]) : (!llvm.ptr) -> memref<?x3xf32>
// CHECK-NEXT:     enzymexla.xla_wrapper @negate (%[[HOST_0]]) : (memref<?x3xf32>) -> ()
// CHECK-NEXT:     llvm.return
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func private @negate(%[[BODY_ARG0:[a-zA-Z0-9_]+]]: tensor<?x3xf32>) -> tensor<?x3xf32> {
// CHECK-NEXT:     %[[BODY_C:[a-zA-Z0-9_]+]] = stablehlo.constant dense<[0, 3, 4]> : tensor<3xi32>
// CHECK-NEXT:     %[[BODY_C_0:[a-zA-Z0-9_]+]] = stablehlo.constant dense<12> : tensor<i32>
// CHECK-NEXT:     %[[BODY_C_1:[a-zA-Z0-9_]+]] = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT:     %[[BODY_C_2:[a-zA-Z0-9_]+]] = stablehlo.constant dense<[2, 0]> : tensor<2xi32>
// CHECK-NEXT:     %[[BODY_C_3:[a-zA-Z0-9_]+]] = stablehlo.constant dense<2> : tensor<i32>
// CHECK-NEXT:     %[[BODY_C_4:[a-zA-Z0-9_]+]] = stablehlo.constant dense<4> : tensor<1xi32>
// CHECK-NEXT:     %[[BODY_C_5:[a-zA-Z0-9_]+]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:     %[[BODY_C_6:[a-zA-Z0-9_]+]] = stablehlo.constant dense<0> : tensor<1xi32>
// CHECK-NEXT:     %[[BODY_C_7:[a-zA-Z0-9_]+]] = stablehlo.constant dense<3> : tensor<i32>
// CHECK-NEXT:     %[[BODY_0:[a-zA-Z0-9_]+]] = stablehlo.negate %[[BODY_ARG0]] : tensor<?x3xf32>
// CHECK-NEXT:     %[[BODY_1:[a-zA-Z0-9_]+]] = stablehlo.get_dimension_size %[[BODY_0]], dim = 0 : (tensor<?x3xf32>) -> tensor<i32>
// CHECK-NEXT:     %[[BODY_2:[a-zA-Z0-9_]+]] = stablehlo.multiply %[[BODY_1]], %[[BODY_C_7]] : tensor<i32>
// CHECK-NEXT:     %[[BODY_3:[a-zA-Z0-9_]+]] = stablehlo.reshape %[[BODY_2]] : (tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT:     %[[BODY_4:[a-zA-Z0-9_]+]] = stablehlo.dynamic_update_slice %[[BODY_C_6]], %[[BODY_3]], %[[BODY_C_5]] : (tensor<1xi32>, tensor<1xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT:     %[[BODY_5:[a-zA-Z0-9_]+]] = stablehlo.dynamic_reshape %[[BODY_0]], %[[BODY_4]] : (tensor<?x3xf32>, tensor<1xi32>) -> tensor<?xf32>
// CHECK-NEXT:     %[[BODY_6:[a-zA-Z0-9_]+]] = stablehlo.get_dimension_size %[[BODY_5]], dim = 0 : (tensor<?xf32>) -> tensor<i32>
// CHECK-NEXT:     %[[BODY_7:[a-zA-Z0-9_]+]] = stablehlo.reshape %[[BODY_6]] : (tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT:     %[[BODY_8:[a-zA-Z0-9_]+]] = stablehlo.multiply %[[BODY_7]], %[[BODY_C_4]] : tensor<1xi32>
// CHECK-NEXT:     %[[BODY_9:[a-zA-Z0-9_]+]] = stablehlo.bitcast_convert %[[BODY_5]] : (tensor<?xf32>) -> tensor<?x4xi8>
// CHECK-NEXT:     %[[BODY_10:[a-zA-Z0-9_]+]] = stablehlo.dynamic_reshape %[[BODY_9]], %[[BODY_8]] : (tensor<?x4xi8>, tensor<1xi32>) -> tensor<?xi8>
// CHECK-NEXT:     %[[BODY_11:[a-zA-Z0-9_]+]] = stablehlo.get_dimension_size %[[BODY_10]], dim = 0 : (tensor<?xi8>) -> tensor<i32>
// CHECK-NEXT:     %[[BODY_12:[a-zA-Z0-9_]+]] = stablehlo.divide %[[BODY_11]], %[[BODY_C_3]] : tensor<i32>
// CHECK-NEXT:     %[[BODY_13:[a-zA-Z0-9_]+]] = stablehlo.reshape %[[BODY_12]] : (tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT:     %[[BODY_14:[a-zA-Z0-9_]+]] = stablehlo.dynamic_update_slice %[[BODY_C_2]], %[[BODY_13]], %[[BODY_C_1]] : (tensor<2xi32>, tensor<1xi32>, tensor<i32>) -> tensor<2xi32>
// CHECK-NEXT:     %[[BODY_15:[a-zA-Z0-9_]+]] = stablehlo.dynamic_reshape %[[BODY_10]], %[[BODY_14]] : (tensor<?xi8>, tensor<2xi32>) -> tensor<2x?xi8>
// CHECK-NEXT:     %[[BODY_16:[a-zA-Z0-9_]+]] = stablehlo.not %[[BODY_15]] : tensor<2x?xi8>
// CHECK-NEXT:     %[[BODY_17:[a-zA-Z0-9_]+]] = stablehlo.get_dimension_size %[[BODY_16]], dim = 1 : (tensor<2x?xi8>) -> tensor<i32>
// CHECK-NEXT:     %[[BODY_18:[a-zA-Z0-9_]+]] = stablehlo.multiply %[[BODY_17]], %[[BODY_C_3]] : tensor<i32>
// CHECK-NEXT:     %[[BODY_19:[a-zA-Z0-9_]+]] = stablehlo.reshape %[[BODY_18]] : (tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT:     %[[BODY_20:[a-zA-Z0-9_]+]] = stablehlo.dynamic_update_slice %[[BODY_C_6]], %[[BODY_19]], %[[BODY_C_5]] : (tensor<1xi32>, tensor<1xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT:     %[[BODY_21:[a-zA-Z0-9_]+]] = stablehlo.dynamic_reshape %[[BODY_16]], %[[BODY_20]] : (tensor<2x?xi8>, tensor<1xi32>) -> tensor<?xi8>
// CHECK-NEXT:     %[[BODY_22:[a-zA-Z0-9_]+]] = stablehlo.get_dimension_size %[[BODY_21]], dim = 0 : (tensor<?xi8>) -> tensor<i32>
// CHECK-NEXT:     %[[BODY_23:[a-zA-Z0-9_]+]] = stablehlo.divide %[[BODY_22]], %[[BODY_C_0]] : tensor<i32>
// CHECK-NEXT:     %[[BODY_24:[a-zA-Z0-9_]+]] = stablehlo.reshape %[[BODY_23]] : (tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT:     %[[BODY_25:[a-zA-Z0-9_]+]] = stablehlo.dynamic_update_slice %[[BODY_C]], %[[BODY_24]], %[[BODY_C_5]] : (tensor<3xi32>, tensor<1xi32>, tensor<i32>) -> tensor<3xi32>
// CHECK-NEXT:     %[[BODY_26:[a-zA-Z0-9_]+]] = stablehlo.dynamic_reshape %[[BODY_21]], %[[BODY_25]] : (tensor<?xi8>, tensor<3xi32>) -> tensor<?x3x4xi8>
// CHECK-NEXT:     %[[BODY_27:[a-zA-Z0-9_]+]] = stablehlo.bitcast_convert %[[BODY_26]] : (tensor<?x3x4xi8>) -> tensor<?x3xf32>
// CHECK-NEXT:     return %[[BODY_27]] : tensor<?x3xf32>
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NOT: {{.}}
