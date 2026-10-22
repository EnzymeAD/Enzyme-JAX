// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --enzyme-hlo-opt --split-input-file | FileCheck %s

// A boolean is stored as a byte: a scalar Bool argument that the kernel
// loads as an i8 converts, rather than reinterpreting its single bit.
module {
  func.func private @load_bool(%arg0: memref<8xf32, 1>, %arg1: memref<i1, 1>) {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %0 = "enzymexla.memref2pointer"(%arg1) : (memref<i1, 1>) -> !llvm.ptr<1>
    %1 = "enzymexla.pointer2memref"(%0) : (!llvm.ptr<1>) -> memref<?xi8, 1>
    affine.parallel (%arg2) = (0) to (8) {
      %2 = affine.load %1[0] : memref<?xi8, 1>
      %3 = arith.trunci %2 : i8 to i1
      %4 = arith.select %3, %cst_0, %cst : f32
      affine.store %4, %arg0[%arg2] : memref<8xf32, 1>
    }
    return
  }
}

// CHECK-LABEL: func.func private @load_bool_raised(%arg0: tensor<8xf32>, %arg1: tensor<i1>) -> (tensor<8xf32>, tensor<i1>) {
// CHECK-NOT: stablehlo.get_dimension_size
// CHECK: stablehlo.convert %arg1 : (tensor<i1>) -> tensor<i8>
// CHECK-NOT: stablehlo.bitcast_convert
// CHECK: return %{{.*}}, %arg1 : tensor<8xf32>, tensor<i1>

// -----

// A Bool array the kernel writes as bytes: the stores land in the byte view,
// and the result reads that view back as booleans.
module {
  func.func private @store_bool(%arg0: memref<8xi1, 1>, %arg1: memref<8xf32, 1>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c1_i8 = arith.constant 1 : i8
    %c0_i8 = arith.constant 0 : i8
    %0 = "enzymexla.memref2pointer"(%arg0) : (memref<8xi1, 1>) -> !llvm.ptr<1>
    %1 = "enzymexla.pointer2memref"(%0) : (!llvm.ptr<1>) -> memref<?xi8, 1>
    affine.parallel (%arg2) = (0) to (8) {
      %2 = affine.load %arg1[%arg2] : memref<8xf32, 1>
      %3 = arith.cmpf ogt, %2, %cst : f32
      %4 = arith.select %3, %c1_i8, %c0_i8 : i8
      affine.store %4, %1[%arg2] : memref<?xi8, 1>
    }
    return
  }
}

// CHECK-LABEL: func.func private @store_bool_raised(%arg0: tensor<8xi1>, %arg1: tensor<8xf32>) -> (tensor<8xi1>, tensor<8xf32>) {
// CHECK: %[[BYTES:.+]] = arith.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<8xi1>, tensor<8xi8>
// CHECK: %[[BACK:.+]] = stablehlo.convert %[[BYTES]] : (tensor<8xi8>) -> tensor<8xi1>
// CHECK: return %[[BACK]], %arg1 : tensor<8xi1>, tensor<8xf32>

// -----

// A store through a view of another type of the same width reaches the
// argument's result.
module {
  func.func private @store_bitcast(%arg0: memref<8xf32, 1>) {
    %c0_i32 = arith.constant 0 : i32
    %0 = "enzymexla.memref2pointer"(%arg0) : (memref<8xf32, 1>) -> !llvm.ptr<1>
    %1 = "enzymexla.pointer2memref"(%0) : (!llvm.ptr<1>) -> memref<?xi32, 1>
    affine.parallel (%arg2) = (0) to (8) {
      affine.store %c0_i32, %1[%arg2] : memref<?xi32, 1>
    }
    return
  }
}

// CHECK-LABEL: func.func private @store_bitcast_raised(%arg0: tensor<8xf32>) -> tensor<8xf32> {
// CHECK: %[[Z:.+]] = stablehlo.constant dense<0> : tensor<8xi32>
// CHECK: %[[BACK:.+]] = stablehlo.bitcast_convert %[[Z]] : (tensor<8xi32>) -> tensor<8xf32>
// CHECK: return %[[BACK]] : tensor<8xf32>

// -----

// A wider view groups runs of the narrower elements.
module {
  func.func private @store_wide(%arg0: memref<16xi8, 1>) {
    %cst = arith.constant 1.000000e+00 : f32
    %0 = "enzymexla.memref2pointer"(%arg0) : (memref<16xi8, 1>) -> !llvm.ptr<1>
    %1 = "enzymexla.pointer2memref"(%0) : (!llvm.ptr<1>) -> memref<?xf32, 1>
    affine.parallel (%arg2) = (0) to (4) {
      affine.store %cst, %1[%arg2] : memref<?xf32, 1>
    }
    return
  }
}

// CHECK-LABEL: func.func private @store_wide_raised(%arg0: tensor<16xi8>) -> tensor<16xi8> {
// CHECK: %[[ONES:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<4xf32>
// CHECK: %[[BITS:.+]] = stablehlo.bitcast_convert %[[ONES]] : (tensor<4xf32>) -> tensor<4x4xi8>
// CHECK: %[[BACK:.+]] = stablehlo.reshape %[[BITS]] : (tensor<4x4xi8>) -> tensor<16xi8>
// CHECK: return %[[BACK]] : tensor<16xi8>
