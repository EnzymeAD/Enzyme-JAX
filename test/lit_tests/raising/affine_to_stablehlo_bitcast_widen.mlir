// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --enzyme-hlo-opt | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func.func @kern_widen_dyn(%buf: memref<?xi8>, %out: memref<?xf64>) {
    affine.parallel (%i) = (0) to (8) {
      %ptr = "enzymexla.memref2pointer"(%buf) : (memref<?xi8>) -> !llvm.ptr
      %m64 = "enzymexla.pointer2memref"(%ptr) : (!llvm.ptr) -> memref<?xf64>
      %v64 = affine.load %m64[%i * 2 + 4] : memref<?xf64>
      affine.store %v64, %out[%i] : memref<?xf64>
    }
    return
  }

  func.func @kern_widen_static(%buf: memref<128xi8>, %out: memref<8xf64>) {
    affine.parallel (%i) = (0) to (8) {
      %ptr = "enzymexla.memref2pointer"(%buf) : (memref<128xi8>) -> !llvm.ptr
      %m64 = "enzymexla.pointer2memref"(%ptr) : (!llvm.ptr) -> memref<16xf64>
      %v64 = affine.load %m64[%i * 2] : memref<16xf64>
      affine.store %v64, %out[%i] : memref<8xf64>
    }
    return
  }

  func.func @kern_i8_to_i1_dyn(%buf: memref<?xi8>, %out: memref<?xi1>) {
    affine.parallel (%i) = (0) to (8) {
      %ptr = "enzymexla.memref2pointer"(%buf) : (memref<?xi8>) -> !llvm.ptr
      %m1 = "enzymexla.pointer2memref"(%ptr) : (!llvm.ptr) -> memref<?xi1>
      %v1 = affine.load %m1[%i * 2] : memref<?xi1>
      affine.store %v1, %out[%i] : memref<?xi1>
    }
    return
  }
}

// CHECK-LABEL: func.func private @kern_i8_to_i1_dyn_raised(%arg0: tensor<?xi8>, %arg1: tensor<?xi1>) -> (tensor<?xi8>, tensor<?xi1>)
// CHECK: stablehlo.convert %arg0 : (tensor<?xi8>) -> tensor<?xi1>

// CHECK-LABEL: func.func private @kern_widen_static_raised(%arg0: tensor<128xi8>, %arg1: tensor<8xf64>) -> (tensor<128xi8>, tensor<8xf64>)
// CHECK-NOT: stablehlo.dynamic_reshape
// CHECK: %[[R0:.+]] = stablehlo.reshape %arg0 : (tensor<128xi8>) -> tensor<16x8xi8>
// CHECK: %[[BC0:.+]] = stablehlo.bitcast_convert %[[R0]] : (tensor<16x8xi8>) -> tensor<16xf64>
// CHECK: stablehlo.slice %[[BC0]] [0:16:2] : (tensor<16xf64>) -> tensor<8xf64>

// CHECK-LABEL: func.func private @kern_widen_dyn_raised(%arg0: tensor<?xi8>, %arg1: tensor<?xf64>) -> (tensor<?xi8>, tensor<?xf64>)
// CHECK: stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?xi8>) -> tensor<i32>
// CHECK: stablehlo.divide
// CHECK: stablehlo.dynamic_reshape %arg0, {{.*}} : (tensor<?xi8>, tensor<2xi32>) -> tensor<?x8xi8>
// CHECK: stablehlo.bitcast_convert {{.*}} : (tensor<?x8xi8>) -> tensor<?xf64>
