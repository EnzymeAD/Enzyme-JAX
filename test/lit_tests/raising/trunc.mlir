// RUN: enzymexlamlir-opt %s --libdevice-funcs-raise  --raise-affine-to-stablehlo --arith-raise | FileCheck %s

module {
  func.func @tr(%a : f64) -> f32 {
    %trunc = llvm.fptrunc %a : f64 to f32
    return %trunc : f32
  }
  func.func @trmem(%in : memref<1xf64>, %out : memref<1xf32>) {
    affine.parallel (%i) = (0) to (1) {
      %a = affine.load %in[0] : memref<1xf64>
      %trunc = llvm.fptrunc %a : f64 to f32
      affine.store %trunc, %out[0] : memref<1xf32>
    }
    return
  }
  func.func @ctr(%a : f64) -> f32 {
     %trunc = llvm.intr.experimental.constrained.fptrunc %a towardzero ignore : f64 to f32
     return %trunc : f32
   }
}

// CHECK-LABEL:   func.func @tr(
// CHECK-SAME:                  %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: f64) -> f32 {
// CHECK:           %[[VAL_1:.*]] = arith.truncf %[[VAL_0]] : f64 to f32
// CHECK:           return %[[VAL_1]] : f32

// CHECK-LABEL:   func.func @ctr(
// CHECK-SAME:                   %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: f64) -> f32 {
// CHECK:           %[[VAL_1:.*]] = arith.truncf %[[VAL_0]] {fpExceptionBehavior = 0 : i64} : f64 to f32
// CHECK:           return %[[VAL_1]] : f32

// CHECK-LABEL:   func.func private @trmem_raised(
// CHECK-SAME:                                    %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<1xf64>,
// CHECK-SAME:                                    %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: tensor<1xf32>) -> (tensor<1xf64>, tensor<1xf32>) {
// CHECK:           stablehlo.convert %{{.*}} : (tensor<f64>) -> tensor<f32>
