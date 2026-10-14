// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo="prefer_while_raising=false err_if_not_fully_raised=true" --arith-raise --canonicalize | FileCheck %s

// A store under a guard varying along a lane the store does not index (vd)
// picks the admitted lane's value. The guard here also varies along dy
// (the scf.if), and the store's one index runs over qx and dy: both are
// axes the store keeps, so only vd reduces away. (Reducing dy too wrote
// lane dy = 0's values into every row, with qx's value taken at dy.)
func.func private @k(%in: memref<48xf64, 1>, %out: memref<16xf64, 1>) {
  affine.parallel (%e) = (0) to (1) {
    affine.parallel (%qx, %dy, %vd) = (0, 0, 0) to (4, 4, 3) {
      %n = affine.apply affine_map<(d0) -> (d0 + 2)>(%vd)
      %c = arith.index_cast %n : index to i32
      %d = arith.index_cast %dy : index to i32
      %lt = arith.cmpi ult, %d, %c : i32
      scf.if %lt {
        %v = affine.load %in[%vd * 16 + %dy * 4 + %qx] : memref<48xf64, 1>
        affine.if affine_set<(d0) : (d0 - 2 == 0)>(%vd) {
          affine.store %v, %out[%qx + %dy * 4] : memref<16xf64, 1>
        }
      }
    }
  }
  return
}

// CHECK:  func.func private @k_raised(%arg0: tensor<48xf64>, %arg1: tensor<16xf64>) -> (tensor<48xf64>, tensor<16xf64>) {
// CHECK-NEXT:   %c = stablehlo.constant dense<false> : tensor<i1>
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<-2> : tensor<i64>
// CHECK-NEXT:   %c_2 = stablehlo.constant dense<4> : tensor<i64>
// CHECK-NEXT:   %c_3 = stablehlo.constant dense<16> : tensor<i64>
// CHECK-NEXT:   %c_4 = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:   %c_5 = stablehlo.constant dense<1> : tensor<3xi64>
// CHECK-NEXT:   %c_6 = stablehlo.constant dense<0> : tensor<3xi64>
// CHECK-NEXT:   %c_7 = stablehlo.constant dense<1> : tensor<4xi64>
// CHECK-NEXT:   %c_8 = stablehlo.constant dense<0> : tensor<4xi64>
// CHECK-NEXT:   %0 = stablehlo.iota dim = 0 : tensor<4xi64>
// CHECK-NEXT:   %1 = stablehlo.add %0, %c_8 : tensor<4xi64>
// CHECK-NEXT:   %2 = stablehlo.multiply %1, %c_7 : tensor<4xi64>
// CHECK-NEXT:   %3 = stablehlo.iota dim = 0 : tensor<4xi64>
// CHECK-NEXT:   %4 = stablehlo.add %3, %c_8 : tensor<4xi64>
// CHECK-NEXT:   %5 = stablehlo.multiply %4, %c_7 : tensor<4xi64>
// CHECK-NEXT:   %6 = stablehlo.iota dim = 0 : tensor<3xi64>
// CHECK-NEXT:   %7 = stablehlo.add %6, %c_6 : tensor<3xi64>
// CHECK-NEXT:   %8 = stablehlo.multiply %7, %c_5 : tensor<3xi64>
// CHECK-NEXT:   %9 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:   %10 = stablehlo.add %8, %9 : tensor<3xi64>
// CHECK-NEXT:   %11 = stablehlo.convert %10 : (tensor<3xi64>) -> tensor<3xi32>
// CHECK-NEXT:   %12 = stablehlo.convert %5 : (tensor<4xi64>) -> tensor<4xi32>
// CHECK-NEXT:   %13 = stablehlo.broadcast_in_dim %12, dims = [0] : (tensor<4xi32>) -> tensor<4x3xi32>
// CHECK-NEXT:   %14 = stablehlo.broadcast_in_dim %11, dims = [1] : (tensor<3xi32>) -> tensor<4x3xi32>
// CHECK-NEXT:   %15 = stablehlo.convert %13 : (tensor<4x3xi32>) -> tensor<4x3xui32>
// CHECK-NEXT:   %16 = stablehlo.convert %14 : (tensor<4x3xi32>) -> tensor<4x3xui32>
// CHECK-NEXT:   %17 = stablehlo.compare LT, %15, %16, UNSIGNED : (tensor<4x3xui32>, tensor<4x3xui32>) -> tensor<4x3xi1>
// CHECK-NEXT:   %18 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:   %19 = stablehlo.multiply %8, %18 : tensor<3xi64>
// CHECK-NEXT:   %20 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:   %21 = stablehlo.multiply %5, %20 : tensor<4xi64>
// CHECK-NEXT:   %22 = stablehlo.broadcast_in_dim %19, dims = [0] : (tensor<3xi64>) -> tensor<3x4xi64>
// CHECK-NEXT:   %23 = stablehlo.broadcast_in_dim %21, dims = [1] : (tensor<4xi64>) -> tensor<3x4xi64>
// CHECK-NEXT:   %24 = stablehlo.add %22, %23 : tensor<3x4xi64>
// CHECK-NEXT:   %25 = stablehlo.broadcast_in_dim %24, dims = [0, 1] : (tensor<3x4xi64>) -> tensor<3x4x4xi64>
// CHECK-NEXT:   %26 = stablehlo.broadcast_in_dim %2, dims = [2] : (tensor<4xi64>) -> tensor<3x4x4xi64>
// CHECK-NEXT:   %27 = stablehlo.add %25, %26 : tensor<3x4x4xi64>
// CHECK-NEXT:   %28 = stablehlo.reshape %27 : (tensor<3x4x4xi64>) -> tensor<3x4x4x1xi64>
// CHECK-NEXT:   %29 = "stablehlo.gather"(%arg0, %28) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 3>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<48xf64>, tensor<3x4x4x1xi64>) -> tensor<3x4x4xf64>
// CHECK-NEXT:   %30 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i64>) -> tensor<3xi64>
// CHECK-NEXT:   %31 = stablehlo.add %8, %30 : tensor<3xi64>
// CHECK-NEXT:   %32 = stablehlo.compare EQ, %31, %c_6 : (tensor<3xi64>, tensor<3xi64>) -> tensor<3xi1>
// CHECK-NEXT:   %33 = stablehlo.broadcast_in_dim %32, dims = [1] : (tensor<3xi1>) -> tensor<4x3xi1>
// CHECK-NEXT:   %34 = stablehlo.and %17, %33 : tensor<4x3xi1>
// CHECK-NEXT:   %35 = stablehlo.broadcast_in_dim %34, dims = [0, 1] : (tensor<4x3xi1>) -> tensor<4x3x4xi1>
// CHECK-NEXT:   %36 = stablehlo.broadcast_in_dim %29, dims = [1, 0, 2] : (tensor<3x4x4xf64>) -> tensor<4x3x4xf64>
// CHECK-NEXT:   %37:2 = stablehlo.reduce(%36 init: %cst), (%35 init: %c) across dimensions = [1] : (tensor<4x3x4xf64>, tensor<4x3x4xi1>, tensor<f64>, tensor<i1>) -> (tensor<4x4xf64>, tensor<4x4xi1>)
// CHECK-NEXT:    reducer(%arg2: tensor<f64>, %arg4: tensor<f64>) (%arg3: tensor<i1>, %arg5: tensor<i1>)  {
// CHECK-NEXT:     %44 = stablehlo.select %arg3, %arg2, %arg4 : tensor<i1>, tensor<f64>
// CHECK-NEXT:     %45 = stablehlo.or %arg3, %arg5 : tensor<i1>
// CHECK-NEXT:     stablehlo.return %44, %45 : tensor<f64>, tensor<i1>
// CHECK-NEXT:   }
// CHECK-NEXT:   %38 = stablehlo.reduce(%34 init: %c) applies stablehlo.or across dimensions = [1] : (tensor<4x3xi1>, tensor<i1>) -> tensor<4xi1>
// CHECK-NEXT:   %39 = stablehlo.reshape %arg1 : (tensor<16xf64>) -> tensor<4x4xf64>
// CHECK-NEXT:   %40 = stablehlo.broadcast_in_dim %38, dims = [0] : (tensor<4xi1>) -> tensor<4x4xi1>
// CHECK-NEXT:   %41 = stablehlo.select %40, %37#0, %39 : tensor<4x4xi1>, tensor<4x4xf64>
// CHECK-NEXT:   %42 = stablehlo.reshape %41 : (tensor<4x4xf64>) -> tensor<16xf64>
// CHECK-NEXT:   %43 = stablehlo.dynamic_update_slice %arg1, %42, %c_0 : (tensor<16xf64>, tensor<16xf64>, tensor<i64>) -> tensor<16xf64>
// CHECK-NEXT:   return %arg0, %43 : tensor<48xf64>, tensor<16xf64>
// CHECK-NEXT: }
