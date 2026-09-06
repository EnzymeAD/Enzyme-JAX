// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo | FileCheck %s

// arith.index_cast of an i1 sign-extends (true -> -1), unlike index_castui
// (true -> 1). stablehlo.convert reads a boolean as 0/1, so the signed cast
// negates the conversion, as the arith raising already does for extsi.
// (mfem's H(curl)/H(div) mass kernels offset an index by `- flag` this way;
// raising it as +1 broke "Hcurl/Hdiv PA Coefficient" once the cast was no
// longer hoisted out of the kernel.)
func.func @signed(%out: memref<4xf64, 1>) {
  %c2 = arith.constant 2 : index
  affine.parallel (%i) = (0) to (4) {
    %b = arith.cmpi eq, %i, %c2 : index
    %s = arith.index_cast %b : i1 to index
    %si = arith.index_cast %s : index to i64
    %f = arith.sitofp %si : i64 to f64
    affine.store %f, %out[%i] : memref<4xf64, 1>
  }
  return
}

func.func @unsigned(%out: memref<4xf64, 1>) {
  %c2 = arith.constant 2 : index
  affine.parallel (%i) = (0) to (4) {
    %b = arith.cmpi eq, %i, %c2 : index
    %u = arith.index_castui %b : i1 to index
    %ui = arith.index_cast %u : index to i64
    %f = arith.sitofp %ui : i64 to f64
    affine.store %f, %out[%i] : memref<4xf64, 1>
  }
  return
}

// CHECK:    func.func private @unsigned_raised(%arg0: tensor<4xf64>) -> tensor<4xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<4xi64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<4xi64>
// CHECK-NEXT:    %1 = stablehlo.add %0, %c_0 : tensor<4xi64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<1> : tensor<4xi64>
// CHECK-NEXT:    %2 = stablehlo.multiply %1, %c_1 : tensor<4xi64>
// CHECK-NEXT:    %3 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:    %4 = arith.cmpi eq, %2, %3 : tensor<4xi64>
// CHECK-NEXT:    %5 = stablehlo.convert %4 : (tensor<4xi1>) -> tensor<4xi64>
// CHECK-NEXT:    %6 = arith.sitofp %5 : tensor<4xi64> to tensor<4xf64>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_4 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_5 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_6 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_7 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %7 = stablehlo.broadcast_in_dim %6, dims = [0] : (tensor<4xf64>) -> tensor<4xf64>
// CHECK-NEXT:    %8 = stablehlo.dynamic_update_slice %arg0, %7, %c_7 : (tensor<4xf64>, tensor<4xf64>, tensor<i64>) -> tensor<4xf64>
// CHECK-NEXT:    return %8 : tensor<4xf64>
// CHECK-NEXT:  }

// CHECK:    func.func private @signed_raised(%arg0: tensor<4xf64>) -> tensor<4xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<4xi64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<4xi64>
// CHECK-NEXT:    %1 = stablehlo.add %0, %c_0 : tensor<4xi64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<1> : tensor<4xi64>
// CHECK-NEXT:    %2 = stablehlo.multiply %1, %c_1 : tensor<4xi64>
// CHECK-NEXT:    %3 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:    %4 = arith.cmpi eq, %2, %3 : tensor<4xi64>
// CHECK-NEXT:    %5 = stablehlo.convert %4 : (tensor<4xi1>) -> tensor<4xi64>
// CHECK-NEXT:    %6 = stablehlo.negate %5 : tensor<4xi64>
// CHECK-NEXT:    %7 = arith.sitofp %6 : tensor<4xi64> to tensor<4xf64>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_4 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_5 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_6 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_7 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %8 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<4xf64>) -> tensor<4xf64>
// CHECK-NEXT:    %9 = stablehlo.dynamic_update_slice %arg0, %8, %c_7 : (tensor<4xf64>, tensor<4xf64>, tensor<i64>) -> tensor<4xf64>
// CHECK-NEXT:    return %9 : tensor<4xf64>
// CHECK-NEXT:  }
