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

// CHECK-LABEL: func.func private @unsigned_raised(
// CHECK-NOT: stablehlo.negate
// CHECK: return

// CHECK-LABEL: func.func private @signed_raised(
// CHECK: %[[CMP:.+]] = arith.cmpi eq
// CHECK: %[[CV:.+]] = stablehlo.convert %[[CMP]] : (tensor<4xi1>) -> tensor<4xi64>
// CHECK-NEXT: stablehlo.negate %[[CV]] : tensor<4xi64>
