// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --split-input-file | FileCheck %s

// An affine.if may reference symbols: a runtime scalar in the set becomes a
// broadcast lane mask and the guarded store a select.
func.func @symguard(%out: memref<10xf64, 1>, %nbuf: memref<i64, 1>) {
  %n = affine.load %nbuf[] : memref<i64, 1>
  %ni = arith.index_cast %n : i64 to index
  affine.parallel (%i) = (0) to (10) {
    affine.if affine_set<(d0)[s0] : (s0 - d0 - 1 >= 0)>(%i)[%ni] {
      %c = arith.constant 2.0 : f64
      affine.store %c, %out[%i] : memref<10xf64, 1>
    }
  }
  return
}

// CHECK-LABEL: func.func private @symguard_raised(
// CHECK: %[[MASK:.+]] = stablehlo.compare GE, %{{.+}}, %{{.+}} : (tensor<10xi64>, tensor<10xi64>) -> tensor<10xi1>
// CHECK: stablehlo.select %[[MASK]], %{{.+}}, %{{.+}} : tensor<10xi1>, tensor<10xf64>

// -----

// The same guard inside a loop raised as a while: the symbol condition
// raises within the while body.
func.func @symloop(%out: memref<10xf64, 1>, %nbuf: memref<i64, 1>) {
  %n = affine.load %nbuf[] : memref<i64, 1>
  %ni = arith.index_cast %n : i64 to index
  affine.for %t = 0 to 4 {
    affine.parallel (%i) = (0) to (10) {
      affine.if affine_set<(d0)[s0] : (s0 - d0 - 1 >= 0)>(%i)[%ni] {
        %v = affine.load %out[%i] : memref<10xf64, 1>
        %c = arith.constant 1.0 : f64
        %s = arith.addf %v, %c : f64
        affine.store %s, %out[%i] : memref<10xf64, 1>
      }
    }
  }
  return
}

// CHECK-LABEL: func.func private @symloop_raised(
// CHECK: stablehlo.while
// CHECK: } do {
// CHECK: %[[M:.+]] = stablehlo.compare GE, %{{.+}}, %{{.+}} : (tensor<10xi64>, tensor<10xi64>) -> tensor<10xi1>
// CHECK: stablehlo.select %[[M]], %{{.+}}, %{{.+}} : tensor<10xi1>, tensor<10xf64>
