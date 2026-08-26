// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --split-input-file | FileCheck %s

// Scratch conditionally written under an affine.if: lanes the guard covers
// see the stored value, the rest read the zero initialization.
func.func @if_alloca(%out: memref<10xf64, 1>) {
  %tmp = memref.alloca() : memref<10xf64>
  affine.parallel (%i) = (0) to (10) {
    affine.if affine_set<(d0) : (d0 - 5 >= 0)>(%i) {
      %c = arith.constant 3.0 : f64
      affine.store %c, %tmp[%i] : memref<10xf64>
    }
    %v = affine.load %tmp[%i] : memref<10xf64>
    affine.store %v, %out[%i] : memref<10xf64, 1>
  }
  return
}

// CHECK-LABEL: func.func private @if_alloca_raised(
// CHECK-DAG: stablehlo.constant dense<3.000000e+00> : tensor<f64>
// CHECK-DAG: stablehlo.constant dense<0.000000e+00> : tensor<10xf64>
// CHECK: %[[MASK:.+]] = stablehlo.compare GE, %{{.+}}, %{{.+}} : (tensor<10xi64>, tensor<10xi64>) -> tensor<10xi1>
// CHECK: stablehlo.select %[[MASK]], %{{.+}}, %{{.+}} : tensor<10xi1>, tensor<10xf64>

// -----

// Both branches write the scratch: the read after the if selects between the
// two stored values, the zero initialization is fully overwritten.
func.func @ifelse_alloca(%out: memref<10xf64, 1>) {
  %tmp = memref.alloca() : memref<10xf64>
  affine.parallel (%i) = (0) to (10) {
    affine.if affine_set<(d0) : (d0 - 5 >= 0)>(%i) {
      %c = arith.constant 1.0 : f64
      affine.store %c, %tmp[%i] : memref<10xf64>
    } else {
      %c = arith.constant 2.0 : f64
      affine.store %c, %tmp[%i] : memref<10xf64>
    }
    %v = affine.load %tmp[%i] : memref<10xf64>
    affine.store %v, %out[%i] : memref<10xf64, 1>
  }
  return
}

// CHECK-LABEL: func.func private @ifelse_alloca_raised(
// CHECK-DAG: stablehlo.constant dense<1.000000e+00> : tensor<f64>
// CHECK-DAG: stablehlo.constant dense<2.000000e+00> : tensor<f64>
// CHECK: %[[MASK2:.+]] = stablehlo.compare GE, %{{.+}}, %{{.+}} : (tensor<10xi64>, tensor<10xi64>) -> tensor<10xi1>
// CHECK: stablehlo.select %[[MASK2]], %{{.+}}, %{{.+}} : tensor<10xi1>, tensor<10xf64>

// -----

// Scratch written under a uniform scf.if: the condition broadcasts across
// the lanes and selects against the zero-initialized scratch.
func.func @scfif_alloca(%out: memref<10xf64, 1>, %nbuf: memref<i64, 1>) {
  %tmp = memref.alloca() : memref<10xf64>
  %n = affine.load %nbuf[] : memref<i64, 1>
  %c0 = arith.constant 0 : i64
  %cond = arith.cmpi sgt, %n, %c0 : i64
  affine.parallel (%i) = (0) to (10) {
    scf.if %cond {
      %c = arith.constant 3.0 : f64
      affine.store %c, %tmp[%i] : memref<10xf64>
    }
    %v = affine.load %tmp[%i] : memref<10xf64>
    affine.store %v, %out[%i] : memref<10xf64, 1>
  }
  return
}

// CHECK-LABEL: func.func private @scfif_alloca_raised(
// CHECK-DAG: %[[ZERO3:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<10xf64>
// CHECK: %[[COND:.+]] = arith.cmpi sgt, %{{.+}}, %{{.+}} : tensor<i64>
// CHECK: %[[BCAST:.+]] = stablehlo.broadcast_in_dim %[[COND]], dims = [] : (tensor<i1>) -> tensor<10xi1>
// CHECK: stablehlo.select %[[BCAST]], %{{.+}}, %{{.+}} : tensor<10xi1>, tensor<10xf64>
