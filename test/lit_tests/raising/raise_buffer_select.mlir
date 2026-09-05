// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --split-input-file | FileCheck %s

// A uniform branch choosing between two read-only buffers expands per access
// and raises as a select of the gathered values.
func.func @bufsel(%a: memref<100xf64, 1>, %b: memref<100xf64, 1>, %out: memref<100xf64, 1>, %flagbuf: memref<i64, 1>) {
  %f = affine.load %flagbuf[] : memref<i64, 1>
  %fi = arith.index_cast %f : i64 to index
  affine.parallel (%i) = (0) to (100) {
    %buf = affine.if affine_set<()[s0] : (s0 - 1 >= 0)>()[%fi] -> memref<100xf64, 1> {
      affine.yield %a : memref<100xf64, 1>
    } else {
      affine.yield %b : memref<100xf64, 1>
    }
    %v = affine.load %buf[%i] : memref<100xf64, 1>
    affine.store %v, %out[%i] : memref<100xf64, 1>
  }
  return
}

// CHECK-LABEL: func.func private @bufsel_raised(
// CHECK: stablehlo.select %{{.+}}, %{{.+}}, %{{.+}} : tensor<100xi1>, tensor<100xf64>

