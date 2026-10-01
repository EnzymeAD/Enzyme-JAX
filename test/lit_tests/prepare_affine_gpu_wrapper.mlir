// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(prepare-affine-gpu-wrapper)" | FileCheck %s

// A 300x256 1-d launch whose loop was split into 50x16x16x3x2: the grid
// dimensions 50, 3, 2 and the block dimensions 16, 16 are regrouped under the
// wrapper's bounds.
func.func @split(%m: memref<50x16x16x3x2xf32, 1>) {
  %c1 = arith.constant 1 : index
  %c256 = arith.constant 256 : index
  %c300 = arith.constant 300 : index
  %0 = "enzymexla.gpu_wrapper"(%c300, %c1, %c1, %c256, %c1, %c1) ({
    affine.parallel (%a, %b, %c, %d, %e) = (0, 0, 0, 0, 0) to (50, 16, 16, 3, 2) {
      %v = affine.load %m[%a, %b, %c, %d, %e] : memref<50x16x16x3x2xf32, 1>
      %w = arith.addf %v, %v : f32
      affine.store %w, %m[%a, %b, %c, %d, %e] : memref<50x16x16x3x2xf32, 1>
    }
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  return
}

// CHECK-LABEL: func.func @split
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C256:.+]] = arith.constant 256 : index
// CHECK-DAG:     %[[C300:.+]] = arith.constant 300 : index
// CHECK:         "enzymexla.gpu_wrapper"(%[[C300]], %[[C1]], %[[C1]], %[[C256]], %[[C1]], %[[C1]])
// CHECK:           affine.parallel (%[[G:.+]], %{{.+}}, %{{.+}}, %[[T:.+]], %{{.+}}, %{{.+}}) = (0, 0, 0, 0, 0, 0) to (symbol(%[[C300]]), symbol(%[[C1]]), symbol(%[[C1]]), symbol(%[[C256]]), symbol(%[[C1]]), symbol(%[[C1]]))
// CHECK-DAG:       %[[A:.+]] = affine.apply #{{.+}}(%[[G]])
// CHECK-DAG:       %[[B:.+]] = affine.apply #{{.+}}(%[[T]])
// CHECK-DAG:       %[[C:.+]] = affine.apply #{{.+}}(%[[T]])
// CHECK-DAG:       %[[D:.+]] = affine.apply #{{.+}}(%[[G]])
// CHECK-DAG:       %[[E:.+]] = affine.apply #{{.+}}(%[[G]])
// CHECK:           affine.load %{{.+}}[%[[A]], %[[B]], %[[C]], %[[D]], %[[E]]]

// Already aligned: left alone.
func.func @aligned(%m: memref<300x256xf32, 1>) {
  %c1 = arith.constant 1 : index
  %c256 = arith.constant 256 : index
  %c300 = arith.constant 300 : index
  %0 = "enzymexla.gpu_wrapper"(%c300, %c1, %c1, %c256, %c1, %c1) ({
    affine.parallel (%a, %b, %c, %d, %e, %f) = (0, 0, 0, 0, 0, 0) to (symbol(%c300), symbol(%c1), symbol(%c1), symbol(%c256), symbol(%c1), symbol(%c1)) {
      %v = affine.load %m[%a, %d] : memref<300x256xf32, 1>
      affine.store %v, %m[%a, %d] : memref<300x256xf32, 1>
    }
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  return
}

// CHECK-LABEL: func.func @aligned
// CHECK:         affine.parallel
// CHECK-NOT:     affine.apply

// Extents that do not factor into the launch: left alone.
func.func @mismatch(%m: memref<7xf32, 1>) {
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %0 = "enzymexla.gpu_wrapper"(%c8, %c1, %c1, %c1, %c1, %c1) ({
    affine.parallel (%a) = (0) to (7) {
      %v = affine.load %m[%a] : memref<7xf32, 1>
      affine.store %v, %m[%a] : memref<7xf32, 1>
    }
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  return
}

// CHECK-LABEL: func.func @mismatch
// CHECK:         affine.parallel (%{{.+}}) = (0) to (7)
