// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

// CHECK: #set = affine_set<()[s0] : (s0 >= 0)>

// The row width nd is computed in a conditional of the function, above the
// kernel's affine scope: a symbol of the kernel, though not a valid symbol
// where it is defined. The inner loop's iterations write elements of their
// own all the same, and as the rows are indexed i + e * nd, the two loops
// merge into one over ne * nd, run where nd is not negative: were both
// negative, the nest would run nothing and the product would be positive.
func.func @add_rows(%x: memref<?xf64>, %y: memref<?xf64>, %sizes: memref<2xi32>, %go: i1) {
  %c1 = arith.constant 1 : index
  %c256 = arith.constant 256 : index
  scf.if %go {
    %ne32 = memref.load %sizes[%c1] : memref<2xi32>
    %c0 = arith.constant 0 : index
    %nd32 = memref.load %sizes[%c0] : memref<2xi32>
    %ne = arith.index_cast %ne32 : i32 to index
    %nd = arith.index_cast %nd32 : i32 to index
    %w = "enzymexla.gpu_wrapper"(%ne, %c1, %c1, %c256, %c1, %c1) ({
      affine.parallel (%e) = (0) to (symbol(%ne)) {
        affine.for %i = 0 to %nd {
          %a = affine.load %x[%i + %e * symbol(%nd)] : memref<?xf64>
          %b = affine.load %y[%i + %e * symbol(%nd)] : memref<?xf64>
          %s = arith.addf %a, %b : f64
          affine.store %s, %y[%i + %e * symbol(%nd)] : memref<?xf64>
        }
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
  }
  return
}

// CHECK:  func.func @add_rows(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<2xi32>, %arg3: i1) {
// CHECK-NEXT:   %c1 = arith.constant 1 : index
// CHECK-NEXT:   %c256 = arith.constant 256 : index
// CHECK-NEXT:   scf.if %arg3 {
// CHECK-NEXT:     %0 = affine.load %arg2[1] : memref<2xi32>
// CHECK-NEXT:     %1 = affine.load %arg2[0] : memref<2xi32>
// CHECK-NEXT:     %2 = arith.index_cast %0 : i32 to index
// CHECK-NEXT:     %3 = arith.index_cast %1 : i32 to index
// CHECK-NEXT:     %4 = "enzymexla.gpu_wrapper"(%2, %c1, %c1, %c256, %c1, %c1) ({
// CHECK-NEXT:       affine.if #set()[%3] {
// CHECK-NEXT:         affine.parallel (%arg4) = (0) to (symbol(%2) * symbol(%3)) {
// CHECK-NEXT:           %5 = affine.load %arg0[%arg4] : memref<?xf64>
// CHECK-NEXT:           %6 = affine.load %arg1[%arg4] : memref<?xf64>
// CHECK-NEXT:           %7 = arith.addf %5, %6 : f64
// CHECK-NEXT:           affine.store %7, %arg1[%arg4] : memref<?xf64>
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:       "enzymexla.polygeist_yield"() : () -> ()
// CHECK-NEXT:     }) : (index, index, index, index, index, index) -> index
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// The kernel is launched in a loop of the host: the dependence analysis
// counts the loops of the kernel's affine scope only, which is where its
// accesses' domains stop.
func.func @launch_loop(%x: memref<?xf64>, %y: memref<?xf64>, %n: index, %ne: index, %nd: index) {
  %c1 = arith.constant 1 : index
  %c256 = arith.constant 256 : index
  affine.for %k = 0 to %n {
    %w = "enzymexla.gpu_wrapper"(%ne, %c1, %c1, %c256, %c1, %c1) ({
      affine.parallel (%e) = (0) to (symbol(%ne)) {
        affine.for %i = 0 to %nd {
          %a = affine.load %x[%i + %e * symbol(%nd)] : memref<?xf64>
          %b = affine.load %y[%i + %e * symbol(%nd)] : memref<?xf64>
          %s = arith.addf %a, %b : f64
          affine.store %s, %y[%i + %e * symbol(%nd)] : memref<?xf64>
        }
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
  }
  return
}

// CHECK:  func.func @launch_loop(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: index, %arg3: index, %arg4: index) {
// CHECK-NEXT:   %c1 = arith.constant 1 : index
// CHECK-NEXT:   %c256 = arith.constant 256 : index
// CHECK-NEXT:   affine.for %arg5 = 0 to %arg2 {
// CHECK-NEXT:     %0 = "enzymexla.gpu_wrapper"(%arg3, %c1, %c1, %c256, %c1, %c1) ({
// CHECK-NEXT:       affine.if #set()[%arg4] {
// CHECK-NEXT:         affine.parallel (%arg6) = (0) to (symbol(%arg3) * symbol(%arg4)) {
// CHECK-NEXT:           %1 = affine.load %arg0[%arg6] : memref<?xf64>
// CHECK-NEXT:           %2 = affine.load %arg1[%arg6] : memref<?xf64>
// CHECK-NEXT:           %3 = arith.addf %1, %2 : f64
// CHECK-NEXT:           affine.store %3, %arg1[%arg6] : memref<?xf64>
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:       "enzymexla.polygeist_yield"() : () -> ()
// CHECK-NEXT:     }) : (index, index, index, index, index, index) -> index
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// The rows are nd wide but laid out ns apart: the loops do not merge.
func.func @strided_rows(%x: memref<?xf64>, %ne: index, %nd: index, %ns: index) {
  %c1 = arith.constant 1 : index
  %c256 = arith.constant 256 : index
  %w = "enzymexla.gpu_wrapper"(%ne, %c1, %c1, %c256, %c1, %c1) ({
    affine.parallel (%e) = (0) to (symbol(%ne)) {
      affine.parallel (%i) = (0) to (symbol(%nd)) {
        %a = affine.load %x[%i + %e * symbol(%ns)] : memref<?xf64>
        %s = arith.addf %a, %a : f64
        affine.store %s, %x[%i + %e * symbol(%ns)] : memref<?xf64>
      }
    }
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  return
}

// CHECK:  func.func @strided_rows(%arg0: memref<?xf64>, %arg1: index, %arg2: index, %arg3: index) {
// CHECK-NEXT:   %c1 = arith.constant 1 : index
// CHECK-NEXT:   %c256 = arith.constant 256 : index
// CHECK-NEXT:   %0 = "enzymexla.gpu_wrapper"(%arg1, %c1, %c1, %c256, %c1, %c1) ({
// CHECK-NEXT:     affine.parallel (%arg4, %arg5) = (0, 0) to (symbol(%arg1), symbol(%arg2)) {
// CHECK-NEXT:       %1 = affine.load %arg0[%arg5 + %arg4 * symbol(%arg3)] : memref<?xf64>
// CHECK-NEXT:       %2 = arith.addf %1, %1 : f64
// CHECK-NEXT:       affine.store %2, %arg0[%arg5 + %arg4 * symbol(%arg3)] : memref<?xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:     "enzymexla.polygeist_yield"() : () -> ()
// CHECK-NEXT:   }) : (index, index, index, index, index, index) -> index
// CHECK-NEXT:   return
// CHECK-NEXT: }
