// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=elementwise_slice_to_batch" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file %s | FileCheck %s

// Three pieces of one value that do not follow one another, each converted.
// Stacking them takes a concatenate, which the simplifications take apart
// again and hand the ops back to this pattern; the group is left alone.
func.func @scattered(%x: tensor<9xi32>) -> (tensor<f64>, tensor<f64>, tensor<f64>) {
  %a = stablehlo.slice %x [1:2] : (tensor<9xi32>) -> tensor<1xi32>
  %b = stablehlo.slice %x [6:7] : (tensor<9xi32>) -> tensor<1xi32>
  %c = stablehlo.slice %x [8:9] : (tensor<9xi32>) -> tensor<1xi32>
  %ca = stablehlo.convert %a : (tensor<1xi32>) -> tensor<1xf64>
  %cb = stablehlo.convert %b : (tensor<1xi32>) -> tensor<1xf64>
  %cc = stablehlo.convert %c : (tensor<1xi32>) -> tensor<1xf64>
  %ra = stablehlo.reshape %ca : (tensor<1xf64>) -> tensor<f64>
  %rb = stablehlo.reshape %cb : (tensor<1xf64>) -> tensor<f64>
  %rc = stablehlo.reshape %cc : (tensor<1xf64>) -> tensor<f64>
  return %ra, %rb, %rc : tensor<f64>, tensor<f64>, tensor<f64>
}

// CHECK:  func.func @scattered(%arg0: tensor<9xi32>) -> (tensor<f64>, tensor<f64>, tensor<f64>) {
// CHECK-NEXT:   %0 = stablehlo.slice %arg0 [1:2] : (tensor<9xi32>) -> tensor<1xi32>
// CHECK-NEXT:   %1 = stablehlo.slice %arg0 [6:7] : (tensor<9xi32>) -> tensor<1xi32>
// CHECK-NEXT:   %2 = stablehlo.slice %arg0 [8:9] : (tensor<9xi32>) -> tensor<1xi32>
// CHECK-NEXT:   %3 = stablehlo.convert %0 : (tensor<1xi32>) -> tensor<1xf64>
// CHECK-NEXT:   %4 = stablehlo.convert %1 : (tensor<1xi32>) -> tensor<1xf64>
// CHECK-NEXT:   %5 = stablehlo.convert %2 : (tensor<1xi32>) -> tensor<1xf64>
// CHECK-NEXT:   %6 = stablehlo.reshape %3 : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:   %7 = stablehlo.reshape %4 : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:   %8 = stablehlo.reshape %5 : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:   return %6, %7, %8 : tensor<f64>, tensor<f64>, tensor<f64>
// CHECK-NEXT: }

// -----

// The same three pieces following one another merge into one slice, and the
// three converts become one.
func.func @contiguous(%x: tensor<9xi32>) -> (tensor<f64>, tensor<f64>, tensor<f64>) {
  %a = stablehlo.slice %x [1:2] : (tensor<9xi32>) -> tensor<1xi32>
  %b = stablehlo.slice %x [2:3] : (tensor<9xi32>) -> tensor<1xi32>
  %c = stablehlo.slice %x [3:4] : (tensor<9xi32>) -> tensor<1xi32>
  %ca = stablehlo.convert %a : (tensor<1xi32>) -> tensor<1xf64>
  %cb = stablehlo.convert %b : (tensor<1xi32>) -> tensor<1xf64>
  %cc = stablehlo.convert %c : (tensor<1xi32>) -> tensor<1xf64>
  %ra = stablehlo.reshape %ca : (tensor<1xf64>) -> tensor<f64>
  %rb = stablehlo.reshape %cb : (tensor<1xf64>) -> tensor<f64>
  %rc = stablehlo.reshape %cc : (tensor<1xf64>) -> tensor<f64>
  return %ra, %rb, %rc : tensor<f64>, tensor<f64>, tensor<f64>
}

// CHECK:  func.func @contiguous(%arg0: tensor<9xi32>) -> (tensor<f64>, tensor<f64>, tensor<f64>) {
// CHECK-NEXT:   %0 = stablehlo.slice %arg0 [1:4] : (tensor<9xi32>) -> tensor<3xi32>
// CHECK-NEXT:   %1 = stablehlo.reshape %0 : (tensor<3xi32>) -> tensor<3x1xi32>
// CHECK-NEXT:   %2 = stablehlo.convert %1 : (tensor<3x1xi32>) -> tensor<3x1xf64>
// CHECK-NEXT:   %3 = stablehlo.slice %2 [0:1, 0:1] : (tensor<3x1xf64>) -> tensor<1x1xf64>
// CHECK-NEXT:   %4 = stablehlo.reshape %3 : (tensor<1x1xf64>) -> tensor<1xf64>
// CHECK-NEXT:   %5 = stablehlo.slice %2 [1:2, 0:1] : (tensor<3x1xf64>) -> tensor<1x1xf64>
// CHECK-NEXT:   %6 = stablehlo.reshape %5 : (tensor<1x1xf64>) -> tensor<1xf64>
// CHECK-NEXT:   %7 = stablehlo.slice %2 [2:3, 0:1] : (tensor<3x1xf64>) -> tensor<1x1xf64>
// CHECK-NEXT:   %8 = stablehlo.reshape %7 : (tensor<1x1xf64>) -> tensor<1xf64>
// CHECK-NEXT:   %9 = stablehlo.reshape %4 : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:   %10 = stablehlo.reshape %6 : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:   %11 = stablehlo.reshape %8 : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:   return %9, %10, %11 : tensor<f64>, tensor<f64>, tensor<f64>
// CHECK-NEXT: }
