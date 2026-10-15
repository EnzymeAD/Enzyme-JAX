// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=transpose_elementwise(1)" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s --check-prefix=SINGLE-USER
// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=transpose_elementwise(0)" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s --check-prefix=AGG

module {
  func.func @main(%a : tensor<100x200x300xbf16>, %b: tensor<100x200x300xbf16>) -> tensor<300x100x200xbf16> {
    %1909 = stablehlo.subtract %a, %b : tensor<100x200x300xbf16>
    %1910 = stablehlo.transpose %1909, dims = [2, 0, 1] : (tensor<100x200x300xbf16>) -> tensor<300x100x200xbf16> 
    return %1910 : tensor<300x100x200xbf16> 
  }
  func.func @main2(%a : tensor<100x200x300xbf16> ) -> tensor<300x100x200xf32> {
    %1909 = stablehlo.convert %a : (tensor<100x200x300xbf16>) -> tensor<100x200x300xf32>
    %1910 = stablehlo.transpose %1909, dims = [2, 0, 1] : (tensor<100x200x300xf32>) -> tensor<300x100x200xf32> 
    return %1910 : tensor<300x100x200xf32> 
  }
  func.func @main3(%a : tensor<100x200x300xbf16>, %b: tensor<100x200x300xbf16>) -> (tensor<300x100x200xbf16>, tensor<100x200x300xbf16>) {
    %1909 = stablehlo.subtract %a, %b : tensor<100x200x300xbf16>
    %1910 = stablehlo.transpose %1909, dims = [2, 0, 1] : (tensor<100x200x300xbf16>) -> tensor<300x100x200xbf16>
    %1911 = stablehlo.negate %1909 : tensor<100x200x300xbf16>
    return %1910, %1911 : tensor<300x100x200xbf16>, tensor<100x200x300xbf16>
  }
  func.func @main4(%a : tensor<100xbf16>, %b: tensor<200xbf16>) -> tensor<300x100x200xbf16> {
    %0 = stablehlo.broadcast_in_dim %a, dims = [0] : (tensor<100xbf16>) -> tensor<100x200x300xbf16>
    %1 = stablehlo.broadcast_in_dim %b, dims = [1] : (tensor<200xbf16>) -> tensor<100x200x300xbf16>
    %1909 = stablehlo.subtract %0, %1 : tensor<100x200x300xbf16>
    %1910 = stablehlo.transpose %1909, dims = [2, 0, 1] : (tensor<100x200x300xbf16>) -> tensor<300x100x200xbf16>
    return %1910 : tensor<300x100x200xbf16>
  }
}

// The transpose of a difference of two arguments stays on the result (pushed
// onto the arguments it would copy both); of a convert of one it moves up; of
// a difference of two broadcasts it moves onto them, which take it for free.


// SINGLE-USER:  func.func @main(%arg0: tensor<100x200x300xbf16>, %arg1: tensor<100x200x300xbf16>) -> tensor<300x100x200xbf16> {
// SINGLE-USER-NEXT:    %0 = stablehlo.subtract %arg0, %arg1 : tensor<100x200x300xbf16>
// SINGLE-USER-NEXT:    %1 = stablehlo.transpose %0, dims = [2, 0, 1] : (tensor<100x200x300xbf16>) -> tensor<300x100x200xbf16>
// SINGLE-USER-NEXT:    return %1 : tensor<300x100x200xbf16>
// SINGLE-USER-NEXT:  }
// SINGLE-USER:  func.func @main2(%arg0: tensor<100x200x300xbf16>) -> tensor<300x100x200xf32> {
// SINGLE-USER-NEXT:    %0 = stablehlo.transpose %arg0, dims = [2, 0, 1] : (tensor<100x200x300xbf16>) -> tensor<300x100x200xbf16>
// SINGLE-USER-NEXT:    %1 = stablehlo.convert %0 : (tensor<300x100x200xbf16>) -> tensor<300x100x200xf32>
// SINGLE-USER-NEXT:    return %1 : tensor<300x100x200xf32>
// SINGLE-USER-NEXT:  }
// SINGLE-USER: func.func @main3(%arg0: tensor<100x200x300xbf16>, %arg1: tensor<100x200x300xbf16>) -> (tensor<300x100x200xbf16>, tensor<100x200x300xbf16>) {
// SINGLE-USER-NEXT:    %0 = stablehlo.subtract %arg0, %arg1 : tensor<100x200x300xbf16>
// SINGLE-USER-NEXT:    %1 = stablehlo.transpose %0, dims = [2, 0, 1] : (tensor<100x200x300xbf16>) -> tensor<300x100x200xbf16>
// SINGLE-USER-NEXT:    %2 = stablehlo.negate %0 : tensor<100x200x300xbf16>
// SINGLE-USER-NEXT:    return %1, %2 : tensor<300x100x200xbf16>, tensor<100x200x300xbf16>
// SINGLE-USER-NEXT:  }
// SINGLE-USER: func.func @main4(%arg0: tensor<100xbf16>, %arg1: tensor<200xbf16>) -> tensor<300x100x200xbf16> {
// SINGLE-USER-NEXT:    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<100xbf16>) -> tensor<100x200x300xbf16>
// SINGLE-USER-NEXT:    %[[b0:.+]] = stablehlo.transpose %0, dims = [2, 0, 1] : (tensor<100x200x300xbf16>) -> tensor<300x100x200xbf16>
// SINGLE-USER-NEXT:    %2 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<200xbf16>) -> tensor<100x200x300xbf16>
// SINGLE-USER-NEXT:    %[[b1:.+]] = stablehlo.transpose %2, dims = [2, 0, 1] : (tensor<100x200x300xbf16>) -> tensor<300x100x200xbf16>
// SINGLE-USER-NEXT:    %4 = stablehlo.subtract %[[b0]], %[[b1]] : tensor<300x100x200xbf16>
// SINGLE-USER-NEXT:    return %4 : tensor<300x100x200xbf16>
// SINGLE-USER-NEXT:  }

// AGG:  func.func @main(%arg0: tensor<100x200x300xbf16>, %arg1: tensor<100x200x300xbf16>) -> tensor<300x100x200xbf16> {
// AGG-NEXT:    %0 = stablehlo.subtract %arg0, %arg1 : tensor<100x200x300xbf16>
// AGG-NEXT:    %1 = stablehlo.transpose %0, dims = [2, 0, 1] : (tensor<100x200x300xbf16>) -> tensor<300x100x200xbf16>
// AGG-NEXT:    return %1 : tensor<300x100x200xbf16>
// AGG-NEXT:  }
// AGG:  func.func @main2(%arg0: tensor<100x200x300xbf16>) -> tensor<300x100x200xf32> {
// AGG-NEXT:    %0 = stablehlo.transpose %arg0, dims = [2, 0, 1] : (tensor<100x200x300xbf16>) -> tensor<300x100x200xbf16>
// AGG-NEXT:    %1 = stablehlo.convert %0 : (tensor<300x100x200xbf16>) -> tensor<300x100x200xf32>
// AGG-NEXT:    return %1 : tensor<300x100x200xf32>
// AGG-NEXT:  }
// AGG: func.func @main3(%arg0: tensor<100x200x300xbf16>, %arg1: tensor<100x200x300xbf16>) -> (tensor<300x100x200xbf16>, tensor<100x200x300xbf16>) {
// AGG-NEXT:    %0 = stablehlo.subtract %arg0, %arg1 : tensor<100x200x300xbf16>
// AGG-NEXT:    %1 = stablehlo.transpose %0, dims = [2, 0, 1] : (tensor<100x200x300xbf16>) -> tensor<300x100x200xbf16>
// AGG-NEXT:    %2 = stablehlo.negate %0 : tensor<100x200x300xbf16>
// AGG-NEXT:    return %1, %2 : tensor<300x100x200xbf16>, tensor<100x200x300xbf16>
// AGG-NEXT:  }
// AGG: func.func @main4(%arg0: tensor<100xbf16>, %arg1: tensor<200xbf16>) -> tensor<300x100x200xbf16> {
// AGG-NEXT:    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<100xbf16>) -> tensor<100x200x300xbf16>
// AGG-NEXT:    %[[b0:.+]] = stablehlo.transpose %0, dims = [2, 0, 1] : (tensor<100x200x300xbf16>) -> tensor<300x100x200xbf16>
// AGG-NEXT:    %2 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<200xbf16>) -> tensor<100x200x300xbf16>
// AGG-NEXT:    %[[b1:.+]] = stablehlo.transpose %2, dims = [2, 0, 1] : (tensor<100x200x300xbf16>) -> tensor<300x100x200xbf16>
// AGG-NEXT:    %4 = stablehlo.subtract %[[b0]], %[[b1]] : tensor<300x100x200xbf16>
// AGG-NEXT:    return %4 : tensor<300x100x200xbf16>
// AGG-NEXT:  }
