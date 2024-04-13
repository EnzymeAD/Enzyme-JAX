// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

  func.func @t1(%arg0: tensor<1x30x1x10xbf16>, %arg1: tensor<1x30x1x10xbf16>) -> tensor<1x30x1x100xbf16> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1 = stablehlo.pad %arg1, %0, low = [0, 0, 0, 80], high = [0, 0, 0, 0], interior = [0, 0, 0, 0] : (tensor<1x30x1x10xbf16>, tensor<bf16>) -> tensor<1x30x1x90xbf16>
    %2 = stablehlo.concatenate %1, %arg0, dim = 3 : (tensor<1x30x1x90xbf16>, tensor<1x30x1x10xbf16>) -> tensor<1x30x1x100xbf16>
    return %2 : tensor<1x30x1x100xbf16>
  }

// CHECK:  func.func @t1(%arg0: tensor<1x30x1x10xbf16>, %arg1: tensor<1x30x1x10xbf16>) -> tensor<1x30x1x100xbf16> {
// CHECK-NEXT:    %0 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
// CHECK-NEXT:    %1 = stablehlo.concatenate %arg1, %arg0, dim = 3 : (tensor<1x30x1x10xbf16>, tensor<1x30x1x10xbf16>) -> tensor<1x30x1x20xbf16>
// CHECK-NEXT:    %2 = stablehlo.pad %1, %0, low = [0, 0, 0, 80], high = [0, 0, 0, 0], interior = [0, 0, 0, 0] : (tensor<1x30x1x20xbf16>, tensor<bf16>) -> tensor<1x30x1x100xbf16>
// CHECK-NEXT:    return %2 : tensor<1x30x1x100xbf16>
// CHECK-NEXT:  }

  func.func @t2(%arg0: tensor<1x30x1x10xbf16>, %arg1: tensor<1x30x1x10xbf16>) -> tensor<1x30x1x100xbf16> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1 = stablehlo.pad %arg1, %0, low = [0, 0, 0, 0], high = [0, 0, 0, 80], interior = [0, 0, 0, 0] : (tensor<1x30x1x10xbf16>, tensor<bf16>) -> tensor<1x30x1x90xbf16>
    %2 = stablehlo.concatenate %1, %arg0, dim = 3 : (tensor<1x30x1x90xbf16>, tensor<1x30x1x10xbf16>) -> tensor<1x30x1x100xbf16>
    return %2 : tensor<1x30x1x100xbf16>
  }

// doesn't do if wrong side
// CHECK:  func.func @t2(%arg0: tensor<1x30x1x10xbf16>, %arg1: tensor<1x30x1x10xbf16>) -> tensor<1x30x1x100xbf16> {
// CHECK-NEXT:    %0 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
// CHECK-NEXT:    %1 = stablehlo.pad %arg1, %0, low = [0, 0, 0, 0], high = [0, 0, 0, 80], interior = [0, 0, 0, 0] : (tensor<1x30x1x10xbf16>, tensor<bf16>) -> tensor<1x30x1x90xbf16>
// CHECK-NEXT:    %2 = stablehlo.concatenate %1, %arg0, dim = 3 : (tensor<1x30x1x90xbf16>, tensor<1x30x1x10xbf16>) -> tensor<1x30x1x100xbf16>
// CHECK-NEXT:    return %2 : tensor<1x30x1x100xbf16>
// CHECK-NEXT:  }

  func.func @t3(%arg0: tensor<1x30x1x10xbf16>, %arg1: tensor<1x30x1x10xbf16>) -> tensor<1x30x1x100xbf16> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1 = stablehlo.pad %arg1, %0, low = [0, 0, 0, 0], high = [0, 0, 0, 80], interior = [0, 0, 0, 0] : (tensor<1x30x1x10xbf16>, tensor<bf16>) -> tensor<1x30x1x90xbf16>
    %2 = stablehlo.concatenate %arg0, %1, dim = 3 : (tensor<1x30x1x10xbf16>, tensor<1x30x1x90xbf16>) -> tensor<1x30x1x100xbf16>
    return %2 : tensor<1x30x1x100xbf16>
  }

// CHECK:  func.func @t3(%arg0: tensor<1x30x1x10xbf16>, %arg1: tensor<1x30x1x10xbf16>) -> tensor<1x30x1x100xbf16> {
// CHECK-NEXT:    %0 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
// CHECK-NEXT:    %1 = stablehlo.concatenate %arg0, %arg1, dim = 3 : (tensor<1x30x1x10xbf16>, tensor<1x30x1x10xbf16>) -> tensor<1x30x1x20xbf16>
// CHECK-NEXT:    %2 = stablehlo.pad %1, %0, low = [0, 0, 0, 0], high = [0, 0, 0, 80], interior = [0, 0, 0, 0] : (tensor<1x30x1x20xbf16>, tensor<bf16>) -> tensor<1x30x1x100xbf16>
// CHECK-NEXT:    return %2 : tensor<1x30x1x100xbf16>
// CHECK-NEXT:  }

  func.func @t4(%arg0: tensor<1x30x1x10xbf16>, %arg1: tensor<1x30x1x10xbf16>) -> tensor<1x30x1x100xbf16> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %1 = stablehlo.pad %arg1, %0, low = [0, 0, 0, 80], high = [0, 0, 0, 0], interior = [0, 0, 0, 0] : (tensor<1x30x1x10xbf16>, tensor<bf16>) -> tensor<1x30x1x90xbf16>
    %2 = stablehlo.concatenate %arg0, %1, dim = 3 : (tensor<1x30x1x10xbf16>, tensor<1x30x1x90xbf16>) -> tensor<1x30x1x100xbf16>
    return %2 : tensor<1x30x1x100xbf16>
  }

// doesn't do if wrong side
// CHECK:  func.func @t4(%arg0: tensor<1x30x1x10xbf16>, %arg1: tensor<1x30x1x10xbf16>) -> tensor<1x30x1x100xbf16> {
// CHECK-NEXT:    %0 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
// CHECK-NEXT:    %1 = stablehlo.pad %arg1, %0, low = [0, 0, 0, 80], high = [0, 0, 0, 0], interior = [0, 0, 0, 0] : (tensor<1x30x1x10xbf16>, tensor<bf16>) -> tensor<1x30x1x90xbf16>
// CHECK-NEXT:    %2 = stablehlo.concatenate %arg0, %1, dim = 3 : (tensor<1x30x1x10xbf16>, tensor<1x30x1x90xbf16>) -> tensor<1x30x1x100xbf16>
// CHECK-NEXT:    return %2 : tensor<1x30x1x100xbf16>
// CHECK-NEXT:  }

  func.func @c1(%arg0: tensor<1x30x1x10xbf16>) -> tensor<1x30x1x100xbf16> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %c1 = stablehlo.constant dense<0.000000e+00> : tensor<1x30x1x10xbf16>
    %1 = stablehlo.pad %arg0, %0, low = [0, 0, 0, 80], high = [0, 0, 0, 0], interior = [0, 0, 0, 0] : (tensor<1x30x1x10xbf16>, tensor<bf16>) -> tensor<1x30x1x90xbf16>
    %2 = stablehlo.concatenate %1, %c1, dim = 3 : (tensor<1x30x1x90xbf16>, tensor<1x30x1x10xbf16>) -> tensor<1x30x1x100xbf16>
    return %2 : tensor<1x30x1x100xbf16>
  }

// CHECK:  func.func @c1(%arg0: tensor<1x30x1x10xbf16>) -> tensor<1x30x1x100xbf16> {
// CHECK-NEXT:    %0 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
// CHECK-NEXT:    %1 = stablehlo.pad %arg0, %0, low = [0, 0, 0, 80], high = [0, 0, 0, 10], interior = [0, 0, 0, 0] : (tensor<1x30x1x10xbf16>, tensor<bf16>) -> tensor<1x30x1x100xbf16>
// CHECK-NEXT:    return %1 : tensor<1x30x1x100xbf16>
// CHECK-NEXT:  }

  func.func @c2(%771: tensor<1x30x1x25xbf16>) -> tensor<1x1x160x1x25xbf16> {
    %cst_200 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %cst_146 = stablehlo.constant dense<0.000000e+00> : tensor<1x1x80x1x25xbf16> 
	%773 = stablehlo.pad %771, %cst_200, low = [0, 0, 0, 0], high = [0, 50, 0, 0], interior = [0, 0, 0, 0] : (tensor<1x30x1x25xbf16>, tensor<bf16>) -> tensor<1x80x1x25xbf16>
	%774 = stablehlo.reshape %773 : (tensor<1x80x1x25xbf16>) -> tensor<1x1x80x1x25xbf16>
	%775 = stablehlo.concatenate %cst_146, %774, dim = 2 : (tensor<1x1x80x1x25xbf16>, tensor<1x1x80x1x25xbf16>) -> tensor<1x1x160x1x25xbf16>
    return %775 : tensor<1x1x160x1x25xbf16>
  }

// CHECK:  func.func @c2(%arg0: tensor<1x30x1x25xbf16>) -> tensor<1x1x160x1x25xbf16> {
// CHECK-NEXT:    %0 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
// CHECK-NEXT:    %1 = stablehlo.reshape %arg0 : (tensor<1x30x1x25xbf16>) -> tensor<1x1x30x1x25xbf16>
// CHECK-NEXT:    %2 = stablehlo.pad %1, %0, low = [0, 0, 80, 0, 0], high = [0, 0, 50, 0, 0], interior = [0, 0, 0, 0, 0] : (tensor<1x1x30x1x25xbf16>, tensor<bf16>) -> tensor<1x1x160x1x25xbf16>
// CHECK-NEXT:    return %2 : tensor<1x1x160x1x25xbf16>
// CHECK-NEXT:  }
