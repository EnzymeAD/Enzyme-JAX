// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @t1(%arg0: tensor<3x2xf64>, %arg1: tensor<3x2xf64>) -> tensor<3x2xf64> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<3x2xf64>) -> tensor<2x3xf64>
    %1 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<3x2xf64>) -> tensor<2x3xf64>
    %2 = stablehlo.multiply %0, %1 : tensor<2x3xf64>
    %3 = stablehlo.transpose %2, dims = [1, 0] : (tensor<2x3xf64>) -> tensor<3x2xf64>
    return %3 : tensor<3x2xf64>
}

// CHECK:  func.func @t1(%arg0: tensor<3x2xf64>, %arg1: tensor<3x2xf64>) -> tensor<3x2xf64> {
// CHECK-NEXT:    %0 = stablehlo.multiply %arg0, %arg1 : tensor<3x2xf64>
// CHECK-NEXT:    return %0 : tensor<3x2xf64>
// CHECK-NEXT:  }

func.func @t2(%arg0: tensor<3x2xf64>, %arg1: tensor<3x2xf64>) -> tensor<3x2xf64> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<3x2xf64>) -> tensor<2x3xf64>
    %1 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<3x2xf64>) -> tensor<2x3xf64>
    %2 = stablehlo.add %0, %1 : tensor<2x3xf64>
    %3 = stablehlo.transpose %2, dims = [1, 0] : (tensor<2x3xf64>) -> tensor<3x2xf64>
    return %3 : tensor<3x2xf64>
}

// CHECK:  func.func @t2(%arg0: tensor<3x2xf64>, %arg1: tensor<3x2xf64>) -> tensor<3x2xf64> {
// CHECK-NEXT:    %0 = stablehlo.add %arg0, %arg1 : tensor<3x2xf64>
// CHECK-NEXT:    return %0 : tensor<3x2xf64>
// CHECK-NEXT:  }

func.func @t4(%arg0: tensor<3x2xf64>, %arg1: tensor<3x2xf64>) -> tensor<3x2xf64> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<3x2xf64>) -> tensor<2x3xf64>
    %1 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<3x2xf64>) -> tensor<2x3xf64>
    %2 = stablehlo.subtract %0, %1 : tensor<2x3xf64>
    %3 = stablehlo.transpose %2, dims = [1, 0] : (tensor<2x3xf64>) -> tensor<3x2xf64>
    return %3 : tensor<3x2xf64>
}

// CHECK:  func.func @t4(%arg0: tensor<3x2xf64>, %arg1: tensor<3x2xf64>) -> tensor<3x2xf64> {
// CHECK-NEXT:    %0 = stablehlo.subtract %arg0, %arg1 : tensor<3x2xf64>
// CHECK-NEXT:    return %0 : tensor<3x2xf64>
// CHECK-NEXT:  }

func.func @t5(%arg0: tensor<3x2xf64>, %arg1: tensor<3x2xf64>) -> tensor<3x2xf64> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<3x2xf64>) -> tensor<2x3xf64>
    %1 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<3x2xf64>) -> tensor<2x3xf64>
    %2 = stablehlo.divide %0, %1 : tensor<2x3xf64>
    %3 = stablehlo.transpose %2, dims = [1, 0] : (tensor<2x3xf64>) -> tensor<3x2xf64>
    return %3 : tensor<3x2xf64>
}

// CHECK:  func.func @t5(%arg0: tensor<3x2xf64>, %arg1: tensor<3x2xf64>) -> tensor<3x2xf64> {
// CHECK-NEXT:    %0 = stablehlo.divide %arg0, %arg1 : tensor<3x2xf64>
// CHECK-NEXT:    return %0 : tensor<3x2xf64>
// CHECK-NEXT:  }

func.func @t7(%arg0: tensor<3x2xf64>, %arg1: tensor<3x2xf64>) -> tensor<3x2xf64> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<3x2xf64>) -> tensor<2x3xf64>
    %1 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<3x2xf64>) -> tensor<2x3xf64>
    %2 = stablehlo.minimum %0, %1 : tensor<2x3xf64>
    %3 = stablehlo.transpose %2, dims = [1, 0] : (tensor<2x3xf64>) -> tensor<3x2xf64>
    return %3 : tensor<3x2xf64>
}

// CHECK:  func.func @t7(%arg0: tensor<3x2xf64>, %arg1: tensor<3x2xf64>) -> tensor<3x2xf64> {
// CHECK-NEXT:    %0 = stablehlo.minimum %arg0, %arg1 : tensor<3x2xf64>
// CHECK-NEXT:    return %0 : tensor<3x2xf64>
// CHECK-NEXT:  }

func.func @t8(%arg0: tensor<3x2xf64>, %arg1: tensor<3x2xf64>) -> tensor<3x2xf64> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<3x2xf64>) -> tensor<2x3xf64>
    %1 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<3x2xf64>) -> tensor<2x3xf64>
    %2 = stablehlo.maximum %0, %1 : tensor<2x3xf64>
    %3 = stablehlo.transpose %2, dims = [1, 0] : (tensor<2x3xf64>) -> tensor<3x2xf64>
    return %3 : tensor<3x2xf64>
}

// CHECK:  func.func @t8(%arg0: tensor<3x2xf64>, %arg1: tensor<3x2xf64>) -> tensor<3x2xf64> {
// CHECK-NEXT:    %0 = stablehlo.maximum %arg0, %arg1 : tensor<3x2xf64>
// CHECK-NEXT:    return %0 : tensor<3x2xf64>
// CHECK-NEXT:  }

func.func @t9(%arg0: tensor<3x2xf64>, %arg1: tensor<3x2xf64>) -> tensor<3x2xf64> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<3x2xf64>) -> tensor<2x3xf64>
    %1 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<3x2xf64>) -> tensor<2x3xf64>
    %2 = stablehlo.power %0, %1 : tensor<2x3xf64>
    %3 = stablehlo.transpose %2, dims = [1, 0] : (tensor<2x3xf64>) -> tensor<3x2xf64>
    return %3 : tensor<3x2xf64>
}

// CHECK:  func.func @t9(%arg0: tensor<3x2xf64>, %arg1: tensor<3x2xf64>) -> tensor<3x2xf64> {
// CHECK-NEXT:    %0 = stablehlo.power %arg0, %arg1 : tensor<3x2xf64>
// CHECK-NEXT:    return %0 : tensor<3x2xf64>
// CHECK-NEXT:  }

func.func @t10(%arg0: tensor<3x2xf64>, %arg1: tensor<3x2xf64>) -> tensor<3x2xf64> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<3x2xf64>) -> tensor<2x3xf64>
    %1 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<3x2xf64>) -> tensor<2x3xf64>
    %2 = stablehlo.remainder %0, %1 : tensor<2x3xf64>
    %3 = stablehlo.transpose %2, dims = [1, 0] : (tensor<2x3xf64>) -> tensor<3x2xf64>
    return %3 : tensor<3x2xf64>
}

// CHECK:  func.func @t10(%arg0: tensor<3x2xf64>, %arg1: tensor<3x2xf64>) -> tensor<3x2xf64> {
// CHECK-NEXT:    %0 = stablehlo.remainder %arg0, %arg1 : tensor<3x2xf64>
// CHECK-NEXT:    return %0 : tensor<3x2xf64>
// CHECK-NEXT:  }

func.func @t11(%arg0: tensor<3x2xi1>, %arg1: tensor<3x2xi1>) -> tensor<3x2xi1> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<3x2xi1>) -> tensor<2x3xi1>
    %1 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<3x2xi1>) -> tensor<2x3xi1>
    %2 = stablehlo.and %0, %1 : tensor<2x3xi1>
    %3 = stablehlo.transpose %2, dims = [1, 0] : (tensor<2x3xi1>) -> tensor<3x2xi1>
    return %3 : tensor<3x2xi1>
}

// CHECK:  func.func @t11(%arg0: tensor<3x2xi1>, %arg1: tensor<3x2xi1>) -> tensor<3x2xi1> {
// CHECK-NEXT:    %0 = stablehlo.and %arg0, %arg1 : tensor<3x2xi1>
// CHECK-NEXT:    return %0 : tensor<3x2xi1>
// CHECK-NEXT:  }

func.func @t12(%arg0: tensor<3x2xi1>, %arg1: tensor<3x2xi1>) -> tensor<3x2xi1> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<3x2xi1>) -> tensor<2x3xi1>
    %1 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<3x2xi1>) -> tensor<2x3xi1>
    %2 = stablehlo.or %0, %1 : tensor<2x3xi1>
    %3 = stablehlo.transpose %2, dims = [1, 0] : (tensor<2x3xi1>) -> tensor<3x2xi1>
    return %3 : tensor<3x2xi1>
}

// CHECK:  func.func @t12(%arg0: tensor<3x2xi1>, %arg1: tensor<3x2xi1>) -> tensor<3x2xi1> {
// CHECK-NEXT:    %0 = stablehlo.or %arg0, %arg1 : tensor<3x2xi1>
// CHECK-NEXT:    return %0 : tensor<3x2xi1>
// CHECK-NEXT:  }

func.func @t13(%arg0: tensor<3x2xi1>, %arg1: tensor<3x2xi1>) -> tensor<3x2xi1> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<3x2xi1>) -> tensor<2x3xi1>
    %1 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<3x2xi1>) -> tensor<2x3xi1>
    %2 = stablehlo.xor %0, %1 : tensor<2x3xi1>
    %3 = stablehlo.transpose %2, dims = [1, 0] : (tensor<2x3xi1>) -> tensor<3x2xi1>
    return %3 : tensor<3x2xi1>
}

// CHECK:  func.func @t13(%arg0: tensor<3x2xi1>, %arg1: tensor<3x2xi1>) -> tensor<3x2xi1> {
// CHECK-NEXT:    %0 = stablehlo.xor %arg0, %arg1 : tensor<3x2xi1>
// CHECK-NEXT:    return %0 : tensor<3x2xi1>
// CHECK-NEXT:  }

func.func @t14(%arg0: tensor<3x3xf64>, %arg1: tensor<3x3xf64>) -> tensor<3x3xf64> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<3x3xf64>) -> tensor<3x3xf64>
    %1 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<3x3xf64>) -> tensor<3x3xf64>
    %2 = stablehlo.multiply %0, %1 : tensor<3x3xf64>
    %3 = stablehlo.transpose %2, dims = [1, 0] : (tensor<3x3xf64>) -> tensor<3x3xf64>
    %4 = stablehlo.cosine %2 : tensor<3x3xf64>
    return %4 : tensor<3x3xf64>
}

// CHECK:  func.func @t14(%arg0: tensor<3x3xf64>, %arg1: tensor<3x3xf64>) -> tensor<3x3xf64> {
// CHECK-NEXT:    %0 = stablehlo.multiply %arg0, %arg1 : tensor<3x3xf64>
// CHECK-NEXT:    %1 = stablehlo.cosine %0 : tensor<3x3xf64>
// CHECK-NEXT:    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<3x3xf64>) -> tensor<3x3xf64>
// CHECK-NEXT:    return %2 : tensor<3x3xf64>
// CHECK-NEXT:  }

func.func @t15(%arg0: tensor<3x4xf64>) -> tensor<3x4xf64> {
    %cst = stablehlo.constant dense<[[0.6496222808917268, 0.096212809753773776, 0.15377221949977682], [0.96568572693987941, 0.023023752185516666, 0.79410616419530333], [0.23747479566982688, 0.094921128460392024, 0.79170861871474563], [0.38420536250190751, 0.13376956140378637, 0.91958862661700169]]> : tensor<4x3xf64>
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<3x4xf64>) -> tensor<4x3xf64>
    %1 = stablehlo.add %0, %cst : tensor<4x3xf64>
    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<4x3xf64>) -> tensor<3x4xf64>
    return %2 : tensor<3x4xf64>
}

// CHECK:  func.func @t15(%arg0: tensor<3x4xf64>) -> tensor<3x4xf64> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<{{\[\[}}0.6496222808917268, 0.96568572693987941, 0.23747479566982688, 0.38420536250190751{{\]}}, {{\[}}0.096212809753773776, 0.023023752185516666, 0.094921128460392024, 0.13376956140378637{{\]}}, {{\[}}0.15377221949977682, 0.79410616419530333, 0.79170861871474563, 0.91958862661700169{{\]\]}}> : tensor<3x4xf64>
// CHECK-NEXT:    %0 = stablehlo.add %arg0, %cst : tensor<3x4xf64>
// CHECK-NEXT:    return %0 : tensor<3x4xf64>
// CHECK-NEXT:  }

func.func @t16(%arg0: tensor<3x4xf64>) -> tensor<3x4xf64> {
    %cst = stablehlo.constant dense<[[0.27420692997448848, 0.942463642354195, 0.38939691245710661], [0.78824309336664444, 0.89589669457637566, 0.89695004003823775], [0.29780552679309602, 0.78345118987434825, 0.73322208573165204], [0.76793662184643451, 0.47269648986329182, 0.30380322872102516]]> : tensor<4x3xf64>
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<3x4xf64>) -> tensor<4x3xf64>
    %1 = stablehlo.add %cst, %0 : tensor<4x3xf64>
    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<4x3xf64>) -> tensor<3x4xf64>
    return %2 : tensor<3x4xf64>
}

// CHECK:  func.func @t16(%arg0: tensor<3x4xf64>) -> tensor<3x4xf64> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<{{\[\[}}0.27420692997448848, 0.78824309336664444, 0.29780552679309602, 0.76793662184643451{{\]}}, {{\[}}0.942463642354195, 0.89589669457637566, 0.78345118987434825, 0.47269648986329182{{\]}}, {{\[}}0.38939691245710661, 0.89695004003823775, 0.73322208573165204, 0.30380322872102516{{\]\]}}> : tensor<3x4xf64>
// CHECK-NEXT:    %0 = stablehlo.add %cst, %arg0 : tensor<3x4xf64>
// CHECK-NEXT:    return %0 : tensor<3x4xf64>
// CHECK-NEXT:  }

func.func @t17(%arg0: tensor<3x2xf64>) -> tensor<3x2xf64> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<3x2xf64>) -> tensor<2x3xf64>
    %1 = stablehlo.add %0, %0 : tensor<2x3xf64>
    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<2x3xf64>) -> tensor<3x2xf64>
    return %2 : tensor<3x2xf64>
}

// CHECK:  func.func @t17(%arg0: tensor<3x2xf64>) -> tensor<3x2xf64> {
// CHECK-NEXT:    %0 = stablehlo.add %arg0, %arg0 : tensor<3x2xf64>
// CHECK-NEXT:    return %0 : tensor<3x2xf64>
// CHECK-NEXT:  }

func.func @t18(%arg0: tensor<2x12x12xf32>) -> (tensor<12x12x2xf32>) {
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<12x12x2xf32>
    %7 = stablehlo.transpose %arg0, dims = [1, 2, 0] : (tensor<2x12x12xf32>) -> tensor<12x12x2xf32>
    %8 = stablehlo.multiply %cst, %7 : tensor<12x12x2xf32>
    return %8 : tensor<12x12x2xf32>
}

// CHECK:  func.func @t18(%arg0: tensor<2x12x12xf32>) -> tensor<12x12x2xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<2.000000e+00> : tensor<2x12x12xf32>
// CHECK-NEXT:     %0 = stablehlo.multiply %cst, %arg0 : tensor<2x12x12xf32>
// CHECK-NEXT:     %1 = stablehlo.transpose %0, dims = [1, 2, 0] : (tensor<2x12x12xf32>) -> tensor<12x12x2xf32>
// CHECK-NEXT:     return %1 : tensor<12x12x2xf32>
// CHECK-NEXT: }
