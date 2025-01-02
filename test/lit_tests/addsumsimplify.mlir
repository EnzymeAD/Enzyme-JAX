// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt{no_nan=true})" %s | FileCheck %s --check-prefix=NONAN
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt{no_nan=false})" %s | FileCheck %s --check-prefix=NAN

module {
  func.func @t1(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<3xf64>
    %1 = stablehlo.subtract %arg1, %0 : tensor<3xf64>
    return %1 : tensor<3xf64>
  }
}

// NONAN:  func.func @t1(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
// NONAN-NEXT:    %0 = stablehlo.negate %arg0 : tensor<3xf64>
// NONAN-NEXT:    return %0 : tensor<3xf64>
// NONAN-NEXT:  }

// NAN:  func.func @t1(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
// NAN-NEXT:    %0 = stablehlo.add %arg0, %arg1 : tensor<3xf64>
// NAN-NEXT:    %1 = stablehlo.subtract %arg1, %0 : tensor<3xf64>
// NAN-NEXT:    return %1 : tensor<3xf64>
// NAN-NEXT:  }

module {
  func.func @t2(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<3xf64>
    %1 = stablehlo.subtract %0, %arg1 : tensor<3xf64>
    return %1 : tensor<3xf64>
  }
}

// NONAN:  func.func @t2(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
// NONAN-NEXT:    return %arg0 : tensor<3xf64>
// NONAN-NEXT:  }

// NAN:  func.func @t2(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
// NAN-NEXT:    %0 = stablehlo.add %arg0, %arg1 : tensor<3xf64>
// NAN-NEXT:    %1 = stablehlo.subtract %0, %arg1 : tensor<3xf64>
// NAN-NEXT:    return %1 : tensor<3xf64>
// NAN-NEXT:  }

module {
  func.func @t3(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.add %arg1, %arg0 : tensor<3xf64>
    %1 = stablehlo.subtract %arg1, %0 : tensor<3xf64>
    return %1 : tensor<3xf64>
  }
}

// NONAN:  func.func @t3(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
// NONAN-NEXT:    %0 = stablehlo.negate %arg0 : tensor<3xf64>
// NONAN-NEXT:    return %0 : tensor<3xf64>
// NONAN-NEXT:  }

// NAN:  func.func @t3(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
// NAN-NEXT:    %0 = stablehlo.add %arg1, %arg0 : tensor<3xf64>
// NAN-NEXT:    %1 = stablehlo.subtract %arg1, %0 : tensor<3xf64>
// NAN-NEXT:    return %1 : tensor<3xf64>
// NAN-NEXT:  }

module {
  func.func @t4(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
    %0 = stablehlo.add %arg1, %arg0 : tensor<3xf64>
    %1 = stablehlo.subtract %0, %arg1 : tensor<3xf64>
    return %1 : tensor<3xf64>
  }
}

// NONAN:  func.func @t4(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
// NONAN-NEXT:    return %arg0 : tensor<3xf64>
// NONAN-NEXT:  }

// NAN:  func.func @t4(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
// NAN-NEXT:    %0 = stablehlo.add %arg1, %arg0 : tensor<3xf64>
// NAN-NEXT:    %1 = stablehlo.subtract %0, %arg1 : tensor<3xf64>
// NAN-NEXT:    return %1 : tensor<3xf64>
// NAN-NEXT:  }
