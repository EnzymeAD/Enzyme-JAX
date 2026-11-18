// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=cosc outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=cosc outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined strongZero=true" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=STRONG
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --lower-enzymexla-math --chlo-legalize-to-stablehlo --canonicalize --arith-raise | stablehlo-translate - --interpret

func.func @cosc(%x: tensor<4xf32>) -> tensor<4xf32> {
  %0 = enzymexla.math.cosc %x : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// REVERSE:  func.func @cosc(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
// REVERSE-DAG:     %[[TWO:.+]] = stablehlo.constant dense<2.000000e+00> : tensor<4xf32>
// REVERSE-DAG:     %[[NPI2:.+]] = stablehlo.constant dense<-9.86960411> : tensor<4xf32>
// REVERSE-DAG:     %[[AT0:.+]] = stablehlo.constant dense<-3.28986812> : tensor<4xf32>
// REVERSE-DAG:     %[[ZERO:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<4xf32>
// REVERSE:         %[[ISZERO:.+]] = stablehlo.compare EQ, %arg0, %[[ZERO]] : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
// REVERSE-NEXT:    %[[LIM:.+]] = stablehlo.multiply %arg1, %[[AT0]] : tensor<4xf32>
// REVERSE-NEXT:    %[[SINC:.+]] = enzymexla.math.sinc %arg0 : (tensor<4xf32>) -> tensor<4xf32>
// REVERSE-NEXT:    %[[T0:.+]] = stablehlo.multiply %[[NPI2]], %[[SINC]] : tensor<4xf32>
// REVERSE-NEXT:    %[[COSC:.+]] = enzymexla.math.cosc %arg0 : (tensor<4xf32>) -> tensor<4xf32>
// REVERSE-NEXT:    %[[T1:.+]] = stablehlo.multiply %[[TWO]], %[[COSC]] : tensor<4xf32>
// REVERSE-NEXT:    %[[T2:.+]] = stablehlo.divide %[[T1]], %arg0 : tensor<4xf32>
// REVERSE-NEXT:    %[[DER:.+]] = stablehlo.subtract %[[T0]], %[[T2]] : tensor<4xf32>
// REVERSE-NEXT:    %[[GEN:.+]] = stablehlo.multiply %arg1, %[[DER]] : tensor<4xf32>
// REVERSE-NEXT:    %[[RES:.+]] = stablehlo.select %[[ISZERO]], %[[LIM]], %[[GEN]] : tensor<4xi1>, tensor<4xf32>
// REVERSE-NEXT:    return %[[RES]] : tensor<4xf32>
// REVERSE-NEXT:  }

// STRONG:  func.func @cosc(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
// STRONG-DAG:     %[[TWO:.+]] = stablehlo.constant dense<2.000000e+00> : tensor<4xf32>
// STRONG-DAG:     %[[NPI2:.+]] = stablehlo.constant dense<-9.86960411> : tensor<4xf32>
// STRONG-DAG:     %[[AT0:.+]] = stablehlo.constant dense<-3.28986812> : tensor<4xf32>
// STRONG-DAG:     %[[ZERO:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<4xf32>
// STRONG:         %[[ISZERO:.+]] = stablehlo.compare EQ, %arg0, %[[ZERO]] : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
// STRONG-NEXT:    %[[DZERO0:.+]] = stablehlo.compare EQ, %arg1, %[[ZERO]] : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
// STRONG-NEXT:    %[[LIM:.+]] = stablehlo.multiply %arg1, %[[AT0]] : tensor<4xf32>
// STRONG-NEXT:    %[[LIMSEL:.+]] = stablehlo.select %[[DZERO0]], %[[ZERO]], %[[LIM]] : tensor<4xi1>, tensor<4xf32>
// STRONG-NEXT:    %[[SINC:.+]] = enzymexla.math.sinc %arg0 : (tensor<4xf32>) -> tensor<4xf32>
// STRONG-NEXT:    %[[T0:.+]] = stablehlo.multiply %[[NPI2]], %[[SINC]] : tensor<4xf32>
// STRONG-NEXT:    %[[COSC:.+]] = enzymexla.math.cosc %arg0 : (tensor<4xf32>) -> tensor<4xf32>
// STRONG-NEXT:    %[[T1:.+]] = stablehlo.multiply %[[TWO]], %[[COSC]] : tensor<4xf32>
// STRONG-NEXT:    %[[T2:.+]] = stablehlo.divide %[[T1]], %arg0 : tensor<4xf32>
// STRONG-NEXT:    %[[DER:.+]] = stablehlo.subtract %[[T0]], %[[T2]] : tensor<4xf32>
// STRONG-NEXT:    %[[DZERO1:.+]] = stablehlo.compare EQ, %arg1, %[[ZERO]] : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
// STRONG-NEXT:    %[[GEN:.+]] = stablehlo.multiply %arg1, %[[DER]] : tensor<4xf32>
// STRONG-NEXT:    %[[GENSEL:.+]] = stablehlo.select %[[DZERO1]], %[[ZERO]], %[[GEN]] : tensor<4xi1>, tensor<4xf32>
// STRONG-NEXT:    %[[RES:.+]] = stablehlo.select %[[ISZERO]], %[[LIMSEL]], %[[GENSEL]] : tensor<4xi1>, tensor<4xf32>
// STRONG-NEXT:    return %[[RES]] : tensor<4xf32>
// STRONG-NEXT:  }

func.func @main() {
  // A zero adjoint yields a zero gradient even where cosc'(x) is nan.
  %x = stablehlo.constant dense<[0x7FC00000, 0x7F800000, 0.0, 1.0]> : tensor<4xf32>
  %d = stablehlo.constant dense<[0.0, 0.0, 0.0, 1.0]> : tensor<4xf32>

  %res = enzyme.autodiff @cosc(%x, %d) <{
    activity = [#enzyme.activity<enzyme_active>],
    ret_activity = [#enzyme.activity<enzyme_activenoneed>],
    strong_zero = true
  }> : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

  check.expect_almost_eq_const %res, dense<[0.0, 0.0, 0.0, 2.0]> : tensor<4xf32>
  func.return
}
