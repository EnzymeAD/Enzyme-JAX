// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=trunc_roundtrip outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" --arith-raise --canonicalize --cse --enzyme-hlo-opt | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=trunc_roundtrip outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --arith-raise --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --cse --enzyme-hlo-opt | FileCheck %s --check-prefix=REVERSE
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=widen outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" --arith-raise --canonicalize --cse --enzyme-hlo-opt | FileCheck %s --check-prefix=FORWARD-WIDEN
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=widen outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --arith-raise --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --cse --enzyme-hlo-opt | FileCheck %s --check-prefix=REVERSE-WIDEN

// x -> 2 * float(trunc(x)): a convert to or from an integer type is piecewise
// constant, so activity analysis marks it inactive and the derivative is zero.
// It used to be treated as linear, giving 2.
func.func @trunc_roundtrip(%x: tensor<2xf64>) -> tensor<2xf64> {
  %c2 = stablehlo.constant dense<2.0> : tensor<2xf64>
  %i = stablehlo.convert %x : (tensor<2xf64>) -> tensor<2xi64>
  %f = stablehlo.convert %i : (tensor<2xi64>) -> tensor<2xf64>
  %r = stablehlo.multiply %f, %c2 : tensor<2xf64>
  return %r : tensor<2xf64>
}

// FORWARD:  func.func @trunc_roundtrip(%arg0: tensor<2xf64>, %arg1: tensor<2xf64>) -> (tensor<2xf64>, tensor<2xf64>) {
// FORWARD-DAG:    %[[ZERO:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<2xf64>
// FORWARD-DAG:    %[[TWO:.+]] = stablehlo.constant dense<2.000000e+00> : tensor<2xf64>
// FORWARD-DAG:    %[[I:.+]] = stablehlo.convert %arg0 : (tensor<2xf64>) -> tensor<2xi64>
// FORWARD-DAG:    %[[F:.+]] = stablehlo.convert %[[I]] : (tensor<2xi64>) -> tensor<2xf64>
// FORWARD:        %[[R:.+]] = stablehlo.multiply %[[F]], %[[TWO]] : tensor<2xf64>
// FORWARD:        return %[[R]], %[[ZERO]] : tensor<2xf64>, tensor<2xf64>

// REVERSE:  func.func @trunc_roundtrip(%arg0: tensor<2xf64>, %arg1: tensor<2xf64>) -> tensor<2xf64> {
// REVERSE:        %[[ZERO:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<2xf64>
// REVERSE:        return %[[ZERO]] : tensor<2xf64>

// A convert between floating-point types stays linear.
func.func @widen(%x: tensor<2xf32>) -> tensor<2xf64> {
  %c2 = stablehlo.constant dense<2.0> : tensor<2xf64>
  %f = stablehlo.convert %x : (tensor<2xf32>) -> tensor<2xf64>
  %r = stablehlo.multiply %f, %c2 : tensor<2xf64>
  return %r : tensor<2xf64>
}

// FORWARD-WIDEN:  func.func @widen(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> (tensor<2xf64>, tensor<2xf64>) {
// FORWARD-WIDEN:    %[[TWO:.+]] = stablehlo.constant dense<2.000000e+00> : tensor<2xf64>
// FORWARD-WIDEN-DAG:    %[[DF:.+]] = stablehlo.convert %arg1 : (tensor<2xf32>) -> tensor<2xf64>
// FORWARD-WIDEN-DAG:    %[[F:.+]] = stablehlo.convert %arg0 : (tensor<2xf32>) -> tensor<2xf64>
// FORWARD-WIDEN-DAG:    %[[DR:.+]] = stablehlo.multiply %[[DF]], %[[TWO]] : tensor<2xf64>
// FORWARD-WIDEN-DAG:    %[[R:.+]] = stablehlo.multiply %[[F]], %[[TWO]] : tensor<2xf64>
// FORWARD-WIDEN:    return %[[R]], %[[DR]] : tensor<2xf64>, tensor<2xf64>

// REVERSE-WIDEN:  func.func @widen(%arg0: tensor<2xf32>, %arg1: tensor<2xf64>) -> tensor<2xf32> {
// REVERSE-WIDEN:    %[[TWO:.+]] = stablehlo.constant dense<2.000000e+00> : tensor<2xf64>
// REVERSE-WIDEN:    %[[D:.+]] = stablehlo.multiply %arg1, %[[TWO]] : tensor<2xf64>
// REVERSE-WIDEN:    %[[DX:.+]] = stablehlo.convert %[[D]] : (tensor<2xf64>) -> tensor<2xf32>
// REVERSE-WIDEN:    return %[[DX]] : tensor<2xf32>
