// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=subtract outfn= retTys=enzyme_dup argTys=enzyme_dup,enzyme_dup mode=ForwardMode" --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=subtract outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=REVERSE
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise | stablehlo-translate - --interpret

func.func @subtract(%a : tensor<2xf32>, %b : tensor<2xf32>) -> tensor<2xf32> {
  %c = stablehlo.subtract %a, %b : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  func.return %c : tensor<2xf32>
}

// FORWARD:  func.func @subtract(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>, %arg3: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
// FORWARD-NEXT:    %0 = stablehlo.subtract %arg1, %arg3 : tensor<2xf32>
// FORWARD-NEXT:    %1 = stablehlo.subtract %arg0, %arg2 : tensor<2xf32>
// FORWARD-NEXT:    return %1, %0 : tensor<2xf32>, tensor<2xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @subtract(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
// REVERSE-NEXT:    %0 = stablehlo.negate %arg2 : tensor<2xf32>
// REVERSE-NEXT:    return %arg2, %0 : tensor<2xf32>, tensor<2xf32>
// REVERSE-NEXT:  }

func.func @main() {
  %a = stablehlo.constant dense<[1.0, -2.0]> : tensor<2xf32>
  %b = stablehlo.constant dense<[2.0, 3.0]> : tensor<2xf32>
  %output = stablehlo.constant dense<[-1.0, -5.0]> : tensor<2xf32>

  %done = stablehlo.constant dense<1.0> : tensor<2xf32>
  %dnegone = stablehlo.constant dense<-1.0> : tensor<2xf32>
  %dzero = stablehlo.constant dense<0.0> : tensor<2xf32>

  // fwd diff
  %fwd_a:2 = enzyme.fwddiff @subtract(%a, %done, %b, %dzero) {
    activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>)

  check.expect_almost_eq %fwd_a#0, %output : tensor<2xf32>
  check.expect_almost_eq %fwd_a#1, %done : tensor<2xf32>

  %fwd_b:2 = enzyme.fwddiff @subtract(%a, %dzero, %b, %done) {
    activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>)

  check.expect_almost_eq %fwd_b#0, %output : tensor<2xf32>
  check.expect_almost_eq %fwd_b#1, %dnegone : tensor<2xf32>

  // rev diff
  %rev:3 = enzyme.autodiff @subtract(%a, %b, %done) {
    activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>],
    ret_activity=[#enzyme<activity enzyme_active>]
  } : (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>)

  check.expect_almost_eq %rev#0, %output : tensor<2xf32>
  check.expect_almost_eq %rev#1, %done : tensor<2xf32>
  check.expect_almost_eq %rev#2, %dnegone : tensor<2xf32>

  func.return
}

