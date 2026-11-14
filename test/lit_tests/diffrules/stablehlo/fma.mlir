// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=fma outfn= retTys=enzyme_dup argTys=enzyme_dup,enzyme_dup,enzyme_dup mode=ForwardMode" --arith-raise --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=fma outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_active,enzyme_active mode=ReverseModeCombined" --arith-raise --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=REVERSE
// RUN: enzymexlamlir-opt %s --enzyme --arith-raise --canonicalize --remove-unnecessary-enzyme-ops --arith-raise | stablehlo-translate - --interpret

func.func @fma(%a : tensor<2xf32>, %b : tensor<2xf32>, %c : tensor<2xf32>) -> tensor<2xf32> {
  %r = stablehlo.fma %a, %b, %c : (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  func.return %r : tensor<2xf32>
}

// FORWARD:  func.func @fma(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>, %arg3: tensor<2xf32>, %arg4: tensor<2xf32>, %arg5: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
// FORWARD-NEXT:    %0 = stablehlo.multiply %arg1, %arg2 : tensor<2xf32>
// FORWARD-NEXT:    %1 = stablehlo.multiply %arg3, %arg0 : tensor<2xf32>
// FORWARD-NEXT:    %2 = stablehlo.add %0, %1 : tensor<2xf32>
// FORWARD-NEXT:    %3 = stablehlo.add %2, %arg5 : tensor<2xf32>
// FORWARD-NEXT:    %4 = stablehlo.fma %arg0, %arg2, %arg4 : tensor<2xf32>
// FORWARD-NEXT:    return %4, %3 : tensor<2xf32>, tensor<2xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @fma(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>, %arg3: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) {
// REVERSE-NEXT:    %0 = stablehlo.multiply %arg3, %arg1 : tensor<2xf32>
// REVERSE-NEXT:    %1 = stablehlo.multiply %arg3, %arg0 : tensor<2xf32>
// REVERSE-NEXT:    return %0, %1, %arg3 : tensor<2xf32>, tensor<2xf32>, tensor<2xf32>
// REVERSE-NEXT:  }

func.func @main() {
  %a = stablehlo.constant dense<[2.0, -3.0]> : tensor<2xf32>
  %b = stablehlo.constant dense<[5.0, 7.0]> : tensor<2xf32>
  %c = stablehlo.constant dense<[11.0, 13.0]> : tensor<2xf32>
  %out = stablehlo.constant dense<[21.0, -8.0]> : tensor<2xf32>

  %done = stablehlo.constant dense<1.0> : tensor<2xf32>
  %dzero = stablehlo.constant dense<0.0> : tensor<2xf32>

  %fwd_a:2 = enzyme.fwddiff @fma(%a, %done, %b, %dzero, %c, %dzero) <{
    activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dup>]
  }> : (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>)

  check.expect_almost_eq %fwd_a#0, %out : tensor<2xf32>
  check.expect_almost_eq %fwd_a#1, %b : tensor<2xf32>

  %fwd_b:2 = enzyme.fwddiff @fma(%a, %dzero, %b, %done, %c, %dzero) <{
    activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dup>]
  }> : (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>)

  check.expect_almost_eq %fwd_b#0, %out : tensor<2xf32>
  check.expect_almost_eq %fwd_b#1, %a : tensor<2xf32>

  %fwd_c:2 = enzyme.fwddiff @fma(%a, %dzero, %b, %dzero, %c, %done) <{
    activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dup>]
  }> : (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>)

  check.expect_almost_eq %fwd_c#0, %out : tensor<2xf32>
  check.expect_almost_eq %fwd_c#1, %done : tensor<2xf32>

  %rev:4 = enzyme.autodiff @fma(%a, %b, %c, %done) <{
    activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
    ret_activity=[#enzyme.activity<enzyme_active>]
  }> : (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>)

  check.expect_almost_eq %rev#0, %out : tensor<2xf32>
  check.expect_almost_eq %rev#1, %b : tensor<2xf32>
  check.expect_almost_eq %rev#2, %a : tensor<2xf32>
  check.expect_almost_eq %rev#3, %done : tensor<2xf32>

  func.return
}
