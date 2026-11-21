// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=barrier outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=barrier outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=REVERSE
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=barrier_multi outfn= retTys=enzyme_dup,enzyme_const argTys=enzyme_dup,enzyme_dup,enzyme_const mode=ForwardMode" --arith-raise --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=FORWARD-MULTI
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=barrier_multi outfn= retTys=enzyme_active,enzyme_const argTys=enzyme_active,enzyme_active,enzyme_const mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=REVERSE-MULTI
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @barrier(%x : tensor<3xf32>) -> tensor<3xf32> {
  %y = stablehlo.optimization_barrier %x : tensor<3xf32>
  func.return %y : tensor<3xf32>
}

// FORWARD:  func.func @barrier(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> (tensor<3xf32>, tensor<3xf32>) {
// FORWARD-DAG:    %[[DY:.+]] = stablehlo.optimization_barrier %arg1 : tensor<3xf32>
// FORWARD-DAG:    %[[Y:.+]] = stablehlo.optimization_barrier %arg0 : tensor<3xf32>
// FORWARD-NEXT:   return %[[Y]], %[[DY]] : tensor<3xf32>, tensor<3xf32>
// FORWARD-NEXT: }

// REVERSE:  func.func @barrier(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> tensor<3xf32> {
// REVERSE-NEXT:   %0 = stablehlo.optimization_barrier %arg1 : tensor<3xf32>
// REVERSE-NEXT:   return %0 : tensor<3xf32>
// REVERSE-NEXT: }

func.func @barrier_multi(%x : tensor<2xf32>, %y : tensor<2xf32>, %k : tensor<i64>) -> (tensor<2xf32>, tensor<i64>) {
  %b:3 = stablehlo.optimization_barrier %x, %y, %k : tensor<2xf32>, tensor<2xf32>, tensor<i64>
  %prod = stablehlo.multiply %b#0, %b#1 : tensor<2xf32>
  func.return %prod, %b#2 : tensor<2xf32>, tensor<i64>
}

// FORWARD-MULTI:  func.func @barrier_multi(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>, %arg3: tensor<2xf32>, %arg4: tensor<i64>) -> (tensor<2xf32>, tensor<2xf32>) {
// FORWARD-MULTI-DAG:    %[[DB:.+]]:2 = stablehlo.optimization_barrier %arg1, %arg3 : tensor<2xf32>, tensor<2xf32>
// FORWARD-MULTI-DAG:    %[[B:.+]]:2 = stablehlo.optimization_barrier %arg0, %arg2 : tensor<2xf32>, tensor<2xf32>
// FORWARD-MULTI-DAG:    %[[T0:.+]] = stablehlo.multiply %[[DB]]#0, %[[B]]#1 : tensor<2xf32>
// FORWARD-MULTI-DAG:    %[[T1:.+]] = stablehlo.multiply %[[DB]]#1, %[[B]]#0 : tensor<2xf32>
// FORWARD-MULTI-DAG:    %[[DP:.+]] = stablehlo.add %[[T0]], %[[T1]] : tensor<2xf32>
// FORWARD-MULTI-DAG:    %[[P:.+]] = stablehlo.multiply %[[B]]#0, %[[B]]#1 : tensor<2xf32>
// FORWARD-MULTI:        return %[[P]], %[[DP]] : tensor<2xf32>, tensor<2xf32>

// REVERSE-MULTI:  func.func @barrier_multi(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>, %arg2: tensor<i64>, %arg3: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
// REVERSE-MULTI-DAG:    %[[B:.+]]:2 = stablehlo.optimization_barrier %arg0, %arg1 : tensor<2xf32>, tensor<2xf32>
// REVERSE-MULTI-DAG:    %[[DX:.+]] = stablehlo.multiply %arg3, %[[B]]#1 : tensor<2xf32>
// REVERSE-MULTI-DAG:    %[[DY:.+]] = stablehlo.multiply %arg3, %[[B]]#0 : tensor<2xf32>
// REVERSE-MULTI-DAG:    %[[DB:.+]]:2 = stablehlo.optimization_barrier %[[DX]], %[[DY]] : tensor<2xf32>, tensor<2xf32>
// REVERSE-MULTI:        return %[[DB]]#0, %[[DB]]#1 : tensor<2xf32>, tensor<2xf32>

func.func @main() {
  %x = stablehlo.constant dense<[1.0, -2.0, 3.5]> : tensor<3xf32>
  %dx = stablehlo.constant dense<[0.5, 2.0, -1.0]> : tensor<3xf32>

  %fwd:2 = enzyme.fwddiff @barrier(%x, %dx) <{
    activity=[#enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dup>]
  }> : (tensor<3xf32>, tensor<3xf32>) -> (tensor<3xf32>, tensor<3xf32>)

  check.expect_almost_eq %fwd#0, %x : tensor<3xf32>
  check.expect_almost_eq %fwd#1, %dx : tensor<3xf32>

  %rev:2 = enzyme.autodiff @barrier(%x, %dx) <{
    activity=[#enzyme.activity<enzyme_active>],
    ret_activity=[#enzyme.activity<enzyme_active>]
  }> : (tensor<3xf32>, tensor<3xf32>) -> (tensor<3xf32>, tensor<3xf32>)

  check.expect_almost_eq %rev#0, %x : tensor<3xf32>
  check.expect_almost_eq %rev#1, %dx : tensor<3xf32>

  %bx = stablehlo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %fwd_w2:2 = enzyme.fwddiff @barrier(%x, %bx) <{
    activity=[#enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dup>],
    width=2
  }> : (tensor<3xf32>, tensor<2x3xf32>) -> (tensor<3xf32>, tensor<2x3xf32>)

  check.expect_almost_eq %fwd_w2#0, %x : tensor<3xf32>
  check.expect_almost_eq %fwd_w2#1, %bx : tensor<2x3xf32>

  %rev_w2 = enzyme.autodiff @barrier(%x, %bx) <{
    activity=[#enzyme.activity<enzyme_active>],
    ret_activity=[#enzyme.activity<enzyme_activenoneed>],
    width=2
  }> : (tensor<3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>

  check.expect_almost_eq %rev_w2, %bx : tensor<2x3xf32>

  %mx = stablehlo.constant dense<[2.0, 3.0]> : tensor<2xf32>
  %mdx = stablehlo.constant dense<[1.0, 0.5]> : tensor<2xf32>
  %my = stablehlo.constant dense<[4.0, -5.0]> : tensor<2xf32>
  %mdy = stablehlo.constant dense<[-1.0, 2.0]> : tensor<2xf32>
  %mk = stablehlo.constant dense<7> : tensor<i64>
  %expected_prod = stablehlo.constant dense<[8.0, -15.0]> : tensor<2xf32>
  %expected_dprod = stablehlo.constant dense<[2.0, 3.5]> : tensor<2xf32>

  %fwd_m:3 = enzyme.fwddiff @barrier_multi(%mx, %mdx, %my, %mdy, %mk) <{
    activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_const>],
    ret_activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_const>]
  }> : (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<i64>) -> (tensor<2xf32>, tensor<2xf32>, tensor<i64>)

  check.expect_almost_eq %fwd_m#0, %expected_prod : tensor<2xf32>
  check.expect_almost_eq %fwd_m#1, %expected_dprod : tensor<2xf32>
  check.expect_eq %fwd_m#2, %mk : tensor<i64>

  %seed = stablehlo.constant dense<[1.0, 2.0]> : tensor<2xf32>
  %expected_gx = stablehlo.constant dense<[4.0, -10.0]> : tensor<2xf32>
  %expected_gy = stablehlo.constant dense<[2.0, 6.0]> : tensor<2xf32>
  %rev_m:2 = enzyme.autodiff @barrier_multi(%mx, %my, %mk, %seed) <{
    activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>],
    ret_activity=[#enzyme.activity<enzyme_activenoneed>, #enzyme.activity<enzyme_constnoneed>]
  }> : (tensor<2xf32>, tensor<2xf32>, tensor<i64>, tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>)

  check.expect_almost_eq %rev_m#0, %expected_gx : tensor<2xf32>
  check.expect_almost_eq %rev_m#1, %expected_gy : tensor<2xf32>

  func.return
}
