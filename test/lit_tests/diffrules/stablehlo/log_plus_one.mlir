// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=log_plus_one outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" --enzyme-hlo-opt  | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=log_plus_one outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt | FileCheck %s --check-prefix=REVERSE
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise | stablehlo-translate - --interpret

func.func @log_plus_one(%x : tensor<2xf32>) -> tensor<2xf32> {
  %y = stablehlo.log_plus_one %x : (tensor<2xf32>) -> tensor<2xf32>
  func.return %y : tensor<2xf32>
}

func.func @main() {
  %input = stablehlo.constant dense<[0.0, 1.0]> : tensor<2xf32>
  %dinput = stablehlo.constant dense<1.0> : tensor<2xf32>

  // fwd diff
  %fwd_res:2 = enzyme.fwddiff @log_plus_one(%input, %dinput) {
    activity=[#enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>)

  check.expect_eq_const %fwd_res#0, dense<[0.0, 0.6931471805599453]> : tensor<2xf32>
  check.expect_eq_const %fwd_res#1, dense<[1.0, 0.5]> : tensor<2xf32>

  // rev diff
  %rev_res:2 = enzyme.autodiff @log_plus_one(%input, %dinput) {
    activity=[#enzyme<activity enzyme_active>],
    ret_activity=[#enzyme<activity enzyme_active>]
  } : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>)

  check.expect_eq_const %rev_res#0, dense<[0.0, 0.6931471805599453]> : tensor<2xf32>
  check.expect_eq_const %rev_res#1, dense<[1.0, 0.5]> : tensor<2xf32>

  func.return
}

// FORWARD:  func.func @log_plus_one(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
// FORWARD-NEXT:    %cst = stablehlo.constant dense<1.000000e+00> : tensor<2xf32>
// FORWARD-NEXT:    %0 = stablehlo.add %arg0, %cst : tensor<2xf32>
// FORWARD-NEXT:    %1 = stablehlo.divide %arg1, %0 : tensor<2xf32>
// FORWARD-NEXT:    %2 = stablehlo.log_plus_one %arg0 : tensor<2xf32>
// FORWARD-NEXT:    return %2, %1 : tensor<2xf32>, tensor<2xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @log_plus_one(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
// REVERSE-NEXT:    %cst = stablehlo.constant dense<1.000000e+00> : tensor<2xf32>
// REVERSE-NEXT:    %0 = stablehlo.add %arg0, %cst : tensor<2xf32>
// REVERSE-NEXT:    %1 = stablehlo.divide %arg1, %0 : tensor<2xf32>
// REVERSE-NEXT:    return %1 : tensor<2xf32>
// REVERSE-NEXT:  }
