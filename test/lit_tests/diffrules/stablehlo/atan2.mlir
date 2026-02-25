// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=atan2 outfn= retTys=enzyme_dup argTys=enzyme_dup,enzyme_dup mode=ForwardMode" --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=atan2 outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=REVERSE
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @atan2(%y : tensor<5xf32>, %x : tensor<5xf32>) -> tensor<5xf32> {
  %z = stablehlo.atan2 %y, %x : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
  func.return %z : tensor<5xf32>
}

// FORWARD:  func.func @atan2(%arg0: tensor<5xf32>, %arg1: tensor<5xf32>, %arg2: tensor<5xf32>, %arg3: tensor<5xf32>) -> (tensor<5xf32>, tensor<5xf32>) {
// FORWARD-NEXT:    %0 = stablehlo.multiply %arg2, %arg1 : tensor<5xf32>
// FORWARD-NEXT:    %1 = stablehlo.multiply %arg0, %arg3 : tensor<5xf32>
// FORWARD-NEXT:    %2 = stablehlo.subtract %0, %1 : tensor<5xf32>
// FORWARD-NEXT:    %3 = stablehlo.multiply %arg2, %arg2 : tensor<5xf32>
// FORWARD-NEXT:    %4 = stablehlo.multiply %arg0, %arg0 : tensor<5xf32>
// FORWARD-NEXT:    %5 = stablehlo.add %3, %4 : tensor<5xf32>
// FORWARD-NEXT:    %6 = stablehlo.divide %2, %5 : tensor<5xf32>
// FORWARD-NEXT:    %7 = stablehlo.atan2 %arg0, %arg2 : tensor<5xf32>
// FORWARD-NEXT:    return %7, %6 : tensor<5xf32>, tensor<5xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @atan2(%arg0: tensor<5xf32>, %arg1: tensor<5xf32>, %arg2: tensor<5xf32>) -> (tensor<5xf32>, tensor<5xf32>) {
// REVERSE-NEXT:    %0 = stablehlo.multiply %arg1, %arg1 : tensor<5xf32>
// REVERSE-NEXT:    %1 = stablehlo.multiply %arg0, %arg0 : tensor<5xf32>
// REVERSE-NEXT:    %2 = stablehlo.add %0, %1 : tensor<5xf32>
// REVERSE-NEXT:    %3 = stablehlo.divide %arg1, %2 : tensor<5xf32>
// REVERSE-NEXT:    %4 = stablehlo.multiply %arg2, %3 : tensor<5xf32>
// REVERSE-NEXT:    %5 = stablehlo.negate %arg0 : tensor<5xf32>
// REVERSE-NEXT:    %6 = stablehlo.divide %5, %2 : tensor<5xf32>
// REVERSE-NEXT:    %7 = stablehlo.multiply %arg2, %6 : tensor<5xf32>
// REVERSE-NEXT:    return %4, %7 : tensor<5xf32>, tensor<5xf32>
// REVERSE-NEXT:  }

func.func @main() {
  %y = stablehlo.constant dense<[0.0, 1.0, -1.0, 1.0, -1.0]> : tensor<5xf32>
  %x = stablehlo.constant dense<[1.0, 1.0, 1.0, -1.0, -1.0]> : tensor<5xf32>
  %out = stablehlo.constant dense<[0.0, 0.7853981633974483, -0.7853981633974483, 2.356194490192345, -2.356194490192345]> : tensor<5xf32>
  %expected_dy = stablehlo.constant dense<[1.0, 0.5, 0.5, -0.5, -0.5]> : tensor<5xf32>
  %expected_dx = stablehlo.constant dense<[0.0, -0.5, 0.5, -0.5, 0.5]> : tensor<5xf32>

  %done = stablehlo.constant dense<1.0> : tensor<5xf32>
  %dzero = stablehlo.constant dense<0.0> : tensor<5xf32>

  %fwd_y:2 = enzyme.fwddiff @atan2(%y, %done, %x, %dzero) {
    activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<5xf32>, tensor<5xf32>, tensor<5xf32>, tensor<5xf32>) -> (tensor<5xf32>, tensor<5xf32>)

  check.expect_almost_eq %fwd_y#0, %out : tensor<5xf32>
  check.expect_almost_eq %fwd_y#1, %expected_dy : tensor<5xf32>

  %fwd_x:2 = enzyme.fwddiff @atan2(%y, %done, %x, %dzero) {
    activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<5xf32>, tensor<5xf32>, tensor<5xf32>, tensor<5xf32>) -> (tensor<5xf32>, tensor<5xf32>)

  check.expect_almost_eq %fwd_x#0, %out : tensor<5xf32>
  check.expect_almost_eq %fwd_x#1, %expected_dx : tensor<5xf32>

  %rev:3 = enzyme.autodiff @atan2(%y, %x, %done) {
    activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>],
    ret_activity=[#enzyme<activity enzyme_active>]
  } : (tensor<5xf32>, tensor<5xf32>, tensor<5xf32>) -> (tensor<5xf32>, tensor<5xf32>, tensor<5xf32>)

  check.expect_almost_eq %rev#0, %out : tensor<5xf32>
  check.expect_almost_eq %rev#1, %expected_dy : tensor<5xf32>
  check.expect_almost_eq %rev#2, %expected_dx : tensor<5xf32>

  func.return
}
