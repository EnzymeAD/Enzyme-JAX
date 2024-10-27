// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=clamp outfn= retTys=enzyme_dup argTys=enzyme_dup,enzyme_dup,enzyme_dup mode=ForwardMode" --canonicalize --arith-raise | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=clamp outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_active,enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise | FileCheck %s --check-prefix=REVERSE
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise | $(dirname $(which enzymexlamlir-opt))/external/stablehlo/stablehlo-translate - --interpret

// : # enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise

module {
  func.func @clamp(%min: tensor<10xf32>, %operand: tensor<10xf32>, %max: tensor<10xf32>) -> tensor<10xf32> {
    %0 = stablehlo.clamp %min, %operand, %max  : tensor<10xf32>
    return %0 : tensor<10xf32>
  }

  func.func @main() {
    %min = stablehlo.constant dense<1.0> : tensor<10xf32>
    %operand = stablehlo.constant dense<[1.5, 1.5, 1.0, 0.0, 2.5, 1.5, 1.5, 1.5, 1.5, 1.5]> : tensor<10xf32>
    %max = stablehlo.constant dense<2.0> : tensor<10xf32>

    %dclamp = stablehlo.constant dense<1.0> : tensor<10xf32>

    %res:2 = enzyme.autodiff @clamp(%min, %operand, %max, %dclamp) {
      activity=[#enzyme<activity enzyme_const>, #enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>],
      ret_activity=[#enzyme<activity enzyme_active>]
    } : (tensor<10xf32>, tensor<10xf32>, tensor<10xf32>, tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>)

    check.expect_eq_const %res#0, dense<[1.5, 1.5, 1.0, 1.0, 2.0, 1.5, 1.5, 1.5, 1.5, 1.5]> : tensor<10xf32>
    check.expect_eq_const %res#1, dense<[1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]> : tensor<10xf32>

    func.return
  }
}

// FORWARD:  func.func @clamp(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>, %arg2: tensor<10xf32>, %arg3: tensor<10xf32>, %arg4: tensor<10xf32>, %arg5: tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>) {
// FORWARD-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<10xf32>
// FORWARD-NEXT:    %0 = stablehlo.compare  LT, %arg0, %arg4 : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xi1>
// FORWARD-NEXT:    %1 = stablehlo.compare  GT, %arg0, %arg2 : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xi1>
// FORWARD-NEXT:    %2 = stablehlo.and %0, %1 : tensor<10xi1>
// FORWARD-NEXT:    %3 = stablehlo.select %2, %arg1, %cst : tensor<10xi1>, tensor<10xf32>
// FORWARD-NEXT:    %4 = stablehlo.compare  LT, %arg2, %arg0 : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xi1>
// FORWARD-NEXT:    %5 = stablehlo.compare  GT, %arg2, %arg4 : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xi1>
// FORWARD-NEXT:    %6 = stablehlo.or %4, %5 : tensor<10xi1>
// FORWARD-NEXT:    %7 = stablehlo.select %6, %cst, %arg3 : tensor<10xi1>, tensor<10xf32>
// FORWARD-NEXT:    %8 = stablehlo.add %3, %7 : tensor<10xf32>
// FORWARD-NEXT:    %9 = stablehlo.maximum %arg2, %arg0 : tensor<10xf32>
// FORWARD-NEXT:    %10 = stablehlo.compare  GT, %9, %arg4 : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xi1>
// FORWARD-NEXT:    %11 = stablehlo.select %10, %arg5, %cst : tensor<10xi1>, tensor<10xf32>
// FORWARD-NEXT:    %12 = stablehlo.add %8, %11 : tensor<10xf32>
// FORWARD-NEXT:    %13 = stablehlo.clamp %arg0, %arg2, %arg4 : tensor<10xf32>
// FORWARD-NEXT:    return %13, %12 : tensor<10xf32>, tensor<10xf32>
// FORWARD-NEXT:  }

// REVERSE: func.func @clamp(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>, %arg2: tensor<10xf32>, %arg3: tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>, tensor<10xf32>) {
// REVERSE-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<10xf32>
// REVERSE-NEXT:    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<10xf32>
// REVERSE-NEXT:    %0 = stablehlo.add %arg3, %cst_0 : tensor<10xf32>
// REVERSE-NEXT:    %1 = stablehlo.compare  LT, %arg0, %arg2 : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xi1>
// REVERSE-NEXT:    %2 = stablehlo.compare  GT, %arg0, %arg1 : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xi1>
// REVERSE-NEXT:    %3 = stablehlo.and %1, %2 : tensor<10xi1>
// REVERSE-NEXT:    %4 = stablehlo.select %3, %0, %cst : tensor<10xi1>, tensor<10xf32>
// REVERSE-NEXT:    %5 = stablehlo.add %4, %cst_0 : tensor<10xf32>
// REVERSE-NEXT:    %6 = stablehlo.compare  LT, %arg1, %arg0 : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xi1>
// REVERSE-NEXT:    %7 = stablehlo.compare  GT, %arg1, %arg2 : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xi1>
// REVERSE-NEXT:    %8 = stablehlo.or %6, %7 : tensor<10xi1>
// REVERSE-NEXT:    %9 = stablehlo.select %8, %cst, %0 : tensor<10xi1>, tensor<10xf32>
// REVERSE-NEXT:    %10 = stablehlo.add %9, %cst_0 : tensor<10xf32>
// REVERSE-NEXT:    %11 = stablehlo.maximum %arg1, %arg0 : tensor<10xf32>
// REVERSE-NEXT:    %12 = stablehlo.compare  GT, %11, %arg2 : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xi1>
// REVERSE-NEXT:    %13 = stablehlo.select %12, %0, %cst : tensor<10xi1>, tensor<10xf32>
// REVERSE-NEXT:    %14 = stablehlo.add %13, %cst_0 : tensor<10xf32>
// REVERSE-NEXT:    return %5, %10, %14 : tensor<10xf32>, tensor<10xf32>, tensor<10xf32>
// REVERSE-NEXT:  }
