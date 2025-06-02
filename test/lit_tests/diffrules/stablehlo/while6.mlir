// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" --canonicalize --arith-raise --enzyme-hlo-opt --remove-unnecessary-enzyme-ops --enzyme-hlo-opt --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --arith-raise --enzyme-hlo-opt --remove-unnecessary-enzyme-ops --enzyme-hlo-opt | FileCheck %s --check-prefix=REVERSE

func.func @main(%arg0: tensor<12x16x4xf32>) -> (tensor<12x4xf32>) {
  %c = stablehlo.constant dense<0> : tensor<i32>
  %c_0 = stablehlo.constant dense<15> : tensor<i32>
  %c_1 = stablehlo.constant dense<1> : tensor<i32>
  %0 = stablehlo.slice %arg0 [0:12, 0:1, 0:4] : (tensor<12x16x4xf32>) -> tensor<12x1x4xf32>
  %1 = stablehlo.reshape %0 : (tensor<12x1x4xf32>) -> tensor<12x4xf32>
  %8:2 = stablehlo.while(%iterArg = %c, %iterArg_4 = %1) : tensor<i32>, tensor<12x4xf32>
    cond {
    %9 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %9 : tensor<i1>
  } do {
    %10 = stablehlo.add %c_1, %iterArg : tensor<i32>
    %13 = stablehlo.dynamic_slice %arg0, %c, %10, %c, sizes = [12, 1, 4] : (tensor<12x16x4xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<12x1x4xf32>
    %15 = stablehlo.reshape %13 : (tensor<12x1x4xf32>) -> tensor<12x4xf32>
    stablehlo.return %10, %15 : tensor<i32>, tensor<12x4xf32>
  }
  return %8#1 : tensor<12x4xf32>
}

// FORWARD:  func.func @main(%arg0: tensor<12x16x4xf32>, %arg1: tensor<12x16x4xf32>) -> (tensor<12x4xf32>, tensor<12x4xf32>) {
// FORWARD-NEXT:    %c = stablehlo.constant dense<0> : tensor<i32>
// FORWARD-NEXT:    %c_0 = stablehlo.constant dense<15> : tensor<i32>
// FORWARD-NEXT:    %c_1 = stablehlo.constant dense<1> : tensor<i32>
// FORWARD-NEXT:    %0 = stablehlo.slice %arg1 [0:12, 0:1, 0:4] : (tensor<12x16x4xf32>) -> tensor<12x1x4xf32>
// FORWARD-NEXT:    %1 = stablehlo.slice %arg0 [0:12, 0:1, 0:4] : (tensor<12x16x4xf32>) -> tensor<12x1x4xf32>
// FORWARD-NEXT:    %2 = stablehlo.reshape %0 : (tensor<12x1x4xf32>) -> tensor<12x4xf32>
// FORWARD-NEXT:    %3 = stablehlo.reshape %1 : (tensor<12x1x4xf32>) -> tensor<12x4xf32>
// FORWARD-NEXT:    %4:3 = stablehlo.while(%iterArg = %c, %iterArg_2 = %3, %iterArg_3 = %2) : tensor<i32>, tensor<12x4xf32>, tensor<12x4xf32>
// FORWARD-NEXT:     cond {
// FORWARD-NEXT:      %5 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i32>, tensor<i32>) -> tensor<i1>
// FORWARD-NEXT:      stablehlo.return %5 : tensor<i1>
// FORWARD-NEXT:    } do {
// FORWARD-NEXT:      %5 = stablehlo.add %c_1, %iterArg : tensor<i32>
// FORWARD-NEXT:      %6 = stablehlo.dynamic_slice %arg1, %c, %5, %c, sizes = [12, 1, 4] : (tensor<12x16x4xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<12x1x4xf32>
// FORWARD-NEXT:      %7 = stablehlo.dynamic_slice %arg0, %c, %5, %c, sizes = [12, 1, 4] : (tensor<12x16x4xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<12x1x4xf32>
// FORWARD-NEXT:      %8 = stablehlo.reshape %6 : (tensor<12x1x4xf32>) -> tensor<12x4xf32>
// FORWARD-NEXT:      %9 = stablehlo.reshape %7 : (tensor<12x1x4xf32>) -> tensor<12x4xf32>
// FORWARD-NEXT:      stablehlo.return %5, %9, %8 : tensor<i32>, tensor<12x4xf32>, tensor<12x4xf32>
// FORWARD-NEXT:    }
// FORWARD-NEXT:    return %4#1, %4#2 : tensor<12x4xf32>, tensor<12x4xf32>
// FORWARD-NEXT:  }

// REVERSE: func.func @main(%arg0: tensor<12x16x4xf32>, %arg1: tensor<12x4xf32>) -> tensor<12x16x4xf32> {
// REVERSE-DAG:     %[[C14:.+]] = stablehlo.constant dense<14> : tensor<i64>
// REVERSE-DAG:     %[[ZERO:.+]] = arith.constant dense<0> : tensor<15xi32>
// REVERSE-DAG:     %[[C15:.+]] = stablehlo.constant dense<15> : tensor<i64>
// REVERSE-DAG:     %[[cst_1:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<12x16x4xf32>
// REVERSE-DAG:     %[[c_2:.+]] = stablehlo.constant dense<1> : tensor<i64>
// REVERSE-DAG:     %[[c_3:.+]] = stablehlo.constant dense<0> : tensor<i64>
// REVERSE-DAG:     %[[c_4:.+]] = stablehlo.constant dense<1> : tensor<i32>
// REVERSE-DAG:     %[[C15_I32:.+]] = stablehlo.constant dense<15> : tensor<i32>
// REVERSE-DAG:     %[[c_6:.+]] = stablehlo.constant dense<0> : tensor<i32>
// REVERSE-DAG:     %[[cst_7:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<12x4xf32>
// REVERSE-NEXT:     %[[a0:.+]]:2 = stablehlo.while(%iterArg = %[[c_6]], %iterArg_8 = %[[ZERO]]) : tensor<i32>, tensor<15xi32>
// REVERSE-NEXT:      cond {
// REVERSE-NEXT:       %[[a7:.+]] = stablehlo.compare  LT, %iterArg, %[[C15_I32]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
// REVERSE-NEXT:       stablehlo.return %[[a7]] : tensor<i1>
// REVERSE-NEXT:     } do {
// REVERSE-NEXT:       %[[a7:.+]] = stablehlo.add %[[c_4]], %iterArg : tensor<i32>
// REVERSE-NEXT:       %[[a8:.+]] = stablehlo.reshape %[[a7]] : (tensor<i32>) -> tensor<1xi32>
// REVERSE-NEXT:       %[[a9:.+]] = stablehlo.dynamic_update_slice %iterArg_8, %[[a8]], %iterArg : (tensor<15xi32>, tensor<1xi32>, tensor<i32>) -> tensor<15xi32>
// REVERSE-NEXT:       stablehlo.return %[[a7]], %[[a9]] : tensor<i32>, tensor<15xi32>
// REVERSE-NEXT:     }
// REVERSE-NEXT:     %[[a1:.+]]:3 = stablehlo.while(%iterArg = %[[c_3]], %iterArg_8 = %[[C14]], %iterArg_9 = %[[cst_1]]) : tensor<i64>, tensor<i64>, tensor<12x16x4xf32>
// REVERSE-NEXT:      cond {
// REVERSE-NEXT:       %[[a7:.+]] = stablehlo.compare  LT, %iterArg, %[[C15]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
// REVERSE-NEXT:       stablehlo.return %[[a7]] : tensor<i1>
// REVERSE-NEXT:     } do {
// REVERSE-NEXT:       %[[a7:.+]] = stablehlo.compare  EQ, %iterArg, %[[c_3]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
// REVERSE-NEXT:       %[[a8:.+]] = stablehlo.select %[[a7]], %arg1, %[[cst_7]] : tensor<i1>, tensor<12x4xf32>
// REVERSE-NEXT:       %[[a11:.+]] = stablehlo.dynamic_slice %[[a0]]#1, %iterArg_8, sizes = [1] : (tensor<15xi32>, tensor<i64>) -> tensor<1xi32>
// REVERSE-NEXT:       %[[a12:.+]] = stablehlo.reshape %[[a11]] : (tensor<1xi32>) -> tensor<i32>
// REVERSE-NEXT:       %[[a9:.+]] = stablehlo.add %iterArg, %[[c_2]] : tensor<i64>
// REVERSE-NEXT:       %[[a10:.+]] = stablehlo.reshape %[[a8]] : (tensor<12x4xf32>) -> tensor<12x1x4xf32>
// REVERSE-NEXT:       %[[a13:.+]] = stablehlo.dynamic_update_slice %[[cst_1]], %[[a10]], %[[c_6]], %[[a12]], %[[c_6]] : (tensor<12x16x4xf32>, tensor<12x1x4xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<12x16x4xf32>
// REVERSE-NEXT:       %[[a14:.+]] = stablehlo.add %iterArg_9, %[[a13]] : tensor<12x16x4xf32>
// REVERSE-NEXT:       %[[a15:.+]] = stablehlo.subtract %iterArg_8, %[[c_2]] : tensor<i64>
// REVERSE-NEXT:       stablehlo.return %[[a9]], %[[a15]], %[[a14]] : tensor<i64>, tensor<i64>, tensor<12x16x4xf32>
// REVERSE-NEXT:     }
// REVERSE-NEXT:     return %[[a1]]#2 : tensor<12x16x4xf32>
// REVERSE-NEXT:   }

