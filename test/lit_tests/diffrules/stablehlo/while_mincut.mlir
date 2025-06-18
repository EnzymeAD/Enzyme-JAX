// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize | FileCheck %s

module {
  func.func public @main(%arg0: tensor<3xf64>) -> (tensor<3xf64>) {
    %c = stablehlo.constant dense<0> : tensor<i64>
    %c_10 = stablehlo.constant dense<10> : tensor<i64>
    %c_1 = stablehlo.constant dense<1> : tensor<i64>
    %0:2 = stablehlo.while(%iterArg = %c, %iterArg_0 = %arg0) : tensor<i64>, tensor<3xf64>
     cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_10,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %2 = stablehlo.add %iterArg, %c_1 : tensor<i64>
      %3 = stablehlo.cosine %iterArg_0 : tensor<3xf64>
      %4 = stablehlo.multiply %3, %3 : tensor<3xf64>
      stablehlo.return %2, %4 : tensor<i64>, tensor<3xf64>
    }
    return %0#1 : tensor<3xf64>
  }
}

// CHECK:  func.func @main(%[[ARG0:.+]]: tensor<3xf64>, %[[ARG1:.+]]: tensor<3xf64>) -> tensor<3xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<9> : tensor<i64>
// CHECK-NEXT:    %cst = arith.constant dense<0.000000e+00> : tensor<10x3xf64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<10> : tensor<i64>
// CHECK-NEXT:    %[[ZEROI:.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %[[ZERO:.+]] = arith.constant dense<0.000000e+00> : tensor<3xf64>
// CHECK-NEXT:    %[[FWD:.+]]:3 = stablehlo.while(%iterArg = %[[ZEROI]], %iterArg_4 = %[[ARG0]], %iterArg_5 = %cst) : tensor<i64>, tensor<3xf64>, tensor<10x3xf64>
// CHECK-NEXT:     cond {
// CHECK-NEXT:      %[[CMP:.+]] = stablehlo.compare  LT, %iterArg, %c_1,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %[[CMP]] : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %[[v6:.+]] = stablehlo.reshape %iterArg_4 : (tensor<3xf64>) -> tensor<1x3xf64>
// CHECK-NEXT:      %[[v7:.+]] = stablehlo.dynamic_update_slice %iterArg_5, %[[v6]], %iterArg, %c_2 : (tensor<10x3xf64>, tensor<1x3xf64>, tensor<i64>, tensor<i64>) -> tensor<10x3xf64>
// CHECK-NEXT:      %[[v8:.+]] = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:      %[[v9:.+]] = stablehlo.cosine %iterArg_4 : tensor<3xf64>
// CHECK-NEXT:      %[[v10:.+]] = stablehlo.multiply %[[v9]], %[[v9]] : tensor<3xf64>
// CHECK-NEXT:      stablehlo.return %[[v8]], %[[v10]], %[[v7]] : tensor<i64>, tensor<3xf64>, tensor<10x3xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[ADD:.+]] = arith.addf %[[ARG1]], %[[ZERO]] : tensor<3xf64>
// CHECK-NEXT:    %[[REV:.+]]:3 = stablehlo.while(%iterArg = %[[ZEROI]], %iterArg_4 = %[[ADD]], %iterArg_5 = %c) : tensor<i64>, tensor<3xf64>, tensor<i64>
// CHECK-NEXT:     cond {
// CHECK-NEXT:      %[[CMP:.+]] = stablehlo.compare  LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %[[CMP]] : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %[[v6:.+]] = stablehlo.dynamic_slice %[[FWD]]#2, %iterArg_5, %c_2, sizes = [1, 3] : (tensor<10x3xf64>, tensor<i64>, tensor<i64>) -> tensor<1x3xf64>
// CHECK-NEXT:      %[[v7:.+]] = stablehlo.reshape %[[v6]] : (tensor<1x3xf64>) -> tensor<3xf64>
// CHECK-NEXT:      %[[v8:.+]] = stablehlo.cosine %[[v7]] : tensor<3xf64>
// CHECK-NEXT:      %[[v9:.+]] = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:      %[[a4:.+]] = arith.addf %iterArg_4, %[[ZERO]] : tensor<3xf64>
// CHECK-NEXT:      %[[v10:.+]] = stablehlo.multiply %[[a4]], %[[v8]] : tensor<3xf64>
// CHECK-NEXT:      %[[v11:.+]] = arith.addf %[[v10]], %[[ZERO]] : tensor<3xf64>
// CHECK-NEXT:      %[[v12:.+]] = stablehlo.multiply %[[a4]], %[[v8]] : tensor<3xf64>
// CHECK-NEXT:      %[[v13:.+]] = arith.addf %[[v11]], %[[v12]] : tensor<3xf64>
// CHECK-NEXT:      %[[v14:.+]] = stablehlo.sine %[[v7]] : tensor<3xf64>
// CHECK-NEXT:      %[[v15:.+]] = stablehlo.negate %[[v14]] : tensor<3xf64>
// CHECK-NEXT:      %[[v16:.+]] = stablehlo.multiply %[[v13]], %[[v15]] : tensor<3xf64>
// CHECK-NEXT:      %[[v17:.+]] = arith.addf %[[v16]], %[[ZERO]] : tensor<3xf64>
// CHECK-NEXT:      %[[v18:.+]] = stablehlo.subtract %iterArg_5, %c_0 : tensor<i64>
// CHECK-NEXT:      stablehlo.return %[[v9]], %[[v17]], %[[v18]] : tensor<i64>, tensor<3xf64>, tensor<i64>
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[v5:.+]] = arith.addf %[[REV]]#1, %[[ZERO]] : tensor<3xf64>
// CHECK-NEXT:    return %[[v5]] : tensor<3xf64>
// CHECK-NEXT:  }
