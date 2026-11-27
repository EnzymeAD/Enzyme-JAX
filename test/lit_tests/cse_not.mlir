// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s
// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=cse_not<16>},transform-interpreter,enzyme-hlo-remove-transform)" | FileCheck %s

// Regression test for #490: deduplicate identical stablehlo.not operations.

module {
  func.func @cse_not_mask(%m: tensor<90x180xi1>, %a: tensor<90x180xi1>, %b: tensor<90x180xi1>, %c: tensor<90x180xi1>) -> (tensor<90x180xi1>, tensor<90x180xi1>) {
    %nm0 = stablehlo.not %m : tensor<90x180xi1>
    %x0 = stablehlo.and %a, %nm0 : tensor<90x180xi1>
    %nm1 = stablehlo.not %m : tensor<90x180xi1>
    %x1 = stablehlo.and %b, %nm1 : tensor<90x180xi1>
    %u = stablehlo.or %x0, %x1 : tensor<90x180xi1>
    %nm2 = stablehlo.not %m : tensor<90x180xi1>
    %x2 = stablehlo.and %c, %nm2 : tensor<90x180xi1>
    return %u, %x2 : tensor<90x180xi1>, tensor<90x180xi1>
  }
}

// CHECK-LABEL: func.func @cse_not_mask(
// CHECK: %[[NM:.+]] = stablehlo.not %arg0 : tensor<90x180xi1>
// CHECK-NOT: stablehlo.not
// CHECK: return
