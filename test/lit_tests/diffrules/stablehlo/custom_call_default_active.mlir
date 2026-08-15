// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_active,enzyme_const mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

// Without enzyme.active_operands the active operands default to the
// float-typed ones, so the mask is skipped and @axpy_rev returns two
// cotangents, for operands 0 and 1 in that order.

func.func @axpy_rev(%a: tensor<4xf32>, %b: tensor<4xf32>, %mask: tensor<4xi1>,
                    %y: tensor<4xf32>,
                    %dy: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %two = stablehlo.constant dense<2.000000e+00> : tensor<4xf32>
  %da = stablehlo.multiply %dy, %two : tensor<4xf32>
  func.return %da, %dy : tensor<4xf32>, tensor<4xf32>
}

func.func @main(%a : tensor<4xf32>, %b : tensor<4xf32>, %mask : tensor<4xi1>) -> tensor<4xf32> {
  %y = stablehlo.custom_call @axpy(%a, %b, %mask) {
    enzyme.reverse = @axpy_rev
  } : (tensor<4xf32>, tensor<4xf32>, tensor<4xi1>) -> tensor<4xf32>
  func.return %y : tensor<4xf32>
}

// REVERSE: func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xi1>, %arg3: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
// REVERSE:   stablehlo.custom_call @axpy(%arg0, %arg1, %arg2)
// REVERSE:   call @axpy_rev(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) : (tensor<4xf32>, tensor<4xf32>, tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)
// REVERSE:   return
