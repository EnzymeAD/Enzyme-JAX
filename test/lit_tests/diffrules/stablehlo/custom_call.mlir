// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_const mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_const mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=KEEP

// A custom call whose adjoint is supplied by the call site: `@scale` multiplies
// by 3 and also returns a residual, so its VJP is a single multiply of the
// output cotangent by 3. The i1 mask operand is excluded from
// enzyme.active_operands, and the residual's cotangent is never used.

func.func @scale_rev(%x: tensor<4xf32>, %mask: tensor<4xi1>,
                     %y: tensor<4xf32>, %sum: tensor<f32>,
                     %dy: tensor<4xf32>, %dsum: tensor<f32>) -> tensor<4xf32> {
  %three = stablehlo.constant dense<3.000000e+00> : tensor<4xf32>
  %res = stablehlo.multiply %dy, %three : tensor<4xf32>
  func.return %res : tensor<4xf32>
}

func.func @main(%x : tensor<4xf32>, %mask : tensor<4xi1>) -> tensor<4xf32> {
  %y:2 = stablehlo.custom_call @scale(%x, %mask) {
    enzyme.reverse = @scale_rev,
    enzyme.active_operands = array<i64: 0>
  } : (tensor<4xf32>, tensor<4xi1>) -> (tensor<4xf32>, tensor<f32>)
  func.return %y#0 : tensor<4xf32>
}

// The reverse function is only referenced from an attribute, and survives.
// KEEP-DAG: func.func @scale_rev(
// KEEP-DAG: func.func @main(

// REVERSE: func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xi1>, %arg2: tensor<4xf32>) -> tensor<4xf32> {
// The primal call is still there, and the cotangent goes through @scale_rev,
// which takes (primal operands..., primal results..., result cotangents...)
// and returns exactly one cotangent -- for operand 0, the only active one.
// REVERSE:   %{{.+}}:2 = stablehlo.custom_call @scale(%arg0, %arg1)
// REVERSE:   call @scale_rev(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) : (tensor<4xf32>, tensor<4xi1>, tensor<4xf32>, tensor<f32>, tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
// REVERSE:   return
