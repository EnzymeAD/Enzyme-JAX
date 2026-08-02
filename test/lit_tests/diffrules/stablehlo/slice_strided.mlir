// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise | FileCheck %s --check-prefix=MAIN
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=offset outfn= argTys=enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise | FileCheck %s --check-prefix=OFFSET
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=exact outfn= argTys=enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise | FileCheck %s --check-prefix=EXACT
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=unit outfn= argTys=enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise | FileCheck %s --check-prefix=UNIT

// Trailing pad is dim - (start + (outDim - 1) * stride) - 1, not dim - limit.
func.func @main(%x: tensor<10xf32>) -> tensor<3xf32> {
  %0 = stablehlo.slice %x [0:6:2] : (tensor<10xf32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// MAIN-LABEL: func.func @main
// MAIN-SAME:    (%[[X:.+]]: tensor<10xf32>, %[[DR:.+]]: tensor<3xf32>) -> tensor<10xf32>
// MAIN:         %[[PAD:.+]] = stablehlo.pad %{{.+}}, %{{.+}}, low = [0], high = [5], interior = [1] : (tensor<3xf32>, tensor<f32>) -> tensor<10xf32>

func.func @offset(%x: tensor<12xf32>) -> tensor<3xf32> {
  %0 = stablehlo.slice %x [1:9:3] : (tensor<12xf32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// OFFSET-LABEL: func.func @offset
// OFFSET-SAME:    (%[[X:.+]]: tensor<12xf32>, %[[DR:.+]]: tensor<3xf32>) -> tensor<12xf32>
// OFFSET:         %[[PAD:.+]] = stablehlo.pad %{{.+}}, %{{.+}}, low = [1], high = [4], interior = [2] : (tensor<3xf32>, tensor<f32>) -> tensor<12xf32>

func.func @exact(%x: tensor<10xf32>) -> tensor<4xf32> {
  %0 = stablehlo.slice %x [0:8:2] : (tensor<10xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// EXACT-LABEL: func.func @exact
// EXACT-SAME:    (%[[X:.+]]: tensor<10xf32>, %[[DR:.+]]: tensor<4xf32>) -> tensor<10xf32>
// EXACT:         %[[PAD:.+]] = stablehlo.pad %{{.+}}, %{{.+}}, low = [0], high = [3], interior = [1] : (tensor<4xf32>, tensor<f32>) -> tensor<10xf32>

func.func @unit(%x: tensor<10xf32>) -> tensor<6xf32> {
  %0 = stablehlo.slice %x [0:6:1] : (tensor<10xf32>) -> tensor<6xf32>
  return %0 : tensor<6xf32>
}

// UNIT-LABEL: func.func @unit
// UNIT-SAME:    (%[[X:.+]]: tensor<10xf32>, %[[DR:.+]]: tensor<6xf32>) -> tensor<10xf32>
// UNIT:         %[[PAD:.+]] = stablehlo.pad %{{.+}}, %{{.+}}, low = [0], high = [4], interior = [0] : (tensor<6xf32>, tensor<f32>) -> tensor<10xf32>
