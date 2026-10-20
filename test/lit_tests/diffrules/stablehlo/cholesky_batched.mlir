// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s

func.func @main(%a : tensor<2x40x40xf64>) -> tensor<2x40x40xf64> {
  %0 = stablehlo.cholesky %a, lower = false : tensor<2x40x40xf64>
  return %0 : tensor<2x40x40xf64>
}

// CHECK-LABEL: func.func @main(
// CHECK-SAME:      %[[A:.+]]: tensor<2x40x40xf64>, %[[UBAR:.+]]: tensor<2x40x40xf64>) -> tensor<2x40x40xf64> {
// CHECK:         %[[U:.+]] = stablehlo.cholesky %[[A]] : tensor<2x40x40xf64>
// CHECK-NEXT:    %{{.+}} = stablehlo.slice %[[U]] [0:2, 20:40, 20:40] : (tensor<2x40x40xf64>) -> tensor<2x20x20xf64>
// CHECK:         %[[T:.+]] = stablehlo.multiply %{{.+}}, %{{.+}} : tensor<2x20x20xf64>
// CHECK-NEXT:    %[[D0:.+]] = stablehlo.slice %[[U]] [0:2, 0:20, 0:20] : (tensor<2x40x40xf64>) -> tensor<2x20x20xf64>
// CHECK-NEXT:    %[[DB0:.+]] = stablehlo.slice %[[UBAR]] [0:2, 0:20, 0:20] : (tensor<2x40x40xf64>) -> tensor<2x20x20xf64>
// CHECK-NEXT:    %[[R:.+]] = stablehlo.iota dim = 1 : tensor<2x20x20xi64>
// CHECK-NEXT:    %[[C:.+]] = stablehlo.iota dim = 2 : tensor<2x20x20xi64>
// CHECK-NEXT:    %{{.+}} = stablehlo.compare EQ, %[[R]], %[[C]] : (tensor<2x20x20xi64>, tensor<2x20x20xi64>) -> tensor<2x20x20xi1>
// CHECK-NEXT:    %[[EYE:.+]] = stablehlo.select %{{.+}}, %{{.+}}, %{{.+}} : tensor<2x20x20xi1>, tensor<2x20x20xf64>
// CHECK-NEXT:    %[[DINVT:.+]] = "stablehlo.triangular_solve"(%[[D0]], %[[EYE]]) <{left_side = true, lower = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<2x20x20xf64>, tensor<2x20x20xf64>) -> tensor<2x20x20xf64>
// CHECK-NEXT:    %[[CT:.+]] = stablehlo.slice %[[U]] [0:2, 0:20, 20:40] : (tensor<2x40x40xf64>) -> tensor<2x20x20xf64>
// CHECK-NEXT:    %[[CBT:.+]] = stablehlo.slice %[[UBAR]] [0:2, 0:20, 20:40] : (tensor<2x40x40xf64>) -> tensor<2x20x20xf64>
// CHECK-NEXT:    %[[CTT:.+]] = stablehlo.dot_general %[[CT]], %[[T]], batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<2x20x20xf64>, tensor<2x20x20xf64>) -> tensor<2x20x20xf64>
// CHECK-NEXT:    %[[CTT2:.+]] = stablehlo.multiply %[[CTT]], %{{.+}} : tensor<2x20x20xf64>
// CHECK-NEXT:    %[[CBART:.+]] = stablehlo.subtract %[[CBT]], %[[CTT2]] : tensor<2x20x20xf64>
// CHECK-NEXT:    %[[MT:.+]] = stablehlo.dot_general %[[DINVT]], %[[CBART]], batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<2x20x20xf64>, tensor<2x20x20xf64>) -> tensor<2x20x20xf64>
// CHECK-NEXT:    %[[NEG:.+]] = stablehlo.negate %[[CBART]] : tensor<2x20x20xf64>
// CHECK-NEXT:    %[[LHS:.+]] = stablehlo.concatenate %[[D0]], %[[NEG]], dim = 2 : (tensor<2x20x20xf64>, tensor<2x20x20xf64>) -> tensor<2x20x40xf64>
// CHECK-NEXT:    %[[RHS:.+]] = stablehlo.concatenate %[[DB0]], %[[CT]], dim = 2 : (tensor<2x20x20xf64>, tensor<2x20x20xf64>) -> tensor<2x20x40xf64>
// CHECK-NEXT:    %[[S:.+]] = stablehlo.dot_general %[[LHS]], %[[RHS]], batching_dims = [0] x [0], contracting_dims = [2] x [2] : (tensor<2x20x40xf64>, tensor<2x20x40xf64>) -> tensor<2x20x20xf64>
// CHECK:         %[[ST:.+]] = stablehlo.transpose %[[S]], dims = [0, 2, 1] : (tensor<2x20x20xf64>) -> tensor<2x20x20xf64>
// CHECK:         %[[TOP:.+]] = stablehlo.concatenate %{{.+}}, %{{.+}}, dim = 2 : (tensor<2x20x20xf64>, tensor<2x20x20xf64>) -> tensor<2x20x40xf64>
// CHECK-NEXT:    %[[BOT:.+]] = stablehlo.concatenate %{{.+}}, %[[T]], dim = 2 : (tensor<2x20x20xf64>, tensor<2x20x20xf64>) -> tensor<2x20x40xf64>
// CHECK-NEXT:    %{{.+}} = stablehlo.concatenate %[[TOP]], %[[BOT]], dim = 1 : (tensor<2x20x40xf64>, tensor<2x20x40xf64>) -> tensor<2x40x40xf64>
