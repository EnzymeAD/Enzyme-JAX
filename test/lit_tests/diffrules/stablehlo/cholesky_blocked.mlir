// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=upper outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=UPPER
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=lower outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=LOWER

// Reversed in two blocks of 20 rows, last block first.

func.func @upper(%a : tensor<40x40xf64>) -> tensor<40x40xf64> {
  %0 = stablehlo.cholesky %a, lower = false : tensor<40x40xf64>
  return %0 : tensor<40x40xf64>
}

func.func @lower(%a : tensor<40x40xf64>) -> tensor<40x40xf64> {
  %0 = stablehlo.cholesky %a, lower = true : tensor<40x40xf64>
  return %0 : tensor<40x40xf64>
}

// UPPER-LABEL: func.func @upper(
// UPPER-SAME:      %[[A:.+]]: tensor<40x40xf64>, %[[UBAR:.+]]: tensor<40x40xf64>) -> tensor<40x40xf64> {
// UPPER-DAG:     %[[HALF:.+]] = stablehlo.constant dense<5.000000e-01> : tensor<20x20xf64>
// UPPER-DAG:     %[[TWO:.+]] = stablehlo.constant dense<2.000000e+00> : tensor<20x20xf64>
// UPPER-DAG:     %[[QUARTER:.+]] = stablehlo.constant dense<2.500000e-01> : tensor<20x20xf64>
// UPPER:         %[[U:.+]] = stablehlo.cholesky %[[A]] : tensor<40x40xf64>
// UPPER-NEXT:    %[[D1:.+]] = stablehlo.slice %[[U]] [20:40, 20:40] : (tensor<40x40xf64>) -> tensor<20x20xf64>
// UPPER-NEXT:    %[[DB1:.+]] = stablehlo.slice %[[UBAR]] [20:40, 20:40] : (tensor<40x40xf64>) -> tensor<20x20xf64>
// UPPER:         %[[DINVT1:.+]] = "stablehlo.triangular_solve"(%[[D1]], %{{.+}}) <{left_side = true, lower = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
// UPPER-NEXT:    %{{.+}} = stablehlo.dot_general %[[D1]], %[[DB1]], contracting_dims = [1] x [1] : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
// UPPER:         %[[T:.+]] = stablehlo.multiply %{{.+}}, %[[QUARTER]] : tensor<20x20xf64>
// UPPER-NEXT:    %[[D0:.+]] = stablehlo.slice %[[U]] [0:20, 0:20] : (tensor<40x40xf64>) -> tensor<20x20xf64>
// UPPER-NEXT:    %[[DB0:.+]] = stablehlo.slice %[[UBAR]] [0:20, 0:20] : (tensor<40x40xf64>) -> tensor<20x20xf64>
// UPPER-NEXT:    %{{.+}} = stablehlo.iota dim = 0 : tensor<20x20xi64>
// UPPER-NEXT:    %{{.+}} = stablehlo.iota dim = 1 : tensor<20x20xi64>
// UPPER-NEXT:    %{{.+}} = stablehlo.compare EQ, %{{.+}}, %{{.+}} : (tensor<20x20xi64>, tensor<20x20xi64>) -> tensor<20x20xi1>
// UPPER-NEXT:    %[[EYE:.+]] = stablehlo.select %{{.+}}, %{{.+}}, %{{.+}} : tensor<20x20xi1>, tensor<20x20xf64>
// UPPER-NEXT:    %[[DINVT0:.+]] = "stablehlo.triangular_solve"(%[[D0]], %[[EYE]]) <{left_side = true, lower = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
// UPPER-NEXT:    %[[CT:.+]] = stablehlo.slice %[[U]] [0:20, 20:40] : (tensor<40x40xf64>) -> tensor<20x20xf64>
// UPPER-NEXT:    %[[CBT:.+]] = stablehlo.slice %[[UBAR]] [0:20, 20:40] : (tensor<40x40xf64>) -> tensor<20x20xf64>
// UPPER-NEXT:    %[[CTT:.+]] = stablehlo.dot_general %[[CT]], %[[T]], contracting_dims = [1] x [0] : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
// UPPER-NEXT:    %[[CTT2:.+]] = stablehlo.multiply %[[CTT]], %[[TWO]] : tensor<20x20xf64>
// UPPER-NEXT:    %[[CBART:.+]] = stablehlo.subtract %[[CBT]], %[[CTT2]] : tensor<20x20xf64>
// UPPER-NEXT:    %[[MT:.+]] = stablehlo.dot_general %[[DINVT0]], %[[CBART]], contracting_dims = [1] x [0] : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
// UPPER-NEXT:    %[[NEG:.+]] = stablehlo.negate %[[CBART]] : tensor<20x20xf64>
// UPPER-NEXT:    %[[LHS:.+]] = stablehlo.concatenate %[[D0]], %[[NEG]], dim = 1 : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x40xf64>
// UPPER-NEXT:    %[[RHS:.+]] = stablehlo.concatenate %[[DB0]], %[[CT]], dim = 1 : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x40xf64>
// UPPER-NEXT:    %[[S:.+]] = stablehlo.dot_general %[[LHS]], %[[RHS]], contracting_dims = [1] x [1] : (tensor<20x40xf64>, tensor<20x40xf64>) -> tensor<20x20xf64>
// UPPER-NEXT:    %[[HALFMT:.+]] = stablehlo.multiply %[[MT]], %[[HALF]] : tensor<20x20xf64>
// UPPER-NEXT:    %{{.+}} = stablehlo.iota dim = 0 : tensor<20x20xi64>
// UPPER-NEXT:    %{{.+}} = stablehlo.iota dim = 1 : tensor<20x20xi64>
// UPPER-NEXT:    %[[LOW:.+]] = stablehlo.compare GE, %{{.+}}, %{{.+}} : (tensor<20x20xi64>, tensor<20x20xi64>) -> tensor<20x20xi1>
// UPPER-NEXT:    %[[ST:.+]] = stablehlo.transpose %[[S]], dims = [1, 0] : (tensor<20x20xf64>) -> tensor<20x20xf64>
// UPPER-NEXT:    %[[P:.+]] = stablehlo.select %[[LOW]], %[[S]], %[[ST]] : tensor<20x20xi1>, tensor<20x20xf64>
// UPPER-NEXT:    %[[W:.+]] = stablehlo.dot_general %[[DINVT0]], %[[P]], contracting_dims = [1] x [0] : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
// UPPER-NEXT:    %[[Z:.+]] = stablehlo.dot_general %[[W]], %[[DINVT0]], contracting_dims = [1] x [1] : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
// UPPER-NEXT:    %[[ZT:.+]] = stablehlo.transpose %[[Z]], dims = [1, 0] : (tensor<20x20xf64>) -> tensor<20x20xf64>
// UPPER-NEXT:    %[[ZS:.+]] = stablehlo.add %[[Z]], %[[ZT]] : tensor<20x20xf64>
// UPPER-NEXT:    %[[AJJ:.+]] = stablehlo.multiply %[[ZS]], %[[QUARTER]] : tensor<20x20xf64>
// UPPER-NEXT:    %[[HALFM:.+]] = stablehlo.transpose %[[HALFMT]], dims = [1, 0] : (tensor<20x20xf64>) -> tensor<20x20xf64>
// UPPER-NEXT:    %[[TOP:.+]] = stablehlo.concatenate %[[AJJ]], %[[HALFMT]], dim = 1 : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x40xf64>
// UPPER-NEXT:    %[[BOT:.+]] = stablehlo.concatenate %[[HALFM]], %[[T]], dim = 1 : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x40xf64>
// UPPER-NEXT:    %[[ABAR:.+]] = stablehlo.concatenate %[[TOP]], %[[BOT]], dim = 0 : (tensor<20x40xf64>, tensor<20x40xf64>) -> tensor<40x40xf64>
// UPPER-NEXT:    return %[[ABAR]] : tensor<40x40xf64>

// LOWER-LABEL: func.func @lower(
// LOWER-SAME:      %[[A:.+]]: tensor<40x40xf64>, %[[LBAR:.+]]: tensor<40x40xf64>) -> tensor<40x40xf64> {
// LOWER:         %[[L:.+]] = stablehlo.cholesky %[[A]], lower = true : tensor<40x40xf64>
// LOWER-NEXT:    %{{.+}} = stablehlo.slice %[[L]] [20:40, 20:40] : (tensor<40x40xf64>) -> tensor<20x20xf64>
// LOWER:         %[[T:.+]] = stablehlo.multiply %{{.+}}, %{{.+}} : tensor<20x20xf64>
// LOWER-NEXT:    %[[D0:.+]] = stablehlo.slice %[[L]] [0:20, 0:20] : (tensor<40x40xf64>) -> tensor<20x20xf64>
// LOWER-NEXT:    %[[DB0:.+]] = stablehlo.slice %[[LBAR]] [0:20, 0:20] : (tensor<40x40xf64>) -> tensor<20x20xf64>
// LOWER:         %[[DINVT0:.+]] = "stablehlo.triangular_solve"(%[[D0]], %{{.+}}) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose TRANSPOSE>, unit_diagonal = false}> : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
// LOWER-NEXT:    %[[C:.+]] = stablehlo.slice %[[L]] [20:40, 0:20] : (tensor<40x40xf64>) -> tensor<20x20xf64>
// LOWER-NEXT:    %[[CB:.+]] = stablehlo.slice %[[LBAR]] [20:40, 0:20] : (tensor<40x40xf64>) -> tensor<20x20xf64>
// LOWER-NEXT:    %[[TC:.+]] = stablehlo.dot_general %[[T]], %[[C]], contracting_dims = [1] x [0] : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
// LOWER-NEXT:    %[[TC2:.+]] = stablehlo.multiply %[[TC]], %{{.+}} : tensor<20x20xf64>
// LOWER-NEXT:    %[[CBAR:.+]] = stablehlo.subtract %[[CB]], %[[TC2]] : tensor<20x20xf64>
// LOWER-NEXT:    %[[M:.+]] = stablehlo.dot_general %[[CBAR]], %[[DINVT0]], contracting_dims = [1] x [1] : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
// LOWER-NEXT:    %[[NEGC:.+]] = stablehlo.negate %[[C]] : tensor<20x20xf64>
// LOWER-NEXT:    %[[LHS:.+]] = stablehlo.concatenate %[[D0]], %[[CBAR]], dim = 0 : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<40x20xf64>
// LOWER-NEXT:    %[[RHS:.+]] = stablehlo.concatenate %[[DB0]], %[[NEGC]], dim = 0 : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<40x20xf64>
// LOWER-NEXT:    %{{.+}} = stablehlo.dot_general %[[LHS]], %[[RHS]], contracting_dims = [0] x [0] : (tensor<40x20xf64>, tensor<40x20xf64>) -> tensor<20x20xf64>
// LOWER-NEXT:    %[[HALFM:.+]] = stablehlo.multiply %[[M]], %{{.+}} : tensor<20x20xf64>
// LOWER:         %[[AJJ:.+]] = stablehlo.multiply %{{.+}}, %{{.+}} : tensor<20x20xf64>
// LOWER-NEXT:    %[[HALFMT:.+]] = stablehlo.transpose %[[HALFM]], dims = [1, 0] : (tensor<20x20xf64>) -> tensor<20x20xf64>
// LOWER-NEXT:    %[[TOP:.+]] = stablehlo.concatenate %[[AJJ]], %[[HALFMT]], dim = 1 : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x40xf64>
// LOWER-NEXT:    %[[BOT:.+]] = stablehlo.concatenate %[[HALFM]], %[[T]], dim = 1 : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x40xf64>
// LOWER-NEXT:    %{{.+}} = stablehlo.concatenate %[[TOP]], %[[BOT]], dim = 0 : (tensor<20x40xf64>, tensor<20x40xf64>) -> tensor<40x40xf64>
