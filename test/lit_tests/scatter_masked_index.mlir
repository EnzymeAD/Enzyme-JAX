// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

// A masked store scatters with its index sent out of bounds on the dead
// path (scatter drops it). The single-point scatter must not become a
// plain dynamic_update_slice — DUS clamps the index back in bounds — so
// the mask is resolved first: slice the original value and write it back
// when masked off.
func.func @main(%buf: tensor<50xf64>, %i: tensor<i64>, %p: tensor<i1>, %v: tensor<f64>) -> tensor<50xf64> {
  %cm1 = stablehlo.constant dense<-1> : tensor<1xi64>
  %ir = stablehlo.reshape %i : (tensor<i64>) -> tensor<1xi64>
  %pb = stablehlo.reshape %p : (tensor<i1>) -> tensor<1xi1>
  %sel = stablehlo.select %pb, %ir, %cm1 : tensor<1xi1>, tensor<1xi64>
  %r = "stablehlo.scatter"(%buf, %sel, %v) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<50xf64>, tensor<1xi64>, tensor<f64>) -> tensor<50xf64>
  return %r : tensor<50xf64>
}

// CHECK-LABEL: func.func @main(
// CHECK:         %[[ORIG:.+]] = stablehlo.dynamic_slice %arg0, %arg1, sizes = [1]
// CHECK:         %[[ORIGS:.+]] = stablehlo.reshape %[[ORIG]] : (tensor<1xf64>) -> tensor<f64>
// CHECK:         %[[UPD:.+]] = stablehlo.select %arg2, %arg3, %[[ORIGS]]
// CHECK:         %[[UPD1:.+]] = stablehlo.reshape %[[UPD]] : (tensor<f64>) -> tensor<1xf64>
// CHECK:         %[[RES:.+]] = stablehlo.dynamic_update_slice %arg0, %[[UPD1]], %arg1
// CHECK:         return %[[RES]]
