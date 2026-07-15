// RUN: enzymexlamlir-opt --split-input-file %s | FileCheck %s

func.func @roundtrip_axis_ops() -> (!axis.shape_axis<tensor<6x4xf32>, 1>, !axis.factor_group<6>) {
  %axis0 = axis.getaxis tensor<6x4xf32> 0
  %axis1 = axis.getaxis tensor<6x4xf32> 1

  %f0, %f1 = axis.factor %axis0 [2, 3] : !axis.shape_axis<tensor<6x4xf32>, 0>
  %g = axis.group %f0, %f1 : !axis.axis_factor<!axis.shape_axis<tensor<6x4xf32>, 0>, 2, 3>, !axis.axis_factor<!axis.shape_axis<tensor<6x4xf32>, 0>, 3, 1>

  return %axis1, %g : !axis.shape_axis<tensor<6x4xf32>, 1>, !axis.factor_group<6>
}

// CHECK-LABEL: func.func @roundtrip_axis_ops()
// CHECK: %[[AX0:.*]] = axis.getaxis tensor<6x4xf32> 0
// CHECK: %[[AX1:.*]] = axis.getaxis tensor<6x4xf32> 1
// CHECK: %[[FPAIR:.*]]:2 = axis.factor %[[AX0]] [2, 3] : !axis.shape_axis<tensor<6x4xf32>, 0>
// CHECK: %[[G:.*]] = axis.group %[[FPAIR]]#0, %[[FPAIR]]#1 : !axis.axis_factor<!axis.shape_axis<tensor<6x4xf32>, 0>, 2, 3>, !axis.axis_factor<!axis.shape_axis<tensor<6x4xf32>, 0>, 3, 1>
// CHECK: return %[[AX1]], %[[G]] : !axis.shape_axis<tensor<6x4xf32>, 1>, !axis.factor_group<6>
