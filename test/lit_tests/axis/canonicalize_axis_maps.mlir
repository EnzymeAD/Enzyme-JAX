// RUN: enzymexlamlir-opt %s --canonicalize-axis-maps --canonicalize | FileCheck %s

// CHECK-LABEL: func.func @map_6_to_2_3() -> !axis.map {
// CHECK: %[[L0:.*]] = axis.product {{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 2, 3>
// CHECK: %[[L1:.*]] = axis.product {{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 3, 1>
// CHECK: %[[R0:.*]] = axis.product {{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 2, 3>
// CHECK: %[[R1:.*]] = axis.product {{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 3, 1>
// CHECK: %[[MAP:.*]] = axis.map %[[L0]], %[[L1]] to %[[R0]], %[[R1]] : [!axis.factor_group<2>, !axis.factor_group<3>] [!axis.factor_group<2>, !axis.factor_group<3>]
// CHECK: return %[[MAP]] : !axis.map
func.func @map_6_to_2_3() -> !axis.map {
  %axis = axis.getaxis tensor<21600xf32> 0

  %l0 = axis.factor %axis : (!axis.shape_axis<tensor<21600xf32>, 0>) -> !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 6, 1>
  %lhs = axis.product %l0 : !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 6, 1>

  %r0 = axis.factor %axis : (!axis.shape_axis<tensor<21600xf32>, 0>) -> !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 2, 3>
  %r1 = axis.factor %axis : (!axis.shape_axis<tensor<21600xf32>, 0>) -> !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 3, 1>
  %rhs = axis.product %r0, %r1 : !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 2, 3>, !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 3, 1>

  %m = axis.map %lhs to %rhs : [!axis.factor_group<6>] [!axis.factor_group<6>]
  return %m : !axis.map
}

// CHECK-LABEL: func.func @map_8_4_to_4_8() -> !axis.map {
// CHECK: %[[L0:.*]] = axis.product {{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 4, 8>
// CHECK: %[[L1:.*]] = axis.product {{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 2, 4>
// CHECK: %[[L2:.*]] = axis.product {{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 4, 1>
// CHECK: %[[R0:.*]] = axis.product {{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 4, 8>
// CHECK: %[[R1:.*]] = axis.product {{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 2, 4>
// CHECK: %[[R2:.*]] = axis.product {{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 4, 1>
// CHECK: %[[MAP:.*]] = axis.map %[[L0]], %[[L1]], %[[L2]] to %[[R0]], %[[R1]], %[[R2]] : [!axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<4>] [!axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<4>]
// CHECK: return %[[MAP]] : !axis.map
func.func @map_8_4_to_4_8() -> !axis.map {
  %axis = axis.getaxis tensor<21600xf32> 0

  %l0 = axis.factor %axis : (!axis.shape_axis<tensor<21600xf32>, 0>) -> !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 8, 4>
  %l1 = axis.factor %axis : (!axis.shape_axis<tensor<21600xf32>, 0>) -> !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 4, 1>
  %lhs = axis.product %l0, %l1 : !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 8, 4>, !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 4, 1>

  %r0 = axis.factor %axis : (!axis.shape_axis<tensor<21600xf32>, 0>) -> !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 4, 8>
  %r1 = axis.factor %axis : (!axis.shape_axis<tensor<21600xf32>, 0>) -> !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 8, 1>
  %rhs = axis.product %r0, %r1 : !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 4, 8>, !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 8, 1>

  %m = axis.map %lhs to %rhs : [!axis.factor_group<32>] [!axis.factor_group<32>]
  return %m : !axis.map
}

// CHECK-LABEL: func.func @map_6_to_6() -> !axis.map {
// CHECK: %[[L0:.*]] = axis.product {{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 6, 1>
// CHECK: %[[R0:.*]] = axis.product {{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 6, 1>
// CHECK: %[[MAP:.*]] = axis.map %[[L0]] to %[[R0]] : [!axis.factor_group<6>] [!axis.factor_group<6>]
// CHECK: return %[[MAP]] : !axis.map
func.func @map_6_to_6() -> !axis.map {
  %axis = axis.getaxis tensor<21600xf32> 0

  %l0 = axis.factor %axis : (!axis.shape_axis<tensor<21600xf32>, 0>) -> !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 6, 1>
  %lhs = axis.product %l0 : !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 6, 1>

  %r0 = axis.factor %axis : (!axis.shape_axis<tensor<21600xf32>, 0>) -> !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 6, 1>
  %rhs = axis.product %r0 : !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 6, 1>

  %m = axis.map %lhs to %rhs : [!axis.factor_group<6>] [!axis.factor_group<6>]
  return %m : !axis.map
}

// CHECK-LABEL: func.func @map_2_9_to_3_6() -> !axis.map {
// CHECK: %[[L0:.*]] = axis.product {{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 2, 9>, !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 3, 3>
// CHECK: %[[L1:.*]] = axis.product {{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 3, 1>
// CHECK: %[[R0:.*]] = axis.product {{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 3, 6>, !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 2, 3>
// CHECK: %[[R1:.*]] = axis.product {{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 3, 1>
// CHECK: %[[MAP:.*]] = axis.map %[[L0]], %[[L1]] to %[[R0]], %[[R1]] : [!axis.factor_group<6>, !axis.factor_group<3>] [!axis.factor_group<6>, !axis.factor_group<3>]
// CHECK: return %[[MAP]] : !axis.map
func.func @map_2_9_to_3_6() -> !axis.map {
  %axis = axis.getaxis tensor<21600xf32> 0

  %l0 = axis.factor %axis : (!axis.shape_axis<tensor<21600xf32>, 0>) -> !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 2, 9>
  %l1 = axis.factor %axis : (!axis.shape_axis<tensor<21600xf32>, 0>) -> !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 9, 1>
  %lhs = axis.product %l0, %l1 : !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 2, 9>, !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 9, 1>

  %r0 = axis.factor %axis : (!axis.shape_axis<tensor<21600xf32>, 0>) -> !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 3, 6>
  %r1 = axis.factor %axis : (!axis.shape_axis<tensor<21600xf32>, 0>) -> !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 6, 1>
  %rhs = axis.product %r0, %r1 : !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 3, 6>, !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 6, 1>

  %m = axis.map %lhs to %rhs : [!axis.factor_group<18>] [!axis.factor_group<18>]
  return %m : !axis.map
}

// CHECK-LABEL: func.func @map_2_15_3_to_3_6_5() -> !axis.map {
// CHECK: %[[L0:.*]] = axis.product {{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 2, 45>, !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 3, 15>
// CHECK: %[[L1:.*]] = axis.product {{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 5, 3>, !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 3, 1>
// CHECK: %[[R0:.*]] = axis.product {{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 3, 30>, !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 2, 15>
// CHECK: %[[R1:.*]] = axis.product {{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 3, 5>, !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 5, 1>
// CHECK: %[[MAP:.*]] = axis.map %[[L0]], %[[L1]] to %[[R0]], %[[R1]] : [!axis.factor_group<6>, !axis.factor_group<15>] [!axis.factor_group<6>, !axis.factor_group<15>]
// CHECK: return %[[MAP]] : !axis.map
func.func @map_2_15_3_to_3_6_5() -> !axis.map {
  %axis = axis.getaxis tensor<21600xf32> 0

  %l0 = axis.factor %axis : (!axis.shape_axis<tensor<21600xf32>, 0>) -> !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 2, 45>
  %l1 = axis.factor %axis : (!axis.shape_axis<tensor<21600xf32>, 0>) -> !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 15, 3>
  %l2 = axis.factor %axis : (!axis.shape_axis<tensor<21600xf32>, 0>) -> !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 3, 1>
  %lhs = axis.product %l0, %l1, %l2 : !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 2, 45>, !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 15, 3>, !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 3, 1>

  %r0 = axis.factor %axis : (!axis.shape_axis<tensor<21600xf32>, 0>) -> !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 3, 30>
  %r1 = axis.factor %axis : (!axis.shape_axis<tensor<21600xf32>, 0>) -> !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 6, 5>
  %r2 = axis.factor %axis : (!axis.shape_axis<tensor<21600xf32>, 0>) -> !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 5, 1>
  %rhs = axis.product %r0, %r1, %r2 : !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 3, 30>, !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 6, 5>, !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 5, 1>

  %m = axis.map %lhs to %rhs : [!axis.factor_group<90>] [!axis.factor_group<90>]
  return %m : !axis.map
}

// CHECK-LABEL: func.func @map_2_3_4_to_3_2_4() -> !axis.map {
// CHECK: %[[L0:.*]] = axis.product {{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 2, 12>, !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 3, 4>
// CHECK: %[[L1:.*]] = axis.product {{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 4, 1>
// CHECK: %[[R0:.*]] = axis.product {{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 3, 8>, !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 2, 4>
// CHECK: %[[R1:.*]] = axis.product {{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 4, 1>
// CHECK: %[[MAP:.*]] = axis.map %[[L0]], %[[L1]] to %[[R0]], %[[R1]] : [!axis.factor_group<6>, !axis.factor_group<4>] [!axis.factor_group<6>, !axis.factor_group<4>]
// CHECK: return %[[MAP]] : !axis.map
func.func @map_2_3_4_to_3_2_4() -> !axis.map {
  %axis = axis.getaxis tensor<21600xf32> 0

  %l0 = axis.factor %axis : (!axis.shape_axis<tensor<21600xf32>, 0>) -> !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 2, 12>
  %l1 = axis.factor %axis : (!axis.shape_axis<tensor<21600xf32>, 0>) -> !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 3, 4>
  %l2 = axis.factor %axis : (!axis.shape_axis<tensor<21600xf32>, 0>) -> !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 4, 1>
  %lhs = axis.product %l0, %l1, %l2 : !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 2, 12>, !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 3, 4>, !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 4, 1>

  %r0 = axis.factor %axis : (!axis.shape_axis<tensor<21600xf32>, 0>) -> !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 3, 8>
  %r1 = axis.factor %axis : (!axis.shape_axis<tensor<21600xf32>, 0>) -> !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 2, 4>
  %r2 = axis.factor %axis : (!axis.shape_axis<tensor<21600xf32>, 0>) -> !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 4, 1>
  %rhs = axis.product %r0, %r1, %r2 : !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 3, 8>, !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 2, 4>, !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 4, 1>

  %m = axis.map %lhs to %rhs : [!axis.factor_group<24>] [!axis.factor_group<24>]
  return %m : !axis.map
}

// CHECK-LABEL: func.func @map_2_3_5_to_5_3_2() -> !axis.map {
// CHECK: %[[L0:.*]] = axis.product {{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 2, 15>, !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 3, 5>, !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 5, 1>
// CHECK: %[[R0:.*]] = axis.product {{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 5, 6>, !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 3, 2>, !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 2, 1>
// CHECK: %[[MAP:.*]] = axis.map %[[L0]] to %[[R0]] : [!axis.factor_group<30>] [!axis.factor_group<30>]
// CHECK: return %[[MAP]] : !axis.map
func.func @map_2_3_5_to_5_3_2() -> !axis.map {
  %axis = axis.getaxis tensor<21600xf32> 0

  %l0 = axis.factor %axis : (!axis.shape_axis<tensor<21600xf32>, 0>) -> !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 2, 15>
  %l1 = axis.factor %axis : (!axis.shape_axis<tensor<21600xf32>, 0>) -> !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 3, 5>
  %l2 = axis.factor %axis : (!axis.shape_axis<tensor<21600xf32>, 0>) -> !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 5, 1>
  %lhs = axis.product %l0, %l1, %l2 : !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 2, 15>, !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 3, 5>, !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 5, 1>

  %r0 = axis.factor %axis : (!axis.shape_axis<tensor<21600xf32>, 0>) -> !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 5, 6>
  %r1 = axis.factor %axis : (!axis.shape_axis<tensor<21600xf32>, 0>) -> !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 3, 2>
  %r2 = axis.factor %axis : (!axis.shape_axis<tensor<21600xf32>, 0>) -> !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 2, 1>
  %rhs = axis.product %r0, %r1, %r2 : !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 5, 6>, !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 3, 2>, !axis.axis_factor<!axis.shape_axis<tensor<21600xf32>, 0>, 2, 1>

  %m = axis.map %lhs to %rhs : [!axis.factor_group<30>] [!axis.factor_group<30>]
  return %m : !axis.map
}
