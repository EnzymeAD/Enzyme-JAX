// RUN: enzymexlamlir-opt %s --canonicalize-axis-maps --canonicalize | FileCheck %s

// Each axis.map below is used as a distributed.Collective's own mapping, both
// to give --canonicalize a real (non-Pure) consumer to keep it from folding
// the whole chain away, and because that's exactly how these values get used
// in practice.

// CHECK-LABEL: module @map_6_to_2_3
// CHECK: %[[L0:.*]] = axis.product ({{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<6xf32>, 0>, 2, 3>)
// CHECK: %[[L1:.*]] = axis.product ({{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<6xf32>, 0>, 3, 1>)
// CHECK: %[[R0:.*]] = axis.product ({{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<6xf32>, 0>, 2, 3>)
// CHECK: %[[R1:.*]] = axis.product ({{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<6xf32>, 0>, 3, 1>)
// CHECK: %[[MAP:.*]] = axis.map %[[L0]], %[[L1]] to %[[R0]], %[[R1]] : [!axis.factor_group<2>, !axis.factor_group<3>] [!axis.factor_group<2>, !axis.factor_group<3>]
// CHECK: maps %[[MAP]] : !axis.map
module @map_6_to_2_3 {
  func.func @main() {
    return
  }
  %axis = axis.getaxis tensor<6xf32> 0

  %l0 = axis.factor %axis : !axis.shape_axis<tensor<6xf32>, 0> <6, 1>
  %lhs = axis.product (%l0 : !axis.axis_factor<!axis.shape_axis<tensor<6xf32>, 0>, 6, 1>)

  %r0 = axis.factor %axis : !axis.shape_axis<tensor<6xf32>, 0> <2, 3>
  %r1 = axis.factor %axis : !axis.shape_axis<tensor<6xf32>, 0> <3, 1>
  %rhs = axis.product (%r0 : !axis.axis_factor<!axis.shape_axis<tensor<6xf32>, 0>, 2, 3>, %r1 : !axis.axis_factor<!axis.shape_axis<tensor<6xf32>, 0>, 3, 1>)

  %m = axis.map %lhs to %rhs : [!axis.factor_group<6>] [!axis.factor_group<6>]

  %mesh_empty = axis.product ()
  %input = tensor.empty() : tensor<6xf32>
  %h = distributed.Collective %input : tensor<6xf32> on %mesh_empty : !axis.factor_group<1> to tensor<6xf32> on %mesh_empty : !axis.factor_group<1> reduces () maps %m : !axis.map
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<6xf32>> -> tensor<6xf32>
}

// -----

// CHECK-LABEL: module @map_8_4_to_4_8
// CHECK: %[[L0:.*]] = axis.product ({{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<32xf32>, 0>, 4, 8>)
// CHECK: %[[L1:.*]] = axis.product ({{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<32xf32>, 0>, 2, 4>)
// CHECK: %[[L2:.*]] = axis.product ({{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<32xf32>, 0>, 4, 1>)
// CHECK: %[[R0:.*]] = axis.product ({{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<32xf32>, 0>, 4, 8>)
// CHECK: %[[R1:.*]] = axis.product ({{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<32xf32>, 0>, 2, 4>)
// CHECK: %[[R2:.*]] = axis.product ({{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<32xf32>, 0>, 4, 1>)
// CHECK: %[[MAP:.*]] = axis.map %[[L0]], %[[L1]], %[[L2]] to %[[R0]], %[[R1]], %[[R2]] : [!axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<4>] [!axis.factor_group<4>, !axis.factor_group<2>, !axis.factor_group<4>]
// CHECK: maps %[[MAP]] : !axis.map
module @map_8_4_to_4_8 {
  func.func @main() {
    return
  }
  %axis = axis.getaxis tensor<32xf32> 0

  %l0 = axis.factor %axis : !axis.shape_axis<tensor<32xf32>, 0> <8, 4>
  %l1 = axis.factor %axis : !axis.shape_axis<tensor<32xf32>, 0> <4, 1>
  %lhs = axis.product (%l0 : !axis.axis_factor<!axis.shape_axis<tensor<32xf32>, 0>, 8, 4>, %l1 : !axis.axis_factor<!axis.shape_axis<tensor<32xf32>, 0>, 4, 1>)

  %r0 = axis.factor %axis : !axis.shape_axis<tensor<32xf32>, 0> <4, 8>
  %r1 = axis.factor %axis : !axis.shape_axis<tensor<32xf32>, 0> <8, 1>
  %rhs = axis.product (%r0 : !axis.axis_factor<!axis.shape_axis<tensor<32xf32>, 0>, 4, 8>, %r1 : !axis.axis_factor<!axis.shape_axis<tensor<32xf32>, 0>, 8, 1>)

  %m = axis.map %lhs to %rhs : [!axis.factor_group<32>] [!axis.factor_group<32>]

  %mesh_empty = axis.product ()
  %input = tensor.empty() : tensor<32xf32>
  %h = distributed.Collective %input : tensor<32xf32> on %mesh_empty : !axis.factor_group<1> to tensor<32xf32> on %mesh_empty : !axis.factor_group<1> reduces () maps %m : !axis.map
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<32xf32>> -> tensor<32xf32>
}

// -----

// CHECK-LABEL: module @map_6_to_6
// CHECK: %[[L0:.*]] = axis.product ({{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<6xf32>, 0>, 6, 1>)
// CHECK: %[[R0:.*]] = axis.product ({{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<6xf32>, 0>, 6, 1>)
// CHECK: %[[MAP:.*]] = axis.map %[[L0]] to %[[R0]] : [!axis.factor_group<6>] [!axis.factor_group<6>]
// CHECK: maps %[[MAP]] : !axis.map
module @map_6_to_6 {
  func.func @main() {
    return
  }
  %axis = axis.getaxis tensor<6xf32> 0

  %l0 = axis.factor %axis : !axis.shape_axis<tensor<6xf32>, 0> <6, 1>
  %lhs = axis.product (%l0 : !axis.axis_factor<!axis.shape_axis<tensor<6xf32>, 0>, 6, 1>)

  %r0 = axis.factor %axis : !axis.shape_axis<tensor<6xf32>, 0> <6, 1>
  %rhs = axis.product (%r0 : !axis.axis_factor<!axis.shape_axis<tensor<6xf32>, 0>, 6, 1>)

  %m = axis.map %lhs to %rhs : [!axis.factor_group<6>] [!axis.factor_group<6>]

  %mesh_empty = axis.product ()
  %input = tensor.empty() : tensor<6xf32>
  %h = distributed.Collective %input : tensor<6xf32> on %mesh_empty : !axis.factor_group<1> to tensor<6xf32> on %mesh_empty : !axis.factor_group<1> reduces () maps %m : !axis.map
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<6xf32>> -> tensor<6xf32>
}

// -----

// CHECK-LABEL: module @map_2_9_to_3_6
// CHECK: %[[L0:.*]] = axis.product ({{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<18xf32>, 0>, 2, 9>, {{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<18xf32>, 0>, 3, 3>)
// CHECK: %[[L1:.*]] = axis.product ({{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<18xf32>, 0>, 3, 1>)
// CHECK: %[[R0:.*]] = axis.product ({{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<18xf32>, 0>, 3, 6>, {{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<18xf32>, 0>, 2, 3>)
// CHECK: %[[R1:.*]] = axis.product ({{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<18xf32>, 0>, 3, 1>)
// CHECK: %[[MAP:.*]] = axis.map %[[L0]], %[[L1]] to %[[R0]], %[[R1]] : [!axis.factor_group<6>, !axis.factor_group<3>] [!axis.factor_group<6>, !axis.factor_group<3>]
// CHECK: maps %[[MAP]] : !axis.map
module @map_2_9_to_3_6 {
  func.func @main() {
    return
  }
  %axis = axis.getaxis tensor<18xf32> 0

  %l0 = axis.factor %axis : !axis.shape_axis<tensor<18xf32>, 0> <2, 9>
  %l1 = axis.factor %axis : !axis.shape_axis<tensor<18xf32>, 0> <9, 1>
  %lhs = axis.product (%l0 : !axis.axis_factor<!axis.shape_axis<tensor<18xf32>, 0>, 2, 9>, %l1 : !axis.axis_factor<!axis.shape_axis<tensor<18xf32>, 0>, 9, 1>)

  %r0 = axis.factor %axis : !axis.shape_axis<tensor<18xf32>, 0> <3, 6>
  %r1 = axis.factor %axis : !axis.shape_axis<tensor<18xf32>, 0> <6, 1>
  %rhs = axis.product (%r0 : !axis.axis_factor<!axis.shape_axis<tensor<18xf32>, 0>, 3, 6>, %r1 : !axis.axis_factor<!axis.shape_axis<tensor<18xf32>, 0>, 6, 1>)

  %m = axis.map %lhs to %rhs : [!axis.factor_group<18>] [!axis.factor_group<18>]

  %mesh_empty = axis.product ()
  %input = tensor.empty() : tensor<18xf32>
  %h = distributed.Collective %input : tensor<18xf32> on %mesh_empty : !axis.factor_group<1> to tensor<18xf32> on %mesh_empty : !axis.factor_group<1> reduces () maps %m : !axis.map
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<18xf32>> -> tensor<18xf32>
}

// -----

// CHECK-LABEL: module @map_2_15_3_to_3_6_5
// CHECK: %[[L0:.*]] = axis.product ({{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<90xf32>, 0>, 2, 45>, {{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<90xf32>, 0>, 3, 15>)
// CHECK: %[[L1:.*]] = axis.product ({{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<90xf32>, 0>, 5, 3>, {{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<90xf32>, 0>, 3, 1>)
// CHECK: %[[R0:.*]] = axis.product ({{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<90xf32>, 0>, 3, 30>, {{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<90xf32>, 0>, 2, 15>)
// CHECK: %[[R1:.*]] = axis.product ({{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<90xf32>, 0>, 3, 5>, {{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<90xf32>, 0>, 5, 1>)
// CHECK: %[[MAP:.*]] = axis.map %[[L0]], %[[L1]] to %[[R0]], %[[R1]] : [!axis.factor_group<6>, !axis.factor_group<15>] [!axis.factor_group<6>, !axis.factor_group<15>]
// CHECK: maps %[[MAP]] : !axis.map
module @map_2_15_3_to_3_6_5 {
  func.func @main() {
    return
  }
  %axis = axis.getaxis tensor<90xf32> 0

  %l0 = axis.factor %axis : !axis.shape_axis<tensor<90xf32>, 0> <2, 45>
  %l1 = axis.factor %axis : !axis.shape_axis<tensor<90xf32>, 0> <15, 3>
  %l2 = axis.factor %axis : !axis.shape_axis<tensor<90xf32>, 0> <3, 1>
  %lhs = axis.product (%l0 : !axis.axis_factor<!axis.shape_axis<tensor<90xf32>, 0>, 2, 45>, %l1 : !axis.axis_factor<!axis.shape_axis<tensor<90xf32>, 0>, 15, 3>, %l2 : !axis.axis_factor<!axis.shape_axis<tensor<90xf32>, 0>, 3, 1>)

  %r0 = axis.factor %axis : !axis.shape_axis<tensor<90xf32>, 0> <3, 30>
  %r1 = axis.factor %axis : !axis.shape_axis<tensor<90xf32>, 0> <6, 5>
  %r2 = axis.factor %axis : !axis.shape_axis<tensor<90xf32>, 0> <5, 1>
  %rhs = axis.product (%r0 : !axis.axis_factor<!axis.shape_axis<tensor<90xf32>, 0>, 3, 30>, %r1 : !axis.axis_factor<!axis.shape_axis<tensor<90xf32>, 0>, 6, 5>, %r2 : !axis.axis_factor<!axis.shape_axis<tensor<90xf32>, 0>, 5, 1>)

  %m = axis.map %lhs to %rhs : [!axis.factor_group<90>] [!axis.factor_group<90>]

  %mesh_empty = axis.product ()
  %input = tensor.empty() : tensor<90xf32>
  %h = distributed.Collective %input : tensor<90xf32> on %mesh_empty : !axis.factor_group<1> to tensor<90xf32> on %mesh_empty : !axis.factor_group<1> reduces () maps %m : !axis.map
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<90xf32>> -> tensor<90xf32>
}

// -----

// CHECK-LABEL: module @map_2_3_4_to_3_2_4
// CHECK: %[[L0:.*]] = axis.product ({{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<24xf32>, 0>, 2, 12>, {{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<24xf32>, 0>, 3, 4>)
// CHECK: %[[L1:.*]] = axis.product ({{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<24xf32>, 0>, 4, 1>)
// CHECK: %[[R0:.*]] = axis.product ({{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<24xf32>, 0>, 3, 8>, {{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<24xf32>, 0>, 2, 4>)
// CHECK: %[[R1:.*]] = axis.product ({{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<24xf32>, 0>, 4, 1>)
// CHECK: %[[MAP:.*]] = axis.map %[[L0]], %[[L1]] to %[[R0]], %[[R1]] : [!axis.factor_group<6>, !axis.factor_group<4>] [!axis.factor_group<6>, !axis.factor_group<4>]
// CHECK: maps %[[MAP]] : !axis.map
module @map_2_3_4_to_3_2_4 {
  func.func @main() {
    return
  }
  %axis = axis.getaxis tensor<24xf32> 0

  %l0 = axis.factor %axis : !axis.shape_axis<tensor<24xf32>, 0> <2, 12>
  %l1 = axis.factor %axis : !axis.shape_axis<tensor<24xf32>, 0> <3, 4>
  %l2 = axis.factor %axis : !axis.shape_axis<tensor<24xf32>, 0> <4, 1>
  %lhs = axis.product (%l0 : !axis.axis_factor<!axis.shape_axis<tensor<24xf32>, 0>, 2, 12>, %l1 : !axis.axis_factor<!axis.shape_axis<tensor<24xf32>, 0>, 3, 4>, %l2 : !axis.axis_factor<!axis.shape_axis<tensor<24xf32>, 0>, 4, 1>)

  %r0 = axis.factor %axis : !axis.shape_axis<tensor<24xf32>, 0> <3, 8>
  %r1 = axis.factor %axis : !axis.shape_axis<tensor<24xf32>, 0> <2, 4>
  %r2 = axis.factor %axis : !axis.shape_axis<tensor<24xf32>, 0> <4, 1>
  %rhs = axis.product (%r0 : !axis.axis_factor<!axis.shape_axis<tensor<24xf32>, 0>, 3, 8>, %r1 : !axis.axis_factor<!axis.shape_axis<tensor<24xf32>, 0>, 2, 4>, %r2 : !axis.axis_factor<!axis.shape_axis<tensor<24xf32>, 0>, 4, 1>)

  %m = axis.map %lhs to %rhs : [!axis.factor_group<24>] [!axis.factor_group<24>]

  %mesh_empty = axis.product ()
  %input = tensor.empty() : tensor<24xf32>
  %h = distributed.Collective %input : tensor<24xf32> on %mesh_empty : !axis.factor_group<1> to tensor<24xf32> on %mesh_empty : !axis.factor_group<1> reduces () maps %m : !axis.map
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<24xf32>> -> tensor<24xf32>
}

// -----

// CHECK-LABEL: module @map_2_3_5_to_5_3_2
// CHECK: %[[L0:.*]] = axis.product ({{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<30xf32>, 0>, 2, 15>, {{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<30xf32>, 0>, 3, 5>, {{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<30xf32>, 0>, 5, 1>)
// CHECK: %[[R0:.*]] = axis.product ({{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<30xf32>, 0>, 5, 6>, {{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<30xf32>, 0>, 3, 2>, {{.*}} : !axis.axis_factor<!axis.shape_axis<tensor<30xf32>, 0>, 2, 1>)
// CHECK: %[[MAP:.*]] = axis.map %[[L0]] to %[[R0]] : [!axis.factor_group<30>] [!axis.factor_group<30>]
// CHECK: maps %[[MAP]] : !axis.map
module @map_2_3_5_to_5_3_2 {
  func.func @main() {
    return
  }
  %axis = axis.getaxis tensor<30xf32> 0

  %l0 = axis.factor %axis : !axis.shape_axis<tensor<30xf32>, 0> <2, 15>
  %l1 = axis.factor %axis : !axis.shape_axis<tensor<30xf32>, 0> <3, 5>
  %l2 = axis.factor %axis : !axis.shape_axis<tensor<30xf32>, 0> <5, 1>
  %lhs = axis.product (%l0 : !axis.axis_factor<!axis.shape_axis<tensor<30xf32>, 0>, 2, 15>, %l1 : !axis.axis_factor<!axis.shape_axis<tensor<30xf32>, 0>, 3, 5>, %l2 : !axis.axis_factor<!axis.shape_axis<tensor<30xf32>, 0>, 5, 1>)

  %r0 = axis.factor %axis : !axis.shape_axis<tensor<30xf32>, 0> <5, 6>
  %r1 = axis.factor %axis : !axis.shape_axis<tensor<30xf32>, 0> <3, 2>
  %r2 = axis.factor %axis : !axis.shape_axis<tensor<30xf32>, 0> <2, 1>
  %rhs = axis.product (%r0 : !axis.axis_factor<!axis.shape_axis<tensor<30xf32>, 0>, 5, 6>, %r1 : !axis.axis_factor<!axis.shape_axis<tensor<30xf32>, 0>, 3, 2>, %r2 : !axis.axis_factor<!axis.shape_axis<tensor<30xf32>, 0>, 2, 1>)

  %m = axis.map %lhs to %rhs : [!axis.factor_group<30>] [!axis.factor_group<30>]

  %mesh_empty = axis.product ()
  %input = tensor.empty() : tensor<30xf32>
  %h = distributed.Collective %input : tensor<30xf32> on %mesh_empty : !axis.factor_group<1> to tensor<30xf32> on %mesh_empty : !axis.factor_group<1> reduces () maps %m : !axis.map
  %v = distributed.Await %h : !distributed.asynch_handle<tensor<30xf32>> -> tensor<30xf32>
}
