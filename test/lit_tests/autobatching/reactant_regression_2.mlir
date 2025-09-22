// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=if_remove_unused;dot_general_slice_to_batch;gather_slice_to_batch;iota_slice_to_batch;reduce_slice_to_batch;sort_slice_to_batch;transpose_slice_to_batch;broadcastindim_slice_to_batch;reducewindow_slice_to_batch;elementwise_slice_to_batch;slice_reshape_pad<1>;elementwise_reshape_like;reorder_elementwise_and_shape_op<16>;transpose_all_users_slice},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s

module @reactant_f_gener... attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func @main(%arg0: tensor<6x2xf32> {tf.aliasing_output = 1 : i32}, %arg1: tensor<2x4xf32> {tf.aliasing_output = 2 : i32}) -> (tensor<4xf32>, tensor<6x2xf32>, tensor<2x4xf32>) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<6x2xf32>) -> tensor<2x6xf32>
    %1 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<2x4xf32>) -> tensor<4x2xf32>
    %2 = stablehlo.slice %0 [0:2, 0:1] : (tensor<2x6xf32>) -> tensor<2x1xf32>
    %3 = stablehlo.transpose %2, dims = [1, 0] : (tensor<2x1xf32>) -> tensor<1x2xf32>
    %4 = stablehlo.reshape %3 : (tensor<1x2xf32>) -> tensor<2xf32>
    %5 = stablehlo.transpose %4, dims = [0] : (tensor<2xf32>) -> tensor<2xf32>
    %6 = stablehlo.transpose %5, dims = [0] : (tensor<2xf32>) -> tensor<2xf32>
    %7 = stablehlo.reshape %6 : (tensor<2xf32>) -> tensor<1x2xf32>
    %8 = stablehlo.transpose %7, dims = [1, 0] : (tensor<1x2xf32>) -> tensor<2x1xf32>
    %9 = stablehlo.convert %8 : tensor<2x1xf32>
    %10 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<4x2xf32>) -> tensor<4x2xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<4x2xf32>) -> tensor<4x2xf32>
    %12 = stablehlo.broadcast_in_dim %9, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %14 = stablehlo.dot_general %11, %13, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x2xf32>, tensor<2x1xf32>) -> tensor<4x1xf32>
    %15 = stablehlo.transpose %14, dims = [1, 0] : (tensor<4x1xf32>) -> tensor<1x4xf32>
    %16 = stablehlo.reshape %15 : (tensor<1x4xf32>) -> tensor<4xf32>
    %17 = stablehlo.transpose %16, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %18 = stablehlo.slice %0 [0:2, 1:2] : (tensor<2x6xf32>) -> tensor<2x1xf32>
    %19 = stablehlo.transpose %18, dims = [1, 0] : (tensor<2x1xf32>) -> tensor<1x2xf32>
    %20 = stablehlo.reshape %19 : (tensor<1x2xf32>) -> tensor<2xf32>
    %21 = stablehlo.transpose %20, dims = [0] : (tensor<2xf32>) -> tensor<2xf32>
    %22 = stablehlo.transpose %21, dims = [0] : (tensor<2xf32>) -> tensor<2xf32>
    %23 = stablehlo.reshape %22 : (tensor<2xf32>) -> tensor<1x2xf32>
    %24 = stablehlo.transpose %23, dims = [1, 0] : (tensor<1x2xf32>) -> tensor<2x1xf32>
    %25 = stablehlo.convert %24 : tensor<2x1xf32>
    %26 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<4x2xf32>) -> tensor<4x2xf32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [0, 1] : (tensor<4x2xf32>) -> tensor<4x2xf32>
    %28 = stablehlo.broadcast_in_dim %25, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %29 = stablehlo.broadcast_in_dim %28, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %30 = stablehlo.dot_general %27, %29, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x2xf32>, tensor<2x1xf32>) -> tensor<4x1xf32>
    %31 = stablehlo.transpose %30, dims = [1, 0] : (tensor<4x1xf32>) -> tensor<1x4xf32>
    %32 = stablehlo.reshape %31 : (tensor<1x4xf32>) -> tensor<4xf32>
    %33 = stablehlo.transpose %32, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %34 = stablehlo.slice %0 [0:2, 2:3] : (tensor<2x6xf32>) -> tensor<2x1xf32>
    %35 = stablehlo.transpose %34, dims = [1, 0] : (tensor<2x1xf32>) -> tensor<1x2xf32>
    %36 = stablehlo.reshape %35 : (tensor<1x2xf32>) -> tensor<2xf32>
    %37 = stablehlo.transpose %36, dims = [0] : (tensor<2xf32>) -> tensor<2xf32>
    %38 = stablehlo.transpose %37, dims = [0] : (tensor<2xf32>) -> tensor<2xf32>
    %39 = stablehlo.reshape %38 : (tensor<2xf32>) -> tensor<1x2xf32>
    %40 = stablehlo.transpose %39, dims = [1, 0] : (tensor<1x2xf32>) -> tensor<2x1xf32>
    %41 = stablehlo.convert %40 : tensor<2x1xf32>
    %42 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<4x2xf32>) -> tensor<4x2xf32>
    %43 = stablehlo.broadcast_in_dim %42, dims = [0, 1] : (tensor<4x2xf32>) -> tensor<4x2xf32>
    %44 = stablehlo.broadcast_in_dim %41, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %45 = stablehlo.broadcast_in_dim %44, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %46 = stablehlo.dot_general %43, %45, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x2xf32>, tensor<2x1xf32>) -> tensor<4x1xf32>
    %47 = stablehlo.transpose %46, dims = [1, 0] : (tensor<4x1xf32>) -> tensor<1x4xf32>
    %48 = stablehlo.reshape %47 : (tensor<1x4xf32>) -> tensor<4xf32>
    %49 = stablehlo.transpose %48, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %50 = stablehlo.slice %0 [0:2, 3:4] : (tensor<2x6xf32>) -> tensor<2x1xf32>
    %51 = stablehlo.transpose %50, dims = [1, 0] : (tensor<2x1xf32>) -> tensor<1x2xf32>
    %52 = stablehlo.reshape %51 : (tensor<1x2xf32>) -> tensor<2xf32>
    %53 = stablehlo.transpose %52, dims = [0] : (tensor<2xf32>) -> tensor<2xf32>
    %54 = stablehlo.transpose %53, dims = [0] : (tensor<2xf32>) -> tensor<2xf32>
    %55 = stablehlo.reshape %54 : (tensor<2xf32>) -> tensor<1x2xf32>
    %56 = stablehlo.transpose %55, dims = [1, 0] : (tensor<1x2xf32>) -> tensor<2x1xf32>
    %57 = stablehlo.convert %56 : tensor<2x1xf32>
    %58 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<4x2xf32>) -> tensor<4x2xf32>
    %59 = stablehlo.broadcast_in_dim %58, dims = [0, 1] : (tensor<4x2xf32>) -> tensor<4x2xf32>
    %60 = stablehlo.broadcast_in_dim %57, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %61 = stablehlo.broadcast_in_dim %60, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %62 = stablehlo.dot_general %59, %61, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x2xf32>, tensor<2x1xf32>) -> tensor<4x1xf32>
    %63 = stablehlo.transpose %62, dims = [1, 0] : (tensor<4x1xf32>) -> tensor<1x4xf32>
    %64 = stablehlo.reshape %63 : (tensor<1x4xf32>) -> tensor<4xf32>
    %65 = stablehlo.transpose %64, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %66 = stablehlo.slice %0 [0:2, 4:5] : (tensor<2x6xf32>) -> tensor<2x1xf32>
    %67 = stablehlo.transpose %66, dims = [1, 0] : (tensor<2x1xf32>) -> tensor<1x2xf32>
    %68 = stablehlo.reshape %67 : (tensor<1x2xf32>) -> tensor<2xf32>
    %69 = stablehlo.transpose %68, dims = [0] : (tensor<2xf32>) -> tensor<2xf32>
    %70 = stablehlo.transpose %69, dims = [0] : (tensor<2xf32>) -> tensor<2xf32>
    %71 = stablehlo.reshape %70 : (tensor<2xf32>) -> tensor<1x2xf32>
    %72 = stablehlo.transpose %71, dims = [1, 0] : (tensor<1x2xf32>) -> tensor<2x1xf32>
    %73 = stablehlo.convert %72 : tensor<2x1xf32>
    %74 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<4x2xf32>) -> tensor<4x2xf32>
    %75 = stablehlo.broadcast_in_dim %74, dims = [0, 1] : (tensor<4x2xf32>) -> tensor<4x2xf32>
    %76 = stablehlo.broadcast_in_dim %73, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %77 = stablehlo.broadcast_in_dim %76, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %78 = stablehlo.dot_general %75, %77, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x2xf32>, tensor<2x1xf32>) -> tensor<4x1xf32>
    %79 = stablehlo.transpose %78, dims = [1, 0] : (tensor<4x1xf32>) -> tensor<1x4xf32>
    %80 = stablehlo.reshape %79 : (tensor<1x4xf32>) -> tensor<4xf32>
    %81 = stablehlo.transpose %80, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %82 = stablehlo.slice %0 [0:2, 5:6] : (tensor<2x6xf32>) -> tensor<2x1xf32>
    %83 = stablehlo.transpose %82, dims = [1, 0] : (tensor<2x1xf32>) -> tensor<1x2xf32>
    %84 = stablehlo.reshape %83 : (tensor<1x2xf32>) -> tensor<2xf32>
    %85 = stablehlo.transpose %84, dims = [0] : (tensor<2xf32>) -> tensor<2xf32>
    %86 = stablehlo.transpose %85, dims = [0] : (tensor<2xf32>) -> tensor<2xf32>
    %87 = stablehlo.reshape %86 : (tensor<2xf32>) -> tensor<1x2xf32>
    %88 = stablehlo.transpose %87, dims = [1, 0] : (tensor<1x2xf32>) -> tensor<2x1xf32>
    %89 = stablehlo.convert %88 : tensor<2x1xf32>
    %90 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<4x2xf32>) -> tensor<4x2xf32>
    %91 = stablehlo.broadcast_in_dim %90, dims = [0, 1] : (tensor<4x2xf32>) -> tensor<4x2xf32>
    %92 = stablehlo.broadcast_in_dim %89, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %93 = stablehlo.broadcast_in_dim %92, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %94 = stablehlo.dot_general %91, %93, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x2xf32>, tensor<2x1xf32>) -> tensor<4x1xf32>
    %95 = stablehlo.transpose %94, dims = [1, 0] : (tensor<4x1xf32>) -> tensor<1x4xf32>
    %96 = stablehlo.reshape %95 : (tensor<1x4xf32>) -> tensor<4xf32>
    %97 = stablehlo.transpose %96, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %98 = stablehlo.broadcast_in_dim %17, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %99 = stablehlo.broadcast_in_dim %98, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %100 = stablehlo.broadcast_in_dim %33, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %101 = stablehlo.broadcast_in_dim %100, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %102 = stablehlo.add %99, %101 : tensor<4xf32>
    %103 = stablehlo.broadcast_in_dim %102, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %104 = stablehlo.broadcast_in_dim %103, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %105 = stablehlo.broadcast_in_dim %49, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %106 = stablehlo.broadcast_in_dim %105, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %107 = stablehlo.add %104, %106 : tensor<4xf32>
    %108 = stablehlo.broadcast_in_dim %107, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %109 = stablehlo.broadcast_in_dim %108, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %110 = stablehlo.broadcast_in_dim %65, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %111 = stablehlo.broadcast_in_dim %110, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %112 = stablehlo.add %109, %111 : tensor<4xf32>
    %113 = stablehlo.broadcast_in_dim %112, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %114 = stablehlo.broadcast_in_dim %113, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %115 = stablehlo.broadcast_in_dim %81, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %116 = stablehlo.broadcast_in_dim %115, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %117 = stablehlo.add %114, %116 : tensor<4xf32>
    %118 = stablehlo.broadcast_in_dim %117, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %119 = stablehlo.broadcast_in_dim %118, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %120 = stablehlo.broadcast_in_dim %97, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %121 = stablehlo.broadcast_in_dim %120, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %122 = stablehlo.add %119, %121 : tensor<4xf32>
    %123 = stablehlo.transpose %122, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %124 = stablehlo.transpose %0, dims = [1, 0] : (tensor<2x6xf32>) -> tensor<6x2xf32>
    %125 = stablehlo.transpose %1, dims = [1, 0] : (tensor<4x2xf32>) -> tensor<2x4xf32>
    return %123, %124, %125 : tensor<4xf32>, tensor<6x2xf32>, tensor<2x4xf32>
  }
}

// CHECK:  module @reactant_f_gener... attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
// CHECK-NEXT:    func.func private @enzymexla_unbatched_SliceToBatch_1(%arg0: tensor<1x1x2xf32>) -> tensor<1x1x2xf32> {
// CHECK-NEXT:      %0 = stablehlo.convert %arg0 : tensor<1x1x2xf32>
// CHECK-NEXT:      return %0 : tensor<1x1x2xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func private @enzymexla_unbatched_SliceToBatch_0(%arg0: tensor<2x1xf32>) -> tensor<1x2xf32> {
// CHECK-NEXT:      %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x1xf32>) -> tensor<1x2xf32>
// CHECK-NEXT:      return %0 : tensor<1x2xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @main(%arg0: tensor<6x2xf32> {tf.aliasing_output = 1 : i32}, %arg1: tensor<2x4xf32> {tf.aliasing_output = 2 : i32}) -> (tensor<4xf32>, tensor<6x2xf32>, tensor<2x4xf32>) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
// CHECK-NEXT:      %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<6x2xf32>) -> tensor<2x6xf32>
// CHECK-NEXT:      %1 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<2x4xf32>) -> tensor<4x2xf32>
// CHECK-NEXT:      %2 = stablehlo.broadcast_in_dim %0, dims = [1, 0] : (tensor<2x6xf32>) -> tensor<6x2x1xf32>
// CHECK-NEXT:      %3 = call @batched_enzymexla_unbatched_SliceToBatch_0(%2) : (tensor<6x2x1xf32>) -> tensor<6x1x2xf32>
// CHECK-NEXT:      %4 = stablehlo.broadcast_in_dim %3, dims = [0, 2, 3] : (tensor<6x1x2xf32>) -> tensor<6x1x1x2xf32>
// CHECK-NEXT:      %5 = call @batched_enzymexla_unbatched_SliceToBatch_1(%4) : (tensor<6x1x1x2xf32>) -> tensor<6x1x1x2xf32>
// CHECK-NEXT:      %6 = stablehlo.slice %5 [0:1, 0:1, 0:1, 0:2] : (tensor<6x1x1x2xf32>) -> tensor<1x1x1x2xf32>
// CHECK-NEXT:      %7 = stablehlo.reshape %6 : (tensor<1x1x1x2xf32>) -> tensor<1x1x2xf32>
// CHECK-NEXT:      %8 = stablehlo.slice %5 [1:2, 0:1, 0:1, 0:2] : (tensor<6x1x1x2xf32>) -> tensor<1x1x1x2xf32>
// CHECK-NEXT:      %9 = stablehlo.reshape %8 : (tensor<1x1x1x2xf32>) -> tensor<1x1x2xf32>
// CHECK-NEXT:      %10 = stablehlo.slice %5 [2:3, 0:1, 0:1, 0:2] : (tensor<6x1x1x2xf32>) -> tensor<1x1x1x2xf32>
// CHECK-NEXT:      %11 = stablehlo.reshape %10 : (tensor<1x1x1x2xf32>) -> tensor<1x1x2xf32>
// CHECK-NEXT:      %12 = stablehlo.slice %5 [3:4, 0:1, 0:1, 0:2] : (tensor<6x1x1x2xf32>) -> tensor<1x1x1x2xf32>
// CHECK-NEXT:      %13 = stablehlo.reshape %12 : (tensor<1x1x1x2xf32>) -> tensor<1x1x2xf32>
// CHECK-NEXT:      %14 = stablehlo.slice %5 [4:5, 0:1, 0:1, 0:2] : (tensor<6x1x1x2xf32>) -> tensor<1x1x1x2xf32>
// CHECK-NEXT:      %15 = stablehlo.reshape %14 : (tensor<1x1x1x2xf32>) -> tensor<1x1x2xf32>
// CHECK-NEXT:      %16 = stablehlo.slice %5 [5:6, 0:1, 0:1, 0:2] : (tensor<6x1x1x2xf32>) -> tensor<1x1x1x2xf32>
// CHECK-NEXT:      %17 = stablehlo.reshape %16 : (tensor<1x1x1x2xf32>) -> tensor<1x1x2xf32>
// CHECK-NEXT:      %18 = stablehlo.reshape %17 : (tensor<1x1x2xf32>) -> tensor<1x2xf32>
// CHECK-NEXT:      %19 = stablehlo.reshape %18 : (tensor<1x2xf32>) -> tensor<2xf32>
// CHECK-NEXT:      %20 = stablehlo.transpose %19, dims = [0] : (tensor<2xf32>) -> tensor<2xf32>
// CHECK-NEXT:      %21 = stablehlo.transpose %20, dims = [0] : (tensor<2xf32>) -> tensor<2xf32>
// CHECK-NEXT:      %22 = stablehlo.reshape %21 : (tensor<2xf32>) -> tensor<1x2xf32>
// CHECK-NEXT:      %23 = stablehlo.transpose %22, dims = [1, 0] : (tensor<1x2xf32>) -> tensor<2x1xf32>
// CHECK-NEXT:      %24 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<4x2xf32>) -> tensor<4x2xf32>
// CHECK-NEXT:      %25 = stablehlo.broadcast_in_dim %24, dims = [0, 1] : (tensor<4x2xf32>) -> tensor<4x2xf32>
// CHECK-NEXT:      %26 = stablehlo.broadcast_in_dim %23, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1xf32>
// CHECK-NEXT:      %27 = stablehlo.broadcast_in_dim %26, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1xf32>
// CHECK-NEXT:      %28 = stablehlo.dot_general %25, %27, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x2xf32>, tensor<2x1xf32>) -> tensor<4x1xf32>
// CHECK-NEXT:      %29 = stablehlo.transpose %28, dims = [1, 0] : (tensor<4x1xf32>) -> tensor<1x4xf32>
// CHECK-NEXT:      %30 = stablehlo.reshape %29 : (tensor<1x4xf32>) -> tensor<4xf32>
// CHECK-NEXT:      %31 = stablehlo.transpose %30, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:      %32 = stablehlo.reshape %15 : (tensor<1x1x2xf32>) -> tensor<1x2xf32>
// CHECK-NEXT:      %33 = stablehlo.reshape %32 : (tensor<1x2xf32>) -> tensor<2xf32>
// CHECK-NEXT:      %34 = stablehlo.transpose %33, dims = [0] : (tensor<2xf32>) -> tensor<2xf32>
// CHECK-NEXT:      %35 = stablehlo.transpose %34, dims = [0] : (tensor<2xf32>) -> tensor<2xf32>
// CHECK-NEXT:      %36 = stablehlo.reshape %35 : (tensor<2xf32>) -> tensor<1x2xf32>
// CHECK-NEXT:      %37 = stablehlo.transpose %36, dims = [1, 0] : (tensor<1x2xf32>) -> tensor<2x1xf32>
// CHECK-NEXT:      %38 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<4x2xf32>) -> tensor<4x2xf32>
// CHECK-NEXT:      %39 = stablehlo.broadcast_in_dim %38, dims = [0, 1] : (tensor<4x2xf32>) -> tensor<4x2xf32>
// CHECK-NEXT:      %40 = stablehlo.broadcast_in_dim %37, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1xf32>
// CHECK-NEXT:      %41 = stablehlo.broadcast_in_dim %40, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1xf32>
// CHECK-NEXT:      %42 = stablehlo.dot_general %39, %41, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x2xf32>, tensor<2x1xf32>) -> tensor<4x1xf32>
// CHECK-NEXT:      %43 = stablehlo.transpose %42, dims = [1, 0] : (tensor<4x1xf32>) -> tensor<1x4xf32>
// CHECK-NEXT:      %44 = stablehlo.reshape %43 : (tensor<1x4xf32>) -> tensor<4xf32>
// CHECK-NEXT:      %45 = stablehlo.transpose %44, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:      %46 = stablehlo.reshape %13 : (tensor<1x1x2xf32>) -> tensor<1x2xf32>
// CHECK-NEXT:      %47 = stablehlo.reshape %46 : (tensor<1x2xf32>) -> tensor<2xf32>
// CHECK-NEXT:      %48 = stablehlo.transpose %47, dims = [0] : (tensor<2xf32>) -> tensor<2xf32>
// CHECK-NEXT:      %49 = stablehlo.transpose %48, dims = [0] : (tensor<2xf32>) -> tensor<2xf32>
// CHECK-NEXT:      %50 = stablehlo.reshape %49 : (tensor<2xf32>) -> tensor<1x2xf32>
// CHECK-NEXT:      %51 = stablehlo.transpose %50, dims = [1, 0] : (tensor<1x2xf32>) -> tensor<2x1xf32>
// CHECK-NEXT:      %52 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<4x2xf32>) -> tensor<4x2xf32>
// CHECK-NEXT:      %53 = stablehlo.broadcast_in_dim %52, dims = [0, 1] : (tensor<4x2xf32>) -> tensor<4x2xf32>
// CHECK-NEXT:      %54 = stablehlo.broadcast_in_dim %51, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1xf32>
// CHECK-NEXT:      %55 = stablehlo.broadcast_in_dim %54, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1xf32>
// CHECK-NEXT:      %56 = stablehlo.dot_general %53, %55, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x2xf32>, tensor<2x1xf32>) -> tensor<4x1xf32>
// CHECK-NEXT:      %57 = stablehlo.transpose %56, dims = [1, 0] : (tensor<4x1xf32>) -> tensor<1x4xf32>
// CHECK-NEXT:      %58 = stablehlo.reshape %57 : (tensor<1x4xf32>) -> tensor<4xf32>
// CHECK-NEXT:      %59 = stablehlo.transpose %58, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:      %60 = stablehlo.reshape %11 : (tensor<1x1x2xf32>) -> tensor<1x2xf32>
// CHECK-NEXT:      %61 = stablehlo.reshape %60 : (tensor<1x2xf32>) -> tensor<2xf32>
// CHECK-NEXT:      %62 = stablehlo.transpose %61, dims = [0] : (tensor<2xf32>) -> tensor<2xf32>
// CHECK-NEXT:      %63 = stablehlo.transpose %62, dims = [0] : (tensor<2xf32>) -> tensor<2xf32>
// CHECK-NEXT:      %64 = stablehlo.reshape %63 : (tensor<2xf32>) -> tensor<1x2xf32>
// CHECK-NEXT:      %65 = stablehlo.transpose %64, dims = [1, 0] : (tensor<1x2xf32>) -> tensor<2x1xf32>
// CHECK-NEXT:      %66 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<4x2xf32>) -> tensor<4x2xf32>
// CHECK-NEXT:      %67 = stablehlo.broadcast_in_dim %66, dims = [0, 1] : (tensor<4x2xf32>) -> tensor<4x2xf32>
// CHECK-NEXT:      %68 = stablehlo.broadcast_in_dim %65, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1xf32>
// CHECK-NEXT:      %69 = stablehlo.broadcast_in_dim %68, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1xf32>
// CHECK-NEXT:      %70 = stablehlo.dot_general %67, %69, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x2xf32>, tensor<2x1xf32>) -> tensor<4x1xf32>
// CHECK-NEXT:      %71 = stablehlo.transpose %70, dims = [1, 0] : (tensor<4x1xf32>) -> tensor<1x4xf32>
// CHECK-NEXT:      %72 = stablehlo.reshape %71 : (tensor<1x4xf32>) -> tensor<4xf32>
// CHECK-NEXT:      %73 = stablehlo.transpose %72, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:      %74 = stablehlo.reshape %9 : (tensor<1x1x2xf32>) -> tensor<1x2xf32>
// CHECK-NEXT:      %75 = stablehlo.reshape %74 : (tensor<1x2xf32>) -> tensor<2xf32>
// CHECK-NEXT:      %76 = stablehlo.transpose %75, dims = [0] : (tensor<2xf32>) -> tensor<2xf32>
// CHECK-NEXT:      %77 = stablehlo.transpose %76, dims = [0] : (tensor<2xf32>) -> tensor<2xf32>
// CHECK-NEXT:      %78 = stablehlo.reshape %77 : (tensor<2xf32>) -> tensor<1x2xf32>
// CHECK-NEXT:      %79 = stablehlo.transpose %78, dims = [1, 0] : (tensor<1x2xf32>) -> tensor<2x1xf32>
// CHECK-NEXT:      %80 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<4x2xf32>) -> tensor<4x2xf32>
// CHECK-NEXT:      %81 = stablehlo.broadcast_in_dim %80, dims = [0, 1] : (tensor<4x2xf32>) -> tensor<4x2xf32>
// CHECK-NEXT:      %82 = stablehlo.broadcast_in_dim %79, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1xf32>
// CHECK-NEXT:      %83 = stablehlo.broadcast_in_dim %82, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1xf32>
// CHECK-NEXT:      %84 = stablehlo.dot_general %81, %83, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x2xf32>, tensor<2x1xf32>) -> tensor<4x1xf32>
// CHECK-NEXT:      %85 = stablehlo.transpose %84, dims = [1, 0] : (tensor<4x1xf32>) -> tensor<1x4xf32>
// CHECK-NEXT:      %86 = stablehlo.reshape %85 : (tensor<1x4xf32>) -> tensor<4xf32>
// CHECK-NEXT:      %87 = stablehlo.transpose %86, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:      %88 = stablehlo.reshape %7 : (tensor<1x1x2xf32>) -> tensor<1x2xf32>
// CHECK-NEXT:      %89 = stablehlo.reshape %88 : (tensor<1x2xf32>) -> tensor<2xf32>
// CHECK-NEXT:      %90 = stablehlo.transpose %89, dims = [0] : (tensor<2xf32>) -> tensor<2xf32>
// CHECK-NEXT:      %91 = stablehlo.transpose %90, dims = [0] : (tensor<2xf32>) -> tensor<2xf32>
// CHECK-NEXT:      %92 = stablehlo.reshape %91 : (tensor<2xf32>) -> tensor<1x2xf32>
// CHECK-NEXT:      %93 = stablehlo.transpose %92, dims = [1, 0] : (tensor<1x2xf32>) -> tensor<2x1xf32>
// CHECK-NEXT:      %94 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<4x2xf32>) -> tensor<4x2xf32>
// CHECK-NEXT:      %95 = stablehlo.broadcast_in_dim %94, dims = [0, 1] : (tensor<4x2xf32>) -> tensor<4x2xf32>
// CHECK-NEXT:      %96 = stablehlo.broadcast_in_dim %93, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1xf32>
// CHECK-NEXT:      %97 = stablehlo.broadcast_in_dim %96, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1xf32>
// CHECK-NEXT:      %98 = stablehlo.dot_general %95, %97, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x2xf32>, tensor<2x1xf32>) -> tensor<4x1xf32>
// CHECK-NEXT:      %99 = stablehlo.transpose %98, dims = [1, 0] : (tensor<4x1xf32>) -> tensor<1x4xf32>
// CHECK-NEXT:      %100 = stablehlo.reshape %99 : (tensor<1x4xf32>) -> tensor<4xf32>
// CHECK-NEXT:      %101 = stablehlo.transpose %100, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:      %102 = stablehlo.add %31, %45 : tensor<4xf32>
// CHECK-NEXT:      %103 = stablehlo.broadcast_in_dim %102, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:      %104 = stablehlo.broadcast_in_dim %103, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:      %105 = stablehlo.add %104, %59 : tensor<4xf32>
// CHECK-NEXT:      %106 = stablehlo.broadcast_in_dim %105, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:      %107 = stablehlo.broadcast_in_dim %106, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:      %108 = stablehlo.add %107, %73 : tensor<4xf32>
// CHECK-NEXT:      %109 = stablehlo.broadcast_in_dim %108, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:      %110 = stablehlo.broadcast_in_dim %109, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:      %111 = stablehlo.add %110, %87 : tensor<4xf32>
// CHECK-NEXT:      %112 = stablehlo.broadcast_in_dim %111, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:      %113 = stablehlo.broadcast_in_dim %112, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:      %114 = stablehlo.add %113, %101 : tensor<4xf32>
// CHECK-NEXT:      %115 = stablehlo.broadcast_in_dim %114, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:      %116 = stablehlo.broadcast_in_dim %115, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:      %117 = stablehlo.transpose %116, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:      %118 = stablehlo.transpose %0, dims = [1, 0] : (tensor<2x6xf32>) -> tensor<6x2xf32>
// CHECK-NEXT:      %119 = stablehlo.transpose %1, dims = [1, 0] : (tensor<4x2xf32>) -> tensor<2x4xf32>
// CHECK-NEXT:      return %117, %118, %119 : tensor<4xf32>, tensor<6x2xf32>, tensor<2x4xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func private @batched_enzymexla_unbatched_SliceToBatch_0(%arg0: tensor<6x2x1xf32>) -> tensor<6x1x2xf32> {
// CHECK-NEXT:      %0 = stablehlo.transpose %arg0, dims = [0, 2, 1] : (tensor<6x2x1xf32>) -> tensor<6x1x2xf32>
// CHECK-NEXT:      return %0 : tensor<6x1x2xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func private @batched_enzymexla_unbatched_SliceToBatch_1(%arg0: tensor<6x1x1x2xf32>) -> tensor<6x1x1x2xf32> {
// CHECK-NEXT:      %0 = stablehlo.convert %arg0 : tensor<6x1x1x2xf32>
// CHECK-NEXT:      return %0 : tensor<6x1x1x2xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
