// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=if_remove_unused;dot_general_slice_to_batch;gather_slice_to_batch;iota_slice_to_batch;reduce_slice_to_batch;sort_slice_to_batch;transpose_slice_to_batch;broadcastindim_slice_to_batch;reducewindow_slice_to_batch;elementwise_slice_to_batch;slice_reshape_pad<1>;elementwise_reshape_like;reorder_elementwise_and_shape_op<16>;transpose_all_users_slice},transform-interpreter,enzyme-hlo-remove-transform,enzyme-hlo-opt)" %s | FileCheck %s

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

// CHECK:  func.func @main(%arg0: tensor<6x2xf32> {tf.aliasing_output = 1 : i32}, %arg1: tensor<2x4xf32> {tf.aliasing_output = 2 : i32}) -> (tensor<4xf32>, tensor<6x2xf32>, tensor<2x4xf32>) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
// CHECK-NEXT:    %0 = stablehlo.reshape %arg0 : (tensor<6x2xf32>) -> tensor<6x1x2xf32>
// CHECK-NEXT:    %1 = stablehlo.slice %0 [0:1, 0:1, 0:2] : (tensor<6x1x2xf32>) -> tensor<1x1x2xf32>
// CHECK-NEXT:    %2 = stablehlo.slice %0 [1:2, 0:1, 0:2] : (tensor<6x1x2xf32>) -> tensor<1x1x2xf32>
// CHECK-NEXT:    %3 = stablehlo.slice %0 [2:3, 0:1, 0:2] : (tensor<6x1x2xf32>) -> tensor<1x1x2xf32>
// CHECK-NEXT:    %4 = stablehlo.slice %0 [3:4, 0:1, 0:2] : (tensor<6x1x2xf32>) -> tensor<1x1x2xf32>
// CHECK-NEXT:    %5 = stablehlo.slice %0 [4:5, 0:1, 0:2] : (tensor<6x1x2xf32>) -> tensor<1x1x2xf32>
// CHECK-NEXT:    %6 = stablehlo.slice %0 [5:6, 0:1, 0:2] : (tensor<6x1x2xf32>) -> tensor<1x1x2xf32>
// CHECK-NEXT:    %7 = stablehlo.reshape %1 : (tensor<1x1x2xf32>) -> tensor<2xf32>
// CHECK-NEXT:    %8 = stablehlo.reshape %2 : (tensor<1x1x2xf32>) -> tensor<2xf32>
// CHECK-NEXT:    %9 = stablehlo.reshape %3 : (tensor<1x1x2xf32>) -> tensor<2xf32>
// CHECK-NEXT:    %10 = stablehlo.reshape %4 : (tensor<1x1x2xf32>) -> tensor<2xf32>
// CHECK-NEXT:    %11 = stablehlo.reshape %5 : (tensor<1x1x2xf32>) -> tensor<2xf32>
// CHECK-NEXT:    %12 = stablehlo.reshape %6 : (tensor<1x1x2xf32>) -> tensor<2xf32>
// CHECK-NEXT:    %13 = stablehlo.add %7, %8 : tensor<2xf32>
// CHECK-NEXT:    %14 = stablehlo.add %13, %9 : tensor<2xf32>
// CHECK-NEXT:    %15 = stablehlo.add %14, %10 : tensor<2xf32>
// CHECK-NEXT:    %16 = stablehlo.add %15, %11 : tensor<2xf32>
// CHECK-NEXT:    %17 = stablehlo.add %16, %12 : tensor<2xf32>
// CHECK-NEXT:    %18 = stablehlo.dot_general %17, %arg1, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2xf32>, tensor<2x4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    return %18, %arg0, %arg1 : tensor<4xf32>, tensor<6x2xf32>, tensor<2x4xf32>
// CHECK-NEXT:  }
