// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=transpose_is_reshape<16>;transpose_slice_to_batch},transform-interpreter,enzyme-hlo-remove-transform,enzyme-hlo-opt{passses=65536},enzyme-hlo-opt)" %s | FileCheck %s

module @reactant_f_gener... attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func @main(%arg0: tensor<6x2xf32> {tf.aliasing_output = 1 : i32}, %arg1: tensor<2x4xf32> {tf.aliasing_output = 2 : i32}) -> (tensor<4xf32>, tensor<6x2xf32>, tensor<2x4xf32>) attributes {enzymexla.memory_effects = []} {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<6x2xf32>) -> tensor<2x6xf32>
    %1 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<2x4xf32>) -> tensor<4x2xf32>
    %2 = stablehlo.slice %0 [0:2, 0:1] : (tensor<2x6xf32>) -> tensor<2x1xf32>
    %3 = stablehlo.transpose %2, dims = [1, 0] : (tensor<2x1xf32>) -> tensor<1x2xf32>
    %4 = stablehlo.reshape %3 : (tensor<1x2xf32>) -> tensor<2xf32>
    %5 = stablehlo.transpose %4, dims = [0] : (tensor<2xf32>) -> tensor<2xf32>
    %6 = stablehlo.reshape %5 : (tensor<2xf32>) -> tensor<1x2xf32>
    %7 = stablehlo.transpose %6, dims = [1, 0] : (tensor<1x2xf32>) -> tensor<2x1xf32>
    %8 = stablehlo.convert %7 : tensor<2x1xf32>
    %9 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<4x2xf32>) -> tensor<4x2xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1] : (tensor<4x2xf32>) -> tensor<4x2xf32>
    %11 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %13 = stablehlo.dot_general %10, %12, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x2xf32>, tensor<2x1xf32>) -> tensor<4x1xf32>
    %14 = stablehlo.transpose %13, dims = [1, 0] : (tensor<4x1xf32>) -> tensor<1x4xf32>
    %15 = stablehlo.reshape %14 : (tensor<1x4xf32>) -> tensor<4xf32>
    %16 = stablehlo.transpose %15, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %17 = stablehlo.slice %0 [0:2, 1:2] : (tensor<2x6xf32>) -> tensor<2x1xf32>
    %18 = stablehlo.transpose %17, dims = [1, 0] : (tensor<2x1xf32>) -> tensor<1x2xf32>
    %19 = stablehlo.reshape %18 : (tensor<1x2xf32>) -> tensor<2xf32>
    %20 = stablehlo.transpose %19, dims = [0] : (tensor<2xf32>) -> tensor<2xf32>
    %21 = stablehlo.reshape %20 : (tensor<2xf32>) -> tensor<1x2xf32>
    %22 = stablehlo.transpose %21, dims = [1, 0] : (tensor<1x2xf32>) -> tensor<2x1xf32>
    %23 = stablehlo.convert %22 : tensor<2x1xf32>
    %24 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<4x2xf32>) -> tensor<4x2xf32>
    %25 = stablehlo.broadcast_in_dim %24, dims = [0, 1] : (tensor<4x2xf32>) -> tensor<4x2xf32>
    %26 = stablehlo.broadcast_in_dim %23, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %28 = stablehlo.dot_general %25, %27, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x2xf32>, tensor<2x1xf32>) -> tensor<4x1xf32>
    %29 = stablehlo.transpose %28, dims = [1, 0] : (tensor<4x1xf32>) -> tensor<1x4xf32>
    %30 = stablehlo.reshape %29 : (tensor<1x4xf32>) -> tensor<4xf32>
    %31 = stablehlo.transpose %30, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %32 = stablehlo.slice %0 [0:2, 2:3] : (tensor<2x6xf32>) -> tensor<2x1xf32>
    %33 = stablehlo.transpose %32, dims = [1, 0] : (tensor<2x1xf32>) -> tensor<1x2xf32>
    %34 = stablehlo.reshape %33 : (tensor<1x2xf32>) -> tensor<2xf32>
    %35 = stablehlo.transpose %34, dims = [0] : (tensor<2xf32>) -> tensor<2xf32>
    %36 = stablehlo.reshape %35 : (tensor<2xf32>) -> tensor<1x2xf32>
    %37 = stablehlo.transpose %36, dims = [1, 0] : (tensor<1x2xf32>) -> tensor<2x1xf32>
    %38 = stablehlo.convert %37 : tensor<2x1xf32>
    %39 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<4x2xf32>) -> tensor<4x2xf32>
    %40 = stablehlo.broadcast_in_dim %39, dims = [0, 1] : (tensor<4x2xf32>) -> tensor<4x2xf32>
    %41 = stablehlo.broadcast_in_dim %38, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %42 = stablehlo.broadcast_in_dim %41, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %43 = stablehlo.dot_general %40, %42, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x2xf32>, tensor<2x1xf32>) -> tensor<4x1xf32>
    %44 = stablehlo.transpose %43, dims = [1, 0] : (tensor<4x1xf32>) -> tensor<1x4xf32>
    %45 = stablehlo.reshape %44 : (tensor<1x4xf32>) -> tensor<4xf32>
    %46 = stablehlo.transpose %45, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %47 = stablehlo.slice %0 [0:2, 3:4] : (tensor<2x6xf32>) -> tensor<2x1xf32>
    %48 = stablehlo.transpose %47, dims = [1, 0] : (tensor<2x1xf32>) -> tensor<1x2xf32>
    %49 = stablehlo.reshape %48 : (tensor<1x2xf32>) -> tensor<2xf32>
    %50 = stablehlo.transpose %49, dims = [0] : (tensor<2xf32>) -> tensor<2xf32>
    %51 = stablehlo.reshape %50 : (tensor<2xf32>) -> tensor<1x2xf32>
    %52 = stablehlo.transpose %51, dims = [1, 0] : (tensor<1x2xf32>) -> tensor<2x1xf32>
    %53 = stablehlo.convert %52 : tensor<2x1xf32>
    %54 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<4x2xf32>) -> tensor<4x2xf32>
    %55 = stablehlo.broadcast_in_dim %54, dims = [0, 1] : (tensor<4x2xf32>) -> tensor<4x2xf32>
    %56 = stablehlo.broadcast_in_dim %53, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %57 = stablehlo.broadcast_in_dim %56, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %58 = stablehlo.dot_general %55, %57, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x2xf32>, tensor<2x1xf32>) -> tensor<4x1xf32>
    %59 = stablehlo.transpose %58, dims = [1, 0] : (tensor<4x1xf32>) -> tensor<1x4xf32>
    %60 = stablehlo.reshape %59 : (tensor<1x4xf32>) -> tensor<4xf32>
    %61 = stablehlo.transpose %60, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %62 = stablehlo.slice %0 [0:2, 4:5] : (tensor<2x6xf32>) -> tensor<2x1xf32>
    %63 = stablehlo.transpose %62, dims = [1, 0] : (tensor<2x1xf32>) -> tensor<1x2xf32>
    %64 = stablehlo.reshape %63 : (tensor<1x2xf32>) -> tensor<2xf32>
    %65 = stablehlo.transpose %64, dims = [0] : (tensor<2xf32>) -> tensor<2xf32>
    %66 = stablehlo.reshape %65 : (tensor<2xf32>) -> tensor<1x2xf32>
    %67 = stablehlo.transpose %66, dims = [1, 0] : (tensor<1x2xf32>) -> tensor<2x1xf32>
    %68 = stablehlo.convert %67 : tensor<2x1xf32>
    %69 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<4x2xf32>) -> tensor<4x2xf32>
    %70 = stablehlo.broadcast_in_dim %69, dims = [0, 1] : (tensor<4x2xf32>) -> tensor<4x2xf32>
    %71 = stablehlo.broadcast_in_dim %68, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %72 = stablehlo.broadcast_in_dim %71, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %73 = stablehlo.dot_general %70, %72, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x2xf32>, tensor<2x1xf32>) -> tensor<4x1xf32>
    %74 = stablehlo.transpose %73, dims = [1, 0] : (tensor<4x1xf32>) -> tensor<1x4xf32>
    %75 = stablehlo.reshape %74 : (tensor<1x4xf32>) -> tensor<4xf32>
    %76 = stablehlo.transpose %75, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %77 = stablehlo.slice %0 [0:2, 5:6] : (tensor<2x6xf32>) -> tensor<2x1xf32>
    %78 = stablehlo.transpose %77, dims = [1, 0] : (tensor<2x1xf32>) -> tensor<1x2xf32>
    %79 = stablehlo.reshape %78 : (tensor<1x2xf32>) -> tensor<2xf32>
    %80 = stablehlo.transpose %79, dims = [0] : (tensor<2xf32>) -> tensor<2xf32>
    %81 = stablehlo.reshape %80 : (tensor<2xf32>) -> tensor<1x2xf32>
    %82 = stablehlo.transpose %81, dims = [1, 0] : (tensor<1x2xf32>) -> tensor<2x1xf32>
    %83 = stablehlo.convert %82 : tensor<2x1xf32>
    %84 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<4x2xf32>) -> tensor<4x2xf32>
    %85 = stablehlo.broadcast_in_dim %84, dims = [0, 1] : (tensor<4x2xf32>) -> tensor<4x2xf32>
    %86 = stablehlo.broadcast_in_dim %83, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %87 = stablehlo.broadcast_in_dim %86, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %88 = stablehlo.dot_general %85, %87, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x2xf32>, tensor<2x1xf32>) -> tensor<4x1xf32>
    %89 = stablehlo.transpose %88, dims = [1, 0] : (tensor<4x1xf32>) -> tensor<1x4xf32>
    %90 = stablehlo.reshape %89 : (tensor<1x4xf32>) -> tensor<4xf32>
    %91 = stablehlo.transpose %90, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %92 = stablehlo.broadcast_in_dim %16, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %93 = stablehlo.broadcast_in_dim %92, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %94 = stablehlo.broadcast_in_dim %31, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %95 = stablehlo.broadcast_in_dim %94, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %96 = stablehlo.add %93, %95 : tensor<4xf32>
    %97 = stablehlo.broadcast_in_dim %96, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %98 = stablehlo.broadcast_in_dim %97, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %99 = stablehlo.broadcast_in_dim %46, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %100 = stablehlo.broadcast_in_dim %99, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %101 = stablehlo.add %98, %100 : tensor<4xf32>
    %102 = stablehlo.broadcast_in_dim %101, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %103 = stablehlo.broadcast_in_dim %102, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %104 = stablehlo.broadcast_in_dim %61, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %105 = stablehlo.broadcast_in_dim %104, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %106 = stablehlo.add %103, %105 : tensor<4xf32>
    %107 = stablehlo.broadcast_in_dim %106, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %108 = stablehlo.broadcast_in_dim %107, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %109 = stablehlo.broadcast_in_dim %76, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %110 = stablehlo.broadcast_in_dim %109, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %111 = stablehlo.add %108, %110 : tensor<4xf32>
    %112 = stablehlo.broadcast_in_dim %111, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %113 = stablehlo.broadcast_in_dim %112, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %114 = stablehlo.broadcast_in_dim %91, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %115 = stablehlo.broadcast_in_dim %114, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %116 = stablehlo.add %113, %115 : tensor<4xf32>
    %117 = stablehlo.transpose %116, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
    %118 = stablehlo.transpose %arg0, dims = [0, 1] : (tensor<6x2xf32>) -> tensor<6x2xf32>
    %119 = stablehlo.transpose %arg1, dims = [0, 1] : (tensor<2x4xf32>) -> tensor<2x4xf32>
    return %117, %118, %119 : tensor<4xf32>, tensor<6x2xf32>, tensor<2x4xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<6x2xf32> {tf.aliasing_output = 1 : i32}, %arg1: tensor<2x4xf32> {tf.aliasing_output = 2 : i32}) -> (tensor<4xf32>, tensor<6x2xf32>, tensor<2x4xf32>) attributes {enzymexla.memory_effects = []} {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<6x2xf32>, tensor<f32>) -> tensor<2xf32>
// CHECK-NEXT:    %1 = stablehlo.dot_general %arg1, %0, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4xf32>, tensor<2xf32>) -> tensor<4xf32>
// CHECK-NEXT:    return %1, %arg0, %arg1 : tensor<4xf32>, tensor<6x2xf32>, tensor<2x4xf32>
// CHECK-NEXT:  }
