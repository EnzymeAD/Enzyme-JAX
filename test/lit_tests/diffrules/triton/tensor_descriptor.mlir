// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=test_tensor_descriptor outfn= argTys=enzyme_dup,enzyme_dup,enzyme_const retTys=enzyme_dup mode=ForwardMode" --canonicalize | FileCheck %s

module {
  tt.func @test_tensor_descriptor(
      %src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %size: i32)
      -> tensor<16x16xf32> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c16_i64 = arith.constant 16 : i64
    %src_desc = tt.make_tensor_descriptor
        %src, [%size, %size], [%c16_i64, %c1_i64]
        : <f32>, <16x16xf32>
    %dst_desc = tt.make_tensor_descriptor
        %dst, [%size, %size], [%c16_i64, %c1_i64]
        : <f32>, <16x16xf32>
    %tile = tt.descriptor_load %src_desc[%c0_i32, %c0_i32]
        : !tt.tensordesc<16x16xf32> -> tensor<16x16xf32>
    %transposed = tt.trans %tile {order = array<i32: 1, 0>}
        : tensor<16x16xf32> -> tensor<16x16xf32>
    %logged = math.log2 %transposed : tensor<16x16xf32>
    tt.descriptor_store %dst_desc[%c0_i32, %c0_i32], %logged
        : !tt.tensordesc<16x16xf32>, tensor<16x16xf32>
    tt.return %logged : tensor<16x16xf32>
  }
}

// CHECK-LABEL: tt.func @test_tensor_descriptor(
// CHECK-SAME:      %[[SRC:arg[0-9]+]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[DSRC:arg[0-9]+]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[DST:arg[0-9]+]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[DDST:arg[0-9]+]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[SIZE:arg[0-9]+]]: i32)
// CHECK:         %[[LN2:.+]] = arith.constant dense<0.693147{{.*}}>
// CHECK:         %[[DSRC_DESC:[0-9]+]] = tt.make_tensor_descriptor %[[DSRC]],
// CHECK-NEXT:    %[[SRC_DESC:[0-9]+]] = tt.make_tensor_descriptor %[[SRC]],
// CHECK-NEXT:    %[[DDST_DESC:[0-9]+]] = tt.make_tensor_descriptor %[[DDST]],
// CHECK-NEXT:    %[[DST_DESC:[0-9]+]] = tt.make_tensor_descriptor %[[DST]],
// CHECK-NEXT:    %[[DTILE:[0-9]+]] = tt.descriptor_load %[[DSRC_DESC]]
// CHECK-NEXT:    %[[TILE:[0-9]+]] = tt.descriptor_load %[[SRC_DESC]]
// CHECK-NEXT:    %[[DTRANSPOSED:[0-9]+]] = tt.trans %[[DTILE]]
// CHECK-NEXT:    %[[TRANSPOSED:[0-9]+]] = tt.trans %[[TILE]]
// CHECK-NEXT:    %[[DENOM:[0-9]+]] = arith.mulf %[[TRANSPOSED]], %[[LN2]]
// CHECK-NEXT:    %[[DLOGGED:[0-9]+]] = arith.divf %[[DTRANSPOSED]], %[[DENOM]]
// CHECK-NEXT:    %[[LOGGED:[0-9]+]] = math.log2 %[[TRANSPOSED]]
// CHECK-NEXT:    tt.descriptor_store %[[DDST_DESC]]{{.*}}, %[[DLOGGED]]
// CHECK-NEXT:    tt.descriptor_store %[[DST_DESC]]{{.*}}, %[[LOGGED]]
// CHECK-NEXT:    tt.return %[[LOGGED]], %[[DLOGGED]]
