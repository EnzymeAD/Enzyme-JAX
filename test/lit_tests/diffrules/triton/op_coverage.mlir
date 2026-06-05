// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=test_triton_ops outfn= argTys=enzyme_dup,enzyme_dup,enzyme_dup retTys=enzyme_dup,enzyme_dup mode=ForwardMode" --canonicalize | FileCheck %s

module {
  tt.func @test_triton_ops(%ptr_in: !tt.ptr<f32>, %ptr_out: !tt.ptr<f32>, %vec: tensor<64xf32>) -> (tensor<64x32xf32>, f32) {
    %pid = tt.get_program_id x : i32
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>

    // splat + addptr + load
    %in_splat = tt.splat %ptr_in : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %in_ptrs = tt.addptr %in_splat, %range : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %loaded = tt.load %in_ptrs : tensor<64x!tt.ptr<f32>>

    // store loaded data to output pointer
    %out_splat = tt.splat %ptr_out : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %out_ptrs = tt.addptr %out_splat, %range : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    tt.store %out_ptrs, %loaded : tensor<64x!tt.ptr<f32>>

    // expand_dims + broadcast to build a matrix from %vec
    %col = tt.expand_dims %vec {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
    %mat = tt.broadcast %col : tensor<64x1xf32> -> tensor<64x32xf32>

    // dot: mat @ constant_matrix + zero
    %cst = arith.constant dense<1.0> : tensor<32x32xf32>
    %zero = arith.constant dense<0.0> : tensor<64x32xf32>
    %dot = tt.dot %mat, %cst, %zero, inputPrecision = tf32 : tensor<64x32xf32> * tensor<32x32xf32> -> tensor<64x32xf32>

    // reduce sum over %vec
    %sum = "tt.reduce"(%vec) <{axis = 0 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %add = arith.addf %lhs, %rhs : f32
      tt.reduce.return %add : f32
    }) : (tensor<64xf32>) -> f32

    tt.return %dot, %sum : tensor<64x32xf32>, f32
  }
}

// After forward-mode AD with enzyme_dup on all 3 args:
//   ptr_in (%arg0) + shadow (%arg1)
//   ptr_out (%arg2) + shadow (%arg3)
//   vec (%arg4) + shadow vec (%arg5)
// Returns: (primal_dot, shadow_dot, primal_sum, shadow_sum)
//
// The differentiated function interleaves shadow and primal operations:
//   - get_program_id is inactive and unused (DCE'd)
//   - make_range is inactive (shared by shadow and primal)
//   - splat/addptr/load are duplicated for shadow and primal pointers
//   - store writes shadow data to shadow ptr, primal data to primal ptr
//   - expand_dims/broadcast/dot are duplicated for shadow and primal vec
//   - reduce is duplicated for shadow and primal vec

// CHECK:      tt.func @test_triton_ops(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: !tt.ptr<f32>, %arg4: tensor<64xf32>, %arg5: tensor<64xf32>) -> (tensor<64x32xf32>, tensor<64x32xf32>, f32, f32) {
// CHECK-NEXT:   %[[cst:.+]] = arith.constant dense<0.000000e+00> : tensor<64x32xf32>
// CHECK-NEXT:   %[[cst_0:.+]] = arith.constant dense<1.000000e+00> : tensor<32x32xf32>
// CHECK-NEXT:   %[[v0:.+]] = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK-NEXT:   %[[v1:.+]] = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:   %[[v2:.+]] = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:   %[[v3:.+]] = tt.addptr %[[v1]], %[[v0]] : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
// CHECK-NEXT:   %[[v4:.+]] = tt.addptr %[[v2]], %[[v0]] : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
// CHECK-NEXT:   %[[v5:.+]] = tt.load %[[v3]] : tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:   %[[v6:.+]] = tt.load %[[v4]] : tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:   %[[v7:.+]] = tt.splat %arg3 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:   %[[v8:.+]] = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:   %[[v9:.+]] = tt.addptr %[[v7]], %[[v0]] : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
// CHECK-NEXT:   %[[v10:.+]] = tt.addptr %[[v8]], %[[v0]] : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
// CHECK-NEXT:   tt.store %[[v9]], %[[v5]] : tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:   tt.store %[[v10]], %[[v6]] : tensor<64x!tt.ptr<f32>>
// CHECK-NEXT:   %[[v11:.+]] = tt.expand_dims %arg5 {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
// CHECK-NEXT:   %[[v12:.+]] = tt.expand_dims %arg4 {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
// CHECK-NEXT:   %[[v13:.+]] = tt.broadcast %[[v11]] : tensor<64x1xf32> -> tensor<64x32xf32>
// CHECK-NEXT:   %[[v14:.+]] = tt.broadcast %[[v12]] : tensor<64x1xf32> -> tensor<64x32xf32>
// CHECK-NEXT:   %[[v15:.+]] = tt.dot %[[v13]], %[[cst_0]], %[[cst]], inputPrecision = tf32 : tensor<64x32xf32> * tensor<32x32xf32> -> tensor<64x32xf32>
// CHECK-NEXT:   %[[v16:.+]] = tt.dot %[[v14]], %[[cst_0]], %[[cst]], inputPrecision = tf32 : tensor<64x32xf32> * tensor<32x32xf32> -> tensor<64x32xf32>
// CHECK-NEXT:   %[[v17:.+]] = "tt.reduce"(%arg5) <{axis = 0 : i32}> ({
// CHECK-NEXT:   ^bb0(%arg6: f32, %arg7: f32):
// CHECK-NEXT:     %[[v19:.+]] = arith.addf %arg6, %arg7 : f32
// CHECK-NEXT:     tt.reduce.return %[[v19]] : f32
// CHECK-NEXT:   }) : (tensor<64xf32>) -> f32
// CHECK-NEXT:   %[[v18:.+]] = "tt.reduce"(%arg4) <{axis = 0 : i32}> ({
// CHECK-NEXT:   ^bb0(%arg6: f32, %arg7: f32):
// CHECK-NEXT:     %[[v20:.+]] = arith.addf %arg6, %arg7 : f32
// CHECK-NEXT:     tt.reduce.return %[[v20]] : f32
// CHECK-NEXT:   }) : (tensor<64xf32>) -> f32
// CHECK-NEXT:   tt.return %[[v16]], %[[v15]], %[[v18]], %[[v17]] : tensor<64x32xf32>, tensor<64x32xf32>, f32, f32
// CHECK-NEXT: }
