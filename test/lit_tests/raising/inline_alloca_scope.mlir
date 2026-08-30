// RUN: enzymexlamlir-opt %s --split-input-file --raise-affine-to-stablehlo | FileCheck %s

// An alloca scope only delimits stack lifetime, which raised value semantics
// make meaningless: the descent recurses into its body and forwards the
// yielded result.

module {
  func.func private @scoped(%out: memref<16xf64, 1>, %in: memref<16xf64, 1>) {
    affine.parallel (%t) = (0) to (16) {
      %r = memref.alloca_scope -> f64 {
        %v = affine.load %in[%t] : memref<16xf64, 1>
        %two = arith.constant 2.0 : f64
        %d = arith.mulf %v, %two : f64
        memref.alloca_scope.return %d : f64
      }
      affine.store %r, %out[%t] : memref<16xf64, 1>
    }
    return
  }
}

// CHECK:  func.func private @scoped_raised(%[[v1:.+]]: tensor<16xf64>, %[[v2:.+]]: tensor<16xf64>) -> (tensor<16xf64>, tensor<16xf64>) {
// CHECK-NEXT:  %[[v3:.+]] = stablehlo.constant dense<2.000000e+00> : tensor<f64>
// CHECK-NEXT:  %[[v4:.+]] = stablehlo.iota dim = 0 : tensor<16xi64>
// CHECK-NEXT:  %[[v5:.+]] = stablehlo.constant dense<0> : tensor<16xi64>
// CHECK-NEXT:  %[[v6:.+]] = stablehlo.add %[[v4]], %[[v5]] : tensor<16xi64>
// CHECK-NEXT:  %[[v7:.+]] = stablehlo.constant dense<1> : tensor<16xi64>
// CHECK-NEXT:  %[[v8:.+]] = stablehlo.multiply %[[v6]], %[[v7]] : tensor<16xi64>
// CHECK-NEXT:  %[[v9:.+]] = stablehlo.reshape %[[v2]] : (tensor<16xf64>) -> tensor<16xf64>
// CHECK-NEXT:  %[[v10:.+]] = stablehlo.broadcast_in_dim %[[v3]], dims = [] : (tensor<f64>) -> tensor<16xf64>
// CHECK-NEXT:  %[[v11:.+]] = arith.mulf %[[v9]], %[[v10]] : tensor<16xf64>
// CHECK-NEXT:  %[[v12:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:  %[[v13:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:  %[[v14:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:  %[[v15:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:  %[[v16:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:  %[[v17:.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:  %[[v18:.+]] = stablehlo.broadcast_in_dim %[[v11]], dims = [0] : (tensor<16xf64>) -> tensor<16xf64>
// CHECK-NEXT:  %[[v19:.+]] = stablehlo.dynamic_update_slice %[[v1]], %[[v18]], %[[v17]] : (tensor<16xf64>, tensor<16xf64>, tensor<i64>) -> tensor<16xf64>
// CHECK-NEXT:  return %[[v19]], %[[v2]] : tensor<16xf64>, tensor<16xf64>
// CHECK-NEXT:  }

// -----


// Inliner wrappers leave scf.execute_region regions whose extra blocks only
// branch to a trap: blocks guaranteed to end in llvm.unreachable do not
// count as live, so the unique live path raises, forwarding block arguments
// along the way.

module {
  func.func private @trapped(%out: memref<16xf64, 1>, %in: memref<16xf64, 1>) {
    affine.parallel (%t) = (0) to (16) {
      %r = scf.execute_region -> f64 {
        %v = affine.load %in[%t] : memref<16xf64, 1>
        %c0 = arith.constant 0.0 : f64
        %ok = arith.cmpf uge, %v, %c0 : f64
        cf.cond_br %ok, ^live(%v : f64), ^trap
      ^live(%lv: f64):
        %s = arith.addf %lv, %lv : f64
        scf.yield %s : f64
      ^trap:
        llvm.unreachable
      }
      affine.store %r, %out[%t] : memref<16xf64, 1>
    }
    return
  }
}

// CHECK:  func.func private @trapped_raised(%[[v1:.+]]: tensor<16xf64>, %[[v2:.+]]: tensor<16xf64>) -> (tensor<16xf64>, tensor<16xf64>) {
// CHECK-NEXT:  %[[v3:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:  %[[v4:.+]] = stablehlo.iota dim = 0 : tensor<16xi64>
// CHECK-NEXT:  %[[v5:.+]] = stablehlo.constant dense<0> : tensor<16xi64>
// CHECK-NEXT:  %[[v6:.+]] = stablehlo.add %[[v4]], %[[v5]] : tensor<16xi64>
// CHECK-NEXT:  %[[v7:.+]] = stablehlo.constant dense<1> : tensor<16xi64>
// CHECK-NEXT:  %[[v8:.+]] = stablehlo.multiply %[[v6]], %[[v7]] : tensor<16xi64>
// CHECK-NEXT:  %[[v9:.+]] = stablehlo.reshape %[[v2]] : (tensor<16xf64>) -> tensor<16xf64>
// CHECK-NEXT:  %[[v10:.+]] = stablehlo.broadcast_in_dim %[[v3]], dims = [] : (tensor<f64>) -> tensor<16xf64>
// CHECK-NEXT:  %[[v11:.+]] = arith.cmpf uge, %[[v9]], %[[v10]] : tensor<16xf64>
// CHECK-NEXT:  %[[v12:.+]] = arith.addf %[[v9]], %[[v9]] : tensor<16xf64>
// CHECK-NEXT:  %[[v13:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:  %[[v14:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:  %[[v15:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:  %[[v16:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:  %[[v17:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:  %[[v18:.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:  %[[v19:.+]] = stablehlo.broadcast_in_dim %[[v12]], dims = [0] : (tensor<16xf64>) -> tensor<16xf64>
// CHECK-NEXT:  %[[v20:.+]] = stablehlo.dynamic_update_slice %[[v1]], %[[v19]], %[[v18]] : (tensor<16xf64>, tensor<16xf64>, tensor<i64>) -> tensor<16xf64>
// CHECK-NEXT:  return %[[v20]], %[[v2]] : tensor<16xf64>, tensor<16xf64>
// CHECK-NEXT:  }

// -----


// A chain of trap-guarded branches (stacked MFEM_VERIFY-style aborts, the
// second trap reached through its own guard block) still has one live path.

module {
  func.func private @chained(%out: memref<16xf64, 1>, %in: memref<16xf64, 1>) {
    affine.parallel (%t) = (0) to (16) {
      %r = scf.execute_region -> f64 {
        %v = affine.load %in[%t] : memref<16xf64, 1>
        %c0 = arith.constant 0.0 : f64
        %ok = arith.cmpf uge, %v, %c0 : f64
        cf.cond_br %ok, ^next, ^pretrap
      ^next:
        %c1 = arith.constant 1.0 : f64
        %ok2 = arith.cmpf ule, %v, %c1 : f64
        cf.cond_br %ok2, ^live, ^trap
      ^pretrap:
        cf.br ^trap
      ^live:
        %s = arith.mulf %v, %v : f64
        scf.yield %s : f64
      ^trap:
        llvm.unreachable
      }
      affine.store %r, %out[%t] : memref<16xf64, 1>
    }
    return
  }
}

// CHECK:  func.func private @chained_raised(%[[v1:.+]]: tensor<16xf64>, %[[v2:.+]]: tensor<16xf64>) -> (tensor<16xf64>, tensor<16xf64>) {
// CHECK-NEXT:  %[[v3:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// CHECK-NEXT:  %[[v4:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:  %[[v5:.+]] = stablehlo.iota dim = 0 : tensor<16xi64>
// CHECK-NEXT:  %[[v6:.+]] = stablehlo.constant dense<0> : tensor<16xi64>
// CHECK-NEXT:  %[[v7:.+]] = stablehlo.add %[[v5]], %[[v6]] : tensor<16xi64>
// CHECK-NEXT:  %[[v8:.+]] = stablehlo.constant dense<1> : tensor<16xi64>
// CHECK-NEXT:  %[[v9:.+]] = stablehlo.multiply %[[v7]], %[[v8]] : tensor<16xi64>
// CHECK-NEXT:  %[[v10:.+]] = stablehlo.reshape %[[v2]] : (tensor<16xf64>) -> tensor<16xf64>
// CHECK-NEXT:  %[[v11:.+]] = stablehlo.broadcast_in_dim %[[v4]], dims = [] : (tensor<f64>) -> tensor<16xf64>
// CHECK-NEXT:  %[[v12:.+]] = arith.cmpf uge, %[[v10]], %[[v11]] : tensor<16xf64>
// CHECK-NEXT:  %[[v13:.+]] = stablehlo.broadcast_in_dim %[[v3]], dims = [] : (tensor<f64>) -> tensor<16xf64>
// CHECK-NEXT:  %[[v14:.+]] = arith.cmpf ule, %[[v10]], %[[v13]] : tensor<16xf64>
// CHECK-NEXT:  %[[v15:.+]] = arith.mulf %[[v10]], %[[v10]] : tensor<16xf64>
// CHECK-NEXT:  %[[v16:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:  %[[v17:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:  %[[v18:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:  %[[v19:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:  %[[v20:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:  %[[v21:.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:  %[[v22:.+]] = stablehlo.broadcast_in_dim %[[v15]], dims = [0] : (tensor<16xf64>) -> tensor<16xf64>
// CHECK-NEXT:  %[[v23:.+]] = stablehlo.dynamic_update_slice %[[v1]], %[[v22]], %[[v21]] : (tensor<16xf64>, tensor<16xf64>, tensor<i64>) -> tensor<16xf64>
// CHECK-NEXT:  return %[[v23]], %[[v2]] : tensor<16xf64>, tensor<16xf64>
// CHECK-NEXT:  }
