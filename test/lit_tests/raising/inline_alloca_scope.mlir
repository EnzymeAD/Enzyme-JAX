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

// CHECK:    func.func private @scoped_raised(%[[a1:.+]]: tensor<16xf64>, %[[a2:.+]]: tensor<16xf64>) -> (tensor<16xf64>, tensor<16xf64>) {
// CHECK-NEXT:    %[[a3:.+]] = stablehlo.constant dense<2.000000e+00> : tensor<f64>
// CHECK-NEXT:    %[[a4:.+]] = stablehlo.iota dim = 0 : tensor<16xi64>
// CHECK-NEXT:    %[[a5:.+]] = stablehlo.constant dense<0> : tensor<16xi64>
// CHECK-NEXT:    %[[a6:.+]] = stablehlo.add %[[a4]], %[[a5]] : tensor<16xi64>
// CHECK-NEXT:    %[[a7:.+]] = stablehlo.constant dense<1> : tensor<16xi64>
// CHECK-NEXT:    %[[a8:.+]] = stablehlo.multiply %[[a6]], %[[a7]] : tensor<16xi64>
// CHECK-NEXT:    %[[a9:.+]] = stablehlo.broadcast_in_dim %[[a3]], dims = [] : (tensor<f64>) -> tensor<16xf64>
// CHECK-NEXT:    %[[a10:.+]] = arith.mulf %[[a2]], %[[a9]] : tensor<16xf64>
// CHECK-NEXT:    %[[a11:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a12:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a13:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a14:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a15:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a16:.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %[[a17:.+]] = stablehlo.dynamic_update_slice %[[a1]], %[[a10]], %[[a16]] : (tensor<16xf64>, tensor<16xf64>, tensor<i64>) -> tensor<16xf64>
// CHECK-NEXT:    return %[[a17]], %[[a2]] : tensor<16xf64>, tensor<16xf64>
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

// CHECK:    func.func private @trapped_raised(%[[a1:.+]]: tensor<16xf64>, %[[a2:.+]]: tensor<16xf64>) -> (tensor<16xf64>, tensor<16xf64>) {
// CHECK-NEXT:    %[[a3:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %[[a4:.+]] = stablehlo.iota dim = 0 : tensor<16xi64>
// CHECK-NEXT:    %[[a5:.+]] = stablehlo.constant dense<0> : tensor<16xi64>
// CHECK-NEXT:    %[[a6:.+]] = stablehlo.add %[[a4]], %[[a5]] : tensor<16xi64>
// CHECK-NEXT:    %[[a7:.+]] = stablehlo.constant dense<1> : tensor<16xi64>
// CHECK-NEXT:    %[[a8:.+]] = stablehlo.multiply %[[a6]], %[[a7]] : tensor<16xi64>
// CHECK-NEXT:    %[[a9:.+]] = stablehlo.broadcast_in_dim %[[a3]], dims = [] : (tensor<f64>) -> tensor<16xf64>
// CHECK-NEXT:    %[[a10:.+]] = arith.cmpf uge, %[[a2]], %[[a9]] : tensor<16xf64>
// CHECK-NEXT:    %[[a11:.+]] = arith.addf %[[a2]], %[[a2]] : tensor<16xf64>
// CHECK-NEXT:    %[[a12:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a13:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a14:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a15:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a16:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a17:.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %[[a18:.+]] = stablehlo.dynamic_update_slice %[[a1]], %[[a11]], %[[a17]] : (tensor<16xf64>, tensor<16xf64>, tensor<i64>) -> tensor<16xf64>
// CHECK-NEXT:    return %[[a18]], %[[a2]] : tensor<16xf64>, tensor<16xf64>
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

// CHECK:    func.func private @chained_raised(%[[a1:.+]]: tensor<16xf64>, %[[a2:.+]]: tensor<16xf64>) -> (tensor<16xf64>, tensor<16xf64>) {
// CHECK-NEXT:    %[[a3:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// CHECK-NEXT:    %[[a4:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %[[a5:.+]] = stablehlo.iota dim = 0 : tensor<16xi64>
// CHECK-NEXT:    %[[a6:.+]] = stablehlo.constant dense<0> : tensor<16xi64>
// CHECK-NEXT:    %[[a7:.+]] = stablehlo.add %[[a5]], %[[a6]] : tensor<16xi64>
// CHECK-NEXT:    %[[a8:.+]] = stablehlo.constant dense<1> : tensor<16xi64>
// CHECK-NEXT:    %[[a9:.+]] = stablehlo.multiply %[[a7]], %[[a8]] : tensor<16xi64>
// CHECK-NEXT:    %[[a10:.+]] = stablehlo.broadcast_in_dim %[[a4]], dims = [] : (tensor<f64>) -> tensor<16xf64>
// CHECK-NEXT:    %[[a11:.+]] = arith.cmpf uge, %[[a2]], %[[a10]] : tensor<16xf64>
// CHECK-NEXT:    %[[a12:.+]] = stablehlo.broadcast_in_dim %[[a3]], dims = [] : (tensor<f64>) -> tensor<16xf64>
// CHECK-NEXT:    %[[a13:.+]] = arith.cmpf ule, %[[a2]], %[[a12]] : tensor<16xf64>
// CHECK-NEXT:    %[[a14:.+]] = arith.mulf %[[a2]], %[[a2]] : tensor<16xf64>
// CHECK-NEXT:    %[[a15:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a16:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a17:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a18:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a19:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a20:.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %[[a21:.+]] = stablehlo.dynamic_update_slice %[[a1]], %[[a14]], %[[a20]] : (tensor<16xf64>, tensor<16xf64>, tensor<i64>) -> tensor<16xf64>
// CHECK-NEXT:    return %[[a21]], %[[a2]] : tensor<16xf64>, tensor<16xf64>
// CHECK-NEXT:  }
