module @"reactant_run!" attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func private @foo(%array: memref<85x180x18xf64, 1>, %sum_res: memref<85x180xf64, 1>) {
    %cst = arith.constant 5.000000e-01 : f64
    affine.parallel (%i, %j) = (0, 0) to (85, 180) {
      %rsum = affine.for %k = 0 to 18 iter_args(%running_sum = %cst) -> (f64) {
        %ld = affine.load %array[%i, %j, %k] : memref<85x180x18xf64, 1>
        %add = arith.addf %ld, %running_sum : f64
        affine.yield %add : f64
      }
      affine.store %rsum, %sum_res[%i, %j] : memref<85x180xf64, 1>
    }
    return
  }
}
