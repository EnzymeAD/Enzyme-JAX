



func.func @affine(%arg0: f32, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>) -> () {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %24 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() : memref<4xf32>
    affine.parallel (%arg5) = (0) to (4) {
      %1 = affine.load %arg1[%arg5] : memref<?xf32>
      memref.store %1, %alloc[%arg5] : memref<4xf32>
      %2 = arith.mulf %1, %arg0 : f32
      affine.store %2, %arg3[%arg5] : memref<?xf32>
    }
    %0 = affine.parallel (%arg5) = (0) to (4) reduce ("addf") -> (f32) {
      %1 = memref.load %alloc[%arg5] : memref<4xf32>
      %2 = memref.load %arg4[%arg5] : memref<?xf32>
      memref.store %cst, %arg4[%arg5] : memref<?xf32>
      %3 = arith.mulf %2, %arg0 : f32
      %4 = arith.mulf %2, %1 : f32
      %5 = memref.atomic_rmw addf %3, %arg2[%arg5] : (f32, memref<?xf32>) -> f32
      affine.yield %4 : f32
    }
    memref.dealloc %alloc : memref<4xf32>
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  return
}
