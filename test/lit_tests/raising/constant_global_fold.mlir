// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo | FileCheck %s

// A kernel-local constant table promoted to a rodata global folds into the
// module instead of becoming a kernel argument: constant offsets read the
// element directly, runtime indices select over the elements.

// CHECK-LABEL: @fold_raised
// CHECK-SAME: (%{{.*}}: tensor<16xf64>) -> tensor<16xf64>
// CHECK-NOT: llvm.mlir.addressof

module {
  llvm.mlir.global private constant @map(dense<[0, 3, 6, 1]> : tensor<4xi32>) {addr_space = 0 : i32} : !llvm.array<4 x i32>
  func.func private @fold(%out: memref<16xf64, 1>) {
    %g = llvm.mlir.addressof @map : !llvm.ptr
    %m = "enzymexla.pointer2memref"(%g) : (!llvm.ptr) -> memref<?xi32>
    affine.parallel (%t) = (0) to (16) {
      %c = affine.load %m[2] : memref<?xi32>
      %d = affine.load %m[%t mod 4] : memref<?xi32>
      %s = arith.addi %c, %d : i32
      %f = arith.sitofp %s : i32 to f64
      affine.store %f, %out[%t] : memref<16xf64, 1>
    }
    return
  }
}
