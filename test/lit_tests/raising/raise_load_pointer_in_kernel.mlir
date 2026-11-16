// RUN: not enzymexlamlir-opt --raise-affine-to-stablehlo %s 2>&1 | FileCheck %s

// The kernel reads a buffer's pointer out of a table of pointers: a pointer
// read inside the kernel is no value a tensor holds, so the kernel fails to
// raise (rather than reading it as one and viewing that).
module {
  func.func @table(%tbl: !llvm.ptr, %out: memref<?xi32>) {
    %ptrs = "enzymexla.pointer2memref"(%tbl) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c4, %c1, %c1) ({
      affine.parallel (%i) = (0) to (4) {
        %p = affine.load %ptrs[1] : memref<?x!llvm.ptr>
        %v = "enzymexla.pointer2memref"(%p) : (!llvm.ptr) -> memref<?xi32>
        %x = affine.load %v[%i] : memref<?xi32>
        affine.store %x, %out[%i] : memref<?xi32>
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    func.return
  }
}

// CHECK-NOT: LLVM ERROR
// CHECK: failed to raise operation: affine.parallel (%{{.+}}) = (0) to (4) {
// CHECK-NOT: LLVM ERROR
