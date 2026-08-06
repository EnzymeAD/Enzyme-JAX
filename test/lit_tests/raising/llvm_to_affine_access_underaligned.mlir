// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(llvm-to-affine-access)" | FileCheck %s --check-prefix=RAISE
// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(llvm-to-affine-access,lower-aligned-affine-accesses,lower-affine)" | FileCheck %s --check-prefix=MEMREF
// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(llvm-to-affine-access,lower-aligned-affine-accesses,lower-affine,convert-polygeist-to-llvm{backend=cpu})" | FileCheck %s --check-prefix=LLVM

// An access must come back out of the raising at the alignment it promised
// going in. clang says `load i128, align 8` for the 16-byte member swap in
// MFEM's Mesh::Swap, at a struct offset that is 8 mod 16; the raised
// affine.load carried the alignment only as an attribute, upstream
// lower-affine rebuilt the access without it, and the final llvm.load was
// emitted at i128's ABI alignment of 16 -- so the backend's movaps trapped.
//
// The alignment rides the attribute through every stage:
// llvm-to-affine-access records it, lower-aligned-affine-accesses lowers the
// accesses that carry it before lower-affine can drop it, and the polygeist
// lowering reads it back onto the llvm access it emits.

module {
  llvm.func @swap16(%a: !llvm.ptr, %b: !llvm.ptr) {
    %pa = llvm.getelementptr inbounds|nuw %a[5488] : (!llvm.ptr) -> !llvm.ptr, i8
    %pb = llvm.getelementptr inbounds|nuw %b[5488] : (!llvm.ptr) -> !llvm.ptr, i8
    %old = llvm.load %pa {alignment = 8 : i64} : !llvm.ptr -> i128
    llvm.store %old, %pb {alignment = 8 : i64} : i128, !llvm.ptr
    llvm.return
  }
}

// The raising still converts the access; the alignment is on the attribute.
// RAISE-LABEL: llvm.func @swap16
// RAISE:         affine.load %{{.*}}[343] {alignment = 8 : i64, ordering = 0 : i64} : memref<?xi128>
// RAISE:         affine.store %{{.*}}[343] {alignment = 8 : i64, ordering = 0 : i64} : memref<?xi128>

// lower-aligned-affine-accesses takes the attributed accesses down to memref
// before lower-affine would have dropped the attribute.
// MEMREF-LABEL: llvm.func @swap16
// MEMREF:         memref.load %{{.*}} {alignment = 8 : i64, ordering = 0 : i64} : memref<?xi128>
// MEMREF:         memref.store %{{.*}} {alignment = 8 : i64, ordering = 0 : i64} : memref<?xi128>

// And the llvm access comes out at the alignment that went in.
// LLVM-LABEL: llvm.func @swap16
// LLVM:         llvm.load %{{.*}} {alignment = 8 : i64{{.*}}} : !llvm.ptr -> i128
// LLVM:         llvm.store %{{.*}} {alignment = 8 : i64{{.*}}} : i128, !llvm.ptr
