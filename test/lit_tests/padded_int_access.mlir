// RUN: enzymexlamlir-opt %s --llvm-to-affine-access | FileCheck %s

// Memref indexing strides by an element's allocation size, and both the
// affine-map division here and the later byte-GEP folds assume that stride
// equals the store size. i480 stores 60 bytes but strides 64: indexed as a
// memref it reads element k at byte 64k instead of 60k, and its memref form
// also trades the op's explicit align 4 for the type's preferred 16. SROA
// coalesces struct copies into exactly such wide stores -- libstdc++'s
// stable_sort moving 60-byte values handed MFEM reads of freed memory that
// way -- so accesses of padded types must stay in llvm dialect form.

module attributes {dlti.dl_spec = #dlti.dl_spec<i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, "dlti.endianness" = "little", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>>} {
  llvm.func @padded_stays_llvm(%src: !llvm.ptr, %dst: !llvm.ptr, %i: i64) {
    %p = llvm.getelementptr inbounds %src[%i] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %v = llvm.load %p {alignment = 4 : i64} : !llvm.ptr -> i480
    %q = llvm.getelementptr inbounds %dst[%i] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %v, %q {alignment = 4 : i64} : i480, !llvm.ptr
    llvm.return
  }

  llvm.func @unpadded_converts(%src: !llvm.ptr, %i: i64) -> i32 {
    %p = llvm.getelementptr inbounds %src[%i] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %v = llvm.load %p {alignment = 4 : i64} : !llvm.ptr -> i32
    llvm.return %v : i32
  }
}

// CHECK-LABEL: llvm.func @padded_stays_llvm
// CHECK: llvm.load %{{.*}} {alignment = 4 : i64} : !llvm.ptr -> i480
// CHECK: llvm.store %{{.*}}, %{{.*}} {alignment = 4 : i64} : i480, !llvm.ptr

// CHECK-LABEL: llvm.func @unpadded_converts
// CHECK: %[[M:.+]] = "enzymexla.pointer2memref"(%arg0)
// CHECK: affine.load %[[M]][symbol(%{{.+}})] {alignment = 4 : i64
