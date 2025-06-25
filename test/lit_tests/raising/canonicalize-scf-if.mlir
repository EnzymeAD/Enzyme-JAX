// RUN: enzymexlamlir-opt --allow-unregistered-dialect --canonicalize-scf-for -split-input-file %s | FileCheck %s

module {
  llvm.mlir.global external @_ZSt4cout() {addr_space = 0 : i32, alignment = 8 : i64, sym_visibility = "private"} : !llvm.struct<"class.std::basic_ostream.1.1", (ptr, struct<"class.std::basic_ios.1.1", (struct<"class.std::ios_base.1.1", (ptr, i64, i64, i32, i32, i32, ptr, struct<"struct.std::ios_base::_Words.1.1", (ptr, i64)>, array<8 x struct<"struct.std::ios_base::_Words.1.1", (ptr, i64)>>, i32, ptr, struct<"class.std::locale.1.1", (ptr)>)>, ptr, i8, i8, ptr, ptr, ptr, ptr)>)>
  func.func @main() -> (i32, i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    %0 = ub.poison : i32
    %c2_i32 = arith.constant 2 : i32
    %1 = ub.poison : !llvm.ptr
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %2 = llvm.mlir.addressof @_ZSt4cout : !llvm.ptr
    %c54_i64 = arith.constant 54 : i64
    %4 = llvm.mlir.zero : !llvm.ptr
    %19 = llvm.load %2 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %20 = llvm.icmp "eq" %19, %4 : !llvm.ptr
    %21:2 = scf.if %20 -> (i32, i32) {
      scf.yield %0, %c0_i32 : i32, i32
    } else {
      scf.yield %c0_i32, %c0_i32 : i32, i32
    }
    func.return %21#0, %21#1 : i32, i32
  }
}

// CHECK:  func.func @main() -> (i32, i32) {
// CHECK-NEXT:    %0 = ub.poison : i32
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    %1 = llvm.mlir.addressof @_ZSt4cout : !llvm.ptr
// CHECK-NEXT:    %2 = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %3 = llvm.load %1 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %4 = llvm.icmp "eq" %3, %2 : !llvm.ptr
// CHECK-NEXT:    %5 = scf.if %4 -> (i32) {
// CHECK-NEXT:      scf.yield %0 : i32
// CHECK-NEXT:    } else {
// CHECK-NEXT:      scf.yield %c0_i32 : i32
// CHECK-NEXT:    }
// CHECK-NEXT:    return %5, %c0_i32 : i32, i32
// CHECK-NEXT:  }
