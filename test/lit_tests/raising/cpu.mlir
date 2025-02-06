// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-llvm-to-cf,enzyme-lift-cf-to-scf,libdevice-funcs-raise,canonicalize,llvm-to-affine-access)" | FileCheck %s
module {
  llvm.func internal unnamed_addr fastcc @throw_boundserror_2676() attributes {dso_local, no_inline, sym_visibility = "private"} {
    llvm.unreachable
  }
  func.func private @kern$par0(%arg0: !llvm.ptr<1>) {
    %0 = llvm.mlir.constant(63 : i32) : i32
    affine.parallel (%arg1, %arg2, %arg3, %arg4, %arg5, %arg6) = (0, 0, 0, 0, 0, 0) to (1, 1, 1, 1, 1, 40) {
      scf.execute_region {
        %1 = arith.index_cast %arg4 : index to i32
        %2 = llvm.icmp "ugt" %1, %0 : i32
        llvm.cond_br %2, ^bb2, ^bb1
      ^bb1:  // pred: ^bb0
        %3 = llvm.zext %1 : i32 to i64
        %4 = llvm.getelementptr inbounds %arg0[%3] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i64
        %5 = llvm.load %4 {alignment = 1 : i64} : !llvm.ptr<1> -> i64
        %6 = llvm.mul %5, %5 : i64
        llvm.store %6, %4 {alignment = 1 : i64} : i64, !llvm.ptr<1>
        scf.yield
      ^bb2:  // pred: ^bb0
        llvm.call fastcc @throw_boundserror_2676() : () -> ()
        scf.yield
      }
    }
    return
  }
  func.func @main(%arg0: tensor<64xi64>) -> tensor<64xi64> {
    %0 = enzymexla.jit_call @kern$par0 (%arg0) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<64xi64>) -> tensor<64xi64>
    return %0 : tensor<64xi64>
  }
}

// CHECK:   func.func private @kern$par0(%arg0: !llvm.ptr<1>) {
// CHECK-NEXT:     %0 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr<1>) -> memref<?xi64, 1>
// CHECK-NEXT:     affine.parallel (%arg1, %arg2, %arg3, %arg4, %arg5, %arg6) = (0, 0, 0, 0, 0, 0) to (1, 1, 1, 1, 1, 40) {
// CHECK-NEXT:       affine.if #set(%arg4) {
// CHECK-NEXT:         llvm.call fastcc @throw_boundserror_2676() : () -> ()
// CHECK-NEXT:       } else {
// CHECK-NEXT:         %1 = affine.load %0[%arg4] : memref<?xi64, 1>
// CHECK-NEXT:         %2 = arith.muli %1, %1 : i64
// CHECK-NEXT:         affine.store %2, %0[%arg4] : memref<?xi64, 1>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
