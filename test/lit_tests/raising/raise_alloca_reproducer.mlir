// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(polygeist-mem2reg,canonicalize,llvm-to-affine-access,canonicalize,raise-affine-to-stablehlo)" | FileCheck %s

module {
  func.func @test_alloca(%arg0: memref<2xi64>, %out: memref<1xi64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %e0 = memref.load %arg0[%c0] : memref<2xi64>
    %e1 = memref.load %arg0[%c1] : memref<2xi64>
    
    %undef = llvm.mlir.undef : !llvm.array<2 x i64>
    %v0 = llvm.insertvalue %e0, %undef[0] : !llvm.array<2 x i64>
    %val = llvm.insertvalue %e1, %v0[1] : !llvm.array<2 x i64>
    
    %c1_i32 = arith.constant 1 : i32
    %alloca = llvm.alloca %c1_i32 x !llvm.array<2 x i64> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    
    // Store the whole array
    llvm.store %val, %alloca : !llvm.array<2 x i64>, !llvm.ptr
    
    // GEP for index 1
    %c0_i32_llvm = llvm.mlir.constant(0 : i32) : i32
    %c1_i32_llvm = llvm.mlir.constant(1 : i32) : i32
    %gep1 = llvm.getelementptr %alloca[%c0_i32_llvm, %c1_i32_llvm] : (!llvm.ptr, i32, i32) -> !llvm.ptr, !llvm.array<2 x i64>
    %res = llvm.load %gep1 : !llvm.ptr -> i64
    
    memref.store %res, %out[%c0] : memref<1xi64>
    return
  }
}

// CHECK:      func.func private @test_alloca_raised(%arg0: tensor<2xi64>, %arg1: tensor<1xi64>) -> (tensor<2xi64>, tensor<1xi64>) {
// CHECK:        %[[GATHER:.+]] = "stablehlo.gather"
// CHECK:        %[[BCAST:.+]] = stablehlo.broadcast_in_dim %[[GATHER]]
// CHECK:        %[[SCATTER:.+]] = "stablehlo.scatter"(%arg1, %{{.+}}, %[[BCAST]])
// CHECK:        return %arg0, %[[SCATTER]] : tensor<2xi64>, tensor<1xi64>
// CHECK:      }
