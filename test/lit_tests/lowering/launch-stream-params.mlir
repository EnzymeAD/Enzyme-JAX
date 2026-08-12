// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-polygeist-to-llvm{backend=cuda})" | FileCheck %s

module attributes {gpu.container_module} {
  llvm.func @launch(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i32) {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c1_i64 = arith.constant 1 : i64
    %stream = llvm.inttoptr %c1_i64 : i64 to !llvm.ptr
    gpu.launch_func <%stream : !llvm.ptr> @test_module::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : !llvm.ptr, %arg1 : !llvm.ptr, %arg2 : i32)
    llvm.return
  }

  gpu.module @test_module {
    gpu.func @test_kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i32) kernel {
      gpu.return
    }
  }
}

// CHECK-LABEL: llvm.func @launch
// CHECK-DAG:     %[[STREAM:.+]] = llvm.inttoptr

// CHECK-DAG:     %[[THREE:.+]] = llvm.mlir.constant(3 : i32) : i32
// CHECK-DAG:     %[[ONE:.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-DAG:     %[[SLOT2:.+]] = llvm.alloca %[[ONE]] x i32
// CHECK-DAG:     %[[SLOT1:.+]] = llvm.alloca %[[ONE]] x !llvm.ptr
// CHECK-DAG:     %[[SLOT0:.+]] = llvm.alloca %[[ONE]] x !llvm.ptr
// CHECK-DAG:     %[[ARRAY:.+]] = llvm.alloca %[[THREE]] x !llvm.ptr

// CHECK:         llvm.store %arg0, %[[SLOT0]]
// CHECK:         llvm.store %[[SLOT0]], %[[ARRAY]]
// CHECK:         llvm.store %arg1, %[[SLOT1]]
// CHECK:         %[[GEP1:.+]] = llvm.getelementptr %[[ARRAY]][1]
// CHECK:         llvm.store %[[SLOT1]], %[[GEP1]]
// CHECK:         llvm.store %arg2, %[[SLOT2]]
// CHECK:         %[[GEP2:.+]] = llvm.getelementptr %[[ARRAY]][2]
// CHECK:         llvm.store %[[SLOT2]], %[[GEP2]]

// CHECK:         llvm.call @cudaLaunchKernel({{.*}}, %[[ARRAY]], {{.*}}, %[[STREAM]])
