// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-polygeist-to-llvm{backend=cuda})" | FileCheck %s

module attributes {gpu.container_module} {
  gpu.module @gpum {
    gpu.func @kern() kernel attributes {polygeist.host_symbol = "kern_stub"} {
      gpu.return
    }
    gpu.func @other() kernel {
      gpu.return
    }
  }
  llvm.func @kern_stub() {
    llvm.return
  }
  func.func @caller(%attrs: !llvm.ptr) -> i32 {
    %0 = "enzymexla.gpu_func_attributes"(%attrs) <{fn = @gpum::@kern}> : (!llvm.ptr) -> i32
    %1 = "enzymexla.gpu_func_attributes"(%attrs) <{fn = @gpum::@other}> : (!llvm.ptr) -> i32
    %2 = arith.addi %0, %1 : i32
    return %2 : i32
  }
}

// CHECK-LABEL: llvm.func @caller
// CHECK-DAG:     %[[K:.+]] = llvm.mlir.addressof @kern_stub : !llvm.ptr
// CHECK-DAG:     %[[O:.+]] = llvm.mlir.addressof @__polygeist_gpum_other_device_stub : !llvm.ptr
// CHECK:         llvm.call @cudaFuncGetAttributes(%arg0, %[[K]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.call @cudaFuncGetAttributes(%arg0, %[[O]]) : (!llvm.ptr, !llvm.ptr) -> i32
