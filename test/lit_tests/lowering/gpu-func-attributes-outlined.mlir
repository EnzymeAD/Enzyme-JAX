// RUN: enzymexlamlir-opt %s --split-input-file --pass-pipeline="builtin.module(convert-parallel-to-gpu2{backend=cuda})" | FileCheck %s

module attributes {gpu.container_module} {
  llvm.func @host(%attrs: !llvm.ptr, %out: !llvm.ptr, %n: i64) {
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %nt = arith.index_cast %n : i64 to index
    %0 = "enzymexla.gpu_func_attributes"(%attrs) <{fn = @__mlir_gpu_module::@__polygeist_empty_kernel, host_symbol = "_Z18__device_stub__fooPi"}> : (!llvm.ptr) -> i32
    %1 = "enzymexla.gpu_func_attributes"(%attrs) <{fn = @__mlir_gpu_module::@__polygeist_empty_kernel, host_symbol = "_Z18__device_stub__barPi"}> : (!llvm.ptr) -> i32
    %2 = "enzymexla.gpu_func_attributes"(%attrs) <{fn = @__mlir_gpu_module::@__polygeist_empty_kernel, host_symbol = "_Z18__device_stub__bazPi"}> : (!llvm.ptr) -> i32
    %3 = "enzymexla.gpu_error"() ({
      gpu.launch_func @main_kernel::@main_kernel blocks in (%c1, %c1, %c1) threads in (%c128, %c1, %c1) args(%out : !llvm.ptr)
      "enzymexla.polygeist_yield"() : () -> ()
    }) {passthrough = [["polygeist.host_symbol", "_Z18__device_stub__fooPi"], ["target-cpu", "sm_120"]]} : () -> index
    %4 = "enzymexla.gpu_error"() ({
      gpu.launch_func @fixed_kernel::@fixed_kernel blocks in (%c1, %c1, %c1) threads in (%c128, %c1, %c1) args(%out : !llvm.ptr)
      "enzymexla.polygeist_yield"() : () -> ()
    }) {passthrough = [["polygeist.host_symbol", "_Z18__device_stub__bazPi"], ["target-cpu", "sm_120"]]} : () -> index
    %5 = "enzymexla.gpu_error"() ({
      gpu.launch_func @free_kernel::@free_kernel blocks in (%c1, %c1, %c1) threads in (%nt, %c1, %c1) args(%out : !llvm.ptr)
      "enzymexla.polygeist_yield"() : () -> ()
    }) {passthrough = [["polygeist.host_symbol", "_Z18__device_stub__bazPi"], ["target-cpu", "sm_120"]]} : () -> index
    llvm.return
  }
  gpu.module @__mlir_gpu_module [#nvvm.target] {
    gpu.func @__polygeist_empty_kernel() kernel {
      gpu.return
    }
  }
  gpu.module @main_kernel [#nvvm.target] {
    gpu.func @main_kernel(%arg0: !llvm.ptr) kernel {
      gpu.return
    }
  }
  gpu.module @fixed_kernel [#nvvm.target] {
    gpu.func @fixed_kernel(%arg0: !llvm.ptr) kernel attributes {known_block_size = array<i32: 128, 1, 1>} {
      gpu.return
    }
  }
  gpu.module @free_kernel [#nvvm.target] {
    gpu.func @free_kernel(%arg0: !llvm.ptr) kernel {
      gpu.return
    }
  }
}

// CHECK-LABEL: llvm.func @host(
// CHECK:         "enzymexla.gpu_func_attributes"(%arg0) <{fn = @main_kernel::@main_kernel}> : (!llvm.ptr) -> i32
// CHECK:         "enzymexla.gpu_func_attributes"(%arg0) <{fn = @__mlir_gpu_module::@__polygeist_empty_kernel}> : (!llvm.ptr) -> i32
// CHECK:         "enzymexla.gpu_func_attributes"(%arg0) <{fn = @free_kernel::@free_kernel}> : (!llvm.ptr) -> i32
// CHECK:       gpu.func @__polygeist_empty_kernel() kernel
// CHECK:       gpu.func @main_kernel(%arg0: !llvm.ptr) kernel attributes {polygeist.host_symbol = "_Z18__device_stub__fooPi"}

// -----

// Once every query has its outlined kernel, no query asks the stand-in, and it
// goes, with its module when that holds nothing else.
module attributes {gpu.container_module} {
  llvm.func @host(%attrs: !llvm.ptr, %out: !llvm.ptr) {
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %0 = "enzymexla.gpu_func_attributes"(%attrs) <{fn = @__mlir_gpu_module::@__polygeist_empty_kernel, host_symbol = "_Z18__device_stub__fooPi"}> : (!llvm.ptr) -> i32
    %1 = "enzymexla.gpu_error"() ({
      gpu.launch_func @main_kernel::@main_kernel blocks in (%c1, %c1, %c1) threads in (%c128, %c1, %c1) args(%out : !llvm.ptr)
      "enzymexla.polygeist_yield"() : () -> ()
    }) {passthrough = [["polygeist.host_symbol", "_Z18__device_stub__fooPi"], ["target-cpu", "sm_120"]]} : () -> index
    llvm.return
  }
  gpu.module @__mlir_gpu_module [#nvvm.target] {
    gpu.func @__polygeist_empty_kernel() kernel {
      gpu.return
    }
  }
  gpu.module @main_kernel [#nvvm.target] {
    gpu.func @main_kernel(%arg0: !llvm.ptr) kernel {
      gpu.return
    }
  }
}

// CHECK-LABEL: llvm.func @host(
// CHECK:         "enzymexla.gpu_func_attributes"(%arg0) <{fn = @main_kernel::@main_kernel}> : (!llvm.ptr) -> i32
// CHECK-NOT:   __mlir_gpu_module
// CHECK:       gpu.module @main_kernel

// -----

// A stand-in whose module also holds a captured kernel goes alone.
module attributes {gpu.container_module} {
  llvm.func @host(%attrs: !llvm.ptr, %out: !llvm.ptr) {
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %0 = "enzymexla.gpu_func_attributes"(%attrs) <{fn = @__mlir_gpu_module::@__polygeist_empty_kernel, host_symbol = "_Z18__device_stub__fooPi"}> : (!llvm.ptr) -> i32
    %1 = "enzymexla.gpu_error"() ({
      gpu.launch_func @main_kernel::@main_kernel blocks in (%c1, %c1, %c1) threads in (%c128, %c1, %c1) args(%out : !llvm.ptr)
      "enzymexla.polygeist_yield"() : () -> ()
    }) {passthrough = [["polygeist.host_symbol", "_Z18__device_stub__fooPi"], ["target-cpu", "sm_120"]]} : () -> index
    %2 = "enzymexla.gpu_error"() ({
      gpu.launch_func @__mlir_gpu_module::@captured blocks in (%c1, %c1, %c1) threads in (%c128, %c1, %c1) args(%out : !llvm.ptr)
      "enzymexla.polygeist_yield"() : () -> ()
    }) : () -> index
    llvm.return
  }
  gpu.module @__mlir_gpu_module [#nvvm.target] {
    gpu.func @__polygeist_empty_kernel() kernel {
      gpu.return
    }
    gpu.func @captured(%arg0: !llvm.ptr) kernel {
      gpu.return
    }
  }
  gpu.module @main_kernel [#nvvm.target] {
    gpu.func @main_kernel(%arg0: !llvm.ptr) kernel {
      gpu.return
    }
  }
}

// CHECK-LABEL: llvm.func @host(
// CHECK:         "enzymexla.gpu_func_attributes"(%arg0) <{fn = @main_kernel::@main_kernel}> : (!llvm.ptr) -> i32
// CHECK:       gpu.module @__mlir_gpu_module
// CHECK-NOT:     __polygeist_empty_kernel
// CHECK:         gpu.func @captured
