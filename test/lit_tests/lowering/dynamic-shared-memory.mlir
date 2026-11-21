// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-parallel-to-gpu2{backend=cuda})" --verify-diagnostics | FileCheck %s

// Block-scope arrays over 48 KiB move to one dynamic shared memory region;
// every launch passes its size and opts the kernel in right before launching.

module attributes {gpu.container_module} {
  gpu.module @big {
    gpu.func @big(%out: memref<?xf32, 1>, %v: f32) kernel {
      %c0 = arith.constant 0 : index
      %a = memref.alloca() : memref<10241xf32, 5>
      %b = memref.alloca() alignment = 32 : memref<2048xf64, 5>
      memref.store %v, %a[%c0] : memref<10241xf32, 5>
      %w = arith.extf %v : f32 to f64
      memref.store %w, %b[%c0] : memref<2048xf64, 5>
      %x = memref.load %a[%c0] : memref<10241xf32, 5>
      %y = memref.load %b[%c0] : memref<2048xf64, 5>
      %z = arith.truncf %y : f64 to f32
      %s = arith.addf %x, %z : f32
      memref.store %s, %out[%c0] : memref<?xf32, 1>
      gpu.return
    }
  }

  // 4 KiB: stays static.
  gpu.module @small {
    gpu.func @small(%out: memref<?xf32, 1>, %v: f32) kernel {
      %c0 = arith.constant 0 : index
      %a = memref.alloca() : memref<1024xf32, 5>
      memref.store %v, %a[%c0] : memref<1024xf32, 5>
      %x = memref.load %a[%c0] : memref<1024xf32, 5>
      memref.store %x, %out[%c0] : memref<?xf32, 1>
      gpu.return
    }
  }

  // Launched with a size of its own: left alone, with a warning.
  gpu.module @own {
    // expected-warning @+1 {{kernel is launched with dynamic shared memory of its own; its 65536 bytes of block-scope arrays stay static, over the 49152-byte limit}}
    gpu.func @own(%out: memref<?xf32, 1>, %v: f32) kernel {
      %c0 = arith.constant 0 : index
      %a = memref.alloca() : memref<16384xf32, 5>
      memref.store %v, %a[%c0] : memref<16384xf32, 5>
      %x = memref.load %a[%c0] : memref<16384xf32, 5>
      memref.store %x, %out[%c0] : memref<?xf32, 1>
      gpu.return
    }
  }

  // A vector element type needs its own alignment (32) with no attribute.
  gpu.module @vec {
    gpu.func @vec(%out: memref<?xf32, 1>, %v: f32, %w: vector<8xf32>) kernel {
      %c0 = arith.constant 0 : index
      %a = memref.alloca() : memref<4097xf32, 5>
      %b = memref.alloca() : memref<1024xvector<8xf32>, 5>
      memref.store %v, %a[%c0] : memref<4097xf32, 5>
      memref.store %w, %b[%c0] : memref<1024xvector<8xf32>, 5>
      %x = memref.load %a[%c0] : memref<4097xf32, 5>
      memref.store %x, %out[%c0] : memref<?xf32, 1>
      gpu.return
    }
  }

  func.func @host(%out: memref<?xf32, 1>, %v: f32, %n: i32, %w: vector<8xf32>) {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    gpu.launch_func @big::@big blocks in (%c1, %c1, %c1) threads in (%c32, %c1, %c1) args(%out : memref<?xf32, 1>, %v : f32)
    gpu.launch_func @small::@small blocks in (%c1, %c1, %c1) threads in (%c32, %c1, %c1) args(%out : memref<?xf32, 1>, %v : f32)
    gpu.launch_func @big::@big blocks in (%c1, %c1, %c1) threads in (%c32, %c1, %c1) args(%out : memref<?xf32, 1>, %v : f32)
    gpu.launch_func @own::@own blocks in (%c1, %c1, %c1) threads in (%c32, %c1, %c1) dynamic_shared_memory_size %n args(%out : memref<?xf32, 1>, %v : f32)
    gpu.launch_func @vec::@vec blocks in (%c1, %c1, %c1) threads in (%c32, %c1, %c1) args(%out : memref<?xf32, 1>, %v : f32, %w : vector<8xf32>)
    return
  }
}

// The second array starts at 40964 rounded up to its alignment of 32.
// CHECK-LABEL: gpu.module @big
// CHECK: llvm.mlir.global external @big_dynamic_shared_memory() {addr_space = 3 : i32, alignment = 32 : i64} : !llvm.array<0 x i8>
// CHECK: gpu.func @big
// CHECK-NOT: memref.alloca
// CHECK: %[[BASE:.+]] = llvm.mlir.addressof @big_dynamic_shared_memory : !llvm.ptr<3>
// CHECK: "enzymexla.pointer2memref"(%[[BASE]]) : (!llvm.ptr<3>) -> memref<10241xf32, 3>
// CHECK: %[[SECOND:.+]] = llvm.getelementptr %[[BASE]][40992] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
// CHECK: "enzymexla.pointer2memref"(%[[SECOND]]) : (!llvm.ptr<3>) -> memref<2048xf64, 3>

// CHECK-LABEL: gpu.module @small
// CHECK-NOT: dynamic_shared_memory
// CHECK: memref.global @shared_mem_{{[0-9]+}} : memref<1024xf32, 3>

// CHECK-LABEL: gpu.module @own
// CHECK-NOT: dynamic_shared_memory
// CHECK: memref.global @shared_mem_{{[0-9]+}} : memref<16384xf32, 3>

// 16388 rounded up to 32, the ABI alignment of vector<8xf32>.
// CHECK-LABEL: gpu.module @vec
// CHECK: llvm.mlir.global external @vec_dynamic_shared_memory() {addr_space = 3 : i32, alignment = 32 : i64} : !llvm.array<0 x i8>
// CHECK: %[[VBASE:.+]] = llvm.mlir.addressof @vec_dynamic_shared_memory : !llvm.ptr<3>
// CHECK: llvm.getelementptr %[[VBASE]][16416] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8

// 57376 = 40992 + 2048 * 8.
// CHECK-LABEL: func.func @host
// CHECK-DAG: %[[S1:.+]] = arith.constant 57376 : i32
// CHECK-DAG: %[[S4:.+]] = arith.constant 49184 : i32
// CHECK: %[[K1:.+]] = "enzymexla.gpu_kernel_address"() <{fn = @big::@big}> : () -> !llvm.ptr
// CHECK: llvm.call @cudaFuncSetAttribute(%[[K1]], %{{.+}}, %[[S1]])
// CHECK-NEXT: gpu.launch_func @big::@big {{.*}} dynamic_shared_memory_size %[[S1]]
// CHECK-NOT: cudaFuncSetAttribute
// CHECK: gpu.launch_func @small::@small blocks in ({{.*}}) threads in ({{.*}}) args
// CHECK: %[[K2:.+]] = "enzymexla.gpu_kernel_address"() <{fn = @big::@big}> : () -> !llvm.ptr
// CHECK: llvm.call @cudaFuncSetAttribute(%[[K2]],
// CHECK-NEXT: gpu.launch_func @big::@big {{.*}} dynamic_shared_memory_size
// CHECK-NOT: cudaFuncSetAttribute
// CHECK: gpu.launch_func @own::@own {{.*}} dynamic_shared_memory_size %arg2
// CHECK: llvm.call @cudaFuncSetAttribute(%{{.+}}, %{{.+}}, %[[S4]])
// CHECK-NEXT: gpu.launch_func @vec::@vec {{.*}} dynamic_shared_memory_size %[[S4]]
