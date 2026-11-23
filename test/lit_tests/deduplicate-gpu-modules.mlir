// RUN: enzymexlamlir-opt %s --deduplicate-gpu-modules --split-input-file | FileCheck %s

// Equivalent modules with distinct kernel, shared-memory, private-helper, and
// dead host-stub symbol names collapse to the first module and rewrite host
// launch/query symbol references.
// CHECK-LABEL: module attributes {gpu.container_module}
module attributes {gpu.container_module} {
  // CHECK: gpu.module @kmod_a [#nvvm.target<O = 3, chip = "sm_80">] {
  // CHECK-NEXT: llvm.mlir.global internal @shared_mem_100() {addr_space = 3 : i32} : !llvm.array<8 x f32>
  // CHECK-NEXT: memref.global @shared_mem_memref_100 : memref<4xf32, 3> = uninitialized
  // CHECK-NEXT: func.func private @helper_a(%{{.+}}: f32) -> f32
  // CHECK: gpu.func @f1_kernel(%{{.+}}: !llvm.ptr) kernel
  // CHECK-NOT: polygeist.host_symbol
  gpu.module @kmod_a [#nvvm.target<O = 3, chip = "sm_80">] {
    llvm.mlir.global internal @shared_mem_100() {addr_space = 3 : i32} : !llvm.array<8 x f32>
    memref.global @shared_mem_memref_100 : memref<4xf32, 3> = uninitialized
    func.func private @helper_a(%x: f32) -> f32 {
      %y = arith.addf %x, %x : f32
      return %y : f32
    }
    gpu.func @f1_kernel(%out: !llvm.ptr) kernel attributes {"polygeist.host_symbol" = "dead_stub_a"} {
      %shm = llvm.mlir.addressof @shared_mem_100 : !llvm.ptr<3>
      %mshm = memref.get_global @shared_mem_memref_100 : memref<4xf32, 3>
      %c0 = arith.constant 0 : index
      %v = memref.load %mshm[%c0] : memref<4xf32, 3>
      %doubled = func.call @helper_a(%v) : (f32) -> f32
      llvm.store %doubled, %shm : f32, !llvm.ptr<3>
      %loaded = llvm.load %shm : !llvm.ptr<3> -> f32
      llvm.store %loaded, %out : f32, !llvm.ptr
      gpu.return
    }
  }

  // CHECK-NOT: gpu.module @kmod_b
  gpu.module @kmod_b [#nvvm.target<O = 3, chip = "sm_80">] {
    llvm.mlir.global internal @shared_mem_200() {addr_space = 3 : i32} : !llvm.array<8 x f32>
    memref.global @shared_mem_memref_200 : memref<4xf32, 3> = uninitialized
    func.func private @helper_b(%x: f32) -> f32 {
      %y = arith.addf %x, %x : f32
      return %y : f32
    }
    gpu.func @f2_kernel(%out: !llvm.ptr) kernel attributes {"polygeist.host_symbol" = "dead_stub_b"} {
      %shm = llvm.mlir.addressof @shared_mem_200 : !llvm.ptr<3>
      %mshm = memref.get_global @shared_mem_memref_200 : memref<4xf32, 3>
      %c0 = arith.constant 0 : index
      %v = memref.load %mshm[%c0] : memref<4xf32, 3>
      %doubled = func.call @helper_b(%v) : (f32) -> f32
      llvm.store %doubled, %shm : f32, !llvm.ptr<3>
      %loaded = llvm.load %shm : !llvm.ptr<3> -> f32
      llvm.store %loaded, %out : f32, !llvm.ptr
      gpu.return
    }
  }

  // CHECK-LABEL: func.func @host
  func.func @host(%ptr: !llvm.ptr) -> !llvm.ptr {
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i64 = arith.constant 0 : i64
    // CHECK: gpu.launch_func @kmod_a::@f1_kernel
    gpu.launch_func @kmod_a::@f1_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%ptr : !llvm.ptr)
    // CHECK: gpu.launch_func @kmod_a::@f1_kernel
    gpu.launch_func @kmod_b::@f2_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%ptr : !llvm.ptr)
    // CHECK: "enzymexla.gpu_occupancy"{{.+}}fn = @kmod_a::@f1_kernel
    %occ = "enzymexla.gpu_occupancy"(%c64_i32, %c0_i64, %c0_i32) <{fn = @kmod_b::@f2_kernel}> : (i32, i64, i32) -> i32
    // CHECK: "enzymexla.gpu_kernel_address"() <{fn = @kmod_a::@f1_kernel}>
    %addr = "enzymexla.gpu_kernel_address"() <{fn = @kmod_b::@f2_kernel}> : () -> !llvm.ptr
    return %addr : !llvm.ptr
  }
}

// -----

// Modules that differ in live host stubs, target attributes, device globals,
// or `polygeist_stubs` lowering state must remain distinct.
// CHECK-LABEL: module attributes {gpu.container_module}
module attributes {gpu.container_module} {
  llvm.mlir.global external @dev_a() {addr_space = 1 : i32} : i32
  llvm.mlir.global external @dev_b() {addr_space = 1 : i32} : i32
  llvm.func @live_stub_a() { llvm.return }
  llvm.func @live_stub_b() { llvm.return }

  // CHECK: gpu.module @live_a
  gpu.module @live_a [#nvvm.target<O = 3, chip = "sm_80">] {
    gpu.func @k() kernel attributes {"polygeist.host_symbol" = "live_stub_a"} {
      gpu.return
    }
  }
  // CHECK: gpu.module @live_b
  gpu.module @live_b [#nvvm.target<O = 3, chip = "sm_80">] {
    gpu.func @k() kernel attributes {"polygeist.host_symbol" = "live_stub_b"} {
      gpu.return
    }
  }
  // CHECK: gpu.module @fast_target
  gpu.module @fast_target [#nvvm.target<O = 3, chip = "sm_80", flags = {fast = true}>] {
    gpu.func @k() kernel {
      gpu.return
    }
  }
  // CHECK: gpu.module @dev_glob_a
  gpu.module @dev_glob_a [#nvvm.target<O = 3, chip = "sm_80">] {
    llvm.mlir.global external @dev_a() {addr_space = 1 : i32} : i32
    gpu.func @k() kernel {
      %p = llvm.mlir.addressof @dev_a : !llvm.ptr<1>
      gpu.return
    }
  }
  // CHECK: gpu.module @dev_glob_b
  gpu.module @dev_glob_b [#nvvm.target<O = 3, chip = "sm_80">] {
    llvm.mlir.global external @dev_b() {addr_space = 1 : i32} : i32
    gpu.func @k() kernel {
      %p = llvm.mlir.addressof @dev_b : !llvm.ptr<1>
      gpu.return
    }
  }
  // CHECK: gpu.module @already_stubbed_a
  gpu.module @already_stubbed_a [#nvvm.target<O = 3, chip = "sm_80">] attributes {polygeist_stubs} {
    gpu.func @k() kernel {
      gpu.return
    }
  }
  // CHECK: gpu.module @already_stubbed_b
  gpu.module @already_stubbed_b [#nvvm.target<O = 3, chip = "sm_80">] attributes {polygeist_stubs} {
    gpu.func @k() kernel {
      gpu.return
    }
  }
  // CHECK: gpu.module @non_shared_internal_a
  gpu.module @non_shared_internal_a [#nvvm.target<O = 3, chip = "sm_80">] {
    llvm.mlir.global internal @lut_a(dense<[1, 2]> : tensor<2xi32>) {addr_space = 0 : i32} : !llvm.array<2 x i32>
    gpu.func @k() kernel {
      %p = llvm.mlir.addressof @lut_a : !llvm.ptr
      gpu.return
    }
  }
  // CHECK: gpu.module @non_shared_internal_b
  gpu.module @non_shared_internal_b [#nvvm.target<O = 3, chip = "sm_80">] {
    llvm.mlir.global internal @lut_b(dense<[1, 2]> : tensor<2xi32>) {addr_space = 0 : i32} : !llvm.array<2 x i32>
    gpu.func @k() kernel {
      %p = llvm.mlir.addressof @lut_b : !llvm.ptr
      gpu.return
    }
  }
}

// -----

// Multi-block CFG kernels deduplicate when control flow and successor targets
// match, stay distinct when successor targets are swapped, and handle blocks
// whose lexical order differs from dominance order without miscompiling.
// CHECK-LABEL: module attributes {gpu.container_module}
module attributes {gpu.container_module} {
  // CHECK: gpu.module @cfg_a
  gpu.module @cfg_a [#nvvm.target<O = 3, chip = "sm_80">] {
    gpu.func @k(%cond: i1, %out: !llvm.ptr) kernel {
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      cf.cond_br %cond, ^bb1, ^bb2
    ^bb1:
      llvm.store %c1, %out : i32, !llvm.ptr
      gpu.return
    ^bb2:
      llvm.store %c2, %out : i32, !llvm.ptr
      gpu.return
    }
  }
  // CHECK-NOT: gpu.module @cfg_a_dup
  gpu.module @cfg_a_dup [#nvvm.target<O = 3, chip = "sm_80">] {
    gpu.func @k2(%cond: i1, %out: !llvm.ptr) kernel {
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      cf.cond_br %cond, ^bb1, ^bb2
    ^bb1:
      llvm.store %c1, %out : i32, !llvm.ptr
      gpu.return
    ^bb2:
      llvm.store %c2, %out : i32, !llvm.ptr
      gpu.return
    }
  }
  // CHECK: gpu.module @cfg_swapped_succs
  gpu.module @cfg_swapped_succs [#nvvm.target<O = 3, chip = "sm_80">] {
    gpu.func @k3(%cond: i1, %out: !llvm.ptr) kernel {
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      cf.cond_br %cond, ^bb2, ^bb1
    ^bb1:
      llvm.store %c1, %out : i32, !llvm.ptr
      gpu.return
    ^bb2:
      llvm.store %c2, %out : i32, !llvm.ptr
      gpu.return
    }
  }
  // OperationEquivalence::isRegionEquivalentTo populates its value map in
  // lexical block order, so non-topological CFG blocks conservatively stay
  // distinct while hashing handles forward-defined SSA operands cleanly.
  // CHECK: gpu.module @cfg_non_topo_a
  gpu.module @cfg_non_topo_a [#nvvm.target<O = 3, chip = "sm_80">] {
    gpu.func @k4(%cond: i1, %out: !llvm.ptr) kernel {
      cf.br ^bb2
    ^bb1:
      llvm.store %dom_val, %out : i32, !llvm.ptr
      gpu.return
    ^bb2:
      %dom_val = arith.constant 42 : i32
      cf.cond_br %cond, ^bb1, ^bb3
    ^bb3:
      gpu.return
    }
  }
  // CHECK: gpu.module @cfg_non_topo_b
  gpu.module @cfg_non_topo_b [#nvvm.target<O = 3, chip = "sm_80">] {
    gpu.func @k5(%cond: i1, %out: !llvm.ptr) kernel {
      cf.br ^bb2
    ^bb1:
      llvm.store %dom_val, %out : i32, !llvm.ptr
      gpu.return
    ^bb2:
      %dom_val = arith.constant 42 : i32
      cf.cond_br %cond, ^bb1, ^bb3
    ^bb3:
      gpu.return
    }
  }
}
