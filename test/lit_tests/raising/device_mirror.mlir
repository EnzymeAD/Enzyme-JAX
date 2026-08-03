// RUN: enzymexlamlir-opt --libdevice-funcs-raise %s | FileCheck %s
// RUN: enzymexlamlir-opt --libdevice-funcs-raise --convert-polygeist-to-llvm='backend=cpu' %s | FileCheck --check-prefix=LOWER %s

llvm.func external @_Z22__enzyme_device_mirrorIiPiET_S0_T0_(i32, !llvm.ptr)
    -> i32

llvm.func @raise_device_mirror(%host: i32, %device: !llvm.ptr) -> i32 {
  // CHECK-LABEL: llvm.func @raise_device_mirror
  // CHECK-NOT:     llvm.call
  // CHECK:         %[[TAG:.*]] = enzymexla.device_mirror %arg0, %arg1 : (i32, !llvm.ptr) -> i32
  // CHECK:         llvm.return %[[TAG]] : i32
  %tag = llvm.call @_Z22__enzyme_device_mirrorIiPiET_S0_T0_(%host, %device) :
      (i32, !llvm.ptr) -> i32
  llvm.return %tag : i32
}

// LOWER-LABEL: llvm.func @raise_device_mirror
// LOWER-NOT:     enzymexla.device_mirror
// LOWER:         llvm.return %arg0 : i32
