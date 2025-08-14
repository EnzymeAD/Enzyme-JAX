// RUN: enzymexlamlir-opt --allow-unregistered-dialect --libdevice-funcs-raise -split-input-file %s | FileCheck %s

func.func @simple() {
    %c1_i32 = arith.constant 1 : i32
    %c7_i64 = arith.constant 7 : i64
    %1 = llvm.alloca %c1_i32 x !llvm.array<7 x i8> {alignment = 1 : i64} : (i32) -> !llvm.ptr
    %2 = llvm.alloca %c1_i32 x !llvm.array<7 x i8> {alignment = 1 : i64} : (i32) -> !llvm.ptr
    "llvm.intr.memcpy"(%2, %1, %c7_i64) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    "Just.keepalive"(%c1_i32) : (i32) -> ()
    return
}
// CHECK-LABEL:   func.func @simple() {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i32
// CHECK:           "Just.keepalive"(%[[VAL_0]]) : (i32) -> ()
// CHECK:           return
// CHECK:         }

func.func @simple_with_lifetime() {
    %c1_i32 = arith.constant 1 : i32
    %c7_i64 = arith.constant 7 : i64
    %1 = llvm.alloca %c1_i32 x !llvm.array<7 x i8> {alignment = 1 : i64} : (i32) -> !llvm.ptr
    %2 = llvm.alloca %c1_i32 x !llvm.array<7 x i8> {alignment = 1 : i64} : (i32) -> !llvm.ptr
    llvm.intr.lifetime.start %2 : !llvm.ptr
    "llvm.intr.memcpy"(%2, %1, %c7_i64) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.intr.lifetime.end %2 : !llvm.ptr
    "Just.keepalive"(%c1_i32) : (i32) -> ()
    return
}
// CHECK-LABEL:   func.func @simple_with_lifetime() {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i32
// CHECK:           "Just.keepalive"(%[[VAL_0]]) : (i32) -> ()
// CHECK:           return
// CHECK:         }

func.func @no_canonicalize(%1: !llvm.ptr) {
    %c1_i32 = arith.constant 1 : i32
    %c7_i64 = arith.constant 7 : i64
    %2 = llvm.alloca %c1_i32 x !llvm.array<7 x i8> {alignment = 1 : i64} : (i32) -> !llvm.ptr
    llvm.intr.lifetime.start %2 : !llvm.ptr
    "llvm.intr.memcpy"(%2, %1, %c7_i64) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.intr.lifetime.end %2 : !llvm.ptr
    "Just.keepalive"(%c1_i32) : (i32) -> ()
    return
}
// CHECK-LABEL:   func.func @no_canonicalize(
// CHECK-SAME:                               %[[VAL_0:.*]]: !llvm.ptr) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 7 : i64
// CHECK:           %[[VAL_3:.*]] = llvm.alloca %[[VAL_1]] x !llvm.array<7 x i8> {alignment = 1 : i64} : (i32) -> !llvm.ptr
// CHECK:           llvm.intr.lifetime.start %[[VAL_3]] : !llvm.ptr
// CHECK:           "llvm.intr.memcpy"(%[[VAL_3]], %[[VAL_0]], %[[VAL_2]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK:           llvm.intr.lifetime.end %[[VAL_3]] : !llvm.ptr
// CHECK:           "Just.keepalive"(%[[VAL_1]]) : (i32) -> ()
// CHECK:           return
// CHECK:         }

func.func @allocaLifetime_oceans() {
    %c1_i32 = arith.constant 1 : i32
    %c7_i64 = arith.constant 7 : i64
    %1 = llvm.alloca %c1_i32 x !llvm.array<7 x i8> {alignment = 1 : i64} : (i32) -> !llvm.ptr
    %2 = llvm.alloca %c1_i32 x !llvm.array<7 x i8> {alignment = 1 : i64} : (i32) -> !llvm.ptr
    %3 = llvm.alloca %c1_i32 x !llvm.array<7 x i8> {alignment = 1 : i64} : (i32) -> !llvm.ptr
    %4 = llvm.alloca %c1_i32 x !llvm.array<7 x i8> {alignment = 1 : i64} : (i32) -> !llvm.ptr
    llvm.intr.lifetime.start %2 : !llvm.ptr
    "llvm.intr.memcpy"(%2, %4, %c7_i64) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    "llvm.intr.memcpy"(%3, %2, %c7_i64) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.intr.lifetime.end %2 : !llvm.ptr

    llvm.intr.lifetime.start %2 : !llvm.ptr
    "llvm.intr.memcpy"(%2, %1, %c7_i64) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    "llvm.intr.memcpy"(%3, %2, %c7_i64) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.intr.lifetime.end %2 : !llvm.ptr

    "Just.keepalive"(%c1_i32) : (i32) -> ()
    return
}
// CHECK-LABEL:   func.func @allocaLifetime_oceans() {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i32
// CHECK:           "Just.keepalive"(%[[VAL_0]]) : (i32) -> ()
// CHECK:           return
// CHECK:         }

func.func @allocaLifetime_memset() {
    %c1_i32 = arith.constant 1 : i32
    %c7_i64 = arith.constant 7 : i64
    %c0_i8 = arith.constant 0 : i8
    %1 = llvm.alloca %c1_i32 x !llvm.array<7 x i8> {alignment = 1 : i64} : (i32) -> !llvm.ptr
    %2 = llvm.alloca %c1_i32 x !llvm.array<7 x i8> {alignment = 1 : i64} : (i32) -> !llvm.ptr
    %3 = llvm.alloca %c1_i32 x !llvm.array<7 x i8> {alignment = 1 : i64} : (i32) -> !llvm.ptr
    %4 = llvm.alloca %c1_i32 x !llvm.array<7 x i8> {alignment = 1 : i64} : (i32) -> !llvm.ptr

    "llvm.intr.memset"(%4, %c0_i8, %c7_i64) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()

    llvm.intr.lifetime.start %2 : !llvm.ptr
    "llvm.intr.memcpy"(%2, %4, %c7_i64) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    "llvm.intr.memcpy"(%3, %2, %c7_i64) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.intr.lifetime.end %2 : !llvm.ptr

    llvm.intr.lifetime.start %2 : !llvm.ptr
    "llvm.intr.memcpy"(%2, %1, %c7_i64) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    "llvm.intr.memcpy"(%3, %2, %c7_i64) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.intr.lifetime.end %2 : !llvm.ptr

    "Just.keepalive"(%c1_i32) : (i32) -> ()
    return
}
// CHECK-LABEL:   func.func @allocaLifetime_memset() {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 7 : i64
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i8
// CHECK:           %[[VAL_3:.*]] = llvm.alloca %[[VAL_0]] x !llvm.array<7 x i8> {alignment = 1 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_4:.*]] = llvm.alloca %[[VAL_0]] x !llvm.array<7 x i8> {alignment = 1 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_5:.*]] = llvm.alloca %[[VAL_0]] x !llvm.array<7 x i8> {alignment = 1 : i64} : (i32) -> !llvm.ptr
// CHECK:           "llvm.intr.memset"(%[[VAL_5]], %[[VAL_2]], %[[VAL_1]]) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
// CHECK:           llvm.intr.lifetime.start %[[VAL_3]] : !llvm.ptr
// CHECK:           "llvm.intr.memcpy"(%[[VAL_3]], %[[VAL_5]], %[[VAL_1]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK:           "llvm.intr.memcpy"(%[[VAL_4]], %[[VAL_3]], %[[VAL_1]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK:           llvm.intr.lifetime.end %[[VAL_3]] : !llvm.ptr
// CHECK:           llvm.intr.lifetime.start %[[VAL_3]] : !llvm.ptr
// CHECK:           "llvm.intr.memcpy"(%[[VAL_4]], %[[VAL_3]], %[[VAL_1]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK:           llvm.intr.lifetime.end %[[VAL_3]] : !llvm.ptr
// CHECK:           "Just.keepalive"(%[[VAL_0]]) : (i32) -> ()
// CHECK:           return
// CHECK:         }

func.func @no_alloca() {
  %c1_i32 = arith.constant 1 : i32
  %c42_i8 = arith.constant 42 : i8  // Value to set (42)
  %c10_i64 = arith.constant 10 : i64  // Size of the memory to set
  
  %ptr = llvm.alloca %c1_i32 x !llvm.array<10 x i8> {alignment = 1 : i64} : (i32) -> !llvm.ptr
  
  llvm.intr.lifetime.start %ptr : !llvm.ptr
  
  "llvm.intr.memset"(%ptr, %c42_i8, %c10_i64) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
  
  llvm.intr.lifetime.end %ptr : !llvm.ptr
  
  return
}
// CHECK-LABEL:   func.func @no_alloca() {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 42 : i8
// CHECK:           %[[VAL_2:.*]] = arith.constant 10 : i64
// CHECK:           %[[VAL_3:.*]] = llvm.alloca %[[VAL_0]] x !llvm.array<10 x i8> {alignment = 1 : i64} : (i32) -> !llvm.ptr
// CHECK:           llvm.intr.lifetime.start %[[VAL_3]] : !llvm.ptr
// CHECK:           "llvm.intr.memset"(%[[VAL_3]], %[[VAL_1]], %[[VAL_2]]) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
// CHECK:           llvm.intr.lifetime.end %[[VAL_3]] : !llvm.ptr
// CHECK:           return
// CHECK:         }
