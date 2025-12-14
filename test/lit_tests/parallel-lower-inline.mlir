// RUN: enzymexlamlir-opt %s --parallel-lower | FileCheck %s

module {
  llvm.func internal @bar(%arg0: i32, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr {llvm.noundef}, %arg5: !llvm.ptr {llvm.noundef}, %arg6: i32 {llvm.noundef}, %arg7: i64 {llvm.noundef}) attributes { sym_visibility = "private" } {
    %0 = "enzymexla.stream2token"(%arg3) : (!llvm.ptr) -> !gpu.async.token
    %1 = arith.index_cast %arg0 : i32 to index
    %2 = gpu.launch async [%0] blocks(%arg8, %arg9, %arg10) in (%arg14 = %1, %arg15 = %1, %arg16 = %1) threads(%arg11, %arg12, %arg13) in (%arg17 = %1, %arg18 = %1, %arg19 = %1) dynamic_shared_memory_size %arg0 {
      llvm.call @foo(%arg4, %arg5, %arg6, %arg1, %arg7) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr, i64) -> ()
      gpu.terminator
    }
    llvm.return
  }
  llvm.func private @foo(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readnone}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}, %arg2: i32 {llvm.noundef}, %arg3: !llvm.ptr {llvm.align = 1 : i64, llvm.byval = !llvm.struct<(i8)>, llvm.nocapture, llvm.noundef, llvm.readnone}, %arg4: i64 {llvm.noundef}) attributes { sym_visibility = "private" } {
    %rld = llvm.load %arg3 : !llvm.ptr -> !llvm.struct<(i8)>
    llvm.store %rld, %arg1 : !llvm.struct<(i8)>, !llvm.ptr
    llvm.return
  }
}

// CHECK:   llvm.func internal @bar(%arg0: i32, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr {llvm.noundef}, %arg5: !llvm.ptr {llvm.noundef}, %arg6: i32 {llvm.noundef}, %arg7: i64 {llvm.noundef}) attributes {sym_visibility = "private"} {
// CHECK:     %[[ld:.*]] = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(i8)>
// CHECK:       scf.parallel 
// CHECK:         scf.parallel 
// CHECK:           memref.alloca_scope  {
// CHECK:             scf.execute_region {
// CHECK-DAG:               %[[a1:.*]] = llvm.alloca %0 x !llvm.struct<(i8)> {alignment = 1 : i64} : (i64) -> !llvm.ptr
// CHECK-DAG:               %[[a2:.*]] = llvm.alloca %0 x !llvm.struct<(i8)> {alignment = 1 : i64} : (i64) -> !llvm.ptr
// CHECK:               llvm.store %[[ld]], %[[a2]] : !llvm.struct<(i8)>, !llvm.ptr
// CHECK:               memref.alloca_scope  {
// CHECK:                 scf.execute_region {
// CHECK:                   "llvm.intr.memcpy"(%[[a1]], %[[a2]], %0) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK:                   %[[rld:.*]] = llvm.load %4 : !llvm.ptr -> !llvm.struct<(i8)>
// CHECK:                   llvm.store %6, %arg5 : !llvm.struct<(i8)>, !llvm.ptr
// CHECK:                   scf.yield
// CHECK:                 }
// CHECK:               }
// CHECK:               scf.yield
// CHECK:             }
// CHECK:           }
// CHECK:           scf.reduce
// CHECK:         }
// CHECK:         scf.reduce
// CHECK:       }
// CHECK:       async.yield
// CHECK:     }
// CHECK:     llvm.return
// CHECK:   }
