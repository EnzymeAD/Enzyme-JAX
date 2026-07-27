// RUN: enzymexlamlir-opt %s --convert-polygeist-to-llvm="backend=xla-gpu" | FileCheck %s

module {
  func.func private @my_xla_fn(!llvm.ptr, !llvm.ptr) -> ()
  
  func.func @test_xla_wrapper(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    // CHECK: llvm.func @test_xla_wrapper
    // CHECK: %[[ALLOCA:.+]] = llvm.alloca
    // CHECK: llvm.br ^bb1
    // CHECK: ^bb2:
    // CHECK: llvm.call @reactantXLAExec(%{{.+}}, %{{.+}}, %{{.+}}, %[[ALLOCA]])
    // CHECK-NOT: llvm.alloca
    
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    
    scf.for %i = %c0 to %c10 step %c1 {
      "enzymexla.xla_wrapper"(%arg0, %arg1) {fn = @my_xla_fn} : (!llvm.ptr, !llvm.ptr) -> ()
    }
    return
  }
}
