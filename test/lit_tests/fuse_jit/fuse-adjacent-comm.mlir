// RUN: enzymexlamlir-opt --fuse-jit="dag=false adjacent=true" %s | FileCheck %s

module {
  llvm.func @send(!llvm.ptr, !llvm.ptr)
  llvm.func @recv(!llvm.ptr, !llvm.ptr)
  llvm.func @kernel(!llvm.ptr)

  llvm.func @jitwrap_send(%arg0: !llvm.ptr, %arg1: !llvm.ptr) attributes {enzymexla.comm} {
    llvm.call @send(%arg0, %arg1) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @jitwrap_recv(%arg0: !llvm.ptr, %arg1: !llvm.ptr) attributes {enzymexla.comm} {
    llvm.call @recv(%arg0, %arg1) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }
  // CHECK-LABEL: llvm.func @fused__jitwrap_send_jitwrap_recv(
  // CHECK-SAME: %[[SEND_BUF:[^:]+]]: !llvm.ptr, %[[SEND_PEER:[^:]+]]: !llvm.ptr, %[[RECV_BUF:[^:]+]]: !llvm.ptr, %[[RECV_PEER:[^:]+]]: !llvm.ptr) attributes {enzymexla.comm} {
  // CHECK-NEXT: llvm.call @send(%[[SEND_BUF]], %[[SEND_PEER]])
  // CHECK-NEXT: llvm.call @recv(%[[RECV_BUF]], %[[RECV_PEER]])
  // CHECK-NEXT: llvm.return

  // Not a comm wrapper.
  llvm.func @jitwrap_kernel(%arg0: !llvm.ptr) {
    llvm.call @kernel(%arg0) : (!llvm.ptr) -> ()
    llvm.return
  }

  // Inputs computed between the calls are hoisted above the fused call.
  func.func @hoists_inputs(%buf: tensor<4xf32>, %peer: tensor<i32>) -> tensor<4xf32> {
    enzymexla.jit_call @jitwrap_send (%buf, %peer) : (tensor<4xf32>, tensor<i32>) -> ()
    %next = stablehlo.add %peer, %peer : tensor<i32>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<4xf32>
    %r = enzymexla.jit_call @jitwrap_recv (%zero, %next) <{
      output_operand_aliases = [
        #stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>
      ]}> : (tensor<4xf32>, tensor<i32>) -> tensor<4xf32>
    return %r : tensor<4xf32>
  }
  // CHECK-LABEL: func.func @hoists_inputs
  // CHECK-SAME: (%[[BUF:[^:]+]]: tensor<4xf32>, %[[PEER:[^:]+]]: tensor<i32>)
  // CHECK-NEXT: %[[NEXT:.*]] = stablehlo.add %[[PEER]], %[[PEER]]
  // CHECK-NEXT: %[[ZERO:.*]] = stablehlo.constant
  // CHECK-NEXT: %[[R:.*]] = enzymexla.jit_call @fused__jitwrap_send_jitwrap_recv (%[[BUF]], %[[PEER]], %[[ZERO]], %[[NEXT]])
  // CHECK-SAME: operand_index = 2
  // CHECK-NEXT: return %[[R]]

  // Recvs sharing one constant (e.g. after CSE) each write it in place, so each
  // gets its own slot rather than sharing one.
  func.func @shared_constant(%buf: tensor<4xf32>, %peer: tensor<i32>) -> (tensor<4xf32>, tensor<4xf32>) {
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<4xf32>
    enzymexla.jit_call @jitwrap_send (%buf, %peer) : (tensor<4xf32>, tensor<i32>) -> ()
    %r0 = enzymexla.jit_call @jitwrap_recv (%zero, %peer) <{
      output_operand_aliases = [
        #stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>
      ]}> : (tensor<4xf32>, tensor<i32>) -> tensor<4xf32>
    %r1 = enzymexla.jit_call @jitwrap_recv (%zero, %peer) <{
      output_operand_aliases = [
        #stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>
      ]}> : (tensor<4xf32>, tensor<i32>) -> tensor<4xf32>
    return %r0, %r1 : tensor<4xf32>, tensor<4xf32>
  }
  // CHECK-LABEL: func.func @shared_constant
  // CHECK-SAME: (%[[BUF:[^:]+]]: tensor<4xf32>, %[[PEER:[^:]+]]: tensor<i32>)
  // CHECK-NEXT: %[[ZERO:.*]] = stablehlo.constant
  // CHECK-NEXT: %[[R:.*]]:2 = enzymexla.jit_call @fused__jitwrap_send_jitwrap_recv_jitwrap_recv (%[[BUF]], %[[PEER]], %[[ZERO]], %[[PEER]], %[[ZERO]], %[[PEER]])
  // CHECK-SAME: output_tuple_indices = [0], operand_index = 2
  // CHECK-SAME: output_tuple_indices = [1], operand_index = 4
  // CHECK-NEXT: return %[[R]]#0, %[[R]]#1

  // A non-comm jit call with effects (jitwrap_kernel) ends the fusion
  func.func @effect_splits(%buf: tensor<4xf32>, %peer: tensor<i32>) -> tensor<4xf32> {
    enzymexla.jit_call @jitwrap_send (%buf, %peer) : (tensor<4xf32>, tensor<i32>) -> ()
    enzymexla.jit_call @jitwrap_kernel (%buf) : (tensor<4xf32>) -> ()
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<4xf32>
    %r = enzymexla.jit_call @jitwrap_recv (%zero, %peer) <{
      output_operand_aliases = [
        #stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>
      ]}> : (tensor<4xf32>, tensor<i32>) -> tensor<4xf32>
    return %r : tensor<4xf32>
  }
  // CHECK-LABEL: func.func @effect_splits
  // CHECK-NEXT: enzymexla.jit_call @jitwrap_send
  // CHECK-NEXT: enzymexla.jit_call @jitwrap_kernel
  // CHECK-NEXT: stablehlo.constant
  // CHECK-NEXT: enzymexla.jit_call @jitwrap_recv

  // Non-comm calls (jitwrap_kernel) are not fused.
  func.func @adjacent_non_comm(%buf: tensor<4xf32>) {
    enzymexla.jit_call @jitwrap_kernel (%buf) : (tensor<4xf32>) -> ()
    enzymexla.jit_call @jitwrap_kernel (%buf) : (tensor<4xf32>) -> ()
    return
  }
  // CHECK-LABEL: func.func @adjacent_non_comm
  // CHECK-NEXT: enzymexla.jit_call @jitwrap_kernel
  // CHECK-NEXT: enzymexla.jit_call @jitwrap_kernel

  // The send reads the received data through a non-member op, which cannot be
  // hoisted above the recv.
  func.func @member_dependency(%peer: tensor<i32>) {
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<4xf32>
    %r = enzymexla.jit_call @jitwrap_recv (%zero, %peer) <{
      output_operand_aliases = [
        #stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>
      ]}> : (tensor<4xf32>, tensor<i32>) -> tensor<4xf32>
    %doubled = stablehlo.add %r, %r : tensor<4xf32>
    enzymexla.jit_call @jitwrap_send (%doubled, %peer) : (tensor<4xf32>, tensor<i32>) -> ()
    return
  }
  // CHECK-LABEL: func.func @member_dependency
  // CHECK-NEXT: stablehlo.constant
  // CHECK-NEXT: enzymexla.jit_call @jitwrap_recv
  // CHECK-NEXT: stablehlo.add
  // CHECK-NEXT: enzymexla.jit_call @jitwrap_send

  // A result used between the calls reads the fused call's result instead.
  func.func @escaping_result(%buf: tensor<4xf32>, %peer: tensor<i32>) -> tensor<4xf32> {
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<4xf32>
    %r = enzymexla.jit_call @jitwrap_recv (%zero, %peer) <{
      output_operand_aliases = [
        #stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>
      ]}> : (tensor<4xf32>, tensor<i32>) -> tensor<4xf32>
    %abs = stablehlo.abs %r : tensor<4xf32>
    enzymexla.jit_call @jitwrap_send (%buf, %peer) : (tensor<4xf32>, tensor<i32>) -> ()
    return %abs : tensor<4xf32>
  }
  // CHECK-LABEL: func.func @escaping_result
  // CHECK-SAME: (%[[BUF:[^:]+]]: tensor<4xf32>, %[[PEER:[^:]+]]: tensor<i32>)
  // CHECK-NEXT: %[[ZERO:.*]] = stablehlo.constant
  // CHECK-NEXT: %[[R:.*]] = enzymexla.jit_call @fused__jitwrap_recv_jitwrap_send (%[[ZERO]], %[[PEER]], %[[BUF]], %[[PEER]])
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[R]]
  // CHECK-NEXT: return %[[ABS]]

  // Runs inside a loop body are fused; the loop itself ends the outer run.
  func.func @in_while(%buf: tensor<4xf32>, %peer: tensor<i32>, %n: tensor<i32>) {
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<4xf32>
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    enzymexla.jit_call @jitwrap_send (%buf, %peer) : (tensor<4xf32>, tensor<i32>) -> ()
    %w = stablehlo.while(%iter = %c0) : tensor<i32>
     cond {
      %p = stablehlo.compare LT, %iter, %n : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %p : tensor<i1>
     } do {
      enzymexla.jit_call @jitwrap_send (%buf, %iter) : (tensor<4xf32>, tensor<i32>) -> ()
      %r = enzymexla.jit_call @jitwrap_recv (%zero, %iter) <{
        output_operand_aliases = [
          #stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>
        ]}> : (tensor<4xf32>, tensor<i32>) -> tensor<4xf32>
      %one = stablehlo.constant dense<1> : tensor<i32>
      %next = stablehlo.add %iter, %one : tensor<i32>
      stablehlo.return %next : tensor<i32>
     }
    return
  }
  // CHECK-LABEL: func.func @in_while
  // CHECK: enzymexla.jit_call @jitwrap_send
  // CHECK-NEXT: stablehlo.while
  // CHECK: do {
  // CHECK-NEXT: enzymexla.jit_call @fused__jitwrap_send_jitwrap_recv_1
  // CHECK-NOT: enzymexla.jit_call
  // CHECK: stablehlo.return
}
