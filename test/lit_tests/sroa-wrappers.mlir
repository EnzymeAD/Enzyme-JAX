// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(inline{default-pipeline=canonicalize max-iterations=4},sroa-wrappers{instsimplify=true instcombine=false},canonicalize)" | FileCheck %s
#tbaa_root = #llvm.tbaa_root<id = "custom_tbaa">
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "custom_tbaa_addrspace(1)", members = {<#tbaa_root, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc, access_type = #tbaa_type_desc, offset = 0>
module {
  llvm.func local_unnamed_addr @_Z8tuplef2_5TupleI5Int6413CuTracedArrayIS0_Li0ELi1E2__EE(%arg0: !llvm.struct<(i64, array<1 x ptr<1>>)>) attributes {sym_visibility = "private"} {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<(i64, array<1 x ptr<1>>)>
    %1 = llvm.extractvalue %arg0[1] : !llvm.struct<(i64, array<1 x ptr<1>>)>
    %2 = llvm.extractvalue %1[0] : !llvm.array<1 x ptr<1>>
    %3 = llvm.bitcast %2 : !llvm.ptr<1> to !llvm.ptr<1>
    %4 = llvm.load %3 {alignment = 1 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr<1> -> i64
    %5 = llvm.mul %4, %0 : i64
    llvm.store %5, %3 {alignment = 1 : i64, tbaa = [#tbaa_tag]} : i64, !llvm.ptr<1>
    llvm.return
  }
  llvm.func ptx_kernelcc @"##call__Z8tuplef2_5TupleI5Int6413CuTracedArrayIS0_Li0ELi1E2__EE#258"(%arg0: !llvm.ptr<1>) attributes {sym_visibility = "private"} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(i64, array<1 x ptr<1>>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(dense<[5, 0, 0, 0, 0, 0, 0, 0, 112, 231, 165, 87, 9, 117, 0, 0]> : tensor<16xui8>) : !llvm.array<16 x i8>
    llvm.store %2, %1 : !llvm.array<16 x i8>, !llvm.ptr
    %3 = llvm.getelementptr %1[8] : (!llvm.ptr) -> !llvm.ptr, ui8
    llvm.store %arg0, %3 : !llvm.ptr<1>, !llvm.ptr
    %4 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(i64, array<1 x ptr<1>>)>
    llvm.call @_Z8tuplef2_5TupleI5Int6413CuTracedArrayIS0_Li0ELi1E2__EE(%4) : (!llvm.struct<(i64, array<1 x ptr<1>>)>) -> ()
    llvm.return
  }
  func.func @main(%arg0: tensor<i64>) -> tensor<i64> {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %c_0 = stablehlo.constant dense<1> : tensor<i64>
    %c_1 = stablehlo.constant dense<1> : tensor<i64>
    %c_2 = stablehlo.constant dense<1> : tensor<i64>
    %c_3 = stablehlo.constant dense<1> : tensor<i64>
    %c_4 = stablehlo.constant dense<1> : tensor<i64>
    %c_5 = stablehlo.constant dense<0> : tensor<i64>
    %0 = enzymexla.kernel_call @"##call__Z8tuplef2_5TupleI5Int6413CuTracedArrayIS0_Li0ELi1E2__EE#258" blocks in(%c, %c_0, %c_1) threads in(%c_2, %c_3, %c_4) shmem = %c_5 (%arg0) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<i64>) -> tensor<i64>
    return %0 : tensor<i64>
  }
}

// CHECK:  func.func @main(%arg0: tensor<i64>) -> tensor<i64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %0 = enzymexla.kernel_call @"##call__Z8tuplef2_5TupleI5Int6413CuTracedArrayIS0_Li0ELi1E2__EE#258" blocks in(%c, %c, %c) threads in(%c, %c, %c) shmem = %c_0 (%arg0) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<i64>) -> tensor<i64>
// CHECK-NEXT:    return %0 : tensor<i64>
// CHECK-NEXT:  }
// CHECK:  llvm.func ptx_kernelcc @"##call__Z8tuplef2_5TupleI5Int6413CuTracedArrayIS0_Li0ELi1E2__EE#258"(%arg0: !llvm.ptr<1>
// CHECK-NEXT:    %0 = llvm.mlir.constant(5 : i64) : i64
// CHECK-NEXT:    %1 = llvm.load %arg0 {alignment = 1 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr<1> -> i64
// CHECK-NEXT:    %2 = llvm.mul %1, %0 : i64
// CHECK-NEXT:    llvm.store %2, %arg0 {alignment = 1 : i64, tbaa = [#tbaa_tag]} : i64, !llvm.ptr<1>
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }

