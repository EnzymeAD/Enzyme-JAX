// RUN: enzymexlamlir-opt %s -polygeist-mem2reg -canonicalize -tessera-apply-pdl | FileCheck %s

#loop_unroll = #llvm.loop_unroll<disable = true>
#tbaa_root = #llvm.tbaa_root<id = "Simple C++ TBAA">
#loop_annotation = #llvm.loop_annotation<unroll = #loop_unroll, mustProgress = true>
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc, access_type = #tbaa_type_desc, offset = 0>
#tbaa_type_desc1 = #llvm.tbaa_type_desc<id = "float", members = {<#tbaa_type_desc, 0>}>
#tbaa_tag1 = #llvm.tbaa_tag<base_type = #tbaa_type_desc1, access_type = #tbaa_type_desc1, offset = 0>

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {

  tessera.define @eigen.inv(%arg0: !llvm.ptr {llvm.align = 16 : i64, llvm.noalias, llvm.sret = !llvm.struct<"class.Eigen::Matrix", (struct<"class.Eigen::PlainObjectBase", (struct<"class.Eigen::DenseStorage", (struct<"struct.Eigen::internal::plain_array", (array<16 x f32>)>)>)>)>, llvm.writeonly}, %arg1: !llvm.ptr {llvm.noundef, llvm.readonly}) attributes {CConv = #llvm.cconv<ccc>, dso_local, linkage = #llvm.linkage<external>, no_inline, no_unwind, passthrough = ["mustprogress", ["min-legal-vector-width", "128"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tessera.convert = #tessera<convert "eigen.inv" byref = [true, false] sizes = [64, 4] pure = true>, tessera.original_name = "_Z7inverseN5Eigen6MatrixIfLi4ELi4ELi0ELi4ELi4EEE", tessera.side_effect_free, tessera.sret_attrs = {llvm.align = 16 : i64, llvm.noalias, llvm.sret = !llvm.struct<"class.Eigen::Matrix", (struct<"class.Eigen::PlainObjectBase", (struct<"class.Eigen::DenseStorage", (struct<"struct.Eigen::internal::plain_array", (array<16 x f32>)>)>)>)>, llvm.writeonly}, tune_cpu = "generic", unnamed_addr = 0 : i64, uwtable_kind = #llvm.uwtableKind<async>, visibility_ = 0 : i64} {

  tessera.return
  }

  // CHECK-LABEL: llvm.func local_unnamed_addr @main
  llvm.func local_unnamed_addr @main(%12: !llvm.ptr, %10: !llvm.ptr) -> (!llvm.ptr {llvm.noundef}) attributes {dso_local, no_unwind, passthrough = ["mustprogress", "norecurse", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uwtable_kind = #llvm.uwtableKind<async>} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %11 = llvm.alloca %0 x !llvm.struct<"class.Eigen::Matrix", (struct<"class.Eigen::PlainObjectBase", (struct<"class.Eigen::DenseStorage", (struct<"struct.Eigen::internal::plain_array", (array<16 x f32>)>)>)>)> {alignment = 16 : i64} : (i32) -> !llvm.ptr
    %14 = llvm.load %12 : !llvm.ptr -> i512 
    %15 = tessera.call @eigen.inv(%14) {CConv = #llvm.cconv<ccc>, TailCallKind = #llvm.tailcallkind<none>, arg_attrs = [{llvm.nonnull, llvm.noundef}], fastmathFlags = #llvm.fastmath<none>, op_bundle_sizes = array<i32>, operandSegmentSizes = array<i32: 2, 0>, tessera.loaded_operands = array<i32: 0>} : (i512) -> !llvm.struct<"class.Eigen::Matrix", (struct<"class.Eigen::PlainObjectBase", (struct<"class.Eigen::DenseStorage", (struct<"struct.Eigen::internal::plain_array", (array<16 x f32>)>)>)>)>
    llvm.store %15, %11 : !llvm.struct<"class.Eigen::Matrix", (struct<"class.Eigen::PlainObjectBase", (struct<"class.Eigen::DenseStorage", (struct<"struct.Eigen::internal::plain_array", (array<16 x f32>)>)>)>)>, !llvm.ptr
    %16 = llvm.load %11 : !llvm.ptr -> i512
    %17 = tessera.call @eigen.inv(%16) {CConv = #llvm.cconv<ccc>, TailCallKind = #llvm.tailcallkind<none>, arg_attrs = [{llvm.nonnull, llvm.noundef}], fastmathFlags = #llvm.fastmath<none>, op_bundle_sizes = array<i32>, operandSegmentSizes = array<i32: 2, 0>, tessera.loaded_operands = array<i32: 0>} : (i512) -> !llvm.struct<"class.Eigen::Matrix", (struct<"class.Eigen::PlainObjectBase", (struct<"class.Eigen::DenseStorage", (struct<"struct.Eigen::internal::plain_array", (array<16 x f32>)>)>)>)>
    llvm.store %17, %10 : !llvm.struct<"class.Eigen::Matrix", (struct<"class.Eigen::PlainObjectBase", (struct<"class.Eigen::DenseStorage", (struct<"struct.Eigen::internal::plain_array", (array<16 x f32>)>)>)>)>, !llvm.ptr
    llvm.return %10 : !llvm.ptr
    // CHECK-NOT: llvm.alloca
    // CHECK-NOT: tessera.call @eigen.inv
    // CHECK: %[[RES:.*]] = llvm.load %arg0
    // CHECK: llvm.store %[[RES]], %arg1
    // CHECK: llvm.return %arg1
  }

  module @patterns {
    pdl.pattern : benefit(1) {
      %0 = operand
      %1 = attribute = @eigen.inv
      %2 = type
      %3 = operation "tessera.call"(%0 : !pdl.value)  {"callee" = %1} -> (%2 : !pdl.type)
      %4 = result 0 of %3
      %5 = attribute = @eigen.inv
      %6 = type
      %7 = operation "tessera.call"(%4 : !pdl.value)  {"callee" = %5} -> (%6 : !pdl.type)
      rewrite %7 {
        replace %7 with(%0 : !pdl.value)
      }
    }
  }
}

