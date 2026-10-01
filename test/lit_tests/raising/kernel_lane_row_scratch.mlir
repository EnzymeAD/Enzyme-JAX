// RUN: enzymexlamlir-opt %s --pass-pipeline='builtin.module(raise-affine-to-stablehlo{prefer_while_raising=false})' | FileCheck %s

// A face-restriction kernel (mfem's 
// L2NormalDerivativeFaceRestriction::Mult3D<1>) stores into a 2x1 scratch at 
// [lane + t, 0] where the lane loop is sized to one iteration in context and t 
// is a thread of two: the row-major check of the flattened pair must skip the 
// one-element dimension, or the masked store's update (2) and mask (1) disagree 
// and the select cannot be built.
// The kernel is kept whole (declarations only for what it calls); the check is that it raises.

#set = affine_set<(d0) : (d0 == 0)>
#set2 = affine_set<()[s0] : (s0 * 2 - 1 >= 0)>
#set3 = affine_set<()[s0] : (s0 - 1 >= 0)>
#set4 = affine_set<()[s0] : (s0 - 1 == 0)>
#set5 = affine_set<()[s0] : (s0 * 6 - 1 >= 0)>
#tbaa_root = #llvm.tbaa_root<id = "Simple C++ TBAA">
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root, 0>}>
#tbaa_type_desc1 = #llvm.tbaa_type_desc<id = "int", members = {<#tbaa_type_desc, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc1, access_type = #tbaa_type_desc1, offset = 0>
module {
  llvm.mlir.global external local_unnamed_addr @_ZN4mfem6Device16device_singletonE() {addr_space = 0 : i32, alignment = 1 : i64} : !llvm.struct<"class.mfem::Device.1", packed (i32, i32, i64, i8, i8, array<2 x i8>, i32, i32, i32, i32, array<4 x i8>)>
  llvm.mlir.global linkonce_odr constant @_ZTVN4mfem6VectorE() {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : !llvm.struct<(array<12 x ptr>)> {
    %0 = llvm.mlir.undef : !llvm.struct<(array<12 x ptr>)>
    %1 = llvm.mlir.addressof @_ZN4mfem6Vector13HostReadWriteEv : !llvm.ptr
    %2 = llvm.mlir.addressof @_ZN4mfem6Vector9ReadWriteEb : !llvm.ptr
    %3 = llvm.mlir.addressof @_ZN4mfem6Vector9HostWriteEv : !llvm.ptr
    %4 = llvm.mlir.addressof @_ZN4mfem6Vector5WriteEb : !llvm.ptr
    %5 = llvm.mlir.addressof @_ZNK4mfem6Vector8HostReadEv : !llvm.ptr
    %6 = llvm.mlir.addressof @_ZNK4mfem6Vector4ReadEb : !llvm.ptr
    %7 = llvm.mlir.addressof @_ZN4mfem6VectorD0Ev : !llvm.ptr
    %8 = llvm.mlir.addressof @_ZN4mfem6VectorD2Ev : !llvm.ptr
    %9 = llvm.mlir.addressof @_ZNK4mfem6Vector9UseDeviceEv : !llvm.ptr
    %10 = llvm.mlir.addressof @_ZNK4mfem6Vector9UseDeviceEb : !llvm.ptr
    %11 = llvm.mlir.addressof @_ZTIN4mfem6VectorE : !llvm.ptr
    %12 = llvm.mlir.zero : !llvm.ptr
    %13 = llvm.mlir.undef : !llvm.array<12 x ptr>
    %14 = llvm.insertvalue %12, %13[0] : !llvm.array<12 x ptr> 
    %15 = llvm.insertvalue %11, %14[1] : !llvm.array<12 x ptr> 
    %16 = llvm.insertvalue %10, %15[2] : !llvm.array<12 x ptr> 
    %17 = llvm.insertvalue %9, %16[3] : !llvm.array<12 x ptr> 
    %18 = llvm.insertvalue %8, %17[4] : !llvm.array<12 x ptr> 
    %19 = llvm.insertvalue %7, %18[5] : !llvm.array<12 x ptr> 
    %20 = llvm.insertvalue %6, %19[6] : !llvm.array<12 x ptr> 
    %21 = llvm.insertvalue %5, %20[7] : !llvm.array<12 x ptr> 
    %22 = llvm.insertvalue %4, %21[8] : !llvm.array<12 x ptr> 
    %23 = llvm.insertvalue %3, %22[9] : !llvm.array<12 x ptr> 
    %24 = llvm.insertvalue %2, %23[10] : !llvm.array<12 x ptr> 
    %25 = llvm.insertvalue %1, %24[11] : !llvm.array<12 x ptr> 
    %26 = llvm.insertvalue %25, %0[0] : !llvm.struct<(array<12 x ptr>)> 
    llvm.return %26 : !llvm.struct<(array<12 x ptr>)>
  }
  llvm.mlir.global private unnamed_addr constant @".str.2"("Verification failed: (\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.4"(") is false:\0A --> \00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.6"("\0A ... in function: \00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.7"("\0A ... in file: \00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.8"("/mnt3/wmoses/git/mfem/fem/normal_deriv_restriction.cpp\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global external @_ZTVN10__cxxabiv117__class_type_infoE() {addr_space = 0 : i32} : !llvm.array<0 x ptr>
  llvm.mlir.global linkonce_odr constant @_ZTIN4mfem6VectorE() {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : !llvm.struct<(ptr, ptr)> {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr)>
    %1 = llvm.mlir.addressof @_ZTSN4mfem6VectorE : !llvm.ptr
    %2 = llvm.mlir.addressof @_ZTVN10__cxxabiv117__class_type_infoE : !llvm.ptr
    %3 = llvm.getelementptr inbounds|nuw %2[16] : (!llvm.ptr) -> !llvm.ptr, i8
    %4 = llvm.insertvalue %3, %0[0] : !llvm.struct<(ptr, ptr)> 
    %5 = llvm.insertvalue %1, %4[1] : !llvm.struct<(ptr, ptr)> 
    llvm.return %5 : !llvm.struct<(ptr, ptr)>
  }
  llvm.mlir.global linkonce_odr constant @_ZTSN4mfem6VectorE("N4mfem6VectorE\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.14"(dense<0> : tensor<1xi8>) {addr_space = 0 : i32, alignment = 1 : i64, dso_local} : !llvm.array<1 x i8>
  llvm.mlir.global private unnamed_addr constant @".str.19"("q == d\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.20"("T_D1D == d || T_D1D == 0\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.22"("/mnt3/wmoses/git/mfem/fem/../general/forall.hpp\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.23"("cudaGetLastError()\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @__PRETTY_FUNCTION__._ZNK4mfem33L2NormalDerivativeFaceRestriction6Mult3DILi1EEEvRKNS_6VectorERS2_("void mfem::L2NormalDerivativeFaceRestriction::Mult3D(const Vector &, Vector &) const [T_D1D = 1]\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @__PRETTY_FUNCTION__._ZN4mfem8CuWrap2DIRZNKS_33L2NormalDerivativeFaceRestriction6Mult3DILi1EEEvRKNS_6VectorERS3_EUliE_EEviOT_iii("void mfem::CuWrap2D(const int, DBODY &&, const int, const int, const int) [DBODY = (lambda at /mnt3/wmoses/git/mfem/fem/normal_deriv_restriction.cpp:462:32) &]\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.func @__gxx_personality_v0(...) -> i32 attributes {sym_visibility = "private"}
  llvm.func local_unnamed_addr @_ZdlPvm(!llvm.ptr {llvm.noundef}, i64 {llvm.noundef}) attributes {no_unwind, passthrough = ["nobuiltin", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], "uniform-work-group-size"], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @_ZN4mfem10mfem_errorEPKc(!llvm.ptr {llvm.noundef}) attributes {noreturn, passthrough = ["enzyme_inactive", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], "uniform-work-group-size"], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @_ZN4mfem13MemoryManager7Delete_EPvNS_10MemoryTypeEj(!llvm.ptr {llvm.noundef}, i32 {llvm.noundef}, i32 {llvm.noundef}) attributes {passthrough = [["enzyme_math", "free"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], "uniform-work-group-size"], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func hidden local_unnamed_addr @__clang_call_terminate(%arg0: !llvm.ptr {llvm.noundef}) attributes {dso_local, no_inline, no_unwind, noreturn, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uwtable_kind = #llvm.uwtableKind<async>}
  llvm.func unnamed_addr @_ZN4mfem6VectorD0Ev(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 36 : i64, llvm.nonnull, llvm.noundef}) attributes {alignment = 2 : i64, dso_local, inline_hint, no_unwind, passthrough = ["mustprogress", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], "uniform-work-group-size"], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uwtable_kind = #llvm.uwtableKind<async>}
  llvm.func unnamed_addr @_ZN4mfem6VectorD2Ev(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dead_on_return = 36 : i64, llvm.dereferenceable = 36 : i64, llvm.nonnull, llvm.noundef}) attributes {alignment = 2 : i64, dso_local, inline_hint, no_unwind, passthrough = ["mustprogress", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], "uniform-work-group-size"], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uwtable_kind = #llvm.uwtableKind<async>}
  llvm.func unnamed_addr @_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEC1Ev(!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 112 : i64, llvm.nonnull, llvm.noundef}) attributes {alignment = 2 : i64, passthrough = ["mustprogress", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], "uniform-work-group-size"], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, uwtable_kind = #llvm.uwtableKind<async>}
  llvm.func local_unnamed_addr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_c(!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, i8 {llvm.noundef, llvm.signext}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}) attributes {inline_hint, passthrough = ["mustprogress", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], "uniform-work-group-size"], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, uwtable_kind = #llvm.uwtableKind<async>}
  llvm.func local_unnamed_addr @_ZNSolsEi(!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, i32 {llvm.noundef}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}) attributes {passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], "uniform-work-group-size"], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @_ZNKSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEE3strEv(!llvm.ptr {llvm.align = 8 : i64, llvm.dead_on_unwind, llvm.noalias, llvm.sret = !llvm.struct<"class.std::__cxx11::basic_string.1", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider.1", (ptr)>, i64, struct<"union.anon.1", (i64, array<8 x i8>)>)>, llvm.writable}, !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 112 : i64, llvm.nonnull, llvm.noundef}) attributes {alignment = 2 : i64, passthrough = ["mustprogress", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], "uniform-work-group-size"], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, uwtable_kind = #llvm.uwtableKind<async>}
  llvm.func unnamed_addr @_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev(!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 112 : i64, llvm.nonnull, llvm.noundef}) attributes {alignment = 2 : i64, no_unwind, passthrough = ["mustprogress", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], "uniform-work-group-size"], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, uwtable_kind = #llvm.uwtableKind<async>}
  llvm.func local_unnamed_addr @_ZNK4mfem18FiniteElementSpace12GetTypicalFEEv(!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 1426 : i64, llvm.nonnull, llvm.noundef}) -> (!llvm.ptr {llvm.noundef}) attributes {passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], "uniform-work-group-size"], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func linkonce_odr local_unnamed_addr @_ZNK4mfem33L2NormalDerivativeFaceRestriction6Mult3DILi1EEEvRKNS_6VectorERS2_(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 128 : i64, llvm.nonnull, llvm.noundef}, %arg1: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 36 : i64, llvm.nonnull, llvm.noundef}, %arg2: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 36 : i64, llvm.nonnull, llvm.noundef}) attributes {alignment = 2 : i64, dso_local, passthrough = ["mustprogress", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], "uniform-work-group-size"], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uwtable_kind = #llvm.uwtableKind<async>} {
    %c-1 = arith.constant -1 : index
    %c8 = arith.constant 8 : index
    %0 = llvm.mlir.addressof @_ZN4mfem6Device16device_singletonE : !llvm.ptr
    %c64_i32 = arith.constant 64 : i32
    %c449_i32 = arith.constant 449 : i32
    %c24_i64 = arith.constant 24 : i64
    %1 = llvm.mlir.addressof @".str.20" : !llvm.ptr
    %c10_i8 = arith.constant 10 : i8
    %c448_i32 = arith.constant 448 : i32
    %c58_i8 = arith.constant 58 : i8
    %c54_i64 = arith.constant 54 : i64
    %2 = llvm.mlir.addressof @".str.8" : !llvm.ptr
    %c15_i64 = arith.constant 15 : i64
    %3 = llvm.mlir.addressof @".str.7" : !llvm.ptr
    %c96_i64 = arith.constant 96 : i64
    %4 = llvm.mlir.addressof @__PRETTY_FUNCTION__._ZNK4mfem33L2NormalDerivativeFaceRestriction6Mult3DILi1EEEvRKNS_6VectorERS2_ : !llvm.ptr
    %c19_i64 = arith.constant 19 : i64
    %5 = llvm.mlir.addressof @".str.6" : !llvm.ptr
    %c0_i64 = arith.constant 0 : i64
    %6 = llvm.mlir.addressof @".str.14" : !llvm.ptr
    %c17_i64 = arith.constant 17 : i64
    %7 = llvm.mlir.addressof @".str.4" : !llvm.ptr
    %c6_i64 = arith.constant 6 : i64
    %8 = llvm.mlir.addressof @".str.19" : !llvm.ptr
    %c22_i64 = arith.constant 22 : i64
    %9 = llvm.mlir.addressof @".str.2" : !llvm.ptr
    %c256_i32 = arith.constant 256 : i32
    %c16_i64 = arith.constant 16 : i64
    %c1_i64 = arith.constant 1 : i64
    %c1_i32 = arith.constant 1 : i32
    %c6_i32 = arith.constant 6 : i32
    %c0_i32 = arith.constant 0 : i32
    %10 = llvm.mlir.zero : !llvm.ptr
    %true = arith.constant true
    %c4_i64 = arith.constant 4 : i64
    %c2_i32 = arith.constant 2 : i32
    %11 = llvm.mlir.addressof @".str.23" : !llvm.ptr
    %12 = llvm.mlir.addressof @__PRETTY_FUNCTION__._ZN4mfem8CuWrap2DIRZNKS_33L2NormalDerivativeFaceRestriction6Mult3DILi1EEEvRKNS_6VectorERS3_EUliE_EEviOT_iii : !llvm.ptr
    %13 = llvm.mlir.addressof @".str.22" : !llvm.ptr
    %c785_i32 = arith.constant 785 : i32
    %c2_i64 = arith.constant 2 : i64
    %c5_i32 = arith.constant 5 : i32
    %c-3_i32 = arith.constant -3 : i32
    %cst = arith.constant 0.000000e+00 : f64
    %false = arith.constant false
    %14 = llvm.mlir.addressof @_ZTVN4mfem6VectorE : !llvm.ptr
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %15 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xi8>
    %16 = llvm.getelementptr inbounds|nuw %14[16] : (!llvm.ptr) -> !llvm.ptr, i8
    %17 = llvm.alloca %c1_i32 x !llvm.array<2 x array<1 x i32>> {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %18 = llvm.alloca %c1_i32 x !llvm.struct<"class.mfem::Vector.1", packed (ptr, struct<"class.mfem::Memory.1", packed (ptr, i32, i32, i32, array<4 x i8>)>, i32, array<4 x i8>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %19 = llvm.alloca %c1_i32 x !llvm.struct<"class.std::__cxx11::basic_ostringstream.1", (struct<"class.std::basic_ostream.base.1", (ptr)>, struct<"class.std::__cxx11::basic_stringbuf.1", (struct<"class.std::basic_streambuf.1", (ptr, ptr, ptr, ptr, ptr, ptr, ptr, struct<"class.std::locale.1", (ptr)>)>, i32, struct<"class.std::__cxx11::basic_string.1", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider.1", (ptr)>, i64, struct<"union.anon.1", (i64, array<8 x i8>)>)>)>, struct<"class.std::basic_ios.1", (struct<"class.std::ios_base.1", (ptr, i64, i64, i32, i32, i32, ptr, struct<"struct.std::ios_base::_Words.1", (ptr, i64)>, array<8 x struct<"struct.std::ios_base::_Words.1", (ptr, i64)>>, i32, ptr, struct<"class.std::locale.1", (ptr)>)>, ptr, i8, i8, ptr, ptr, ptr, ptr)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %20 = llvm.alloca %c1_i32 x !llvm.struct<"class.std::__cxx11::basic_string.1", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider.1", (ptr)>, i64, struct<"union.anon.1", (i64, array<8 x i8>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %21 = llvm.alloca %c1_i32 x !llvm.struct<"class.std::__cxx11::basic_ostringstream.1", (struct<"class.std::basic_ostream.base.1", (ptr)>, struct<"class.std::__cxx11::basic_stringbuf.1", (struct<"class.std::basic_streambuf.1", (ptr, ptr, ptr, ptr, ptr, ptr, ptr, struct<"class.std::locale.1", (ptr)>)>, i32, struct<"class.std::__cxx11::basic_string.1", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider.1", (ptr)>, i64, struct<"union.anon.1", (i64, array<8 x i8>)>)>)>, struct<"class.std::basic_ios.1", (struct<"class.std::ios_base.1", (ptr, i64, i64, i32, i32, i32, ptr, struct<"struct.std::ios_base::_Words.1", (ptr, i64)>, array<8 x struct<"struct.std::ios_base::_Words.1", (ptr, i64)>>, i32, ptr, struct<"class.std::locale.1", (ptr)>)>, ptr, i8, i8, ptr, ptr, ptr, ptr)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %22 = llvm.alloca %c1_i32 x !llvm.struct<"class.std::__cxx11::basic_string.1", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider.1", (ptr)>, i64, struct<"union.anon.1", (i64, array<8 x i8>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %23 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    %24 = affine.load %23[0] : memref<?x!llvm.ptr>
    %25 = "enzymexla.pointer2memref"(%24) : (!llvm.ptr) -> memref<?xi32>
    %26 = affine.load %25[6] : memref<?xi32>
    %27 = arith.index_cast %26 : i32 to index
    %28 = affine.load %25[7] : memref<?xi32>
    %29 = arith.index_cast %28 : i32 to index
    %30 = arith.cmpi eq, %28, %c1_i32 : i32
    %31 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xi32>
    %32 = affine.load %31[5] : memref<?xi32>
    %33 = llvm.call tail @_ZNK4mfem18FiniteElementSpace12GetTypicalFEEv(%24) : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 1426 : i64, llvm.nonnull, llvm.noundef}) -> (!llvm.ptr {llvm.noundef})
    %34 = llvm.getelementptr inbounds|nuw %33[72] : (!llvm.ptr) -> !llvm.ptr, i8
    %35 = "enzymexla.pointer2memref"(%33) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    %36 = affine.load %35[0] : memref<?x!llvm.ptr>
    %37 = "enzymexla.pointer2memref"(%36) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    %38 = affine.load %37[23] : memref<?x!llvm.ptr>
    %39 = llvm.call tail %38(%33, %34, %c1_i32) : !llvm.ptr, (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 216 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 64 : i64, llvm.nonnull, llvm.noundef}, i32 {llvm.noundef}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 168 : i64, llvm.nonnull, llvm.noundef})
    %40 = "enzymexla.pointer2memref"(%39) : (!llvm.ptr) -> memref<?xi32>
    %41 = affine.load %40[8] : memref<?xi32>
    %42 = affine.load %40[7] : memref<?xi32>
    %43 = arith.muli %41, %41 overflow<nsw> : i32
    %44 = arith.index_cast %43 : i32 to index
    llvm.intr.lifetime.start %18 : !llvm.ptr
    %45 = affine.load %23[0] : memref<?x!llvm.ptr>
    %46 = affine.load %15[8] : memref<?xi8>
    %47 = arith.trunci %46 : i8 to i1
    llvm.call @_ZN4mfem21GetLVectorFaceNbrDataERKNS_18FiniteElementSpaceERKNS_6VectorENS_8FaceTypeE(%18, %45, %arg1, %47) : (!llvm.ptr {llvm.align = 8 : i64, llvm.dead_on_unwind, llvm.nonnull, llvm.sret = !llvm.struct<"class.mfem::Vector.1", packed (ptr, struct<"class.mfem::Memory.1", packed (ptr, i32, i32, i32, array<4 x i8>)>, i32, array<4 x i8>)>, llvm.writable}, !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 1426 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 36 : i64, llvm.nonnull, llvm.noundef}, i1 {llvm.noundef, llvm.zeroext}) -> ()
    %48 = "enzymexla.pointer2memref"(%18) : (!llvm.ptr) -> memref<?xi32>
    %49 = affine.load %48[8] : memref<?xi32>
    %50 = arith.divsi %49, %42 : i32
    %51 = arith.divsi %50, %42 : i32
    %52 = arith.divsi %51, %42 : i32
    %53 = arith.divsi %52, %26 : i32
    %54 = arith.cmpi eq, %41, %42 : i32
    %55 = llvm.getelementptr inbounds|nuw %17[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i8>
    cf.cond_br %54, ^bb25, ^bb1
  ^bb1:  // pred: ^bb0
    llvm.intr.lifetime.start %19 : !llvm.ptr
    llvm.invoke @_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEC1Ev(%19) to ^bb2 unwind ^bb16 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 112 : i64, llvm.nonnull, llvm.noundef}) -> ()
  ^bb2:  // pred: ^bb1
    %56 = "enzymexla.pointer2memref"(%19) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    %57 = affine.load %56[0] : memref<?x!llvm.ptr>
    %58 = "enzymexla.pointer2memref"(%57) : (!llvm.ptr) -> memref<?xi64>
    %59 = affine.load %58[-3] : memref<?xi64>
    %60 = arith.index_cast %59 : i64 to index
    %61 = "enzymexla.pointer2memref"(%19) : (!llvm.ptr) -> memref<?xi64>
    %62 = arith.cmpi slt, %60, %c0 : index
    %63 = arith.subi %c-1, %60 : index
    %64 = arith.select %62, %63, %60 : index
    %65 = arith.divsi %64, %c8 : index
    %66 = arith.subi %c-1, %65 : index
    %67 = arith.select %62, %66, %65 : index
    affine.store %c16_i64, %61[symbol(%67) + 1] : memref<?xi64>
    %68 = affine.load %58[-3] : memref<?xi64>
    %69 = arith.index_cast %68 : i64 to index
    %70 = "enzymexla.pointer2memref"(%19) : (!llvm.ptr) -> memref<?xi32>
    %71 = arith.cmpi slt, %69, %c0 : index
    %72 = arith.subi %c-1, %69 : index
    %73 = arith.select %71, %72, %69 : index
    %74 = arith.divsi %73, %c4 : index
    %75 = arith.subi %c-1, %74 : index
    %76 = arith.select %71, %75, %74 : index
    %77 = affine.load %70[symbol(%76) + 6] : memref<?xi32>
    %78 = arith.ori %77, %c256_i32 : i32
    affine.store %78, %70[symbol(%76) + 6] : memref<?xi32>
    %79 = llvm.invoke @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%19, %9, %c22_i64) to ^bb3 unwind ^bb17 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, i64 {llvm.noundef}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef})
  ^bb3:  // pred: ^bb2
    %80 = llvm.invoke @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%19, %8, %c6_i64) to ^bb4 unwind ^bb17 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, i64 {llvm.noundef}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef})
  ^bb4:  // pred: ^bb3
    %81 = llvm.invoke @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%19, %7, %c17_i64) to ^bb5 unwind ^bb17 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, i64 {llvm.noundef}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef})
  ^bb5:  // pred: ^bb4
    %82 = llvm.invoke @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%19, %6, %c0_i64) to ^bb6 unwind ^bb17 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, i64 {llvm.noundef}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef})
  ^bb6:  // pred: ^bb5
    %83 = llvm.invoke @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%19, %5, %c19_i64) to ^bb7 unwind ^bb17 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, i64 {llvm.noundef}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef})
  ^bb7:  // pred: ^bb6
    %84 = llvm.invoke @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%19, %4, %c96_i64) to ^bb8 unwind ^bb17 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, i64 {llvm.noundef}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef})
  ^bb8:  // pred: ^bb7
    %85 = llvm.invoke @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%19, %3, %c15_i64) to ^bb9 unwind ^bb17 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, i64 {llvm.noundef}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef})
  ^bb9:  // pred: ^bb8
    %86 = llvm.invoke @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%19, %2, %c54_i64) to ^bb10 unwind ^bb17 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, i64 {llvm.noundef}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef})
  ^bb10:  // pred: ^bb9
    %87 = llvm.invoke @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_c(%19, %c58_i8) to ^bb11 unwind ^bb17 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, i8 {llvm.noundef, llvm.signext}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef})
  ^bb11:  // pred: ^bb10
    %88 = llvm.invoke @_ZNSolsEi(%87, %c448_i32) to ^bb12 unwind ^bb17 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, i32 {llvm.noundef}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef})
  ^bb12:  // pred: ^bb11
    %89 = llvm.invoke @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_c(%88, %c10_i8) to ^bb13 unwind ^bb17 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, i8 {llvm.noundef, llvm.signext}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef})
  ^bb13:  // pred: ^bb12
    llvm.intr.lifetime.start %20 : !llvm.ptr
    llvm.invoke @_ZNKSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEE3strEv(%20, %19) to ^bb14 unwind ^bb18 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dead_on_unwind, llvm.nonnull, llvm.sret = !llvm.struct<"class.std::__cxx11::basic_string.1", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider.1", (ptr)>, i64, struct<"union.anon.1", (i64, array<8 x i8>)>)>, llvm.writable}, !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 112 : i64, llvm.nonnull, llvm.noundef}) -> ()
  ^bb14:  // pred: ^bb13
    %90 = "enzymexla.pointer2memref"(%20) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    %91 = affine.load %90[0] : memref<?x!llvm.ptr>
    llvm.invoke @_ZN4mfem10mfem_errorEPKc(%91) to ^bb15 unwind ^bb19 : (!llvm.ptr {llvm.noundef}) -> ()
  ^bb15:  // pred: ^bb14
    llvm.unreachable
  ^bb16:  // pred: ^bb1
    %92 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    cf.br ^bb24(%92 : !llvm.struct<(ptr, i32)>)
  ^bb17:  // 11 preds: ^bb2, ^bb3, ^bb4, ^bb5, ^bb6, ^bb7, ^bb8, ^bb9, ^bb10, ^bb11, ^bb12
    %93 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    cf.br ^bb23(%93 : !llvm.struct<(ptr, i32)>)
  ^bb18:  // pred: ^bb13
    %94 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    cf.br ^bb22(%94 : !llvm.struct<(ptr, i32)>)
  ^bb19:  // pred: ^bb14
    %95 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    %96 = affine.load %90[0] : memref<?x!llvm.ptr>
    %97 = llvm.getelementptr inbounds|nuw %20[16] : (!llvm.ptr) -> !llvm.ptr, i8
    %98 = llvm.icmp "eq" %96, %97 : !llvm.ptr
    cf.cond_br %98, ^bb20, ^bb21
  ^bb20:  // pred: ^bb19
    %99 = "enzymexla.pointer2memref"(%20) : (!llvm.ptr) -> memref<?xi64>
    %100 = affine.load %99[1] : memref<?xi64>
    %101 = arith.cmpi ult, %100, %c16_i64 : i64
    llvm.intr.assume %101 : i1
    cf.br ^bb22(%95 : !llvm.struct<(ptr, i32)>)
  ^bb21:  // pred: ^bb19
    %102 = "enzymexla.pointer2memref"(%20) : (!llvm.ptr) -> memref<?xi64>
    %103 = affine.load %102[2] : memref<?xi64>
    %104 = arith.addi %103, %c1_i64 : i64
    llvm.call @_ZdlPvm(%96, %104) {builtin, no_unwind} : (!llvm.ptr {llvm.noundef}, i64 {llvm.noundef}) -> ()
    cf.br ^bb22(%95 : !llvm.struct<(ptr, i32)>)
  ^bb22(%105: !llvm.struct<(ptr, i32)>):  // 3 preds: ^bb18, ^bb20, ^bb21
    llvm.intr.lifetime.end %20 : !llvm.ptr
    cf.br ^bb23(%105 : !llvm.struct<(ptr, i32)>)
  ^bb23(%106: !llvm.struct<(ptr, i32)>):  // 2 preds: ^bb17, ^bb22
    llvm.call @_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev(%19) {no_unwind} : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 112 : i64, llvm.nonnull, llvm.noundef}) -> ()
    cf.br ^bb24(%106 : !llvm.struct<(ptr, i32)>)
  ^bb24(%107: !llvm.struct<(ptr, i32)>):  // 2 preds: ^bb16, ^bb23
    llvm.intr.lifetime.end %19 : !llvm.ptr
    cf.br ^bb115(%107 : !llvm.struct<(ptr, i32)>)
  ^bb25:  // pred: ^bb0
    %108 = arith.cmpi eq, %41, %c1_i32 : i32
    cf.cond_br %108, ^bb50, ^bb26
  ^bb26:  // pred: ^bb25
    llvm.intr.lifetime.start %21 : !llvm.ptr
    llvm.invoke @_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEC1Ev(%21) to ^bb27 unwind ^bb41 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 112 : i64, llvm.nonnull, llvm.noundef}) -> ()
  ^bb27:  // pred: ^bb26
    %109 = "enzymexla.pointer2memref"(%21) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    %110 = affine.load %109[0] : memref<?x!llvm.ptr>
    %111 = "enzymexla.pointer2memref"(%110) : (!llvm.ptr) -> memref<?xi64>
    %112 = affine.load %111[-3] : memref<?xi64>
    %113 = arith.index_cast %112 : i64 to index
    %114 = "enzymexla.pointer2memref"(%21) : (!llvm.ptr) -> memref<?xi64>
    %115 = arith.cmpi slt, %113, %c0 : index
    %116 = arith.subi %c-1, %113 : index
    %117 = arith.select %115, %116, %113 : index
    %118 = arith.divsi %117, %c8 : index
    %119 = arith.subi %c-1, %118 : index
    %120 = arith.select %115, %119, %118 : index
    affine.store %c16_i64, %114[symbol(%120) + 1] : memref<?xi64>
    %121 = affine.load %111[-3] : memref<?xi64>
    %122 = arith.index_cast %121 : i64 to index
    %123 = "enzymexla.pointer2memref"(%21) : (!llvm.ptr) -> memref<?xi32>
    %124 = arith.cmpi slt, %122, %c0 : index
    %125 = arith.subi %c-1, %122 : index
    %126 = arith.select %124, %125, %122 : index
    %127 = arith.divsi %126, %c4 : index
    %128 = arith.subi %c-1, %127 : index
    %129 = arith.select %124, %128, %127 : index
    %130 = affine.load %123[symbol(%129) + 6] : memref<?xi32>
    %131 = arith.ori %130, %c256_i32 : i32
    affine.store %131, %123[symbol(%129) + 6] : memref<?xi32>
    %132 = llvm.invoke @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%21, %9, %c22_i64) to ^bb28 unwind ^bb42 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, i64 {llvm.noundef}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef})
  ^bb28:  // pred: ^bb27
    %133 = llvm.invoke @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%21, %1, %c24_i64) to ^bb29 unwind ^bb42 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, i64 {llvm.noundef}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef})
  ^bb29:  // pred: ^bb28
    %134 = llvm.invoke @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%21, %7, %c17_i64) to ^bb30 unwind ^bb42 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, i64 {llvm.noundef}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef})
  ^bb30:  // pred: ^bb29
    %135 = llvm.invoke @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%21, %6, %c0_i64) to ^bb31 unwind ^bb42 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, i64 {llvm.noundef}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef})
  ^bb31:  // pred: ^bb30
    %136 = llvm.invoke @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%21, %5, %c19_i64) to ^bb32 unwind ^bb42 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, i64 {llvm.noundef}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef})
  ^bb32:  // pred: ^bb31
    %137 = llvm.invoke @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%21, %4, %c96_i64) to ^bb33 unwind ^bb42 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, i64 {llvm.noundef}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef})
  ^bb33:  // pred: ^bb32
    %138 = llvm.invoke @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%21, %3, %c15_i64) to ^bb34 unwind ^bb42 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, i64 {llvm.noundef}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef})
  ^bb34:  // pred: ^bb33
    %139 = llvm.invoke @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%21, %2, %c54_i64) to ^bb35 unwind ^bb42 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, i64 {llvm.noundef}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef})
  ^bb35:  // pred: ^bb34
    %140 = llvm.invoke @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_c(%21, %c58_i8) to ^bb36 unwind ^bb42 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, i8 {llvm.noundef, llvm.signext}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef})
  ^bb36:  // pred: ^bb35
    %141 = llvm.invoke @_ZNSolsEi(%140, %c449_i32) to ^bb37 unwind ^bb42 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, i32 {llvm.noundef}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef})
  ^bb37:  // pred: ^bb36
    %142 = llvm.invoke @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_c(%141, %c10_i8) to ^bb38 unwind ^bb42 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, i8 {llvm.noundef, llvm.signext}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef})
  ^bb38:  // pred: ^bb37
    llvm.intr.lifetime.start %22 : !llvm.ptr
    llvm.invoke @_ZNKSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEE3strEv(%22, %21) to ^bb39 unwind ^bb43 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dead_on_unwind, llvm.nonnull, llvm.sret = !llvm.struct<"class.std::__cxx11::basic_string.1", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider.1", (ptr)>, i64, struct<"union.anon.1", (i64, array<8 x i8>)>)>, llvm.writable}, !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 112 : i64, llvm.nonnull, llvm.noundef}) -> ()
  ^bb39:  // pred: ^bb38
    %143 = "enzymexla.pointer2memref"(%22) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    %144 = affine.load %143[0] : memref<?x!llvm.ptr>
    llvm.invoke @_ZN4mfem10mfem_errorEPKc(%144) to ^bb40 unwind ^bb44 : (!llvm.ptr {llvm.noundef}) -> ()
  ^bb40:  // pred: ^bb39
    llvm.unreachable
  ^bb41:  // pred: ^bb26
    %145 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    cf.br ^bb49(%145 : !llvm.struct<(ptr, i32)>)
  ^bb42:  // 11 preds: ^bb27, ^bb28, ^bb29, ^bb30, ^bb31, ^bb32, ^bb33, ^bb34, ^bb35, ^bb36, ^bb37
    %146 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    cf.br ^bb48(%146 : !llvm.struct<(ptr, i32)>)
  ^bb43:  // pred: ^bb38
    %147 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    cf.br ^bb47(%147 : !llvm.struct<(ptr, i32)>)
  ^bb44:  // pred: ^bb39
    %148 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    %149 = affine.load %143[0] : memref<?x!llvm.ptr>
    %150 = llvm.getelementptr inbounds|nuw %22[16] : (!llvm.ptr) -> !llvm.ptr, i8
    %151 = llvm.icmp "eq" %149, %150 : !llvm.ptr
    cf.cond_br %151, ^bb45, ^bb46
  ^bb45:  // pred: ^bb44
    %152 = "enzymexla.pointer2memref"(%22) : (!llvm.ptr) -> memref<?xi64>
    %153 = affine.load %152[1] : memref<?xi64>
    %154 = arith.cmpi ult, %153, %c16_i64 : i64
    llvm.intr.assume %154 : i1
    cf.br ^bb47(%148 : !llvm.struct<(ptr, i32)>)
  ^bb46:  // pred: ^bb44
    %155 = "enzymexla.pointer2memref"(%22) : (!llvm.ptr) -> memref<?xi64>
    %156 = affine.load %155[2] : memref<?xi64>
    %157 = arith.addi %156, %c1_i64 : i64
    llvm.call @_ZdlPvm(%149, %157) {builtin, no_unwind} : (!llvm.ptr {llvm.noundef}, i64 {llvm.noundef}) -> ()
    cf.br ^bb47(%148 : !llvm.struct<(ptr, i32)>)
  ^bb47(%158: !llvm.struct<(ptr, i32)>):  // 3 preds: ^bb43, ^bb45, ^bb46
    llvm.intr.lifetime.end %22 : !llvm.ptr
    cf.br ^bb48(%158 : !llvm.struct<(ptr, i32)>)
  ^bb48(%159: !llvm.struct<(ptr, i32)>):  // 2 preds: ^bb42, ^bb47
    llvm.call @_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev(%21) {no_unwind} : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 112 : i64, llvm.nonnull, llvm.noundef}) -> ()
    cf.br ^bb49(%159 : !llvm.struct<(ptr, i32)>)
  ^bb49(%160: !llvm.struct<(ptr, i32)>):  // 2 preds: ^bb41, ^bb48
    llvm.intr.lifetime.end %21 : !llvm.ptr
    cf.br ^bb115(%160 : !llvm.struct<(ptr, i32)>)
  ^bb50:  // pred: ^bb25
    %161 = llvm.getelementptr inbounds|nuw %39[104] : (!llvm.ptr) -> !llvm.ptr, i8
    %162 = affine.load %40[32] : memref<?xi32>
    %163 = affine.load %40[30] : memref<?xi32>
    %164 = arith.ori %163, %c64_i32 : i32
    affine.store %164, %40[30] : memref<?xi32>
    %165 = "enzymexla.pointer2memref"(%0) : (!llvm.ptr) -> memref<?xi32>
    %166 = affine.load %165[8] : memref<?xi32>
    %167 = llvm.invoke @_ZNK4mfem6MemoryIdE4ReadENS_11MemoryClassEi(%161, %166, %162) to ^bb51 unwind ^bb108 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 28 : i64, llvm.nonnull, llvm.noundef}, i32 {llvm.noundef}, i32 {llvm.noundef}) -> (!llvm.ptr {llvm.noundef})
  ^bb51:  // pred: ^bb50
    %168 = llvm.getelementptr inbounds|nuw %arg0[32] : (!llvm.ptr) -> !llvm.ptr, i8
    %169 = affine.load %31[14] : memref<?xi32>
    %170 = affine.load %31[12] : memref<?xi32>
    %171 = arith.ori %170, %c64_i32 : i32
    affine.store %171, %31[12] : memref<?xi32>
    %172 = affine.load %165[8] : memref<?xi32>
    %173 = llvm.invoke @_ZNK4mfem6MemoryIiE4ReadENS_11MemoryClassEi(%168, %172, %169) to ^bb52 unwind ^bb109 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 28 : i64, llvm.nonnull, llvm.noundef}, i32 {llvm.noundef}, i32 {llvm.noundef}) -> (!llvm.ptr {llvm.noundef})
  ^bb52:  // pred: ^bb51
    %174 = affine.load %31[4] : memref<?xi32>
    %175 = arith.index_cast %174 : i32 to index
    %176 = arith.muli %174, %c6_i32 : i32
    %177 = arith.cmpi sgt, %176, %c0_i32 : i32
    %178 = llvm.getelementptr inbounds|nuw %arg0[96] : (!llvm.ptr) -> !llvm.ptr, i8
    %179 = affine.load %31[30] : memref<?xi32>
    %180 = affine.load %31[28] : memref<?xi32>
    %181 = arith.ori %180, %c64_i32 : i32
    affine.store %181, %31[28] : memref<?xi32>
    %182 = affine.load %165[8] : memref<?xi32>
    %183 = llvm.invoke @_ZNK4mfem6MemoryIiE4ReadENS_11MemoryClassEi(%178, %182, %179) to ^bb53 unwind ^bb110 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 28 : i64, llvm.nonnull, llvm.noundef}, i32 {llvm.noundef}, i32 {llvm.noundef}) -> (!llvm.ptr {llvm.noundef})
  ^bb53:  // pred: ^bb52
    %184 = affine.load %31[4] : memref<?xi32>
    %185 = arith.index_cast %184 : i32 to index
    %186 = "enzymexla.pointer2memref"(%arg1) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    %187 = affine.load %186[0] : memref<?x!llvm.ptr>
    %188 = "enzymexla.pointer2memref"(%187) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    %189 = affine.load %188[4] : memref<?x!llvm.ptr>
    %190 = llvm.invoke %189(%arg1, %true) to ^bb54 unwind ^bb111 : !llvm.ptr, (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 36 : i64, llvm.nonnull, llvm.noundef}, i1 {llvm.noundef, llvm.zeroext}) -> (!llvm.ptr {llvm.noundef})
  ^bb54:  // pred: ^bb53
    %191 = arith.select %30, %26, %c1_i32 {fastmathFlags = #llvm.fastmath<none>} : i32
    %192 = affine.load %31[5] : memref<?xi32>
    %193 = arith.select %30, %c1_i32, %192 {fastmathFlags = #llvm.fastmath<none>} : i32
    %194 = arith.muli %192, %26 : i32
    %195 = arith.cmpi sgt, %194, %c0_i32 : i32
    %196 = affine.load %48[8] : memref<?xi32>
    %197 = affine.load %48[6] : memref<?xi32>
    %198 = arith.ori %197, %c64_i32 : i32
    affine.store %198, %48[6] : memref<?xi32>
    %199 = llvm.getelementptr inbounds|nuw %18[8] : (!llvm.ptr) -> !llvm.ptr, i8
    %200 = affine.load %165[8] : memref<?xi32>
    %201 = llvm.invoke @_ZNK4mfem6MemoryIdE4ReadENS_11MemoryClassEi(%199, %200, %196) to ^bb55 unwind ^bb112 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 20 : i64, llvm.nonnull, llvm.noundef}, i32 {llvm.noundef}, i32 {llvm.noundef}) -> (!llvm.ptr {llvm.noundef})
  ^bb55:  // pred: ^bb54
    %202 = "enzymexla.pointer2memref"(%arg2) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    %203 = affine.load %202[0] : memref<?x!llvm.ptr>
    %204 = "enzymexla.pointer2memref"(%203) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    %205 = affine.load %204[6] : memref<?x!llvm.ptr>
    %206 = llvm.invoke %205(%arg2, %true) to ^bb56 unwind ^bb113 : !llvm.ptr, (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 36 : i64, llvm.nonnull, llvm.noundef}, i1 {llvm.noundef, llvm.zeroext}) -> (!llvm.ptr {llvm.noundef})
  ^bb56:  // pred: ^bb55
    %207 = arith.muli %53, %26 : i32
    %208 = arith.cmpi sgt, %207, %c0_i32 : i32
    %209 = arith.select %30, %c1_i32, %53 {fastmathFlags = #llvm.fastmath<none>} : i32
    %210 = affine.load %31[4] : memref<?xi32>
    %211 = "enzymexla.pointer2memref"(%0) : (!llvm.ptr) -> memref<?xi64>
    %212 = affine.load %211[1] : memref<?xi64>
    %213 = arith.andi %212, %c4_i64 : i64
    %214 = arith.cmpi eq, %213, %c0_i64 : i64
    cf.cond_br %214, ^bb57, ^bb58
  ^bb57:  // pred: ^bb56
    %215 = arith.cmpi sgt, %210, %c0_i32 : i32
    cf.cond_br %215, ^bb62(%c0_i32 : i32), ^bb99
  ^bb58:  // pred: ^bb56
    %216 = arith.cmpi eq, %210, %c0_i32 : i32
    cf.cond_br %216, ^bb99, ^bb59
  ^bb59:  // pred: ^bb58
    %217 = arith.index_cast %210 : i32 to index
    %218 = "enzymexla.gpu_wrapper"(%217, %c1, %c1, %c1, %c2, %c1) ({
      affine.parallel (%arg3) = (0) to (symbol(%217)) {
        %alloca = memref.alloca() alignment = 8 : memref<1xf64>
        %alloca_0 = memref.alloca() alignment = 4 : memref<2xi32>
        %alloca_1 = memref.alloca() alignment = 4 : memref<2xi32>
        %alloca_2 = memref.alloca() alignment = 4 : memref<2x1xi32>
        affine.parallel (%arg4) = (0) to (2) {
          affine.if #set(%arg4) {
            %488 = "enzymexla.pointer2memref"(%167) : (!llvm.ptr) -> memref<?xf64>
            %489 = affine.load %488[0] : memref<?xf64>
            %memspacecast_4 = memref.memory_space_cast %alloca : memref<1xf64> to memref<1xf64, 3>
            affine.store %489, %memspacecast_4[0] : memref<1xf64, 3>
          }
          %485 = "enzymexla.pointer2memref"(%173) : (!llvm.ptr) -> memref<?xi32>
          %486 = affine.if #set5()[%175] -> i32 {
            %488 = affine.load %485[%arg4 + %arg3 * 6] : memref<?xi32>
            affine.yield %488 : i32
          } else {
            affine.yield %c0_i32 : i32
          }
          %memspacecast = memref.memory_space_cast %alloca_0 : memref<2xi32> to memref<2xi32, 3>
          affine.store %486, %memspacecast[%arg4] : memref<2xi32, 3>
          %487 = affine.if #set5()[%175] -> i32 {
            %488 = affine.load %485[%arg4 + %arg3 * 6 + 2] : memref<?xi32>
            affine.yield %488 : i32
          } else {
            affine.yield %c0_i32 : i32
          }
          %memspacecast_3 = memref.memory_space_cast %alloca_1 : memref<2xi32> to memref<2xi32, 3>
          affine.store %487, %memspacecast_3[%arg4] : memref<2xi32, 3>
          affine.if #set3()[%44] {
            affine.parallel (%arg5) = (0) to (symbol(%44)) {
              %488 = "enzymexla.pointer2memref"(%183) : (!llvm.ptr) -> memref<?xi32>
              %489 = affine.if #set2()[%185] -> i32 {
                %490 = affine.load %488[%arg5 + (%arg4 + %arg3 * 2) * symbol(%44)] : memref<?xi32>
                affine.yield %490 : i32
              } else {
                affine.yield %c0_i32 : i32
              }
              affine.store %489, %alloca_2[%arg5 + %arg4, 0] : memref<2x1xi32>
            }
          }
          "enzymexla.barrier"(%c0, %arg4, %c0) : (index, index, index) -> ()
          affine.if #set3()[%44] {
            %488 = affine.load %memspacecast[%arg4] : memref<2xi32, 3>
            %489 = arith.cmpi slt, %488, %32 : i32
            %490 = arith.select %489, %c0_i32, %32 {fastmathFlags = #llvm.fastmath<none>} : i32
            %491 = arith.subi %488, %490 overflow<nsw> : i32
            %492 = affine.load %memspacecast_3[%arg4] : memref<2xi32, 3>
            %493 = arith.cmpi eq, %492, %c0_i32 : i32
            %494 = arith.cmpi eq, %492, %c5_i32 : i32
            %495 = arith.ori %493, %494 : i1
            %496 = arith.andi %492, %c-3_i32 : i32
            %497 = arith.cmpi eq, %496, %c1_i32 : i32
            %498 = arith.cmpi slt, %491, %c0_i32 : i32
            affine.for %arg5 = 0 to %44 {
              scf.if %498 {
                affine.for %arg6 = 0 to %27 {
                  %499 = "enzymexla.pointer2memref"(%206) : (!llvm.ptr) -> memref<?xf64>
                  affine.store %cst, %499[%arg5 + (%arg6 + (%arg4 + %arg3 * 2) * symbol(%27)) * symbol(%44)] : memref<?xf64>
                }
              } else {
                %499 = affine.load %alloca_2[%arg5 + %arg4, 0] : memref<2x1xi32>
                %500 = arith.divsi %499, %43 : i32
                %501 = arith.muli %500, %43 overflow<nsw> : i32
                %502 = arith.subi %499, %501 overflow<nsw> : i32
                %503 = arith.index_castui %492 : i32 to index
                %504 = arith.cmpi eq, %503, %c4 : index
                %505 = arith.cmpi eq, %503, %c2 : index
                %506 = arith.select %497, %502, %500 {fastmathFlags = #llvm.fastmath<none>} : i32
                %507 = arith.select %505, %c0_i32, %506 : i32
                %508 = arith.select %504, %c0_i32, %507 : i32
                affine.for %arg6 = 0 to %27 {
                  %509 = arith.index_cast %arg6 : index to i32
                  %510 = affine.if #set4()[%29] -> i32 {
                    affine.yield %491 : i32
                  } else {
                    affine.yield %509 : i32
                  }
                  %511 = arith.select %489, %193, %209 : i32
                  %512 = arith.muli %511, %510 overflow<nsw> : i32
                  %513 = arith.select %497, %c0_i32, %502 {fastmathFlags = #llvm.fastmath<none>} : i32
                  %514 = arith.select %495, %c0_i32, %500 {fastmathFlags = #llvm.fastmath<none>} : i32
                  %515 = arith.index_cast %508 : i32 to index
                  %516 = memref.load %alloca[%515] : memref<1xf64>
                  %517:4 = affine.if #set4()[%29] -> (i32, i32, i32, i32) {
                    affine.yield %509, %c0_i32, %513, %514 : i32, i32, i32, i32
                  } else {
                    affine.yield %c0_i32, %513, %514, %491 : i32, i32, i32, i32
                  }
                  %518 = arith.addi %517#3, %512 overflow<nsw> : i32
                  %519 = arith.addi %518, %517#2 overflow<nsw> : i32
                  %520 = arith.addi %519, %517#1 overflow<nsw> : i32
                  %521 = arith.muli %520, %191 overflow<nsw> : i32
                  %522 = arith.index_cast %521 : i32 to index
                  %523 = arith.index_cast %517#0 : i32 to index
                  %524 = arith.addi %522, %523 : index
                  %525 = scf.if %489 -> (f64) {
                    %529 = "enzymexla.pointer2memref"(%190) : (!llvm.ptr) -> memref<?xf64>
                    %530 = memref.load %529[%524] : memref<?xf64>
                    scf.yield %530 : f64
                  } else {
                    %529 = "enzymexla.pointer2memref"(%201) : (!llvm.ptr) -> memref<?xf64>
                    %530 = memref.load %529[%524] : memref<?xf64>
                    scf.yield %530 : f64
                  }
                  %526 = arith.mulf %516, %525 fastmath<contract> : f64
                  %527 = arith.addf %526, %cst fastmath<contract> : f64
                  %528 = "enzymexla.pointer2memref"(%206) : (!llvm.ptr) -> memref<?xf64>
                  affine.store %527, %528[%arg5 + (%arg6 + (%arg4 + %arg3 * 2) * symbol(%27)) * symbol(%44)] : memref<?xf64>
                }
              }
            }
          }
        }
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) {passthrough = ["mustprogress", "norecurse", ["no-trapping-math", "true"], ["polygeist.host_symbol", "_ZN4mfemL25__device_stub__CuKernel2DIZNKS_33L2NormalDerivativeFaceRestriction6Mult3DILi1EEEvRKNS_6VectorERS3_EUliE_EEviT_"], ["stack-protector-buffer-size", "8"], ["target-cpu", "sm_120"], "uniform-work-group-size"], target_cpu = "sm_120", target_features = #llvm.target_features<["+ptx88", "+sm_120"]>} : (index, index, index, index, index, index) -> index
    %219 = llvm.invoke @cudaGetLastError() to ^bb60 unwind ^bb114 : () -> i32
  ^bb60:  // pred: ^bb59
    %220 = arith.cmpi eq, %219, %c0_i32 : i32
    cf.cond_br %220, ^bb99, ^bb61
  ^bb61:  // pred: ^bb60
    llvm.invoke @_ZN4mfem15mfem_cuda_errorE9cudaErrorPKcS2_S2_i(%219, %11, %12, %13, %c785_i32) to ^bb99 unwind ^bb114 : (i32 {llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, i32 {llvm.noundef}) -> ()
  ^bb62(%221: i32):  // 2 preds: ^bb57, ^bb98
    %222 = arith.index_cast %221 : i32 to index
    llvm.intr.lifetime.start %17 : !llvm.ptr
    cf.br ^bb64(%c0_i32 : i32)
  ^bb63:  // pred: ^bb64
    %223 = arith.index_cast %240 : i32 to index
    %224 = "enzymexla.pointer2memref"(%167) : (!llvm.ptr) -> memref<?xf64>
    %225 = affine.load %224[symbol(%223)] : memref<?xf64>
    %226 = arith.muli %43, %c2_i32 : i32
    %227 = arith.muli %226, %221 : i32
    %228 = arith.extui %43 nneg : i32 to i64
    %229 = arith.shli %228, %c2_i64 overflow<nsw, nuw> : i64
    %230 = "enzymexla.pointer2memref"(%173) : (!llvm.ptr) -> memref<?xi32>
    %231:4 = scf.if %177 -> (i32, i32, i32, i32) {
      %485 = affine.load %230[symbol(%222) * 6] : memref<?xi32>
      %486 = affine.load %230[symbol(%222) * 6 + 1] : memref<?xi32>
      %487 = affine.load %230[symbol(%222) * 6 + 2] : memref<?xi32>
      %488 = affine.load %230[symbol(%222) * 6 + 3] : memref<?xi32>
      scf.yield %485, %486, %487, %488 : i32, i32, i32, i32
    } else {
      scf.yield %c0_i32, %c0_i32, %c0_i32, %c0_i32 : i32, i32, i32, i32
    }
    %232 = arith.extsi %227 : i32 to i64
    %233 = arith.shli %232, %c2_i64 overflow<nsw> : i64
    %234 = llvm.getelementptr %183[%233] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    "llvm.intr.memcpy"(%17, %234, %229) <{arg_attrs = [{llvm.align = 4 : i64, llvm.nonnull}, {llvm.align = 4 : i64}, {}], isVolatile = false, tbaa = [#tbaa_tag]}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %235 = llvm.getelementptr inbounds|nuw %17[4] : (!llvm.ptr) -> !llvm.ptr, i8
    %236 = arith.addi %227, %43 : i32
    %237 = arith.extsi %236 : i32 to i64
    %238 = arith.shli %237, %c2_i64 overflow<nsw> : i64
    %239 = llvm.getelementptr %183[%238] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    "llvm.intr.memcpy"(%235, %239, %229) <{arg_attrs = [{llvm.align = 4 : i64, llvm.nonnull}, {llvm.align = 4 : i64}, {}], isVolatile = false, tbaa = [#tbaa_tag]}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    cf.br ^bb65(%true, %17, %231#2, %231#0, %c0_i64 : i1, !llvm.ptr, i32, i32, i64)
  ^bb64(%240: i32):  // 2 preds: ^bb62, ^bb64
    %241 = arith.addi %240, %c1_i32 overflow<nsw, nuw> : i32
    %242 = arith.cmpi eq, %241, %c1_i32 : i32
    cf.cond_br %242, ^bb63, ^bb64(%241 : i32)
  ^bb65(%243: i1, %244: !llvm.ptr, %245: i32, %246: i32, %247: i64):  // 2 preds: ^bb63, ^bb67
    %248 = arith.cmpi slt, %246, %32 : i32
    %249 = arith.select %248, %c0_i32, %32 {fastmathFlags = #llvm.fastmath<none>} : i32
    %250 = arith.subi %246, %249 overflow<nsw> : i32
    %251 = arith.cmpi eq, %245, %c0_i32 : i32
    %252 = arith.cmpi eq, %245, %c5_i32 : i32
    %253 = arith.ori %251, %252 : i1
    %254 = arith.andi %245, %c-3_i32 : i32
    %255 = arith.cmpi eq, %254, %c1_i32 : i32
    %256 = arith.cmpi slt, %250, %c0_i32 : i32
    %257 = arith.cmpi sgt, %26, %c0_i32 : i32
    cf.cond_br %256, ^bb68, ^bb66
  ^bb66:  // pred: ^bb65
    cf.cond_br %257, ^bb72, ^bb67
  ^bb67:  // 4 preds: ^bb66, ^bb68, ^bb71, ^bb74
    cf.cond_br %243, ^bb65(%false, %55, %231#3, %231#1, %c1_i64 : i1, !llvm.ptr, i32, i32, i64), ^bb98
  ^bb68:  // pred: ^bb65
    cf.cond_br %257, ^bb97, ^bb67
  ^bb69(%258: i64):  // 2 preds: ^bb71, ^bb97
    %259 = arith.index_cast %258 : i64 to index
    cf.br ^bb70(%c0_i64 : i64)
  ^bb70(%260: i64):  // 2 preds: ^bb69, ^bb70
    %261 = arith.addi %260, %456 overflow<nsw> : i64
    %262 = arith.muli %261, %457 overflow<nsw> : i64
    %263 = arith.index_cast %262 : i64 to index
    %264 = "enzymexla.pointer2memref"(%206) : (!llvm.ptr) -> memref<?xf64>
    %265 = arith.addi %263, %259 : index
    affine.store %cst, %264[symbol(%265)] : memref<?xf64>
    %266 = arith.addi %260, %c1_i64 overflow<nsw, nuw> : i64
    %267 = arith.cmpi eq, %266, %458 : i64
    cf.cond_br %267, ^bb71, ^bb70(%266 : i64)
  ^bb71:  // pred: ^bb70
    %268 = arith.addi %258, %c1_i64 overflow<nsw, nuw> : i64
    %269 = arith.cmpi eq, %268, %228 : i64
    cf.cond_br %269, ^bb67, ^bb69(%268 : i64)
  ^bb72:  // pred: ^bb66
    %270 = arith.muli %221, %c2_i32 overflow<nsw> : i32
    %271 = arith.trunci %247 : i64 to i32
    %272 = arith.addi %270, %271 overflow<nsw> : i32
    %273 = arith.muli %272, %26 overflow<nsw> : i32
    %274 = arith.extsi %273 : i32 to i64
    %275 = arith.extsi %43 : i32 to i64
    %276 = arith.extui %26 nneg : i32 to i64
    cf.br ^bb73(%c0_i64 : i64)
  ^bb73(%277: i64):  // 2 preds: ^bb72, ^bb74
    %278 = arith.index_cast %277 : i64 to index
    %279 = "enzymexla.pointer2memref"(%244) : (!llvm.ptr) -> memref<?xi32>
    %280 = affine.load %279[symbol(%278)] : memref<?xi32>
    %281 = arith.divsi %280, %43 : i32
    %282 = arith.muli %281, %43 overflow<nsw> : i32
    %283 = arith.subi %280, %282 overflow<nsw> : i32
    %284 = arith.select %248, %193, %209 {fastmathFlags = #llvm.fastmath<none>} : i32
    %285 = arith.select %30, %c0_i32, %283 {fastmathFlags = #llvm.fastmath<none>} : i32
    %286 = arith.select %30, %283, %281 {fastmathFlags = #llvm.fastmath<none>} : i32
    %287 = arith.select %30, %281, %250 {fastmathFlags = #llvm.fastmath<none>} : i32
    cf.switch %245 : i32, [
      default: ^bb75(%c0_i64 : i64),
      4: ^bb80,
      2: ^bb80
    ]
  ^bb74:  // 3 preds: ^bb78, ^bb85, ^bb91
    %288 = arith.addi %277, %c1_i64 overflow<nsw, nuw> : i64
    %289 = arith.cmpi eq, %288, %228 : i64
    cf.cond_br %289, ^bb67, ^bb73(%288 : i64)
  ^bb75(%290: i64):  // 2 preds: ^bb73, ^bb78
    %291 = arith.trunci %290 : i64 to i32
    %292 = arith.select %30, %250, %291 {fastmathFlags = #llvm.fastmath<none>} : i32
    %293 = arith.muli %284, %292 overflow<nsw> : i32
    %294 = arith.select %30, %291, %c0_i32 {fastmathFlags = #llvm.fastmath<none>} : i32
    cf.cond_br %255, ^bb79(%c0_i32, %cst : i32, f64), ^bb77
  ^bb76(%295: i32, %296: f64):  // 2 preds: ^bb76, ^bb95
    %297 = enzymexla.math.fmuladd %225, %434, %296 : f64
    %298 = arith.addi %295, %c1_i32 overflow<nsw, nuw> : i32
    %299 = arith.cmpi eq, %298, %c1_i32 : i32
    cf.cond_br %299, ^bb78(%297 : f64), ^bb76(%298, %297 : i32, f64)
  ^bb77:  // pred: ^bb75
    cf.switch %245 : i32, [
      default: ^bb95,
      5: ^bb96(%c0_i32, %cst : i32, f64),
      0: ^bb96(%c0_i32, %cst : i32, f64)
    ]
  ^bb78(%300: f64):  // 3 preds: ^bb76, ^bb79, ^bb96
    %301 = arith.addi %290, %274 overflow<nsw> : i64
    %302 = arith.muli %301, %275 overflow<nsw> : i64
    %303 = arith.index_cast %302 : i64 to index
    %304 = "enzymexla.pointer2memref"(%206) : (!llvm.ptr) -> memref<?xf64>
    %305 = arith.addi %303, %278 : index
    affine.store %300, %304[symbol(%305)] : memref<?xf64>
    %306 = arith.addi %290, %c1_i64 overflow<nsw, nuw> : i64
    %307 = arith.cmpi eq, %306, %276 : i64
    cf.cond_br %307, ^bb74, ^bb75(%306 : i64)
  ^bb79(%308: i32, %309: f64):  // 2 preds: ^bb75, ^bb79
    %310 = arith.select %253, %308, %281 {fastmathFlags = #llvm.fastmath<none>} : i32
    %311 = arith.select %30, %c0_i32, %308 {fastmathFlags = #llvm.fastmath<none>} : i32
    %312 = arith.select %30, %308, %310 {fastmathFlags = #llvm.fastmath<none>} : i32
    %313 = arith.select %30, %310, %250 {fastmathFlags = #llvm.fastmath<none>} : i32
    %314 = arith.addi %293, %313 overflow<nsw> : i32
    %315 = arith.addi %314, %312 overflow<nsw> : i32
    %316 = arith.addi %315, %311 overflow<nsw> : i32
    %317 = arith.muli %316, %191 overflow<nsw> : i32
    %318 = arith.index_cast %317 : i32 to index
    %319 = arith.index_cast %294 : i32 to index
    %320 = arith.addi %318, %319 : index
    %321 = "enzymexla.pointer2memref"(%190) : (!llvm.ptr) -> memref<?xf64>
    %322 = "enzymexla.pointer2memref"(%201) : (!llvm.ptr) -> memref<?xf64>
    %323 = scf.if %248 -> (f64) {
      %485 = scf.if %195 -> (f64) {
        %486 = affine.load %321[symbol(%320)] : memref<?xf64>
        scf.yield %486 : f64
      } else {
        scf.yield %cst : f64
      }
      scf.yield %485 : f64
    } else {
      %485 = scf.if %208 -> (f64) {
        %486 = affine.load %322[symbol(%320)] : memref<?xf64>
        scf.yield %486 : f64
      } else {
        scf.yield %cst : f64
      }
      scf.yield %485 : f64
    }
    %324 = enzymexla.math.fmuladd %225, %323, %309 : f64
    %325 = arith.addi %308, %c1_i32 overflow<nsw, nuw> : i32
    %326 = arith.cmpi eq, %325, %c1_i32 : i32
    cf.cond_br %326, ^bb78(%324 : f64), ^bb79(%325, %324 : i32, f64)
  ^bb80:  // 2 preds: ^bb73, ^bb73
    cf.cond_br %255, ^bb88(%c0_i64 : i64), ^bb87
  ^bb81(%327: i64):  // 2 preds: ^bb85, ^bb87
    %328 = arith.index_cast %327 : i64 to index
    %329 = arith.trunci %327 : i64 to i32
    %330 = arith.select %30, %250, %329 {fastmathFlags = #llvm.fastmath<none>} : i32
    %331 = arith.muli %284, %330 overflow<nsw> : i32
    %332 = arith.addi %331, %287 overflow<nsw> : i32
    %333 = arith.addi %332, %286 overflow<nsw> : i32
    cf.cond_br %30, ^bb82, ^bb84
  ^bb82:  // pred: ^bb81
    %334 = arith.extsi %333 : i32 to i64
    cf.br ^bb86(%c0_i64, %cst : i64, f64)
  ^bb83(%335: i64, %336: f64):  // 2 preds: ^bb83, ^bb84
    %337 = arith.index_cast %335 : i64 to index
    %338 = "enzymexla.pointer2memref"(%190) : (!llvm.ptr) -> memref<?xf64>
    %339 = "enzymexla.pointer2memref"(%201) : (!llvm.ptr) -> memref<?xf64>
    %340 = scf.if %248 -> (f64) {
      %485 = scf.if %195 -> (f64) {
        %486 = arith.addi %337, %346 : index
        %487 = affine.load %338[symbol(%486)] : memref<?xf64>
        scf.yield %487 : f64
      } else {
        scf.yield %cst : f64
      }
      scf.yield %485 : f64
    } else {
      %485 = scf.if %208 -> (f64) {
        %486 = arith.addi %337, %346 : index
        %487 = affine.load %339[symbol(%486)] : memref<?xf64>
        scf.yield %487 : f64
      } else {
        scf.yield %cst : f64
      }
      scf.yield %485 : f64
    }
    %341 = enzymexla.math.fmuladd %225, %340, %336 : f64
    %342 = arith.addi %335, %c1_i64 overflow<nsw, nuw> : i64
    %343 = arith.cmpi eq, %342, %c1_i64 : i64
    cf.cond_br %343, ^bb85(%341 : f64), ^bb83(%342, %341 : i64, f64)
  ^bb84:  // pred: ^bb81
    %344 = arith.addi %333, %283 overflow<nsw> : i32
    %345 = arith.muli %344, %191 overflow<nsw> : i32
    %346 = arith.index_cast %345 : i32 to index
    cf.br ^bb83(%c0_i64, %cst : i64, f64)
  ^bb85(%347: f64):  // 2 preds: ^bb83, ^bb86
    %348 = arith.addi %327, %274 overflow<nsw> : i64
    %349 = arith.muli %348, %275 overflow<nsw> : i64
    %350 = arith.index_cast %349 : i64 to index
    %351 = "enzymexla.pointer2memref"(%206) : (!llvm.ptr) -> memref<?xf64>
    %352 = arith.addi %350, %278 : index
    affine.store %347, %351[symbol(%352)] : memref<?xf64>
    %353 = arith.addi %327, %c1_i64 overflow<nsw, nuw> : i64
    %354 = arith.cmpi eq, %353, %276 : i64
    cf.cond_br %354, ^bb74, ^bb81(%353 : i64)
  ^bb86(%355: i64, %356: f64):  // 2 preds: ^bb82, ^bb86
    %357 = arith.addi %355, %334 overflow<nsw> : i64
    %358 = arith.muli %357, %366 overflow<nsw> : i64
    %359 = arith.index_cast %358 : i64 to index
    %360 = "enzymexla.pointer2memref"(%190) : (!llvm.ptr) -> memref<?xf64>
    %361 = "enzymexla.pointer2memref"(%201) : (!llvm.ptr) -> memref<?xf64>
    %362 = scf.if %248 -> (f64) {
      %485 = scf.if %195 -> (f64) {
        %486 = arith.addi %359, %328 : index
        %487 = affine.load %360[symbol(%486)] : memref<?xf64>
        scf.yield %487 : f64
      } else {
        scf.yield %cst : f64
      }
      scf.yield %485 : f64
    } else {
      %485 = scf.if %208 -> (f64) {
        %486 = arith.addi %359, %328 : index
        %487 = affine.load %361[symbol(%486)] : memref<?xf64>
        scf.yield %487 : f64
      } else {
        scf.yield %cst : f64
      }
      scf.yield %485 : f64
    }
    %363 = enzymexla.math.fmuladd %225, %362, %356 : f64
    %364 = arith.addi %355, %c1_i64 overflow<nsw, nuw> : i64
    %365 = arith.cmpi eq, %364, %c1_i64 : i64
    cf.cond_br %365, ^bb85(%363 : f64), ^bb86(%364, %363 : i64, f64)
  ^bb87:  // pred: ^bb80
    %366 = arith.extsi %191 : i32 to i64
    cf.br ^bb81(%c0_i64 : i64)
  ^bb88(%367: i64):  // 2 preds: ^bb80, ^bb91
    %368 = arith.index_cast %367 : i64 to index
    %369 = arith.trunci %367 : i64 to i32
    %370 = arith.select %30, %250, %369 {fastmathFlags = #llvm.fastmath<none>} : i32
    %371 = arith.muli %284, %370 overflow<nsw> : i32
    cf.switch %245 : i32, [
      default: ^bb90,
      5: ^bb92(%c0_i32, %cst : i32, f64),
      0: ^bb92(%c0_i32, %cst : i32, f64)
    ]
  ^bb89(%372: i64, %373: f64):  // 2 preds: ^bb89, ^bb93
    %374 = arith.index_cast %372 : i64 to index
    %375 = arith.trunci %372 : i64 to i32
    %376 = arith.addi %411, %375 overflow<nsw> : i32
    %377 = arith.muli %376, %191 overflow<nsw> : i32
    %378 = arith.index_cast %377 : i32 to index
    %379 = "enzymexla.pointer2memref"(%190) : (!llvm.ptr) -> memref<?xf64>
    %380 = "enzymexla.pointer2memref"(%201) : (!llvm.ptr) -> memref<?xf64>
    %381 = scf.if %248 -> (f64) {
      %485 = scf.if %195 -> (f64) {
        %486 = arith.addi %378, %374 : index
        %487 = affine.load %379[symbol(%486)] : memref<?xf64>
        scf.yield %487 : f64
      } else {
        scf.yield %cst : f64
      }
      scf.yield %485 : f64
    } else {
      %485 = scf.if %208 -> (f64) {
        %486 = arith.addi %378, %374 : index
        %487 = affine.load %380[symbol(%486)] : memref<?xf64>
        scf.yield %487 : f64
      } else {
        scf.yield %cst : f64
      }
      scf.yield %485 : f64
    }
    %382 = enzymexla.math.fmuladd %225, %381, %373 : f64
    %383 = arith.addi %372, %c1_i64 overflow<nsw, nuw> : i64
    %384 = arith.cmpi eq, %383, %c1_i64 : i64
    cf.cond_br %384, ^bb91(%382 : f64), ^bb89(%383, %382 : i64, f64)
  ^bb90:  // pred: ^bb88
    %385 = arith.addi %371, %287 overflow<nsw> : i32
    cf.cond_br %30, ^bb94(%c0_i64, %cst : i64, f64), ^bb93
  ^bb91(%386: f64):  // 3 preds: ^bb89, ^bb92, ^bb94
    %387 = arith.addi %367, %274 overflow<nsw> : i64
    %388 = arith.muli %387, %275 overflow<nsw> : i64
    %389 = arith.index_cast %388 : i64 to index
    %390 = "enzymexla.pointer2memref"(%206) : (!llvm.ptr) -> memref<?xf64>
    %391 = arith.addi %389, %278 : index
    affine.store %386, %390[symbol(%391)] : memref<?xf64>
    %392 = arith.addi %367, %c1_i64 overflow<nsw, nuw> : i64
    %393 = arith.cmpi eq, %392, %276 : i64
    cf.cond_br %393, ^bb74, ^bb88(%392 : i64)
  ^bb92(%394: i32, %395: f64):  // 3 preds: ^bb88, ^bb88, ^bb92
    %396 = arith.select %30, %369, %394 {fastmathFlags = #llvm.fastmath<none>} : i32
    %397 = arith.select %30, %394, %250 {fastmathFlags = #llvm.fastmath<none>} : i32
    %398 = arith.addi %371, %397 overflow<nsw> : i32
    %399 = arith.addi %398, %394 overflow<nsw> : i32
    %400 = arith.addi %399, %394 overflow<nsw> : i32
    %401 = arith.muli %400, %191 overflow<nsw> : i32
    %402 = arith.index_cast %401 : i32 to index
    %403 = arith.index_cast %396 : i32 to index
    %404 = arith.addi %402, %403 : index
    %405 = "enzymexla.pointer2memref"(%190) : (!llvm.ptr) -> memref<?xf64>
    %406 = "enzymexla.pointer2memref"(%201) : (!llvm.ptr) -> memref<?xf64>
    %407 = scf.if %248 -> (f64) {
      %485 = scf.if %195 -> (f64) {
        %486 = affine.load %405[symbol(%404)] : memref<?xf64>
        scf.yield %486 : f64
      } else {
        scf.yield %cst : f64
      }
      scf.yield %485 : f64
    } else {
      %485 = scf.if %208 -> (f64) {
        %486 = affine.load %406[symbol(%404)] : memref<?xf64>
        scf.yield %486 : f64
      } else {
        scf.yield %cst : f64
      }
      scf.yield %485 : f64
    }
    %408 = enzymexla.math.fmuladd %225, %407, %395 : f64
    %409 = arith.addi %394, %c1_i32 overflow<nsw, nuw> : i32
    %410 = arith.cmpi eq, %409, %c1_i32 : i32
    cf.cond_br %410, ^bb91(%408 : f64), ^bb92(%409, %408 : i32, f64)
  ^bb93:  // pred: ^bb90
    %411 = arith.addi %385, %281 overflow<nsw> : i32
    cf.br ^bb89(%c0_i64, %cst : i64, f64)
  ^bb94(%412: i64, %413: f64):  // 2 preds: ^bb90, ^bb94
    %414 = arith.trunci %412 : i64 to i32
    %415 = arith.addi %385, %414 overflow<nsw> : i32
    %416 = arith.addi %415, %414 overflow<nsw> : i32
    %417 = arith.muli %416, %191 overflow<nsw> : i32
    %418 = arith.index_cast %417 : i32 to index
    %419 = "enzymexla.pointer2memref"(%190) : (!llvm.ptr) -> memref<?xf64>
    %420 = "enzymexla.pointer2memref"(%201) : (!llvm.ptr) -> memref<?xf64>
    %421 = scf.if %248 -> (f64) {
      %485 = scf.if %195 -> (f64) {
        %486 = arith.addi %418, %368 : index
        %487 = affine.load %419[symbol(%486)] : memref<?xf64>
        scf.yield %487 : f64
      } else {
        scf.yield %cst : f64
      }
      scf.yield %485 : f64
    } else {
      %485 = scf.if %208 -> (f64) {
        %486 = arith.addi %418, %368 : index
        %487 = affine.load %420[symbol(%486)] : memref<?xf64>
        scf.yield %487 : f64
      } else {
        scf.yield %cst : f64
      }
      scf.yield %485 : f64
    }
    %422 = enzymexla.math.fmuladd %225, %421, %413 : f64
    %423 = arith.addi %412, %c1_i64 overflow<nsw, nuw> : i64
    %424 = arith.cmpi eq, %423, %c1_i64 : i64
    cf.cond_br %424, ^bb91(%422 : f64), ^bb94(%423, %422 : i64, f64)
  ^bb95:  // pred: ^bb77
    %425 = arith.addi %293, %287 overflow<nsw> : i32
    %426 = arith.addi %425, %286 overflow<nsw> : i32
    %427 = arith.addi %426, %285 overflow<nsw> : i32
    %428 = arith.muli %427, %191 overflow<nsw> : i32
    %429 = arith.index_cast %428 : i32 to index
    %430 = arith.index_cast %294 : i32 to index
    %431 = arith.addi %429, %430 : index
    %432 = "enzymexla.pointer2memref"(%190) : (!llvm.ptr) -> memref<?xf64>
    %433 = "enzymexla.pointer2memref"(%201) : (!llvm.ptr) -> memref<?xf64>
    %434 = scf.if %248 -> (f64) {
      %485 = scf.if %195 -> (f64) {
        %486 = affine.load %432[symbol(%431)] : memref<?xf64>
        scf.yield %486 : f64
      } else {
        scf.yield %cst : f64
      }
      scf.yield %485 : f64
    } else {
      %485 = scf.if %208 -> (f64) {
        %486 = affine.load %433[symbol(%431)] : memref<?xf64>
        scf.yield %486 : f64
      } else {
        scf.yield %cst : f64
      }
      scf.yield %485 : f64
    }
    cf.br ^bb76(%c0_i32, %cst : i32, f64)
  ^bb96(%435: i32, %436: f64):  // 3 preds: ^bb77, ^bb77, ^bb96
    %437 = arith.select %30, %283, %435 {fastmathFlags = #llvm.fastmath<none>} : i32
    %438 = arith.select %30, %435, %250 {fastmathFlags = #llvm.fastmath<none>} : i32
    %439 = arith.addi %293, %438 overflow<nsw> : i32
    %440 = arith.addi %439, %437 overflow<nsw> : i32
    %441 = arith.addi %440, %285 overflow<nsw> : i32
    %442 = arith.muli %441, %191 overflow<nsw> : i32
    %443 = arith.index_cast %442 : i32 to index
    %444 = arith.index_cast %294 : i32 to index
    %445 = arith.addi %443, %444 : index
    %446 = "enzymexla.pointer2memref"(%190) : (!llvm.ptr) -> memref<?xf64>
    %447 = "enzymexla.pointer2memref"(%201) : (!llvm.ptr) -> memref<?xf64>
    %448 = scf.if %248 -> (f64) {
      %485 = scf.if %195 -> (f64) {
        %486 = affine.load %446[symbol(%445)] : memref<?xf64>
        scf.yield %486 : f64
      } else {
        scf.yield %cst : f64
      }
      scf.yield %485 : f64
    } else {
      %485 = scf.if %208 -> (f64) {
        %486 = affine.load %447[symbol(%445)] : memref<?xf64>
        scf.yield %486 : f64
      } else {
        scf.yield %cst : f64
      }
      scf.yield %485 : f64
    }
    %449 = enzymexla.math.fmuladd %225, %448, %436 : f64
    %450 = arith.addi %435, %c1_i32 overflow<nsw, nuw> : i32
    %451 = arith.cmpi eq, %450, %c1_i32 : i32
    cf.cond_br %451, ^bb78(%449 : f64), ^bb96(%450, %449 : i32, f64)
  ^bb97:  // pred: ^bb68
    %452 = arith.muli %221, %c2_i32 overflow<nsw> : i32
    %453 = arith.trunci %247 : i64 to i32
    %454 = arith.addi %452, %453 overflow<nsw> : i32
    %455 = arith.muli %454, %26 overflow<nsw> : i32
    %456 = arith.extsi %455 : i32 to i64
    %457 = arith.extsi %43 : i32 to i64
    %458 = arith.extui %26 nneg : i32 to i64
    cf.br ^bb69(%c0_i64 : i64)
  ^bb98:  // pred: ^bb67
    llvm.intr.lifetime.end %17 : !llvm.ptr
    %459 = arith.addi %221, %c1_i32 overflow<nsw, nuw> : i32
    %460 = arith.cmpi eq, %459, %210 : i32
    cf.cond_br %460, ^bb99, ^bb62(%459 : i32)
  ^bb99:  // 5 preds: ^bb57, ^bb58, ^bb60, ^bb61, ^bb98
    %461 = "enzymexla.pointer2memref"(%18) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    affine.store %16, %461[0] : memref<?x!llvm.ptr>
    %462 = affine.load %48[6] : memref<?xi32>
    %463 = arith.andi %462, %c1_i32 : i32
    %464 = arith.cmpi eq, %463, %c0_i32 : i32
    %465 = affine.load %48[5] : memref<?xi32>
    %466 = arith.cmpi eq, %465, %c0_i32 : i32
    %467 = arith.andi %464, %466 : i1
    cf.cond_br %467, ^bb103(%462 : i32), ^bb100
  ^bb100:  // pred: ^bb99
    %468 = affine.load %461[1] : memref<?x!llvm.ptr>
    llvm.invoke @_ZN4mfem13MemoryManager7Delete_EPvNS_10MemoryTypeEj(%468, %465, %462) to ^bb101 unwind ^bb106 : (!llvm.ptr {llvm.noundef}, i32 {llvm.noundef}, i32 {llvm.noundef}) -> ()
  ^bb101:  // pred: ^bb100
    cf.cond_br %466, ^bb102, ^bb107
  ^bb102:  // pred: ^bb101
    %469 = affine.load %48[6] : memref<?xi32>
    cf.br ^bb103(%469 : i32)
  ^bb103(%470: i32):  // 2 preds: ^bb99, ^bb102
    %471 = arith.andi %470, %c2_i32 : i32
    %472 = arith.cmpi eq, %471, %c0_i32 : i32
    cf.cond_br %472, ^bb107, ^bb104
  ^bb104:  // pred: ^bb103
    %473 = affine.load %461[1] : memref<?x!llvm.ptr>
    %474 = llvm.icmp "eq" %473, %10 : !llvm.ptr
    cf.cond_br %474, ^bb107, ^bb105
  ^bb105:  // pred: ^bb104
    llvm.call @_ZdaPv(%473) {builtin, no_unwind} : (!llvm.ptr {llvm.nonnull, llvm.noundef}) -> ()
    cf.br ^bb107
  ^bb106:  // pred: ^bb100
    %475 = llvm.landingpad (catch %10 : !llvm.ptr) : !llvm.struct<(ptr, i32)>
    %476 = llvm.extractvalue %475[0] : !llvm.struct<(ptr, i32)> 
    llvm.call @__clang_call_terminate(%476) {no_unwind, noreturn} : (!llvm.ptr) -> ()
    llvm.unreachable
  ^bb107:  // 4 preds: ^bb101, ^bb103, ^bb104, ^bb105
    llvm.intr.lifetime.end %18 : !llvm.ptr
    llvm.return
  ^bb108:  // pred: ^bb50
    %477 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    cf.br ^bb115(%477 : !llvm.struct<(ptr, i32)>)
  ^bb109:  // pred: ^bb51
    %478 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    cf.br ^bb115(%478 : !llvm.struct<(ptr, i32)>)
  ^bb110:  // pred: ^bb52
    %479 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    cf.br ^bb115(%479 : !llvm.struct<(ptr, i32)>)
  ^bb111:  // pred: ^bb53
    %480 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    cf.br ^bb115(%480 : !llvm.struct<(ptr, i32)>)
  ^bb112:  // pred: ^bb54
    %481 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    cf.br ^bb115(%481 : !llvm.struct<(ptr, i32)>)
  ^bb113:  // pred: ^bb55
    %482 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    cf.br ^bb115(%482 : !llvm.struct<(ptr, i32)>)
  ^bb114:  // 2 preds: ^bb59, ^bb61
    %483 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    cf.br ^bb115(%483 : !llvm.struct<(ptr, i32)>)
  ^bb115(%484: !llvm.struct<(ptr, i32)>):  // 9 preds: ^bb24, ^bb49, ^bb108, ^bb109, ^bb110, ^bb111, ^bb112, ^bb113, ^bb114
    llvm.call @_ZN4mfem6VectorD2Ev(%18) {no_unwind} : (!llvm.ptr {llvm.align = 8 : i64, llvm.dead_on_return = 36 : i64, llvm.dereferenceable = 36 : i64, llvm.nonnull, llvm.noundef}) -> ()
    llvm.intr.lifetime.end %18 : !llvm.ptr
    llvm.resume %484 : !llvm.struct<(ptr, i32)>
  }
  llvm.func unnamed_addr @_ZNK4mfem6Vector9UseDeviceEb(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 36 : i64, llvm.nonnull, llvm.noundef}, %arg1: i1 {llvm.noundef, llvm.zeroext}) attributes {alignment = 2 : i64, dso_local, no_unwind, passthrough = ["mustprogress", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], "uniform-work-group-size"], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uwtable_kind = #llvm.uwtableKind<async>}
  llvm.func unnamed_addr @_ZNK4mfem6Vector9UseDeviceEv(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 36 : i64, llvm.nonnull, llvm.noundef}) -> (i1 {llvm.noundef, llvm.zeroext}) attributes {alignment = 2 : i64, dso_local, no_unwind, passthrough = ["mustprogress", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], "uniform-work-group-size"], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uwtable_kind = #llvm.uwtableKind<async>}
  llvm.func unnamed_addr @_ZNK4mfem6Vector4ReadEb(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 36 : i64, llvm.nonnull, llvm.noundef}, %arg1: i1 {llvm.noundef, llvm.zeroext}) -> (!llvm.ptr {llvm.noundef}) attributes {alignment = 2 : i64, dso_local, passthrough = ["mustprogress", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], "uniform-work-group-size"], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uwtable_kind = #llvm.uwtableKind<async>}
  llvm.func unnamed_addr @_ZNK4mfem6Vector8HostReadEv(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 36 : i64, llvm.nonnull, llvm.noundef}) -> (!llvm.ptr {llvm.noundef}) attributes {alignment = 2 : i64, dso_local, passthrough = ["mustprogress", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], "uniform-work-group-size"], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uwtable_kind = #llvm.uwtableKind<async>}
  llvm.func unnamed_addr @_ZN4mfem6Vector5WriteEb(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 36 : i64, llvm.nonnull, llvm.noundef}, %arg1: i1 {llvm.noundef, llvm.zeroext}) -> (!llvm.ptr {llvm.noundef}) attributes {alignment = 2 : i64, dso_local, passthrough = ["mustprogress", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], "uniform-work-group-size"], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uwtable_kind = #llvm.uwtableKind<async>}
  llvm.func unnamed_addr @_ZN4mfem6Vector9HostWriteEv(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 36 : i64, llvm.nonnull, llvm.noundef}) -> (!llvm.ptr {llvm.noundef}) attributes {alignment = 2 : i64, dso_local, passthrough = ["mustprogress", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], "uniform-work-group-size"], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uwtable_kind = #llvm.uwtableKind<async>}
  llvm.func unnamed_addr @_ZN4mfem6Vector9ReadWriteEb(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 36 : i64, llvm.nonnull, llvm.noundef}, %arg1: i1 {llvm.noundef, llvm.zeroext}) -> (!llvm.ptr {llvm.noundef}) attributes {alignment = 2 : i64, dso_local, passthrough = ["mustprogress", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], "uniform-work-group-size"], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uwtable_kind = #llvm.uwtableKind<async>}
  llvm.func unnamed_addr @_ZN4mfem6Vector13HostReadWriteEv(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 36 : i64, llvm.nonnull, llvm.noundef}) -> (!llvm.ptr {llvm.noundef}) attributes {alignment = 2 : i64, dso_local, passthrough = ["mustprogress", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], "uniform-work-group-size"], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uwtable_kind = #llvm.uwtableKind<async>}
  llvm.func local_unnamed_addr @_ZNK4mfem6MemoryIiE4ReadENS_11MemoryClassEi(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 20 : i64, llvm.nonnull, llvm.noundef}, %arg1: i32 {llvm.noundef}, %arg2: i32 {llvm.noundef}) -> (!llvm.ptr {llvm.noundef}) attributes {alignment = 2 : i64, dso_local, no_inline, passthrough = ["mustprogress", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], "uniform-work-group-size"], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uwtable_kind = #llvm.uwtableKind<async>}
  llvm.func local_unnamed_addr @_ZdaPv(!llvm.ptr {llvm.noundef}) attributes {no_unwind, passthrough = ["nobuiltin", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], "uniform-work-group-size"], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @_ZNK4mfem6MemoryIdE4ReadENS_11MemoryClassEi(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 20 : i64, llvm.nonnull, llvm.noundef}, %arg1: i32 {llvm.noundef}, %arg2: i32 {llvm.noundef}) -> (!llvm.ptr {llvm.noundef}) attributes {alignment = 2 : i64, dso_local, no_inline, passthrough = ["mustprogress", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], "uniform-work-group-size"], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uwtable_kind = #llvm.uwtableKind<async>}
  llvm.func local_unnamed_addr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.noundef}, i64 {llvm.noundef}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}) attributes {passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], "uniform-work-group-size"], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @_ZN4mfem21GetLVectorFaceNbrDataERKNS_18FiniteElementSpaceERKNS_6VectorENS_8FaceTypeE(!llvm.ptr {llvm.align = 8 : i64, llvm.dead_on_unwind, llvm.sret = !llvm.struct<"class.mfem::Vector.1", packed (ptr, struct<"class.mfem::Memory.1", packed (ptr, i32, i32, i32, array<4 x i8>)>, i32, array<4 x i8>)>, llvm.writable}, !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 1426 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 36 : i64, llvm.nonnull, llvm.noundef}, i1 {llvm.noundef, llvm.zeroext}) attributes {passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], "uniform-work-group-size"], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @cudaGetLastError() -> i32 attributes {passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], "uniform-work-group-size"], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @_ZN4mfem15mfem_cuda_errorE9cudaErrorPKcS2_S2_i(i32 {llvm.noundef}, !llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.noundef}, i32 {llvm.noundef}) attributes {passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], "uniform-work-group-size"], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
}

// CHECK: func.func private @rxla$raised_0(
