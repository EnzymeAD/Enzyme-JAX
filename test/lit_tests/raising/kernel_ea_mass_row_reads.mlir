// RUN: enzymexlamlir-opt %s --pass-pipeline='builtin.module(raise-affine-to-stablehlo{prefer_while_raising=false})' | FileCheck %s

// An element-assembly kernel (mfem's EAMassAssemble2D<8,8>) reads a shared 
// scratch at [t + k * 8] with t a thread of eight and k a sequential reduction 
// loop: the pair is contiguous but k is scalar per iteration, so the read is a 
// per-lane index and must go through the gather; sliced, its start came out as 
// a tensor<8xi64>.
// The kernel is kept whole (declarations only for what it calls); the check is that it raises.


module {
  llvm.mlir.global external local_unnamed_addr @_ZN4mfem6Device16device_singletonE() {addr_space = 0 : i32, alignment = 1 : i64} : !llvm.struct<"class.mfem::Device.1", packed (i32, i32, i64, i8, i8, array<2 x i8>, i32, i32, i32, i32, array<4 x i8>)>
  llvm.mlir.global private unnamed_addr constant @".str.4"("\0A ... in function: \00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.5"("\0A ... in file: \00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.7"("Verification failed: (\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.8"("D1D <= DeviceDofQuadLimits::Get().MAX_D1D\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.9"(") is false:\0A --> \00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.10"(dense<0> : tensor<1xi8>) {addr_space = 0 : i32, alignment = 1 : i64, dso_local} : !llvm.array<1 x i8>
  llvm.mlir.global private unnamed_addr constant @".str.11"("/mnt3/wmoses/git/mfem/fem/integ/bilininteg_mass_kernels.hpp\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.12"("Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global linkonce_odr @_ZZN4mfem19DeviceDofQuadLimits3GetEvE15dof_quad_limits() {addr_space = 0 : i32, alignment = 4 : i64, dso_local} : !llvm.struct<"struct.mfem::DeviceDofQuadLimits.1", (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> {
    %0 = llvm.mlir.zero : !llvm.struct<"struct.mfem::DeviceDofQuadLimits.1", (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)>
    llvm.return %0 : !llvm.struct<"struct.mfem::DeviceDofQuadLimits.1", (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)>
  }
  llvm.mlir.global linkonce_odr @_ZGVZN4mfem19DeviceDofQuadLimits3GetEvE15dof_quad_limits(0 : i64) {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : i64
  llvm.mlir.global private unnamed_addr constant @".str.14"("/mnt3/wmoses/git/mfem/fem/integ/../../general/forall.hpp\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.15"("cudaGetLastError()\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @__PRETTY_FUNCTION__._ZN4mfem8CuWrap2DIRZNS_8internal16EAMassAssemble2DILi0ELi0EEEviRKNS_5ArrayIdEERKNS_6VectorERS7_biiEUliE_EEviOT_iii("void mfem::CuWrap2D(const int, DBODY &&, const int, const int, const int) [DBODY = (lambda at /mnt3/wmoses/git/mfem/fem/integ/bilininteg_mass_kernels.hpp:1232:34) &]\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @__PRETTY_FUNCTION__._ZN4mfem8internal16EAMassAssemble2DILi8ELi8EEEviRKNS_5ArrayIdEERKNS_6VectorERS6_bii("void mfem::internal::EAMassAssemble2D(const int, const Array<real_t> &, const Vector &, Vector &, const bool, const int, const int) [T_D1D = 8, T_Q1D = 8]\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.func local_unnamed_addr @_ZN4mfem10mfem_errorEPKc(!llvm.ptr {llvm.noundef}) attributes {noreturn, passthrough = ["enzyme_inactive", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], "uniform-work-group-size"], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func @__gxx_personality_v0(...) -> i32 attributes {sym_visibility = "private"}
  llvm.func local_unnamed_addr @_ZdlPvm(!llvm.ptr {llvm.noundef}, i64 {llvm.noundef}) attributes {no_unwind, passthrough = ["nobuiltin", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], "uniform-work-group-size"], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func linkonce_odr local_unnamed_addr @_ZN4mfem8internal16EAMassAssemble2DILi8ELi8EEEviRKNS_5ArrayIdEERKNS_6VectorERS6_bii(%arg0: i32 {llvm.noundef}, %arg1: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 28 : i64, llvm.nonnull, llvm.noundef}, %arg2: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 36 : i64, llvm.nonnull, llvm.noundef}, %arg3: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 36 : i64, llvm.nonnull, llvm.noundef}, %arg4: i1 {llvm.noundef, llvm.zeroext}, %arg5: i32 {llvm.noundef}, %arg6: i32 {llvm.noundef})  attributes {dso_local, inline_hint, passthrough = ["mustprogress", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], "uniform-work-group-size"], personality = @__gxx_personality_v0, target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uwtable_kind = #llvm.uwtableKind<async>} {
    %c-1 = arith.constant -1 : index
    %c4 = arith.constant 4 : index
    %c48 = arith.constant 48 : index
    %c64 = arith.constant 64 : index
    %0 = llvm.mlir.addressof @_ZN4mfem6Device16device_singletonE : !llvm.ptr
    %c8_i64 = arith.constant 8 : i64
    %c0_i32 = arith.constant 0 : i32
    %c0_i8 = arith.constant 0 : i8
    %1 = llvm.mlir.addressof @_ZGVZN4mfem19DeviceDofQuadLimits3GetEvE15dof_quad_limits : !llvm.ptr
    %c1_i32 = arith.constant 1 : i32
    %c5188_i64 = arith.constant 5188 : i64
    %c0_i64 = arith.constant 0 : i64
    %c14_i32 = arith.constant 14 : i32
    %2 = llvm.mlir.addressof @_ZZN4mfem19DeviceDofQuadLimits3GetEvE15dof_quad_limits : !llvm.ptr
    %c4_i64 = arith.constant 4 : i64
    %c5_i32 = arith.constant 5 : i32
    %c16_i64 = arith.constant 16 : i64
    %c6_i32 = arith.constant 6 : i32
    %c8_i32 = arith.constant 8 : i32
    %c8328_i64 = arith.constant 8328 : i64
    %c10_i32 = arith.constant 10 : i32
    %c9_i32 = arith.constant 9 : i32
    %c24_i32 = arith.constant 24 : i32
    %c7_i32 = arith.constant 7 : i32
    %c256_i32 = arith.constant 256 : i32
    %3 = llvm.mlir.addressof @".str.7" : !llvm.ptr
    %c22_i64 = arith.constant 22 : i64
    %4 = llvm.mlir.addressof @".str.8" : !llvm.ptr
    %c41_i64 = arith.constant 41 : i64
    %5 = llvm.mlir.addressof @".str.9" : !llvm.ptr
    %c17_i64 = arith.constant 17 : i64
    %6 = llvm.mlir.addressof @".str.10" : !llvm.ptr
    %7 = llvm.mlir.addressof @".str.4" : !llvm.ptr
    %c19_i64 = arith.constant 19 : i64
    %8 = llvm.mlir.addressof @__PRETTY_FUNCTION__._ZN4mfem8internal16EAMassAssemble2DILi8ELi8EEEviRKNS_5ArrayIdEERKNS_6VectorERS6_bii : !llvm.ptr
    %c154_i64 = arith.constant 154 : i64
    %9 = llvm.mlir.addressof @".str.5" : !llvm.ptr
    %c15_i64 = arith.constant 15 : i64
    %10 = llvm.mlir.addressof @".str.11" : !llvm.ptr
    %c59_i64 = arith.constant 59 : i64
    %c58_i8 = arith.constant 58 : i8
    %c1226_i32 = arith.constant 1226 : i32
    %c10_i8 = arith.constant 10 : i8
    %c1_i64 = arith.constant 1 : i64
    %11 = llvm.mlir.addressof @".str.12" : !llvm.ptr
    %c1227_i32 = arith.constant 1227 : i32
    %c64_i32 = arith.constant 64 : i32
    %true = arith.constant true
    %12 = llvm.mlir.addressof @".str.15" : !llvm.ptr
    %13 = llvm.mlir.addressof @__PRETTY_FUNCTION__._ZN4mfem8CuWrap2DIRZNS_8internal16EAMassAssemble2DILi0ELi0EEEviRKNS_5ArrayIdEERKNS_6VectorERS7_biiEUliE_EEviOT_iii : !llvm.ptr
    %14 = llvm.mlir.addressof @".str.14" : !llvm.ptr
    %c785_i32 = arith.constant 785 : i32
    %cst = arith.constant 0.000000e+00 : f64
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %15 = "enzymexla.pointer2memref"(%1) : (!llvm.ptr) -> memref<?xi8>
    %alloca = memref.alloca() {alignment = 16 : i64} : memref<64xf64>
    %alloca_0 = memref.alloca() {alignment = 16 : i64} : memref<64xf64>
    %16 = llvm.alloca %c1_i32 x !llvm.struct<"class.std::__cxx11::basic_ostringstream.1", (struct<"class.std::basic_ostream.base.1", (ptr)>, struct<"class.std::__cxx11::basic_stringbuf.1", (struct<"class.std::basic_streambuf.1", (ptr, ptr, ptr, ptr, ptr, ptr, ptr, struct<"class.std::locale.1", (ptr)>)>, i32, struct<"class.std::__cxx11::basic_string.1", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider.1", (ptr)>, i64, struct<"union.anon.1", (i64, array<8 x i8>)>)>)>, struct<"class.std::basic_ios.1", (struct<"class.std::ios_base.1", (ptr, i64, i64, i32, i32, i32, ptr, struct<"struct.std::ios_base::_Words.1", (ptr, i64)>, array<8 x struct<"struct.std::ios_base::_Words.1", (ptr, i64)>>, i32, ptr, struct<"class.std::locale.1", (ptr)>)>, ptr, i8, i8, ptr, ptr, ptr, ptr)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %17 = llvm.alloca %c1_i32 x !llvm.struct<"class.std::__cxx11::basic_string.1", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider.1", (ptr)>, i64, struct<"union.anon.1", (i64, array<8 x i8>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %18 = llvm.alloca %c1_i32 x !llvm.struct<"class.std::__cxx11::basic_ostringstream.1", (struct<"class.std::basic_ostream.base.1", (ptr)>, struct<"class.std::__cxx11::basic_stringbuf.1", (struct<"class.std::basic_streambuf.1", (ptr, ptr, ptr, ptr, ptr, ptr, ptr, struct<"class.std::locale.1", (ptr)>)>, i32, struct<"class.std::__cxx11::basic_string.1", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider.1", (ptr)>, i64, struct<"union.anon.1", (i64, array<8 x i8>)>)>)>, struct<"class.std::basic_ios.1", (struct<"class.std::ios_base.1", (ptr, i64, i64, i32, i32, i32, ptr, struct<"struct.std::ios_base::_Words.1", (ptr, i64)>, array<8 x struct<"struct.std::ios_base::_Words.1", (ptr, i64)>>, i32, ptr, struct<"class.std::locale.1", (ptr)>)>, ptr, i8, i8, ptr, ptr, ptr, ptr)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %19 = llvm.alloca %c1_i32 x !llvm.struct<"class.std::__cxx11::basic_string.1", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider.1", (ptr)>, i64, struct<"union.anon.1", (i64, array<8 x i8>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %20 = affine.load %15[0] : memref<?xi8>
    %21 = arith.cmpi eq, %20, %c0_i8 : i8
    cf.cond_br %21, ^bb1, ^bb9
  ^bb1:  // pred: ^bb0
    %22 = llvm.call tail @__cxa_guard_acquire(%1) {no_unwind} : (!llvm.ptr {llvm.nonnull}) -> i32
    %23 = arith.cmpi eq, %22, %c0_i32 : i32
    cf.cond_br %23, ^bb9, ^bb2
  ^bb2:  // pred: ^bb1
    %24 = "enzymexla.pointer2memref"(%0) : (!llvm.ptr) -> memref<?xi64>
    %25 = affine.load %24[1] : memref<?xi64>
    %26 = arith.andi %25, %c5188_i64 : i64
    %27 = arith.cmpi eq, %26, %c0_i64 : i64
    cf.cond_br %27, ^bb4, ^bb3
  ^bb3:  // pred: ^bb2
    %28 = "enzymexla.pointer2memref"(%2) : (!llvm.ptr) -> memref<?xi32>
    affine.store %c14_i32, %28[0] : memref<?xi32>
    affine.store %c14_i32, %28[1] : memref<?xi32>
    affine.store %c14_i32, %28[2] : memref<?xi32>
    affine.store %c14_i32, %28[3] : memref<?xi32>
    affine.store %c5_i32, %28[4] : memref<?xi32>
    affine.store %c6_i32, %28[5] : memref<?xi32>
    affine.store %c5_i32, %28[6] : memref<?xi32>
    affine.store %c6_i32, %28[7] : memref<?xi32>
    affine.store %c8_i32, %28[8] : memref<?xi32>
    affine.store %c6_i32, %28[9] : memref<?xi32>
    cf.br ^bb7
  ^bb4:  // pred: ^bb2
    %29 = arith.andi %25, %c8328_i64 : i64
    %30 = arith.cmpi eq, %29, %c0_i64 : i64
    cf.cond_br %30, ^bb6, ^bb5
  ^bb5:  // pred: ^bb4
    %31 = "enzymexla.pointer2memref"(%2) : (!llvm.ptr) -> memref<?xi32>
    affine.store %c10_i32, %31[0] : memref<?xi32>
    affine.store %c10_i32, %31[1] : memref<?xi32>
    affine.store %c9_i32, %31[2] : memref<?xi32>
    affine.store %c9_i32, %31[3] : memref<?xi32>
    affine.store %c5_i32, %31[4] : memref<?xi32>
    affine.store %c5_i32, %31[5] : memref<?xi32>
    affine.store %c5_i32, %31[6] : memref<?xi32>
    affine.store %c6_i32, %31[7] : memref<?xi32>
    affine.store %c8_i32, %31[8] : memref<?xi32>
    affine.store %c6_i32, %31[9] : memref<?xi32>
    cf.br ^bb7
  ^bb6:  // pred: ^bb4
    %32 = "enzymexla.pointer2memref"(%2) : (!llvm.ptr) -> memref<?xi32>
    affine.store %c24_i32, %32[0] : memref<?xi32>
    affine.store %c24_i32, %32[1] : memref<?xi32>
    affine.store %c24_i32, %32[2] : memref<?xi32>
    affine.store %c24_i32, %32[3] : memref<?xi32>
    affine.store %c10_i32, %32[4] : memref<?xi32>
    affine.store %c10_i32, %32[5] : memref<?xi32>
    affine.store %c10_i32, %32[6] : memref<?xi32>
    affine.store %c10_i32, %32[7] : memref<?xi32>
    affine.store %c24_i32, %32[8] : memref<?xi32>
    affine.store %c24_i32, %32[9] : memref<?xi32>
    cf.br ^bb7
  ^bb7:  // 3 preds: ^bb3, ^bb5, ^bb6
    %33 = llvm.intr.invariant.start 40, %2 : !llvm.ptr
    llvm.call tail @__cxa_guard_release(%1) {no_unwind} : (!llvm.ptr {llvm.nonnull}) -> ()
    cf.br ^bb9
  ^bb8(%34: !llvm.struct<(ptr, i32)>):  // 2 preds: ^bb30, ^bb60
    llvm.resume %34 : !llvm.struct<(ptr, i32)>
  ^bb9:  // 3 preds: ^bb0, ^bb1, ^bb7
    %35 = "enzymexla.pointer2memref"(%2) : (!llvm.ptr) -> memref<?xi32>
    %36 = affine.load %35[0] : memref<?xi32>
    %37 = arith.cmpi sgt, %36, %c7_i32 : i32
    cf.cond_br %37, ^bb31, ^bb10
  ^bb10:  // pred: ^bb9
    llvm.intr.lifetime.start %16 : !llvm.ptr
    llvm.call @_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEC1Ev(%16) : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 112 : i64, llvm.nonnull, llvm.noundef}) -> ()
    %38 = "enzymexla.pointer2memref"(%16) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    %39 = affine.load %38[0] : memref<?x!llvm.ptr>
    %40 = "enzymexla.pointer2memref"(%39) : (!llvm.ptr) -> memref<?xi64>
    %41 = affine.load %40[-3] : memref<?xi64>
    %42 = arith.index_cast %41 : i64 to index
    %43 = "enzymexla.pointer2memref"(%16) : (!llvm.ptr) -> memref<?xi64>
    %44 = arith.cmpi slt, %42, %c0 : index
    %45 = arith.subi %c-1, %42 : index
    %46 = arith.select %44, %45, %42 : index
    %47 = arith.divsi %46, %c8 : index
    %48 = arith.subi %c-1, %47 : index
    %49 = arith.select %44, %48, %47 : index
    affine.store %c16_i64, %43[symbol(%49) + 1] : memref<?xi64>
    %50 = affine.load %40[-3] : memref<?xi64>
    %51 = arith.index_cast %50 : i64 to index
    %52 = "enzymexla.pointer2memref"(%16) : (!llvm.ptr) -> memref<?xi32>
    %53 = arith.cmpi slt, %51, %c0 : index
    %54 = arith.subi %c-1, %51 : index
    %55 = arith.select %53, %54, %51 : index
    %56 = arith.divsi %55, %c4 : index
    %57 = arith.subi %c-1, %56 : index
    %58 = arith.select %53, %57, %56 : index
    %59 = affine.load %52[symbol(%58) + 6] : memref<?xi32>
    %60 = arith.ori %59, %c256_i32 : i32
    affine.store %60, %52[symbol(%58) + 6] : memref<?xi32>
    %61 = llvm.invoke @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%16, %3, %c22_i64) to ^bb11 unwind ^bb24 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, i64 {llvm.noundef}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef})
  ^bb11:  // pred: ^bb10
    %62 = llvm.invoke @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%16, %4, %c41_i64) to ^bb12 unwind ^bb24 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, i64 {llvm.noundef}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef})
  ^bb12:  // pred: ^bb11
    %63 = llvm.invoke @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%16, %5, %c17_i64) to ^bb13 unwind ^bb24 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, i64 {llvm.noundef}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef})
  ^bb13:  // pred: ^bb12
    %64 = llvm.invoke @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%16, %6, %c0_i64) to ^bb14 unwind ^bb24 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, i64 {llvm.noundef}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef})
  ^bb14:  // pred: ^bb13
    %65 = llvm.invoke @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%16, %7, %c19_i64) to ^bb15 unwind ^bb24 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, i64 {llvm.noundef}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef})
  ^bb15:  // pred: ^bb14
    %66 = llvm.invoke @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%16, %8, %c154_i64) to ^bb16 unwind ^bb24 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, i64 {llvm.noundef}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef})
  ^bb16:  // pred: ^bb15
    %67 = llvm.invoke @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%16, %9, %c15_i64) to ^bb17 unwind ^bb24 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, i64 {llvm.noundef}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef})
  ^bb17:  // pred: ^bb16
    %68 = llvm.invoke @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%16, %10, %c59_i64) to ^bb18 unwind ^bb24 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, i64 {llvm.noundef}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef})
  ^bb18:  // pred: ^bb17
    %69 = llvm.invoke @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_c(%16, %c58_i8) to ^bb19 unwind ^bb24 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, i8 {llvm.noundef, llvm.signext}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef})
  ^bb19:  // pred: ^bb18
    %70 = llvm.invoke @_ZNSolsEi(%69, %c1226_i32) to ^bb20 unwind ^bb24 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, i32 {llvm.noundef}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef})
  ^bb20:  // pred: ^bb19
    %71 = llvm.invoke @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_c(%70, %c10_i8) to ^bb21 unwind ^bb24 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, i8 {llvm.noundef, llvm.signext}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef})
  ^bb21:  // pred: ^bb20
    llvm.intr.lifetime.start %17 : !llvm.ptr
    llvm.invoke @_ZNKSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEE3strEv(%17, %16) to ^bb22 unwind ^bb25 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dead_on_unwind, llvm.nonnull, llvm.sret = !llvm.struct<"class.std::__cxx11::basic_string.1", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider.1", (ptr)>, i64, struct<"union.anon.1", (i64, array<8 x i8>)>)>, llvm.writable}, !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 112 : i64, llvm.nonnull, llvm.noundef}) -> ()
  ^bb22:  // pred: ^bb21
    %72 = "enzymexla.pointer2memref"(%17) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    %73 = affine.load %72[0] : memref<?x!llvm.ptr>
    llvm.invoke @_ZN4mfem10mfem_errorEPKc(%73) to ^bb23 unwind ^bb26 : (!llvm.ptr {llvm.noundef}) -> ()
  ^bb23:  // pred: ^bb22
    llvm.unreachable
  ^bb24:  // 11 preds: ^bb10, ^bb11, ^bb12, ^bb13, ^bb14, ^bb15, ^bb16, ^bb17, ^bb18, ^bb19, ^bb20
    %74 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    cf.br ^bb30(%74 : !llvm.struct<(ptr, i32)>)
  ^bb25:  // pred: ^bb21
    %75 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    cf.br ^bb29(%75 : !llvm.struct<(ptr, i32)>)
  ^bb26:  // pred: ^bb22
    %76 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    %77 = affine.load %72[0] : memref<?x!llvm.ptr>
    %78 = llvm.getelementptr inbounds|nuw %17[16] : (!llvm.ptr) -> !llvm.ptr, i8
    %79 = llvm.icmp "eq" %77, %78 : !llvm.ptr
    cf.cond_br %79, ^bb27, ^bb28
  ^bb27:  // pred: ^bb26
    %80 = "enzymexla.pointer2memref"(%17) : (!llvm.ptr) -> memref<?xi64>
    %81 = affine.load %80[1] : memref<?xi64>
    %82 = arith.cmpi ult, %81, %c16_i64 : i64
    llvm.intr.assume %82  : i1
    cf.br ^bb29(%76 : !llvm.struct<(ptr, i32)>)
  ^bb28:  // pred: ^bb26
    %83 = "enzymexla.pointer2memref"(%17) : (!llvm.ptr) -> memref<?xi64>
    %84 = affine.load %83[2] : memref<?xi64>
    %85 = arith.addi %84, %c1_i64 : i64
    llvm.call @_ZdlPvm(%77, %85) {builtin, no_unwind} : (!llvm.ptr {llvm.noundef}, i64 {llvm.noundef}) -> ()
    cf.br ^bb29(%76 : !llvm.struct<(ptr, i32)>)
  ^bb29(%86: !llvm.struct<(ptr, i32)>):  // 3 preds: ^bb25, ^bb27, ^bb28
    llvm.intr.lifetime.end %17 : !llvm.ptr
    cf.br ^bb30(%86 : !llvm.struct<(ptr, i32)>)
  ^bb30(%87: !llvm.struct<(ptr, i32)>):  // 2 preds: ^bb24, ^bb29
    llvm.call @_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev(%16) {no_unwind} : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 112 : i64, llvm.nonnull, llvm.noundef}) -> ()
    llvm.intr.lifetime.end %16 : !llvm.ptr
    cf.br ^bb8(%87 : !llvm.struct<(ptr, i32)>)
  ^bb31:  // pred: ^bb9
    %88 = affine.load %15[0] : memref<?xi8>
    %89 = arith.cmpi eq, %88, %c0_i8 : i8
    cf.cond_br %89, ^bb32, ^bb39
  ^bb32:  // pred: ^bb31
    %90 = llvm.call tail @__cxa_guard_acquire(%1) {no_unwind} : (!llvm.ptr {llvm.nonnull}) -> i32
    %91 = arith.cmpi eq, %90, %c0_i32 : i32
    cf.cond_br %91, ^bb39, ^bb33
  ^bb33:  // pred: ^bb32
    %92 = "enzymexla.pointer2memref"(%0) : (!llvm.ptr) -> memref<?xi64>
    %93 = affine.load %92[1] : memref<?xi64>
    %94 = arith.andi %93, %c5188_i64 : i64
    %95 = arith.cmpi eq, %94, %c0_i64 : i64
    cf.cond_br %95, ^bb35, ^bb34
  ^bb34:  // pred: ^bb33
    affine.store %c14_i32, %35[0] : memref<?xi32>
    affine.store %c14_i32, %35[1] : memref<?xi32>
    affine.store %c14_i32, %35[2] : memref<?xi32>
    affine.store %c14_i32, %35[3] : memref<?xi32>
    affine.store %c5_i32, %35[4] : memref<?xi32>
    affine.store %c6_i32, %35[5] : memref<?xi32>
    affine.store %c5_i32, %35[6] : memref<?xi32>
    affine.store %c6_i32, %35[7] : memref<?xi32>
    affine.store %c8_i32, %35[8] : memref<?xi32>
    affine.store %c6_i32, %35[9] : memref<?xi32>
    cf.br ^bb38
  ^bb35:  // pred: ^bb33
    %96 = arith.andi %93, %c8328_i64 : i64
    %97 = arith.cmpi eq, %96, %c0_i64 : i64
    cf.cond_br %97, ^bb37, ^bb36
  ^bb36:  // pred: ^bb35
    affine.store %c10_i32, %35[0] : memref<?xi32>
    affine.store %c10_i32, %35[1] : memref<?xi32>
    affine.store %c9_i32, %35[2] : memref<?xi32>
    affine.store %c9_i32, %35[3] : memref<?xi32>
    affine.store %c5_i32, %35[4] : memref<?xi32>
    affine.store %c5_i32, %35[5] : memref<?xi32>
    affine.store %c5_i32, %35[6] : memref<?xi32>
    affine.store %c6_i32, %35[7] : memref<?xi32>
    affine.store %c8_i32, %35[8] : memref<?xi32>
    affine.store %c6_i32, %35[9] : memref<?xi32>
    cf.br ^bb38
  ^bb37:  // pred: ^bb35
    affine.store %c24_i32, %35[0] : memref<?xi32>
    affine.store %c24_i32, %35[1] : memref<?xi32>
    affine.store %c24_i32, %35[2] : memref<?xi32>
    affine.store %c24_i32, %35[3] : memref<?xi32>
    affine.store %c10_i32, %35[4] : memref<?xi32>
    affine.store %c10_i32, %35[5] : memref<?xi32>
    affine.store %c10_i32, %35[6] : memref<?xi32>
    affine.store %c10_i32, %35[7] : memref<?xi32>
    affine.store %c24_i32, %35[8] : memref<?xi32>
    affine.store %c24_i32, %35[9] : memref<?xi32>
    cf.br ^bb38
  ^bb38:  // 3 preds: ^bb34, ^bb36, ^bb37
    %98 = llvm.intr.invariant.start 40, %2 : !llvm.ptr
    llvm.call tail @__cxa_guard_release(%1) {no_unwind} : (!llvm.ptr {llvm.nonnull}) -> ()
    cf.br ^bb39
  ^bb39:  // 3 preds: ^bb31, ^bb32, ^bb38
    %99 = affine.load %35[1] : memref<?xi32>
    %100 = arith.cmpi sgt, %99, %c7_i32 : i32
    cf.cond_br %100, ^bb61, ^bb40
  ^bb40:  // pred: ^bb39
    llvm.intr.lifetime.start %18 : !llvm.ptr
    llvm.call @_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEC1Ev(%18) : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 112 : i64, llvm.nonnull, llvm.noundef}) -> ()
    %101 = "enzymexla.pointer2memref"(%18) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    %102 = affine.load %101[0] : memref<?x!llvm.ptr>
    %103 = "enzymexla.pointer2memref"(%102) : (!llvm.ptr) -> memref<?xi64>
    %104 = affine.load %103[-3] : memref<?xi64>
    %105 = arith.index_cast %104 : i64 to index
    %106 = "enzymexla.pointer2memref"(%18) : (!llvm.ptr) -> memref<?xi64>
    %107 = arith.cmpi slt, %105, %c0 : index
    %108 = arith.subi %c-1, %105 : index
    %109 = arith.select %107, %108, %105 : index
    %110 = arith.divsi %109, %c8 : index
    %111 = arith.subi %c-1, %110 : index
    %112 = arith.select %107, %111, %110 : index
    affine.store %c16_i64, %106[symbol(%112) + 1] : memref<?xi64>
    %113 = affine.load %103[-3] : memref<?xi64>
    %114 = arith.index_cast %113 : i64 to index
    %115 = "enzymexla.pointer2memref"(%18) : (!llvm.ptr) -> memref<?xi32>
    %116 = arith.cmpi slt, %114, %c0 : index
    %117 = arith.subi %c-1, %114 : index
    %118 = arith.select %116, %117, %114 : index
    %119 = arith.divsi %118, %c4 : index
    %120 = arith.subi %c-1, %119 : index
    %121 = arith.select %116, %120, %119 : index
    %122 = affine.load %115[symbol(%121) + 6] : memref<?xi32>
    %123 = arith.ori %122, %c256_i32 : i32
    affine.store %123, %115[symbol(%121) + 6] : memref<?xi32>
    %124 = llvm.invoke @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%18, %3, %c22_i64) to ^bb41 unwind ^bb54 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, i64 {llvm.noundef}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef})
  ^bb41:  // pred: ^bb40
    %125 = llvm.invoke @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%18, %11, %c41_i64) to ^bb42 unwind ^bb54 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, i64 {llvm.noundef}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef})
  ^bb42:  // pred: ^bb41
    %126 = llvm.invoke @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%18, %5, %c17_i64) to ^bb43 unwind ^bb54 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, i64 {llvm.noundef}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef})
  ^bb43:  // pred: ^bb42
    %127 = llvm.invoke @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%18, %6, %c0_i64) to ^bb44 unwind ^bb54 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, i64 {llvm.noundef}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef})
  ^bb44:  // pred: ^bb43
    %128 = llvm.invoke @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%18, %7, %c19_i64) to ^bb45 unwind ^bb54 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, i64 {llvm.noundef}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef})
  ^bb45:  // pred: ^bb44
    %129 = llvm.invoke @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%18, %8, %c154_i64) to ^bb46 unwind ^bb54 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, i64 {llvm.noundef}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef})
  ^bb46:  // pred: ^bb45
    %130 = llvm.invoke @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%18, %9, %c15_i64) to ^bb47 unwind ^bb54 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, i64 {llvm.noundef}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef})
  ^bb47:  // pred: ^bb46
    %131 = llvm.invoke @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%18, %10, %c59_i64) to ^bb48 unwind ^bb54 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, i64 {llvm.noundef}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef})
  ^bb48:  // pred: ^bb47
    %132 = llvm.invoke @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_c(%18, %c58_i8) to ^bb49 unwind ^bb54 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, i8 {llvm.noundef, llvm.signext}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef})
  ^bb49:  // pred: ^bb48
    %133 = llvm.invoke @_ZNSolsEi(%132, %c1227_i32) to ^bb50 unwind ^bb54 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, i32 {llvm.noundef}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef})
  ^bb50:  // pred: ^bb49
    %134 = llvm.invoke @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_c(%133, %c10_i8) to ^bb51 unwind ^bb54 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, i8 {llvm.noundef, llvm.signext}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef})
  ^bb51:  // pred: ^bb50
    llvm.intr.lifetime.start %19 : !llvm.ptr
    llvm.invoke @_ZNKSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEE3strEv(%19, %18) to ^bb52 unwind ^bb55 : (!llvm.ptr {llvm.align = 8 : i64, llvm.dead_on_unwind, llvm.nonnull, llvm.sret = !llvm.struct<"class.std::__cxx11::basic_string.1", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider.1", (ptr)>, i64, struct<"union.anon.1", (i64, array<8 x i8>)>)>, llvm.writable}, !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 112 : i64, llvm.nonnull, llvm.noundef}) -> ()
  ^bb52:  // pred: ^bb51
    %135 = "enzymexla.pointer2memref"(%19) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    %136 = affine.load %135[0] : memref<?x!llvm.ptr>
    llvm.invoke @_ZN4mfem10mfem_errorEPKc(%136) to ^bb53 unwind ^bb56 : (!llvm.ptr {llvm.noundef}) -> ()
  ^bb53:  // pred: ^bb52
    llvm.unreachable
  ^bb54:  // 11 preds: ^bb40, ^bb41, ^bb42, ^bb43, ^bb44, ^bb45, ^bb46, ^bb47, ^bb48, ^bb49, ^bb50
    %137 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    cf.br ^bb60(%137 : !llvm.struct<(ptr, i32)>)
  ^bb55:  // pred: ^bb51
    %138 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    cf.br ^bb59(%138 : !llvm.struct<(ptr, i32)>)
  ^bb56:  // pred: ^bb52
    %139 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    %140 = affine.load %135[0] : memref<?x!llvm.ptr>
    %141 = llvm.getelementptr inbounds|nuw %19[16] : (!llvm.ptr) -> !llvm.ptr, i8
    %142 = llvm.icmp "eq" %140, %141 : !llvm.ptr
    cf.cond_br %142, ^bb57, ^bb58
  ^bb57:  // pred: ^bb56
    %143 = "enzymexla.pointer2memref"(%19) : (!llvm.ptr) -> memref<?xi64>
    %144 = affine.load %143[1] : memref<?xi64>
    %145 = arith.cmpi ult, %144, %c16_i64 : i64
    llvm.intr.assume %145  : i1
    cf.br ^bb59(%139 : !llvm.struct<(ptr, i32)>)
  ^bb58:  // pred: ^bb56
    %146 = "enzymexla.pointer2memref"(%19) : (!llvm.ptr) -> memref<?xi64>
    %147 = affine.load %146[2] : memref<?xi64>
    %148 = arith.addi %147, %c1_i64 : i64
    llvm.call @_ZdlPvm(%140, %148) {builtin, no_unwind} : (!llvm.ptr {llvm.noundef}, i64 {llvm.noundef}) -> ()
    cf.br ^bb59(%139 : !llvm.struct<(ptr, i32)>)
  ^bb59(%149: !llvm.struct<(ptr, i32)>):  // 3 preds: ^bb55, ^bb57, ^bb58
    llvm.intr.lifetime.end %19 : !llvm.ptr
    cf.br ^bb60(%149 : !llvm.struct<(ptr, i32)>)
  ^bb60(%150: !llvm.struct<(ptr, i32)>):  // 2 preds: ^bb54, ^bb59
    llvm.call @_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev(%18) {no_unwind} : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 112 : i64, llvm.nonnull, llvm.noundef}) -> ()
    llvm.intr.lifetime.end %18 : !llvm.ptr
    cf.br ^bb8(%150 : !llvm.struct<(ptr, i32)>)
  ^bb61:  // pred: ^bb39
    %151 = "enzymexla.pointer2memref"(%arg1) : (!llvm.ptr) -> memref<?xi32>
    %152 = affine.load %151[6] : memref<?xi32>
    %153 = affine.load %151[4] : memref<?xi32>
    %154 = arith.ori %153, %c64_i32 : i32
    affine.store %154, %151[4] : memref<?xi32>
    %155 = "enzymexla.pointer2memref"(%0) : (!llvm.ptr) -> memref<?xi32>
    %156 = affine.load %155[8] : memref<?xi32>
    %157 = llvm.call tail @_ZNK4mfem6MemoryIdE4ReadENS_11MemoryClassEi(%arg1, %156, %152) : (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 28 : i64, llvm.nonnull, llvm.noundef}, i32 {llvm.noundef}, i32 {llvm.noundef}) -> (!llvm.ptr {llvm.noundef})
    %158 = "enzymexla.pointer2memref"(%arg2) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    %159 = affine.load %158[0] : memref<?x!llvm.ptr>
    %160 = "enzymexla.pointer2memref"(%159) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    %161 = affine.load %160[4] : memref<?x!llvm.ptr>
    %162 = llvm.call tail %161(%arg2, %true) : !llvm.ptr, (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 36 : i64, llvm.nonnull, llvm.noundef}, i1 {llvm.noundef, llvm.zeroext}) -> (!llvm.ptr {llvm.noundef})
    %163 = "enzymexla.pointer2memref"(%arg3) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    %164 = affine.load %163[0] : memref<?x!llvm.ptr>
    %165 = arith.select %arg4, %c64, %c48 : index
    %166 = "enzymexla.pointer2memref"(%164) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    %167 = arith.cmpi slt, %165, %c0 : index
    %168 = arith.subi %c-1, %165 : index
    %169 = arith.select %167, %168, %165 : index
    %170 = arith.divsi %169, %c8 : index
    %171 = arith.subi %c-1, %170 : index
    %172 = arith.select %167, %171, %170 : index
    %173 = affine.load %166[symbol(%172)] : memref<?x!llvm.ptr>
    %174 = llvm.call tail %173(%arg3, %true) : !llvm.ptr, (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 36 : i64, llvm.nonnull, llvm.noundef}, i1 {llvm.noundef, llvm.zeroext}) -> (!llvm.ptr {llvm.noundef})
    %175 = "enzymexla.pointer2memref"(%0) : (!llvm.ptr) -> memref<?xi64>
    %176 = affine.load %175[1] : memref<?xi64>
    %177 = arith.andi %176, %c4_i64 : i64
    %178 = arith.cmpi eq, %177, %c0_i64 : i64
    cf.cond_br %178, ^bb62, ^bb63
  ^bb62:  // pred: ^bb61
    %179 = arith.cmpi sgt, %arg0, %c0_i32 : i32
    cf.cond_br %179, ^bb66(%c0_i32 : i32), ^bb86
  ^bb63:  // pred: ^bb61
    %180 = arith.cmpi eq, %arg0, %c0_i32 : i32
    cf.cond_br %180, ^bb86, ^bb64
  ^bb64:  // pred: ^bb63
    %181 = arith.index_cast %arg0 : i32 to index
    %182 = "enzymexla.gpu_wrapper"(%181, %c1, %c1, %c8, %c8, %c1) ({
      affine.parallel (%arg7) = (0) to (symbol(%181)) {
        %alloca_1 = memref.alloca() {alignment = 8 : i64} : memref<8x8xf64>
        affine.parallel (%arg8, %arg9) = (0, 0) to (8, 8) {
          %alloca_2 = memref.alloca() {alignment = 8 : i64} : memref<64xf64>
          %491 = "enzymexla.pointer2memref"(%157) : (!llvm.ptr) -> memref<?xf64>
          %492 = affine.load %491[0] : memref<?xf64>
          affine.store %492, %alloca_2[0] : memref<64xf64>
          %493 = affine.load %491[1] : memref<?xf64>
          affine.store %493, %alloca_2[8] : memref<64xf64>
          %494 = affine.load %491[2] : memref<?xf64>
          affine.store %494, %alloca_2[16] : memref<64xf64>
          %495 = affine.load %491[3] : memref<?xf64>
          affine.store %495, %alloca_2[24] : memref<64xf64>
          %496 = affine.load %491[4] : memref<?xf64>
          affine.store %496, %alloca_2[32] : memref<64xf64>
          %497 = affine.load %491[5] : memref<?xf64>
          affine.store %497, %alloca_2[40] : memref<64xf64>
          %498 = affine.load %491[6] : memref<?xf64>
          affine.store %498, %alloca_2[48] : memref<64xf64>
          %499 = affine.load %491[7] : memref<?xf64>
          affine.store %499, %alloca_2[56] : memref<64xf64>
          %500 = affine.load %491[8] : memref<?xf64>
          affine.store %500, %alloca_2[1] : memref<64xf64>
          %501 = affine.load %491[9] : memref<?xf64>
          affine.store %501, %alloca_2[9] : memref<64xf64>
          %502 = affine.load %491[10] : memref<?xf64>
          affine.store %502, %alloca_2[17] : memref<64xf64>
          %503 = affine.load %491[11] : memref<?xf64>
          affine.store %503, %alloca_2[25] : memref<64xf64>
          %504 = affine.load %491[12] : memref<?xf64>
          affine.store %504, %alloca_2[33] : memref<64xf64>
          %505 = affine.load %491[13] : memref<?xf64>
          affine.store %505, %alloca_2[41] : memref<64xf64>
          %506 = affine.load %491[14] : memref<?xf64>
          affine.store %506, %alloca_2[49] : memref<64xf64>
          %507 = affine.load %491[15] : memref<?xf64>
          affine.store %507, %alloca_2[57] : memref<64xf64>
          %508 = affine.load %491[16] : memref<?xf64>
          affine.store %508, %alloca_2[2] : memref<64xf64>
          %509 = affine.load %491[17] : memref<?xf64>
          affine.store %509, %alloca_2[10] : memref<64xf64>
          %510 = affine.load %491[18] : memref<?xf64>
          affine.store %510, %alloca_2[18] : memref<64xf64>
          %511 = affine.load %491[19] : memref<?xf64>
          affine.store %511, %alloca_2[26] : memref<64xf64>
          %512 = affine.load %491[20] : memref<?xf64>
          affine.store %512, %alloca_2[34] : memref<64xf64>
          %513 = affine.load %491[21] : memref<?xf64>
          affine.store %513, %alloca_2[42] : memref<64xf64>
          %514 = affine.load %491[22] : memref<?xf64>
          affine.store %514, %alloca_2[50] : memref<64xf64>
          %515 = affine.load %491[23] : memref<?xf64>
          affine.store %515, %alloca_2[58] : memref<64xf64>
          %516 = affine.load %491[24] : memref<?xf64>
          affine.store %516, %alloca_2[3] : memref<64xf64>
          %517 = affine.load %491[25] : memref<?xf64>
          affine.store %517, %alloca_2[11] : memref<64xf64>
          %518 = affine.load %491[26] : memref<?xf64>
          affine.store %518, %alloca_2[19] : memref<64xf64>
          %519 = affine.load %491[27] : memref<?xf64>
          affine.store %519, %alloca_2[27] : memref<64xf64>
          %520 = affine.load %491[28] : memref<?xf64>
          affine.store %520, %alloca_2[35] : memref<64xf64>
          %521 = affine.load %491[29] : memref<?xf64>
          affine.store %521, %alloca_2[43] : memref<64xf64>
          %522 = affine.load %491[30] : memref<?xf64>
          affine.store %522, %alloca_2[51] : memref<64xf64>
          %523 = affine.load %491[31] : memref<?xf64>
          affine.store %523, %alloca_2[59] : memref<64xf64>
          %524 = affine.load %491[32] : memref<?xf64>
          affine.store %524, %alloca_2[4] : memref<64xf64>
          %525 = affine.load %491[33] : memref<?xf64>
          affine.store %525, %alloca_2[12] : memref<64xf64>
          %526 = affine.load %491[34] : memref<?xf64>
          affine.store %526, %alloca_2[20] : memref<64xf64>
          %527 = affine.load %491[35] : memref<?xf64>
          affine.store %527, %alloca_2[28] : memref<64xf64>
          %528 = affine.load %491[36] : memref<?xf64>
          affine.store %528, %alloca_2[36] : memref<64xf64>
          %529 = affine.load %491[37] : memref<?xf64>
          affine.store %529, %alloca_2[44] : memref<64xf64>
          %530 = affine.load %491[38] : memref<?xf64>
          affine.store %530, %alloca_2[52] : memref<64xf64>
          %531 = affine.load %491[39] : memref<?xf64>
          affine.store %531, %alloca_2[60] : memref<64xf64>
          %532 = affine.load %491[40] : memref<?xf64>
          affine.store %532, %alloca_2[5] : memref<64xf64>
          %533 = affine.load %491[41] : memref<?xf64>
          affine.store %533, %alloca_2[13] : memref<64xf64>
          %534 = affine.load %491[42] : memref<?xf64>
          affine.store %534, %alloca_2[21] : memref<64xf64>
          %535 = affine.load %491[43] : memref<?xf64>
          affine.store %535, %alloca_2[29] : memref<64xf64>
          %536 = affine.load %491[44] : memref<?xf64>
          affine.store %536, %alloca_2[37] : memref<64xf64>
          %537 = affine.load %491[45] : memref<?xf64>
          affine.store %537, %alloca_2[45] : memref<64xf64>
          %538 = affine.load %491[46] : memref<?xf64>
          affine.store %538, %alloca_2[53] : memref<64xf64>
          %539 = affine.load %491[47] : memref<?xf64>
          affine.store %539, %alloca_2[61] : memref<64xf64>
          %540 = affine.load %491[48] : memref<?xf64>
          affine.store %540, %alloca_2[6] : memref<64xf64>
          %541 = affine.load %491[49] : memref<?xf64>
          affine.store %541, %alloca_2[14] : memref<64xf64>
          %542 = affine.load %491[50] : memref<?xf64>
          affine.store %542, %alloca_2[22] : memref<64xf64>
          %543 = affine.load %491[51] : memref<?xf64>
          affine.store %543, %alloca_2[30] : memref<64xf64>
          %544 = affine.load %491[52] : memref<?xf64>
          affine.store %544, %alloca_2[38] : memref<64xf64>
          %545 = affine.load %491[53] : memref<?xf64>
          affine.store %545, %alloca_2[46] : memref<64xf64>
          %546 = affine.load %491[54] : memref<?xf64>
          affine.store %546, %alloca_2[54] : memref<64xf64>
          %547 = affine.load %491[55] : memref<?xf64>
          affine.store %547, %alloca_2[62] : memref<64xf64>
          %548 = affine.load %491[56] : memref<?xf64>
          affine.store %548, %alloca_2[7] : memref<64xf64>
          %549 = affine.load %491[57] : memref<?xf64>
          affine.store %549, %alloca_2[15] : memref<64xf64>
          %550 = affine.load %491[58] : memref<?xf64>
          affine.store %550, %alloca_2[23] : memref<64xf64>
          %551 = affine.load %491[59] : memref<?xf64>
          affine.store %551, %alloca_2[31] : memref<64xf64>
          %552 = affine.load %491[60] : memref<?xf64>
          affine.store %552, %alloca_2[39] : memref<64xf64>
          %553 = affine.load %491[61] : memref<?xf64>
          affine.store %553, %alloca_2[47] : memref<64xf64>
          %554 = affine.load %491[62] : memref<?xf64>
          affine.store %554, %alloca_2[55] : memref<64xf64>
          %555 = affine.load %491[63] : memref<?xf64>
          affine.store %555, %alloca_2[63] : memref<64xf64>
          %556 = "enzymexla.pointer2memref"(%162) : (!llvm.ptr) -> memref<?xf64>
          %557 = affine.load %556[%arg8 + %arg9 * 8 + %arg7 * 64] : memref<?xf64>
          affine.store %557, %alloca_1[%arg8, %arg9] : memref<8x8xf64>
          "enzymexla.barrier"(%arg8, %arg9, %c0) : (index, index, index) -> ()
          affine.for %arg10 = 0 to 8 {
            affine.for %arg11 = 0 to 8 {
              %558 = affine.load %alloca_2[%arg9] : memref<64xf64>
              %559 = affine.load %alloca_2[%arg11] : memref<64xf64>
              %560 = affine.load %alloca_2[%arg9 + 8] : memref<64xf64>
              %561 = affine.load %alloca_2[%arg11 + 8] : memref<64xf64>
              %562 = affine.load %alloca_2[%arg9 + 16] : memref<64xf64>
              %563 = affine.load %alloca_2[%arg11 + 16] : memref<64xf64>
              %564 = affine.load %alloca_2[%arg9 + 24] : memref<64xf64>
              %565 = affine.load %alloca_2[%arg11 + 24] : memref<64xf64>
              %566 = affine.load %alloca_2[%arg9 + 32] : memref<64xf64>
              %567 = affine.load %alloca_2[%arg11 + 32] : memref<64xf64>
              %568 = affine.load %alloca_2[%arg9 + 40] : memref<64xf64>
              %569 = affine.load %alloca_2[%arg11 + 40] : memref<64xf64>
              %570 = affine.load %alloca_2[%arg9 + 48] : memref<64xf64>
              %571 = affine.load %alloca_2[%arg11 + 48] : memref<64xf64>
              %572 = affine.load %alloca_2[%arg9 + 56] : memref<64xf64>
              %573 = affine.load %alloca_2[%arg11 + 56] : memref<64xf64>
              %574 = affine.for %arg12 = 0 to 8 iter_args(%arg13 = %cst) -> (f64) {
                %577 = affine.load %alloca_2[%arg8 + %arg12 * 8] : memref<64xf64>
                %578 = affine.load %alloca_2[%arg10 + %arg12 * 8] : memref<64xf64>
                %579 = arith.mulf %577, %578 fastmath<contract> : f64
                %580 = arith.mulf %558, %579 fastmath<contract> : f64
                %581 = arith.mulf %559, %580 fastmath<contract> : f64
                %582 = affine.load %alloca_1[%arg12, 0] : memref<8x8xf64>
                %583 = arith.mulf %582, %581 fastmath<contract> : f64
                %584 = arith.addf %arg13, %583 fastmath<contract> : f64
                %585 = arith.mulf %560, %579 fastmath<contract> : f64
                %586 = arith.mulf %561, %585 fastmath<contract> : f64
                %587 = affine.load %alloca_1[%arg12, 1] : memref<8x8xf64>
                %588 = arith.mulf %587, %586 fastmath<contract> : f64
                %589 = arith.addf %588, %584 fastmath<contract> : f64
                %590 = arith.mulf %562, %579 fastmath<contract> : f64
                %591 = arith.mulf %563, %590 fastmath<contract> : f64
                %592 = affine.load %alloca_1[%arg12, 2] : memref<8x8xf64>
                %593 = arith.mulf %591, %592 fastmath<contract> : f64
                %594 = arith.addf %593, %589 fastmath<contract> : f64
                %595 = arith.mulf %564, %579 fastmath<contract> : f64
                %596 = arith.mulf %565, %595 fastmath<contract> : f64
                %597 = affine.load %alloca_1[%arg12, 3] : memref<8x8xf64>
                %598 = arith.mulf %596, %597 fastmath<contract> : f64
                %599 = arith.addf %598, %594 fastmath<contract> : f64
                %600 = arith.mulf %566, %579 fastmath<contract> : f64
                %601 = arith.mulf %567, %600 fastmath<contract> : f64
                %602 = affine.load %alloca_1[%arg12, 4] : memref<8x8xf64>
                %603 = arith.mulf %601, %602 fastmath<contract> : f64
                %604 = arith.addf %603, %599 fastmath<contract> : f64
                %605 = arith.mulf %568, %579 fastmath<contract> : f64
                %606 = arith.mulf %569, %605 fastmath<contract> : f64
                %607 = affine.load %alloca_1[%arg12, 5] : memref<8x8xf64>
                %608 = arith.mulf %606, %607 fastmath<contract> : f64
                %609 = arith.addf %608, %604 fastmath<contract> : f64
                %610 = arith.mulf %570, %579 fastmath<contract> : f64
                %611 = arith.mulf %571, %610 fastmath<contract> : f64
                %612 = affine.load %alloca_1[%arg12, 6] : memref<8x8xf64>
                %613 = arith.mulf %611, %612 fastmath<contract> : f64
                %614 = arith.addf %613, %609 fastmath<contract> : f64
                %615 = arith.mulf %572, %579 fastmath<contract> : f64
                %616 = arith.mulf %573, %615 fastmath<contract> : f64
                %617 = affine.load %alloca_1[%arg12, 7] : memref<8x8xf64>
                %618 = arith.mulf %616, %617 fastmath<contract> : f64
                %619 = arith.addf %618, %614 fastmath<contract> : f64
                affine.yield %619 : f64
              }
              %575 = scf.if %arg4 -> (f64) {
                %577 = "enzymexla.pointer2memref"(%174) : (!llvm.ptr) -> memref<?xf64>
                %578 = affine.load %577[%arg8 + %arg9 * 8 + %arg10 * 64 + %arg7 * 4096 + %arg11 * 512] : memref<?xf64>
                %579 = arith.addf %574, %578 fastmath<contract> : f64
                scf.yield %579 : f64
              } else {
                scf.yield %574 : f64
              }
              %576 = "enzymexla.pointer2memref"(%174) : (!llvm.ptr) -> memref<?xf64>
              affine.store %575, %576[%arg8 + %arg9 * 8 + %arg10 * 64 + %arg7 * 4096 + %arg11 * 512] : memref<?xf64>
            }
          }
        }
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) {passthrough = ["mustprogress", "norecurse", ["no-trapping-math", "true"], ["polygeist.host_symbol", "_ZN4mfemL25__device_stub__CuKernel2DIZNS_8internal16EAMassAssemble2DILi8ELi8EEEviRKNS_5ArrayIdEERKNS_6VectorERS7_biiEUliE_EEviT_"], ["stack-protector-buffer-size", "8"], ["target-cpu", "sm_120"], "uniform-work-group-size"], target_cpu = "sm_120", target_features = #llvm.target_features<["+ptx88", "+sm_120"]>} : (index, index, index, index, index, index) -> index
    %183 = llvm.call @cudaGetLastError() : () -> i32
    %184 = arith.cmpi eq, %183, %c0_i32 : i32
    cf.cond_br %184, ^bb86, ^bb65
  ^bb65:  // pred: ^bb64
    llvm.call @_ZN4mfem15mfem_cuda_errorE9cudaErrorPKcS2_S2_i(%183, %12, %13, %14, %c785_i32) : (i32 {llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, i32 {llvm.noundef}) -> ()
    cf.br ^bb86
  ^bb66(%185: i32):  // 2 preds: ^bb62, ^bb77
    %186 = arith.index_cast %185 : i32 to index
    %187 = "enzymexla.pointer2memref"(%157) : (!llvm.ptr) -> memref<?xf64>
    %188 = affine.load %187[0] : memref<?xf64>
    affine.store %188, %alloca[0] : memref<64xf64>
    %189 = affine.load %187[1] : memref<?xf64>
    affine.store %189, %alloca[8] : memref<64xf64>
    %190 = affine.load %187[2] : memref<?xf64>
    affine.store %190, %alloca[16] : memref<64xf64>
    %191 = affine.load %187[3] : memref<?xf64>
    affine.store %191, %alloca[24] : memref<64xf64>
    %192 = affine.load %187[4] : memref<?xf64>
    affine.store %192, %alloca[32] : memref<64xf64>
    %193 = affine.load %187[5] : memref<?xf64>
    affine.store %193, %alloca[40] : memref<64xf64>
    %194 = affine.load %187[6] : memref<?xf64>
    affine.store %194, %alloca[48] : memref<64xf64>
    %195 = affine.load %187[7] : memref<?xf64>
    affine.store %195, %alloca[56] : memref<64xf64>
    %196 = affine.load %187[8] : memref<?xf64>
    affine.store %196, %alloca[1] : memref<64xf64>
    %197 = affine.load %187[9] : memref<?xf64>
    affine.store %197, %alloca[9] : memref<64xf64>
    %198 = affine.load %187[10] : memref<?xf64>
    affine.store %198, %alloca[17] : memref<64xf64>
    %199 = affine.load %187[11] : memref<?xf64>
    affine.store %199, %alloca[25] : memref<64xf64>
    %200 = affine.load %187[12] : memref<?xf64>
    affine.store %200, %alloca[33] : memref<64xf64>
    %201 = affine.load %187[13] : memref<?xf64>
    affine.store %201, %alloca[41] : memref<64xf64>
    %202 = affine.load %187[14] : memref<?xf64>
    affine.store %202, %alloca[49] : memref<64xf64>
    %203 = affine.load %187[15] : memref<?xf64>
    affine.store %203, %alloca[57] : memref<64xf64>
    %204 = affine.load %187[16] : memref<?xf64>
    affine.store %204, %alloca[2] : memref<64xf64>
    %205 = affine.load %187[17] : memref<?xf64>
    affine.store %205, %alloca[10] : memref<64xf64>
    %206 = affine.load %187[18] : memref<?xf64>
    affine.store %206, %alloca[18] : memref<64xf64>
    %207 = affine.load %187[19] : memref<?xf64>
    affine.store %207, %alloca[26] : memref<64xf64>
    %208 = affine.load %187[20] : memref<?xf64>
    affine.store %208, %alloca[34] : memref<64xf64>
    %209 = affine.load %187[21] : memref<?xf64>
    affine.store %209, %alloca[42] : memref<64xf64>
    %210 = affine.load %187[22] : memref<?xf64>
    affine.store %210, %alloca[50] : memref<64xf64>
    %211 = affine.load %187[23] : memref<?xf64>
    affine.store %211, %alloca[58] : memref<64xf64>
    %212 = affine.load %187[24] : memref<?xf64>
    affine.store %212, %alloca[3] : memref<64xf64>
    %213 = affine.load %187[25] : memref<?xf64>
    affine.store %213, %alloca[11] : memref<64xf64>
    %214 = affine.load %187[26] : memref<?xf64>
    affine.store %214, %alloca[19] : memref<64xf64>
    %215 = affine.load %187[27] : memref<?xf64>
    affine.store %215, %alloca[27] : memref<64xf64>
    %216 = affine.load %187[28] : memref<?xf64>
    affine.store %216, %alloca[35] : memref<64xf64>
    %217 = affine.load %187[29] : memref<?xf64>
    affine.store %217, %alloca[43] : memref<64xf64>
    %218 = affine.load %187[30] : memref<?xf64>
    affine.store %218, %alloca[51] : memref<64xf64>
    %219 = affine.load %187[31] : memref<?xf64>
    affine.store %219, %alloca[59] : memref<64xf64>
    %220 = affine.load %187[32] : memref<?xf64>
    affine.store %220, %alloca[4] : memref<64xf64>
    %221 = affine.load %187[33] : memref<?xf64>
    affine.store %221, %alloca[12] : memref<64xf64>
    %222 = affine.load %187[34] : memref<?xf64>
    affine.store %222, %alloca[20] : memref<64xf64>
    %223 = affine.load %187[35] : memref<?xf64>
    affine.store %223, %alloca[28] : memref<64xf64>
    %224 = affine.load %187[36] : memref<?xf64>
    affine.store %224, %alloca[36] : memref<64xf64>
    %225 = affine.load %187[37] : memref<?xf64>
    affine.store %225, %alloca[44] : memref<64xf64>
    %226 = affine.load %187[38] : memref<?xf64>
    affine.store %226, %alloca[52] : memref<64xf64>
    %227 = affine.load %187[39] : memref<?xf64>
    affine.store %227, %alloca[60] : memref<64xf64>
    %228 = affine.load %187[40] : memref<?xf64>
    affine.store %228, %alloca[5] : memref<64xf64>
    %229 = affine.load %187[41] : memref<?xf64>
    affine.store %229, %alloca[13] : memref<64xf64>
    %230 = affine.load %187[42] : memref<?xf64>
    affine.store %230, %alloca[21] : memref<64xf64>
    %231 = affine.load %187[43] : memref<?xf64>
    affine.store %231, %alloca[29] : memref<64xf64>
    %232 = affine.load %187[44] : memref<?xf64>
    affine.store %232, %alloca[37] : memref<64xf64>
    %233 = affine.load %187[45] : memref<?xf64>
    affine.store %233, %alloca[45] : memref<64xf64>
    %234 = affine.load %187[46] : memref<?xf64>
    affine.store %234, %alloca[53] : memref<64xf64>
    %235 = affine.load %187[47] : memref<?xf64>
    affine.store %235, %alloca[61] : memref<64xf64>
    %236 = affine.load %187[48] : memref<?xf64>
    affine.store %236, %alloca[6] : memref<64xf64>
    %237 = affine.load %187[49] : memref<?xf64>
    affine.store %237, %alloca[14] : memref<64xf64>
    %238 = affine.load %187[50] : memref<?xf64>
    affine.store %238, %alloca[22] : memref<64xf64>
    %239 = affine.load %187[51] : memref<?xf64>
    affine.store %239, %alloca[30] : memref<64xf64>
    %240 = affine.load %187[52] : memref<?xf64>
    affine.store %240, %alloca[38] : memref<64xf64>
    %241 = affine.load %187[53] : memref<?xf64>
    affine.store %241, %alloca[46] : memref<64xf64>
    %242 = affine.load %187[54] : memref<?xf64>
    affine.store %242, %alloca[54] : memref<64xf64>
    %243 = affine.load %187[55] : memref<?xf64>
    affine.store %243, %alloca[62] : memref<64xf64>
    %244 = affine.load %187[56] : memref<?xf64>
    affine.store %244, %alloca[7] : memref<64xf64>
    %245 = affine.load %187[57] : memref<?xf64>
    affine.store %245, %alloca[15] : memref<64xf64>
    %246 = affine.load %187[58] : memref<?xf64>
    affine.store %246, %alloca[23] : memref<64xf64>
    %247 = affine.load %187[59] : memref<?xf64>
    affine.store %247, %alloca[31] : memref<64xf64>
    %248 = affine.load %187[60] : memref<?xf64>
    affine.store %248, %alloca[39] : memref<64xf64>
    %249 = affine.load %187[61] : memref<?xf64>
    affine.store %249, %alloca[47] : memref<64xf64>
    %250 = affine.load %187[62] : memref<?xf64>
    affine.store %250, %alloca[55] : memref<64xf64>
    %251 = affine.load %187[63] : memref<?xf64>
    affine.store %251, %alloca[63] : memref<64xf64>
    %252 = "enzymexla.pointer2memref"(%162) : (!llvm.ptr) -> memref<?xf64>
    %253 = affine.load %252[symbol(%186) * 64] : memref<?xf64>
    affine.store %253, %alloca_0[0] : memref<64xf64>
    %254 = affine.load %252[symbol(%186) * 64 + 8] : memref<?xf64>
    affine.store %254, %alloca_0[1] : memref<64xf64>
    %255 = affine.load %252[symbol(%186) * 64 + 16] : memref<?xf64>
    affine.store %255, %alloca_0[2] : memref<64xf64>
    %256 = affine.load %252[symbol(%186) * 64 + 24] : memref<?xf64>
    affine.store %256, %alloca_0[3] : memref<64xf64>
    %257 = affine.load %252[symbol(%186) * 64 + 32] : memref<?xf64>
    affine.store %257, %alloca_0[4] : memref<64xf64>
    %258 = affine.load %252[symbol(%186) * 64 + 40] : memref<?xf64>
    affine.store %258, %alloca_0[5] : memref<64xf64>
    %259 = affine.load %252[symbol(%186) * 64 + 48] : memref<?xf64>
    affine.store %259, %alloca_0[6] : memref<64xf64>
    %260 = affine.load %252[symbol(%186) * 64 + 56] : memref<?xf64>
    affine.store %260, %alloca_0[7] : memref<64xf64>
    %261 = affine.load %252[symbol(%186) * 64 + 1] : memref<?xf64>
    affine.store %261, %alloca_0[8] : memref<64xf64>
    %262 = affine.load %252[symbol(%186) * 64 + 9] : memref<?xf64>
    affine.store %262, %alloca_0[9] : memref<64xf64>
    %263 = affine.load %252[symbol(%186) * 64 + 17] : memref<?xf64>
    affine.store %263, %alloca_0[10] : memref<64xf64>
    %264 = affine.load %252[symbol(%186) * 64 + 25] : memref<?xf64>
    affine.store %264, %alloca_0[11] : memref<64xf64>
    %265 = affine.load %252[symbol(%186) * 64 + 33] : memref<?xf64>
    affine.store %265, %alloca_0[12] : memref<64xf64>
    %266 = affine.load %252[symbol(%186) * 64 + 41] : memref<?xf64>
    affine.store %266, %alloca_0[13] : memref<64xf64>
    %267 = affine.load %252[symbol(%186) * 64 + 49] : memref<?xf64>
    affine.store %267, %alloca_0[14] : memref<64xf64>
    %268 = affine.load %252[symbol(%186) * 64 + 57] : memref<?xf64>
    affine.store %268, %alloca_0[15] : memref<64xf64>
    %269 = affine.load %252[symbol(%186) * 64 + 2] : memref<?xf64>
    affine.store %269, %alloca_0[16] : memref<64xf64>
    %270 = affine.load %252[symbol(%186) * 64 + 10] : memref<?xf64>
    affine.store %270, %alloca_0[17] : memref<64xf64>
    %271 = affine.load %252[symbol(%186) * 64 + 18] : memref<?xf64>
    affine.store %271, %alloca_0[18] : memref<64xf64>
    %272 = affine.load %252[symbol(%186) * 64 + 26] : memref<?xf64>
    affine.store %272, %alloca_0[19] : memref<64xf64>
    %273 = affine.load %252[symbol(%186) * 64 + 34] : memref<?xf64>
    affine.store %273, %alloca_0[20] : memref<64xf64>
    %274 = affine.load %252[symbol(%186) * 64 + 42] : memref<?xf64>
    affine.store %274, %alloca_0[21] : memref<64xf64>
    %275 = affine.load %252[symbol(%186) * 64 + 50] : memref<?xf64>
    affine.store %275, %alloca_0[22] : memref<64xf64>
    %276 = affine.load %252[symbol(%186) * 64 + 58] : memref<?xf64>
    affine.store %276, %alloca_0[23] : memref<64xf64>
    %277 = affine.load %252[symbol(%186) * 64 + 3] : memref<?xf64>
    affine.store %277, %alloca_0[24] : memref<64xf64>
    %278 = affine.load %252[symbol(%186) * 64 + 11] : memref<?xf64>
    affine.store %278, %alloca_0[25] : memref<64xf64>
    %279 = affine.load %252[symbol(%186) * 64 + 19] : memref<?xf64>
    affine.store %279, %alloca_0[26] : memref<64xf64>
    %280 = affine.load %252[symbol(%186) * 64 + 27] : memref<?xf64>
    affine.store %280, %alloca_0[27] : memref<64xf64>
    %281 = affine.load %252[symbol(%186) * 64 + 35] : memref<?xf64>
    affine.store %281, %alloca_0[28] : memref<64xf64>
    %282 = affine.load %252[symbol(%186) * 64 + 43] : memref<?xf64>
    affine.store %282, %alloca_0[29] : memref<64xf64>
    %283 = affine.load %252[symbol(%186) * 64 + 51] : memref<?xf64>
    affine.store %283, %alloca_0[30] : memref<64xf64>
    %284 = affine.load %252[symbol(%186) * 64 + 59] : memref<?xf64>
    affine.store %284, %alloca_0[31] : memref<64xf64>
    %285 = affine.load %252[symbol(%186) * 64 + 4] : memref<?xf64>
    affine.store %285, %alloca_0[32] : memref<64xf64>
    %286 = affine.load %252[symbol(%186) * 64 + 12] : memref<?xf64>
    affine.store %286, %alloca_0[33] : memref<64xf64>
    %287 = affine.load %252[symbol(%186) * 64 + 20] : memref<?xf64>
    affine.store %287, %alloca_0[34] : memref<64xf64>
    %288 = affine.load %252[symbol(%186) * 64 + 28] : memref<?xf64>
    affine.store %288, %alloca_0[35] : memref<64xf64>
    %289 = affine.load %252[symbol(%186) * 64 + 36] : memref<?xf64>
    affine.store %289, %alloca_0[36] : memref<64xf64>
    %290 = affine.load %252[symbol(%186) * 64 + 44] : memref<?xf64>
    affine.store %290, %alloca_0[37] : memref<64xf64>
    %291 = affine.load %252[symbol(%186) * 64 + 52] : memref<?xf64>
    affine.store %291, %alloca_0[38] : memref<64xf64>
    %292 = affine.load %252[symbol(%186) * 64 + 60] : memref<?xf64>
    affine.store %292, %alloca_0[39] : memref<64xf64>
    %293 = affine.load %252[symbol(%186) * 64 + 5] : memref<?xf64>
    affine.store %293, %alloca_0[40] : memref<64xf64>
    %294 = affine.load %252[symbol(%186) * 64 + 13] : memref<?xf64>
    affine.store %294, %alloca_0[41] : memref<64xf64>
    %295 = affine.load %252[symbol(%186) * 64 + 21] : memref<?xf64>
    affine.store %295, %alloca_0[42] : memref<64xf64>
    %296 = affine.load %252[symbol(%186) * 64 + 29] : memref<?xf64>
    affine.store %296, %alloca_0[43] : memref<64xf64>
    %297 = affine.load %252[symbol(%186) * 64 + 37] : memref<?xf64>
    affine.store %297, %alloca_0[44] : memref<64xf64>
    %298 = affine.load %252[symbol(%186) * 64 + 45] : memref<?xf64>
    affine.store %298, %alloca_0[45] : memref<64xf64>
    %299 = affine.load %252[symbol(%186) * 64 + 53] : memref<?xf64>
    affine.store %299, %alloca_0[46] : memref<64xf64>
    %300 = affine.load %252[symbol(%186) * 64 + 61] : memref<?xf64>
    affine.store %300, %alloca_0[47] : memref<64xf64>
    %301 = affine.load %252[symbol(%186) * 64 + 6] : memref<?xf64>
    affine.store %301, %alloca_0[48] : memref<64xf64>
    %302 = affine.load %252[symbol(%186) * 64 + 14] : memref<?xf64>
    affine.store %302, %alloca_0[49] : memref<64xf64>
    %303 = affine.load %252[symbol(%186) * 64 + 22] : memref<?xf64>
    affine.store %303, %alloca_0[50] : memref<64xf64>
    %304 = affine.load %252[symbol(%186) * 64 + 30] : memref<?xf64>
    affine.store %304, %alloca_0[51] : memref<64xf64>
    %305 = affine.load %252[symbol(%186) * 64 + 38] : memref<?xf64>
    affine.store %305, %alloca_0[52] : memref<64xf64>
    %306 = affine.load %252[symbol(%186) * 64 + 46] : memref<?xf64>
    affine.store %306, %alloca_0[53] : memref<64xf64>
    %307 = affine.load %252[symbol(%186) * 64 + 54] : memref<?xf64>
    affine.store %307, %alloca_0[54] : memref<64xf64>
    %308 = affine.load %252[symbol(%186) * 64 + 62] : memref<?xf64>
    affine.store %308, %alloca_0[55] : memref<64xf64>
    %309 = affine.load %252[symbol(%186) * 64 + 7] : memref<?xf64>
    affine.store %309, %alloca_0[56] : memref<64xf64>
    %310 = affine.load %252[symbol(%186) * 64 + 15] : memref<?xf64>
    affine.store %310, %alloca_0[57] : memref<64xf64>
    %311 = affine.load %252[symbol(%186) * 64 + 23] : memref<?xf64>
    affine.store %311, %alloca_0[58] : memref<64xf64>
    %312 = affine.load %252[symbol(%186) * 64 + 31] : memref<?xf64>
    affine.store %312, %alloca_0[59] : memref<64xf64>
    %313 = affine.load %252[symbol(%186) * 64 + 39] : memref<?xf64>
    affine.store %313, %alloca_0[60] : memref<64xf64>
    %314 = affine.load %252[symbol(%186) * 64 + 47] : memref<?xf64>
    affine.store %314, %alloca_0[61] : memref<64xf64>
    %315 = affine.load %252[symbol(%186) * 64 + 55] : memref<?xf64>
    affine.store %315, %alloca_0[62] : memref<64xf64>
    %316 = affine.load %252[symbol(%186) * 64 + 63] : memref<?xf64>
    affine.store %316, %alloca_0[63] : memref<64xf64>
    %317 = arith.muli %185, %c8_i32 overflow<nsw> : i32
    %318 = arith.extsi %317 : i32 to i64
    cf.cond_br %arg4, ^bb67(%c0_i64 : i64), ^bb76(%c0_i64 : i64)
  ^bb67(%319: i64):  // 2 preds: ^bb66, ^bb75
    %320 = arith.index_cast %319 : i64 to index
    cf.br ^bb68(%c0_i64 : i64)
  ^bb68(%321: i64):  // 2 preds: ^bb67, ^bb74
    %322 = arith.index_cast %321 : i64 to index
    cf.br ^bb69(%c0_i64 : i64)
  ^bb69(%323: i64):  // 2 preds: ^bb68, ^bb73
    %324 = arith.index_cast %323 : i64 to index
    cf.br ^bb70(%c0_i64 : i64)
  ^bb70(%325: i64):  // 2 preds: ^bb69, ^bb71
    %326 = arith.index_cast %325 : i64 to index
    %327 = affine.load %alloca[symbol(%322)] : memref<64xf64>
    %328 = affine.load %alloca[symbol(%326)] : memref<64xf64>
    %329 = affine.load %alloca[symbol(%322) + 8] : memref<64xf64>
    %330 = affine.load %alloca[symbol(%326) + 8] : memref<64xf64>
    %331 = affine.load %alloca[symbol(%322) + 16] : memref<64xf64>
    %332 = affine.load %alloca[symbol(%326) + 16] : memref<64xf64>
    %333 = affine.load %alloca[symbol(%322) + 24] : memref<64xf64>
    %334 = affine.load %alloca[symbol(%326) + 24] : memref<64xf64>
    %335 = affine.load %alloca[symbol(%322) + 32] : memref<64xf64>
    %336 = affine.load %alloca[symbol(%326) + 32] : memref<64xf64>
    %337 = affine.load %alloca[symbol(%322) + 40] : memref<64xf64>
    %338 = affine.load %alloca[symbol(%326) + 40] : memref<64xf64>
    %339 = affine.load %alloca[symbol(%322) + 48] : memref<64xf64>
    %340 = affine.load %alloca[symbol(%326) + 48] : memref<64xf64>
    %341 = affine.load %alloca[symbol(%322) + 56] : memref<64xf64>
    %342 = affine.load %alloca[symbol(%326) + 56] : memref<64xf64>
    cf.br ^bb72(%c0_i64, %cst : i64, f64)
  ^bb71:  // pred: ^bb72
    %343 = arith.addi %325, %318 overflow<nsw> : i64
    %344 = arith.muli %343, %c8_i64 overflow<nsw> : i64
    %345 = arith.addi %344, %323 overflow<nsw> : i64
    %346 = arith.muli %345, %c8_i64 overflow<nsw> : i64
    %347 = arith.index_cast %346 : i64 to index
    %348 = arith.addi %347, %322 : index
    %349 = "enzymexla.pointer2memref"(%174) : (!llvm.ptr) -> memref<?xf64>
    %350 = arith.muli %348, %c8 overflow<nsw> : index
    %351 = arith.addi %350, %320 : index
    %352 = affine.load %349[symbol(%351)] : memref<?xf64>
    %353 = arith.addf %396, %352 : f64
    affine.store %353, %349[symbol(%351)] : memref<?xf64>
    %354 = arith.addi %325, %c1_i64 overflow<nsw, nuw> : i64
    %355 = arith.cmpi eq, %354, %c8_i64 : i64
    cf.cond_br %355, ^bb73, ^bb70(%354 : i64)
  ^bb72(%356: i64, %357: f64):  // 2 preds: ^bb70, ^bb72
    %358 = arith.index_cast %356 : i64 to index
    %359 = arith.muli %358, %c8 overflow<nsw> : index
    %360 = arith.addi %320, %359 : index
    %361 = affine.load %alloca[symbol(%360)] : memref<64xf64>
    %362 = arith.addi %324, %359 : index
    %363 = affine.load %alloca[symbol(%362)] : memref<64xf64>
    %364 = arith.mulf %361, %363 : f64
    %365 = arith.mulf %364, %327 : f64
    %366 = arith.mulf %365, %328 : f64
    %367 = affine.load %alloca_0[symbol(%358) * 8] : memref<64xf64>
    %368 = enzymexla.math.fmuladd %366, %367, %357 : f64
    %369 = arith.mulf %364, %329 : f64
    %370 = arith.mulf %369, %330 : f64
    %371 = affine.load %alloca_0[symbol(%358) * 8 + 1] : memref<64xf64>
    %372 = enzymexla.math.fmuladd %370, %371, %368 : f64
    %373 = arith.mulf %364, %331 : f64
    %374 = arith.mulf %373, %332 : f64
    %375 = affine.load %alloca_0[symbol(%358) * 8 + 2] : memref<64xf64>
    %376 = enzymexla.math.fmuladd %374, %375, %372 : f64
    %377 = arith.mulf %364, %333 : f64
    %378 = arith.mulf %377, %334 : f64
    %379 = affine.load %alloca_0[symbol(%358) * 8 + 3] : memref<64xf64>
    %380 = enzymexla.math.fmuladd %378, %379, %376 : f64
    %381 = arith.mulf %364, %335 : f64
    %382 = arith.mulf %381, %336 : f64
    %383 = affine.load %alloca_0[symbol(%358) * 8 + 4] : memref<64xf64>
    %384 = enzymexla.math.fmuladd %382, %383, %380 : f64
    %385 = arith.mulf %364, %337 : f64
    %386 = arith.mulf %385, %338 : f64
    %387 = affine.load %alloca_0[symbol(%358) * 8 + 5] : memref<64xf64>
    %388 = enzymexla.math.fmuladd %386, %387, %384 : f64
    %389 = arith.mulf %364, %339 : f64
    %390 = arith.mulf %389, %340 : f64
    %391 = affine.load %alloca_0[symbol(%358) * 8 + 6] : memref<64xf64>
    %392 = enzymexla.math.fmuladd %390, %391, %388 : f64
    %393 = arith.mulf %364, %341 : f64
    %394 = arith.mulf %393, %342 : f64
    %395 = affine.load %alloca_0[symbol(%358) * 8 + 7] : memref<64xf64>
    %396 = enzymexla.math.fmuladd %394, %395, %392 : f64
    %397 = arith.addi %356, %c1_i64 overflow<nsw, nuw> : i64
    %398 = arith.cmpi eq, %397, %c8_i64 : i64
    cf.cond_br %398, ^bb71, ^bb72(%397, %396 : i64, f64)
  ^bb73:  // pred: ^bb71
    %399 = arith.addi %323, %c1_i64 overflow<nsw, nuw> : i64
    %400 = arith.cmpi eq, %399, %c8_i64 : i64
    cf.cond_br %400, ^bb74, ^bb69(%399 : i64)
  ^bb74:  // pred: ^bb73
    %401 = arith.addi %321, %c1_i64 overflow<nsw, nuw> : i64
    %402 = arith.cmpi eq, %401, %c8_i64 : i64
    cf.cond_br %402, ^bb75, ^bb68(%401 : i64)
  ^bb75:  // pred: ^bb74
    %403 = arith.addi %319, %c1_i64 overflow<nsw, nuw> : i64
    %404 = arith.cmpi eq, %403, %c8_i64 : i64
    cf.cond_br %404, ^bb77, ^bb67(%403 : i64)
  ^bb76(%405: i64):  // 2 preds: ^bb66, ^bb79
    %406 = arith.index_cast %405 : i64 to index
    cf.br ^bb78(%c0_i64 : i64)
  ^bb77:  // 2 preds: ^bb75, ^bb79
    %407 = arith.addi %185, %c1_i32 overflow<nsw, nuw> : i32
    %408 = arith.cmpi eq, %407, %arg0 : i32
    cf.cond_br %408, ^bb86, ^bb66(%407 : i32)
  ^bb78(%409: i64):  // 2 preds: ^bb76, ^bb81
    %410 = arith.index_cast %409 : i64 to index
    cf.br ^bb80(%c0_i64 : i64)
  ^bb79:  // pred: ^bb81
    %411 = arith.addi %405, %c1_i64 overflow<nsw, nuw> : i64
    %412 = arith.cmpi eq, %411, %c8_i64 : i64
    cf.cond_br %412, ^bb77, ^bb76(%411 : i64)
  ^bb80(%413: i64):  // 2 preds: ^bb78, ^bb83
    %414 = arith.index_cast %413 : i64 to index
    cf.br ^bb82(%c0_i64 : i64)
  ^bb81:  // pred: ^bb83
    %415 = arith.addi %409, %c1_i64 overflow<nsw, nuw> : i64
    %416 = arith.cmpi eq, %415, %c8_i64 : i64
    cf.cond_br %416, ^bb79, ^bb78(%415 : i64)
  ^bb82(%417: i64):  // 2 preds: ^bb80, ^bb85
    %418 = arith.index_cast %417 : i64 to index
    %419 = affine.load %alloca[symbol(%410)] : memref<64xf64>
    %420 = affine.load %alloca[symbol(%418)] : memref<64xf64>
    %421 = affine.load %alloca[symbol(%410) + 8] : memref<64xf64>
    %422 = affine.load %alloca[symbol(%418) + 8] : memref<64xf64>
    %423 = affine.load %alloca[symbol(%410) + 16] : memref<64xf64>
    %424 = affine.load %alloca[symbol(%418) + 16] : memref<64xf64>
    %425 = affine.load %alloca[symbol(%410) + 24] : memref<64xf64>
    %426 = affine.load %alloca[symbol(%418) + 24] : memref<64xf64>
    %427 = affine.load %alloca[symbol(%410) + 32] : memref<64xf64>
    %428 = affine.load %alloca[symbol(%418) + 32] : memref<64xf64>
    %429 = affine.load %alloca[symbol(%410) + 40] : memref<64xf64>
    %430 = affine.load %alloca[symbol(%418) + 40] : memref<64xf64>
    %431 = affine.load %alloca[symbol(%410) + 48] : memref<64xf64>
    %432 = affine.load %alloca[symbol(%418) + 48] : memref<64xf64>
    %433 = affine.load %alloca[symbol(%410) + 56] : memref<64xf64>
    %434 = affine.load %alloca[symbol(%418) + 56] : memref<64xf64>
    cf.br ^bb84(%c0_i64, %cst : i64, f64)
  ^bb83:  // pred: ^bb85
    %435 = arith.addi %413, %c1_i64 overflow<nsw, nuw> : i64
    %436 = arith.cmpi eq, %435, %c8_i64 : i64
    cf.cond_br %436, ^bb81, ^bb80(%435 : i64)
  ^bb84(%437: i64, %438: f64):  // 2 preds: ^bb82, ^bb84
    %439 = arith.index_cast %437 : i64 to index
    %440 = arith.muli %439, %c8 overflow<nsw> : index
    %441 = arith.addi %406, %440 : index
    %442 = affine.load %alloca[symbol(%441)] : memref<64xf64>
    %443 = arith.addi %414, %440 : index
    %444 = affine.load %alloca[symbol(%443)] : memref<64xf64>
    %445 = arith.mulf %442, %444 : f64
    %446 = arith.mulf %445, %419 : f64
    %447 = arith.mulf %446, %420 : f64
    %448 = affine.load %alloca_0[symbol(%439) * 8] : memref<64xf64>
    %449 = enzymexla.math.fmuladd %447, %448, %438 : f64
    %450 = arith.mulf %445, %421 : f64
    %451 = arith.mulf %450, %422 : f64
    %452 = affine.load %alloca_0[symbol(%439) * 8 + 1] : memref<64xf64>
    %453 = enzymexla.math.fmuladd %451, %452, %449 : f64
    %454 = arith.mulf %445, %423 : f64
    %455 = arith.mulf %454, %424 : f64
    %456 = affine.load %alloca_0[symbol(%439) * 8 + 2] : memref<64xf64>
    %457 = enzymexla.math.fmuladd %455, %456, %453 : f64
    %458 = arith.mulf %445, %425 : f64
    %459 = arith.mulf %458, %426 : f64
    %460 = affine.load %alloca_0[symbol(%439) * 8 + 3] : memref<64xf64>
    %461 = enzymexla.math.fmuladd %459, %460, %457 : f64
    %462 = arith.mulf %445, %427 : f64
    %463 = arith.mulf %462, %428 : f64
    %464 = affine.load %alloca_0[symbol(%439) * 8 + 4] : memref<64xf64>
    %465 = enzymexla.math.fmuladd %463, %464, %461 : f64
    %466 = arith.mulf %445, %429 : f64
    %467 = arith.mulf %466, %430 : f64
    %468 = affine.load %alloca_0[symbol(%439) * 8 + 5] : memref<64xf64>
    %469 = enzymexla.math.fmuladd %467, %468, %465 : f64
    %470 = arith.mulf %445, %431 : f64
    %471 = arith.mulf %470, %432 : f64
    %472 = affine.load %alloca_0[symbol(%439) * 8 + 6] : memref<64xf64>
    %473 = enzymexla.math.fmuladd %471, %472, %469 : f64
    %474 = arith.mulf %445, %433 : f64
    %475 = arith.mulf %474, %434 : f64
    %476 = affine.load %alloca_0[symbol(%439) * 8 + 7] : memref<64xf64>
    %477 = enzymexla.math.fmuladd %475, %476, %473 : f64
    %478 = arith.addi %437, %c1_i64 overflow<nsw, nuw> : i64
    %479 = arith.cmpi eq, %478, %c8_i64 : i64
    cf.cond_br %479, ^bb85, ^bb84(%478, %477 : i64, f64)
  ^bb85:  // pred: ^bb84
    %480 = arith.addi %417, %318 overflow<nsw> : i64
    %481 = arith.muli %480, %c8_i64 overflow<nsw> : i64
    %482 = arith.addi %481, %413 overflow<nsw> : i64
    %483 = arith.muli %482, %c8_i64 overflow<nsw> : i64
    %484 = arith.index_cast %483 : i64 to index
    %485 = arith.addi %484, %410 : index
    %486 = "enzymexla.pointer2memref"(%174) : (!llvm.ptr) -> memref<?xf64>
    %487 = arith.muli %485, %c8 overflow<nsw> : index
    %488 = arith.addi %487, %406 : index
    affine.store %477, %486[symbol(%488)] : memref<?xf64>
    %489 = arith.addi %417, %c1_i64 overflow<nsw, nuw> : i64
    %490 = arith.cmpi eq, %489, %c8_i64 : i64
    cf.cond_br %490, ^bb83, ^bb82(%489 : i64)
  ^bb86:  // 5 preds: ^bb62, ^bb63, ^bb64, ^bb65, ^bb77
    llvm.return
  }
  llvm.func unnamed_addr @_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEC1Ev(!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 112 : i64, llvm.nonnull, llvm.noundef}) attributes {alignment = 2 : i64, passthrough = ["mustprogress", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], "uniform-work-group-size"], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, uwtable_kind = #llvm.uwtableKind<async>}
  llvm.func local_unnamed_addr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_c(!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, i8 {llvm.noundef, llvm.signext}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}) attributes {inline_hint, passthrough = ["mustprogress", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], "uniform-work-group-size"], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, uwtable_kind = #llvm.uwtableKind<async>}
  llvm.func local_unnamed_addr @_ZNSolsEi(!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, i32 {llvm.noundef}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}) attributes {passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], "uniform-work-group-size"], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @_ZNKSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEE3strEv(!llvm.ptr {llvm.align = 8 : i64, llvm.dead_on_unwind, llvm.noalias, llvm.sret = !llvm.struct<"class.std::__cxx11::basic_string.1", (struct<"struct.std::__cxx11::basic_string<char>::_Alloc_hider.1", (ptr)>, i64, struct<"union.anon.1", (i64, array<8 x i8>)>)>, llvm.writable}, !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 112 : i64, llvm.nonnull, llvm.noundef}) attributes {alignment = 2 : i64, passthrough = ["mustprogress", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], "uniform-work-group-size"], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, uwtable_kind = #llvm.uwtableKind<async>}
  llvm.func unnamed_addr @_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev(!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 112 : i64, llvm.nonnull, llvm.noundef}) attributes {alignment = 2 : i64, no_unwind, passthrough = ["mustprogress", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], "uniform-work-group-size"], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, uwtable_kind = #llvm.uwtableKind<async>}
  llvm.func local_unnamed_addr @_ZNK4mfem6MemoryIdE4ReadENS_11MemoryClassEi(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 20 : i64, llvm.nonnull, llvm.noundef}, %arg1: i32 {llvm.noundef}, %arg2: i32 {llvm.noundef}) -> (!llvm.ptr {llvm.noundef}) attributes {alignment = 2 : i64, dso_local, no_inline, passthrough = ["mustprogress", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], "uniform-work-group-size"], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uwtable_kind = #llvm.uwtableKind<async>}
  llvm.func local_unnamed_addr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.noundef}, i64 {llvm.noundef}) -> (!llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}) attributes {passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], "uniform-work-group-size"], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @__cxa_guard_acquire(!llvm.ptr) -> i32 attributes {no_unwind, passthrough = ["nofree"], sym_visibility = "private"}
  llvm.func local_unnamed_addr @__cxa_guard_release(!llvm.ptr) attributes {no_unwind, passthrough = ["nofree"], sym_visibility = "private"}
  llvm.func local_unnamed_addr @cudaGetLastError() -> i32 attributes {passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], "uniform-work-group-size"], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @_ZN4mfem15mfem_cuda_errorE9cudaErrorPKcS2_S2_i(i32 {llvm.noundef}, !llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.noundef}, i32 {llvm.noundef}) attributes {passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], "uniform-work-group-size"], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
}

// CHECK: func.func private @rxla$raised_0(
// CHECK-NOT: tensor<8xi64>) -> tensor<8x8x1xf64>
