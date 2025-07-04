// RUN: enzymexlamlir-opt --affine-cfg --split-input-file --allow-unregistered-dialect %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.target_triple = "x86_64-unknown-linux-gnu"} {

func.func @f(%119: memref<?xf64>, %arg0: i32 {llvm.noundef}, %arg1: i32 {llvm.noundef}, %arg2: i32 {llvm.noundef}, %arg3: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 24 : i64, llvm.nocapture, llvm.nofree, llvm.nonnull, llvm.noundef, llvm.readonly}, %arg4: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 24 : i64, llvm.nocapture, llvm.nofree, llvm.nonnull, llvm.noundef, llvm.readonly}) attributes {dso_local, no_unwind, passthrough = ["mustprogress", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uwtable_kind = #llvm.uwtableKind<async>} {
  %c-1 = arith.constant -1 : index
  %c229_i32 = arith.constant 229 : i32
  %c228_i32 = arith.constant 228 : i32
  %c227_i32 = arith.constant 227 : i32
  %c226_i32 = arith.constant 226 : i32
  %c221_i32 = arith.constant 221 : i32
  %c220_i32 = arith.constant 220 : i32
  %c206_i32 = arith.constant 206 : i32
  %c216_i32 = arith.constant 216 : i32
  %c32_i64 = arith.constant 32 : i64
  %c4294967297_i64 = arith.constant 4294967297 : i64
  %c32_i32 = arith.constant 32 : i32
  %c31_i32 = arith.constant 31 : i32
  %cst = arith.constant 1.000000e+00 : f64
  %cst_0 = arith.constant -1.000000e+00 : f64
  %c196_i32 = arith.constant 196 : i32
  %c195_i32 = arith.constant 195 : i32
  %c194_i32 = arith.constant 194 : i32
  %cst_1 = arith.constant 9.9999999999999998E-13 : f64
  %c1_i64 = arith.constant 1 : i64
  %c0_i64 = arith.constant 0 : i64
  %c3_i64 = arith.constant 3 : i64
  %c139_i32 = arith.constant 139 : i32
  %4 = llvm.mlir.zero : !llvm.ptr
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %c5_i32 = arith.constant 5 : i32
  %cst_2 = arith.constant 0.000000e+00 : f64
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %5 = ub.poison : i32
  %c2_i32 = arith.constant 2 : i32
  %c3_i32 = arith.constant 3 : i32
  %c-1_i64 = arith.constant -1 : i64
  %c-1_i32 = arith.constant -1 : i32
  %6 = ub.poison : f64
  %7 = llvm.alloca %c1_i32 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %8 = llvm.alloca %c1_i32 x f64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %9 = llvm.alloca %c1_i32 x f64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
  llvm.intr.lifetime.start 8, %7 : !llvm.ptr
  %10 = "enzymexla.pointer2memref"(%7) : (!llvm.ptr) -> memref<?x!llvm.ptr>
  affine.store %4, %10[0] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "p1 _ZTS13cublasContext", members = {<#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, 0>}>, access_type = <id = "p1 _ZTS13cublasContext", members = {<#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, 0>}>, offset = 0>]} : memref<?x!llvm.ptr>
  %11 = arith.cmpi eq, %arg2, %c0_i32 : i32
  %13 = arith.index_castui %c0_i32 : i32 to index
  %14 = arith.cmpi eq, %13, %c0 : index
    %18 = arith.subi %arg0, %arg1 : i32
    %19 = arith.extsi %18 : i32 to i64
    %20 = arith.extsi %arg1 : i32 to i64
    %21 = arith.shli %20, %c3_i64 : i64
    %22 = arith.muli %21, %19 : i64
    %23 = arith.index_cast %22 : i64 to index
    %memref = gpu.alloc  (%23) : memref<?xi8, 1>
    %24 = "enzymexla.memref2pointer"(%memref) : (memref<?xi8, 1>) -> !llvm.ptr
    %25 = arith.index_cast %22 : i64 to index
    %memref_3 = gpu.alloc  (%25) : memref<?xi8, 1>
    %26 = "enzymexla.memref2pointer"(%memref_3) : (memref<?xi8, 1>) -> !llvm.ptr
    %27 = arith.shli %19, %c3_i64 : i64
    %28 = arith.muli %27, %19 : i64
    %29 = arith.index_cast %28 : i64 to index
    %memref_4 = gpu.alloc  (%29) : memref<?xi8, 1>
    %30 = "enzymexla.memref2pointer"(%memref_4) : (memref<?xi8, 1>) -> !llvm.ptr
    %31 = arith.cmpi sgt, %arg0, %c0_i32 : i32
      %38 = arith.extui %arg0 {nonNeg} : i32 to i64
      %39 = arith.extui %arg0 {nonNeg} : i32 to i64
      %40 = arith.shli %39, %c3_i64 : i64
      %41 = arith.extsi %arg1 : i32 to i64
      %42 = arith.extui %arg0 {nonNeg} : i32 to i64
      %43 = arith.extui %arg0 {nonNeg} : i32 to i64
      %44 = arith.extui %arg0 {nonNeg} : i32 to i64
        %48 = arith.trunci %c1_i64 : i64 to i32
        %49 = arith.subi %arg0, %48 : i32
        %50 = arith.minsi %49, %arg1 : i32
        %51 = arith.trunci %c1_i64 : i64 to i32
        %52 = arith.addi %50, %51 : i32
        %53 = arith.cmpi sgt, %50, %c0_i32 : i32
          %65 = "enzymexla.pointer2memref"(%arg3) : (!llvm.ptr) -> memref<?x!llvm.ptr>
          %66 = affine.load %65[0] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "_ZTSNSt12_Vector_baseIdSaIdEE17_Vector_impl_dataE", members = {<#llvm.tbaa_type_desc<id = "p1 double", members = {<#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "p1 double", members = {<#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, 0>}>, 8>, <#llvm.tbaa_type_desc<id = "p1 double", members = {<#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, 0>}>, 16>}>, access_type = <id = "p1 double", members = {<#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, 0>}>, offset = 0>]} : memref<?x!llvm.ptr>
          %67 = "enzymexla.pointer2memref"(%arg4) : (!llvm.ptr) -> memref<?x!llvm.ptr>
          %68 = affine.load %67[0] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "_ZTSNSt12_Vector_baseIiSaIiEE17_Vector_impl_dataE", members = {<#llvm.tbaa_type_desc<id = "p1 int", members = {<#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, 0>}>, 0>, <#llvm.tbaa_type_desc<id = "p1 int", members = {<#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, 0>}>, 8>, <#llvm.tbaa_type_desc<id = "p1 int", members = {<#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, 0>}>, 16>}>, access_type = <id = "p1 int", members = {<#llvm.tbaa_type_desc<id = "any pointer", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, 0>}>, offset = 0>]} : memref<?x!llvm.ptr>
          %69 = arith.extsi %52 : i32 to i64
          %70 = arith.addi %c1_i64, %c1_i64 : i64
            %72 = arith.addi %c-1_i64, %c-1_i64 : i64
            %73 = arith.index_cast %c-1_i64 : i64 to index
            %74 = arith.addi %73, %c-1 : index
            %75 = arith.index_cast %c-1_i64 : i64 to index
            %76 = arith.addi %75, %c-1 : index
            %77 = arith.index_cast %c-1_i64 : i64 to index
            %78 = arith.addi %77, %c-1 : index
            %79 = arith.muli %72, %38 : i64
            %80 = arith.index_cast %79 : i64 to index
            %81 = arith.index_cast %79 : i64 to index
            %82 = arith.index_cast %79 : i64 to index
            %83 = arith.index_cast %79 : i64 to index
            %84 = arith.cmpi slt, %c-1_i64, %42 : i64
            %85 = arith.trunci %72 : i64 to i32
            %86 = scf.if %84 -> (i32) {
              %101 = "enzymexla.pointer2memref"(%66) : (!llvm.ptr) -> memref<?xf64>
              %102 = arith.addi %74, %80 : index
              %103 = memref.load %101[%102] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : memref<?xf64>
              %104 = math.absf %103 : f64
              %105 = arith.trunci %72 : i64 to i32
              scf.yield %105 : i32
            } else {
              scf.yield %85 : i32
            }
              %103 = arith.index_cast %86 : i32 to index
              %106 = scf.while (%arg9 = %c1_i64) : (i64) -> i64 {
                %110 = arith.index_cast %arg9 : i64 to index
                %117 = arith.addi %110, %103 : index
                %118 = memref.load %119[%117] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, access_type = <id = "double", members = {<#llvm.tbaa_type_desc<id = "omnipotent char", members = {<#llvm.tbaa_root<id = "Simple C++ TBAA">, 0>}>, 0>}>, offset = 0>]} : memref<?xf64>
                affine.store %118, %119[0] : memref<?xf64>
                %121 = arith.cmpi ne, %arg9, %c1_i64 : i64
                scf.condition(%121) %arg9 : i64
              } do {
              ^bb0(%arg9: i64):
                %107 = arith.addi %arg9, %c1_i64 : i64
                scf.yield %107 : i64
              }
            %91 = "enzymexla.pointer2memref"(%66) : (!llvm.ptr) -> memref<?xf64>
            %100 = arith.addi %c1_i64, %c1_i64 : i64
  llvm.return
}
}

// CHECK:  func.func @f(%arg0: memref<?xf64>, %arg1: i32 {llvm.noundef}, %arg2: i32 {llvm.noundef}, %arg3: i32 {llvm.noundef}, %arg4: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 24 : i64, llvm.nocapture, llvm.nofree, llvm.nonnull, llvm.noundef, llvm.readonly}, %arg5: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 24 : i64, llvm.nocapture, llvm.nofree, llvm.nonnull, llvm.noundef, llvm.readonly}) attributes {dso_local, no_unwind, passthrough = ["mustprogress", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uwtable_kind = #llvm.uwtableKind<async>} {
// CHECK-NEXT:    %c-2_i32 = arith.constant -2 : i32
// CHECK-NEXT:    %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:    %0 = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:    %c-1_i64 = arith.constant -1 : i64
// CHECK-NEXT:    %1 = llvm.alloca %c1_i32 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:    llvm.intr.lifetime.start 8, %1 : !llvm.ptr
// CHECK-NEXT:    %2 = "enzymexla.pointer2memref"(%1) : (!llvm.ptr) -> memref<?x!llvm.ptr>
// CHECK-NEXT:    affine.store %0, %2[0] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag1]} : memref<?x!llvm.ptr>
// CHECK-NEXT:    %3 = arith.extui %arg1 {nonNeg} : i32 to i64
// CHECK-NEXT:    %4 = arith.cmpi sgt, %3, %c-1_i64 : i64
// CHECK-NEXT:    %5 = scf.if %4 -> (i32) {
// CHECK-NEXT:      scf.yield %c-2_i32 : i32
// CHECK-NEXT:    } else {
// CHECK-NEXT:      scf.yield %c-2_i32 : i32
// CHECK-NEXT:    }
// CHECK-NEXT:    %6 = arith.index_cast %5 : i32 to index
// CHECK-NEXT:    %7 = scf.while (%arg6 = %c1_i64) : (i64) -> i64 {
// CHECK-NEXT:      %8 = arith.index_cast %arg6 : i64 to index
// CHECK-NEXT:      %9 = arith.addi %8, %6 : index
// CHECK-NEXT:      %10 = memref.load %arg0[%9] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<?xf64>
// CHECK-NEXT:      affine.store %10, %arg0[0] : memref<?xf64>
// CHECK-NEXT:      %11 = arith.cmpi ne, %arg6, %c1_i64 : i64
// CHECK-NEXT:      scf.condition(%11) %arg6 : i64
// CHECK-NEXT:    } do {
// CHECK-NEXT:    ^bb0(%arg6: i64):
// CHECK-NEXT:      %8 = arith.addi %arg6, %c1_i64 : i64
// CHECK-NEXT:      scf.yield %8 : i64
// CHECK-NEXT:    }
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }
