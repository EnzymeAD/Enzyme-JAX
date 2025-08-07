// RUN: enzymexlamlir-opt --llvm-to-affine-access %s | FileCheck %s

#alias_scope_domain = #llvm.alias_scope_domain<id = distinct[0]<>, description = "_ZL40Inner_CalcTimeConstraintsForElems_kernelILi128EEviddPiPdS1_S1_S1_S1_">
#tbaa_root = #llvm.tbaa_root<id = "Simple C++ TBAA">
#alias_scope = #llvm.alias_scope<id = distinct[1]<>, domain = #alias_scope_domain, description = "_ZL40Inner_CalcTimeConstraintsForElems_kernelILi128EEviddPiPdS1_S1_S1_S1_: %matElemlist">
#alias_scope1 = #llvm.alias_scope<id = distinct[2]<>, domain = #alias_scope_domain, description = "_ZL40Inner_CalcTimeConstraintsForElems_kernelILi128EEviddPiPdS1_S1_S1_S1_: %ss">
#alias_scope2 = #llvm.alias_scope<id = distinct[3]<>, domain = #alias_scope_domain, description = "_ZL40Inner_CalcTimeConstraintsForElems_kernelILi128EEviddPiPdS1_S1_S1_S1_: %vdov">
#alias_scope3 = #llvm.alias_scope<id = distinct[4]<>, domain = #alias_scope_domain, description = "_ZL40Inner_CalcTimeConstraintsForElems_kernelILi128EEviddPiPdS1_S1_S1_S1_: %arealg">
#alias_scope4 = #llvm.alias_scope<id = distinct[5]<>, domain = #alias_scope_domain, description = "_ZL40Inner_CalcTimeConstraintsForElems_kernelILi128EEviddPiPdS1_S1_S1_S1_: %dev_mindtcourant">
#alias_scope5 = #llvm.alias_scope<id = distinct[6]<>, domain = #alias_scope_domain, description = "_ZL40Inner_CalcTimeConstraintsForElems_kernelILi128EEviddPiPdS1_S1_S1_S1_: %dev_mindthydro">
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root, 0>}>
#tbaa_type_desc1 = #llvm.tbaa_type_desc<id = "int", members = {<#tbaa_type_desc, 0>}>
#tbaa_type_desc2 = #llvm.tbaa_type_desc<id = "double", members = {<#tbaa_type_desc, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc1, access_type = #tbaa_type_desc1, offset = 0>
#tbaa_tag1 = #llvm.tbaa_tag<base_type = #tbaa_type_desc2, access_type = #tbaa_type_desc2, offset = 0>
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<6> = dense<32> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.legal_int_widths" = array<i32: 16, 32, 64>>, llvm.target_triple = "nvptx64-nvidia-cuda"} {
  llvm.comdat @__llvm_global_comdat {
    llvm.comdat_selector @_Z34CalcTimeConstraintsForElems_kernelILi128EEviddPiPdS1_S1_S1_S1_S1_S1_ any
  }
  llvm.mlir.global internal unnamed_addr @_ZZL40Inner_CalcTimeConstraintsForElems_kernelILi128EEviddPiPdS1_S1_S1_S1_E12s_mindthydro() {addr_space = 3 : i32, alignment = 8 : i64, dso_local, sym_visibility = "private"} : !llvm.array<128 x f64> {
    %0 = llvm.mlir.undef : !llvm.array<128 x f64>
    llvm.return %0 : !llvm.array<128 x f64>
  }
  llvm.mlir.global internal unnamed_addr @_ZZL40Inner_CalcTimeConstraintsForElems_kernelILi128EEviddPiPdS1_S1_S1_S1_E14s_mindtcourant() {addr_space = 3 : i32, alignment = 8 : i64, dso_local, sym_visibility = "private"} : !llvm.array<128 x f64> {
    %0 = llvm.mlir.undef : !llvm.array<128 x f64>
    llvm.return %0 : !llvm.array<128 x f64>
  }
  llvm.func local_unnamed_addr ptx_kernelcc @_Z34CalcTimeConstraintsForElems_kernelILi128EEviddPiPdS1_S1_S1_S1_S1_S1_(%arg0: i32 {llvm.noundef}, %arg1: f64 {llvm.noundef}, %arg2: f64 {llvm.noundef}, %arg3: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.nofree, llvm.noundef, llvm.readonly}, %arg4: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.nofree, llvm.noundef, llvm.readonly}, %arg5: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.nofree, llvm.noundef, llvm.readonly}, %arg6: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.nofree, llvm.noundef, llvm.readonly}, %arg7: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.nofree, llvm.noundef, llvm.writeonly}, %arg8: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.nofree, llvm.noundef, llvm.readnone}, %arg9: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.nofree, llvm.noundef, llvm.writeonly}, %arg10: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.nofree, llvm.noundef, llvm.readnone}) comdat(@__llvm_global_comdat::@_Z34CalcTimeConstraintsForElems_kernelILi128EEviddPiPdS1_S1_S1_S1_S1_S1_) attributes {convergent, dso_local, frame_pointer = #llvm.framePointerKind<all>, no_unwind, passthrough = ["mustprogress", "norecurse", ["no-trapping-math", "true"], ["nvvm.maxntid", "128"], ["nvvm.minctasm", "16"], ["stack-protector-buffer-size", "8"], ["target-cpu", "sm_60"], ["uniform-work-group-size", "true"]], target_cpu = "sm_60", target_features = #llvm.target_features<["+ptx85", "+sm_60"]>} {
    %0 = llvm.mlir.addressof @_ZZL40Inner_CalcTimeConstraintsForElems_kernelILi128EEviddPiPdS1_S1_S1_S1_E12s_mindthydro : !llvm.ptr<3>
    %cst = arith.constant 9.9999999999999995E-21 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %cst_1 = arith.constant 1.000000e+20 : f64
    %1 = llvm.mlir.addressof @_ZZL40Inner_CalcTimeConstraintsForElems_kernelILi128EEviddPiPdS1_S1_S1_S1_E14s_mindtcourant : !llvm.ptr<3>
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %c16_i32 = arith.constant 16 : i32
    %c8_i32 = arith.constant 8 : i32
    %c4_i32 = arith.constant 4 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %2 = llvm.addrspacecast %0 : !llvm.ptr<3> to !llvm.ptr
    %3 = llvm.addrspacecast %1 : !llvm.ptr<3> to !llvm.ptr
    llvm.intr.experimental.noalias.scope.decl #alias_scope
    llvm.intr.experimental.noalias.scope.decl #alias_scope1
    llvm.intr.experimental.noalias.scope.decl #alias_scope2
    llvm.intr.experimental.noalias.scope.decl #alias_scope3
    llvm.intr.experimental.noalias.scope.decl #alias_scope4
    llvm.intr.experimental.noalias.scope.decl #alias_scope5
    %thread_id_x = gpu.thread_id  x
    %4 = arith.index_castui %thread_id_x : index to i64
    %5 = arith.index_cast %4 : i64 to index
    %6 = arith.index_cast %4 : i64 to index
    %7 = arith.index_cast %4 : i64 to index
    %8 = arith.index_cast %4 : i64 to index
    %9 = arith.index_cast %4 : i64 to index
    %10 = arith.index_cast %4 : i64 to index
    %11 = arith.index_cast %4 : i64 to index
    %12 = arith.index_cast %4 : i64 to index
    %13 = arith.index_cast %4 : i64 to index
    %14 = arith.index_cast %4 : i64 to index
    %15 = arith.index_cast %4 : i64 to index
    %16 = arith.index_cast %4 : i64 to index
    %17 = arith.index_cast %4 : i64 to index
    %18 = arith.index_cast %4 : i64 to index
    %19 = arith.index_cast %4 : i64 to index
    %20 = arith.index_cast %4 : i64 to index
    %21 = arith.index_cast %4 : i64 to index
    %22 = arith.index_cast %4 : i64 to index
    %23 = arith.index_cast %4 : i64 to index
    %24 = arith.index_cast %4 : i64 to index
    %25 = arith.index_cast %4 : i64 to index
    %26 = arith.index_cast %4 : i64 to index
    %27 = arith.index_cast %4 : i64 to index
    %28 = arith.index_cast %4 : i64 to index
    %29 = arith.index_cast %4 : i64 to index
    %30 = arith.index_cast %4 : i64 to index
    %31 = arith.index_cast %4 : i64 to index
    %32 = arith.index_cast %4 : i64 to index
    %33 = arith.index_cast %4 : i64 to index
    %34 = arith.index_cast %4 : i64 to index
    %35 = arith.index_castui %thread_id_x : index to i32
    %block_dim_x = gpu.block_dim  x
    %36 = arith.index_castui %block_dim_x : index to i32
    %block_id_x = gpu.block_id  x
    %37 = arith.index_castui %block_id_x : index to i64
    %38 = arith.index_cast %37 : i64 to index
    %39 = arith.index_cast %37 : i64 to index
    %40 = arith.index_castui %block_id_x : index to i32
    %41 = arith.muli %36, %40 : i32
    %42 = arith.addi %41, %35 : i32
    %43 = arith.cmpi slt, %42, %arg0 : i32
    %44:2 = scf.if %43 -> (f64, f64) {
      %grid_dim_x = gpu.grid_dim  x
      %54 = arith.index_castui %grid_dim_x : index to i32
      %55 = arith.muli %54, %36 : i32
      %56 = arith.addi %42, %55 : i32
      %57:5 = scf.while (%arg11 = %56, %arg12 = %cst_1, %arg13 = %cst_1, %arg14 = %cst_1, %arg15 = %cst_1) : (i32, f64, f64, f64, f64) -> (i32, f64, f64, f64, f64) {
        %58 = arith.subi %arg11, %55 : i32
        %59 = arith.index_cast %58 : i32 to index
        %60 = "enzymexla.pointer2memref"(%arg3) : (!llvm.ptr) -> memref<?xi32>
        %61 = memref.load %60[%59] {alias_scopes = [#alias_scope], alignment = 4 : i64, noalias_scopes = [#alias_scope1, #alias_scope2, #alias_scope3, #alias_scope4, #alias_scope5], ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<?xi32>
        %62 = arith.index_cast %61 : i32 to index
        %63 = arith.index_cast %61 : i32 to index
        %64 = arith.index_cast %61 : i32 to index
        %65 = "enzymexla.pointer2memref"(%arg5) : (!llvm.ptr) -> memref<?xf64>
        %66 = memref.load %65[%62] {alias_scopes = [#alias_scope2], alignment = 8 : i64, noalias_scopes = [#alias_scope, #alias_scope1, #alias_scope3, #alias_scope4, #alias_scope5], ordering = 0 : i64, tbaa = [#tbaa_tag1]} : memref<?xf64>
        %67 = arith.cmpf une, %66, %cst_0 {fastmathFlags = #llvm.fastmath<contract>} : f64
        %68 = math.absf %66 : f64
        %69 = arith.addf %68, %cst {fastmathFlags = #llvm.fastmath<contract>} : f64
        %70 = arith.divf %arg2, %69 {fastmathFlags = #llvm.fastmath<contract>} : f64
        %71 = arith.cmpf ogt, %arg14, %70 {fastmathFlags = #llvm.fastmath<contract>} : f64
        %72 = arith.select %71, %70, %arg14 {fastmathFlags = #llvm.fastmath<none>} : f64
        %73 = arith.select %67, %72, %arg14 : f64
        %74 = arith.cmpf olt, %73, %arg12 {fastmathFlags = #llvm.fastmath<contract>} : f64
        %75 = arith.select %74, %73, %arg12 {fastmathFlags = #llvm.fastmath<none>} : f64
        %76 = "enzymexla.pointer2memref"(%arg4) : (!llvm.ptr) -> memref<?xf64>
        %77 = memref.load %76[%63] {alias_scopes = [#alias_scope1], alignment = 8 : i64, noalias_scopes = [#alias_scope, #alias_scope2, #alias_scope3, #alias_scope4, #alias_scope5], ordering = 0 : i64, tbaa = [#tbaa_tag1]} : memref<?xf64>
        %78 = "enzymexla.pointer2memref"(%arg6) : (!llvm.ptr) -> memref<?xf64>
        %79 = memref.load %78[%64] {alias_scopes = [#alias_scope3], alignment = 8 : i64, noalias_scopes = [#alias_scope, #alias_scope1, #alias_scope2, #alias_scope4, #alias_scope5], ordering = 0 : i64, tbaa = [#tbaa_tag1]} : memref<?xf64>
        %80 = arith.mulf %77, %77 {fastmathFlags = #llvm.fastmath<contract>} : f64
        %81 = arith.cmpf olt, %66, %cst_0 {fastmathFlags = #llvm.fastmath<contract>} : f64
        %82 = arith.mulf %arg1, %79 {fastmathFlags = #llvm.fastmath<contract>} : f64
        %83 = arith.mulf %79, %82 {fastmathFlags = #llvm.fastmath<contract>} : f64
        %84 = arith.mulf %66, %83 {fastmathFlags = #llvm.fastmath<contract>} : f64
        %85 = arith.mulf %66, %84 {fastmathFlags = #llvm.fastmath<contract>} : f64
        %86 = arith.select %81, %85, %cst_0 {fastmathFlags = #llvm.fastmath<contract>} : f64
        %87 = arith.addf %80, %86 {fastmathFlags = #llvm.fastmath<contract>} : f64
        %88 = math.sqrt %87 : f64
        %89 = arith.divf %79, %88 {fastmathFlags = #llvm.fastmath<contract>} : f64
        %90 = arith.cmpf olt, %89, %arg15 {fastmathFlags = #llvm.fastmath<contract>} : f64
        %91 = arith.andi %67, %90 : i1
        %92 = arith.select %91, %89, %arg15 {fastmathFlags = #llvm.fastmath<none>} : f64
        %93 = arith.cmpf olt, %92, %arg13 {fastmathFlags = #llvm.fastmath<contract>} : f64
        %94 = arith.select %93, %92, %arg13 {fastmathFlags = #llvm.fastmath<none>} : f64
        %95 = arith.cmpi slt, %arg11, %arg0 : i32
        scf.condition(%95) %arg11, %75, %94, %73, %92 : i32, f64, f64, f64, f64
      } do {
      ^bb0(%arg11: i32, %arg12: f64, %arg13: f64, %arg14: f64, %arg15: f64):
        %58 = arith.addi %arg11, %55 : i32
        scf.yield %58, %arg12, %arg13, %arg14, %arg15 : i32, f64, f64, f64, f64
      }
      scf.yield %57#2, %57#1 : f64, f64
    } else {
      scf.yield %cst_1, %cst_1 : f64, f64
    }
    %45 = "enzymexla.pointer2memref"(%0) : (!llvm.ptr<3>) -> memref<?xf64, 3>
    affine.store %44#1, %45[symbol(%34)] {alignment = 8 : i64, noalias_scopes = [#alias_scope, #alias_scope1, #alias_scope2, #alias_scope3, #alias_scope4, #alias_scope5], ordering = 0 : i64, tbaa = [#tbaa_tag1]} : memref<?xf64, 3>
    %46 = "enzymexla.pointer2memref"(%1) : (!llvm.ptr<3>) -> memref<?xf64, 3>
    affine.store %44#0, %46[symbol(%33)] {alignment = 8 : i64, noalias_scopes = [#alias_scope, #alias_scope1, #alias_scope2, #alias_scope3, #alias_scope4, #alias_scope5], ordering = 0 : i64, tbaa = [#tbaa_tag1]} : memref<?xf64, 3>
    llvm.return
  }
}

// CHECK:  llvm.func local_unnamed_addr ptx_kernelcc @_Z34CalcTimeConstraintsForElems_kernelILi128EEviddPiPdS1_S1_S1_S1_S1_S1_(%arg0: i32 {llvm.noundef}, %arg1: f64 {llvm.noundef}, %arg2: f64 {llvm.noundef}, %arg3: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.nofree, llvm.noundef, llvm.readonly}, %arg4: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.nofree, llvm.noundef, llvm.readonly}, %arg5: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.nofree, llvm.noundef, llvm.readonly}, %arg6: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.nofree, llvm.noundef, llvm.readonly}, %arg7: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.nofree, llvm.noundef, llvm.writeonly}, %arg8: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.nofree, llvm.noundef, llvm.readnone}, %arg9: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.nofree, llvm.noundef, llvm.writeonly}, %arg10: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.nofree, llvm.noundef, llvm.readnone}) comdat(@__llvm_global_comdat::@_Z34CalcTimeConstraintsForElems_kernelILi128EEviddPiPdS1_S1_S1_S1_S1_S1_) attributes {convergent, dso_local, frame_pointer = #llvm.framePointerKind<all>, no_unwind, passthrough = ["mustprogress", "norecurse", ["no-trapping-math", "true"], ["nvvm.maxntid", "128"], ["nvvm.minctasm", "16"], ["stack-protector-buffer-size", "8"], ["target-cpu", "sm_60"], ["uniform-work-group-size", "true"]], target_cpu = "sm_60", target_features = #llvm.target_features<["+ptx85", "+sm_60"]>} {
// CHECK-NEXT:    %0 = llvm.mlir.addressof @_ZZL40Inner_CalcTimeConstraintsForElems_kernelILi128EEviddPiPdS1_S1_S1_S1_E14s_mindtcourant : !llvm.ptr<3>
// CHECK-NEXT:    %cst = arith.constant 1.000000e+20 : f64
// CHECK-NEXT:    %cst_0 = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %cst_1 = arith.constant 9.9999999999999995E-21 : f64
// CHECK-NEXT:    %1 = llvm.mlir.addressof @_ZZL40Inner_CalcTimeConstraintsForElems_kernelILi128EEviddPiPdS1_S1_S1_S1_E12s_mindthydro : !llvm.ptr<3>
// CHECK-NEXT:    %grid_dim_x = gpu.grid_dim  x
// CHECK-NEXT:    %2 = arith.index_castui %grid_dim_x : index to i32
// CHECK-NEXT:    llvm.intr.experimental.noalias.scope.decl #alias_scope
// CHECK-NEXT:    llvm.intr.experimental.noalias.scope.decl #alias_scope1
// CHECK-NEXT:    llvm.intr.experimental.noalias.scope.decl #alias_scope2
// CHECK-NEXT:    llvm.intr.experimental.noalias.scope.decl #alias_scope3
// CHECK-NEXT:    llvm.intr.experimental.noalias.scope.decl #alias_scope4
// CHECK-NEXT:    llvm.intr.experimental.noalias.scope.decl #alias_scope5
// CHECK-NEXT:    %thread_id_x = gpu.thread_id  x
// CHECK-NEXT:    %3 = arith.index_castui %thread_id_x : index to i64
// CHECK-NEXT:    %4 = arith.index_cast %3 : i64 to index
// CHECK-NEXT:    %5 = arith.index_cast %3 : i64 to index
// CHECK-NEXT:    %6 = arith.index_castui %thread_id_x : index to i32
// CHECK-NEXT:    %block_dim_x = gpu.block_dim  x
// CHECK-NEXT:    %7 = arith.index_castui %block_dim_x : index to i32
// CHECK-NEXT:    %8 = arith.muli %2, %7 : i32
// CHECK-NEXT:    %9 = arith.index_cast %8 : i32 to index
// CHECK-NEXT:    %block_id_x = gpu.block_id  x
// CHECK-NEXT:    %10 = arith.index_castui %block_id_x : index to i32
// CHECK-NEXT:    %11 = arith.muli %7, %10 : i32
// CHECK-NEXT:    %12 = arith.addi %11, %6 : i32
// CHECK-NEXT:    %13 = arith.cmpi slt, %12, %arg0 : i32
// CHECK-NEXT:    %14:2 = scf.if %13 -> (f64, f64) {
// CHECK-NEXT:      %17 = arith.addi %12, %8 : i32
// CHECK-NEXT:      %18:5 = scf.while (%arg11 = %17, %arg12 = %cst, %arg13 = %cst, %arg14 = %cst, %arg15 = %cst) : (i32, f64, f64, f64, f64) -> (i32, f64, f64, f64, f64) {
// CHECK-NEXT:        %19 = arith.index_cast %arg11 : i32 to index
// CHECK-NEXT:        %20 = "enzymexla.pointer2memref"(%arg3) : (!llvm.ptr) -> memref<?xi32>
// CHECK-NEXT:        %21 = affine.apply #map()[%9]
// CHECK-NEXT:        %22 = arith.addi %21, %19 : index
// CHECK-NEXT:        %23 = memref.load %20[%22] {alias_scopes = [#alias_scope], alignment = 4 : i64, noalias_scopes = [#alias_scope1, #alias_scope2, #alias_scope3, #alias_scope4, #alias_scope5], ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<?xi32>
// CHECK-NEXT:        %24 = arith.index_cast %23 : i32 to index
// CHECK-NEXT:        %25 = arith.index_cast %23 : i32 to index
// CHECK-NEXT:        %26 = arith.index_cast %23 : i32 to index
// CHECK-NEXT:        %27 = "enzymexla.pointer2memref"(%arg5) : (!llvm.ptr) -> memref<?xf64>
// CHECK-NEXT:        %28 = memref.load %27[%24] {alias_scopes = [#alias_scope2], alignment = 8 : i64, noalias_scopes = [#alias_scope, #alias_scope1, #alias_scope3, #alias_scope4, #alias_scope5], ordering = 0 : i64, tbaa = [#tbaa_tag1]} : memref<?xf64>
// CHECK-NEXT:        %29 = arith.cmpf une, %28, %cst_0 {fastmathFlags = #llvm.fastmath<contract>} : f64
// CHECK-NEXT:        %30 = math.absf %28 : f64
// CHECK-NEXT:        %31 = arith.addf %30, %cst_1 {fastmathFlags = #llvm.fastmath<contract>} : f64
// CHECK-NEXT:        %32 = arith.divf %arg2, %31 {fastmathFlags = #llvm.fastmath<contract>} : f64
// CHECK-NEXT:        %33 = arith.cmpf ogt, %arg14, %32 {fastmathFlags = #llvm.fastmath<contract>} : f64
// CHECK-NEXT:        %34 = arith.select %33, %32, %arg14 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:        %35 = arith.select %29, %34, %arg14 : f64
// CHECK-NEXT:        %36 = arith.cmpf olt, %35, %arg12 {fastmathFlags = #llvm.fastmath<contract>} : f64
// CHECK-NEXT:        %37 = arith.select %36, %35, %arg12 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:        %38 = "enzymexla.pointer2memref"(%arg4) : (!llvm.ptr) -> memref<?xf64>
// CHECK-NEXT:        %39 = memref.load %38[%25] {alias_scopes = [#alias_scope1], alignment = 8 : i64, noalias_scopes = [#alias_scope, #alias_scope2, #alias_scope3, #alias_scope4, #alias_scope5], ordering = 0 : i64, tbaa = [#tbaa_tag1]} : memref<?xf64>
// CHECK-NEXT:        %40 = "enzymexla.pointer2memref"(%arg6) : (!llvm.ptr) -> memref<?xf64>
// CHECK-NEXT:        %41 = memref.load %40[%26] {alias_scopes = [#alias_scope3], alignment = 8 : i64, noalias_scopes = [#alias_scope, #alias_scope1, #alias_scope2, #alias_scope4, #alias_scope5], ordering = 0 : i64, tbaa = [#tbaa_tag1]} : memref<?xf64>
// CHECK-NEXT:        %42 = arith.mulf %39, %39 {fastmathFlags = #llvm.fastmath<contract>} : f64
// CHECK-NEXT:        %43 = arith.cmpf olt, %28, %cst_0 {fastmathFlags = #llvm.fastmath<contract>} : f64
// CHECK-NEXT:        %44 = arith.mulf %arg1, %41 {fastmathFlags = #llvm.fastmath<contract>} : f64
// CHECK-NEXT:        %45 = arith.mulf %41, %44 {fastmathFlags = #llvm.fastmath<contract>} : f64
// CHECK-NEXT:        %46 = arith.mulf %28, %45 {fastmathFlags = #llvm.fastmath<contract>} : f64
// CHECK-NEXT:        %47 = arith.mulf %28, %46 {fastmathFlags = #llvm.fastmath<contract>} : f64
// CHECK-NEXT:        %48 = arith.select %43, %47, %cst_0 {fastmathFlags = #llvm.fastmath<contract>} : f64
// CHECK-NEXT:        %49 = arith.addf %42, %48 {fastmathFlags = #llvm.fastmath<contract>} : f64
// CHECK-NEXT:        %50 = math.sqrt %49 : f64
// CHECK-NEXT:        %51 = arith.divf %41, %50 {fastmathFlags = #llvm.fastmath<contract>} : f64
// CHECK-NEXT:        %52 = arith.cmpf olt, %51, %arg15 {fastmathFlags = #llvm.fastmath<contract>} : f64
// CHECK-NEXT:        %53 = arith.andi %29, %52 : i1
// CHECK-NEXT:        %54 = arith.select %53, %51, %arg15 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:        %55 = arith.cmpf olt, %54, %arg13 {fastmathFlags = #llvm.fastmath<contract>} : f64
// CHECK-NEXT:        %56 = arith.select %55, %54, %arg13 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:        %57 = arith.cmpi slt, %arg11, %arg0 : i32
// CHECK-NEXT:        scf.condition(%57) %arg11, %37, %56, %35, %54 : i32, f64, f64, f64, f64
// CHECK-NEXT:      } do {
// CHECK-NEXT:      ^bb0(%arg11: i32, %arg12: f64, %arg13: f64, %arg14: f64, %arg15: f64):
// CHECK-NEXT:        %19 = arith.addi %arg11, %8 : i32
// CHECK-NEXT:        scf.yield %19, %arg12, %arg13, %arg14, %arg15 : i32, f64, f64, f64, f64
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %18#2, %18#1 : f64, f64
// CHECK-NEXT:    } else {
// CHECK-NEXT:      scf.yield %cst, %cst : f64, f64
// CHECK-NEXT:    }
// CHECK-NEXT:    %15 = "enzymexla.pointer2memref"(%1) : (!llvm.ptr<3>) -> memref<?xf64, 3>
// CHECK-NEXT:    affine.store %14#1, %15[symbol(%5)] {alignment = 8 : i64, noalias_scopes = [#alias_scope, #alias_scope1, #alias_scope2, #alias_scope3, #alias_scope4, #alias_scope5], ordering = 0 : i64, tbaa = [#tbaa_tag1]} : memref<?xf64, 3>
// CHECK-NEXT:    %16 = "enzymexla.pointer2memref"(%0) : (!llvm.ptr<3>) -> memref<?xf64, 3>
// CHECK-NEXT:    affine.store %14#0, %16[symbol(%4)] {alignment = 8 : i64, noalias_scopes = [#alias_scope, #alias_scope1, #alias_scope2, #alias_scope3, #alias_scope4, #alias_scope5], ordering = 0 : i64, tbaa = [#tbaa_tag1]} : memref<?xf64, 3>
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }
