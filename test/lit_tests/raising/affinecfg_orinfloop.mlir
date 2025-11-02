// RUN: enzymexlamlir-opt %s --affine-cfg --mlir-print-local-scope | FileCheck %s

#tbaa_root = #llvm.tbaa_root<id = "Simple C++ TBAA">
#tbaa_root1 = #llvm.tbaa_root<id = "_ZTSSt4lessIcE">
#tbaa_root2 = #llvm.tbaa_root<id = "_ZTSSt4lessISt4pairIccEE">
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root, 0>}>
#tbaa_type_desc1 = #llvm.tbaa_type_desc<id = "_ZTSSt20_Rb_tree_key_compareISt4lessIcEE", members = {<#tbaa_root1, 0>}>
#tbaa_type_desc2 = #llvm.tbaa_type_desc<id = "_ZTSSt20_Rb_tree_key_compareISt4lessISt4pairIccEEE", members = {<#tbaa_root2, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc, access_type = #tbaa_type_desc, offset = 0>
#tbaa_type_desc3 = #llvm.tbaa_type_desc<id = "_ZTSSt14_Rb_tree_color", members = {<#tbaa_type_desc, 0>}>
#tbaa_type_desc4 = #llvm.tbaa_type_desc<id = "any pointer", members = {<#tbaa_type_desc, 0>}>
#tbaa_type_desc5 = #llvm.tbaa_type_desc<id = "long", members = {<#tbaa_type_desc, 0>}>
#tbaa_type_desc6 = #llvm.tbaa_type_desc<id = "bool", members = {<#tbaa_type_desc, 0>}>
#tbaa_type_desc7 = #llvm.tbaa_type_desc<id = "int", members = {<#tbaa_type_desc, 0>}>
#tbaa_type_desc8 = #llvm.tbaa_type_desc<id = "double", members = {<#tbaa_type_desc, 0>}>
#tbaa_tag1 = #llvm.tbaa_tag<base_type = #tbaa_type_desc5, access_type = #tbaa_type_desc5, offset = 0>
#tbaa_tag2 = #llvm.tbaa_tag<base_type = #tbaa_type_desc7, access_type = #tbaa_type_desc7, offset = 0>
#tbaa_tag3 = #llvm.tbaa_tag<base_type = #tbaa_type_desc8, access_type = #tbaa_type_desc8, offset = 0>
#tbaa_type_desc9 = #llvm.tbaa_type_desc<id = "p1 _ZTSSt18_Rb_tree_node_base", members = {<#tbaa_type_desc4, 0>}>
#tbaa_type_desc10 = #llvm.tbaa_type_desc<id = "p1 _ZTS20cuda_run_time_vectorIdE", members = {<#tbaa_type_desc4, 0>}>
#tbaa_type_desc11 = #llvm.tbaa_type_desc<id = "p1 double", members = {<#tbaa_type_desc4, 0>}>
#tbaa_type_desc12 = #llvm.tbaa_type_desc<id = "p1 omnipotent char", members = {<#tbaa_type_desc4, 0>}>
#tbaa_type_desc13 = #llvm.tbaa_type_desc<id = "p1 _ZTS15cuda_Class_Grid", members = {<#tbaa_type_desc4, 0>}>
#tbaa_type_desc14 = #llvm.tbaa_type_desc<id = "p1 int", members = {<#tbaa_type_desc4, 0>}>
#tbaa_type_desc15 = #llvm.tbaa_type_desc<id = "p1 _ZTS10Class_Grid", members = {<#tbaa_type_desc4, 0>}>
#tbaa_type_desc16 = #llvm.tbaa_type_desc<id = "_ZTS16cuda_Struct_Grid", members = {<#tbaa_type_desc, 0>, <#tbaa_type_desc, 1>, <#tbaa_type_desc7, 4>, <#tbaa_type_desc7, 8>, <#tbaa_type_desc7, 12>, <#tbaa_type_desc7, 16>, <#tbaa_type_desc7, 20>, <#tbaa_type_desc7, 24>, <#tbaa_type_desc7, 28>, <#tbaa_type_desc7, 32>}>
#tbaa_type_desc17 = #llvm.tbaa_type_desc<id = "p1 _ZTS20cuda_run_time_matrixIdE", members = {<#tbaa_type_desc4, 0>}>
#tbaa_type_desc18 = #llvm.tbaa_type_desc<id = "p1 _ZTS11CUstream_st", members = {<#tbaa_type_desc4, 0>}>
#tbaa_tag4 = #llvm.tbaa_tag<base_type = #tbaa_type_desc9, access_type = #tbaa_type_desc9, offset = 0>
#tbaa_tag5 = #llvm.tbaa_tag<base_type = #tbaa_type_desc10, access_type = #tbaa_type_desc10, offset = 0>
#tbaa_tag6 = #llvm.tbaa_tag<base_type = #tbaa_type_desc13, access_type = #tbaa_type_desc13, offset = 0>
#tbaa_tag7 = #llvm.tbaa_tag<base_type = #tbaa_type_desc11, access_type = #tbaa_type_desc11, offset = 0>
#tbaa_tag8 = #llvm.tbaa_tag<base_type = #tbaa_type_desc14, access_type = #tbaa_type_desc14, offset = 0>
#tbaa_tag9 = #llvm.tbaa_tag<base_type = #tbaa_type_desc16, access_type = #tbaa_type_desc, offset = 1>
#tbaa_tag10 = #llvm.tbaa_tag<base_type = #tbaa_type_desc16, access_type = #tbaa_type_desc7, offset = 8>
#tbaa_tag11 = #llvm.tbaa_tag<base_type = #tbaa_type_desc16, access_type = #tbaa_type_desc7, offset = 20>
#tbaa_type_desc19 = #llvm.tbaa_type_desc<id = "_ZTSSt18_Rb_tree_node_base", members = {<#tbaa_type_desc3, 0>, <#tbaa_type_desc9, 8>, <#tbaa_type_desc9, 16>, <#tbaa_type_desc9, 24>}>
#tbaa_type_desc20 = #llvm.tbaa_type_desc<id = "_ZTSNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderE", members = {<#tbaa_type_desc12, 0>}>
#tbaa_type_desc21 = #llvm.tbaa_type_desc<id = "_ZTSNSt12_Vector_baseI20cuda_run_time_vectorIdESaIS1_EE17_Vector_impl_dataE", members = {<#tbaa_type_desc10, 0>, <#tbaa_type_desc10, 8>, <#tbaa_type_desc10, 16>}>
#tbaa_type_desc22 = #llvm.tbaa_type_desc<id = "_ZTS33cuda_run_time_vector_shallow_copyIdE", members = {<#tbaa_type_desc5, 0>, <#tbaa_type_desc11, 8>}>
#tbaa_type_desc23 = #llvm.tbaa_type_desc<id = "_ZTSNSt12_Vector_baseI20cuda_run_time_matrixIdESaIS1_EE17_Vector_impl_dataE", members = {<#tbaa_type_desc17, 0>, <#tbaa_type_desc17, 8>, <#tbaa_type_desc17, 16>}>
#tbaa_tag12 = #llvm.tbaa_tag<base_type = #tbaa_type_desc21, access_type = #tbaa_type_desc10, offset = 8>
#tbaa_tag13 = #llvm.tbaa_tag<base_type = #tbaa_type_desc21, access_type = #tbaa_type_desc10, offset = 0>
#tbaa_tag14 = #llvm.tbaa_tag<base_type = #tbaa_type_desc22, access_type = #tbaa_type_desc5, offset = 0>
#tbaa_type_desc24 = #llvm.tbaa_type_desc<id = "_ZTSSt15_Rb_tree_header", members = {<#tbaa_type_desc19, 0>, <#tbaa_type_desc5, 32>}>
#tbaa_type_desc25 = #llvm.tbaa_type_desc<id = "_ZTSNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE", members = {<#tbaa_type_desc20, 0>, <#tbaa_type_desc5, 8>, <#tbaa_type_desc, 16>}>
#tbaa_type_desc26 = #llvm.tbaa_type_desc<id = "_ZTSNSt12_Vector_baseI20cuda_run_time_vectorIdESaIS1_EE12_Vector_implE", members = {<#tbaa_type_desc21, 0>}>
#tbaa_type_desc27 = #llvm.tbaa_type_desc<id = "_ZTSNSt12_Vector_baseI20cuda_run_time_matrixIdESaIS1_EE12_Vector_implE", members = {<#tbaa_type_desc23, 0>}>
#tbaa_tag15 = #llvm.tbaa_tag<base_type = #tbaa_type_desc24, access_type = #tbaa_type_desc9, offset = 8>
#tbaa_type_desc28 = #llvm.tbaa_type_desc<id = "_ZTS20cuda_run_time_vectorIdE", members = {<#tbaa_type_desc6, 0>, <#tbaa_type_desc5, 8>, <#tbaa_type_desc11, 16>, <#tbaa_type_desc25, 24>}>
#tbaa_type_desc29 = #llvm.tbaa_type_desc<id = "_ZTSNSt8_Rb_treeIcSt4pairIKcP15cuda_Class_GridESt10_Select1stIS4_ESt4lessIcESaIS4_EE13_Rb_tree_implIS8_Lb1EEE", members = {<#tbaa_type_desc1, 0>, <#tbaa_type_desc24, 8>}>
#tbaa_type_desc30 = #llvm.tbaa_type_desc<id = "_ZTSNSt8_Rb_treeIcSt4pairIKcP20cuda_run_time_vectorIdEESt10_Select1stIS5_ESt4lessIcESaIS5_EE13_Rb_tree_implIS9_Lb1EEE", members = {<#tbaa_type_desc1, 0>, <#tbaa_type_desc24, 8>}>
#tbaa_type_desc31 = #llvm.tbaa_type_desc<id = "_ZTSSt12_Vector_baseI20cuda_run_time_vectorIdESaIS1_EE", members = {<#tbaa_type_desc26, 0>}>
#tbaa_type_desc32 = #llvm.tbaa_type_desc<id = "_ZTSNSt8_Rb_treeIcSt4pairIKc20cuda_run_time_vectorIiEESt10_Select1stIS4_ESt4lessIcESaIS4_EE13_Rb_tree_implIS8_Lb1EEE", members = {<#tbaa_type_desc1, 0>, <#tbaa_type_desc24, 8>}>
#tbaa_type_desc33 = #llvm.tbaa_type_desc<id = "_ZTSNSt8_Rb_treeISt4pairIccES0_IKS1_20cuda_run_time_matrixIdEESt10_Select1stIS5_ESt4lessIS1_ESaIS5_EE13_Rb_tree_implIS9_Lb1EEE", members = {<#tbaa_type_desc2, 0>, <#tbaa_type_desc24, 8>}>
#tbaa_type_desc34 = #llvm.tbaa_type_desc<id = "_ZTSNSt8_Rb_treeISt4pairIccES0_IKS1_20cuda_run_time_vectorIdEESt10_Select1stIS5_ESt4lessIS1_ESaIS5_EE13_Rb_tree_implIS9_Lb1EEE", members = {<#tbaa_type_desc2, 0>, <#tbaa_type_desc24, 8>}>
#tbaa_type_desc35 = #llvm.tbaa_type_desc<id = "_ZTSNSt8_Rb_treeIcSt4pairIKciESt10_Select1stIS2_ESt4lessIcESaIS2_EE13_Rb_tree_implIS6_Lb1EEE", members = {<#tbaa_type_desc1, 0>, <#tbaa_type_desc24, 8>}>
#tbaa_type_desc36 = #llvm.tbaa_type_desc<id = "_ZTSNSt8_Rb_treeISt4pairIccES0_IKS1_iESt10_Select1stIS3_ESt4lessIS1_ESaIS3_EE13_Rb_tree_implIS7_Lb1EEE", members = {<#tbaa_type_desc2, 0>, <#tbaa_type_desc24, 8>}>
#tbaa_type_desc37 = #llvm.tbaa_type_desc<id = "_ZTSNSt8_Rb_treeIcSt4pairIKc33cuda_run_time_vector_shallow_copyIiEESt10_Select1stIS4_ESt4lessIcESaIS4_EE13_Rb_tree_implIS8_Lb1EEE", members = {<#tbaa_type_desc1, 0>, <#tbaa_type_desc24, 8>}>
#tbaa_type_desc38 = #llvm.tbaa_type_desc<id = "_ZTSNSt8_Rb_treeISt4pairIccES0_IKS1_33cuda_run_time_matrix_shallow_copyIdEESt10_Select1stIS5_ESt4lessIS1_ESaIS5_EE13_Rb_tree_implIS9_Lb1EEE", members = {<#tbaa_type_desc2, 0>, <#tbaa_type_desc24, 8>}>
#tbaa_type_desc39 = #llvm.tbaa_type_desc<id = "_ZTSNSt8_Rb_treeISt4pairIccES0_IKS1_33cuda_run_time_vector_shallow_copyIdEESt10_Select1stIS5_ESt4lessIS1_ESaIS5_EE13_Rb_tree_implIS9_Lb1EEE", members = {<#tbaa_type_desc2, 0>, <#tbaa_type_desc24, 8>}>
#tbaa_type_desc40 = #llvm.tbaa_type_desc<id = "_ZTSSt12_Vector_baseI20cuda_run_time_matrixIdESaIS1_EE", members = {<#tbaa_type_desc27, 0>}>
#tbaa_tag16 = #llvm.tbaa_tag<base_type = #tbaa_type_desc28, access_type = #tbaa_type_desc11, offset = 16>
#tbaa_type_desc41 = #llvm.tbaa_type_desc<id = "_ZTSSt8_Rb_treeIcSt4pairIKcP15cuda_Class_GridESt10_Select1stIS4_ESt4lessIcESaIS4_EE", members = {<#tbaa_type_desc29, 0>}>
#tbaa_type_desc42 = #llvm.tbaa_type_desc<id = "_ZTSSt8_Rb_treeIcSt4pairIKcP20cuda_run_time_vectorIdEESt10_Select1stIS5_ESt4lessIcESaIS5_EE", members = {<#tbaa_type_desc30, 0>}>
#tbaa_type_desc43 = #llvm.tbaa_type_desc<id = "_ZTSSt6vectorI20cuda_run_time_vectorIdESaIS1_EE", members = {<#tbaa_type_desc31, 0>}>
#tbaa_type_desc44 = #llvm.tbaa_type_desc<id = "_ZTSSt8_Rb_treeIcSt4pairIKc20cuda_run_time_vectorIiEESt10_Select1stIS4_ESt4lessIcESaIS4_EE", members = {<#tbaa_type_desc32, 0>}>
#tbaa_type_desc45 = #llvm.tbaa_type_desc<id = "_ZTSSt8_Rb_treeISt4pairIccES0_IKS1_20cuda_run_time_matrixIdEESt10_Select1stIS5_ESt4lessIS1_ESaIS5_EE", members = {<#tbaa_type_desc33, 0>}>
#tbaa_type_desc46 = #llvm.tbaa_type_desc<id = "_ZTSSt8_Rb_treeISt4pairIccES0_IKS1_20cuda_run_time_vectorIdEESt10_Select1stIS5_ESt4lessIS1_ESaIS5_EE", members = {<#tbaa_type_desc34, 0>}>
#tbaa_type_desc47 = #llvm.tbaa_type_desc<id = "_ZTSSt8_Rb_treeIcSt4pairIKciESt10_Select1stIS2_ESt4lessIcESaIS2_EE", members = {<#tbaa_type_desc35, 0>}>
#tbaa_type_desc48 = #llvm.tbaa_type_desc<id = "_ZTSSt8_Rb_treeISt4pairIccES0_IKS1_iESt10_Select1stIS3_ESt4lessIS1_ESaIS3_EE", members = {<#tbaa_type_desc36, 0>}>
#tbaa_type_desc49 = #llvm.tbaa_type_desc<id = "_ZTSSt8_Rb_treeIcSt4pairIKc33cuda_run_time_vector_shallow_copyIiEESt10_Select1stIS4_ESt4lessIcESaIS4_EE", members = {<#tbaa_type_desc37, 0>}>
#tbaa_type_desc50 = #llvm.tbaa_type_desc<id = "_ZTSSt8_Rb_treeISt4pairIccES0_IKS1_33cuda_run_time_matrix_shallow_copyIdEESt10_Select1stIS5_ESt4lessIS1_ESaIS5_EE", members = {<#tbaa_type_desc38, 0>}>
#tbaa_type_desc51 = #llvm.tbaa_type_desc<id = "_ZTSSt8_Rb_treeISt4pairIccES0_IKS1_33cuda_run_time_vector_shallow_copyIdEESt10_Select1stIS5_ESt4lessIS1_ESaIS5_EE", members = {<#tbaa_type_desc39, 0>}>
#tbaa_type_desc52 = #llvm.tbaa_type_desc<id = "_ZTSSt6vectorI20cuda_run_time_matrixIdESaIS1_EE", members = {<#tbaa_type_desc40, 0>}>
#tbaa_type_desc53 = #llvm.tbaa_type_desc<id = "_ZTSSt3mapIcP15cuda_Class_GridSt4lessIcESaISt4pairIKcS1_EEE", members = {<#tbaa_type_desc41, 0>}>
#tbaa_type_desc54 = #llvm.tbaa_type_desc<id = "_ZTSSt3mapIcP20cuda_run_time_vectorIdESt4lessIcESaISt4pairIKcS2_EEE", members = {<#tbaa_type_desc42, 0>}>
#tbaa_type_desc55 = #llvm.tbaa_type_desc<id = "_ZTSSt3mapIc20cuda_run_time_vectorIiESt4lessIcESaISt4pairIKcS1_EEE", members = {<#tbaa_type_desc44, 0>}>
#tbaa_type_desc56 = #llvm.tbaa_type_desc<id = "_ZTSSt3mapISt4pairIccE20cuda_run_time_matrixIdESt4lessIS1_ESaIS0_IKS1_S3_EEE", members = {<#tbaa_type_desc45, 0>}>
#tbaa_type_desc57 = #llvm.tbaa_type_desc<id = "_ZTSSt3mapISt4pairIccE20cuda_run_time_vectorIdESt4lessIS1_ESaIS0_IKS1_S3_EEE", members = {<#tbaa_type_desc46, 0>}>
#tbaa_type_desc58 = #llvm.tbaa_type_desc<id = "_ZTSSt3mapIciSt4lessIcESaISt4pairIKciEEE", members = {<#tbaa_type_desc47, 0>}>
#tbaa_type_desc59 = #llvm.tbaa_type_desc<id = "_ZTSSt3mapISt4pairIccEiSt4lessIS1_ESaIS0_IKS1_iEEE", members = {<#tbaa_type_desc48, 0>}>
#tbaa_type_desc60 = #llvm.tbaa_type_desc<id = "_ZTSSt3mapIc33cuda_run_time_vector_shallow_copyIiESt4lessIcESaISt4pairIKcS1_EEE", members = {<#tbaa_type_desc49, 0>}>
#tbaa_type_desc61 = #llvm.tbaa_type_desc<id = "_ZTSSt3mapISt4pairIccE33cuda_run_time_matrix_shallow_copyIdESt4lessIS1_ESaIS0_IKS1_S3_EEE", members = {<#tbaa_type_desc50, 0>}>
#tbaa_type_desc62 = #llvm.tbaa_type_desc<id = "_ZTSSt3mapISt4pairIccE33cuda_run_time_vector_shallow_copyIdESt4lessIS1_ESaIS0_IKS1_S3_EEE", members = {<#tbaa_type_desc51, 0>}>
#tbaa_type_desc63 = #llvm.tbaa_type_desc<id = "_ZTS15cuda_Class_Grid", members = {<#tbaa_type_desc15, 0>, <#tbaa_type_desc16, 8>, <#tbaa_type_desc12, 48>, <#tbaa_type_desc12, 56>, <#tbaa_type_desc14, 64>, <#tbaa_type_desc14, 72>, <#tbaa_type_desc14, 80>, <#tbaa_type_desc14, 88>, <#tbaa_type_desc14, 96>, <#tbaa_type_desc14, 104>, <#tbaa_type_desc14, 112>, <#tbaa_type_desc14, 120>, <#tbaa_type_desc7, 128>, <#tbaa_type_desc7, 132>, <#tbaa_type_desc13, 136>, <#tbaa_type_desc13, 144>, <#tbaa_type_desc53, 152>, <#tbaa_type_desc54, 200>, <#tbaa_type_desc43, 248>, <#tbaa_type_desc43, 272>, <#tbaa_type_desc43, 296>, <#tbaa_type_desc43, 320>, <#tbaa_type_desc43, 344>, <#tbaa_type_desc43, 368>, <#tbaa_type_desc28, 392>, <#tbaa_type_desc28, 448>, <#tbaa_type_desc7, 504>, <#tbaa_type_desc7, 508>, <#tbaa_type_desc7, 512>, <#tbaa_type_desc7, 516>, <#tbaa_type_desc25, 520>, <#tbaa_type_desc6, 552>, <#tbaa_type_desc28, 560>, <#tbaa_type_desc55, 616>, <#tbaa_type_desc56, 664>, <#tbaa_type_desc57, 712>, <#tbaa_type_desc57, 760>, <#tbaa_type_desc57, 808>, <#tbaa_type_desc58, 856>, <#tbaa_type_desc58, 904>, <#tbaa_type_desc7, 952>, <#tbaa_type_desc7, 956>, <#tbaa_type_desc59, 960>, <#tbaa_type_desc59, 1008>, <#tbaa_type_desc57, 1056>, <#tbaa_type_desc22, 1104>, <#tbaa_type_desc60, 1120>, <#tbaa_type_desc61, 1168>, <#tbaa_type_desc62, 1216>, <#tbaa_type_desc62, 1264>, <#tbaa_type_desc62, 1312>, <#tbaa_type_desc52, 1360>, <#tbaa_type_desc6, 1384>, <#tbaa_type_desc18, 1392>, <#tbaa_type_desc18, 1400>, <#tbaa_type_desc18, 1408>, <#tbaa_type_desc18, 1416>, <#tbaa_type_desc18, 1424>, <#tbaa_type_desc18, 1432>, <#tbaa_type_desc18, 1440>, <#tbaa_type_desc18, 1448>, <#tbaa_type_desc18, 1456>, <#tbaa_type_desc18, 1464>, <#tbaa_type_desc18, 1472>}>
#tbaa_tag17 = #llvm.tbaa_tag<base_type = #tbaa_type_desc63, access_type = #tbaa_type_desc6, offset = 1384>
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  llvm.func linkonce_odr local_unnamed_addr @_ZN15cuda_Class_Grid36kernel_launch_cuda_periodic_y_moduloILb0EEEvii(%arg0: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 1480 : i64, llvm.nonnull, llvm.noundef}, %arg1: i32 {llvm.noundef}, %arg2: i32 {llvm.noundef}) attributes {alignment = 2 : i64, dso_local, no_unwind, passthrough = ["mustprogress", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uwtable_kind = #llvm.uwtableKind<async>} {
    %c-1_i64 = arith.constant -1 : i64
    %0 = ub.poison : f64
    %true = arith.constant true
    %1 = ub.poison : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f64
    %c78_i8 = arith.constant 78 : i8
    %c4_i32 = arith.constant 4 : i32
    %c77_i8 = arith.constant 77 : i8
    %c1_i64 = arith.constant 1 : i64
    %2 = llvm.mlir.constant(16 : i64) : i64
    %3 = llvm.mlir.constant(36 : i64) : i64
    %4 = llvm.mlir.constant(1 : i64) : i64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %5 = llvm.mlir.zero : !llvm.ptr
    %c121_i8 = arith.constant 121 : i8
    %c24_i64 = arith.constant 24 : i64
    %c16_i64 = arith.constant 16 : i64
    %c-1_i32 = arith.constant -1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c4294967296_i64 = arith.constant 4294967296 : i64
    %c0_i64 = arith.constant 0 : i64
    %c36_i64 = arith.constant 36 : i64
    %6 = llvm.alloca %c1_i32 x !llvm.struct<"class.cuda_run_time_vector_shallow_copy", (i64, ptr)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %7 = llvm.alloca %c1_i32 x !llvm.struct<"class.cuda_run_time_vector_shallow_copy.298", (i64, ptr)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %8 = llvm.alloca %c1_i32 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %9 = llvm.alloca %c1_i32 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %10 = llvm.alloca %c1_i32 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %11 = llvm.alloca %c1_i32 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %12 = llvm.alloca %c1_i32 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %13 = llvm.alloca %c1_i32 x !llvm.array<9 x ptr> {alignment = 16 : i64} : (i32) -> !llvm.ptr
    %14 = llvm.alloca %c1_i32 x !llvm.struct<"struct.cuda_Struct_Grid", (i8, i8, i32, i32, i32, i32, i32, i32, i32, i32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %15 = llvm.alloca %c1_i32 x !llvm.struct<"struct.cuda_Struct_Grid", (i8, i8, i32, i32, i32, i32, i32, i32, i32, i32)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %16 = llvm.getelementptr inbounds|nuw %arg0[216] : (!llvm.ptr) -> !llvm.ptr, i8
    %17 = llvm.load %16 {alignment = 8 : i64, tbaa = [#tbaa_tag15]} : !llvm.ptr -> !llvm.ptr
    %18 = llvm.getelementptr inbounds|nuw %arg0[208] : (!llvm.ptr) -> !llvm.ptr, i8
    %19 = llvm.icmp "eq" %17, %5 : !llvm.ptr
    %20 = scf.if %19 -> (i32) {
      scf.yield %c0_i32 : i32
    } else {
      %23:2 = scf.while (%arg3 = %17, %arg4 = %18) : (!llvm.ptr, !llvm.ptr) -> (!llvm.ptr, !llvm.ptr) {
        %26 = llvm.getelementptr inbounds|nuw %arg3[32] : (!llvm.ptr) -> !llvm.ptr, i8
        %27 = llvm.load %26 {alignment = 1 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i8
        %28 = arith.cmpi slt, %27, %c121_i8 : i8
        %29 = arith.select %28, %arg4, %arg3 {fastmathFlags = #llvm.fastmath<none>} : !llvm.ptr
        %30 = arith.select %28, %c24_i64, %c16_i64 {fastmathFlags = #llvm.fastmath<none>} : i64
        %31 = llvm.getelementptr inbounds|nuw %arg3[%30] : (!llvm.ptr, i64) -> !llvm.ptr, i8
        %32 = llvm.load %31 {alignment = 8 : i64, tbaa = [#tbaa_tag4]} : !llvm.ptr -> !llvm.ptr
        %33 = llvm.icmp "eq" %32, %5 : !llvm.ptr
        %34 = arith.xori %33, %true : i1
        scf.condition(%34) %32, %29 : !llvm.ptr, !llvm.ptr
      } do {
      ^bb0(%arg3: !llvm.ptr, %arg4: !llvm.ptr):
        scf.yield %arg3, %arg4 : !llvm.ptr, !llvm.ptr
      }
      %24 = llvm.icmp "eq" %23#1, %18 : !llvm.ptr
      %25 = scf.if %24 -> (i32) {
        scf.yield %c0_i32 : i32
      } else {
        %26 = llvm.getelementptr inbounds|nuw %23#1[32] : (!llvm.ptr) -> !llvm.ptr, i8
        %27 = llvm.load %26 {alignment = 1 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i8
        %28 = arith.cmpi sgt, %27, %c121_i8 : i8
        %29 = scf.if %28 -> (i32) {
          scf.yield %c0_i32 : i32
        } else {
          %30 = llvm.getelementptr inbounds|nuw %23#1[40] : (!llvm.ptr) -> !llvm.ptr, i8
          %31 = llvm.load %30 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
          %32 = llvm.getelementptr inbounds|nuw %31[16] : (!llvm.ptr) -> !llvm.ptr, i8
          %33 = llvm.load %32 {alignment = 8 : i64, tbaa = [#tbaa_tag16]} : !llvm.ptr -> !llvm.ptr
          %34 = llvm.getelementptr inbounds|nuw %arg0[368] : (!llvm.ptr) -> !llvm.ptr, i8
          %35 = llvm.getelementptr inbounds|nuw %arg0[376] : (!llvm.ptr) -> !llvm.ptr, i8
          %36 = llvm.load %35 {alignment = 8 : i64, tbaa = [#tbaa_tag12]} : !llvm.ptr -> !llvm.ptr
          %37 = llvm.load %34 {alignment = 8 : i64, tbaa = [#tbaa_tag13]} : !llvm.ptr -> !llvm.ptr
          %38 = llvm.icmp "eq" %36, %37 : !llvm.ptr
          %39:2 = scf.if %38 -> (i32, i32) {
            scf.yield %1, %c0_i32 : i32, i32
          } else {
            %43 = llvm.getelementptr inbounds|nuw %37[16] : (!llvm.ptr) -> !llvm.ptr, i8
            %44 = llvm.load %43 {alignment = 8 : i64, tbaa = [#tbaa_tag16]} : !llvm.ptr -> !llvm.ptr
            %45 = llvm.getelementptr inbounds|nuw %arg0[296] : (!llvm.ptr) -> !llvm.ptr, i8
            %46 = llvm.getelementptr inbounds|nuw %arg0[304] : (!llvm.ptr) -> !llvm.ptr, i8
            %47 = llvm.load %46 {alignment = 8 : i64, tbaa = [#tbaa_tag12]} : !llvm.ptr -> !llvm.ptr
            %48 = llvm.load %45 {alignment = 8 : i64, tbaa = [#tbaa_tag13]} : !llvm.ptr -> !llvm.ptr
            %49 = llvm.icmp "eq" %47, %48 : !llvm.ptr
            %50 = arith.xori %49, %true : i1
            %51 = arith.extui %50 : i1 to i32
            %52 = scf.if %49 -> (i32) {
              scf.yield %1 : i32
            } else {
              %53 = llvm.getelementptr inbounds|nuw %48[16] : (!llvm.ptr) -> !llvm.ptr, i8
              %54 = llvm.load %53 {alignment = 8 : i64, tbaa = [#tbaa_tag16]} : !llvm.ptr -> !llvm.ptr
              %55 = llvm.getelementptr inbounds|nuw %arg0[8] : (!llvm.ptr) -> !llvm.ptr, i8
              %56 = llvm.getelementptr inbounds|nuw %arg0[168] : (!llvm.ptr) -> !llvm.ptr, i8
              %57 = llvm.load %56 {alignment = 8 : i64, tbaa = [#tbaa_tag15]} : !llvm.ptr -> !llvm.ptr
              %58 = llvm.getelementptr inbounds|nuw %arg0[160] : (!llvm.ptr) -> !llvm.ptr, i8
              %59 = llvm.icmp "eq" %57, %5 : !llvm.ptr
              %60 = scf.if %59 -> (i32) {
                scf.yield %c0_i32 : i32
              } else {
                %61:2 = scf.while (%arg3 = %57, %arg4 = %58) : (!llvm.ptr, !llvm.ptr) -> (!llvm.ptr, !llvm.ptr) {
                  %64 = llvm.getelementptr inbounds|nuw %arg3[32] : (!llvm.ptr) -> !llvm.ptr, i8
                  %65 = llvm.load %64 {alignment = 1 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i8
                  %66 = arith.cmpi slt, %65, %c121_i8 : i8
                  %67 = arith.select %66, %arg4, %arg3 {fastmathFlags = #llvm.fastmath<none>} : !llvm.ptr
                  %68 = arith.select %66, %c24_i64, %c16_i64 {fastmathFlags = #llvm.fastmath<none>} : i64
                  %69 = llvm.getelementptr inbounds|nuw %arg3[%68] : (!llvm.ptr, i64) -> !llvm.ptr, i8
                  %70 = llvm.load %69 {alignment = 8 : i64, tbaa = [#tbaa_tag4]} : !llvm.ptr -> !llvm.ptr
                  %71 = llvm.icmp "eq" %70, %5 : !llvm.ptr
                  %72 = arith.xori %71, %true : i1
                  scf.condition(%72) %70, %67 : !llvm.ptr, !llvm.ptr
                } do {
                ^bb0(%arg3: !llvm.ptr, %arg4: !llvm.ptr):
                  scf.yield %arg3, %arg4 : !llvm.ptr, !llvm.ptr
                }
                %62 = llvm.icmp "eq" %61#1, %58 : !llvm.ptr
                %63 = scf.if %62 -> (i32) {
                  scf.yield %c0_i32 : i32
                } else {
                  %64 = llvm.getelementptr inbounds|nuw %61#1[32] : (!llvm.ptr) -> !llvm.ptr, i8
                  %65 = llvm.load %64 {alignment = 1 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i8
                  %66 = arith.cmpi sgt, %65, %c121_i8 : i8
                  %67 = scf.if %66 -> (i32) {
                    scf.yield %c0_i32 : i32
                  } else {
                    %68 = llvm.getelementptr inbounds|nuw %61#1[40] : (!llvm.ptr) -> !llvm.ptr, i8
                    %69 = llvm.load %68 {alignment = 8 : i64, tbaa = [#tbaa_tag6]} : !llvm.ptr -> !llvm.ptr
                    %70 = llvm.getelementptr inbounds|nuw %69[8] : (!llvm.ptr) -> !llvm.ptr, i8
                    %71 = llvm.getelementptr inbounds|nuw %arg0[1104] : (!llvm.ptr) -> !llvm.ptr, i8
                    %72 = llvm.load %71 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> i64
                    %73 = llvm.getelementptr inbounds|nuw %arg0[1112] : (!llvm.ptr) -> !llvm.ptr, i8
                    %74 = llvm.load %73 {alignment = 8 : i64, tbaa = [#tbaa_tag7]} : !llvm.ptr -> !llvm.ptr
                    %75 = llvm.getelementptr inbounds|nuw %arg0[1136] : (!llvm.ptr) -> !llvm.ptr, i8
                    %76 = llvm.load %75 {alignment = 8 : i64, tbaa = [#tbaa_tag15]} : !llvm.ptr -> !llvm.ptr
                    %77 = llvm.getelementptr inbounds|nuw %arg0[1128] : (!llvm.ptr) -> !llvm.ptr, i8
                    %78 = llvm.icmp "eq" %76, %5 : !llvm.ptr
                    %79 = scf.if %78 -> (i32) {
                      scf.yield %c0_i32 : i32
                    } else {
                      %80:2 = scf.while (%arg3 = %76, %arg4 = %77) : (!llvm.ptr, !llvm.ptr) -> (!llvm.ptr, !llvm.ptr) {
                        %83 = llvm.getelementptr inbounds|nuw %arg3[32] : (!llvm.ptr) -> !llvm.ptr, i8
                        %84 = llvm.load %83 {alignment = 1 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i8
                        %85 = arith.cmpi slt, %84, %c121_i8 : i8
                        %86 = arith.select %85, %arg4, %arg3 {fastmathFlags = #llvm.fastmath<none>} : !llvm.ptr
                        %87 = arith.select %85, %c24_i64, %c16_i64 {fastmathFlags = #llvm.fastmath<none>} : i64
                        %88 = llvm.getelementptr inbounds|nuw %arg3[%87] : (!llvm.ptr, i64) -> !llvm.ptr, i8
                        %89 = llvm.load %88 {alignment = 8 : i64, tbaa = [#tbaa_tag4]} : !llvm.ptr -> !llvm.ptr
                        %90 = llvm.icmp "eq" %89, %5 : !llvm.ptr
                        %91 = arith.xori %90, %true : i1
                        scf.condition(%91) %89, %86 : !llvm.ptr, !llvm.ptr
                      } do {
                      ^bb0(%arg3: !llvm.ptr, %arg4: !llvm.ptr):
                        scf.yield %arg3, %arg4 : !llvm.ptr, !llvm.ptr
                      }
                      %81 = llvm.icmp "eq" %80#1, %77 : !llvm.ptr
                      %82 = scf.if %81 -> (i32) {
                        scf.yield %c0_i32 : i32
                      } else {
                        %83 = llvm.getelementptr inbounds|nuw %80#1[32] : (!llvm.ptr) -> !llvm.ptr, i8
                        %84 = llvm.load %83 {alignment = 1 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i8
                        %85 = arith.cmpi sgt, %84, %c121_i8 : i8
                        %86 = arith.select %85, %c0_i32, %c2_i32 : i32
                        %87 = arith.cmpi sle, %84, %c121_i8 : i8
                        scf.if %87 {
                          %88 = llvm.getelementptr inbounds|nuw %80#1[40] : (!llvm.ptr) -> !llvm.ptr, i8
                          %89 = llvm.load %88 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> i64
                          %90 = llvm.getelementptr inbounds|nuw %80#1[48] : (!llvm.ptr) -> !llvm.ptr, i8
                          %91 = llvm.load %90 {alignment = 8 : i64, tbaa = [#tbaa_tag8]} : !llvm.ptr -> !llvm.ptr
                          %92 = arith.cmpi eq, %arg1, %c-1_i32 : i32
                          %93 = llvm.getelementptr inbounds|nuw %arg0[12] : (!llvm.ptr) -> !llvm.ptr, i8
                          %94 = llvm.load %93 {alignment = 4 : i64} : !llvm.ptr -> i32
                          %95 = arith.select %92, %94, %arg1 {fastmathFlags = #llvm.fastmath<none>} : i32
                          %96 = arith.cmpi eq, %arg2, %c-1_i32 : i32
                          %97 = arith.select %96, %c32_i32, %arg2 {fastmathFlags = #llvm.fastmath<none>} : i32
                          %98 = llvm.getelementptr inbounds|nuw %arg0[1384] : (!llvm.ptr) -> !llvm.ptr, i8
                          %99 = llvm.load %98 {alignment = 8 : i64, tbaa = [#tbaa_tag17]} : !llvm.ptr -> i8
                          %100 = arith.trunci %99 : i8 to i1
                          %101 = llvm.getelementptr inbounds|nuw %arg0[1432] : (!llvm.ptr) -> !llvm.ptr, i8
                          %102 = llvm.load %101 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
                          %103 = arith.select %100, %102, %5 {fastmathFlags = #llvm.fastmath<none>} : !llvm.ptr
                          %104 = arith.extui %95 : i32 to i64
                          %105 = arith.ori %104, %c4294967296_i64 {isDisjoint} : i64
                          %106 = arith.extui %97 : i32 to i64
                          %107 = arith.ori %106, %c4294967296_i64 {isDisjoint} : i64
                          "llvm.intr.memcpy"(%15, %55, %c36_i64) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
                          "llvm.intr.memcpy"(%14, %70, %c36_i64) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
                          llvm.store %72, %6 {alignment = 8 : i64} : i64, !llvm.ptr
                          %108 = llvm.getelementptr inbounds|nuw %6[8] : (!llvm.ptr) -> !llvm.ptr, i8
                          llvm.store %74, %108 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
                          llvm.store %89, %7 {alignment = 8 : i64} : i64, !llvm.ptr
                          %109 = llvm.getelementptr inbounds|nuw %7[8] : (!llvm.ptr) -> !llvm.ptr, i8
                          llvm.store %91, %109 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
                          llvm.store %33, %8 {alignment = 8 : i64, tbaa = [#tbaa_tag7]} : !llvm.ptr, !llvm.ptr
                          llvm.store %54, %9 {alignment = 8 : i64, tbaa = [#tbaa_tag7]} : !llvm.ptr, !llvm.ptr
                          llvm.store %5, %10 {alignment = 8 : i64, tbaa = [#tbaa_tag7]} : !llvm.ptr, !llvm.ptr
                          llvm.store %44, %11 {alignment = 8 : i64, tbaa = [#tbaa_tag7]} : !llvm.ptr, !llvm.ptr
                          llvm.store %5, %12 {alignment = 8 : i64, tbaa = [#tbaa_tag7]} : !llvm.ptr, !llvm.ptr
                          llvm.store %15, %13 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
                          %110 = llvm.getelementptr inbounds|nuw %13[8] : (!llvm.ptr) -> !llvm.ptr, i8
                          llvm.store %14, %110 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
                          %111 = llvm.getelementptr inbounds|nuw %13[16] : (!llvm.ptr) -> !llvm.ptr, i8
                          llvm.store %6, %111 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
                          %112 = llvm.getelementptr inbounds|nuw %13[24] : (!llvm.ptr) -> !llvm.ptr, i8
                          llvm.store %7, %112 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
                          %113 = llvm.getelementptr inbounds|nuw %13[32] : (!llvm.ptr) -> !llvm.ptr, i8
                          llvm.store %8, %113 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
                          %114 = llvm.getelementptr inbounds|nuw %13[40] : (!llvm.ptr) -> !llvm.ptr, i8
                          llvm.store %9, %114 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
                          %115 = llvm.getelementptr inbounds|nuw %13[48] : (!llvm.ptr) -> !llvm.ptr, i8
                          llvm.store %10, %115 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
                          %116 = llvm.getelementptr inbounds|nuw %13[56] : (!llvm.ptr) -> !llvm.ptr, i8
                          llvm.store %11, %116 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
                          %117 = llvm.getelementptr inbounds|nuw %13[64] : (!llvm.ptr) -> !llvm.ptr, i8
                          llvm.store %12, %117 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
                          %118 = arith.trunci %105 : i64 to i32
                          %119 = arith.trunci %107 : i64 to i32
                          %120 = llvm.load %13 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
                          %121 = llvm.getelementptr inbounds %13[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
                          %122 = llvm.load %121 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
                          %123 = llvm.getelementptr inbounds %13[2] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
                          %124 = llvm.load %123 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
                          %125 = llvm.getelementptr inbounds %13[3] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
                          %126 = llvm.load %125 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
                          %127 = llvm.getelementptr inbounds %13[4] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
                          %128 = llvm.load %127 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
                          %129 = llvm.load %128 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
                          %130 = llvm.getelementptr inbounds %13[5] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
                          %131 = llvm.load %130 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
                          %132 = llvm.load %131 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
                          %133 = llvm.getelementptr inbounds %13[7] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
                          %134 = llvm.load %133 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
                          %135 = llvm.load %134 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
                          %136 = arith.index_cast %118 : i32 to index
                          %137 = arith.index_cast %119 : i32 to index
                          %138 = llvm.load %120 : !llvm.ptr -> !llvm.struct<"struct.cuda_Struct_Grid", (i8, i8, i32, i32, i32, i32, i32, i32, i32, i32)>
                          %139 = llvm.load %122 : !llvm.ptr -> !llvm.struct<"struct.cuda_Struct_Grid", (i8, i8, i32, i32, i32, i32, i32, i32, i32, i32)>
                          %140 = llvm.load %124 : !llvm.ptr -> !llvm.struct<"class.cuda_run_time_vector_shallow_copy", (i64, ptr)>
                          %141 = llvm.load %126 : !llvm.ptr -> !llvm.struct<"class.cuda_run_time_vector_shallow_copy", (i64, ptr)>
                          %142 = "enzymexla.stream2token"(%103) : (!llvm.ptr) -> !async.token
                          %token = async.execute [%142] {
                            %143 = "enzymexla.gpu_wrapper"(%136, %c1, %c1, %137, %c1, %c1) ({
                              affine.parallel (%arg3, %arg4) = (0, 0) to (symbol(%136), symbol(%137)) {
                                %144 = llvm.alloca %4 x !llvm.struct<"class.cuda_run_time_vector_shallow_copy", (i64, ptr)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
                                %145 = llvm.alloca %4 x !llvm.struct<"class.cuda_run_time_vector_shallow_copy", (i64, ptr)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
                                %146 = llvm.alloca %4 x !llvm.struct<"struct.cuda_Struct_Grid", (i8, i8, i32, i32, i32, i32, i32, i32, i32, i32)> {alignment = 4 : i64} : (i64) -> !llvm.ptr
                                %147 = llvm.alloca %4 x !llvm.struct<"struct.cuda_Struct_Grid", (i8, i8, i32, i32, i32, i32, i32, i32, i32, i32)> {alignment = 4 : i64} : (i64) -> !llvm.ptr
                                %148 = llvm.alloca %4 x !llvm.struct<"struct.cuda_Struct_Grid", (i8, i8, i32, i32, i32, i32, i32, i32, i32, i32)> {alignment = 4 : i64} : (i64) -> !llvm.ptr
                                llvm.store %138, %148 : !llvm.struct<"struct.cuda_Struct_Grid", (i8, i8, i32, i32, i32, i32, i32, i32, i32, i32)>, !llvm.ptr
                                %149 = llvm.alloca %4 x !llvm.struct<"struct.cuda_Struct_Grid", (i8, i8, i32, i32, i32, i32, i32, i32, i32, i32)> {alignment = 4 : i64} : (i64) -> !llvm.ptr
                                llvm.store %139, %149 : !llvm.struct<"struct.cuda_Struct_Grid", (i8, i8, i32, i32, i32, i32, i32, i32, i32, i32)>, !llvm.ptr
                                %150 = llvm.alloca %4 x !llvm.struct<"class.cuda_run_time_vector_shallow_copy", (i64, ptr)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
                                llvm.store %140, %150 : !llvm.struct<"class.cuda_run_time_vector_shallow_copy", (i64, ptr)>, !llvm.ptr
                                %151 = llvm.alloca %4 x !llvm.struct<"class.cuda_run_time_vector_shallow_copy", (i64, ptr)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
                                llvm.store %141, %151 : !llvm.struct<"class.cuda_run_time_vector_shallow_copy", (i64, ptr)>, !llvm.ptr
                                "llvm.intr.memcpy"(%147, %148, %3) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
                                "llvm.intr.memcpy"(%146, %149, %3) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
                                "llvm.intr.memcpy"(%145, %150, %2) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
                                "llvm.intr.memcpy"(%144, %151, %2) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
                                %152 = arith.index_castui %137 : index to i32
                                %153 = arith.index_castui %arg4 : index to i32
                                %154 = llvm.getelementptr inbounds|nuw %146[1] : (!llvm.ptr) -> !llvm.ptr, i8
                                %155 = llvm.load %154 {alignment = 1 : i64, tbaa = [#tbaa_tag9]} : !llvm.ptr -> i8
                                %156 = arith.cmpi eq, %155, %c77_i8 : i8
                                %157 = llvm.getelementptr inbounds|nuw %146[8] : (!llvm.ptr) -> !llvm.ptr, i8
                                %158 = llvm.load %157 {alignment = 4 : i64} : !llvm.ptr -> i32
                                %159 = arith.select %156, %158, %c4_i32 {fastmathFlags = #llvm.fastmath<none>} : i32
                                %160 = arith.cmpi eq, %155, %c78_i8 : i8
                                %161 = arith.addi %158, %c-1_i32 : i32
                                %162 = arith.select %160, %161, %159 {fastmathFlags = #llvm.fastmath<none>} : i32
                                %163 = llvm.getelementptr inbounds|nuw %147[8] : (!llvm.ptr) -> !llvm.ptr, i8
                                %164 = llvm.load %163 {alignment = 4 : i64, tbaa = [#tbaa_tag10]} : !llvm.ptr -> i32
                                %165 = arith.cmpi slt, %153, %164 : i32
                                scf.if %165 {
                                  %166 = arith.index_castui %arg3 : index to i32
                                  %167 = llvm.getelementptr inbounds|nuw %147[20] : (!llvm.ptr) -> !llvm.ptr, i8
                                  %168 = llvm.load %167 {alignment = 4 : i64, tbaa = [#tbaa_tag11]} : !llvm.ptr -> i32
                                  %169 = arith.muli %168, %166 : i32
                                  %170 = llvm.getelementptr inbounds|nuw %146[20] : (!llvm.ptr) -> !llvm.ptr, i8
                                  %171 = llvm.load %170 {alignment = 4 : i64, tbaa = [#tbaa_tag11]} : !llvm.ptr -> i32
                                  %172 = arith.muli %171, %166 : i32
                                  %173 = llvm.load %145 {alignment = 8 : i64, tbaa = [#tbaa_tag14]} : !llvm.ptr -> i64
                                  %174 = arith.cmpi sgt, %173, %c0_i64 : i64
                                  %175 = llvm.getelementptr inbounds|nuw %144[8] : (!llvm.ptr) -> !llvm.ptr, i8
                                  %176 = llvm.load %175 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
                                  %177 = llvm.getelementptr inbounds|nuw %145[8] : (!llvm.ptr) -> !llvm.ptr, i8
                                  %178 = llvm.load %177 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
                                  %179 = arith.addi %153, %152 : i32
                                  %180 = arith.maxsi %164, %179 : i32
                                  %181 = arith.addi %180, %152 : i32
                                  scf.for %arg5 = %179 to %181 step %152  : i32 {
                                    %182 = arith.subi %arg5, %179 : i32
                                    %183 = arith.addi %153, %182 : i32
                                    %184 = arith.addi %169, %183 : i32
                                    %185 = arith.addi %173, %c1_i64 : i64
                                    %186 = scf.if %174 -> (f64) {
                                      %189 = arith.addi %183, %162 : i32
                                      %190:2 = scf.for %arg6 = %c1_i64 to %185 step %c1_i64 iter_args(%arg7 = %cst, %arg8 = %0) -> (f64, f64)  : i64 {
                                        %191 = arith.addi %arg6, %c-1_i64 : i64
                                        %192 = llvm.getelementptr inbounds|nuw %176[%191] : (!llvm.ptr, i64) -> !llvm.ptr, i32
                                        %193 = llvm.load %192 {alignment = 4 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> i32
                                        %194 = arith.addi %189, %193 : i32
                                        %195 = arith.remsi %194, %162 : i32
                                        %196 = arith.addi %195, %172 : i32
                                        %197 = arith.extsi %196 : i32 to i64
                                        %198 = llvm.getelementptr inbounds %129[%197] : (!llvm.ptr, i64) -> !llvm.ptr, f64
                                        %199 = llvm.load %198 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : !llvm.ptr -> f64
                                        %200 = llvm.getelementptr inbounds|nuw %178[%191] : (!llvm.ptr, i64) -> !llvm.ptr, f64
                                        %201 = llvm.load %200 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : !llvm.ptr -> f64
                                        %202 = arith.mulf %199, %201 {fastmathFlags = #llvm.fastmath<contract>} : f64
                                        %203 = arith.addf %arg7, %202 {fastmathFlags = #llvm.fastmath<contract>} : f64
                                        scf.yield %203, %203 : f64, f64
                                      }
                                      scf.yield %190#1 : f64
                                    } else {
                                      scf.yield %cst : f64
                                    }
                                    %187 = arith.extsi %184 : i32 to i64
                                    %188 = llvm.getelementptr inbounds %135[%187] : (!llvm.ptr, i64) -> !llvm.ptr, f64
                                    llvm.store %186, %188 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : f64, !llvm.ptr
                                  }
                                }
                              }
                              "enzymexla.polygeist_yield"() : () -> ()
                            }) : (index, index, index, index, index, index) -> index
                            async.yield
                          }
                        }
                        scf.yield %86 : i32
                      }
                      scf.yield %82 : i32
                    }
                    scf.yield %79 : i32
                  }
                  scf.yield %67 : i32
                }
                scf.yield %63 : i32
              }
              scf.yield %60 : i32
            }
            scf.yield %52, %51 : i32, i32
          }
          %40 = arith.index_castui %39#1 : i32 to index
          %41 = arith.cmpi eq, %40, %c0 : index
          %42 = arith.select %41, %c1_i32, %39#0 : i32
          scf.yield %42 : i32
        }
        scf.yield %29 : i32
      }
      scf.yield %25 : i32
    }
    %21 = arith.index_castui %20 : i32 to index
    %22 = scf.index_switch %21 -> i32 
    case 0 {
      scf.yield %c1_i32 : i32
    }
    case 1 {
      scf.yield %c1_i32 : i32
    }
    default {
      scf.yield %c0_i32 : i32
    }
    cf.switch %22 : i32, [
      default: ^bb1,
      0: ^bb2
    ]
  ^bb1:  // pred: ^bb0
    llvm.unreachable
  ^bb2:  // pred: ^bb0
    llvm.return
  }
}

// CHECK: affine.parallel (%arg3, %arg4) = (0, 0) to (symbol(%133), symbol(%134))
