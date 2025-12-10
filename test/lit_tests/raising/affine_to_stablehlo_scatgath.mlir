// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo | FileCheck %s

#alias_scope_domain = #llvm.alias_scope_domain<id = distinct[0]<>, description = "_Z27performStreamCollide_kernelPvPf">
#map = affine_map<(d0) -> (d0 * 4 + 179773440)>
#set = affine_set<(d0) : (d0 - 2 >= 0, -d0 + 117 >= 0)>
#set1 = affine_set<()[s0] : (s0 == 0)>
#tbaa_root = #llvm.tbaa_root<id = "Simple C++ TBAA">
#alias_scope = #llvm.alias_scope<id = distinct[1]<>, domain = #alias_scope_domain, description = "_Z27performStreamCollide_kernelPvPf: argument 0">
#alias_scope1 = #llvm.alias_scope<id = distinct[2]<>, domain = #alias_scope_domain, description = "_Z27performStreamCollide_kernelPvPf: argument 1">
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root, 0>}>
#tbaa_type_desc1 = #llvm.tbaa_type_desc<id = "any pointer", members = {<#tbaa_type_desc, 0>}>
#tbaa_type_desc2 = #llvm.tbaa_type_desc<id = "int", members = {<#tbaa_type_desc, 0>}>
#tbaa_type_desc3 = #llvm.tbaa_type_desc<id = "float", members = {<#tbaa_type_desc, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc2, access_type = #tbaa_type_desc2, offset = 0>
#tbaa_tag1 = #llvm.tbaa_tag<base_type = #tbaa_type_desc3, access_type = #tbaa_type_desc3, offset = 0>
#tbaa_type_desc4 = #llvm.tbaa_type_desc<id = "p1 _ZTS8_IO_FILE", members = {<#tbaa_type_desc1, 0>}>
#tbaa_type_desc5 = #llvm.tbaa_type_desc<id = "p1 float", members = {<#tbaa_type_desc1, 0>}>
#tbaa_tag2 = #llvm.tbaa_tag<base_type = #tbaa_type_desc4, access_type = #tbaa_type_desc4, offset = 0>
#tbaa_tag3 = #llvm.tbaa_tag<base_type = #tbaa_type_desc5, access_type = #tbaa_type_desc5, offset = 0>
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  llvm.func internal unnamed_addr fastcc @_ZL4kernPfS_(%arg0: !llvm.ptr {llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}) attributes {dso_local, no_infs_fp_math = true, no_inline, no_nans_fp_math = true, no_signed_zeros_fp_math = true, no_unwind, passthrough = ["mustprogress", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uwtable_kind = #llvm.uwtableKind<async>} {
    %c-1_i32 = arith.constant -1 : i32
    %c51_i32 = arith.constant 51 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.0541666672 : f32
    %cst_0 = arith.constant -3.000000e+00 : f32
    %cst_1 = arith.constant 3.000000e+00 : f32
    %cst_2 = arith.constant 4.500000e+00 : f32
    %cst_3 = arith.constant 0.950000047 : f32
    %cst_4 = arith.constant 0.108333334 : f32
    %cst_5 = arith.constant -0.950000047 : f32
    %cst_6 = arith.constant 0.650000035 : f32
    %cst_7 = arith.constant -1.000000e+00 : f32
    %cst_8 = arith.constant 1.500000e+00 : f32
    %cst_9 = arith.constant 0.000000e+00 : f32
    %cst_10 = arith.constant 2.000000e-03 : f32
    %cst_11 = arith.constant 5.000000e-03 : f32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %c150 = arith.constant 150 : index
    %c120 = arith.constant 120 : index
    %2 = "enzymexla.gpu_wrapper"(%c120, %c150, %c1, %c120, %c1, %c1) ({
      affine.parallel (%arg2, %arg3) = (0, 0) to (18000, 120) {
        %9 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf32>
        %10 = affine.load %9[%arg3 + %arg2 * 128] : memref<?xf32>
        %272 = "enzymexla.pointer2memref"(%arg1) : (!llvm.ptr) -> memref<?xf32>
        affine.store %10, %272[%arg3 + %arg2 * 128] : memref<?xf32>
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    llvm.return
  }
}

// Define arg2 = 0 - 18000
// Define arg3 = 0 - 120

// CHECK:  func.func private @raised(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> (tensor<?xf32>, tensor<?xf32>) {

// arg2
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<18000xi64>          

// 0
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<18000xi64>     

// arg2
// CHECK-NEXT:    %1 = stablehlo.add %0, %c : tensor<18000xi64>            

// 1
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<18000xi64>   

// arg2
// CHECK-NEXT:    %2 = stablehlo.multiply %1, %c_0 : tensor<18000xi64>     

// arg3
// CHECK-NEXT:    %3 = stablehlo.iota dim = 0 : tensor<120xi64>            

// 0
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<0> : tensor<120xi64>     

// arg3
// CHECK-NEXT:    %4 = stablehlo.add %3, %c_1 : tensor<120xi64>            

// 1
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<1> : tensor<120xi64>     

// arg3
// CHECK-NEXT:    %5 = stablehlo.multiply %4, %c_2 : tensor<120xi64>       

// 128
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<128> : tensor<i64>       

// 128
// CHECK-NEXT:    %6 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i64>) -> tensor<18000xi64>   

// arg2 * 128
// CHECK-NEXT:    %7 = stablehlo.multiply %2, %6 : tensor<18000xi64>       

// [arg3, :         ]
// CHECK-NEXT:    %8 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<120xi64>) -> tensor<120x18000xi64>   

// [:   , arg2 * 128] 
// CHECK-NEXT:    %9 = stablehlo.broadcast_in_dim %7, dims = [1] : (tensor<18000xi64>) -> tensor<120x18000xi64> 

// arg3 + arg2 * 128 (axis is arg3, arg2)
// CHECK-NEXT:    %10 = stablehlo.add %8, %9 : tensor<120x18000xi64>       

//  arg3 + arg2 * 128 (axis is reshape(arg3, arg2))
// CHECK-NEXT:    %11 = stablehlo.reshape %10 : (tensor<120x18000xi64>) -> tensor<120x18000x1xi64> 

//  arg3 + arg2 * 128 (axis is reshape(arg3, arg2))
// CHECK-NEXT:    %12 = stablehlo.reshape %11 : (tensor<120x18000x1xi64>) -> tensor<2160000x1xi64> 

// arg0[arg3 + arg2 * 128] (axis is reshape(arg3, arg2))
// CHECK-NEXT:    %13 = "stablehlo.gather"(%arg0, %12) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<?xf32>, tensor<2160000x1xi64>) -> tensor<2160000xf32>    

// arg0[arg3 + arg2 * 128] (axis is (arg3, arg2))
// CHECK-NEXT:    %14 = stablehlo.reshape %13 : (tensor<2160000xf32>) -> tensor<120x18000xf32>  

// 128
// CHECK-NEXT:    %c_4 = stablehlo.constant dense<128> : tensor<i64>       

// 128
// CHECK-NEXT:    %15 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i64>) -> tensor<18000xi64> 

// arg2 * 128
// CHECK-NEXT:    %16 = stablehlo.multiply %2, %15 : tensor<18000xi64>     

// [arg3, :]
// CHECK-NEXT:    %17 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<120xi64>) -> tensor<120x18000xi64>    

// [:   , arg2 * 128]
// CHECK-NEXT:    %18 = stablehlo.broadcast_in_dim %16, dims = [1] : (tensor<18000xi64>) -> tensor<120x18000xi64> 

// arg3 + arg2 * 128 (axis is arg3, arg2)
// CHECK-NEXT:    %19 = stablehlo.add %17, %18 : tensor<120x18000xi64>     

//    arg3 + arg2 * 128 (axis is reshape(arg3, arg2))
// CHECK-NEXT:    %20 = stablehlo.reshape %19 : (tensor<120x18000xi64>) -> tensor<2160000x1xi64> 

// arg0[arg3 + arg2 * 128] (axis is (arg3, arg2))
// CHECK-NEXT:    %21 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<120x18000xf32>) -> tensor<120x18000xf32> 

//  arg3 + arg2 * 128 (axis is reshape(arg3, arg2))
// CHECK-NEXT:    %22 = stablehlo.reshape %21 : (tensor<120x18000xf32>) -> tensor<2160000xf32>  

// CHECK-NEXT:    %23 = "stablehlo.scatter"(%arg1, %20, %22) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
// CHECK-NEXT:    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
// CHECK-NEXT:      stablehlo.return %arg3 : tensor<f32>
// CHECK-NEXT:    }) : (tensor<?xf32>, tensor<2160000x1xi64>, tensor<2160000xf32>) -> tensor<?xf32>  
// CHECK-NEXT:    return %arg0, %23 : tensor<?xf32>, tensor<?xf32>
// CHECK-NEXT:  }

