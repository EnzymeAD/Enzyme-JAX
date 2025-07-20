// RUN: enzymexlamlir-opt -allow-unregistered-dialect --canonicalize-scf-for --split-input-file %s | FileCheck %s

#tbaa_root = #llvm.tbaa_root<id = "Simple C++ TBAA">
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root, 0>}>
#tbaa_type_desc1 = #llvm.tbaa_type_desc<id = "double", members = {<#tbaa_type_desc, 0>}>
#tbaa_type_desc2 = #llvm.tbaa_type_desc<id = "_ZTS4Vec3", members = {<#tbaa_type_desc1, 0>, <#tbaa_type_desc1, 8>, <#tbaa_type_desc1, 16>}>
#tbaa_type_desc3 = #llvm.tbaa_type_desc<id = "_ZTS8Particle", members = {<#tbaa_type_desc2, 0>, <#tbaa_type_desc2, 24>, <#tbaa_type_desc2, 48>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc3, access_type = #tbaa_type_desc1, offset = 48>
#tbaa_tag1 = #llvm.tbaa_tag<base_type = #tbaa_type_desc3, access_type = #tbaa_type_desc1, offset = 56>
#tbaa_tag2 = #llvm.tbaa_tag<base_type = #tbaa_type_desc3, access_type = #tbaa_type_desc1, offset = 64>
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  llvm.func local_unnamed_addr @__mlir_launch_kernel__Z35__device_stub__negate_forces_kernelP8Particleii(!llvm.ptr, i32, i32, i32, i32, i32, i32, i64, !llvm.ptr, !llvm.ptr, i32, i32) attributes {sym_visibility = "private"}
  llvm.func internal @_Z35__device_stub__negate_forces_kernelP8Particleii(%arg0: !llvm.ptr {llvm.nocapture, llvm.nofree, llvm.noundef}, %arg1: i32 {llvm.noundef}, %arg2: i32 {llvm.noundef}) attributes {dso_local, frame_pointer = #llvm.framePointerKind<all>, memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "sm_52"], ["uniform-work-group-size", "true"]], sym_visibility = "private", target_cpu = "sm_52", target_features = #llvm.target_features<["+ptx80", "+sm_52"]>, will_return} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %block_id_x = gpu.block_id  x
    %0 = arith.index_castui %block_id_x : index to i32
    %1 = arith.cmpi slt, %0, %arg2 : i32
    scf.if %1 {
      %2 = arith.muli %arg1, %0 : i32
      %3 = arith.extsi %2 : i32 to i64
      %4 = llvm.getelementptr inbounds %arg0[%3] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Particle.1", (struct<"struct.Vec3.1", (f64, f64, f64)>, struct<"struct.Vec3.1", (f64, f64, f64)>, struct<"struct.Vec3.1", (f64, f64, f64)>)>
      %5 = arith.cmpi sgt, %arg1, %c0_i32 : i32
      scf.if %5 {
        %6 = scf.while (%arg3 = %c0_i32) : (i32) -> i32 {
          %7 = arith.extui %arg3 {nonNeg} : i32 to i64
          %8 = llvm.getelementptr inbounds|nuw %4[%7, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Particle.1", (struct<"struct.Vec3.1", (f64, f64, f64)>, struct<"struct.Vec3.1", (f64, f64, f64)>, struct<"struct.Vec3.1", (f64, f64, f64)>)>
          %9 = llvm.load %8 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f64
          %10 = arith.negf %9 {fastmathFlags = #llvm.fastmath<contract>} : f64
          llvm.store %10, %8 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : f64, !llvm.ptr
          %11 = llvm.getelementptr inbounds|nuw %8[8] : (!llvm.ptr) -> !llvm.ptr, i8
          %12 = llvm.load %11 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
          %13 = arith.negf %12 {fastmathFlags = #llvm.fastmath<contract>} : f64
          llvm.store %13, %11 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
          %14 = llvm.getelementptr inbounds|nuw %8[16] : (!llvm.ptr) -> !llvm.ptr, i8
          %15 = llvm.load %14 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> f64
          %16 = arith.negf %15 {fastmathFlags = #llvm.fastmath<contract>} : f64
          llvm.store %16, %14 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : f64, !llvm.ptr
          %17 = arith.addi %arg3, %c1_i32 : i32
          %18 = arith.cmpi ne, %17, %arg1 : i32
          scf.condition(%18) %17 : i32
        } do {
        ^bb0(%arg3: i32):
          scf.yield %arg3 : i32
        }
      }
    }
    llvm.return
  }
}

// CHECK:  llvm.func internal @_Z35__device_stub__negate_forces_kernelP8Particleii(%arg0: !llvm.ptr {llvm.nocapture, llvm.nofree, llvm.noundef}, %arg1: i32 {llvm.noundef}, %arg2: i32 {llvm.noundef}) attributes {dso_local, frame_pointer = #llvm.framePointerKind<all>, memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "sm_52"], ["uniform-work-group-size", "true"]], sym_visibility = "private", target_cpu = "sm_52", target_features = #llvm.target_features<["+ptx80", "+sm_52"]>, will_return} {
// CHECK-NEXT:    %c-1_i32 = arith.constant -1 : i32
// CHECK-NEXT:    %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:    %block_id_x = gpu.block_id  x
// CHECK-NEXT:    %0 = arith.index_castui %block_id_x : index to i32
// CHECK-NEXT:    %1 = arith.cmpi slt, %0, %arg2 : i32
// CHECK-NEXT:    scf.if %1 {
// CHECK-NEXT:      %2 = arith.muli %arg1, %0 : i32
// CHECK-NEXT:      %3 = arith.extsi %2 : i32 to i64
// CHECK-NEXT:      %4 = llvm.getelementptr inbounds %arg0[%3] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Particle.1", (struct<"struct.Vec3.1", (f64, f64, f64)>, struct<"struct.Vec3.1", (f64, f64, f64)>, struct<"struct.Vec3.1", (f64, f64, f64)>)>
// CHECK-NEXT:      %5 = arith.addi %arg1, %c1_i32 : i32
// CHECK-NEXT:      scf.for %arg3 = %c1_i32 to %5 step %c1_i32  : i32 {
// CHECK-NEXT:        %6 = arith.addi %arg3, %c-1_i32 : i32
// CHECK-NEXT:        %7 = arith.extui %6 {nonNeg} : i32 to i64
// CHECK-NEXT:        %8 = llvm.getelementptr inbounds|nuw %4[%7, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Particle.1", (struct<"struct.Vec3.1", (f64, f64, f64)>, struct<"struct.Vec3.1", (f64, f64, f64)>, struct<"struct.Vec3.1", (f64, f64, f64)>)>
// CHECK-NEXT:        %9 = llvm.load %8 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f64
// CHECK-NEXT:        %10 = arith.negf %9 {fastmathFlags = #llvm.fastmath<contract>} : f64
// CHECK-NEXT:        llvm.store %10, %8 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : f64, !llvm.ptr
// CHECK-NEXT:        %11 = llvm.getelementptr inbounds|nuw %8[8] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK-NEXT:        %12 = llvm.load %11 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
// CHECK-NEXT:        %13 = arith.negf %12 {fastmathFlags = #llvm.fastmath<contract>} : f64
// CHECK-NEXT:        llvm.store %13, %11 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
// CHECK-NEXT:        %14 = llvm.getelementptr inbounds|nuw %8[16] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK-NEXT:        %15 = llvm.load %14 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> f64
// CHECK-NEXT:        %16 = arith.negf %15 {fastmathFlags = #llvm.fastmath<contract>} : f64
// CHECK-NEXT:        llvm.store %16, %14 {alignment = 8 : i64, tbaa = [#tbaa_tag2]} : f64, !llvm.ptr
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }