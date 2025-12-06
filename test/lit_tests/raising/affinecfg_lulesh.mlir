// RUN: enzymexlamlir-opt --affine-cfg --split-input-file --allow-unregistered-dialect %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<6> = dense<32> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.legal_int_widths" = array<i32: 16, 32, 64>>, llvm.target_triple = "nvptx64-nvidia-cuda"} {
  llvm.mlir.global internal unnamed_addr constant @__cudart_i2opi_f(dense<[1011060801, -614296167, -181084736, -64530479, 1313084713, -1560706194]> : tensor<6xi32>) {addr_space = 1 : i32, alignment = 4 : i64, dso_local} : !llvm.array<6 x i32>
  llvm.func local_unnamed_addr @__nv_jnf(%arg0: i32 {llvm.noundef}, %arg1: f32) -> f32 attributes {dso_local, no_infs_fp_math = false, no_inline, no_nans_fp_math = false, no_unwind, passthrough = ["nofree", "norecurse", "nosync", ["disable-tail-calls", "false"], ["enzyme_math", "jnf"], ["implements", "llvm.jn.f32"], ["implements2", "jnf"], ["less-precise-fpmad", "false"], ["no-trapping-math", "true"], "prev_always_inline", "prev_fixup", ["prev_linkage", "7"], ["stack-protector-buffer-size", "8"], ["target-cpu", "sm_60"], ["use-soft-float", "false"]], target_cpu = "sm_60", target_features = #llvm.target_features<["+ptx85", "+sm_60"]>, unsafe_fp_math = false} {
    %0 = ub.poison : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 8.000000e+00 : f32
    %cst_0 = arith.constant -3.83170605 : f32
    %cst_1 = arith.constant 7.68505898E-8 : f32
    %cst_2 = arith.constant 7.78064881E-14 : f32
    %cst_3 = arith.constant 9.219086E-13 : f32
    %cst_4 = arith.constant -2.57068828E-11 : f32
    %cst_5 = arith.constant -2.0179057E-10 : f32
    %cst_6 = arith.constant 4.51252768E-9 : f32
    %cst_7 = arith.constant 2.70162896E-8 : f32
    %cst_8 = arith.constant -5.34779701E-7 : f32
    %cst_9 = arith.constant -2.36027631E-6 : f32
    %cst_10 = arith.constant 4.12108093E-5 : f32
    %cst_11 = arith.constant 1.1917029E-4 : f32
    %cst_12 = arith.constant -0.00180755882 : f32
    %cst_13 = arith.constant -0.0025548928 : f32
    %cst_14 = arith.constant 0.0330138914 : f32
    %cst_15 = arith.constant -7.01558685 : f32
    %cst_16 = arith.constant 1.83211725E-7 : f32
    %cst_17 = arith.constant 0x7F800000 : f32
    %cst_18 = arith.constant 0.000000e+00 : f32
    %cst_19 = arith.constant -4.0873065 : f32
    %cst_20 = arith.constant 0.749876558 : f32
    %cst_21 = arith.constant -0.192915648 : f32
    %cst_22 = arith.constant 0.187498257 : f32
    %cst_23 = arith.constant 1.000000e+00 : f32
    %cst_24 = arith.constant -1.57994485 : f32
    %cst_25 = arith.constant 0.361485869 : f32
    %cst_26 = arith.constant -0.164012611 : f32
    %cst_27 = arith.constant 0.374999911 : f32
    %cst_28 = arith.constant 0.797884583 : f32
    %cst_29 = arith.constant 0.636619746 : f32
    %cst_30 = arith.constant -1.57079625 : f32
    %cst_31 = arith.constant -7.54978942E-8 : f32
    %cst_32 = arith.constant -5.39030295E-15 : f32
    %cst_33 = arith.constant 1.056150e+05 : f32
    %c8_i32 = arith.constant 8 : i32
    %c-2147483648_i32 = arith.constant -2147483648 : i32
    %1 = llvm.mlir.addressof @__cudart_i2opi_f : !llvm.ptr<1>
    %c6_i32 = arith.constant 6 : i32
    %c23_i32 = arith.constant 23 : i32
    %c224_i32 = arith.constant 224 : i32
    %c-128_i32 = arith.constant -128 : i32
    %c5_i32 = arith.constant 5 : i32
    %c260046848_i32 = arith.constant 260046848 : i32
    %c4_i32 = arith.constant 4 : i32
    %c30_i32 = arith.constant 30 : i32
    %c2_i32 = arith.constant 2 : i32
    %c31_i32 = arith.constant 31 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c32_i64 = arith.constant 32 : i64
    %cst_34 = arith.constant 8.5153039502163873E-20 : f64
    %c3_i32 = arith.constant 3 : i32
    %cst_35 = arith.constant 1.57079637 : f32
    %cst_36 = arith.constant -2.3561945 : f32
    %cst_37 = arith.constant 2.42795795E-5 : f32
    %cst_38 = arith.constant -0.00138878601 : f32
    %cst_39 = arith.constant -1.95746587E-4 : f32
    %cst_40 = arith.constant 0.0416667275 : f32
    %cst_41 = arith.constant 0.00833270326 : f32
    %cst_42 = arith.constant -0.49999997 : f32
    %cst_43 = arith.constant -0.166666627 : f32
    %cst_44 = arith.constant 1.000000e-30 : f32
    %cst_45 = arith.constant -2.40482545 : f32
    %cst_46 = arith.constant -1.08705898E-7 : f32
    %cst_47 = arith.constant -1.2470738E-15 : f32
    %cst_48 = arith.constant -7.66777253E-14 : f32
    %cst_49 = arith.constant 2.71505561E-12 : f32
    %cst_50 = arith.constant -6.02801186E-12 : f32
    %cst_51 = arith.constant -4.23932667E-10 : f32
    %cst_52 = arith.constant 5.82763782E-10 : f32
    %cst_53 = arith.constant 5.80778412E-8 : f32
    %cst_54 = arith.constant 1.80033111E-9 : f32
    %cst_55 = arith.constant -5.45007333E-6 : f32
    %cst_56 = arith.constant -7.34322065E-6 : f32
    %cst_57 = arith.constant 3.017032E-4 : f32
    %cst_58 = arith.constant 7.73954438E-4 : f32
    %cst_59 = arith.constant -0.00728346175 : f32
    %cst_60 = arith.constant -0.026668204 : f32
    %cst_61 = arith.constant -5.52007818 : f32
    %cst_62 = arith.constant 7.19341457E-8 : f32
    %cst_63 = arith.constant -8.65372753 : f32
    %cst_64 = arith.constant -3.81477918E-7 : f32
    %cst_65 = arith.constant 3.35928798 : f32
    %cst_66 = arith.constant -0.514522672 : f32
    %cst_67 = arith.constant 0.10337057 : f32
    %cst_68 = arith.constant -0.0624997243 : f32
    %cst_69 = arith.constant 1.13964951 : f32
    %cst_70 = arith.constant -0.205326751 : f32
    %cst_71 = arith.constant 0.0650917366 : f32
    %cst_72 = arith.constant -0.124999993 : f32
    %cst_73 = arith.constant -0.785398185 : f32
    %cst_74 = arith.constant 0x7FFFFFFF : f32
    %c40_i32 = arith.constant 40 : i32
    %cst_75 = arith.constant -0.000000e+00 : f32
    %c2147483646_i32 = arith.constant 2147483646 : i32
    %cst_76 = arith.constant 2.000000e+00 : f32
    %cst_77 = arith.constant 9.99999986E+14 : f32
    %cst_78 = arith.constant 1.000000e-15 : f32
    %2 = llvm.alloca %c1_i32 x !llvm.array<7 x i32> {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %3 = arith.index_castui %arg0 : i32 to index
    %4 = scf.index_switch %3 -> f32
    case 0 {
      %5 = llvm.call_intrinsic "llvm.nvvm.fabs.f32"(%arg1) : (f32) -> f32
      %6 = arith.cmpf ugt, %5, %cst {fastmathFlags = #llvm.fastmath<none>} : f32
      %7 = scf.if %6 -> (f32) {
        %8 = llvm.call_intrinsic "llvm.nvvm.fabs.f32"(%5) : (f32) -> f32
        %9 = arith.cmpf oeq, %8, %cst_17 {fastmathFlags = #llvm.fastmath<none>} : f32
        %10 = scf.if %9 -> (f32) {
          scf.yield %cst_18 : f32
        } else {
          %11 = llvm.inline_asm tail_call_kind = <tail> asm_dialect = att "rcp.approx.ftz.f32 $0,$1;", "=f,f" %5 : (f32) -> f32
          %12 = arith.mulf %11, %11 {fastmathFlags = #llvm.fastmath<none>} : f32
          %13 = math.fma %12, %cst_65, %cst_66 : f32
          %14 = math.fma %13, %12, %cst_67 : f32
          %15 = math.fma %14, %12, %cst_68 : f32
          %16 = math.fma %15, %12, %cst_23 : f32
          %17 = math.fma %12, %cst_69, %cst_70 : f32
          %18 = math.fma %17, %12, %cst_71 : f32
          %19 = math.fma %18, %12, %cst_72 : f32
          %20 = math.fma %19, %11, %5 : f32
          %21 = llvm.call_intrinsic "llvm.nvvm.rsqrt.approx.f"(%5) : (f32) -> f32
          %22 = arith.mulf %21, %cst_28 {fastmathFlags = #llvm.fastmath<none>} : f32
          %23 = arith.mulf %22, %16 {fastmathFlags = #llvm.fastmath<none>} : f32
          %24 = arith.mulf %20, %cst_29 {fastmathFlags = #llvm.fastmath<none>} : f32
          %25 = llvm.call_intrinsic "llvm.nvvm.f2i.rn"(%24) : (f32) -> i32
          %26 = arith.sitofp %25 : i32 to f32
          %27 = math.fma %26, %cst_30, %20 : f32
          %28 = math.fma %26, %cst_31, %27 : f32
          %29 = math.fma %26, %cst_32, %28 : f32
          %30 = llvm.call_intrinsic "llvm.nvvm.fabs.f32"(%20) : (f32) -> f32
          %31 = arith.cmpf ogt, %30, %cst_33 {fastmathFlags = #llvm.fastmath<none>} : f32
          %32:2 = scf.if %31 -> (i32, f32) {
            %60 = arith.cmpf oeq, %30, %cst_17 {fastmathFlags = #llvm.fastmath<none>} : f32
            %61:2 = scf.if %60 -> (i32, f32) {
              %62 = llvm.call_intrinsic "llvm.nvvm.mul.rn.f"(%20, %cst_18) : (f32, f32 {llvm.noundef}) -> f32
              scf.yield %25, %62 : i32, f32
            } else {
              %62 = arith.bitcast %20 : f32 to i32
              %63 = arith.shli %62, %c8_i32 : i32
              %64 = arith.ori %63, %c-2147483648_i32 : i32
              %65:2 = scf.for %arg2 = %c0_i32 to %c6_i32 step %c1_i32 iter_args(%arg3 = %c0_i32, %arg4 = %0) -> (i32, i32)  : i32 {
                %110 = arith.extui %arg2 {nonNeg} : i32 to i64
                %111 = llvm.getelementptr inbounds|nuw %1[0, %110] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, !llvm.array<6 x i32>
                %112 = llvm.load %111 {alignment = 4 : i64} : !llvm.ptr<1> -> i32
                %113 = llvm.inline_asm tail_call_kind = <tail> asm_dialect = att "{\0A\09mad.lo.cc.u32   $0, $2, $3, $4;\0A\09madc.hi.u32     $1, $2, $3,  0;\0A\09}", "=r,=r,r,r,r" %112, %64, %arg3 : (i32, i32, i32) -> !llvm.struct<(i32, i32)>
                %114 = llvm.extractvalue %113[0] : !llvm.struct<(i32, i32)>
                %115 = llvm.extractvalue %113[1] : !llvm.struct<(i32, i32)>
                %116 = llvm.getelementptr inbounds|nuw %2[0, %110] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<7 x i32>
                llvm.store %114, %116 {alignment = 4 : i64} : i32, !llvm.ptr
                scf.yield %115, %115 : i32, i32
              }
              %66 = arith.shrui %62, %c23_i32 : i32
              %67 = arith.andi %66, %c224_i32 : i32
              %68 = arith.addi %67, %c-128_i32 : i32
              %69 = arith.shrui %68, %c5_i32 exact : i32
              %70 = llvm.getelementptr inbounds|nuw %2[24] : (!llvm.ptr) -> !llvm.ptr, i8
              llvm.store %65#1, %70 {alignment = 4 : i64} : i32, !llvm.ptr
              %71 = arith.subi %c6_i32, %69 : i32
              %72 = arith.extsi %71 : i32 to i64
              %73 = llvm.getelementptr inbounds %2[0, %72] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<7 x i32>
              %74 = llvm.load %73 {alignment = 4 : i64} : !llvm.ptr -> i32
              %75 = arith.subi %c5_i32, %69 : i32
              %76 = arith.extsi %75 : i32 to i64
              %77 = llvm.getelementptr inbounds %2[0, %76] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<7 x i32>
              %78 = llvm.load %77 {alignment = 4 : i64} : !llvm.ptr -> i32
              %79 = arith.andi %62, %c260046848_i32 : i32
              %80 = arith.cmpi eq, %79, %c0_i32 : i32
              %81 = scf.if %80 -> (i32) {
                scf.yield %78 : i32
              } else {
                %110 = arith.subi %c4_i32, %69 : i32
                %111 = arith.extsi %110 : i32 to i64
                %112 = llvm.getelementptr inbounds %2[0, %111] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<7 x i32>
                %113 = llvm.load %112 {alignment = 4 : i64} : !llvm.ptr -> i32
                %114 = llvm.intr.fshl(%78, %113, %66) : (i32, i32, i32) -> i32
                scf.yield %114 : i32
              }
              %82 = llvm.intr.fshl(%74, %78, %66) : (i32, i32, i32) -> i32
              %83 = arith.shrui %82, %c30_i32 : i32
              %84 = llvm.intr.fshl(%82, %81, %c2_i32) : (i32, i32, i32) -> i32
              %85 = arith.shli %81, %c2_i32 : i32
              %86 = arith.shrui %84, %c31_i32 : i32
              %87 = arith.addi %86, %83 : i32
              %88 = arith.subi %c0_i32, %87 : i32
              %89 = arith.cmpi slt, %62, %c0_i32 : i32
              %90 = arith.select %89, %88, %87 {fastmathFlags = #llvm.fastmath<none>} : i32
              %91 = arith.cmpi sgt, %84, %c-1_i32 : i32
              %92 = arith.xori %84, %c-1_i32 : i32
              %93 = arith.subi %c0_i32, %85 : i32
              %94 = arith.cmpi eq, %85, %c0_i32 : i32
              %95 = arith.extui %94 : i1 to i32
              %96 = arith.addi %95, %92 : i32
              %97 = arith.xori %84, %62 : i32
              %98 = arith.select %91, %84, %96 {fastmathFlags = #llvm.fastmath<none>} : i32
              %99 = arith.select %91, %85, %93 {fastmathFlags = #llvm.fastmath<none>} : i32
              %100 = arith.extui %98 : i32 to i64
              %101 = arith.shli %100, %c32_i64 : i64
              %102 = arith.extui %99 : i32 to i64
              %103 = arith.ori %101, %102 {isDisjoint} : i64
              %104 = arith.sitofp %103 : i64 to f64
              %105 = arith.mulf %104, %cst_34 {fastmathFlags = #llvm.fastmath<none>} : f64
              %106 = arith.truncf %105 : f64 to f32
              %107 = arith.negf %106 {fastmathFlags = #llvm.fastmath<none>} : f32
              %108 = arith.cmpi slt, %97, %c0_i32 : i32
              %109 = arith.select %108, %107, %106 {fastmathFlags = #llvm.fastmath<none>} : f32
              scf.yield %90, %109 : i32, f32
            }
            scf.yield %61#0, %61#1 : i32, f32
          } else {
            scf.yield %25, %29 : i32, f32
          }
          %33 = arith.andi %32#0, %c3_i32 : i32
          %34 = arith.uitofp %33 {nonNeg} : i32 to f32
          %35 = math.fma %34, %cst_35, %cst_73 : f32
          %36 = arith.addf %32#1, %35 {fastmathFlags = #llvm.fastmath<none>} : f32
          %37 = arith.mulf %36, %cst_29 {fastmathFlags = #llvm.fastmath<none>} : f32
          %38 = llvm.call_intrinsic "llvm.nvvm.f2i.rn"(%37) : (f32) -> i32
          %39 = arith.sitofp %38 : i32 to f32
          %40 = math.fma %39, %cst_30, %36 : f32
          %41 = math.fma %39, %cst_31, %40 : f32
          %42 = arith.addi %38, %c1_i32 : i32
          %43 = llvm.call_intrinsic "llvm.nvvm.mul.rn.f"(%41, %41) : (f32, f32) -> f32
          %44 = arith.andi %38, %c1_i32 : i32
          %45 = arith.cmpi eq, %44, %c0_i32 : i32
          %46 = arith.select %45, %cst_23, %41 {fastmathFlags = #llvm.fastmath<none>} : f32
          %47 = math.fma %43, %46, %cst_18 : f32
          %48 = math.fma %43, %cst_37, %cst_38 : f32
          %49 = arith.select %45, %48, %cst_39 {fastmathFlags = #llvm.fastmath<none>} : f32
          %50 = arith.select %45, %cst_40, %cst_41 {fastmathFlags = #llvm.fastmath<none>} : f32
          %51 = math.fma %49, %43, %50 : f32
          %52 = arith.select %45, %cst_42, %cst_43 {fastmathFlags = #llvm.fastmath<none>} : f32
          %53 = math.fma %51, %43, %52 : f32
          %54 = math.fma %53, %47, %46 : f32
          %55 = arith.andi %42, %c2_i32 : i32
          %56 = arith.cmpi eq, %55, %c0_i32 : i32
          %57 = arith.subf %cst_18, %54 {fastmathFlags = #llvm.fastmath<none>} : f32
          %58 = arith.select %56, %54, %57 {fastmathFlags = #llvm.fastmath<none>} : f32
          %59 = arith.mulf %23, %58 {fastmathFlags = #llvm.fastmath<none>} : f32
          scf.yield %59 : f32
        }
        scf.yield %10 : f32
      } else {
        %8 = arith.addf %5, %cst_45 {fastmathFlags = #llvm.fastmath<none>} : f32
        %9 = arith.addf %8, %cst_46 {fastmathFlags = #llvm.fastmath<none>} : f32
        %10 = math.fma %9, %cst_47, %cst_48 : f32
        %11 = math.fma %10, %9, %cst_49 : f32
        %12 = math.fma %11, %9, %cst_50 : f32
        %13 = math.fma %12, %9, %cst_51 : f32
        %14 = math.fma %13, %9, %cst_52 : f32
        %15 = math.fma %14, %9, %cst_53 : f32
        %16 = math.fma %15, %9, %cst_54 : f32
        %17 = math.fma %16, %9, %cst_55 : f32
        %18 = math.fma %17, %9, %cst_56 : f32
        %19 = math.fma %18, %9, %cst_57 : f32
        %20 = math.fma %19, %9, %cst_58 : f32
        %21 = math.fma %20, %9, %cst_59 : f32
        %22 = math.fma %21, %9, %cst_60 : f32
        %23 = arith.mulf %9, %22 {fastmathFlags = #llvm.fastmath<none>} : f32
        %24 = arith.addf %5, %cst_61 {fastmathFlags = #llvm.fastmath<none>} : f32
        %25 = arith.addf %24, %cst_62 {fastmathFlags = #llvm.fastmath<none>} : f32
        %26 = arith.mulf %25, %23 {fastmathFlags = #llvm.fastmath<none>} : f32
        %27 = arith.addf %5, %cst_63 {fastmathFlags = #llvm.fastmath<none>} : f32
        %28 = arith.addf %27, %cst_64 {fastmathFlags = #llvm.fastmath<none>} : f32
        %29 = arith.mulf %28, %26 {fastmathFlags = #llvm.fastmath<none>} : f32
        scf.yield %29 : f32
      }
      scf.yield %7 : f32
    }
    case 1 {
      %5 = llvm.call_intrinsic "llvm.nvvm.fabs.f32"(%arg1) : (f32) -> f32
      %6 = arith.cmpf ugt, %5, %cst {fastmathFlags = #llvm.fastmath<none>} : f32
      %7 = scf.if %6 -> (f32) {
        %14 = llvm.call_intrinsic "llvm.nvvm.fabs.f32"(%5) : (f32) -> f32
        %15 = arith.cmpf oeq, %14, %cst_17 {fastmathFlags = #llvm.fastmath<none>} : f32
        %16 = scf.if %15 -> (f32) {
          scf.yield %cst_18 : f32
        } else {
          %17 = llvm.inline_asm tail_call_kind = <tail> asm_dialect = att "rcp.approx.ftz.f32 $0,$1;", "=f,f" %5 : (f32) -> f32
          %18 = arith.mulf %17, %17 {fastmathFlags = #llvm.fastmath<none>} : f32
          %19 = math.fma %18, %cst_19, %cst_20 : f32
          %20 = math.fma %19, %18, %cst_21 : f32
          %21 = math.fma %20, %18, %cst_22 : f32
          %22 = math.fma %21, %18, %cst_23 : f32
          %23 = math.fma %18, %cst_24, %cst_25 : f32
          %24 = math.fma %23, %18, %cst_26 : f32
          %25 = math.fma %24, %18, %cst_27 : f32
          %26 = math.fma %25, %17, %5 : f32
          %27 = llvm.call_intrinsic "llvm.nvvm.rsqrt.approx.f"(%5) : (f32) -> f32
          %28 = arith.mulf %27, %cst_28 {fastmathFlags = #llvm.fastmath<none>} : f32
          %29 = arith.mulf %28, %22 {fastmathFlags = #llvm.fastmath<none>} : f32
          %30 = arith.mulf %26, %cst_29 {fastmathFlags = #llvm.fastmath<none>} : f32
          %31 = llvm.call_intrinsic "llvm.nvvm.f2i.rn"(%30) : (f32) -> i32
          %32 = arith.sitofp %31 : i32 to f32
          %33 = math.fma %32, %cst_30, %26 : f32
          %34 = math.fma %32, %cst_31, %33 : f32
          %35 = math.fma %32, %cst_32, %34 : f32
          %36 = llvm.call_intrinsic "llvm.nvvm.fabs.f32"(%26) : (f32) -> f32
          %37 = arith.cmpf ogt, %36, %cst_33 {fastmathFlags = #llvm.fastmath<none>} : f32
          %38:2 = scf.if %37 -> (i32, f32) {
            %66 = arith.cmpf oeq, %36, %cst_17 {fastmathFlags = #llvm.fastmath<none>} : f32
            %67:2 = scf.if %66 -> (i32, f32) {
              %68 = llvm.call_intrinsic "llvm.nvvm.mul.rn.f"(%26, %cst_18) : (f32, f32 {llvm.noundef}) -> f32
              scf.yield %31, %68 : i32, f32
            } else {
              %68 = arith.bitcast %26 : f32 to i32
              %69 = arith.shli %68, %c8_i32 : i32
              %70 = arith.ori %69, %c-2147483648_i32 : i32
              %71:2 = scf.for %arg2 = %c0_i32 to %c6_i32 step %c1_i32 iter_args(%arg3 = %c0_i32, %arg4 = %0) -> (i32, i32)  : i32 {
                %116 = arith.extui %arg2 {nonNeg} : i32 to i64
                %117 = llvm.getelementptr inbounds|nuw %1[0, %116] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, !llvm.array<6 x i32>
                %118 = llvm.load %117 {alignment = 4 : i64} : !llvm.ptr<1> -> i32
                %119 = llvm.inline_asm tail_call_kind = <tail> asm_dialect = att "{\0A\09mad.lo.cc.u32   $0, $2, $3, $4;\0A\09madc.hi.u32     $1, $2, $3,  0;\0A\09}", "=r,=r,r,r,r" %118, %70, %arg3 : (i32, i32, i32) -> !llvm.struct<(i32, i32)>
                %120 = llvm.extractvalue %119[0] : !llvm.struct<(i32, i32)>
                %121 = llvm.extractvalue %119[1] : !llvm.struct<(i32, i32)>
                %122 = llvm.getelementptr inbounds|nuw %2[0, %116] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<7 x i32>
                llvm.store %120, %122 {alignment = 4 : i64} : i32, !llvm.ptr
                scf.yield %121, %121 : i32, i32
              }
              %72 = arith.shrui %68, %c23_i32 : i32
              %73 = arith.andi %72, %c224_i32 : i32
              %74 = arith.addi %73, %c-128_i32 : i32
              %75 = arith.shrui %74, %c5_i32 exact : i32
              %76 = llvm.getelementptr inbounds|nuw %2[24] : (!llvm.ptr) -> !llvm.ptr, i8
              llvm.store %71#1, %76 {alignment = 4 : i64} : i32, !llvm.ptr
              %77 = arith.subi %c6_i32, %75 : i32
              %78 = arith.extsi %77 : i32 to i64
              %79 = llvm.getelementptr inbounds %2[0, %78] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<7 x i32>
              %80 = llvm.load %79 {alignment = 4 : i64} : !llvm.ptr -> i32
              %81 = arith.subi %c5_i32, %75 : i32
              %82 = arith.extsi %81 : i32 to i64
              %83 = llvm.getelementptr inbounds %2[0, %82] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<7 x i32>
              %84 = llvm.load %83 {alignment = 4 : i64} : !llvm.ptr -> i32
              %85 = arith.andi %68, %c260046848_i32 : i32
              %86 = arith.cmpi eq, %85, %c0_i32 : i32
              %87 = scf.if %86 -> (i32) {
                scf.yield %84 : i32
              } else {
                %116 = arith.subi %c4_i32, %75 : i32
                %117 = arith.extsi %116 : i32 to i64
                %118 = llvm.getelementptr inbounds %2[0, %117] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<7 x i32>
                %119 = llvm.load %118 {alignment = 4 : i64} : !llvm.ptr -> i32
                %120 = llvm.intr.fshl(%84, %119, %72) : (i32, i32, i32) -> i32
                scf.yield %120 : i32
              }
              %88 = llvm.intr.fshl(%80, %84, %72) : (i32, i32, i32) -> i32
              %89 = arith.shrui %88, %c30_i32 : i32
              %90 = llvm.intr.fshl(%88, %87, %c2_i32) : (i32, i32, i32) -> i32
              %91 = arith.shli %87, %c2_i32 : i32
              %92 = arith.shrui %90, %c31_i32 : i32
              %93 = arith.addi %92, %89 : i32
              %94 = arith.subi %c0_i32, %93 : i32
              %95 = arith.cmpi slt, %68, %c0_i32 : i32
              %96 = arith.select %95, %94, %93 {fastmathFlags = #llvm.fastmath<none>} : i32
              %97 = arith.cmpi sgt, %90, %c-1_i32 : i32
              %98 = arith.xori %90, %c-1_i32 : i32
              %99 = arith.subi %c0_i32, %91 : i32
              %100 = arith.cmpi eq, %91, %c0_i32 : i32
              %101 = arith.extui %100 : i1 to i32
              %102 = arith.addi %101, %98 : i32
              %103 = arith.xori %90, %68 : i32
              %104 = arith.select %97, %90, %102 {fastmathFlags = #llvm.fastmath<none>} : i32
              %105 = arith.select %97, %91, %99 {fastmathFlags = #llvm.fastmath<none>} : i32
              %106 = arith.extui %104 : i32 to i64
              %107 = arith.shli %106, %c32_i64 : i64
              %108 = arith.extui %105 : i32 to i64
              %109 = arith.ori %107, %108 {isDisjoint} : i64
              %110 = arith.sitofp %109 : i64 to f64
              %111 = arith.mulf %110, %cst_34 {fastmathFlags = #llvm.fastmath<none>} : f64
              %112 = arith.truncf %111 : f64 to f32
              %113 = arith.negf %112 {fastmathFlags = #llvm.fastmath<none>} : f32
              %114 = arith.cmpi slt, %103, %c0_i32 : i32
              %115 = arith.select %114, %113, %112 {fastmathFlags = #llvm.fastmath<none>} : f32
              scf.yield %96, %115 : i32, f32
            }
            scf.yield %67#0, %67#1 : i32, f32
          } else {
            scf.yield %31, %35 : i32, f32
          }
          %39 = arith.andi %38#0, %c3_i32 : i32
          %40 = arith.uitofp %39 {nonNeg} : i32 to f32
          %41 = math.fma %40, %cst_35, %cst_36 : f32
          %42 = arith.addf %38#1, %41 {fastmathFlags = #llvm.fastmath<none>} : f32
          %43 = arith.mulf %42, %cst_29 {fastmathFlags = #llvm.fastmath<none>} : f32
          %44 = llvm.call_intrinsic "llvm.nvvm.f2i.rn"(%43) : (f32) -> i32
          %45 = arith.sitofp %44 : i32 to f32
          %46 = math.fma %45, %cst_30, %42 : f32
          %47 = math.fma %45, %cst_31, %46 : f32
          %48 = arith.addi %44, %c1_i32 : i32
          %49 = llvm.call_intrinsic "llvm.nvvm.mul.rn.f"(%47, %47) : (f32, f32) -> f32
          %50 = arith.andi %44, %c1_i32 : i32
          %51 = arith.cmpi eq, %50, %c0_i32 : i32
          %52 = arith.select %51, %cst_23, %47 {fastmathFlags = #llvm.fastmath<none>} : f32
          %53 = math.fma %49, %52, %cst_18 : f32
          %54 = math.fma %49, %cst_37, %cst_38 : f32
          %55 = arith.select %51, %54, %cst_39 {fastmathFlags = #llvm.fastmath<none>} : f32
          %56 = arith.select %51, %cst_40, %cst_41 {fastmathFlags = #llvm.fastmath<none>} : f32
          %57 = math.fma %55, %49, %56 : f32
          %58 = arith.select %51, %cst_42, %cst_43 {fastmathFlags = #llvm.fastmath<none>} : f32
          %59 = math.fma %57, %49, %58 : f32
          %60 = math.fma %59, %53, %52 : f32
          %61 = arith.andi %48, %c2_i32 : i32
          %62 = arith.cmpi eq, %61, %c0_i32 : i32
          %63 = arith.subf %cst_18, %60 {fastmathFlags = #llvm.fastmath<none>} : f32
          %64 = arith.select %62, %60, %63 {fastmathFlags = #llvm.fastmath<none>} : f32
          %65 = arith.mulf %29, %64 {fastmathFlags = #llvm.fastmath<none>} : f32
          scf.yield %65 : f32
        }
        scf.yield %16 : f32
      } else {
        %14 = arith.addf %5, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f32
        %15 = arith.addf %14, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
        %16 = math.fma %15, %cst_2, %cst_3 : f32
        %17 = math.fma %16, %15, %cst_4 : f32
        %18 = math.fma %17, %15, %cst_5 : f32
        %19 = math.fma %18, %15, %cst_6 : f32
        %20 = math.fma %19, %15, %cst_7 : f32
        %21 = math.fma %20, %15, %cst_8 : f32
        %22 = math.fma %21, %15, %cst_9 : f32
        %23 = math.fma %22, %15, %cst_10 : f32
        %24 = math.fma %23, %15, %cst_11 : f32
        %25 = math.fma %24, %15, %cst_12 : f32
        %26 = math.fma %25, %15, %cst_13 : f32
        %27 = math.fma %26, %15, %cst_14 : f32
        %28 = arith.addf %5, %cst_15 {fastmathFlags = #llvm.fastmath<none>} : f32
        %29 = arith.addf %28, %cst_16 {fastmathFlags = #llvm.fastmath<none>} : f32
        %30 = arith.mulf %29, %27 {fastmathFlags = #llvm.fastmath<none>} : f32
        %31 = arith.mulf %15, %30 {fastmathFlags = #llvm.fastmath<none>} : f32
        %32 = arith.mulf %5, %31 {fastmathFlags = #llvm.fastmath<none>} : f32
        scf.yield %32 : f32
      }
      %8 = arith.cmpf olt, %arg1, %cst_18 {fastmathFlags = #llvm.fastmath<none>} : f32
      %9 = arith.negf %7 {fastmathFlags = #llvm.fastmath<none>} : f32
      %10 = arith.select %8, %9, %7 {fastmathFlags = #llvm.fastmath<none>} : f32
      %11 = arith.cmpf olt, %5, %cst_44 {fastmathFlags = #llvm.fastmath<none>} : f32
      %12 = math.copysign %7, %arg1 : f32
      %13 = arith.select %11, %12, %10 : f32
      scf.yield %13 : f32
    }
    default {
      %5 = arith.cmpi slt, %arg0, %c0_i32 : i32
      %6 = scf.if %5 -> (f32) {
        scf.yield %cst_74 : f32
      } else {
        %7 = llvm.call_intrinsic "llvm.nvvm.fabs.f32"(%arg1) : (f32) -> f32
        %8 = arith.cmpi ugt, %arg0, %c4_i32 : i32
        %9 = arith.extui %8 : i1 to i32
        %10 = arith.addi %arg0, %9 : i32
        %11 = arith.uitofp %10 {nonNeg} : i32 to f32
        %12 = arith.cmpf ogt, %7, %11 {fastmathFlags = #llvm.fastmath<none>} : f32
        %13 = scf.if %12 -> (f32) {
          %14 = arith.divf %cst_76, %arg1 {fastmathFlags = #llvm.fastmath<none>} : f32
          %15 = arith.cmpf ole, %7, %cst {fastmathFlags = #llvm.fastmath<none>} : f32
          %16 = scf.if %15 -> (f32) {
            %25 = arith.addf %7, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f32
            %26 = arith.addf %25, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
            %27 = math.fma %26, %cst_2, %cst_3 : f32
            %28 = math.fma %27, %26, %cst_4 : f32
            %29 = math.fma %28, %26, %cst_5 : f32
            %30 = math.fma %29, %26, %cst_6 : f32
            %31 = math.fma %30, %26, %cst_7 : f32
            %32 = math.fma %31, %26, %cst_8 : f32
            %33 = math.fma %32, %26, %cst_9 : f32
            %34 = math.fma %33, %26, %cst_10 : f32
            %35 = math.fma %34, %26, %cst_11 : f32
            %36 = math.fma %35, %26, %cst_12 : f32
            %37 = math.fma %36, %26, %cst_13 : f32
            %38 = math.fma %37, %26, %cst_14 : f32
            %39 = arith.addf %7, %cst_15 {fastmathFlags = #llvm.fastmath<none>} : f32
            %40 = arith.addf %39, %cst_16 {fastmathFlags = #llvm.fastmath<none>} : f32
            %41 = arith.mulf %40, %38 {fastmathFlags = #llvm.fastmath<none>} : f32
            %42 = arith.mulf %26, %41 {fastmathFlags = #llvm.fastmath<none>} : f32
            %43 = arith.mulf %7, %42 {fastmathFlags = #llvm.fastmath<none>} : f32
            scf.yield %43 : f32
          } else {
            %25 = llvm.call_intrinsic "llvm.nvvm.fabs.f32"(%7) : (f32) -> f32
            %26 = arith.cmpf oeq, %25, %cst_17 {fastmathFlags = #llvm.fastmath<none>} : f32
            %27 = scf.if %26 -> (f32) {
              scf.yield %cst_18 : f32
            } else {
              %28 = llvm.inline_asm tail_call_kind = <tail> asm_dialect = att "rcp.approx.ftz.f32 $0,$1;", "=f,f" %7 : (f32) -> f32
              %29 = arith.mulf %28, %28 {fastmathFlags = #llvm.fastmath<none>} : f32
              %30 = math.fma %29, %cst_19, %cst_20 : f32
              %31 = math.fma %30, %29, %cst_21 : f32
              %32 = math.fma %31, %29, %cst_22 : f32
              %33 = math.fma %32, %29, %cst_23 : f32
              %34 = math.fma %29, %cst_24, %cst_25 : f32
              %35 = math.fma %34, %29, %cst_26 : f32
              %36 = math.fma %35, %29, %cst_27 : f32
              %37 = math.fma %36, %28, %7 : f32
              %38 = llvm.call_intrinsic "llvm.nvvm.rsqrt.approx.f"(%7) : (f32) -> f32
              %39 = arith.mulf %38, %cst_28 {fastmathFlags = #llvm.fastmath<none>} : f32
              %40 = arith.mulf %39, %33 {fastmathFlags = #llvm.fastmath<none>} : f32
              %41 = arith.mulf %37, %cst_29 {fastmathFlags = #llvm.fastmath<none>} : f32
              %42 = llvm.call_intrinsic "llvm.nvvm.f2i.rn"(%41) : (f32) -> i32
              %43 = arith.sitofp %42 : i32 to f32
              %44 = math.fma %43, %cst_30, %37 : f32
              %45 = math.fma %43, %cst_31, %44 : f32
              %46 = math.fma %43, %cst_32, %45 : f32
              %47 = llvm.call_intrinsic "llvm.nvvm.fabs.f32"(%37) : (f32) -> f32
              %48 = arith.cmpf ogt, %47, %cst_33 {fastmathFlags = #llvm.fastmath<none>} : f32
              %49:2 = scf.if %48 -> (i32, f32) {
                %77 = arith.cmpf oeq, %47, %cst_17 {fastmathFlags = #llvm.fastmath<none>} : f32
                %78:2 = scf.if %77 -> (i32, f32) {
                  %79 = llvm.call_intrinsic "llvm.nvvm.mul.rn.f"(%37, %cst_18) : (f32, f32 {llvm.noundef}) -> f32
                  scf.yield %42, %79 : i32, f32
                } else {
                  %79 = arith.bitcast %37 : f32 to i32
                  %80 = arith.shli %79, %c8_i32 : i32
                  %81 = arith.ori %80, %c-2147483648_i32 : i32
                  %82:2 = scf.for %arg2 = %c0_i32 to %c6_i32 step %c1_i32 iter_args(%arg3 = %c0_i32, %arg4 = %0) -> (i32, i32)  : i32 {
                    %127 = arith.extui %arg2 {nonNeg} : i32 to i64
                    %128 = llvm.getelementptr inbounds|nuw %1[0, %127] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, !llvm.array<6 x i32>
                    %129 = llvm.load %128 {alignment = 4 : i64} : !llvm.ptr<1> -> i32
                    %130 = llvm.inline_asm tail_call_kind = <tail> asm_dialect = att "{\0A\09mad.lo.cc.u32   $0, $2, $3, $4;\0A\09madc.hi.u32     $1, $2, $3,  0;\0A\09}", "=r,=r,r,r,r" %129, %81, %arg3 : (i32, i32, i32) -> !llvm.struct<(i32, i32)>
                    %131 = llvm.extractvalue %130[0] : !llvm.struct<(i32, i32)>
                    %132 = llvm.extractvalue %130[1] : !llvm.struct<(i32, i32)>
                    %133 = llvm.getelementptr inbounds|nuw %2[0, %127] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<7 x i32>
                    llvm.store %131, %133 {alignment = 4 : i64} : i32, !llvm.ptr
                    scf.yield %132, %132 : i32, i32
                  }
                  %83 = arith.shrui %79, %c23_i32 : i32
                  %84 = arith.andi %83, %c224_i32 : i32
                  %85 = arith.addi %84, %c-128_i32 : i32
                  %86 = arith.shrui %85, %c5_i32 exact : i32
                  %87 = llvm.getelementptr inbounds|nuw %2[24] : (!llvm.ptr) -> !llvm.ptr, i8
                  llvm.store %82#1, %87 {alignment = 4 : i64} : i32, !llvm.ptr
                  %88 = arith.subi %c6_i32, %86 : i32
                  %89 = arith.extsi %88 : i32 to i64
                  %90 = llvm.getelementptr inbounds %2[0, %89] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<7 x i32>
                  %91 = llvm.load %90 {alignment = 4 : i64} : !llvm.ptr -> i32
                  %92 = arith.subi %c5_i32, %86 : i32
                  %93 = arith.extsi %92 : i32 to i64
                  %94 = llvm.getelementptr inbounds %2[0, %93] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<7 x i32>
                  %95 = llvm.load %94 {alignment = 4 : i64} : !llvm.ptr -> i32
                  %96 = arith.andi %79, %c260046848_i32 : i32
                  %97 = arith.cmpi eq, %96, %c0_i32 : i32
                  %98 = scf.if %97 -> (i32) {
                    scf.yield %95 : i32
                  } else {
                    %127 = arith.subi %c4_i32, %86 : i32
                    %128 = arith.extsi %127 : i32 to i64
                    %129 = llvm.getelementptr inbounds %2[0, %128] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<7 x i32>
                    %130 = llvm.load %129 {alignment = 4 : i64} : !llvm.ptr -> i32
                    %131 = llvm.intr.fshl(%95, %130, %83) : (i32, i32, i32) -> i32
                    scf.yield %131 : i32
                  }
                  %99 = llvm.intr.fshl(%91, %95, %83) : (i32, i32, i32) -> i32
                  %100 = arith.shrui %99, %c30_i32 : i32
                  %101 = llvm.intr.fshl(%99, %98, %c2_i32) : (i32, i32, i32) -> i32
                  %102 = arith.shli %98, %c2_i32 : i32
                  %103 = arith.shrui %101, %c31_i32 : i32
                  %104 = arith.addi %103, %100 : i32
                  %105 = arith.subi %c0_i32, %104 : i32
                  %106 = arith.cmpi slt, %79, %c0_i32 : i32
                  %107 = arith.select %106, %105, %104 {fastmathFlags = #llvm.fastmath<none>} : i32
                  %108 = arith.cmpi sgt, %101, %c-1_i32 : i32
                  %109 = arith.xori %101, %c-1_i32 : i32
                  %110 = arith.subi %c0_i32, %102 : i32
                  %111 = arith.cmpi eq, %102, %c0_i32 : i32
                  %112 = arith.extui %111 : i1 to i32
                  %113 = arith.addi %112, %109 : i32
                  %114 = arith.xori %101, %79 : i32
                  %115 = arith.select %108, %101, %113 {fastmathFlags = #llvm.fastmath<none>} : i32
                  %116 = arith.select %108, %102, %110 {fastmathFlags = #llvm.fastmath<none>} : i32
                  %117 = arith.extui %115 : i32 to i64
                  %118 = arith.shli %117, %c32_i64 : i64
                  %119 = arith.extui %116 : i32 to i64
                  %120 = arith.ori %118, %119 {isDisjoint} : i64
                  %121 = arith.sitofp %120 : i64 to f64
                  %122 = arith.mulf %121, %cst_34 {fastmathFlags = #llvm.fastmath<none>} : f64
                  %123 = arith.truncf %122 : f64 to f32
                  %124 = arith.negf %123 {fastmathFlags = #llvm.fastmath<none>} : f32
                  %125 = arith.cmpi slt, %114, %c0_i32 : i32
                  %126 = arith.select %125, %124, %123 {fastmathFlags = #llvm.fastmath<none>} : f32
                  scf.yield %107, %126 : i32, f32
                }
                scf.yield %78#0, %78#1 : i32, f32
              } else {
                scf.yield %42, %46 : i32, f32
              }
              %50 = arith.andi %49#0, %c3_i32 : i32
              %51 = arith.uitofp %50 {nonNeg} : i32 to f32
              %52 = math.fma %51, %cst_35, %cst_36 : f32
              %53 = arith.addf %49#1, %52 {fastmathFlags = #llvm.fastmath<none>} : f32
              %54 = arith.mulf %53, %cst_29 {fastmathFlags = #llvm.fastmath<none>} : f32
              %55 = llvm.call_intrinsic "llvm.nvvm.f2i.rn"(%54) : (f32) -> i32
              %56 = arith.sitofp %55 : i32 to f32
              %57 = math.fma %56, %cst_30, %53 : f32
              %58 = math.fma %56, %cst_31, %57 : f32
              %59 = arith.addi %55, %c1_i32 : i32
              %60 = llvm.call_intrinsic "llvm.nvvm.mul.rn.f"(%58, %58) : (f32, f32) -> f32
              %61 = arith.andi %55, %c1_i32 : i32
              %62 = arith.cmpi eq, %61, %c0_i32 : i32
              %63 = arith.select %62, %cst_23, %58 {fastmathFlags = #llvm.fastmath<none>} : f32
              %64 = math.fma %60, %63, %cst_18 : f32
              %65 = math.fma %60, %cst_37, %cst_38 : f32
              %66 = arith.select %62, %65, %cst_39 {fastmathFlags = #llvm.fastmath<none>} : f32
              %67 = arith.select %62, %cst_40, %cst_41 {fastmathFlags = #llvm.fastmath<none>} : f32
              %68 = math.fma %66, %60, %67 : f32
              %69 = arith.select %62, %cst_42, %cst_43 {fastmathFlags = #llvm.fastmath<none>} : f32
              %70 = math.fma %68, %60, %69 : f32
              %71 = math.fma %70, %64, %63 : f32
              %72 = arith.andi %59, %c2_i32 : i32
              %73 = arith.cmpi eq, %72, %c0_i32 : i32
              %74 = arith.subf %cst_18, %71 {fastmathFlags = #llvm.fastmath<none>} : f32
              %75 = arith.select %73, %71, %74 {fastmathFlags = #llvm.fastmath<none>} : f32
              %76 = arith.mulf %40, %75 {fastmathFlags = #llvm.fastmath<none>} : f32
              scf.yield %76 : f32
            }
            scf.yield %27 : f32
          }
          %17 = arith.cmpf olt, %arg1, %cst_18 {fastmathFlags = #llvm.fastmath<none>} : f32
          %18 = arith.negf %16 {fastmathFlags = #llvm.fastmath<none>} : f32
          %19 = arith.select %17, %18, %16 {fastmathFlags = #llvm.fastmath<none>} : f32
          %20 = arith.cmpf olt, %7, %cst_44 {fastmathFlags = #llvm.fastmath<none>} : f32
          %21 = math.copysign %16, %arg1 : f32
          %22 = arith.select %20, %21, %19 {fastmathFlags = #llvm.fastmath<none>} : f32
          %23 = scf.if %15 -> (f32) {
            %25 = arith.addf %7, %cst_45 {fastmathFlags = #llvm.fastmath<none>} : f32
            %26 = arith.addf %25, %cst_46 {fastmathFlags = #llvm.fastmath<none>} : f32
            %27 = math.fma %26, %cst_47, %cst_48 : f32
            %28 = math.fma %27, %26, %cst_49 : f32
            %29 = math.fma %28, %26, %cst_50 : f32
            %30 = math.fma %29, %26, %cst_51 : f32
            %31 = math.fma %30, %26, %cst_52 : f32
            %32 = math.fma %31, %26, %cst_53 : f32
            %33 = math.fma %32, %26, %cst_54 : f32
            %34 = math.fma %33, %26, %cst_55 : f32
            %35 = math.fma %34, %26, %cst_56 : f32
            %36 = math.fma %35, %26, %cst_57 : f32
            %37 = math.fma %36, %26, %cst_58 : f32
            %38 = math.fma %37, %26, %cst_59 : f32
            %39 = math.fma %38, %26, %cst_60 : f32
            %40 = arith.mulf %26, %39 {fastmathFlags = #llvm.fastmath<none>} : f32
            %41 = arith.addf %7, %cst_61 {fastmathFlags = #llvm.fastmath<none>} : f32
            %42 = arith.addf %41, %cst_62 {fastmathFlags = #llvm.fastmath<none>} : f32
            %43 = arith.mulf %42, %40 {fastmathFlags = #llvm.fastmath<none>} : f32
            %44 = arith.addf %7, %cst_63 {fastmathFlags = #llvm.fastmath<none>} : f32
            %45 = arith.addf %44, %cst_64 {fastmathFlags = #llvm.fastmath<none>} : f32
            %46 = arith.mulf %45, %43 {fastmathFlags = #llvm.fastmath<none>} : f32
            scf.yield %46 : f32
          } else {
            %25 = llvm.call_intrinsic "llvm.nvvm.fabs.f32"(%7) : (f32) -> f32
            %26 = arith.cmpf oeq, %25, %cst_17 {fastmathFlags = #llvm.fastmath<none>} : f32
            %27 = scf.if %26 -> (f32) {
              scf.yield %cst_18 : f32
            } else {
              %28 = llvm.inline_asm tail_call_kind = <tail> asm_dialect = att "rcp.approx.ftz.f32 $0,$1;", "=f,f" %7 : (f32) -> f32
              %29 = arith.mulf %28, %28 {fastmathFlags = #llvm.fastmath<none>} : f32
              %30 = math.fma %29, %cst_65, %cst_66 : f32
              %31 = math.fma %30, %29, %cst_67 : f32
              %32 = math.fma %31, %29, %cst_68 : f32
              %33 = math.fma %32, %29, %cst_23 : f32
              %34 = math.fma %29, %cst_69, %cst_70 : f32
              %35 = math.fma %34, %29, %cst_71 : f32
              %36 = math.fma %35, %29, %cst_72 : f32
              %37 = math.fma %36, %28, %7 : f32
              %38 = llvm.call_intrinsic "llvm.nvvm.rsqrt.approx.f"(%7) : (f32) -> f32
              %39 = arith.mulf %38, %cst_28 {fastmathFlags = #llvm.fastmath<none>} : f32
              %40 = arith.mulf %39, %33 {fastmathFlags = #llvm.fastmath<none>} : f32
              %41 = arith.mulf %37, %cst_29 {fastmathFlags = #llvm.fastmath<none>} : f32
              %42 = llvm.call_intrinsic "llvm.nvvm.f2i.rn"(%41) : (f32) -> i32
              %43 = arith.sitofp %42 : i32 to f32
              %44 = math.fma %43, %cst_30, %37 : f32
              %45 = math.fma %43, %cst_31, %44 : f32
              %46 = math.fma %43, %cst_32, %45 : f32
              %47 = llvm.call_intrinsic "llvm.nvvm.fabs.f32"(%37) : (f32) -> f32
              %48 = arith.cmpf ogt, %47, %cst_33 {fastmathFlags = #llvm.fastmath<none>} : f32
              %49:2 = scf.if %48 -> (i32, f32) {
                %77 = arith.cmpf oeq, %47, %cst_17 {fastmathFlags = #llvm.fastmath<none>} : f32
                %78:2 = scf.if %77 -> (i32, f32) {
                  %79 = llvm.call_intrinsic "llvm.nvvm.mul.rn.f"(%37, %cst_18) : (f32, f32 {llvm.noundef}) -> f32
                  scf.yield %42, %79 : i32, f32
                } else {
                  %79 = arith.bitcast %37 : f32 to i32
                  %80 = arith.shli %79, %c8_i32 : i32
                  %81 = arith.ori %80, %c-2147483648_i32 : i32
                  %82:2 = scf.for %arg2 = %c0_i32 to %c6_i32 step %c1_i32 iter_args(%arg3 = %c0_i32, %arg4 = %0) -> (i32, i32)  : i32 {
                    %127 = arith.extui %arg2 {nonNeg} : i32 to i64
                    %128 = llvm.getelementptr inbounds|nuw %1[0, %127] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, !llvm.array<6 x i32>
                    %129 = llvm.load %128 {alignment = 4 : i64} : !llvm.ptr<1> -> i32
                    %130 = llvm.inline_asm tail_call_kind = <tail> asm_dialect = att "{\0A\09mad.lo.cc.u32   $0, $2, $3, $4;\0A\09madc.hi.u32     $1, $2, $3,  0;\0A\09}", "=r,=r,r,r,r" %129, %81, %arg3 : (i32, i32, i32) -> !llvm.struct<(i32, i32)>
                    %131 = llvm.extractvalue %130[0] : !llvm.struct<(i32, i32)>
                    %132 = llvm.extractvalue %130[1] : !llvm.struct<(i32, i32)>
                    %133 = llvm.getelementptr inbounds|nuw %2[0, %127] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<7 x i32>
                    llvm.store %131, %133 {alignment = 4 : i64} : i32, !llvm.ptr
                    scf.yield %132, %132 : i32, i32
                  }
                  %83 = arith.shrui %79, %c23_i32 : i32
                  %84 = arith.andi %83, %c224_i32 : i32
                  %85 = arith.addi %84, %c-128_i32 : i32
                  %86 = arith.shrui %85, %c5_i32 exact : i32
                  %87 = llvm.getelementptr inbounds|nuw %2[24] : (!llvm.ptr) -> !llvm.ptr, i8
                  llvm.store %82#1, %87 {alignment = 4 : i64} : i32, !llvm.ptr
                  %88 = arith.subi %c6_i32, %86 : i32
                  %89 = arith.extsi %88 : i32 to i64
                  %90 = llvm.getelementptr inbounds %2[0, %89] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<7 x i32>
                  %91 = llvm.load %90 {alignment = 4 : i64} : !llvm.ptr -> i32
                  %92 = arith.subi %c5_i32, %86 : i32
                  %93 = arith.extsi %92 : i32 to i64
                  %94 = llvm.getelementptr inbounds %2[0, %93] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<7 x i32>
                  %95 = llvm.load %94 {alignment = 4 : i64} : !llvm.ptr -> i32
                  %96 = arith.andi %79, %c260046848_i32 : i32
                  %97 = arith.cmpi eq, %96, %c0_i32 : i32
                  %98 = scf.if %97 -> (i32) {
                    scf.yield %95 : i32
                  } else {
                    %127 = arith.subi %c4_i32, %86 : i32
                    %128 = arith.extsi %127 : i32 to i64
                    %129 = llvm.getelementptr inbounds %2[0, %128] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<7 x i32>
                    %130 = llvm.load %129 {alignment = 4 : i64} : !llvm.ptr -> i32
                    %131 = llvm.intr.fshl(%95, %130, %83) : (i32, i32, i32) -> i32
                    scf.yield %131 : i32
                  }
                  %99 = llvm.intr.fshl(%91, %95, %83) : (i32, i32, i32) -> i32
                  %100 = arith.shrui %99, %c30_i32 : i32
                  %101 = llvm.intr.fshl(%99, %98, %c2_i32) : (i32, i32, i32) -> i32
                  %102 = arith.shli %98, %c2_i32 : i32
                  %103 = arith.shrui %101, %c31_i32 : i32
                  %104 = arith.addi %103, %100 : i32
                  %105 = arith.subi %c0_i32, %104 : i32
                  %106 = arith.cmpi slt, %79, %c0_i32 : i32
                  %107 = arith.select %106, %105, %104 {fastmathFlags = #llvm.fastmath<none>} : i32
                  %108 = arith.cmpi sgt, %101, %c-1_i32 : i32
                  %109 = arith.xori %101, %c-1_i32 : i32
                  %110 = arith.subi %c0_i32, %102 : i32
                  %111 = arith.cmpi eq, %102, %c0_i32 : i32
                  %112 = arith.extui %111 : i1 to i32
                  %113 = arith.addi %112, %109 : i32
                  %114 = arith.select %108, %102, %110 {fastmathFlags = #llvm.fastmath<none>} : i32
                  %115 = arith.select %108, %101, %113 {fastmathFlags = #llvm.fastmath<none>} : i32
                  %116 = arith.xori %101, %79 : i32
                  %117 = arith.extui %115 : i32 to i64
                  %118 = arith.shli %117, %c32_i64 : i64
                  %119 = arith.extui %114 : i32 to i64
                  %120 = arith.ori %118, %119 {isDisjoint} : i64
                  %121 = arith.sitofp %120 : i64 to f64
                  %122 = arith.mulf %121, %cst_34 {fastmathFlags = #llvm.fastmath<none>} : f64
                  %123 = arith.truncf %122 : f64 to f32
                  %124 = arith.negf %123 {fastmathFlags = #llvm.fastmath<none>} : f32
                  %125 = arith.cmpi slt, %116, %c0_i32 : i32
                  %126 = arith.select %125, %124, %123 {fastmathFlags = #llvm.fastmath<none>} : f32
                  scf.yield %107, %126 : i32, f32
                }
                scf.yield %78#0, %78#1 : i32, f32
              } else {
                scf.yield %42, %46 : i32, f32
              }
              %50 = arith.andi %49#0, %c3_i32 : i32
              %51 = arith.uitofp %50 {nonNeg} : i32 to f32
              %52 = math.fma %51, %cst_35, %cst_73 : f32
              %53 = arith.addf %49#1, %52 {fastmathFlags = #llvm.fastmath<none>} : f32
              %54 = arith.mulf %53, %cst_29 {fastmathFlags = #llvm.fastmath<none>} : f32
              %55 = llvm.call_intrinsic "llvm.nvvm.f2i.rn"(%54) : (f32) -> i32
              %56 = arith.sitofp %55 : i32 to f32
              %57 = math.fma %56, %cst_30, %53 : f32
              %58 = math.fma %56, %cst_31, %57 : f32
              %59 = arith.addi %55, %c1_i32 : i32
              %60 = llvm.call_intrinsic "llvm.nvvm.mul.rn.f"(%58, %58) : (f32, f32) -> f32
              %61 = arith.andi %55, %c1_i32 : i32
              %62 = arith.cmpi eq, %61, %c0_i32 : i32
              %63 = arith.select %62, %cst_23, %58 {fastmathFlags = #llvm.fastmath<none>} : f32
              %64 = math.fma %60, %63, %cst_18 : f32
              %65 = math.fma %60, %cst_37, %cst_38 : f32
              %66 = arith.select %62, %65, %cst_39 {fastmathFlags = #llvm.fastmath<none>} : f32
              %67 = arith.select %62, %cst_40, %cst_41 {fastmathFlags = #llvm.fastmath<none>} : f32
              %68 = math.fma %66, %60, %67 : f32
              %69 = arith.select %62, %cst_42, %cst_43 {fastmathFlags = #llvm.fastmath<none>} : f32
              %70 = math.fma %68, %60, %69 : f32
              %71 = math.fma %70, %64, %63 : f32
              %72 = arith.andi %59, %c2_i32 : i32
              %73 = arith.cmpi eq, %72, %c0_i32 : i32
              %74 = arith.subf %cst_18, %71 {fastmathFlags = #llvm.fastmath<none>} : f32
              %75 = arith.select %73, %71, %74 {fastmathFlags = #llvm.fastmath<none>} : f32
              %76 = arith.mulf %40, %75 {fastmathFlags = #llvm.fastmath<none>} : f32
              scf.yield %76 : f32
            }
            scf.yield %27 : f32
          }
          %24:3 = scf.while (%arg2 = %23, %arg3 = %22, %arg4 = %c2_i32) : (f32, f32, i32) -> (f32, f32, i32) {
            %25 = arith.addi %arg4, %c-1_i32 : i32
            %26 = arith.uitofp %25 {nonNeg} : i32 to f32
            %27 = arith.mulf %arg3, %26 {fastmathFlags = #llvm.fastmath<none>} : f32
            %28 = arith.negf %arg2 {fastmathFlags = #llvm.fastmath<none>} : f32
            %29 = math.fma %27, %14, %28 : f32
            %30 = arith.cmpi ne, %arg4, %arg0 : i32
            scf.condition(%30) %arg3, %29, %arg4 : f32, f32, i32
          } do {
          ^bb0(%arg2: f32, %arg3: f32, %arg4: i32):
            %25 = arith.addi %arg4, %c1_i32 : i32
            scf.yield %arg2, %arg3, %25 : f32, f32, i32
          }
          scf.yield %24#1 : f32
        } else {
          %14 = arith.muli %arg0, %c40_i32 : i32
          %15 = arith.uitofp %14 {nonNeg} : i32 to f32
          %16 = llvm.call_intrinsic "llvm.nvvm.sqrt.approx.f"(%15) : (f32) -> f32
          %17 = arith.fptosi %16 : f32 to i32
          %18 = arith.addi %arg0, %17 : i32
          %19 = arith.cmpi sgt, %18, %c1_i32 : i32
          %20 = scf.if %19 -> (f32) {
            %21 = arith.andi %18, %c2147483646_i32 : i32
            %22:5 = scf.while (%arg2 = %cst_23, %arg3 = %21, %arg4 = %cst_18, %arg5 = %cst_18, %arg6 = %cst_18) : (f32, i32, f32, f32, f32) -> (f32, i32, f32, f32, f32) {
              %25 = arith.uitofp %arg3 {nonNeg} : i32 to f32
              %26 = arith.mulf %25, %cst_76 {fastmathFlags = #llvm.fastmath<none>} : f32
              %27 = arith.divf %26, %arg1 {fastmathFlags = #llvm.fastmath<none>} : f32
              %28 = arith.mulf %arg2, %27 {fastmathFlags = #llvm.fastmath<none>} : f32
              %29 = arith.subf %28, %arg6 {fastmathFlags = #llvm.fastmath<none>} : f32
              %30 = llvm.call_intrinsic "llvm.nvvm.fabs.f32"(%29) : (f32) -> f32
              %31 = arith.cmpf ogt, %30, %cst_77 {fastmathFlags = #llvm.fastmath<none>} : f32
              %32 = arith.mulf %arg4, %cst_78 {fastmathFlags = #llvm.fastmath<none>} : f32
              %33 = arith.mulf %arg5, %cst_78 {fastmathFlags = #llvm.fastmath<none>} : f32
              %34 = arith.mulf %arg2, %cst_78 {fastmathFlags = #llvm.fastmath<none>} : f32
              %35 = arith.mulf %29, %cst_78 {fastmathFlags = #llvm.fastmath<none>} : f32
              %36 = arith.select %31, %34, %arg2 : f32
              %37 = arith.select %31, %33, %arg5 : f32
              %38 = arith.select %31, %32, %arg4 : f32
              %39 = arith.select %31, %35, %29 : f32
              %40 = arith.addi %arg3, %c-1_i32 : i32
              %41 = arith.cmpi eq, %40, %arg0 : i32
              %42 = arith.select %41, %39, %37 {fastmathFlags = #llvm.fastmath<none>} : f32
              %43 = arith.andi %arg3, %c1_i32 : i32
              %44 = arith.cmpi eq, %43, %c0_i32 : i32
              %45 = arith.mulf %39, %cst_76 {fastmathFlags = #llvm.fastmath<none>} : f32
              %46 = arith.addf %38, %45 {fastmathFlags = #llvm.fastmath<none>} : f32
              %47 = arith.select %44, %38, %46 {fastmathFlags = #llvm.fastmath<none>} : f32
              %48 = arith.cmpi ugt, %arg3, %c1_i32 : i32
              scf.condition(%48) %39, %40, %47, %42, %36 : f32, i32, f32, f32, f32
            } do {
            ^bb0(%arg2: f32, %arg3: i32, %arg4: f32, %arg5: f32, %arg6: f32):
              scf.yield %arg2, %arg3, %arg4, %arg5, %arg6 : f32, i32, f32, f32, f32
            }
            %23 = arith.subf %22#2, %22#0 {fastmathFlags = #llvm.fastmath<none>} : f32
            %24 = arith.divf %22#3, %23 {fastmathFlags = #llvm.fastmath<none>} : f32
            scf.yield %24 : f32
          } else {
            scf.yield %cst_75 : f32
          }
          scf.yield %20 : f32
        }
        scf.yield %13 : f32
      }
      scf.yield %6 : f32
    }
    llvm.return %4 : f32
  }
}


// CHECK:  llvm.func local_unnamed_addr @__nv_jnf(%arg0: i32 {llvm.noundef}, %arg1: f32) -> f32
// CHECK-NEXT:    %0 = ub.poison : i32
// CHECK-NEXT:    %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    %cst = arith.constant 8.000000e+00 : f32
// CHECK-NEXT:    %cst_0 = arith.constant -3.83170605 : f32
// CHECK-NEXT:    %cst_1 = arith.constant 7.68505898E-8 : f32
// CHECK-NEXT:    %cst_2 = arith.constant 7.78064881E-14 : f32
// CHECK-NEXT:    %cst_3 = arith.constant 9.219086E-13 : f32
// CHECK-NEXT:    %cst_4 = arith.constant -2.57068828E-11 : f32
// CHECK-NEXT:    %cst_5 = arith.constant -2.0179057E-10 : f32
// CHECK-NEXT:    %cst_6 = arith.constant 4.51252768E-9 : f32
// CHECK-NEXT:    %cst_7 = arith.constant 2.70162896E-8 : f32
// CHECK-NEXT:    %cst_8 = arith.constant -5.34779701E-7 : f32
// CHECK-NEXT:    %cst_9 = arith.constant -2.36027631E-6 : f32
// CHECK-NEXT:    %cst_10 = arith.constant 4.12108093E-5 : f32
// CHECK-NEXT:    %cst_11 = arith.constant 1.1917029E-4 : f32
// CHECK-NEXT:    %cst_12 = arith.constant -0.00180755882 : f32
// CHECK-NEXT:    %cst_13 = arith.constant -0.0025548928 : f32
// CHECK-NEXT:    %cst_14 = arith.constant 0.0330138914 : f32
// CHECK-NEXT:    %cst_15 = arith.constant -7.01558685 : f32
// CHECK-NEXT:    %cst_16 = arith.constant 1.83211725E-7 : f32
// CHECK-NEXT:    %cst_17 = arith.constant 0x7F800000 : f32
// CHECK-NEXT:    %cst_18 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %cst_19 = arith.constant -4.0873065 : f32
// CHECK-NEXT:    %cst_20 = arith.constant 0.749876558 : f32
// CHECK-NEXT:    %cst_21 = arith.constant -0.192915648 : f32
// CHECK-NEXT:    %cst_22 = arith.constant 0.187498257 : f32
// CHECK-NEXT:    %cst_23 = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:    %cst_24 = arith.constant -1.57994485 : f32
// CHECK-NEXT:    %cst_25 = arith.constant 0.361485869 : f32
// CHECK-NEXT:    %cst_26 = arith.constant -0.164012611 : f32
// CHECK-NEXT:    %cst_27 = arith.constant 0.374999911 : f32
// CHECK-NEXT:    %cst_28 = arith.constant 0.797884583 : f32
// CHECK-NEXT:    %cst_29 = arith.constant 0.636619746 : f32
// CHECK-NEXT:    %cst_30 = arith.constant -1.57079625 : f32
// CHECK-NEXT:    %cst_31 = arith.constant -7.54978942E-8 : f32
// CHECK-NEXT:    %cst_32 = arith.constant -5.39030295E-15 : f32
// CHECK-NEXT:    %cst_33 = arith.constant 1.056150e+05 : f32
// CHECK-NEXT:    %c8_i32 = arith.constant 8 : i32
// CHECK-NEXT:    %c-2147483648_i32 = arith.constant -2147483648 : i32
// CHECK-NEXT:    %1 = llvm.mlir.addressof @__cudart_i2opi_f : !llvm.ptr<1>
// CHECK-NEXT:    %c6_i32 = arith.constant 6 : i32
// CHECK-NEXT:    %c23_i32 = arith.constant 23 : i32
// CHECK-NEXT:    %c224_i32 = arith.constant 224 : i32
// CHECK-NEXT:    %c-128_i32 = arith.constant -128 : i32
// CHECK-NEXT:    %c5_i32 = arith.constant 5 : i32
// CHECK-NEXT:    %c260046848_i32 = arith.constant 260046848 : i32
// CHECK-NEXT:    %c4_i32 = arith.constant 4 : i32
// CHECK-NEXT:    %c30_i32 = arith.constant 30 : i32
// CHECK-NEXT:    %c2_i32 = arith.constant 2 : i32
// CHECK-NEXT:    %c31_i32 = arith.constant 31 : i32
// CHECK-NEXT:    %c-1_i32 = arith.constant -1 : i32
// CHECK-NEXT:    %c32_i64 = arith.constant 32 : i64
// CHECK-NEXT:    %cst_34 = arith.constant 8.5153039502163873E-20 : f64
// CHECK-NEXT:    %c3_i32 = arith.constant 3 : i32
// CHECK-NEXT:    %cst_35 = arith.constant 1.57079637 : f32
// CHECK-NEXT:    %cst_36 = arith.constant -2.3561945 : f32
// CHECK-NEXT:    %cst_37 = arith.constant 2.42795795E-5 : f32
// CHECK-NEXT:    %cst_38 = arith.constant -0.00138878601 : f32
// CHECK-NEXT:    %cst_39 = arith.constant -1.95746587E-4 : f32
// CHECK-NEXT:    %cst_40 = arith.constant 0.0416667275 : f32
// CHECK-NEXT:    %cst_41 = arith.constant 0.00833270326 : f32
// CHECK-NEXT:    %cst_42 = arith.constant -0.49999997 : f32
// CHECK-NEXT:    %cst_43 = arith.constant -0.166666627 : f32
// CHECK-NEXT:    %cst_44 = arith.constant 1.000000e-30 : f32
// CHECK-NEXT:    %cst_45 = arith.constant -2.40482545 : f32
// CHECK-NEXT:    %cst_46 = arith.constant -1.08705898E-7 : f32
// CHECK-NEXT:    %cst_47 = arith.constant -1.2470738E-15 : f32
// CHECK-NEXT:    %cst_48 = arith.constant -7.66777253E-14 : f32
// CHECK-NEXT:    %cst_49 = arith.constant 2.71505561E-12 : f32
// CHECK-NEXT:    %cst_50 = arith.constant -6.02801186E-12 : f32
// CHECK-NEXT:    %cst_51 = arith.constant -4.23932667E-10 : f32
// CHECK-NEXT:    %cst_52 = arith.constant 5.82763782E-10 : f32
// CHECK-NEXT:    %cst_53 = arith.constant 5.80778412E-8 : f32
// CHECK-NEXT:    %cst_54 = arith.constant 1.80033111E-9 : f32
// CHECK-NEXT:    %cst_55 = arith.constant -5.45007333E-6 : f32
// CHECK-NEXT:    %cst_56 = arith.constant -7.34322065E-6 : f32
// CHECK-NEXT:    %cst_57 = arith.constant 3.017032E-4 : f32
// CHECK-NEXT:    %cst_58 = arith.constant 7.73954438E-4 : f32
// CHECK-NEXT:    %cst_59 = arith.constant -0.00728346175 : f32
// CHECK-NEXT:    %cst_60 = arith.constant -0.026668204 : f32
// CHECK-NEXT:    %cst_61 = arith.constant -5.52007818 : f32
// CHECK-NEXT:    %cst_62 = arith.constant 7.19341457E-8 : f32
// CHECK-NEXT:    %cst_63 = arith.constant -8.65372753 : f32
// CHECK-NEXT:    %cst_64 = arith.constant -3.81477918E-7 : f32
// CHECK-NEXT:    %cst_65 = arith.constant 3.35928798 : f32
// CHECK-NEXT:    %cst_66 = arith.constant -0.514522672 : f32
// CHECK-NEXT:    %cst_67 = arith.constant 0.10337057 : f32
// CHECK-NEXT:    %cst_68 = arith.constant -0.0624997243 : f32
// CHECK-NEXT:    %cst_69 = arith.constant 1.13964951 : f32
// CHECK-NEXT:    %cst_70 = arith.constant -0.205326751 : f32
// CHECK-NEXT:    %cst_71 = arith.constant 0.0650917366 : f32
// CHECK-NEXT:    %cst_72 = arith.constant -0.124999993 : f32
// CHECK-NEXT:    %cst_73 = arith.constant -0.785398185 : f32
// CHECK-NEXT:    %cst_74 = arith.constant 0x7FFFFFFF : f32
// CHECK-NEXT:    %c40_i32 = arith.constant 40 : i32
// CHECK-NEXT:    %cst_75 = arith.constant -0.000000e+00 : f32
// CHECK-NEXT:    %c2147483646_i32 = arith.constant 2147483646 : i32
// CHECK-NEXT:    %cst_76 = arith.constant 2.000000e+00 : f32
// CHECK-NEXT:    %cst_77 = arith.constant 9.99999986E+14 : f32
// CHECK-NEXT:    %cst_78 = arith.constant 1.000000e-15 : f32
// CHECK-NEXT:    %2 = llvm.alloca %c1_i32 x !llvm.array<7 x i32> {alignment = 4 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:    %3 = arith.index_castui %arg0 : i32 to index
// CHECK-NEXT:    %4 = scf.index_switch %3 -> f32 
// CHECK-NEXT:    case 0 {
// CHECK-NEXT:      %5 = llvm.call_intrinsic "llvm.nvvm.fabs.f32"(%arg1) : (f32) -> f32
// CHECK-NEXT:      %6 = arith.cmpf ugt, %5, %cst {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:      %7 = scf.if %6 -> (f32) {
// CHECK-NEXT:        %8 = llvm.call_intrinsic "llvm.nvvm.fabs.f32"(%5) : (f32) -> f32
// CHECK-NEXT:        %9 = arith.cmpf oeq, %8, %cst_17 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:        %10 = scf.if %9 -> (f32) {
// CHECK-NEXT:          scf.yield %cst_18 : f32
// CHECK-NEXT:        } else {
// CHECK-NEXT:          %11 = llvm.inline_asm tail_call_kind = <tail> asm_dialect = att "rcp.approx.ftz.f32 $0,$1;", "=f,f" %5 : (f32) -> f32
// CHECK-NEXT:          %12 = arith.mulf %11, %11 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:          %13 = math.fma %12, %cst_65, %cst_66 : f32
// CHECK-NEXT:          %14 = math.fma %13, %12, %cst_67 : f32
// CHECK-NEXT:          %15 = math.fma %14, %12, %cst_68 : f32
// CHECK-NEXT:          %16 = math.fma %15, %12, %cst_23 : f32
// CHECK-NEXT:          %17 = math.fma %12, %cst_69, %cst_70 : f32
// CHECK-NEXT:          %18 = math.fma %17, %12, %cst_71 : f32
// CHECK-NEXT:          %19 = math.fma %18, %12, %cst_72 : f32
// CHECK-NEXT:          %20 = math.fma %19, %11, %5 : f32
// CHECK-NEXT:          %21 = llvm.call_intrinsic "llvm.nvvm.rsqrt.approx.f"(%5) : (f32) -> f32
// CHECK-NEXT:          %22 = arith.mulf %21, %cst_28 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:          %23 = arith.mulf %22, %16 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:          %24 = arith.mulf %20, %cst_29 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:          %25 = llvm.call_intrinsic "llvm.nvvm.f2i.rn"(%24) : (f32) -> i32
// CHECK-NEXT:          %26 = arith.sitofp %25 : i32 to f32
// CHECK-NEXT:          %27 = math.fma %26, %cst_30, %20 : f32
// CHECK-NEXT:          %28 = math.fma %26, %cst_31, %27 : f32
// CHECK-NEXT:          %29 = math.fma %26, %cst_32, %28 : f32
// CHECK-NEXT:          %30 = llvm.call_intrinsic "llvm.nvvm.fabs.f32"(%20) : (f32) -> f32
// CHECK-NEXT:          %31 = arith.cmpf ogt, %30, %cst_33 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:          %32:2 = scf.if %31 -> (i32, f32) {
// CHECK-NEXT:            %60 = arith.cmpf oeq, %30, %cst_17 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:            %61:2 = scf.if %60 -> (i32, f32) {
// CHECK-NEXT:              %62 = llvm.call_intrinsic "llvm.nvvm.mul.rn.f"(%20, %cst_18) : (f32, f32 {llvm.noundef}) -> f32
// CHECK-NEXT:              scf.yield %25, %62 : i32, f32
// CHECK-NEXT:            } else {
// CHECK-NEXT:              %62 = arith.bitcast %20 : f32 to i32
// CHECK-NEXT:              %63 = arith.shli %62, %c8_i32 : i32
// CHECK-NEXT:              %64 = arith.ori %63, %c-2147483648_i32 : i32
// CHECK-NEXT:              %65:2 = affine.for %arg2 = 0 to 6 iter_args(%arg3 = %c0_i32, %arg4 = %0) -> (i32, i32) {
// CHECK-NEXT:                %110 = arith.index_cast %arg2 : index to i32
// CHECK-NEXT:                %111 = arith.extui %110 {nonNeg} : i32 to i64
// CHECK-NEXT:                %112 = llvm.getelementptr inbounds|nuw %1[0, %111] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, !llvm.array<6 x i32>
// CHECK-NEXT:                %113 = llvm.load %112 {alignment = 4 : i64} : !llvm.ptr<1> -> i32
// CHECK-NEXT:                %114 = llvm.inline_asm tail_call_kind = <tail> asm_dialect = att "{\0A\09mad.lo.cc.u32   $0, $2, $3, $4;\0A\09madc.hi.u32     $1, $2, $3,  0;\0A\09}", "=r,=r,r,r,r" %113, %64, %arg3 : (i32, i32, i32) -> !llvm.struct<(i32, i32)>
// CHECK-NEXT:                %115 = llvm.extractvalue %114[0] : !llvm.struct<(i32, i32)> 
// CHECK-NEXT:                %116 = llvm.extractvalue %114[1] : !llvm.struct<(i32, i32)> 
// CHECK-NEXT:                %117 = llvm.getelementptr inbounds|nuw %2[0, %111] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<7 x i32>
// CHECK-NEXT:                llvm.store %115, %117 {alignment = 4 : i64} : i32, !llvm.ptr
// CHECK-NEXT:                affine.yield %116, %116 : i32, i32
// CHECK-NEXT:              }
// CHECK-NEXT:              %66 = arith.shrui %62, %c23_i32 : i32
// CHECK-NEXT:              %67 = arith.andi %66, %c224_i32 : i32
// CHECK-NEXT:              %68 = arith.addi %67, %c-128_i32 : i32
// CHECK-NEXT:              %69 = arith.shrui %68, %c5_i32 exact : i32
// CHECK-NEXT:              %70 = llvm.getelementptr inbounds|nuw %2[24] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK-NEXT:              llvm.store %65#1, %70 {alignment = 4 : i64} : i32, !llvm.ptr
// CHECK-NEXT:              %71 = arith.subi %c6_i32, %69 : i32
// CHECK-NEXT:              %72 = arith.extsi %71 : i32 to i64
// CHECK-NEXT:              %73 = llvm.getelementptr inbounds %2[0, %72] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<7 x i32>
// CHECK-NEXT:              %74 = llvm.load %73 {alignment = 4 : i64} : !llvm.ptr -> i32
// CHECK-NEXT:              %75 = arith.subi %c5_i32, %69 : i32
// CHECK-NEXT:              %76 = arith.extsi %75 : i32 to i64
// CHECK-NEXT:              %77 = llvm.getelementptr inbounds %2[0, %76] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<7 x i32>
// CHECK-NEXT:              %78 = llvm.load %77 {alignment = 4 : i64} : !llvm.ptr -> i32
// CHECK-NEXT:              %79 = arith.andi %62, %c260046848_i32 : i32
// CHECK-NEXT:              %80 = arith.cmpi eq, %79, %c0_i32 : i32
// CHECK-NEXT:              %81 = scf.if %80 -> (i32) {
// CHECK-NEXT:                scf.yield %78 : i32
// CHECK-NEXT:              } else {
// CHECK-NEXT:                %110 = arith.subi %c4_i32, %69 : i32
// CHECK-NEXT:                %111 = arith.extsi %110 : i32 to i64
// CHECK-NEXT:                %112 = llvm.getelementptr inbounds %2[0, %111] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<7 x i32>
// CHECK-NEXT:                %113 = llvm.load %112 {alignment = 4 : i64} : !llvm.ptr -> i32
// CHECK-NEXT:                %114 = llvm.intr.fshl(%78, %113, %66) : (i32, i32, i32) -> i32
// CHECK-NEXT:                scf.yield %114 : i32
// CHECK-NEXT:              }
// CHECK-NEXT:              %82 = llvm.intr.fshl(%74, %78, %66) : (i32, i32, i32) -> i32
// CHECK-NEXT:              %83 = arith.shrui %82, %c30_i32 : i32
// CHECK-NEXT:              %84 = llvm.intr.fshl(%82, %81, %c2_i32) : (i32, i32, i32) -> i32
// CHECK-NEXT:              %85 = arith.shli %81, %c2_i32 : i32
// CHECK-NEXT:              %86 = arith.shrui %84, %c31_i32 : i32
// CHECK-NEXT:              %87 = arith.addi %86, %83 : i32
// CHECK-NEXT:              %88 = arith.subi %c0_i32, %87 : i32
// CHECK-NEXT:              %89 = arith.cmpi slt, %62, %c0_i32 : i32
// CHECK-NEXT:              %90 = arith.select %89, %88, %87 {fastmathFlags = #llvm.fastmath<none>} : i32
// CHECK-NEXT:              %91 = arith.cmpi sgt, %84, %c-1_i32 : i32
// CHECK-NEXT:              %92 = arith.xori %84, %c-1_i32 : i32
// CHECK-NEXT:              %93 = arith.subi %c0_i32, %85 : i32
// CHECK-NEXT:              %94 = arith.cmpi eq, %85, %c0_i32 : i32
// CHECK-NEXT:              %95 = arith.extui %94 : i1 to i32
// CHECK-NEXT:              %96 = arith.addi %95, %92 : i32
// CHECK-NEXT:              %97 = arith.xori %84, %62 : i32
// CHECK-NEXT:              %98 = arith.select %91, %84, %96 {fastmathFlags = #llvm.fastmath<none>} : i32
// CHECK-NEXT:              %99 = arith.select %91, %85, %93 {fastmathFlags = #llvm.fastmath<none>} : i32
// CHECK-NEXT:              %100 = arith.extui %98 : i32 to i64
// CHECK-NEXT:              %101 = arith.shli %100, %c32_i64 : i64
// CHECK-NEXT:              %102 = arith.extui %99 : i32 to i64
// CHECK-NEXT:              %103 = arith.ori %101, %102 {isDisjoint} : i64
// CHECK-NEXT:              %104 = arith.sitofp %103 : i64 to f64
// CHECK-NEXT:              %105 = arith.mulf %104, %cst_34 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:              %106 = arith.truncf %105 : f64 to f32
// CHECK-NEXT:              %107 = arith.negf %106 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:              %108 = arith.cmpi slt, %97, %c0_i32 : i32
// CHECK-NEXT:              %109 = arith.select %108, %107, %106 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:              scf.yield %90, %109 : i32, f32
// CHECK-NEXT:            }
// CHECK-NEXT:            scf.yield %61#0, %61#1 : i32, f32
// CHECK-NEXT:          } else {
// CHECK-NEXT:            scf.yield %25, %29 : i32, f32
// CHECK-NEXT:          }
// CHECK-NEXT:          %33 = arith.andi %32#0, %c3_i32 : i32
// CHECK-NEXT:          %34 = arith.uitofp %33 {nonNeg} : i32 to f32
// CHECK-NEXT:          %35 = math.fma %34, %cst_35, %cst_73 : f32
// CHECK-NEXT:          %36 = arith.addf %32#1, %35 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:          %37 = arith.mulf %36, %cst_29 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:          %38 = llvm.call_intrinsic "llvm.nvvm.f2i.rn"(%37) : (f32) -> i32
// CHECK-NEXT:          %39 = arith.sitofp %38 : i32 to f32
// CHECK-NEXT:          %40 = math.fma %39, %cst_30, %36 : f32
// CHECK-NEXT:          %41 = math.fma %39, %cst_31, %40 : f32
// CHECK-NEXT:          %42 = arith.addi %38, %c1_i32 : i32
// CHECK-NEXT:          %43 = llvm.call_intrinsic "llvm.nvvm.mul.rn.f"(%41, %41) : (f32, f32) -> f32
// CHECK-NEXT:          %44 = arith.andi %38, %c1_i32 : i32
// CHECK-NEXT:          %45 = arith.cmpi eq, %44, %c0_i32 : i32
// CHECK-NEXT:          %46 = arith.select %45, %cst_23, %41 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:          %47 = math.fma %43, %46, %cst_18 : f32
// CHECK-NEXT:          %48 = math.fma %43, %cst_37, %cst_38 : f32
// CHECK-NEXT:          %49 = arith.select %45, %48, %cst_39 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:          %50 = arith.select %45, %cst_40, %cst_41 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:          %51 = math.fma %49, %43, %50 : f32
// CHECK-NEXT:          %52 = arith.select %45, %cst_42, %cst_43 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:          %53 = math.fma %51, %43, %52 : f32
// CHECK-NEXT:          %54 = math.fma %53, %47, %46 : f32
// CHECK-NEXT:          %55 = arith.andi %42, %c2_i32 : i32
// CHECK-NEXT:          %56 = arith.cmpi eq, %55, %c0_i32 : i32
// CHECK-NEXT:          %57 = arith.subf %cst_18, %54 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:          %58 = arith.select %56, %54, %57 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:          %59 = arith.mulf %23, %58 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:          scf.yield %59 : f32
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield %10 : f32
// CHECK-NEXT:      } else {
// CHECK-NEXT:        %8 = arith.addf %5, %cst_45 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:        %9 = arith.addf %8, %cst_46 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:        %10 = math.fma %9, %cst_47, %cst_48 : f32
// CHECK-NEXT:        %11 = math.fma %10, %9, %cst_49 : f32
// CHECK-NEXT:        %12 = math.fma %11, %9, %cst_50 : f32
// CHECK-NEXT:        %13 = math.fma %12, %9, %cst_51 : f32
// CHECK-NEXT:        %14 = math.fma %13, %9, %cst_52 : f32
// CHECK-NEXT:        %15 = math.fma %14, %9, %cst_53 : f32
// CHECK-NEXT:        %16 = math.fma %15, %9, %cst_54 : f32
// CHECK-NEXT:        %17 = math.fma %16, %9, %cst_55 : f32
// CHECK-NEXT:        %18 = math.fma %17, %9, %cst_56 : f32
// CHECK-NEXT:        %19 = math.fma %18, %9, %cst_57 : f32
// CHECK-NEXT:        %20 = math.fma %19, %9, %cst_58 : f32
// CHECK-NEXT:        %21 = math.fma %20, %9, %cst_59 : f32
// CHECK-NEXT:        %22 = math.fma %21, %9, %cst_60 : f32
// CHECK-NEXT:        %23 = arith.mulf %9, %22 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:        %24 = arith.addf %5, %cst_61 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:        %25 = arith.addf %24, %cst_62 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:        %26 = arith.mulf %25, %23 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:        %27 = arith.addf %5, %cst_63 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:        %28 = arith.addf %27, %cst_64 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:        %29 = arith.mulf %28, %26 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:        scf.yield %29 : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %7 : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    case 1 {
// CHECK-NEXT:      %5 = llvm.call_intrinsic "llvm.nvvm.fabs.f32"(%arg1) : (f32) -> f32
// CHECK-NEXT:      %6 = arith.cmpf ugt, %5, %cst {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:      %7 = scf.if %6 -> (f32) {
// CHECK-NEXT:        %14 = llvm.call_intrinsic "llvm.nvvm.fabs.f32"(%5) : (f32) -> f32
// CHECK-NEXT:        %15 = arith.cmpf oeq, %14, %cst_17 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:        %16 = scf.if %15 -> (f32) {
// CHECK-NEXT:          scf.yield %cst_18 : f32
// CHECK-NEXT:        } else {
// CHECK-NEXT:          %17 = llvm.inline_asm tail_call_kind = <tail> asm_dialect = att "rcp.approx.ftz.f32 $0,$1;", "=f,f" %5 : (f32) -> f32
// CHECK-NEXT:          %18 = arith.mulf %17, %17 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:          %19 = math.fma %18, %cst_19, %cst_20 : f32
// CHECK-NEXT:          %20 = math.fma %19, %18, %cst_21 : f32
// CHECK-NEXT:          %21 = math.fma %20, %18, %cst_22 : f32
// CHECK-NEXT:          %22 = math.fma %21, %18, %cst_23 : f32
// CHECK-NEXT:          %23 = math.fma %18, %cst_24, %cst_25 : f32
// CHECK-NEXT:          %24 = math.fma %23, %18, %cst_26 : f32
// CHECK-NEXT:          %25 = math.fma %24, %18, %cst_27 : f32
// CHECK-NEXT:          %26 = math.fma %25, %17, %5 : f32
// CHECK-NEXT:          %27 = llvm.call_intrinsic "llvm.nvvm.rsqrt.approx.f"(%5) : (f32) -> f32
// CHECK-NEXT:          %28 = arith.mulf %27, %cst_28 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:          %29 = arith.mulf %28, %22 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:          %30 = arith.mulf %26, %cst_29 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:          %31 = llvm.call_intrinsic "llvm.nvvm.f2i.rn"(%30) : (f32) -> i32
// CHECK-NEXT:          %32 = arith.sitofp %31 : i32 to f32
// CHECK-NEXT:          %33 = math.fma %32, %cst_30, %26 : f32
// CHECK-NEXT:          %34 = math.fma %32, %cst_31, %33 : f32
// CHECK-NEXT:          %35 = math.fma %32, %cst_32, %34 : f32
// CHECK-NEXT:          %36 = llvm.call_intrinsic "llvm.nvvm.fabs.f32"(%26) : (f32) -> f32
// CHECK-NEXT:          %37 = arith.cmpf ogt, %36, %cst_33 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:          %38:2 = scf.if %37 -> (i32, f32) {
// CHECK-NEXT:            %66 = arith.cmpf oeq, %36, %cst_17 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:            %67:2 = scf.if %66 -> (i32, f32) {
// CHECK-NEXT:              %68 = llvm.call_intrinsic "llvm.nvvm.mul.rn.f"(%26, %cst_18) : (f32, f32 {llvm.noundef}) -> f32
// CHECK-NEXT:              scf.yield %31, %68 : i32, f32
// CHECK-NEXT:            } else {
// CHECK-NEXT:              %68 = arith.bitcast %26 : f32 to i32
// CHECK-NEXT:              %69 = arith.shli %68, %c8_i32 : i32
// CHECK-NEXT:              %70 = arith.ori %69, %c-2147483648_i32 : i32
// CHECK-NEXT:              %71:2 = affine.for %arg2 = 0 to 6 iter_args(%arg3 = %c0_i32, %arg4 = %0) -> (i32, i32) {
// CHECK-NEXT:                %116 = arith.index_cast %arg2 : index to i32
// CHECK-NEXT:                %117 = arith.extui %116 {nonNeg} : i32 to i64
// CHECK-NEXT:                %118 = llvm.getelementptr inbounds|nuw %1[0, %117] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, !llvm.array<6 x i32>
// CHECK-NEXT:                %119 = llvm.load %118 {alignment = 4 : i64} : !llvm.ptr<1> -> i32
// CHECK-NEXT:                %120 = llvm.inline_asm tail_call_kind = <tail> asm_dialect = att "{\0A\09mad.lo.cc.u32   $0, $2, $3, $4;\0A\09madc.hi.u32     $1, $2, $3,  0;\0A\09}", "=r,=r,r,r,r" %119, %70, %arg3 : (i32, i32, i32) -> !llvm.struct<(i32, i32)>
// CHECK-NEXT:                %121 = llvm.extractvalue %120[0] : !llvm.struct<(i32, i32)> 
// CHECK-NEXT:                %122 = llvm.extractvalue %120[1] : !llvm.struct<(i32, i32)> 
// CHECK-NEXT:                %123 = llvm.getelementptr inbounds|nuw %2[0, %117] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<7 x i32>
// CHECK-NEXT:                llvm.store %121, %123 {alignment = 4 : i64} : i32, !llvm.ptr
// CHECK-NEXT:                affine.yield %122, %122 : i32, i32
// CHECK-NEXT:              }
// CHECK-NEXT:              %72 = arith.shrui %68, %c23_i32 : i32
// CHECK-NEXT:              %73 = arith.andi %72, %c224_i32 : i32
// CHECK-NEXT:              %74 = arith.addi %73, %c-128_i32 : i32
// CHECK-NEXT:              %75 = arith.shrui %74, %c5_i32 exact : i32
// CHECK-NEXT:              %76 = llvm.getelementptr inbounds|nuw %2[24] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK-NEXT:              llvm.store %71#1, %76 {alignment = 4 : i64} : i32, !llvm.ptr
// CHECK-NEXT:              %77 = arith.subi %c6_i32, %75 : i32
// CHECK-NEXT:              %78 = arith.extsi %77 : i32 to i64
// CHECK-NEXT:              %79 = llvm.getelementptr inbounds %2[0, %78] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<7 x i32>
// CHECK-NEXT:              %80 = llvm.load %79 {alignment = 4 : i64} : !llvm.ptr -> i32
// CHECK-NEXT:              %81 = arith.subi %c5_i32, %75 : i32
// CHECK-NEXT:              %82 = arith.extsi %81 : i32 to i64
// CHECK-NEXT:              %83 = llvm.getelementptr inbounds %2[0, %82] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<7 x i32>
// CHECK-NEXT:              %84 = llvm.load %83 {alignment = 4 : i64} : !llvm.ptr -> i32
// CHECK-NEXT:              %85 = arith.andi %68, %c260046848_i32 : i32
// CHECK-NEXT:              %86 = arith.cmpi eq, %85, %c0_i32 : i32
// CHECK-NEXT:              %87 = scf.if %86 -> (i32) {
// CHECK-NEXT:                scf.yield %84 : i32
// CHECK-NEXT:              } else {
// CHECK-NEXT:                %116 = arith.subi %c4_i32, %75 : i32
// CHECK-NEXT:                %117 = arith.extsi %116 : i32 to i64
// CHECK-NEXT:                %118 = llvm.getelementptr inbounds %2[0, %117] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<7 x i32>
// CHECK-NEXT:                %119 = llvm.load %118 {alignment = 4 : i64} : !llvm.ptr -> i32
// CHECK-NEXT:                %120 = llvm.intr.fshl(%84, %119, %72) : (i32, i32, i32) -> i32
// CHECK-NEXT:                scf.yield %120 : i32
// CHECK-NEXT:              }
// CHECK-NEXT:              %88 = llvm.intr.fshl(%80, %84, %72) : (i32, i32, i32) -> i32
// CHECK-NEXT:              %89 = arith.shrui %88, %c30_i32 : i32
// CHECK-NEXT:              %90 = llvm.intr.fshl(%88, %87, %c2_i32) : (i32, i32, i32) -> i32
// CHECK-NEXT:              %91 = arith.shli %87, %c2_i32 : i32
// CHECK-NEXT:              %92 = arith.shrui %90, %c31_i32 : i32
// CHECK-NEXT:              %93 = arith.addi %92, %89 : i32
// CHECK-NEXT:              %94 = arith.subi %c0_i32, %93 : i32
// CHECK-NEXT:              %95 = arith.cmpi slt, %68, %c0_i32 : i32
// CHECK-NEXT:              %96 = arith.select %95, %94, %93 {fastmathFlags = #llvm.fastmath<none>} : i32
// CHECK-NEXT:              %97 = arith.cmpi sgt, %90, %c-1_i32 : i32
// CHECK-NEXT:              %98 = arith.xori %90, %c-1_i32 : i32
// CHECK-NEXT:              %99 = arith.subi %c0_i32, %91 : i32
// CHECK-NEXT:              %100 = arith.cmpi eq, %91, %c0_i32 : i32
// CHECK-NEXT:              %101 = arith.extui %100 : i1 to i32
// CHECK-NEXT:              %102 = arith.addi %101, %98 : i32
// CHECK-NEXT:              %103 = arith.xori %90, %68 : i32
// CHECK-NEXT:              %104 = arith.select %97, %90, %102 {fastmathFlags = #llvm.fastmath<none>} : i32
// CHECK-NEXT:              %105 = arith.select %97, %91, %99 {fastmathFlags = #llvm.fastmath<none>} : i32
// CHECK-NEXT:              %106 = arith.extui %104 : i32 to i64
// CHECK-NEXT:              %107 = arith.shli %106, %c32_i64 : i64
// CHECK-NEXT:              %108 = arith.extui %105 : i32 to i64
// CHECK-NEXT:              %109 = arith.ori %107, %108 {isDisjoint} : i64
// CHECK-NEXT:              %110 = arith.sitofp %109 : i64 to f64
// CHECK-NEXT:              %111 = arith.mulf %110, %cst_34 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:              %112 = arith.truncf %111 : f64 to f32
// CHECK-NEXT:              %113 = arith.negf %112 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:              %114 = arith.cmpi slt, %103, %c0_i32 : i32
// CHECK-NEXT:              %115 = arith.select %114, %113, %112 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:              scf.yield %96, %115 : i32, f32
// CHECK-NEXT:            }
// CHECK-NEXT:            scf.yield %67#0, %67#1 : i32, f32
// CHECK-NEXT:          } else {
// CHECK-NEXT:            scf.yield %31, %35 : i32, f32
// CHECK-NEXT:          }
// CHECK-NEXT:          %39 = arith.andi %38#0, %c3_i32 : i32
// CHECK-NEXT:          %40 = arith.uitofp %39 {nonNeg} : i32 to f32
// CHECK-NEXT:          %41 = math.fma %40, %cst_35, %cst_36 : f32
// CHECK-NEXT:          %42 = arith.addf %38#1, %41 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:          %43 = arith.mulf %42, %cst_29 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:          %44 = llvm.call_intrinsic "llvm.nvvm.f2i.rn"(%43) : (f32) -> i32
// CHECK-NEXT:          %45 = arith.sitofp %44 : i32 to f32
// CHECK-NEXT:          %46 = math.fma %45, %cst_30, %42 : f32
// CHECK-NEXT:          %47 = math.fma %45, %cst_31, %46 : f32
// CHECK-NEXT:          %48 = arith.addi %44, %c1_i32 : i32
// CHECK-NEXT:          %49 = llvm.call_intrinsic "llvm.nvvm.mul.rn.f"(%47, %47) : (f32, f32) -> f32
// CHECK-NEXT:          %50 = arith.andi %44, %c1_i32 : i32
// CHECK-NEXT:          %51 = arith.cmpi eq, %50, %c0_i32 : i32
// CHECK-NEXT:          %52 = arith.select %51, %cst_23, %47 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:          %53 = math.fma %49, %52, %cst_18 : f32
// CHECK-NEXT:          %54 = math.fma %49, %cst_37, %cst_38 : f32
// CHECK-NEXT:          %55 = arith.select %51, %54, %cst_39 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:          %56 = arith.select %51, %cst_40, %cst_41 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:          %57 = math.fma %55, %49, %56 : f32
// CHECK-NEXT:          %58 = arith.select %51, %cst_42, %cst_43 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:          %59 = math.fma %57, %49, %58 : f32
// CHECK-NEXT:          %60 = math.fma %59, %53, %52 : f32
// CHECK-NEXT:          %61 = arith.andi %48, %c2_i32 : i32
// CHECK-NEXT:          %62 = arith.cmpi eq, %61, %c0_i32 : i32
// CHECK-NEXT:          %63 = arith.subf %cst_18, %60 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:          %64 = arith.select %62, %60, %63 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:          %65 = arith.mulf %29, %64 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:          scf.yield %65 : f32
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield %16 : f32
// CHECK-NEXT:      } else {
// CHECK-NEXT:        %14 = arith.addf %5, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:        %15 = arith.addf %14, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:        %16 = math.fma %15, %cst_2, %cst_3 : f32
// CHECK-NEXT:        %17 = math.fma %16, %15, %cst_4 : f32
// CHECK-NEXT:        %18 = math.fma %17, %15, %cst_5 : f32
// CHECK-NEXT:        %19 = math.fma %18, %15, %cst_6 : f32
// CHECK-NEXT:        %20 = math.fma %19, %15, %cst_7 : f32
// CHECK-NEXT:        %21 = math.fma %20, %15, %cst_8 : f32
// CHECK-NEXT:        %22 = math.fma %21, %15, %cst_9 : f32
// CHECK-NEXT:        %23 = math.fma %22, %15, %cst_10 : f32
// CHECK-NEXT:        %24 = math.fma %23, %15, %cst_11 : f32
// CHECK-NEXT:        %25 = math.fma %24, %15, %cst_12 : f32
// CHECK-NEXT:        %26 = math.fma %25, %15, %cst_13 : f32
// CHECK-NEXT:        %27 = math.fma %26, %15, %cst_14 : f32
// CHECK-NEXT:        %28 = arith.addf %5, %cst_15 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:        %29 = arith.addf %28, %cst_16 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:        %30 = arith.mulf %29, %27 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:        %31 = arith.mulf %15, %30 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:        %32 = arith.mulf %5, %31 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:        scf.yield %32 : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      %8 = arith.cmpf olt, %arg1, %cst_18 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:      %9 = arith.negf %7 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:      %10 = arith.select %8, %9, %7 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:      %11 = arith.cmpf olt, %5, %cst_44 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:      %12 = math.copysign %7, %arg1 : f32
// CHECK-NEXT:      %13 = arith.select %11, %12, %10 : f32
// CHECK-NEXT:      scf.yield %13 : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    default {
// CHECK-NEXT:      %5 = arith.cmpi slt, %arg0, %c0_i32 : i32
// CHECK-NEXT:      %6 = scf.if %5 -> (f32) {
// CHECK-NEXT:        scf.yield %cst_74 : f32
// CHECK-NEXT:      } else {
// CHECK-NEXT:        %7 = llvm.call_intrinsic "llvm.nvvm.fabs.f32"(%arg1) : (f32) -> f32
// CHECK-NEXT:        %8 = arith.cmpi ugt, %arg0, %c4_i32 : i32
// CHECK-NEXT:        %9 = arith.extui %8 : i1 to i32
// CHECK-NEXT:        %10 = arith.addi %arg0, %9 : i32
// CHECK-NEXT:        %11 = arith.uitofp %10 {nonNeg} : i32 to f32
// CHECK-NEXT:        %12 = arith.cmpf ogt, %7, %11 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:        %13 = scf.if %12 -> (f32) {
// CHECK-NEXT:          %14 = arith.divf %cst_76, %arg1 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:          %15 = arith.cmpf ole, %7, %cst {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:          %16 = scf.if %15 -> (f32) {
// CHECK-NEXT:            %25 = arith.addf %7, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:            %26 = arith.addf %25, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:            %27 = math.fma %26, %cst_2, %cst_3 : f32
// CHECK-NEXT:            %28 = math.fma %27, %26, %cst_4 : f32
// CHECK-NEXT:            %29 = math.fma %28, %26, %cst_5 : f32
// CHECK-NEXT:            %30 = math.fma %29, %26, %cst_6 : f32
// CHECK-NEXT:            %31 = math.fma %30, %26, %cst_7 : f32
// CHECK-NEXT:            %32 = math.fma %31, %26, %cst_8 : f32
// CHECK-NEXT:            %33 = math.fma %32, %26, %cst_9 : f32
// CHECK-NEXT:            %34 = math.fma %33, %26, %cst_10 : f32
// CHECK-NEXT:            %35 = math.fma %34, %26, %cst_11 : f32
// CHECK-NEXT:            %36 = math.fma %35, %26, %cst_12 : f32
// CHECK-NEXT:            %37 = math.fma %36, %26, %cst_13 : f32
// CHECK-NEXT:            %38 = math.fma %37, %26, %cst_14 : f32
// CHECK-NEXT:            %39 = arith.addf %7, %cst_15 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:            %40 = arith.addf %39, %cst_16 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:            %41 = arith.mulf %40, %38 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:            %42 = arith.mulf %26, %41 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:            %43 = arith.mulf %7, %42 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:            scf.yield %43 : f32
// CHECK-NEXT:          } else {
// CHECK-NEXT:            %25 = llvm.call_intrinsic "llvm.nvvm.fabs.f32"(%7) : (f32) -> f32
// CHECK-NEXT:            %26 = arith.cmpf oeq, %25, %cst_17 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:            %27 = scf.if %26 -> (f32) {
// CHECK-NEXT:              scf.yield %cst_18 : f32
// CHECK-NEXT:            } else {
// CHECK-NEXT:              %28 = llvm.inline_asm tail_call_kind = <tail> asm_dialect = att "rcp.approx.ftz.f32 $0,$1;", "=f,f" %7 : (f32) -> f32
// CHECK-NEXT:              %29 = arith.mulf %28, %28 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:              %30 = math.fma %29, %cst_19, %cst_20 : f32
// CHECK-NEXT:              %31 = math.fma %30, %29, %cst_21 : f32
// CHECK-NEXT:              %32 = math.fma %31, %29, %cst_22 : f32
// CHECK-NEXT:              %33 = math.fma %32, %29, %cst_23 : f32
// CHECK-NEXT:              %34 = math.fma %29, %cst_24, %cst_25 : f32
// CHECK-NEXT:              %35 = math.fma %34, %29, %cst_26 : f32
// CHECK-NEXT:              %36 = math.fma %35, %29, %cst_27 : f32
// CHECK-NEXT:              %37 = math.fma %36, %28, %7 : f32
// CHECK-NEXT:              %38 = llvm.call_intrinsic "llvm.nvvm.rsqrt.approx.f"(%7) : (f32) -> f32
// CHECK-NEXT:              %39 = arith.mulf %38, %cst_28 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:              %40 = arith.mulf %39, %33 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:              %41 = arith.mulf %37, %cst_29 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:              %42 = llvm.call_intrinsic "llvm.nvvm.f2i.rn"(%41) : (f32) -> i32
// CHECK-NEXT:              %43 = arith.sitofp %42 : i32 to f32
// CHECK-NEXT:              %44 = math.fma %43, %cst_30, %37 : f32
// CHECK-NEXT:              %45 = math.fma %43, %cst_31, %44 : f32
// CHECK-NEXT:              %46 = math.fma %43, %cst_32, %45 : f32
// CHECK-NEXT:              %47 = llvm.call_intrinsic "llvm.nvvm.fabs.f32"(%37) : (f32) -> f32
// CHECK-NEXT:              %48 = arith.cmpf ogt, %47, %cst_33 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:              %49:2 = scf.if %48 -> (i32, f32) {
// CHECK-NEXT:                %77 = arith.cmpf oeq, %47, %cst_17 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:                %78:2 = scf.if %77 -> (i32, f32) {
// CHECK-NEXT:                  %79 = llvm.call_intrinsic "llvm.nvvm.mul.rn.f"(%37, %cst_18) : (f32, f32 {llvm.noundef}) -> f32
// CHECK-NEXT:                  scf.yield %42, %79 : i32, f32
// CHECK-NEXT:                } else {
// CHECK-NEXT:                  %79 = arith.bitcast %37 : f32 to i32
// CHECK-NEXT:                  %80 = arith.shli %79, %c8_i32 : i32
// CHECK-NEXT:                  %81 = arith.ori %80, %c-2147483648_i32 : i32
// CHECK-NEXT:                  %82:2 = affine.for %arg2 = 0 to 6 iter_args(%arg3 = %c0_i32, %arg4 = %0) -> (i32, i32) {
// CHECK-NEXT:                    %127 = arith.index_cast %arg2 : index to i32
// CHECK-NEXT:                    %128 = arith.extui %127 {nonNeg} : i32 to i64
// CHECK-NEXT:                    %129 = llvm.getelementptr inbounds|nuw %1[0, %128] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, !llvm.array<6 x i32>
// CHECK-NEXT:                    %130 = llvm.load %129 {alignment = 4 : i64} : !llvm.ptr<1> -> i32
// CHECK-NEXT:                    %131 = llvm.inline_asm tail_call_kind = <tail> asm_dialect = att "{\0A\09mad.lo.cc.u32   $0, $2, $3, $4;\0A\09madc.hi.u32     $1, $2, $3,  0;\0A\09}", "=r,=r,r,r,r" %130, %81, %arg3 : (i32, i32, i32) -> !llvm.struct<(i32, i32)>
// CHECK-NEXT:                    %132 = llvm.extractvalue %131[0] : !llvm.struct<(i32, i32)> 
// CHECK-NEXT:                    %133 = llvm.extractvalue %131[1] : !llvm.struct<(i32, i32)> 
// CHECK-NEXT:                    %134 = llvm.getelementptr inbounds|nuw %2[0, %128] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<7 x i32>
// CHECK-NEXT:                    llvm.store %132, %134 {alignment = 4 : i64} : i32, !llvm.ptr
// CHECK-NEXT:                    affine.yield %133, %133 : i32, i32
// CHECK-NEXT:                  }
// CHECK-NEXT:                  %83 = arith.shrui %79, %c23_i32 : i32
// CHECK-NEXT:                  %84 = arith.andi %83, %c224_i32 : i32
// CHECK-NEXT:                  %85 = arith.addi %84, %c-128_i32 : i32
// CHECK-NEXT:                  %86 = arith.shrui %85, %c5_i32 exact : i32
// CHECK-NEXT:                  %87 = llvm.getelementptr inbounds|nuw %2[24] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK-NEXT:                  llvm.store %82#1, %87 {alignment = 4 : i64} : i32, !llvm.ptr
// CHECK-NEXT:                  %88 = arith.subi %c6_i32, %86 : i32
// CHECK-NEXT:                  %89 = arith.extsi %88 : i32 to i64
// CHECK-NEXT:                  %90 = llvm.getelementptr inbounds %2[0, %89] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<7 x i32>
// CHECK-NEXT:                  %91 = llvm.load %90 {alignment = 4 : i64} : !llvm.ptr -> i32
// CHECK-NEXT:                  %92 = arith.subi %c5_i32, %86 : i32
// CHECK-NEXT:                  %93 = arith.extsi %92 : i32 to i64
// CHECK-NEXT:                  %94 = llvm.getelementptr inbounds %2[0, %93] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<7 x i32>
// CHECK-NEXT:                  %95 = llvm.load %94 {alignment = 4 : i64} : !llvm.ptr -> i32
// CHECK-NEXT:                  %96 = arith.andi %79, %c260046848_i32 : i32
// CHECK-NEXT:                  %97 = arith.cmpi eq, %96, %c0_i32 : i32
// CHECK-NEXT:                  %98 = scf.if %97 -> (i32) {
// CHECK-NEXT:                    scf.yield %95 : i32
// CHECK-NEXT:                  } else {
// CHECK-NEXT:                    %127 = arith.subi %c4_i32, %86 : i32
// CHECK-NEXT:                    %128 = arith.extsi %127 : i32 to i64
// CHECK-NEXT:                    %129 = llvm.getelementptr inbounds %2[0, %128] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<7 x i32>
// CHECK-NEXT:                    %130 = llvm.load %129 {alignment = 4 : i64} : !llvm.ptr -> i32
// CHECK-NEXT:                    %131 = llvm.intr.fshl(%95, %130, %83) : (i32, i32, i32) -> i32
// CHECK-NEXT:                    scf.yield %131 : i32
// CHECK-NEXT:                  }
// CHECK-NEXT:                  %99 = llvm.intr.fshl(%91, %95, %83) : (i32, i32, i32) -> i32
// CHECK-NEXT:                  %100 = arith.shrui %99, %c30_i32 : i32
// CHECK-NEXT:                  %101 = llvm.intr.fshl(%99, %98, %c2_i32) : (i32, i32, i32) -> i32
// CHECK-NEXT:                  %102 = arith.shli %98, %c2_i32 : i32
// CHECK-NEXT:                  %103 = arith.shrui %101, %c31_i32 : i32
// CHECK-NEXT:                  %104 = arith.addi %103, %100 : i32
// CHECK-NEXT:                  %105 = arith.subi %c0_i32, %104 : i32
// CHECK-NEXT:                  %106 = arith.cmpi slt, %79, %c0_i32 : i32
// CHECK-NEXT:                  %107 = arith.select %106, %105, %104 {fastmathFlags = #llvm.fastmath<none>} : i32
// CHECK-NEXT:                  %108 = arith.cmpi sgt, %101, %c-1_i32 : i32
// CHECK-NEXT:                  %109 = arith.xori %101, %c-1_i32 : i32
// CHECK-NEXT:                  %110 = arith.subi %c0_i32, %102 : i32
// CHECK-NEXT:                  %111 = arith.cmpi eq, %102, %c0_i32 : i32
// CHECK-NEXT:                  %112 = arith.extui %111 : i1 to i32
// CHECK-NEXT:                  %113 = arith.addi %112, %109 : i32
// CHECK-NEXT:                  %114 = arith.xori %101, %79 : i32
// CHECK-NEXT:                  %115 = arith.select %108, %101, %113 {fastmathFlags = #llvm.fastmath<none>} : i32
// CHECK-NEXT:                  %116 = arith.select %108, %102, %110 {fastmathFlags = #llvm.fastmath<none>} : i32
// CHECK-NEXT:                  %117 = arith.extui %115 : i32 to i64
// CHECK-NEXT:                  %118 = arith.shli %117, %c32_i64 : i64
// CHECK-NEXT:                  %119 = arith.extui %116 : i32 to i64
// CHECK-NEXT:                  %120 = arith.ori %118, %119 {isDisjoint} : i64
// CHECK-NEXT:                  %121 = arith.sitofp %120 : i64 to f64
// CHECK-NEXT:                  %122 = arith.mulf %121, %cst_34 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:                  %123 = arith.truncf %122 : f64 to f32
// CHECK-NEXT:                  %124 = arith.negf %123 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:                  %125 = arith.cmpi slt, %114, %c0_i32 : i32
// CHECK-NEXT:                  %126 = arith.select %125, %124, %123 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:                  scf.yield %107, %126 : i32, f32
// CHECK-NEXT:                }
// CHECK-NEXT:                scf.yield %78#0, %78#1 : i32, f32
// CHECK-NEXT:              } else {
// CHECK-NEXT:                scf.yield %42, %46 : i32, f32
// CHECK-NEXT:              }
// CHECK-NEXT:              %50 = arith.andi %49#0, %c3_i32 : i32
// CHECK-NEXT:              %51 = arith.uitofp %50 {nonNeg} : i32 to f32
// CHECK-NEXT:              %52 = math.fma %51, %cst_35, %cst_36 : f32
// CHECK-NEXT:              %53 = arith.addf %49#1, %52 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:              %54 = arith.mulf %53, %cst_29 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:              %55 = llvm.call_intrinsic "llvm.nvvm.f2i.rn"(%54) : (f32) -> i32
// CHECK-NEXT:              %56 = arith.sitofp %55 : i32 to f32
// CHECK-NEXT:              %57 = math.fma %56, %cst_30, %53 : f32
// CHECK-NEXT:              %58 = math.fma %56, %cst_31, %57 : f32
// CHECK-NEXT:              %59 = arith.addi %55, %c1_i32 : i32
// CHECK-NEXT:              %60 = llvm.call_intrinsic "llvm.nvvm.mul.rn.f"(%58, %58) : (f32, f32) -> f32
// CHECK-NEXT:              %61 = arith.andi %55, %c1_i32 : i32
// CHECK-NEXT:              %62 = arith.cmpi eq, %61, %c0_i32 : i32
// CHECK-NEXT:              %63 = arith.select %62, %cst_23, %58 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:              %64 = math.fma %60, %63, %cst_18 : f32
// CHECK-NEXT:              %65 = math.fma %60, %cst_37, %cst_38 : f32
// CHECK-NEXT:              %66 = arith.select %62, %65, %cst_39 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:              %67 = arith.select %62, %cst_40, %cst_41 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:              %68 = math.fma %66, %60, %67 : f32
// CHECK-NEXT:              %69 = arith.select %62, %cst_42, %cst_43 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:              %70 = math.fma %68, %60, %69 : f32
// CHECK-NEXT:              %71 = math.fma %70, %64, %63 : f32
// CHECK-NEXT:              %72 = arith.andi %59, %c2_i32 : i32
// CHECK-NEXT:              %73 = arith.cmpi eq, %72, %c0_i32 : i32
// CHECK-NEXT:              %74 = arith.subf %cst_18, %71 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:              %75 = arith.select %73, %71, %74 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:              %76 = arith.mulf %40, %75 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:              scf.yield %76 : f32
// CHECK-NEXT:            }
// CHECK-NEXT:            scf.yield %27 : f32
// CHECK-NEXT:          }
// CHECK-NEXT:          %17 = arith.cmpf olt, %arg1, %cst_18 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:          %18 = arith.negf %16 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:          %19 = arith.select %17, %18, %16 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:          %20 = arith.cmpf olt, %7, %cst_44 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:          %21 = math.copysign %16, %arg1 : f32
// CHECK-NEXT:          %22 = arith.select %20, %21, %19 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:          %23 = scf.if %15 -> (f32) {
// CHECK-NEXT:            %25 = arith.addf %7, %cst_45 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:            %26 = arith.addf %25, %cst_46 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:            %27 = math.fma %26, %cst_47, %cst_48 : f32
// CHECK-NEXT:            %28 = math.fma %27, %26, %cst_49 : f32
// CHECK-NEXT:            %29 = math.fma %28, %26, %cst_50 : f32
// CHECK-NEXT:            %30 = math.fma %29, %26, %cst_51 : f32
// CHECK-NEXT:            %31 = math.fma %30, %26, %cst_52 : f32
// CHECK-NEXT:            %32 = math.fma %31, %26, %cst_53 : f32
// CHECK-NEXT:            %33 = math.fma %32, %26, %cst_54 : f32
// CHECK-NEXT:            %34 = math.fma %33, %26, %cst_55 : f32
// CHECK-NEXT:            %35 = math.fma %34, %26, %cst_56 : f32
// CHECK-NEXT:            %36 = math.fma %35, %26, %cst_57 : f32
// CHECK-NEXT:            %37 = math.fma %36, %26, %cst_58 : f32
// CHECK-NEXT:            %38 = math.fma %37, %26, %cst_59 : f32
// CHECK-NEXT:            %39 = math.fma %38, %26, %cst_60 : f32
// CHECK-NEXT:            %40 = arith.mulf %26, %39 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:            %41 = arith.addf %7, %cst_61 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:            %42 = arith.addf %41, %cst_62 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:            %43 = arith.mulf %42, %40 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:            %44 = arith.addf %7, %cst_63 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:            %45 = arith.addf %44, %cst_64 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:            %46 = arith.mulf %45, %43 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:            scf.yield %46 : f32
// CHECK-NEXT:          } else {
// CHECK-NEXT:            %25 = llvm.call_intrinsic "llvm.nvvm.fabs.f32"(%7) : (f32) -> f32
// CHECK-NEXT:            %26 = arith.cmpf oeq, %25, %cst_17 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:            %27 = scf.if %26 -> (f32) {
// CHECK-NEXT:              scf.yield %cst_18 : f32
// CHECK-NEXT:            } else {
// CHECK-NEXT:              %28 = llvm.inline_asm tail_call_kind = <tail> asm_dialect = att "rcp.approx.ftz.f32 $0,$1;", "=f,f" %7 : (f32) -> f32
// CHECK-NEXT:              %29 = arith.mulf %28, %28 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:              %30 = math.fma %29, %cst_65, %cst_66 : f32
// CHECK-NEXT:              %31 = math.fma %30, %29, %cst_67 : f32
// CHECK-NEXT:              %32 = math.fma %31, %29, %cst_68 : f32
// CHECK-NEXT:              %33 = math.fma %32, %29, %cst_23 : f32
// CHECK-NEXT:              %34 = math.fma %29, %cst_69, %cst_70 : f32
// CHECK-NEXT:              %35 = math.fma %34, %29, %cst_71 : f32
// CHECK-NEXT:              %36 = math.fma %35, %29, %cst_72 : f32
// CHECK-NEXT:              %37 = math.fma %36, %28, %7 : f32
// CHECK-NEXT:              %38 = llvm.call_intrinsic "llvm.nvvm.rsqrt.approx.f"(%7) : (f32) -> f32
// CHECK-NEXT:              %39 = arith.mulf %38, %cst_28 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:              %40 = arith.mulf %39, %33 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:              %41 = arith.mulf %37, %cst_29 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:              %42 = llvm.call_intrinsic "llvm.nvvm.f2i.rn"(%41) : (f32) -> i32
// CHECK-NEXT:              %43 = arith.sitofp %42 : i32 to f32
// CHECK-NEXT:              %44 = math.fma %43, %cst_30, %37 : f32
// CHECK-NEXT:              %45 = math.fma %43, %cst_31, %44 : f32
// CHECK-NEXT:              %46 = math.fma %43, %cst_32, %45 : f32
// CHECK-NEXT:              %47 = llvm.call_intrinsic "llvm.nvvm.fabs.f32"(%37) : (f32) -> f32
// CHECK-NEXT:              %48 = arith.cmpf ogt, %47, %cst_33 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:              %49:2 = scf.if %48 -> (i32, f32) {
// CHECK-NEXT:                %77 = arith.cmpf oeq, %47, %cst_17 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:                %78:2 = scf.if %77 -> (i32, f32) {
// CHECK-NEXT:                  %79 = llvm.call_intrinsic "llvm.nvvm.mul.rn.f"(%37, %cst_18) : (f32, f32 {llvm.noundef}) -> f32
// CHECK-NEXT:                  scf.yield %42, %79 : i32, f32
// CHECK-NEXT:                } else {
// CHECK-NEXT:                  %79 = arith.bitcast %37 : f32 to i32
// CHECK-NEXT:                  %80 = arith.shli %79, %c8_i32 : i32
// CHECK-NEXT:                  %81 = arith.ori %80, %c-2147483648_i32 : i32
// CHECK-NEXT:                  %82:2 = affine.for %arg2 = 0 to 6 iter_args(%arg3 = %c0_i32, %arg4 = %0) -> (i32, i32) {
// CHECK-NEXT:                    %127 = arith.index_cast %arg2 : index to i32
// CHECK-NEXT:                    %128 = arith.extui %127 {nonNeg} : i32 to i64
// CHECK-NEXT:                    %129 = llvm.getelementptr inbounds|nuw %1[0, %128] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, !llvm.array<6 x i32>
// CHECK-NEXT:                    %130 = llvm.load %129 {alignment = 4 : i64} : !llvm.ptr<1> -> i32
// CHECK-NEXT:                    %131 = llvm.inline_asm tail_call_kind = <tail> asm_dialect = att "{\0A\09mad.lo.cc.u32   $0, $2, $3, $4;\0A\09madc.hi.u32     $1, $2, $3,  0;\0A\09}", "=r,=r,r,r,r" %130, %81, %arg3 : (i32, i32, i32) -> !llvm.struct<(i32, i32)>
// CHECK-NEXT:                    %132 = llvm.extractvalue %131[0] : !llvm.struct<(i32, i32)> 
// CHECK-NEXT:                    %133 = llvm.extractvalue %131[1] : !llvm.struct<(i32, i32)> 
// CHECK-NEXT:                    %134 = llvm.getelementptr inbounds|nuw %2[0, %128] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<7 x i32>
// CHECK-NEXT:                    llvm.store %132, %134 {alignment = 4 : i64} : i32, !llvm.ptr
// CHECK-NEXT:                    affine.yield %133, %133 : i32, i32
// CHECK-NEXT:                  }
// CHECK-NEXT:                  %83 = arith.shrui %79, %c23_i32 : i32
// CHECK-NEXT:                  %84 = arith.andi %83, %c224_i32 : i32
// CHECK-NEXT:                  %85 = arith.addi %84, %c-128_i32 : i32
// CHECK-NEXT:                  %86 = arith.shrui %85, %c5_i32 exact : i32
// CHECK-NEXT:                  %87 = llvm.getelementptr inbounds|nuw %2[24] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK-NEXT:                  llvm.store %82#1, %87 {alignment = 4 : i64} : i32, !llvm.ptr
// CHECK-NEXT:                  %88 = arith.subi %c6_i32, %86 : i32
// CHECK-NEXT:                  %89 = arith.extsi %88 : i32 to i64
// CHECK-NEXT:                  %90 = llvm.getelementptr inbounds %2[0, %89] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<7 x i32>
// CHECK-NEXT:                  %91 = llvm.load %90 {alignment = 4 : i64} : !llvm.ptr -> i32
// CHECK-NEXT:                  %92 = arith.subi %c5_i32, %86 : i32
// CHECK-NEXT:                  %93 = arith.extsi %92 : i32 to i64
// CHECK-NEXT:                  %94 = llvm.getelementptr inbounds %2[0, %93] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<7 x i32>
// CHECK-NEXT:                  %95 = llvm.load %94 {alignment = 4 : i64} : !llvm.ptr -> i32
// CHECK-NEXT:                  %96 = arith.andi %79, %c260046848_i32 : i32
// CHECK-NEXT:                  %97 = arith.cmpi eq, %96, %c0_i32 : i32
// CHECK-NEXT:                  %98 = scf.if %97 -> (i32) {
// CHECK-NEXT:                    scf.yield %95 : i32
// CHECK-NEXT:                  } else {
// CHECK-NEXT:                    %127 = arith.subi %c4_i32, %86 : i32
// CHECK-NEXT:                    %128 = arith.extsi %127 : i32 to i64
// CHECK-NEXT:                    %129 = llvm.getelementptr inbounds %2[0, %128] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<7 x i32>
// CHECK-NEXT:                    %130 = llvm.load %129 {alignment = 4 : i64} : !llvm.ptr -> i32
// CHECK-NEXT:                    %131 = llvm.intr.fshl(%95, %130, %83) : (i32, i32, i32) -> i32
// CHECK-NEXT:                    scf.yield %131 : i32
// CHECK-NEXT:                  }
// CHECK-NEXT:                  %99 = llvm.intr.fshl(%91, %95, %83) : (i32, i32, i32) -> i32
// CHECK-NEXT:                  %100 = arith.shrui %99, %c30_i32 : i32
// CHECK-NEXT:                  %101 = llvm.intr.fshl(%99, %98, %c2_i32) : (i32, i32, i32) -> i32
// CHECK-NEXT:                  %102 = arith.shli %98, %c2_i32 : i32
// CHECK-NEXT:                  %103 = arith.shrui %101, %c31_i32 : i32
// CHECK-NEXT:                  %104 = arith.addi %103, %100 : i32
// CHECK-NEXT:                  %105 = arith.subi %c0_i32, %104 : i32
// CHECK-NEXT:                  %106 = arith.cmpi slt, %79, %c0_i32 : i32
// CHECK-NEXT:                  %107 = arith.select %106, %105, %104 {fastmathFlags = #llvm.fastmath<none>} : i32
// CHECK-NEXT:                  %108 = arith.cmpi sgt, %101, %c-1_i32 : i32
// CHECK-NEXT:                  %109 = arith.xori %101, %c-1_i32 : i32
// CHECK-NEXT:                  %110 = arith.subi %c0_i32, %102 : i32
// CHECK-NEXT:                  %111 = arith.cmpi eq, %102, %c0_i32 : i32
// CHECK-NEXT:                  %112 = arith.extui %111 : i1 to i32
// CHECK-NEXT:                  %113 = arith.addi %112, %109 : i32
// CHECK-NEXT:                  %114 = arith.select %108, %102, %110 {fastmathFlags = #llvm.fastmath<none>} : i32
// CHECK-NEXT:                  %115 = arith.select %108, %101, %113 {fastmathFlags = #llvm.fastmath<none>} : i32
// CHECK-NEXT:                  %116 = arith.xori %101, %79 : i32
// CHECK-NEXT:                  %117 = arith.extui %115 : i32 to i64
// CHECK-NEXT:                  %118 = arith.shli %117, %c32_i64 : i64
// CHECK-NEXT:                  %119 = arith.extui %114 : i32 to i64
// CHECK-NEXT:                  %120 = arith.ori %118, %119 {isDisjoint} : i64
// CHECK-NEXT:                  %121 = arith.sitofp %120 : i64 to f64
// CHECK-NEXT:                  %122 = arith.mulf %121, %cst_34 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:                  %123 = arith.truncf %122 : f64 to f32
// CHECK-NEXT:                  %124 = arith.negf %123 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:                  %125 = arith.cmpi slt, %116, %c0_i32 : i32
// CHECK-NEXT:                  %126 = arith.select %125, %124, %123 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:                  scf.yield %107, %126 : i32, f32
// CHECK-NEXT:                }
// CHECK-NEXT:                scf.yield %78#0, %78#1 : i32, f32
// CHECK-NEXT:              } else {
// CHECK-NEXT:                scf.yield %42, %46 : i32, f32
// CHECK-NEXT:              }
// CHECK-NEXT:              %50 = arith.andi %49#0, %c3_i32 : i32
// CHECK-NEXT:              %51 = arith.uitofp %50 {nonNeg} : i32 to f32
// CHECK-NEXT:              %52 = math.fma %51, %cst_35, %cst_73 : f32
// CHECK-NEXT:              %53 = arith.addf %49#1, %52 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:              %54 = arith.mulf %53, %cst_29 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:              %55 = llvm.call_intrinsic "llvm.nvvm.f2i.rn"(%54) : (f32) -> i32
// CHECK-NEXT:              %56 = arith.sitofp %55 : i32 to f32
// CHECK-NEXT:              %57 = math.fma %56, %cst_30, %53 : f32
// CHECK-NEXT:              %58 = math.fma %56, %cst_31, %57 : f32
// CHECK-NEXT:              %59 = arith.addi %55, %c1_i32 : i32
// CHECK-NEXT:              %60 = llvm.call_intrinsic "llvm.nvvm.mul.rn.f"(%58, %58) : (f32, f32) -> f32
// CHECK-NEXT:              %61 = arith.andi %55, %c1_i32 : i32
// CHECK-NEXT:              %62 = arith.cmpi eq, %61, %c0_i32 : i32
// CHECK-NEXT:              %63 = arith.select %62, %cst_23, %58 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:              %64 = math.fma %60, %63, %cst_18 : f32
// CHECK-NEXT:              %65 = math.fma %60, %cst_37, %cst_38 : f32
// CHECK-NEXT:              %66 = arith.select %62, %65, %cst_39 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:              %67 = arith.select %62, %cst_40, %cst_41 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:              %68 = math.fma %66, %60, %67 : f32
// CHECK-NEXT:              %69 = arith.select %62, %cst_42, %cst_43 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:              %70 = math.fma %68, %60, %69 : f32
// CHECK-NEXT:              %71 = math.fma %70, %64, %63 : f32
// CHECK-NEXT:              %72 = arith.andi %59, %c2_i32 : i32
// CHECK-NEXT:              %73 = arith.cmpi eq, %72, %c0_i32 : i32
// CHECK-NEXT:              %74 = arith.subf %cst_18, %71 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:              %75 = arith.select %73, %71, %74 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:              %76 = arith.mulf %40, %75 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:              scf.yield %76 : f32
// CHECK-NEXT:            }
// CHECK-NEXT:            scf.yield %27 : f32
// CHECK-NEXT:          }
// CHECK-NEXT:          %24:3 = scf.while (%arg2 = %23, %arg3 = %22, %arg4 = %c2_i32) : (f32, f32, i32) -> (f32, f32, i32) {
// CHECK-NEXT:            %25 = arith.addi %arg4, %c-1_i32 : i32
// CHECK-NEXT:            %26 = arith.uitofp %25 {nonNeg} : i32 to f32
// CHECK-NEXT:            %27 = arith.mulf %arg3, %26 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:            %28 = arith.negf %arg2 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:            %29 = math.fma %27, %14, %28 : f32
// CHECK-NEXT:            %30 = arith.cmpi ne, %arg4, %arg0 : i32
// CHECK-NEXT:            scf.condition(%30) %arg3, %29, %arg4 : f32, f32, i32
// CHECK-NEXT:          } do {
// CHECK-NEXT:          ^bb0(%arg2: f32, %arg3: f32, %arg4: i32):
// CHECK-NEXT:            %25 = arith.addi %arg4, %c1_i32 : i32
// CHECK-NEXT:            scf.yield %arg2, %arg3, %25 : f32, f32, i32
// CHECK-NEXT:          }
// CHECK-NEXT:          scf.yield %24#1 : f32
// CHECK-NEXT:        } else {
// CHECK-NEXT:          %14 = arith.muli %arg0, %c40_i32 : i32
// CHECK-NEXT:          %15 = arith.uitofp %14 {nonNeg} : i32 to f32
// CHECK-NEXT:          %16 = llvm.call_intrinsic "llvm.nvvm.sqrt.approx.f"(%15) : (f32) -> f32
// CHECK-NEXT:          %17 = arith.fptosi %16 : f32 to i32
// CHECK-NEXT:          %18 = arith.addi %arg0, %17 : i32
// CHECK-NEXT:          %19 = arith.cmpi sgt, %18, %c1_i32 : i32
// CHECK-NEXT:          %20 = scf.if %19 -> (f32) {
// CHECK-NEXT:            %21 = arith.andi %18, %c2147483646_i32 : i32
// CHECK-NEXT:            %22:5 = scf.while (%arg2 = %cst_23, %arg3 = %21, %arg4 = %cst_18, %arg5 = %cst_18, %arg6 = %cst_18) : (f32, i32, f32, f32, f32) -> (f32, i32, f32, f32, f32) {
// CHECK-NEXT:              %25 = arith.uitofp %arg3 {nonNeg} : i32 to f32
// CHECK-NEXT:              %26 = arith.mulf %25, %cst_76 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:              %27 = arith.divf %26, %arg1 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:              %28 = arith.mulf %arg2, %27 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:              %29 = arith.subf %28, %arg6 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:              %30 = llvm.call_intrinsic "llvm.nvvm.fabs.f32"(%29) : (f32) -> f32
// CHECK-NEXT:              %31 = arith.cmpf ogt, %30, %cst_77 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:              %32 = arith.mulf %arg4, %cst_78 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:              %33 = arith.mulf %arg5, %cst_78 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:              %34 = arith.mulf %arg2, %cst_78 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:              %35 = arith.mulf %29, %cst_78 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:              %36 = arith.select %31, %34, %arg2 : f32
// CHECK-NEXT:              %37 = arith.select %31, %33, %arg5 : f32
// CHECK-NEXT:              %38 = arith.select %31, %32, %arg4 : f32
// CHECK-NEXT:              %39 = arith.select %31, %35, %29 : f32
// CHECK-NEXT:              %40 = arith.addi %arg3, %c-1_i32 : i32
// CHECK-NEXT:              %41 = arith.cmpi eq, %40, %arg0 : i32
// CHECK-NEXT:              %42 = arith.select %41, %39, %37 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:              %43 = arith.andi %arg3, %c1_i32 : i32
// CHECK-NEXT:              %44 = arith.cmpi eq, %43, %c0_i32 : i32
// CHECK-NEXT:              %45 = arith.mulf %39, %cst_76 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:              %46 = arith.addf %38, %45 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:              %47 = arith.select %44, %38, %46 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:              %48 = arith.cmpi ugt, %arg3, %c1_i32 : i32
// CHECK-NEXT:              scf.condition(%48) %39, %40, %47, %42, %36 : f32, i32, f32, f32, f32
// CHECK-NEXT:            } do {
// CHECK-NEXT:            ^bb0(%arg2: f32, %arg3: i32, %arg4: f32, %arg5: f32, %arg6: f32):
// CHECK-NEXT:              scf.yield %arg2, %arg3, %arg4, %arg5, %arg6 : f32, i32, f32, f32, f32
// CHECK-NEXT:            }
// CHECK-NEXT:            %23 = arith.subf %22#2, %22#0 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:            %24 = arith.divf %22#3, %23 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:            scf.yield %24 : f32
// CHECK-NEXT:          } else {
// CHECK-NEXT:            scf.yield %cst_75 : f32
// CHECK-NEXT:          }
// CHECK-NEXT:          scf.yield %20 : f32
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield %13 : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %6 : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    llvm.return %4 : f32
// CHECK-NEXT:  }
