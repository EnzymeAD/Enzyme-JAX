// RUN: enzymexlamlir-opt --libdevice-funcs-raise %s | FileCheck %s

#tbaa_root = #llvm.tbaa_root<id = "custom_tbaa">
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "custom_tbaa_addrspace(1)", members = {<#tbaa_root, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc, access_type = #tbaa_type_desc, offset = 0>
  llvm.func local_unnamed_addr @__nv_fabs(f64) -> f64 attributes {sym_visibility = "private"}
  llvm.func local_unnamed_addr @__nv_sin(f64) -> f64 attributes {sym_visibility = "private"}
  // CHECK: @raise_constants
  // CHECK-NOT: llvm.mlir.constant
  llvm.func ptx_kernelcc @raise_constants(%arg0: !llvm.ptr<1> {llvm.align = 128 : i64, llvm.dereferenceable = 9336832 : i64, llvm.dereferenceable_or_null = 9336832 : i64, llvm.noalias, llvm.nocapture, llvm.nofree, llvm.noundef, llvm.writeonly}, %arg1: !llvm.ptr<1> {llvm.align = 128 : i64, llvm.dereferenceable = 752 : i64, llvm.dereferenceable_or_null = 752 : i64, llvm.noalias, llvm.nocapture, llvm.nofree, llvm.noundef, llvm.readonly}, %arg2: !llvm.ptr<1> {llvm.align = 128 : i64, llvm.dereferenceable = 752 : i64, llvm.dereferenceable_or_null = 752 : i64, llvm.noalias, llvm.nocapture, llvm.nofree, llvm.noundef, llvm.readonly}, %arg3: !llvm.ptr<1> {llvm.align = 128 : i64, llvm.dereferenceable = 9436160 : i64, llvm.dereferenceable_or_null = 9436160 : i64, llvm.noalias, llvm.nocapture, llvm.nofree, llvm.noundef, llvm.readonly}, %arg4: !llvm.ptr<1> {llvm.align = 128 : i64, llvm.dereferenceable = 145888 : i64, llvm.dereferenceable_or_null = 145888 : i64, llvm.noalias, llvm.nocapture, llvm.nofree, llvm.noundef, llvm.readonly}, %arg5: !llvm.ptr<1> {llvm.align = 128 : i64, llvm.dereferenceable = 9336832 : i64, llvm.dereferenceable_or_null = 9336832 : i64, llvm.noalias, llvm.nocapture, llvm.nofree, llvm.noundef, llvm.readonly}) attributes {sym_visibility = "private"} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.mlir.constant(12 : i32) : i32
    %3 = llvm.mlir.constant(-12 : i64) : i64
    %4 = llvm.mlir.constant(5 : i32) : i32
    %5 = llvm.mlir.constant(-5 : i64) : i64
    %6 = llvm.mlir.constant(16 : i16) : i16
    %7 = llvm.mlir.constant(0 : i64) : i64
    %8 = llvm.mlir.constant(16 : i8) : i8
    %9 = llvm.mlir.constant(16 : i64) : i64
    %10 = llvm.mlir.constant(180 : i64) : i64
    %11 = llvm.mlir.constant(80 : i64) : i64
    %12 = llvm.mlir.constant(50 : i64) : i64
    %13 = llvm.mlir.constant(true) : i1
    %14 = llvm.mlir.constant(6 : i64) : i64
    %15 = llvm.mlir.constant(194 : i64) : i64
    %16 = llvm.mlir.constant(5 : i64) : i64
    %17 = llvm.mlir.constant(1.000000e-01 : f64) : f64
    %18 = llvm.mlir.constant(-0.000000e+00 : f64) : f64
    %19 = llvm.mlir.constant(-7 : i64) : i64
    %20 = llvm.mlir.constant(-48 : i64) : i64
    %21 = llvm.mlir.constant(2.000000e+00 : f64) : f64
    %22 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    // NOTE: specifically this constant should be gone.
    %23 = llvm.mlir.constant(dense<0.000000e+00> : tensor<2xf64>) : !llvm.array<2 x f64>
    %24 = llvm.mlir.constant(3.1415926535897931 : f64) : f64
    %25 = llvm.mlir.constant(1.800000e+02 : f64) : f64
    %26 = llvm.mlir.constant(1.458423E-4 : f64) : f64
    %27 = llvm.mlir.constant(7 : i64) : i64
    %28 = llvm.mlir.constant(5.000000e-01 : f64) : f64
    %29 = llvm.mlir.constant(18430 : i64) : i64
    %30 = llvm.mlir.constant(18236 : i64) : i64
    %31 = nvvm.read.ptx.sreg.ctaid.x range <i32, 0, 3000> : i32
    %32 = llvm.add %31, %0 overflow<nsw, nuw> : i32
    %33 = nvvm.read.ptx.sreg.tid.x range <i32, 0, 256> : i32
    %34 = llvm.add %33, %0 overflow<nsw, nuw> : i32
    %35 = llvm.zext %32 : i32 to i64
    %36 = llvm.sub %35, %1 overflow<nsw, nuw> : i64
    %37 = llvm.trunc %36 : i64 to i32
    %38 = llvm.udiv %37, %2 : i32
    %39 = llvm.zext %38 : i32 to i64
    %40 = llvm.mul %39, %3 overflow<nsw> : i64
    %41 = llvm.udiv %38, %4 : i32
    %42 = llvm.zext %41 : i32 to i64
    %43 = llvm.mul %42, %5 overflow<nsw> : i64
    %44 = llvm.zext %34 : i32 to i64
    %45 = llvm.sub %44, %1 overflow<nsw, nuw> : i64
    %46 = llvm.trunc %45 : i64 to i16
    %47 = llvm.udiv %46, %6 : i16
    %48 = llvm.zext %47 : i16 to i64
    %49 = llvm.sub %7, %48 overflow<nsw> : i64
    %50 = llvm.trunc %48 : i64 to i8
    %51 = llvm.udiv %50, %8 : i8
    %52 = llvm.zext %51 : i8 to i64
    %53 = llvm.sub %7, %52 overflow<nsw> : i64
    %54 = llvm.add %52, %1 overflow<nsw, nuw> : i64
    %55 = llvm.add %49, %36 overflow<nsw> : i64
    %56 = llvm.add %55, %40 overflow<nsw> : i64
    %57 = llvm.mul %56, %9 overflow<nsw> : i64
    %58 = llvm.add %44, %57 overflow<nsw> : i64
    %59 = llvm.add %53, %39 overflow<nsw> : i64
    %60 = llvm.add %59, %43 overflow<nsw> : i64
    %61 = llvm.mul %60, %9 overflow<nsw> : i64
    %62 = llvm.add %48, %1 overflow<nsw, nuw> : i64
    %63 = llvm.add %62, %61 overflow<nsw> : i64
    %64 = llvm.add %54, %42 overflow<nsw, nuw> : i64
    %65 = llvm.icmp "sle" %1, %58 : i64
    %66 = llvm.icmp "sle" %58, %10 : i64
    %67 = llvm.and %65, %66 : i1
    %68 = llvm.icmp "sle" %1, %63 : i64
    %69 = llvm.icmp "sle" %63, %11 : i64
    %70 = llvm.and %68, %69 : i1
    %71 = llvm.icmp "ule" %64, %12 : i64
    %72 = llvm.and %67, %70 : i1
    %73 = llvm.and %71, %72 : i1
    %74 = llvm.xor %73, %13 : i1
    llvm.cond_br %74, ^bb1, ^bb2
  ^bb1:  // 2 preds: ^bb0, ^bb2
    llvm.br ^bb3
  ^bb2:  // pred: ^bb0
    %75 = llvm.add %63, %14 : i64
    %76 = llvm.mul %75, %15 : i64
    %77 = llvm.add %14, %58 : i64
    %78 = llvm.add %77, %76 : i64
    %79 = llvm.getelementptr inbounds %arg4[%78] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
    %80 = llvm.load %79 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr<1> -> f64
    %81 = llvm.add %16, %58 : i64
    %82 = llvm.add %81, %76 : i64
    %83 = llvm.getelementptr inbounds %arg4[%82] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
    %84 = llvm.load %83 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr<1> -> f64
    %85 = llvm.fsub %80, %84 : f64
    %86 = llvm.add %63, %14 : i64
    %87 = llvm.getelementptr inbounds %arg1[%86] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
    %88 = llvm.load %87 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr<1> -> f64
    %89 = llvm.fdiv %85, %88 : f64
    %90 = llvm.fmul %17, %89 : f64
    %91 = llvm.fsub %18, %90 : f64
    %92 = llvm.sub %63, %19 : i64
    %93 = llvm.add %92, %20 : i64
    %94 = llvm.sitofp %93 : i64 to f64
    %95 = llvm.fmul %21, %94 : f64
    %96 = llvm.fmul %22, %94 : f64
    %97 = llvm.call @__nv_fabs(%95) : (f64) -> f64
    %98 = llvm.call @__nv_fabs(%22) : (f64 {llvm.noundef}) -> f64
    %99 = llvm.fcmp "olt" %98, %97 : f64
    %100 = llvm.xor %99, %13 : i1
    %101 = llvm.insertvalue %95, %23[1] : !llvm.array<2 x f64> 
    %102 = llvm.insertvalue %95, %23[0] : !llvm.array<2 x f64> 
    %103 = llvm.insertvalue %22, %102[1] : !llvm.array<2 x f64> 
    %104 = llvm.select %100, %101, %103 : i1, !llvm.array<2 x f64>
    %105 = llvm.extractvalue %104[0] : !llvm.array<2 x f64> 
    %106 = llvm.extractvalue %104[1] : !llvm.array<2 x f64> 
    %107 = llvm.fadd %105, %106 : f64
    %108 = llvm.fsub %105, %107 : f64
    %109 = llvm.fadd %106, %108 : f64
    %110 = llvm.fadd %22, %96 : f64
    %111 = llvm.fadd %110, %109 : f64
    %112 = llvm.fadd %107, %111 : f64
    %113 = llvm.fmul %112, %24 : f64
    %114 = llvm.fdiv %113, %25 : f64
    %115 = llvm.call @__nv_sin(%114) : (f64) -> f64
    %116 = llvm.fmul %26, %115 : f64
    %117 = llvm.add %63, %1 overflow<nsw, nuw> : i64
    %118 = llvm.add %117, %27 : i64
    %119 = llvm.add %118, %20 : i64
    %120 = llvm.sitofp %119 : i64 to f64
    %121 = llvm.fmul %21, %120 : f64
    %122 = llvm.fmul %22, %120 : f64
    %123 = llvm.call @__nv_fabs(%121) : (f64) -> f64
    %124 = llvm.call @__nv_fabs(%22) : (f64 {llvm.noundef}) -> f64
    %125 = llvm.fcmp "olt" %124, %123 : f64
    %126 = llvm.xor %125, %13 : i1
    %127 = llvm.insertvalue %121, %23[1] : !llvm.array<2 x f64> 
    %128 = llvm.insertvalue %121, %23[0] : !llvm.array<2 x f64> 
    %129 = llvm.insertvalue %22, %128[1] : !llvm.array<2 x f64> 
    %130 = llvm.select %126, %127, %129 : i1, !llvm.array<2 x f64>
    %131 = llvm.extractvalue %130[0] : !llvm.array<2 x f64> 
    %132 = llvm.extractvalue %130[1] : !llvm.array<2 x f64> 
    %133 = llvm.fadd %131, %132 : f64
    %134 = llvm.fsub %131, %133 : f64
    %135 = llvm.fadd %132, %134 : f64
    %136 = llvm.fadd %22, %122 : f64
    %137 = llvm.fadd %136, %135 : f64
    %138 = llvm.fadd %133, %137 : f64
    %139 = llvm.fmul %138, %24 : f64
    %140 = llvm.fdiv %139, %25 : f64
    %141 = llvm.call @__nv_sin(%140) : (f64) -> f64
    %142 = llvm.fmul %26, %141 : f64
    %143 = llvm.fadd %116, %142 : f64
    %144 = llvm.fmul %143, %28 : f64
    %145 = llvm.fneg %144 : f64
    %146 = llvm.sub %63, %1 overflow<nsw, nuw> : i64
    %147 = llvm.icmp "ult" %146, %1 : i64
    %148 = llvm.icmp "slt" %11, %146 : i64
    %149 = llvm.or %147, %148 : i1
    %150 = llvm.xor %149, %13 : i1
    %151 = llvm.zext %150 : i1 to i64
    %152 = llvm.add %151, %151 overflow<nsw, nuw> : i64
    %153 = llvm.sitofp %152 : i64 to f64
    %154 = llvm.fmul %153, %28 : f64
    %155 = llvm.icmp "slt" %11, %117 : i64
    %156 = llvm.xor %155, %13 : i1
    %157 = llvm.zext %156 : i1 to i64
    %158 = llvm.add %157, %157 overflow<nsw, nuw> : i64
    %159 = llvm.sitofp %158 : i64 to f64
    %160 = llvm.fmul %159, %28 : f64
    %161 = llvm.fadd %154, %160 : f64
    %162 = llvm.fmul %161, %28 : f64
    %163 = llvm.fcmp "oeq" %162, %22 : f64
    %164 = llvm.add %63, %14 : i64
    %165 = llvm.getelementptr inbounds %arg2[%164] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
    %166 = llvm.load %165 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr<1> -> f64
    %167 = llvm.add %63, %14 : i64
    %168 = llvm.mul %167, %15 : i64
    %169 = llvm.add %64, %14 : i64
    %170 = llvm.mul %169, %29 : i64
    %171 = llvm.add %58, %16 : i64
    %172 = llvm.add %171, %170 : i64
    %173 = llvm.add %172, %168 : i64
    %174 = llvm.getelementptr inbounds %arg3[%173] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
    %175 = llvm.load %174 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr<1> -> f64
    %176 = llvm.fmul %166, %175 : f64
    %177 = llvm.add %58, %14 : i64
    %178 = llvm.add %177, %170 : i64
    %179 = llvm.add %178, %168 : i64
    %180 = llvm.getelementptr inbounds %arg3[%179] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
    %181 = llvm.load %180 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr<1> -> f64
    %182 = llvm.fmul %166, %181 : f64
    %183 = llvm.fadd %176, %182 : f64
    %184 = llvm.fmul %183, %28 : f64
    %185 = llvm.add %117, %14 : i64
    %186 = llvm.getelementptr inbounds %arg2[%185] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
    %187 = llvm.load %186 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr<1> -> f64
    %188 = llvm.add %117, %14 : i64
    %189 = llvm.mul %188, %15 : i64
    %190 = llvm.add %172, %189 : i64
    %191 = llvm.getelementptr inbounds %arg3[%190] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
    %192 = llvm.load %191 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr<1> -> f64
    %193 = llvm.fmul %187, %192 : f64
    %194 = llvm.add %178, %189 : i64
    %195 = llvm.getelementptr inbounds %arg3[%194] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
    %196 = llvm.load %195 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr<1> -> f64
    %197 = llvm.fmul %187, %196 : f64
    %198 = llvm.fadd %193, %197 : f64
    %199 = llvm.fmul %198, %28 : f64
    %200 = llvm.fadd %184, %199 : f64
    %201 = llvm.fmul %200, %28 : f64
    %202 = llvm.fdiv %201, %162 : f64
    %203 = llvm.xor %163, %13 : i1
    %204 = llvm.select %203, %202, %22 : i1, f64
    %205 = llvm.fmul %145, %204 : f64
    %206 = llvm.load %87 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr<1> -> f64
    %207 = llvm.fdiv %205, %206 : f64
    %208 = llvm.fsub %91, %207 : f64
    %209 = llvm.add %63, %14 : i64
    %210 = llvm.mul %209, %15 : i64
    %211 = llvm.add %64, %14 : i64
    %212 = llvm.mul %211, %30 : i64
    %213 = llvm.add %58, %14 : i64
    %214 = llvm.add %213, %212 : i64
    %215 = llvm.add %214, %210 : i64
    %216 = llvm.getelementptr inbounds %arg5[%215] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
    %217 = llvm.load %216 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr<1> -> f64
    %218 = llvm.add %58, %16 : i64
    %219 = llvm.add %218, %212 : i64
    %220 = llvm.add %219, %210 : i64
    %221 = llvm.getelementptr inbounds %arg5[%220] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
    %222 = llvm.load %221 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr<1> -> f64
    %223 = llvm.fsub %217, %222 : f64
    %224 = llvm.fdiv %223, %206 : f64
    %225 = llvm.fsub %208, %224 : f64
    %226 = llvm.fadd %225, %22 : f64
    %227 = llvm.add %63, %14 : i64
    %228 = llvm.mul %227, %15 : i64
    %229 = llvm.add %64, %14 : i64
    %230 = llvm.mul %229, %30 : i64
    %231 = llvm.add %58, %14 : i64
    %232 = llvm.add %231, %230 : i64
    %233 = llvm.add %232, %228 : i64
    %234 = llvm.getelementptr inbounds %arg0[%233] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
    llvm.store %226, %234 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : f64, !llvm.ptr<1>
    llvm.br ^bb1
  ^bb3:  // pred: ^bb1
    llvm.return
  }
