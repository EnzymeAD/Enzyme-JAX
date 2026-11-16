// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

module attributes {gpu.container_module} {
  func.func @test(%112: !llvm.ptr, %113: i64, %2: i32, %232: !llvm.ptr) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c0_i64 = arith.constant 0 : i64
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c31_i32 = arith.constant 31 : i32
    %c-1_i32 = arith.constant -1 : i32

    scf.parallel (%arg3) = (%c0) to (%c256) step (%c1) {
      %243 = llvm.ptrtoint %112 : !llvm.ptr to i64
      %245 = arith.cmpi eq, %243, %c0_i64 : i64
      %247:4 = scf.if %245 -> (i32, i32, i32, i32) {
        scf.yield %c0_i32, %c0_i32, %c0_i32, %c0_i32 : i32, i32, i32, i32
      } else {
        scf.yield %c0_i32, %c0_i32, %c0_i32, %c0_i32 : i32, i32, i32, i32
      }
      %250 = arith.cmpi eq, %113, %c0_i64 : i64
      %251 = arith.select %250, %247#0, %247#2 : i32
      %252 = arith.select %250, %247#1, %247#3 : i32
      %253:3 = scf.if %250 -> (i32, i32, i32) {
        scf.yield %2, %2, %2 : i32, i32, i32
      } else {
        scf.yield %2, %2, %c0_i32 : i32, i32, i32
      }
      %255 = arith.cmpi eq, %253#2, %c0_i32 : i32
      %256:2 = scf.if %255 -> (i32, i32) {
        %258 = llvm.inline_asm has_side_effects tail_call_kind = <tail> asm_dialect = att "shfl.sync.down.b32 $0, $1, $2, $3, $4;", "=r,r,r,r,r" %251, %c1_i32, %c31_i32, %c-1_i32 : (i32, i32, i32, i32) -> i32
        %277 = arith.addi %251, %258 : i32
        %279 = arith.cmpi eq, %252, %c0_i32 : i32
        %282 = scf.if %279 -> (i32) {
          %283 = llvm.load %232 : !llvm.ptr -> i32
          %284 = arith.addi %283, %277 : i32
          scf.yield %284 : i32
        } else {
          scf.yield %2 : i32
        }
        scf.yield %282, %c0_i32 : i32, i32
      } else {
        scf.yield %253#0, %253#1 : i32, i32
      }
      scf.reduce
    }
    func.return
  }
}

// The selects become affine.ifs yielding results of %247, an scf.if keyed off a
// pure ptrtoint that hoists out of the parallel ahead of them; nothing may be
// hoisted above the values it yields.

// CHECK:       #set = affine_set<()[s0] : (s0 == 0)>
// CHECK-LABEL:   func.func @test(
// CHECK-SAME:                    %[[PTR:.+]]: !llvm.ptr, %[[N:.+]]: i64, %[[V:.+]]: i32, %[[PTR2:.+]]: !llvm.ptr) {
// CHECK-NEXT:      %[[CM1_I32:.+]] = arith.constant -1 : i32
// CHECK-NEXT:      %[[C31_I32:.+]] = arith.constant 31 : i32
// CHECK-NEXT:      %[[C1_I32:.+]] = arith.constant 1 : i32
// CHECK-NEXT:      %[[C0_I32:.+]] = arith.constant 0 : i32
// CHECK-NEXT:      %[[C0_I64:.+]] = arith.constant 0 : i64
// CHECK-NEXT:      %[[NIDX:.+]] = arith.index_cast %[[N]] : i64 to index
// CHECK-NEXT:      %[[PI:.+]] = llvm.ptrtoint %[[PTR]] : !llvm.ptr to i64
// CHECK-NEXT:      %[[PZ:.+]] = arith.cmpi eq, %[[PI]], %[[C0_I64]] : i64
// CHECK-NEXT:      %[[NZ:.+]] = arith.cmpi eq, %[[N]], %[[C0_I64]] : i64
// CHECK-NEXT:      %[[IF:.+]]:4 = scf.if %[[PZ]] -> (i32, i32, i32, i32) {
// CHECK-NEXT:        scf.yield %[[C0_I32]], %[[C0_I32]], %[[C0_I32]], %[[C0_I32]] : i32, i32, i32, i32
// CHECK-NEXT:      } else {
// CHECK-NEXT:        scf.yield %[[C0_I32]], %[[C0_I32]], %[[C0_I32]], %[[C0_I32]] : i32, i32, i32, i32
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[SEL1:.+]] = arith.select %[[NZ]], %[[IF]]#1, %[[IF]]#3 : i32
// CHECK-NEXT:      %[[SEL1IDX:.+]] = arith.index_cast %[[SEL1]] : i32 to index
// CHECK-NEXT:      %[[IF2:.+]]:3 = scf.if %[[NZ]] -> (i32, i32, i32) {
// CHECK-NEXT:        scf.yield %[[V]], %[[V]], %[[V]] : i32, i32, i32
// CHECK-NEXT:      } else {
// CHECK-NEXT:        scf.yield %[[V]], %[[V]], %[[C0_I32]] : i32, i32, i32
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[IF2IDX:.+]] = arith.index_cast %[[IF2]]#2 : i32 to index
// CHECK-NEXT:      affine.parallel (%[[TID:.+]]) = (0) to (256) {
// CHECK-NEXT:        %[[SEL0:.+]] = affine.if #set()[%[[NIDX]]] -> i32 {
// CHECK-NEXT:          affine.yield %[[IF]]#0 : i32
// CHECK-NEXT:        } else {
// CHECK-NEXT:          affine.yield %[[IF]]#2 : i32
// CHECK-NEXT:        }
// CHECK-NEXT:        %{{.+}}:2 = affine.if #set()[%[[IF2IDX]]] -> (i32, i32) {
// CHECK-NEXT:          %[[ASM:.+]] = llvm.inline_asm has_side_effects tail_call_kind = <tail> asm_dialect = att "shfl.sync.down.b32 $0, $1, $2, $3, $4;", "=r,r,r,r,r" %[[SEL0]], %[[C1_I32]], %[[C31_I32]], %[[CM1_I32]] : (i32, i32, i32, i32) -> i32
// CHECK-NEXT:          %[[ADD:.+]] = arith.addi %[[SEL0]], %[[ASM]] : i32
// CHECK-NEXT:          %[[INNER:.+]] = affine.if #set()[%[[SEL1IDX]]] -> i32 {
// CHECK-NEXT:            %[[LOAD:.+]] = llvm.load %[[PTR2]] : !llvm.ptr -> i32
// CHECK-NEXT:            %[[ADD2:.+]] = arith.addi %[[LOAD]], %[[ADD]] : i32
// CHECK-NEXT:            affine.yield %[[ADD2]] : i32
// CHECK-NEXT:          } else {
// CHECK-NEXT:            affine.yield %[[V]] : i32
// CHECK-NEXT:          }
// CHECK-NEXT:          affine.yield %[[INNER]], %[[C0_I32]] : i32, i32
// CHECK-NEXT:        } else {
// CHECK-NEXT:          affine.yield %[[IF2]]#0, %[[IF2]]#1 : i32, i32
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }
