//===- Passes.td - EnzymeXLA pass tablegen macros  ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "stablehlo/dialect/StablehloOps.td"

////////
// Mul/Div || Add/Sub - constant computation lifting

// For ops with same precedenct, we can leverage op relationships (commutative,
// associative, inverse properties) to lift constant computations.
// Note: These patterns assume a `CanonicalizeConstantToRhs` run prior, and op
// folding applied after.

// Pattern: op(op(X, cstA), cstB) -> op(X, op(cstA, cstB))
class BinOpLiftConstantComputation<
    Op OpTypeA, Op OpTypeB, Op ResOpTypeA = OpTypeA, Op ResOpTypeB = OpTypeB>
    : Pat<(OpTypeA
           : $opA(OpTypeB
                  : $opB $lhsB, (StableHLO_ConstantOp
                                 : $rhsB $valueB)),
             (StableHLO_ConstantOp
              : $rhsA $valueA)),
          (ResOpTypeA $lhsB, (ResOpTypeB $rhsB, $rhsA))>;

// Flip the order of the constants, useful for divide and subtract.
// Pattern: op(op(X, cstA), cstB) -> op(X, op(cstB, cstA))
class BinOpLiftConstantComputationFlip<
    Op OpTypeA, Op OpTypeB, Op ResOpTypeA = OpTypeA, Op ResOpTypeB = OpTypeB>
    : Pat<(OpTypeA
           : $opA(OpTypeB
                  : $opB $lhsB, (StableHLO_ConstantOp
                                 : $rhsB $valueB)),
             (StableHLO_ConstantOp
              : $rhsA $valueA)),
          (ResOpTypeA $lhsB, (ResOpTypeB $rhsA, $rhsB))>;

// Naming convention: `<RootOp><ChildOp>Const`:

// Pattern: (x / cst_1) * cst_2 -> x * (cst_2 / cst_1)
def MulDivConst
    : BinOpLiftConstantComputationFlip<StableHLO_MulOp, StableHLO_DivOp>;

// Pattern: (x * cst_1) * cst_2 -> x * (cst_1 * cst_2)
def MulMulConst
    : BinOpLiftConstantComputation<StableHLO_MulOp, StableHLO_MulOp>;

// Pattern: (x * cst_1) / cst_2 -> x * (cst_1 / cst_2)
def DivMulConst
    : BinOpLiftConstantComputation<StableHLO_DivOp, StableHLO_MulOp,
                                   StableHLO_MulOp, StableHLO_DivOp>;

// Pattern: (x / cst_1) / cst_2 -> x / (cst_1 * cst_2)
def DivDivConst
    : BinOpLiftConstantComputation<StableHLO_DivOp, StableHLO_DivOp,
                                   StableHLO_DivOp, StableHLO_MulOp>;

// Pattern: (x + cst_1) + cst_2 -> x + (cst_1 + cst_2)
def AddAddConst
    : BinOpLiftConstantComputation<StableHLO_AddOp, StableHLO_AddOp>;

// Pattern: (x - cst_1) + cst_2 -> x + (cst_2 - cst_1)
def AddSubConst
    : BinOpLiftConstantComputationFlip<StableHLO_AddOp, StableHLO_SubtractOp,
                                       StableHLO_AddOp, StableHLO_SubtractOp>;

// Pattern: (x - cst_1) - cst_2 -> x - (cst_1 + cst_2)
def SubSubConst
    : BinOpLiftConstantComputation<StableHLO_SubtractOp, StableHLO_SubtractOp,
                                   StableHLO_SubtractOp, StableHLO_AddOp>;

// Pattern: (x + cst_1) - cst_2 -> x + (cst_1 - cst_2)
def SubAddConst
    : BinOpLiftConstantComputation<StableHLO_SubtractOp, StableHLO_AddOp,
                                   StableHLO_AddOp, StableHLO_SubtractOp>;
