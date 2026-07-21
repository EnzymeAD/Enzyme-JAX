//===- Passes.td - EnzymeXLA pass tablegen macros  ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "stablehlo/dialect/StablehloOps.td"
include "stablehlo/dialect/ChloOps.td"

////////
// Mul/Div || Add/Sub - constant computation lifting

// For ops with same precedence, we can leverage op relationships (commutative,
// associative, inverse properties) to lift constant computations.
// Note: These patterns assume a `CanonicalizeConstantToRhs` run prior, and op
// folding applied after.

// Pattern: op(op(X, cstA), cstB) -> op(X, op(cstA, cstB))
class BinOpLiftConstantComputation<Op OpTypeA, Op OpTypeB,
                                   Op ResOpTypeA = OpTypeA, Op ResOpTypeB = OpTypeB,
                                   list<dag> constraints = []>
  : Pat<(OpTypeA:$opA
          (OpTypeB:$opB $lhsB, (StableHLO_ConstantOp:$rhsB $valueB)),
          (StableHLO_ConstantOp:$rhsA $valueA)),
        (ResOpTypeA $lhsB, (ResOpTypeB $rhsB, $rhsA)),
        constraints>;

// Flip the order of the constants, useful for divide and subtract.
// Pattern: op(op(X, cstA), cstB) -> op(X, op(cstB, cstA))
class BinOpLiftConstantComputationFlip<Op OpTypeA, Op OpTypeB,
                                       Op ResOpTypeA = OpTypeA, Op ResOpTypeB = OpTypeB,
                                       list<dag> constraints = []>
  : Pat<(OpTypeA:$opA
          (OpTypeB:$opB $lhsB, (StableHLO_ConstantOp:$rhsB $valueB)),
          (StableHLO_ConstantOp:$rhsA $valueA)),
        (ResOpTypeA $lhsB, (ResOpTypeB $rhsA, $rhsB)),
        constraints>;

// Naming convention: `<RootOp><ChildOp>Const`:

def IsFloatTensor : Constraint<CPred<
  "isa<mlir::FloatType>(cast<mlir::ShapedType>($0.getType()).getElementType())">,
  "tensor has a floating point element type">;

// Pattern: (x * a) / b -> x * (a / b)
// valid for floats and integers when `b` divides `a` exactly (e.g. `(x * 7) / 7 == x`, but
// `(x * 5) / 2 != x * 2`).
def DivMulReassociable : Constraint<CPred<
  "::mlir::enzyme::divMulReassociable($0, $1, $2)">,
  "reassociating (x * a) / b is value-preserving (float, or b divides a exactly)">;

// Pattern: (x / cst_1) * cst_2 -> x * (cst_2 / cst_1)   (float only)
def MulDivConst : BinOpLiftConstantComputationFlip<StableHLO_MulOp, StableHLO_DivOp,
    StableHLO_MulOp, StableHLO_DivOp, [(IsFloatTensor $opA)]>;

// Pattern: (x * cst_1) * cst_2 -> x * (cst_1 * cst_2)
def MulMulConst : BinOpLiftConstantComputation<StableHLO_MulOp, StableHLO_MulOp>;

// Pattern: (x * cst_1) / cst_2 -> x * (cst_1 / cst_2)
// (float, or integer with cst_2 dividing cst_1)
def DivMulConst : BinOpLiftConstantComputation<StableHLO_DivOp, StableHLO_MulOp,
    StableHLO_MulOp, StableHLO_DivOp, [(DivMulReassociable $opA, $valueB, $valueA)]>;

// Pattern: (x / cst_1) / cst_2 -> x / (cst_1 * cst_2)
def DivDivConst : BinOpLiftConstantComputation<StableHLO_DivOp, StableHLO_DivOp, StableHLO_DivOp, StableHLO_MulOp>;

// Pattern: (x + cst_1) + cst_2 -> x + (cst_1 + cst_2)
def AddAddConst : BinOpLiftConstantComputation<StableHLO_AddOp, StableHLO_AddOp>;

// Pattern: (x - cst_1) + cst_2 -> x + (cst_2 - cst_1)
def AddSubConst : BinOpLiftConstantComputationFlip<StableHLO_AddOp, StableHLO_SubtractOp, StableHLO_AddOp, StableHLO_SubtractOp>;

// Pattern: (x - cst_1) - cst_2 -> x - (cst_1 + cst_2)
def SubSubConst : BinOpLiftConstantComputation<StableHLO_SubtractOp, StableHLO_SubtractOp, StableHLO_SubtractOp, StableHLO_AddOp>;

// Pattern: (x + cst_1) - cst_2 -> x + (cst_1 - cst_2)
def SubAddConst : BinOpLiftConstantComputation<StableHLO_SubtractOp, StableHLO_AddOp, StableHLO_AddOp, StableHLO_SubtractOp>;

def HasSameType : Constraint<CPred<"$0.getType() == $1.getType()">>;
def HasDifferentType : Constraint<CPred<"$0.getType() != $1.getType()">>;
def UseOperand : NativeCodeCall<"$0">;

def BitcastConvertCancellation : Pat<
  (StableHLO_BitcastConvertOp:$outer (StableHLO_BitcastConvertOp $x)),
  (UseOperand $x),
  [(HasSameType $outer, $x)]
>;

// ConvertConvertFloat
def ConvertConvertFloatIdentity : Pat<
  (StableHLO_ConvertOp:$outer (StableHLO_ConvertOp:$inner $x)),
  (UseOperand $x),
  [(IsFloatTensor $outer), (IsFloatTensor $inner), (IsFloatTensor $x), (HasSameType $outer, $x)]>;

def ConvertConvertFloat : Pat<
  (StableHLO_ConvertOp:$outer (StableHLO_ConvertOp:$inner $x)),
  (StableHLO_ConvertOp $x),
  [(IsFloatTensor $outer), (IsFloatTensor $inner), (IsFloatTensor $x), (HasDifferentType $outer, $x)]>;

// ConvertConvertInt
def IsIntegerTensor : Constraint<CPred<
  "isa<mlir::IntegerType>(cast<mlir::ShapedType>($0.getType()).getElementType())">,
  "tensor has an integer element type">;

def IsWideningConvert : Constraint<CPred<
  "cast<mlir::ShapedType>($0.getType()).getElementType().getIntOrFloatBitWidth() <= "
  "cast<mlir::ShapedType>($1.getType()).getElementType().getIntOrFloatBitWidth()"
>, "widening conversion">;

def ConvertConvertIntIdentity : Pat<
  (StableHLO_ConvertOp:$outer (StableHLO_ConvertOp:$inner $x)),
  (UseOperand $x),
  [(IsIntegerTensor $outer), (IsIntegerTensor $inner), (IsIntegerTensor $x),
   (IsWideningConvert $x, $inner), (HasSameType $outer, $x)]>;

def ConvertConvertInt : Pat<
  (StableHLO_ConvertOp:$outer (StableHLO_ConvertOp:$inner $x)),
  (StableHLO_ConvertOp $x),
  [(IsIntegerTensor $outer), (IsIntegerTensor $inner), (IsIntegerTensor $x),
   (IsWideningConvert $x, $inner), (HasDifferentType $outer, $x)]>;

// NotSelectSimplify
def NotSelectSimplify : Pat<
  (StableHLO_SelectOp (StableHLO_NotOp $cond), $lhs, $rhs),
  (StableHLO_SelectOp $cond, $rhs, $lhs)>;

// SelectSelectSameCond
def SelectSelectSameCondFalse : Pat<
  (StableHLO_SelectOp $cond, $c, (StableHLO_SelectOp $cond, $a, $b)),
  (StableHLO_SelectOp $cond, $c, $b)>;

def SelectSelectSameCondTrue : Pat<
  (StableHLO_SelectOp $cond, (StableHLO_SelectOp $cond, $a, $b), $c),
  (StableHLO_SelectOp $cond, $a, $c)>;

// RealConjSimplify
def RealConjSimplify : Pat<
  (StableHLO_RealOp (CHLO_ConjOp $x)),
  (StableHLO_RealOp $x)>;

// RealConvertSimplify
def IsNotComplexTensor : Constraint<CPred<
  "!isa<mlir::ComplexType>(cast<mlir::ShapedType>($0.getType()).getElementType())">,
  "tensor has a non-complex element type">;

def RealConvertSimplify : Pat<
  (StableHLO_RealOp (StableHLO_ConvertOp $x)),
  (UseOperand $x),
  [(IsNotComplexTensor $x)]>;

// ConjComplexNegate
def ConjComplexNegate : Pat<
  (CHLO_ConjOp (StableHLO_ComplexOp $a, (StableHLO_NegOp $b))),
  (StableHLO_ComplexOp $a, $b)>;

// NegateImagConj
def NegateImagConj : Pat<
  (StableHLO_NegOp (StableHLO_ImagOp (CHLO_ConjOp $x))),
  (StableHLO_ImagOp $x)>;

def IsAnyZero : Constraint<CPred<
  "mlir::matchPattern($0, mlir::m_AnyZeroFloat()) || "
  "mlir::matchPattern($0, mlir::m_Zero()) || "
  "mlir::matchPattern($0, ::mlir::enzyme::m_AnyZeroComplex())"
>>;

def IsOne : Constraint<CPred<"mlir::matchPattern($0, mlir::m_One()) || mlir::matchPattern($0, mlir::m_OneFloat())">>;

def IsNegOne : Constraint<CPred<"mlir::matchPattern($0, ::mlir::enzyme::m_NegOne()) || mlir::matchPattern($0, ::mlir::enzyme::m_NegOneFloat())">>;

def IsAllOnes : Constraint<CPred<"mlir::matchPattern($0, ::mlir::enzyme::m_AllOnes())">>;

def IsMinusOneValue : Constraint<CPred<
  "[] (mlir::Value val) { "
  "  mlir::SplatElementsAttr attr; "
  "  if (!mlir::matchPattern(val, mlir::m_Constant(&attr))) return false; "
  "  auto doubleVal = ::mlir::enzyme::getDoubleFromAttr(attr.getSplatValue<mlir::TypedAttr>()); "
  "  return doubleVal && *doubleVal == -1.0; "
  "}($0)"
>>;

def HasOneUse : Constraint<CPred<"$0.hasOneUse()">>;


def GuaranteedPurelyReal : Constraint<CPred<
  "::mlir::enzyme::guaranteedPurelyRealResult($0, rewriter)">,
  "result is guaranteed purely real">;

def GuaranteedNonNegative : Constraint<CPred<
  "::mlir::enzyme::guaranteedNonNegativeResult($0.getDefiningOp(), rewriter)">,
  "result is guaranteed non-negative">;

def IsConstantOp : Constraint<CPred<"mlir::matchPattern($0, mlir::m_Constant())">>;

def IsStaticShapeTensor : Constraint<CPred<
  "cast<mlir::ShapedType>($0.getType()).hasStaticShape()">,
  "tensor has a static shape">;

def CreateConstOp0 : NativeCodeCall<"createConstantOpFromScalar($_builder, $_loc, $0.getType(), 0.0)">;
def CreateConstOp1 : NativeCodeCall<"createConstantOpFromScalar($_builder, $_loc, $0.getType(), 1.0)">;

// InvolutionSimplify
def NegNegSimplify : Pat<
  (StableHLO_NegOp (StableHLO_NegOp $x)),
  (UseOperand $x)>;

def NotNotSimplify : Pat<
  (StableHLO_NotOp (StableHLO_NotOp $x)),
  (UseOperand $x)>;

def ConjConjSimplify : Pat<
  (CHLO_ConjOp (CHLO_ConjOp $x)),
  (UseOperand $x)>;

// SignAbsSimplify
def SignAbsSimplify_1 : Pat<
  (StableHLO_MulOp (StableHLO_SignOp $x), (StableHLO_AbsOp $x)),
  (UseOperand $x)>;

def SignAbsSimplify_2 : Pat<
  (StableHLO_MulOp (StableHLO_AbsOp $x), (StableHLO_SignOp $x)),
  (UseOperand $x)>;

// ReplaceNegAddWithSubtract
def ReplaceNegAddWithSubtractRHS : Pat<
  (StableHLO_AddOp $lhs, (StableHLO_NegOp:$neg $rhs)),
  (StableHLO_SubtractOp $lhs, $rhs),
  [(HasOneUse $neg)]>;

def ReplaceNegAddWithSubtractLHS : Pat<
  (StableHLO_AddOp (StableHLO_NegOp:$neg $lhs), $rhs),
  (StableHLO_SubtractOp $rhs, $lhs),
  [(HasOneUse $neg)]>;

// ReplaceSubtractNegWithAdd
def ReplaceSubtractNegWithAdd : Pat<
  (StableHLO_SubtractOp $lhs, (StableHLO_NegOp:$neg $rhs)),
  (StableHLO_AddOp $lhs, $rhs),
  [(HasOneUse $neg)]>;

// ExponentialMinusOneFuse
def ExponentialMinusOneFuse1 : Pat<
  (StableHLO_SubtractOp (StableHLO_ExpOp:$exp $x, $accuracy), $one),
  (StableHLO_Expm1Op $x, $accuracy),
  [(HasOneUse $exp), (IsOne $one)]>;

def ExponentialMinusOneFuse2 : Pat<
  (StableHLO_SubtractOp $one, (StableHLO_ExpOp:$exp $x, $accuracy)),
  (StableHLO_NegOp (StableHLO_Expm1Op $x, $accuracy)),
  [(HasOneUse $exp), (IsOne $one)]>;

// ExponentialMinusOneAddFuse
def ExponentialMinusOneAddFuse1 : Pat<
  (StableHLO_AddOp (StableHLO_ExpOp:$exp $x, $accuracy), $minus_one),
  (StableHLO_Expm1Op $x, $accuracy),
  [(HasOneUse $exp), (IsMinusOneValue $minus_one)]>;

def ExponentialMinusOneAddFuse2 : Pat<
  (StableHLO_AddOp $minus_one, (StableHLO_ExpOp:$exp $x, $accuracy)),
  (StableHLO_Expm1Op $x, $accuracy),
  [(HasOneUse $exp), (IsMinusOneValue $minus_one)]>;

// MinSimplify / MaxSimplify
def MinSimplify : Pat<
  (StableHLO_MinOp $x, $x),
  (UseOperand $x)>;

def MaxSimplify : Pat<
  (StableHLO_MaxOp $x, $x),
  (UseOperand $x)>;

// AddSimplify
def AddSimplifyLHS : Pat<
  (StableHLO_AddOp $zero, $x),
  (UseOperand $x),
  [(IsAnyZero $zero)]>;

def AddSimplifyRHS : Pat<
  (StableHLO_AddOp $x, $zero),
  (UseOperand $x),
  [(IsAnyZero $zero)]>;

// SubSimplify
def SubSimplifyRHSZero : Pat<
  (StableHLO_SubtractOp $x, $zero),
  (UseOperand $x),
  [(IsAnyZero $zero)]>;

def SubSimplifyLHSZero : Pat<
  (StableHLO_SubtractOp $zero, $x),
  (StableHLO_NegOp $x),
  [(IsAnyZero $zero)]>;

def SubSimplifyIdentical : Pat<
  (StableHLO_SubtractOp:$res $x, $x),
  (CreateConstOp0 $res),
  [(IsIntegerTensor $res)]>;

// MulSimplify
def MulSimplifyOneLHS : Pat<
  (StableHLO_MulOp $one, $x),
  (UseOperand $x),
  [(IsOne $one)]>;

def MulSimplifyOneRHS : Pat<
  (StableHLO_MulOp $x, $one),
  (UseOperand $x),
  [(IsOne $one)]>;

def MulSimplifyNegOneLHS : Pat<
  (StableHLO_MulOp $negone, $x),
  (StableHLO_NegOp $x),
  [(IsNegOne $negone)]>;

def MulSimplifyNegOneRHS : Pat<
  (StableHLO_MulOp $x, $negone),
  (StableHLO_NegOp $x),
  [(IsNegOne $negone)]>;

// AndSimplify
def AndSimplifyIdentical : Pat<
  (StableHLO_AndOp $x, $x),
  (UseOperand $x)>;

def AndSimplifyFalseLHS : Pat<
  (StableHLO_AndOp $zero, $x),
  (UseOperand $zero),
  [(IsAnyZero $zero)]>;

def AndSimplifyFalseRHS : Pat<
  (StableHLO_AndOp $x, $zero),
  (UseOperand $zero),
  [(IsAnyZero $zero)]>;

def AndSimplifyTrueLHS : Pat<
  (StableHLO_AndOp $ones, $x),
  (UseOperand $x),
  [(IsAllOnes $ones)]>;

def AndSimplifyTrueRHS : Pat<
  (StableHLO_AndOp $x, $ones),
  (UseOperand $x),
  [(IsAllOnes $ones)]>;

// OrSimplify
def OrSimplifyIdentical : Pat<
  (StableHLO_OrOp $x, $x),
  (UseOperand $x)>;

def OrSimplifyTrueLHS : Pat<
  (StableHLO_OrOp $ones, $x),
  (UseOperand $ones),
  [(IsAllOnes $ones)]>;

def OrSimplifyTrueRHS : Pat<
  (StableHLO_OrOp $x, $ones),
  (UseOperand $ones),
  [(IsAllOnes $ones)]>;

def OrSimplifyFalseLHS : Pat<
  (StableHLO_OrOp $zero, $x),
  (UseOperand $x),
  [(IsAnyZero $zero)]>;

def OrSimplifyFalseRHS : Pat<
  (StableHLO_OrOp $x, $zero),
  (UseOperand $x),
  [(IsAnyZero $zero)]>;

// XorSimplify
def XorSimplifyFalseLHS : Pat<
  (StableHLO_XorOp $zero, $x),
  (UseOperand $x),
  [(IsAnyZero $zero)]>;

def XorSimplifyFalseRHS : Pat<
  (StableHLO_XorOp $x, $zero),
  (UseOperand $x),
  [(IsAnyZero $zero)]>;

def XorSimplifyTrueLHS : Pat<
  (StableHLO_XorOp $ones, $x),
  (StableHLO_NotOp $x),
  [(IsAllOnes $ones)]>;

def XorSimplifyTrueRHS : Pat<
  (StableHLO_XorOp $x, $ones),
  (StableHLO_NotOp $x),
  [(IsAllOnes $ones)]>;

// ConvertIotaSimplify
def ConvertIotaSimplify : Pat<
  (StableHLO_ConvertOp:$res (StableHLO_IotaOp $dim)),
  (StableHLO_IotaOp $dim),
  [(IsIntegerTensor $res)]>;

// ConjReal
def ConjRealSimplify : Pat<
  (CHLO_ConjOp $x),
  (UseOperand $x),
  [(GuaranteedPurelyReal $x)],
  [],
  (addBenefit 10)>;

// ConjComplexSimplify
def ConjComplexSimplify : Pat<
  (CHLO_ConjOp (StableHLO_ComplexOp $lhs, $rhs)),
  (StableHLO_ComplexOp $lhs, (StableHLO_NegOp $rhs)),
  [(IsConstantOp $rhs)]>;

// DynamicReshapeOpCanon
def DynamicReshapeOpCanon : Pat<
  (StableHLO_DynamicReshapeOp:$res $operand, $shape),
  (StableHLO_ReshapeOp $operand),
  [(IsStaticShapeTensor $res)]>;

// AbsPositiveSimplify
def AbsPositiveSimplify : Pat<
  (StableHLO_AbsOp:$res $x),
  (UseOperand $x),
  [(IsNotComplexTensor $res), (GuaranteedNonNegative $x)]>;

// RemSimplify
def RemSimplify : Pat<
  (StableHLO_RemOp:$res $x, $one),
  (CreateConstOp0 $res),
  [(IsOne $one)]>;

// PowSimplify
def PowSimplifyOneRHS : Pat<
  (StableHLO_PowOp $x, $one),
  (UseOperand $x),
  [(IsOne $one)],
  [],
  (addBenefit 65000)>;

def PowSimplifyZeroRHS : Pat<
  (StableHLO_PowOp:$res $x, $zero),
  (CreateConstOp1 $res),
  [(IsAnyZero $zero)],
  [],
  (addBenefit 65000)>;

def PowSimplifyOneLHS : Pat<
  (StableHLO_PowOp:$res $one, $x),
  (CreateConstOp1 $res),
  [(IsOne $one)],
  [],
  (addBenefit 65000)>;

// Constraints and helper functions for migrated patterns
def ConstDefaultResultAccuracyAttr : ConstantAttr<StableHLO_ResultAccuracyAttr, "::mlir::stablehlo::ResultAccuracyMode::DEFAULT">;

def IsComplexTensor : Constraint<CPred<
  "isa<mlir::ComplexType>(cast<mlir::ShapedType>($0.getType()).getElementType())">,
  "tensor has a complex element type">;

def IsNotConstantOp : Constraint<CPred<"!mlir::matchPattern($0, mlir::m_Constant())">>;

def IsOnlyUsedInOperation : Constraint<CPred<
  "::mlir::enzyme::isOnlyUsedInOperation($0.getDefiningOp(), $1.getDefiningOp())">>;

def IsNegHalf : Constraint<CPred<
  "[] (mlir::Value val) { "
  "  mlir::SplatElementsAttr attr; "
  "  if (!mlir::matchPattern(val, mlir::m_Constant(&attr))) return false; "
  "  auto doubleVal = ::mlir::enzyme::getDoubleFromAttr(attr.getSplatValue<mlir::TypedAttr>()); "
  "  return doubleVal && *doubleVal == -0.5; "
  "}($0)"
>>;

def IsHalf : Constraint<CPred<
  "[] (mlir::Value val) { "
  "  mlir::SplatElementsAttr attr; "
  "  if (!mlir::matchPattern(val, mlir::m_Constant(&attr))) return false; "
  "  auto doubleVal = ::mlir::enzyme::getDoubleFromAttr(attr.getSplatValue<mlir::TypedAttr>()); "
  "  return doubleVal && *doubleVal == 0.5; "
  "}($0)"
>>;

def IsTwo : Constraint<CPred<
  "[] (mlir::Value val) { "
  "  mlir::SplatElementsAttr attr; "
  "  if (!mlir::matchPattern(val, mlir::m_Constant(&attr))) return false; "
  "  auto doubleVal = ::mlir::enzyme::getDoubleFromAttr(attr.getSplatValue<mlir::TypedAttr>()); "
  "  return doubleVal && *doubleVal == 2.0; "
  "}($0)"
>>;

def CreateConstOp2 : NativeCodeCall<"createConstantOpFromScalar($_builder, $_loc, $0.getType(), 2.0)">;
def CreateConstOp3 : NativeCodeCall<"createConstantOpFromScalar($_builder, $_loc, $0.getType(), 3.0)">;
def CreateConstOpNeg2 : NativeCodeCall<"createConstantOpFromScalar($_builder, $_loc, $0.getType(), -2.0)">;

def AnyOperandIsConstant : Constraint<CPred<"::mlir::enzyme::anyOperandIsConstant($0.getDefiningOp())">>;
def NotAllOperandsAreConstant : Constraint<CPred<"!::mlir::enzyme::allOperandsAreConstant($0.getDefiningOp())">>;

def IsEqual : Constraint<CPred<"$0 == $1">>;

// SquareAbsSimplify
def SquareAbsSimplify : Pat<
  (StableHLO_MulOp (StableHLO_AbsOp:$absLhs $x), (StableHLO_AbsOp:$absRhs $y)),
  (StableHLO_MulOp $x, $x),
  [(IsNotComplexTensor $x), (IsEqual $absLhs, $absRhs)]>;

def SquareAbsSimplifyComplex : Pat<
  (StableHLO_MulOp:$res (StableHLO_AbsOp:$absLhs $x), (StableHLO_AbsOp:$absRhs $y)),
  (StableHLO_AddOp (StableHLO_MulOp (StableHLO_RealOp $x), (StableHLO_RealOp $x)), (StableHLO_MulOp (StableHLO_ImagOp $x), (StableHLO_ImagOp $x))),
  [(IsComplexTensor $x), (IsEqual $absLhs, $absRhs), (IsOnlyUsedInOperation $absLhs, $res), (IsOnlyUsedInOperation $absRhs, $res)]>;

// DivSimplify simple patterns
// DivSimplify simple patterns
def DivSimplifyOneRHS : Pat<
  (StableHLO_DivOp $x, $one),
  (UseOperand $x),
  [(IsOne $one)],
  [],
  (addBenefit 65000)>;

def DivSimplifyNegOneRHS : Pat<
  (StableHLO_DivOp $x, $negone),
  (StableHLO_NegOp $x),
  [(IsNegOne $negone)],
  [],
  (addBenefit 65000)>;

// PowSimplify extra patterns
def PowSimplifyNegOneRHS : Pat<
  (StableHLO_PowOp:$res $x, $negone),
  (StableHLO_DivOp (CreateConstOp1 $res), $x),
  [(IsNegOne $negone), (IsFloatTensor $res)],
  [],
  (addBenefit 65000)>;

def PowSimplifyNegHalfRHS : Pat<
  (StableHLO_PowOp:$res $x, $neghalf),
  (StableHLO_RsqrtOp $x, ConstDefaultResultAccuracyAttr),
  [(IsNegHalf $neghalf), (IsFloatTensor $res)],
  [],
  (addBenefit 65000)>;

def PowSimplifyHalfRHS : Pat<
  (StableHLO_PowOp:$res $x, $half),
  (StableHLO_SqrtOp $x, ConstDefaultResultAccuracyAttr),
  [(IsHalf $half), (IsFloatTensor $res)],
  [],
  (addBenefit 65000)>;

def PowSimplifyTwoRHS : Pat<
  (StableHLO_PowOp:$res $x, $two),
  (StableHLO_MulOp $x, $x),
  [(IsTwo $two), (IsFloatTensor $res)],
  [],
  (addBenefit 65000)>;

// LogSimplify patterns
def LogExpSimplify : Pat<
  (StableHLO_LogOp (StableHLO_ExpOp $x, $exp_acc), $log_acc),
  (UseOperand $x),
  [],
  [],
  (addBenefit 65000)>;

def LogPowSimplify : Pat<
  (StableHLO_LogOp (StableHLO_PowOp $x, $y), $log_acc),
  (StableHLO_MulOp $y, (StableHLO_LogOp $x, $log_acc)),
  [],
  [],
  (addBenefit 65000)>;

def LogMulIdenticalSimplify : Pat<
  (StableHLO_LogOp:$res (StableHLO_MulOp $x, $x), $log_acc),
  (StableHLO_MulOp (CreateConstOp2 $res), (StableHLO_LogOp $x, $log_acc)),
  [],
  [],
  (addBenefit 65000)>;

def LogMulConstSimplify : Pat<
  (StableHLO_LogOp (StableHLO_MulOp:$mul $a, $b), $log_acc),
  (StableHLO_AddOp (StableHLO_LogOp $a, $log_acc), (StableHLO_LogOp $b, $log_acc)),
  [(AnyOperandIsConstant $mul), (NotAllOperandsAreConstant $mul)],
  [],
  (addBenefit 65000)>;

def LogAddIdenticalSimplify : Pat<
  (StableHLO_LogOp:$res (StableHLO_AddOp $x, $x), $log_acc),
  (StableHLO_AddOp (StableHLO_LogOp (CreateConstOp2 $res), $log_acc), (StableHLO_LogOp $x, $log_acc)),
  [],
  [],
  (addBenefit 65000)>;

def LogAddOneRHSSimplify : Pat<
  (StableHLO_LogOp (StableHLO_AddOp $x, $one), $log_acc),
  (StableHLO_Log1pOp $x, $log_acc),
  [(IsOne $one)],
  [],
  (addBenefit 65000)>;

def LogAddOneLHSSimplify : Pat<
  (StableHLO_LogOp (StableHLO_AddOp $one, $x), $log_acc),
  (StableHLO_Log1pOp $x, $log_acc),
  [(IsOne $one)],
  [],
  (addBenefit 65000)>;

def LogDivConstSimplify : Pat<
  (StableHLO_LogOp (StableHLO_DivOp:$div $a, $b), $log_acc),
  (StableHLO_SubtractOp (StableHLO_LogOp $a, $log_acc), (StableHLO_LogOp $b, $log_acc)),
  [(AnyOperandIsConstant $div), (NotAllOperandsAreConstant $div)],
  [],
  (addBenefit 65000)>;

def LogSqrtSimplify : Pat<
  (StableHLO_LogOp:$res (StableHLO_SqrtOp:$sqrt $x, $sqrt_acc), $log_acc),
  (StableHLO_DivOp (StableHLO_LogOp $x, $log_acc), (CreateConstOp2 $res)),
  [(IsOnlyUsedInOperation $sqrt, $res)],
  [],
  (addBenefit 65000)>;

def LogCbrtSimplify : Pat<
  (StableHLO_LogOp:$res (StableHLO_CbrtOp:$cbrt $x, $cbrt_acc), $log_acc),
  (StableHLO_DivOp (StableHLO_LogOp $x, $log_acc), (CreateConstOp3 $res)),
  [(IsOnlyUsedInOperation $cbrt, $res)],
  [],
  (addBenefit 65000)>;

def LogRsqrtSimplify : Pat<
  (StableHLO_LogOp:$res (StableHLO_RsqrtOp:$rsqrt $x, $rsqrt_acc), $log_acc),
  (StableHLO_DivOp (StableHLO_LogOp $x, $log_acc), (CreateConstOpNeg2 $res)),
  [(IsOnlyUsedInOperation $rsqrt, $res)],
  [],
  (addBenefit 65000)>;

// NegMulConstSimplify
def NegMulConstLHSSimplify : Pat<
  (StableHLO_NegOp (StableHLO_MulOp $lhs, $rhs)),
  (StableHLO_MulOp (StableHLO_NegOp $lhs), $rhs),
  [(IsConstantOp $lhs), (IsNotConstantOp $rhs)],
  [],
  (addBenefit 65000)>;

def NegMulConstRHSSimplify : Pat<
  (StableHLO_NegOp (StableHLO_MulOp $lhs, $rhs)),
  (StableHLO_MulOp $lhs, (StableHLO_NegOp $rhs)),
  [(IsNotConstantOp $lhs), (IsConstantOp $rhs)],
  [],
  (addBenefit 65000)>;

// NegDivConstSimplify
def NegDivConstLHSSimplify : Pat<
  (StableHLO_NegOp (StableHLO_DivOp $lhs, $rhs)),
  (StableHLO_DivOp (StableHLO_NegOp $lhs), $rhs),
  [(IsConstantOp $lhs), (IsNotConstantOp $rhs)],
  [],
  (addBenefit 65000)>;

def NegDivConstRHSSimplify : Pat<
  (StableHLO_NegOp (StableHLO_DivOp $lhs, $rhs)),
  (StableHLO_DivOp $lhs, (StableHLO_NegOp $rhs)),
  [(IsNotConstantOp $lhs), (IsConstantOp $rhs)],
  [],
  (addBenefit 65000)>;

// DivConstToMulReciprocal
def IsDenseElementsAttr : Constraint<CPred<"llvm::isa<mlir::DenseElementsAttr>($0)">>;

def CreateReciprocalConstOp : NativeCodeCall<
  "::mlir::enzyme::createReciprocalConstantOp($_builder, $_loc, $0, $1.getType())"
>;

def DivConstToMulReciprocal : Pat<
  (StableHLO_DivOp:$op $x, (StableHLO_ConstantOp $value)),
  (StableHLO_MulOp $x, (CreateReciprocalConstOp $value, $op)),
  [(IsFloatTensor $op), (IsNotConstantOp $x), (IsDenseElementsAttr $value)]>;

