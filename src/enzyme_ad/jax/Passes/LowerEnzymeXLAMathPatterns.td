//===- LowerEnzymeXLAMathPatterns.td - EnzymeXLA pass tablegen macros ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
include "stablehlo/dialect/StablehloOps.td"
include "stablehlo/dialect/ChloOps.td"
include "src/enzyme_ad/jax/Dialect/EnzymeXLAOps.td"

def CreateConstOp0 : NativeCodeCall<"createConstantOpFromScalar($_builder, $_loc, $0.getType(), 0.0)">;
def CreateConstOp1 : NativeCodeCall<"createConstantOpFromScalar($_builder, $_loc, $0.getType(), 1.0)">;
def CreatePiOp : NativeCodeCall<"createConstantOpFromScalar($_builder, $_loc, $0.getType(), M_PI)">;
def CreateConstFloatType : NativeCodeCall<"stablehlo::ComparisonTypeAttr::get($_builder.getContext(), stablehlo::ComparisonType::FLOAT)">;
def CreateNanOp : NativeCodeCall<"createConstantOpFromScalar($_builder, $_loc, $0.getType(), std::numeric_limits<double>::quiet_NaN())">;
def CreateHalfOp : NativeCodeCall<"createConstantOpFromScalar($_builder, $_loc, $0.getType(), 0.5)">;
def CreateInvSqrt2Op : NativeCodeCall<"createConstantOpFromScalar($_builder, $_loc, $0.getType(), 1.0 / sqrt(2.0))">;
def CreateSqrt2OverPiOp : NativeCodeCall<"createConstantOpFromScalar($_builder, $_loc, $0.getType(), sqrt(2.0 / M_PI))">;
def CreateSqrt8OverPiOp : NativeCodeCall<"createConstantOpFromScalar($_builder, $_loc, $0.getType(), sqrt(8.0 / M_PI))">;
def CreateGeluCoeffOp : NativeCodeCall<"createConstantOpFromScalar($_builder, $_loc, $0.getType(), 0.044715)">;

class StableHLO_ComparisonDirectionValue<string enumStr> :
  ConstantAttr<StableHLO_ComparisonDirectionAttr, "::mlir::stablehlo::ComparisonDirection::" # enumStr>;

def ConstDefaultResultAccuracyAttr :
  ConstantAttr<StableHLO_ResultAccuracyAttr, "::mlir::stablehlo::ResultAccuracyMode::DEFAULT">;

def GeluApproximationNONE : ConstantAttr<EnzymeXLA_GeluApproximationAttr, "::mlir::enzymexla::GeluApproximation::NONE">;
def GeluApproximationTANH : ConstantAttr<EnzymeXLA_GeluApproximationAttr, "::mlir::enzymexla::GeluApproximation::TANH">;
def GeluApproximationSIGMOID : ConstantAttr<EnzymeXLA_GeluApproximationAttr, "::mlir::enzymexla::GeluApproximation::SIGMOID">;

def : Pat<(ReluOp:$op $input),
          (StableHLO_MaxOp $input, (CreateConstOp0 $input))>;

// Gelu Approximation: NONE
def : Pat<(GeluOp:$op $x, GeluApproximationNONE),
  (StableHLO_MulOp $x,
    (StableHLO_MulOp
      (CreateHalfOp $op),
      (StableHLO_AddOp
        (CreateConstOp1 $op),
        (CHLO_ErfOp (StableHLO_MulOp $x, (CreateInvSqrt2Op $op)))
      )
    )
  )
>;

// Gelu Approximation: TANH
def : Pat<(GeluOp:$op $x, GeluApproximationTANH),
  (StableHLO_MulOp $x,
    (StableHLO_MulOp
      (CreateHalfOp $op),
      (StableHLO_AddOp
        (CreateConstOp1 $op),
        (StableHLO_TanhOp
          (StableHLO_MulOp
            (CreateSqrt2OverPiOp $op),
            (StableHLO_AddOp
              $x,
              (StableHLO_MulOp
                (CreateGeluCoeffOp $op),
                (StableHLO_MulOp $x, (StableHLO_MulOp $x, $x))
              )
            )
          ),
          ConstDefaultResultAccuracyAttr
        )
      )
    )
  )
>;

// Gelu Approximation: SIGMOID
def : Pat<(GeluOp:$op $x, GeluApproximationSIGMOID),
  (StableHLO_MulOp $x,
    (StableHLO_LogisticOp
      (StableHLO_MulOp
        (CreateSqrt8OverPiOp $op),
        (StableHLO_MulOp
          $x,
          (StableHLO_AddOp
            (CreateConstOp1 $op),
            (StableHLO_MulOp
              (CreateGeluCoeffOp $op),
              (StableHLO_MulOp $x, $x)
            )
          )
        )
      ),
      ConstDefaultResultAccuracyAttr
    )
  )
>;

// Softplus
def : Pat<(SoftplusOp:$op $x),
  (StableHLO_SelectOp
    (StableHLO_CompareOp
      $x, $x,
      StableHLO_ComparisonDirectionValue<"NE">,
      (CreateConstFloatType)
    ),
    $x,
    (StableHLO_AddOp
      (StableHLO_MaxOp $x, (CreateConstOp0 $op)),
      (StableHLO_Log1pOp
        (StableHLO_ExpOp (StableHLO_NegOp (StableHLO_AbsOp $x)), ConstDefaultResultAccuracyAttr),
        ConstDefaultResultAccuracyAttr
      )
    )
  )
>;

def : Pat<(LGammaOp $x), (CHLO_LgammaOp $x)>;

def : Pat<(TGammaOp:$op $x),
  (StableHLO_SelectOp
    (StableHLO_CompareOp
      $x,
      (CreateConstOp0 $op),
      StableHLO_ComparisonDirectionValue<"LT">,
      (CreateConstFloatType)
    ),
    (CreateNanOp $op),
    (StableHLO_ExpOp (CHLO_LgammaOp $x), ConstDefaultResultAccuracyAttr)
  )
>;
