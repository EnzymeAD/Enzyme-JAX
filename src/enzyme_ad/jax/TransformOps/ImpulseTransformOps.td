//===- ImpulseTransformOps.td - Impulse (probprog) pattern descriptors ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"

class ImpulsePatternOp<string mnemonic, list<Trait> traits = []>
    : Op<Transform_Dialect,
         "apply_patterns.impulse." # mnemonic,
          // For some reason, inherited methods are not getting declared...
         !listconcat(
            [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>],
            traits)> {
  let arguments = (ins OptionalAttr<I64Attr>:$benefit);
  list<string> patterns = [];
  let assemblyFormat = "attr-dict";
}

def SicmCholeskyScalePatterns : ImpulsePatternOp<
    "sicm_cholesky_scale"> {
  let patterns = ["CholeskyScaleFactorizationHLO"];
}
def SicmCholeskyOuterProductScalePatterns : ImpulsePatternOp<
    "sicm_cholesky_outer_product_scale"> {
  let patterns = ["CholeskyOuterProductScaleHLO"];
}
def SicmCholeskyEigenLiftPatterns : ImpulsePatternOp<
    "sicm_cholesky_eigen_lift"> {
  let patterns = ["CholeskyEigenLiftHLO"];
}
def SicmDotGeneralScalePatterns : ImpulsePatternOp<
    "sicm_dot_general_scale"> {
  let patterns = ["DotGeneralScaleFactorizationHLO"];
}
def SicmTriangularSolveScalePatterns : ImpulsePatternOp<
    "sicm_triangular_solve_scale"> {
  let patterns = ["TriangularSolveScaleFactorizationHLO"];
}
def SicmLogMultiplyPatterns : ImpulsePatternOp<
    "sicm_log_multiply"> {
  let patterns = ["LogMultiplyDistributionHLO"];
}
def SicmDotAbsorbDiagMulPatterns : ImpulsePatternOp<
    "sicm_dot_absorb_diag_mul"> {
  let patterns = ["DotAbsorbDiagMulHLO"];
}
def SicmDotAbsorbFFTPatterns : ImpulsePatternOp<
    "sicm_dot_absorb_fft"> {
  let patterns = ["DotAbsorbFFTHLO"];
}
def SicmDotAbsorbTransposePatterns : ImpulsePatternOp<
    "sicm_dot_absorb_transpose"> {
  let patterns = ["DotAbsorbTransposeHLO"];
}
def SicmDotAbsorbScatterPatterns : ImpulsePatternOp<
    "sicm_dot_absorb_scatter"> {
  let patterns = ["DotAbsorbScatterHLO"];
}
def SicmPowerScalePatterns : ImpulsePatternOp<
    "sicm_power_scale"> {
  let patterns = ["PowerScaleFactorizationHLO"];
}
def SicmSqrtScalePatterns : ImpulsePatternOp<
    "sicm_sqrt_scale"> {
  let patterns = ["SqrtScaleFactorizationHLO"];
}
def SicmRsqrtScalePatterns : ImpulsePatternOp<
    "sicm_rsqrt_scale"> {
  let patterns = ["RsqrtScaleFactorizationHLO"];
}
def SicmAbsScalePatterns : ImpulsePatternOp<
    "sicm_abs_scale"> {
  let patterns = ["AbsScaleFactorizationHLO"];
}
def SicmNegatePatterns : ImpulsePatternOp<
    "sicm_negate"> {
  let patterns = ["NegateDistributionHLO"];
}
def SicmExpAddPatterns : ImpulsePatternOp<
    "sicm_exp_add"> {
  let patterns = ["ExpAddFactorizationHLO"];
}
def SicmDivideScalePatterns : ImpulsePatternOp<
    "sicm_divide_scale"> {
  let patterns = ["DivideScaleFactorizationHLO"];
}
// The Gram variant below is opt-in and deliberately left out of `sicm`.
def SicmGaussianSufficientStatsPatterns : ImpulsePatternOp<
    "sicm_gaussian_sufficient_stats"> {
  let patterns = ["GaussianSufficientStatsHLO",
                  "GaussianSufficientStatsReduceHLO"];
}
def SicmGaussianSufficientStatsGramPatterns : ImpulsePatternOp<
    "sicm_gaussian_sufficient_stats_gram"> {
  let patterns = ["GaussianSufficientStatsGramHLO",
                  "GaussianSufficientStatsGramReduceHLO"];
}

def SicmPatterns : ImpulsePatternOp<"sicm"> {
  let patterns = [
    "CholeskyScaleFactorizationHLO",
    "CholeskyOuterProductScaleHLO",
    "CholeskyEigenLiftHLO",
    "DotGeneralScaleFactorizationHLO",
    "TriangularSolveScaleFactorizationHLO",
    "LogMultiplyDistributionHLO",
    "DotAbsorbDiagMulHLO",
    "DotAbsorbFFTHLO",
    "DotAbsorbTransposeHLO",
    "DotAbsorbScatterHLO",
    "PowerScaleFactorizationHLO",
    "SqrtScaleFactorizationHLO",
    "RsqrtScaleFactorizationHLO",
    "AbsScaleFactorizationHLO",
    "NegateDistributionHLO",
    "ExpAddFactorizationHLO",
    "DivideScaleFactorizationHLO",
    "GaussianSufficientStatsHLO",
    "GaussianSufficientStatsReduceHLO",
  ];
}
