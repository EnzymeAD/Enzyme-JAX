include "src/enzyme_ad/jax/Implementations/Common.td"

class HLODerivative<string opName_, dag patternToMatch, list<dag> resultOps, dag forwardOps=(ForwardFromSummedReverse)> : MLIRDerivative<"enzymexla", opName_, patternToMatch, resultOps, forwardOps>;

class HLOInst<string m, string postopt="", string preopt=""> : Inst<m, "enzymexla", postopt, preopt>;

class HLOMemoryIdentityOp<string opName_, list<int> ptrargs_, list<int> storedargs_ = [], dag patternToMatch=(Unimplemented), list<dag> reverse_ = []>  : MemoryIdentityOp<"enzymexla", opName_, ptrargs_, storedargs_, patternToMatch, reverse_>;

class HLOReadOnlyIdentityOp<string opName_, list<int> ptrargs_ = [0], dag patternToMatch=(Unimplemented), list<dag> reverse_ = []> : ReadOnlyIdentityOp<"enzymexla", opName_, ptrargs_, patternToMatch, reverse_>;

class HLOInactiveOp<string m> : InactiveOp<"enzymexla", m>;

// EnzymeXLA specification
def Gelu : HLOInst<"GeluOp">;
def Relu : HLOInst<"ReluOp">;
def LU : HLOInst<"LUFactorizationOp">;
def SVD : HLOInst<"SVDFactorizationOp">;

// Helpers
class SHLOConstantFP<string m> : ConstantFP<m, "stablehlo", "ConstantOp", "mlir::ElementsAttr">;

class SHLORegionTerminatorOp<string m> : RegionTerminatorOp<"stablehlo", m>;

def Add : Inst<"AddOp", "stablehlo">;
def Compare : Inst<"CompareOp", "stablehlo">;
def Div : Inst<"DivOp", "stablehlo">;
def Erf : Inst<"ErfOp", "chlo">;
def Exp : Inst<"ExpOp", "stablehlo">;
def Mul : Inst<"MulOp", "stablehlo">;
def Neg : Inst<"NegOp", "stablehlo">;
def Pow : Inst<"PowOp", "stablehlo">;
def Rsqrt : Inst<"RsqrtOp", "stablehlo">;
def Select : Inst<"SelectOp", "stablehlo">;
def Sub : Inst<"SubtractOp", "stablehlo">;
def Sqrt : Inst<"SqrtOp", "stablehlo">;
def Tanh : Inst<"TanhOp", "stablehlo">;

def M_PI : SubRoutine<(Op $x),
                        (SHLOConstantFP<"3.141592653589793238462643383279502884197169399375105820974944592307816406286198"> $x)>;

def Sqrt2M_PI : SubRoutine<(Op $x),
                  (SHLOConstantFP<"2.506628274631000502415765284811045253006986740609938316629923576342293654607848"> $x)>;

/// sqrt(2 / pi)
def RsqrtM_PI_2 : SubRoutine<(Op $x),
                    (SHLOConstantFP<"0.7978845608028653558798921198687637369517172623298693153318516593413158517986017"> $x)>;

def EQ : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, "ComparisonDirection::EQ">;
def GE : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, "ComparisonDirection::GE">;
def LT : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, "ComparisonDirection::LT">;
def GT : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, "ComparisonDirection::GT">;

def TanhDerivative : SubRoutine<(Op $x),
                       (Sub (SHLOConstantFP<"1"> $x), (Mul (Tanh $x), (Tanh $x)))>;

def Pow3 : SubRoutine<(Op $x), (Mul (Mul $x, $x), $x)>;

// Derivative rules
/// Relu
def : HLODerivative<"ReluOp",
  (Op $x),
  [
    (Select (Compare $x, (SHLOConstantFP<"0">), (LT)), (SHLOConstantFP<"0"> $x), (DiffeRet)),
  ],
  (
    Select
    (Compare $x, (SHLOConstantFP<"0">), (LT)),
    (SHLOConstantFP<"0"> $x),
    (SelectIfActive $x, (Shadow $x), (SHLOConstantFP<"0"> $x))
  )
>;

/// Gelu
def SelectIfGeluNone : StaticSelect<"cast<GeluOp>(op).getGeluApproximation() == GeluApproximation::NONE">;
def SelectIfGeluTanh : StaticSelect<"cast<GeluOp>(op).getGeluApproximation() == GeluApproximation::TANH">;
def SelectIfGeluSigmoid : StaticSelect<"cast<GeluOp>(op).getGeluApproximation() == GeluApproximation::SIGMOID">;


def : HLODerivative<"GeluOp",
  (Op $x),
  [
    (
      Mul
      (DiffeRet),
      (
        SelectIfGeluNone
        (
          // 0.5 * (1 + erf(x / sqrt(2))) + exp(-x * x / 2) / sqrt(2pi)
          Add
          (
            Mul
            (SHLOConstantFP<"0.5"> $x),
            (Add (SHLOConstantFP<"1"> $x), (Erf (Div $x, (Sqrt (SHLOConstantFP<"2"> $x)))))
          ),
          (Div (Exp (Mul (Mul $x, $x), (SHLOConstantFP<"-0.5"> $x))), (Sqrt2M_PI $x))
        ),
        (
          SelectIfGeluTanh
          (
            // 0.5 * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)) +
            // 0.5 * x * dtanh(sqrt(2 / pi) * (x + 0.044715 * x^3)) * sqrt(2 / pi) * (1 + 0.044715 * 3 * x^2))
            Mul
            (SHLOConstantFP<"0.5"> $x),
            (
              Add
              (SHLOConstantFP<"1"> $x),
              (
                Add
                (
                  Tanh
                  (Mul (RsqrtM_PI_2 $x), (Add $x, (Mul (SHLOConstantFP<"0.044715"> $x), (Pow3 $x))))
                ),
                (
                  Mul
                  $x,
                  (
                    Mul
                    (
                      TanhDerivative
                      (Mul (RsqrtM_PI_2 $x), (Add $x, (Mul (SHLOConstantFP<"0.044715"> $x), (Pow3 $x))))
                    ),
                    (
                      Mul
                      (RsqrtM_PI_2 $x),
                      (Add (SHLOConstantFP<"1"> $x), (Mul (SHLOConstantFP<"0.134145"> $x), (Mul $x, $x)))
                    )
                  )
                )
              )
            )
          ),
          (
            SHLOConstantFP<"0"> $x // TODO: implement this
          )
        )
      )
    )
  ]
>;
