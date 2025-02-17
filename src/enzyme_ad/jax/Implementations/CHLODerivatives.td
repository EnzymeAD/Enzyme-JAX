include "src/enzyme_ad/jax/Implementations/Common.td"

class HLODerivative<string opName_, dag patternToMatch, list<dag> resultOps, dag forwardOps=(ForwardFromSummedReverse)> : MLIRDerivative<"chlo", opName_, patternToMatch, resultOps, forwardOps>;

class HLOInst<string m, string postopt="", string preopt=""> : Inst<m, "chlo", postopt, preopt>;

class HLOMemoryIdentityOp<string opName_, list<int> ptrargs_, list<int> storedargs_ = [], dag patternToMatch=(Unimplemented), list<dag> reverse_ = []>  : MemoryIdentityOp<"chlo", opName_, ptrargs_, storedargs_, patternToMatch, reverse_>;

class HLOReadOnlyIdentityOp<string opName_, list<int> ptrargs_ = [0], dag patternToMatch=(Unimplemented), list<dag> reverse_ = []> : ReadOnlyIdentityOp<"chlo", opName_, ptrargs_, patternToMatch, reverse_>;

class HLOControlFlowOp<string opName_, string impl_> : ControlFlowOp<"chlo", opName_, impl_>;

class HLOConstantFP<string m> : ConstantFP<m, "chlo", "ConstantOp", "mlir::ElementsAttr">;

class HLORegionTerminatorOp<string m> : RegionTerminatorOp<"chlo", m>;

class HLOInactiveOp<string m> : InactiveOp<"chlo", m>;

// Required operations from the StableHLO dialect
def Add : Inst<"AddOp", "stablehlo">;
def Sub : Inst<"SubtractOp", "stablehlo">;
def Mul : Inst<"MulOp", "stablehlo">;
def Div : Inst<"DivOp", "stablehlo">;
def Neg : Inst<"NegOp", "stablehlo">;
def Sqrt : Inst<"SqrtOp", "stablehlo">;
def Pow : Inst<"PowOp", "stablehlo">;
def Cos : Inst<"CosineOp", "stablehlo">;
def Sin : Inst<"SineOp", "stablehlo">;

// Operations
/// CHLO - binary elementwise operations
def BroadcastAdd : HLOInst<"BroadcastAddOp">;
def BroadcastAtan2 : HLOInst<"BroadcastAtan2Op">;
def BroadcastDiv : HLOInst<"BroadcastDivOp">;
def BroadcastMax : HLOInst<"BroadcastMaxOp">;
def BroadcastMin : HLOInst<"BroadcastMinOp">;
def BroadcastMul : HLOInst<"BroadcastMulOp">;
def BroadcastNextAfter : HLOInst<"BroadcastNextAfterOp">;
def BroadcastPolygamma : HLOInst<"BroadcastPolygammaOp">;
def BroadcastPow : HLOInst<"BroadcastPowOp">;
def BroadcastRem : HLOInst<"BroadcastRemOp">;
def BroadcastShiftLeft : HLOInst<"BroadcastShiftLeftOp">;
def BroadcastShiftRightArithmetic : HLOInst<"BroadcastShiftRightArithmeticOp">;
def BroadcastShiftRightLogical : HLOInst<"BroadcastShiftRightLogicalOp">;
def BroadcastSub : HLOInst<"BroadcastSubOp">;
def BroadcastZeta : HLOInst<"BroadcastZetaOp">;

/// CHLO - binary logical elementwise operations
def BroadcastAnd : HLOInst<"BroadcastAndOp">;
def BroadcastOr : HLOInst<"BroadcastOrOp">;
def BroadcastXor : HLOInst<"BroadcastXorOp">;

/// CHLO - non-broadcasting binary operations
def NextAfter : HLOInst<"NextAfterOp">;
def Polygamma : HLOInst<"PolygammaOp">;
def Zeta : HLOInst<"ZetaOp">;

/// CHLO - complex broadcasting operation
def BroadcastComplex : HLOInst<"BroadcastComplexOp">;

/// CHLO - unary elementwise operations
def Acos : HLOInst<"AcosOp">;
def Acosh : HLOInst<"AcoshOp">;
def Asin : HLOInst<"AsinOp">;
def Asinh : HLOInst<"AsinhOp">;
def Atan : HLOInst<"AtanOp">;
def Atanh : HLOInst<"AtanhOp">;
def BesselI1e : HLOInst<"BesselI1eOp">;
def Conj : HLOInst<"ConjOp">;
def Cosh : HLOInst<"CoshOp">;
def Sinh : HLOInst<"SinhOp">;
def Tan : HLOInst<"TanOp">;
def Constant : HLOInst<"ConstantOp">;
def ConstantLike : HLOInst<"ConstantLikeOp">;
def Digamma : HLOInst<"DigammaOp">;
def Erf : HLOInst<"ErfOp">;
def ErfInv : HLOInst<"ErfInvOp">;
def Erfc : HLOInst<"ErfcOp">;
def IsInf : HLOInst<"IsInfOp">;
def IsNegInf : HLOInst<"IsNegInfOp">;
def IsPosInf : HLOInst<"IsPosInfOp">;
def Lgamma : HLOInst<"LgammaOp">;

/// CHLO - broadcasting compare operation
def BroadcastCompare : HLOInst<"BroadcastCompareOp">;

/// CHLO - broadcasting select operation
def BroadcastSelect : HLOInst<"BroadcastSelectOp">;

/// CHLO - miscelaneous operations
def TopK : HLOInst<"TopKOp">;

// Derivative rules
def : HLODerivative<"AcosOp", (Op $x), [
  (Neg (Div (DiffeRet), (Sqrt (Sub (HLOConstantFP<"1"> $x), (Mul $x, $x)))))
]>;

def : HLODerivative<"AcoshOp", (Op $x), [
  (Div (DiffeRet), (Mul (Sqrt (Sub (Mul $x, $x), (HLOConstantFP<"1"> $x))), (Sqrt (Add (Mul $x, $x), (HLOConstantFP<"1"> $x)))))
]>;

def : HLODerivative<"AsinOp", (Op $x), [
  (Div (DiffeRet), (Sqrt (Sub (HLOConstantFP<"1"> $x), (Mul $x, $x))))
]>;

def : HLODerivative<"AsinhOp", (Op $x), [
  (Div (DiffeRet), (Sqrt (Add (HLOConstantFP<"1"> $x), (Mul $x, $x))))
]>;

def : HLODerivative<"AtanOp", (Op $x), [
  (Div (DiffeRet), (Add (Mul $x, $x), (HLOConstantFP<"1"> $x)))
]>;

def : HLODerivative<"AtanhOp", (Op $x), [
  (Div (DiffeRet), (Sub (Mul $x, $x), (HLOConstantFP<"1"> $x)))
]>;

def : HLODerivative<"ConjOp", (Op $z), [(Conj (DiffeRet))], (Conj (Shadow $z))>;

def : HLODerivative<"CoshOp", (Op $x), [(Mul (DiffeRet), (Sinh $x))]>;

def : HLODerivative<"DigammaOp", (Op $x), 
                    [(Mul (DiffeRet), (Polygamma $x, (HLOConstantFP<"2"> $x)))],
                    (Mul (Shadow $x), (Polygamma $x, (HLOConstantFP<"2"> $x)))
>;

def : HLOInactiveOp<"IsInfOp">;

def : HLOInactiveOp<"IsNegInfOp">;

def : HLOInactiveOp<"IsPosInfOp">;

def : HLODerivative<"PolygammaOp", (Op $x, $n),
                    [
                      (Mul (DiffeRet), (Polygamma $x, (Add $n, (HLOConstantFP<"1"> $n)))),
                      (AssertingInactiveArg)
                    ]
                    >;

def : HLODerivative<"SinhOp", (Op $x), [(Mul (DiffeRet), (Cosh $x))]>;

def : HLODerivative<"TanOp", (Op $x), [
  (Div (DiffeRet), (Mul (Cos $x), (Cos $x)))
]>;
