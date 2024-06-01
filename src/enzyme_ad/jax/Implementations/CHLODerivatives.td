include "src/enzyme_ad/jax/Implementations/Common.td"

class HLODerivative<string opName_, dag patternToMatch, list<dag> resultOps, dag forwardOps=(ForwardFromSummedReverse)> : MLIRDerivative<"chlo", opName_, patternToMatch, resultOps, forwardOps>;

class HLOInst<string m, string postopt=""> : Inst<m, "chlo", postopt>;

class HLOMemoryIdentityOp<string opName_, list<int> ptrargs_, list<int> storedargs_ = [], dag patternToMatch=(Unimplemented), list<dag> reverse_ = []>  : MemoryIdentityOp<"chlo", opName_, ptrargs_, storedargs_, patternToMatch, reverse_>;

class HLOReadOnlyIdentityOp<string opName_, list<int> ptrargs_ = [0], dag patternToMatch=(Unimplemented), list<dag> reverse_ = []> : ReadOnlyIdentityOp<"chlo", opName_, ptrargs_, patternToMatch, reverse_>;

class HLOControlFlowOp<string opName_, string impl_> : ControlFlowOp<"chlo", opName_, impl_>;

class HLOConstantFP<string m> : ConstantFP<m, "chlo", "ConstantOp", "mlir::ElementsAttr">;

class HLORegionTerminatorOp<string m> : RegionTerminatorOp<"chlo", m>;

class HLOInactiveOp<string m> : InactiveOp<"chlo", m>;

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
// def Constant : HLOInst<"ConstantOp">;
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
  (Neg (CheckedDiv (DiffeRet), (Sqrt (Sub (HLOConstantFP<"1">), (Pow $x, (HLOConstantFP<"2">))))))
]>;

def : HLODerivative<"AcoshOp", (Op $x), [
  (CheckedDiv (DiffeRet), (CheckedMul (Sqrt (Sub (Pow $x, (HLOConstantFP<"2">)), (HLOConstantFP<"1">))), (Sqrt (Add (Pow $x, (HLOConstantFP<"2">)), (HLOConstantFP<"1">)))))
]>;

def : HLODerivative<"AsinOp", (Op $x), [
  (CheckedDiv (DiffeRet), (Sqrt (Sub (HLOConstantFP<"1">), (Pow $x, (HLOConstantFP<"2">)))))
]>;

def : HLODerivative<"AsinhOp", (Op $x), [
  (CheckedDiv (DiffeRet), (Sqrt (Add (HLOConstantFP<"1">), (Pow $x, (HLOConstantFP<"2">)))))
]>;

def : HLODerivative<"AtanOp", (Op $x), [
  (CheckedDiv (DiffeRet), (Add (Pow $x, (HLOConstantFP<"2">)), (HLOConstantFP<"1">)))
]>;

def : HLODerivative<"AtanhOp", (Op $x), [
  (CheckedDiv (DiffeRet), (Sub (Pow $x, (HLOConstantFP<"2">)), (HLOConstantFP<"1">)))
]>;

def : HLODerivative<"CoshOp", (Op $x), [(CheckedMul (DiffeRet), (Sinh $x))]>;

def : HLOInactiveOp<"IsInfOp">;

def : HLOInactiveOp<"IsNegInfOp">;

def : HLOInactiveOp<"IsPosInfOp">;

def : HLODerivative<"SinhOp", (Op $x), [(CheckedMul (DiffeRet), (Cosh $x))]>;

def : HLODerivative<"TanOp", (Op $x), [
  (CheckedDiv (DiffeRet), (Pow (Cos $x), (HLOConstantFP<"2">)), (HLOConstantFP<"1">))
]>;
