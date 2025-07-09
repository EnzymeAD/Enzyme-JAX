include "src/enzyme_ad/jax/Implementations/Common.td"

class HLODerivative<string opName_, dag patternToMatch, list<dag> resultOps, dag forwardOps=(ForwardFromSummedReverse)> : MLIRDerivative<"enzymexla", opName_, patternToMatch, resultOps, forwardOps>;

class HLOInst<string m, string postopt="", string preopt=""> : Inst<m, "enzymexla", postopt, preopt>;

class SHLOInst<string m, string postopt="", string preopt=""> : Inst<m, "stablehlo", postopt, preopt>;

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

def Select : SHLOInst<"SelectOp">;
def Compare : SHLOInst<"CompareOp">;

def EQ : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, "ComparisonDirection::EQ">;
def GE : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, "ComparisonDirection::GE">;
def LT : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, "ComparisonDirection::LT">;
def GT : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, "ComparisonDirection::GT">;

// Derivative rules
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
