include "Common.td"

class HLODerivative<string opName_, dag patternToMatch, list<dag> resultOps> : MLIRDerivative<"stablehlo", opName_, patternToMatch, resultOps>;

class HLOInst<string m> : Inst<m, "mhlo">;

def Add : HLOInst<"AddOp">;
def Sub : HLOInst<"mhlo::SubOp">;
def Neg : HLOInst<"mhlo::NegOp">;
def Mul : HLOInst<"mhlo::MulOp">;
def Div : HLOInst<"mhlo::DivOp">;
def Rem : HLOInst<"mhlo::RemOp">;

include "HLODerivatives.td"
