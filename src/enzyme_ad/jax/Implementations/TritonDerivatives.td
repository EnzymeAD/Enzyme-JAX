include "src/enzyme_ad/jax/Implementations/Common.td"

class TritonDerivative<string opName_, dag patternToMatch, list<dag> resultOps, dag forwardOps=(ForwardFromSummedReverse)> : MLIRDerivative<"triton", opName_, patternToMatch, resultOps, forwardOps>;

class TritonInst<string m, string postopt="", string preopt=""> : Inst<m, "triton", postopt, preopt>;

class TritonMemoryIdentityOp<string opName_, list<int> ptrargs_, list<int> storedargs_ = [], dag patternToMatch=(Unimplemented), list<dag> reverse_ = []>  : MemoryIdentityOp<"triton", opName_, ptrargs_, storedargs_, patternToMatch, reverse_>;

class TritonReadOnlyIdentityOp<string opName_, list<int> ptrargs_ = [0], dag patternToMatch=(Unimplemented), list<dag> reverse_ = []> : ReadOnlyIdentityOp<"triton", opName_, ptrargs_, patternToMatch, reverse_>;

class ArithConstantFP<string m> : ConstantFP<m, "arith", "ConstantOp", "mlir::ElementsAttr">;

class TritonInactiveOp<string m> : InactiveOp<"triton", m>;

class TritonReturnOp<string m> : ReturnOp<"triton", m>;

def FpToFp : TritonInst<"FpToFpOp">;
def MakeRange : TritonInst<"MakeRangeOp">;

def : TritonReturnOp<"ReturnOp">;
def : TritonInactiveOp<"MakeRangeOp">;
def : TritonInactiveOp<"BitcastOp">;

def : MemoryIdentityOp<"triton", "StoreOp", [1], [0]>;
def : ReadOnlyIdentityOp<"triton", "LoadOp", [0]>;
def : ReadOnlyIdentityOp<"triton", "AddPtrOp", [0]>;
def : ReadOnlyIdentityOp<"triton", "SplatOp", [0]>;

def FpToFpRoundingMode : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.getRoundingAttr();
}]>;

def : TritonDerivative<"FpToFpOp", (Op $x),
                       [
                         (FpToFp (TypeOf $x), (DiffeRet), (FpToFpRoundingMode))
                       ]
                     >;
