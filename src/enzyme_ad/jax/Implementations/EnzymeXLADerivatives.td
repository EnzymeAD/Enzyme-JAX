include "src/enzyme_ad/jax/Implementations/Common.td"

class EnzymeXLADerivative<string opName_, dag patternToMatch, list<dag> resultOps, dag forwardOps=(ForwardFromSummedReverse)> : MLIRDerivative<"enzymexla", opName_, patternToMatch, resultOps, forwardOps>;

class EnzymeXLAInst<string m, string postopt="", string preopt=""> : Inst<m, "enzymexla", postopt, preopt>;

class EnzymeXLAMemoryIdentityOp<string opName_, list<int> ptrargs_, list<int> storedargs_ = [], dag patternToMatch=(Unimplemented), list<dag> reverse_ = []>  : MemoryIdentityOp<"enzymexla", opName_, ptrargs_, storedargs_, patternToMatch, reverse_>;

class EnzymeXLAReadOnlyIdentityOp<string opName_, list<int> ptrargs_ = [0], dag patternToMatch=(Unimplemented), list<dag> reverse_ = []> : ReadOnlyIdentityOp<"enzymexla", opName_, ptrargs_, patternToMatch, reverse_>;

class EnzymeXLAControlFlowOp<string opName_, string impl_> : ControlFlowOp<"enzymexla", opName_, impl_>;

class EnzymeXLAConstantFP<string m> : ConstantFP<m, "enzymexla", "ConstantOp", "mlir::ElementsAttr">;

class EnzymeXLARegionTerminatorOp<string m> : RegionTerminatorOp<"enzymexla", m>;

class EnzymeXLAInactiveOp<string m> : InactiveOp<"enzymexla", m>;

def : EnzymeXLAControlFlowOp<"GPUWrapperOp", [{
  Operation *createWithShadows(Operation *op, OpBuilder &builder,
                               MGradientUtils *gutils, Operation *original,
                               ValueRange remappedOperands,
                               TypeRange rettys) const {
    return enzymexla::GPUWrapperOp::create(builder, original->getLoc(),
                                        remappedOperands);
  }
}]>;

