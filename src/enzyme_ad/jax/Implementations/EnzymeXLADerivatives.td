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

class HLOConstantFP<string m> : ConstantFP<m, "stablehlo", "ConstantOp", "mlir::ElementsAttr">;

// Required operations from the EnzymeXLA dialect
def Gelu : EnzymeXLAInst<"GeluOp">;
def Relu : EnzymeXLAInst<"ReluOp">;

// Required operations from StableHLO/CHLO dialects
def Add : Inst<"AddOp", "stablehlo">;
def Sub : Inst<"SubtractOp", "stablehlo">;
def Mul : Inst<"MulOp", "stablehlo">;
def Div : Inst<"DivOp", "stablehlo">;
def Exp : Inst<"ExpOp", "stablehlo">;
def Tanh : Inst<"TanhOp", "stablehlo">;
def Pow : Inst<"PowOp", "stablehlo">;
def Logistic : Inst<"LogisticOp", "stablehlo">;
def Compare : Inst<"CompareOp", "stablehlo">;
def Select : Inst<"SelectOp", "stablehlo">;
def Erf : Inst<"ErfOp", "chlo">;

def GT : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, "ComparisonDirection::GT">;

def GeluApproximation : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  op.getGeluApproximationAttr();
}]>;

def SelectIfGeluNone : StaticSelect<"cast<enzymexla::GeluOp>(op).getGeluApproximation() == enzymexla::GeluApproximation::NONE">;
def SelectIfGeluTanh : StaticSelect<"cast<enzymexla::GeluOp>(op).getGeluApproximation() == enzymexla::GeluApproximation::TANH">;

// Derivative rules
// ReLU:
//   relu(x) = max(0, x)
//   d/dx relu(x) = 1 if x > 0 else 0
def : EnzymeXLADerivative<"ReluOp", (Op $input),
                    [
                      (Select
                        (Compare $input, (HLOConstantFP<"0"> $input), (GT)),
                        (DiffeRet),
                        (HLOConstantFP<"0"> $input)
                      )
                    ],
                    (Relu (Shadow $input))
                  >;

// GELU (x):
//   NONE:    gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
//            d/dx = 0.5 * (1 + erf(x / sqrt(2))) + x * exp(-x^2 / 2) / sqrt(2π)
//   TANH:    gelu(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
//            d/dx = 0.5 * (1 + tanh(u))
//                   + 0.5 * x * (1 - tanh(u)^2) * sqrt(2/pi) * (1 + 0.134145 * x^2),
//                   where u = sqrt(2/pi) * (x + 0.044715 * x^3)
//   SIGMOID: gelu(x) = x * sigmoid(1.702 * x)
//            d/dx = sigmoid(1.702*x) + 1.702*x*sigmoid(1.702*x)*(1 - sigmoid(1.702*x))
def : EnzymeXLADerivative<"GeluOp", (Op $input),
                    [
                      (Mul
                        (DiffeRet),
                        (SelectIfGeluNone
                          (Add
                            (Mul
                              (HLOConstantFP<"0.5"> $input),
                              (Add
                                (HLOConstantFP<"1"> $input),
                                (Erf (Div $input, (HLOConstantFP<"1.4142135623730951"> $input)))
                              )
                            ),
                            (Mul
                              $input,
                              (Mul
                                (HLOConstantFP<"0.3989422804014327"> $input),
                                (Exp (Mul (HLOConstantFP<"-0.5"> $input), (Mul $input, $input)))
                              )
                            )
                          ),
                          (SelectIfGeluTanh
                            (Add
                              (Mul
                                (HLOConstantFP<"0.5"> $input),
                                (Add
                                  (HLOConstantFP<"1"> $input),
                                  (Tanh
                                    (Mul
                                      (HLOConstantFP<"0.7978845608028654"> $input),
                                      (Add
                                        $input,
                                        (Mul (HLOConstantFP<"0.044715"> $input), (Pow $input, (HLOConstantFP<"3"> $input)))
                                      )
                                    )
                                  )
                                )
                              ),
                              (Mul
                                (Mul (HLOConstantFP<"0.5"> $input), $input),
                                (Mul
                                  (Sub
                                    (HLOConstantFP<"1"> $input),
                                    (Mul
                                      (Tanh
                                        (Mul
                                          (HLOConstantFP<"0.7978845608028654"> $input),
                                          (Add
                                            $input,
                                            (Mul (HLOConstantFP<"0.044715"> $input), (Pow $input, (HLOConstantFP<"3"> $input)))
                                          )
                                        )
                                      ),
                                      (Tanh
                                        (Mul
                                          (HLOConstantFP<"0.7978845608028654"> $input),
                                          (Add
                                            $input,
                                            (Mul (HLOConstantFP<"0.044715"> $input), (Pow $input, (HLOConstantFP<"3"> $input)))
                                          )
                                        )
                                      )
                                    )
                                  ),
                                  (Mul
                                    (HLOConstantFP<"0.7978845608028654"> $input),
                                    (Add
                                      (HLOConstantFP<"1"> $input),
                                      (Mul (HLOConstantFP<"0.134145"> $input), (Mul $input, $input))
                                    )
                                  )
                                )
                              )
                            ),
                            (Add
                              (Logistic (Mul (HLOConstantFP<"1.702"> $input), $input)),
                              (Mul
                                (Mul (HLOConstantFP<"1.702"> $input), $input),
                                (Mul
                                  (Logistic (Mul (HLOConstantFP<"1.702"> $input), $input)),
                                  (Sub
                                    (HLOConstantFP<"1"> $input),
                                    (Logistic (Mul (HLOConstantFP<"1.702"> $input), $input))
                                  )
                                )
                              )
                            )
                          )
                        )
                      )
                    ],
                    (Gelu (Shadow $input), (GeluApproximation))
                  >;

