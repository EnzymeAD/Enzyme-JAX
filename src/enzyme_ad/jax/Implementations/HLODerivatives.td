
def Add : HLOInst<"AddOp">;
def Sub : HLOInst<"SubtractOp">;
def Neg : HLOInst<"NegOp">;
def Mul : HLOInst<"MulOp">;
def Div : HLOInst<"DivOp">;
def Rem : HLOInst<"RemainderOp">;
def Pow : HLOInst<"PowOp">;
def Log : HLOInst<"LogOp">;
def Cos : HLOInst<"CosineOp">;
def Sin : HLOInst<"SineOp">;
def Sqrt : HLOInst<"SqrtOp">;
def Exp : HLOInst<"ExpOp">;

def Dot : HLOInst<"DotGeneralOp">;


def CheckedMul : HLOInst<"MulOp">;
def CheckedDiv : HLOInst<"DivOp">;

def : HLODerivative<"AddOp", (Op $x, $y),
                    [
                      (DiffeRet),
                      (DiffeRet),
                    ]
                  >;

def : HLODerivative<"SubtractOp", (Op $x, $y),
                    [
                      (DiffeRet),
                      (Neg (DiffeRet)),
                    ]
                  >;
def : HLODerivative<"NegOp", (Op $x),
                    [
                      (Neg (DiffeRet)),
                    ]
                  >;
def : HLODerivative<"MulOp", (Op $x, $y),
                    [
                      (CheckedMul (DiffeRet), $y),
                      (CheckedMul (DiffeRet), $x)
                    ]
                  >;
def : HLODerivative<"DivOp", (Op $x, $y), 
                    [
                      (CheckedDiv (DiffeRet), $y),
                      (Neg (Mul (CheckedDiv (DiffeRet), $y), (Div $x, $y)))
                    ]
                    // (CheckedDiv (FSub (SelectIfActive $x, (FMul (Shadow $x), $y), (Zero $x)), (SelectIfActive $y, (FMul (Shadow $y), $x), (Zero $y))), (FMul $y, $y))
                  >;

def : HLODerivative<"PowOp", (Op $x, $y), 
                  [
                    (CheckedMul (DiffeRet), (Mul $y, (Pow $x, (Sub $y, (HLOConstantFP<"1"> $y))))),
                    (CheckedMul (DiffeRet), (Mul (Pow $x, $y), (Log $x)
                    ))
                  ]
                 >;

def : HLODerivative<"CosineOp", (Op $x),
                    [
                      (CheckedMul (DiffeRet), (Neg (Sin $x)))
                    ]
                  >;
def : HLODerivative<"ExpOp", (Op $x),
                    [
                      (CheckedMul (DiffeRet), (Exp $x))
                    ]
                  >;
def : HLODerivative<"SineOp", (Op $x),
                    [
                      (CheckedMul (DiffeRet), (Cos $x))
                      ]
                  >;
def : HLODerivative<"SqrtOp", (Op $x),
                    [
                      // (Select (FCmpUEQ $x, (ConstantFP<"0"> $x)), (ConstantFP<"0"> $x), (FDiv (DiffeRet), (FMul (ConstantFP<"2"> $x), (Call<(SameFunc), [ReadNone,NoUnwind]> $x))))
                      (Div (DiffeRet), (Mul (HLOConstantFP<"2"> $x), (Sqrt $x)))
                      ]
                  >;

def : HLOReadOnlyIdentityOp<"ReshapeOp">;
def : HLOReadOnlyIdentityOp<"SliceOp">;
def : HLOReadOnlyIdentityOp<"BroadcastInDimOp">;
def : HLOReadOnlyIdentityOp<"ConcatenateOp">;
// convert


def ResultDotDim : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, "op.getDotDimensionNumbersAttr()">;
def ResultDotPrec : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, "op.getPrecisionConfigAttr()">;

def : HLODerivative<"DotGeneralOp", (Op $lhs, $rhs),
                    [
                        (Dot (ResultTypes), (DiffeRet), $rhs, (ResultDotDim), (ResultDotPrec)),
                        (Dot (ResultTypes), $lhs, (DiffeRet), (ResultDotDim), (ResultDotPrec))
                      ],
                     (Add (SelectIfActive $lhs, (Dot (ResultTypes), (Shadow $lhs), $rhs, (ResultDotDim), (ResultDotPrec)), (HLOConstantFP<"0">)), (SelectIfActive $rhs, (Dot (ResultTypes), $lhs, (Shadow $rhs), (ResultDotDim), (ResultDotPrec)), (HLOConstantFP<"0">)))
                  >;
