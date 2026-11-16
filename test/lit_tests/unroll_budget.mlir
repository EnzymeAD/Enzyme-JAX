// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=enzyme_hlo_unroll(8)" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s

// A body of more than 128 ops still unrolls when the copies stay within the
// budget of 4096 ops: 4 iterations of 140 ops.
func.func @long_body(%arg0: tensor<4xf64>) -> tensor<4xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c4 = stablehlo.constant dense<4> : tensor<i64>
  %cst = stablehlo.constant dense<1.000000e+00> : tensor<4xf64>
  %0:2 = stablehlo.while(%iterArg = %c0, %iterArg_0 = %arg0) : tensor<i64>, tensor<4xf64>
  cond {
    %1 = stablehlo.compare LT, %iterArg, %c4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %1 : tensor<i1>
  } do {
    %r0 = stablehlo.add %iterArg_0, %cst : tensor<4xf64>
    %r1 = stablehlo.add %r0, %cst : tensor<4xf64>
    %r2 = stablehlo.add %r1, %cst : tensor<4xf64>
    %r3 = stablehlo.add %r2, %cst : tensor<4xf64>
    %r4 = stablehlo.add %r3, %cst : tensor<4xf64>
    %r5 = stablehlo.add %r4, %cst : tensor<4xf64>
    %r6 = stablehlo.add %r5, %cst : tensor<4xf64>
    %r7 = stablehlo.add %r6, %cst : tensor<4xf64>
    %r8 = stablehlo.add %r7, %cst : tensor<4xf64>
    %r9 = stablehlo.add %r8, %cst : tensor<4xf64>
    %r10 = stablehlo.add %r9, %cst : tensor<4xf64>
    %r11 = stablehlo.add %r10, %cst : tensor<4xf64>
    %r12 = stablehlo.add %r11, %cst : tensor<4xf64>
    %r13 = stablehlo.add %r12, %cst : tensor<4xf64>
    %r14 = stablehlo.add %r13, %cst : tensor<4xf64>
    %r15 = stablehlo.add %r14, %cst : tensor<4xf64>
    %r16 = stablehlo.add %r15, %cst : tensor<4xf64>
    %r17 = stablehlo.add %r16, %cst : tensor<4xf64>
    %r18 = stablehlo.add %r17, %cst : tensor<4xf64>
    %r19 = stablehlo.add %r18, %cst : tensor<4xf64>
    %r20 = stablehlo.add %r19, %cst : tensor<4xf64>
    %r21 = stablehlo.add %r20, %cst : tensor<4xf64>
    %r22 = stablehlo.add %r21, %cst : tensor<4xf64>
    %r23 = stablehlo.add %r22, %cst : tensor<4xf64>
    %r24 = stablehlo.add %r23, %cst : tensor<4xf64>
    %r25 = stablehlo.add %r24, %cst : tensor<4xf64>
    %r26 = stablehlo.add %r25, %cst : tensor<4xf64>
    %r27 = stablehlo.add %r26, %cst : tensor<4xf64>
    %r28 = stablehlo.add %r27, %cst : tensor<4xf64>
    %r29 = stablehlo.add %r28, %cst : tensor<4xf64>
    %r30 = stablehlo.add %r29, %cst : tensor<4xf64>
    %r31 = stablehlo.add %r30, %cst : tensor<4xf64>
    %r32 = stablehlo.add %r31, %cst : tensor<4xf64>
    %r33 = stablehlo.add %r32, %cst : tensor<4xf64>
    %r34 = stablehlo.add %r33, %cst : tensor<4xf64>
    %r35 = stablehlo.add %r34, %cst : tensor<4xf64>
    %r36 = stablehlo.add %r35, %cst : tensor<4xf64>
    %r37 = stablehlo.add %r36, %cst : tensor<4xf64>
    %r38 = stablehlo.add %r37, %cst : tensor<4xf64>
    %r39 = stablehlo.add %r38, %cst : tensor<4xf64>
    %r40 = stablehlo.add %r39, %cst : tensor<4xf64>
    %r41 = stablehlo.add %r40, %cst : tensor<4xf64>
    %r42 = stablehlo.add %r41, %cst : tensor<4xf64>
    %r43 = stablehlo.add %r42, %cst : tensor<4xf64>
    %r44 = stablehlo.add %r43, %cst : tensor<4xf64>
    %r45 = stablehlo.add %r44, %cst : tensor<4xf64>
    %r46 = stablehlo.add %r45, %cst : tensor<4xf64>
    %r47 = stablehlo.add %r46, %cst : tensor<4xf64>
    %r48 = stablehlo.add %r47, %cst : tensor<4xf64>
    %r49 = stablehlo.add %r48, %cst : tensor<4xf64>
    %r50 = stablehlo.add %r49, %cst : tensor<4xf64>
    %r51 = stablehlo.add %r50, %cst : tensor<4xf64>
    %r52 = stablehlo.add %r51, %cst : tensor<4xf64>
    %r53 = stablehlo.add %r52, %cst : tensor<4xf64>
    %r54 = stablehlo.add %r53, %cst : tensor<4xf64>
    %r55 = stablehlo.add %r54, %cst : tensor<4xf64>
    %r56 = stablehlo.add %r55, %cst : tensor<4xf64>
    %r57 = stablehlo.add %r56, %cst : tensor<4xf64>
    %r58 = stablehlo.add %r57, %cst : tensor<4xf64>
    %r59 = stablehlo.add %r58, %cst : tensor<4xf64>
    %r60 = stablehlo.add %r59, %cst : tensor<4xf64>
    %r61 = stablehlo.add %r60, %cst : tensor<4xf64>
    %r62 = stablehlo.add %r61, %cst : tensor<4xf64>
    %r63 = stablehlo.add %r62, %cst : tensor<4xf64>
    %r64 = stablehlo.add %r63, %cst : tensor<4xf64>
    %r65 = stablehlo.add %r64, %cst : tensor<4xf64>
    %r66 = stablehlo.add %r65, %cst : tensor<4xf64>
    %r67 = stablehlo.add %r66, %cst : tensor<4xf64>
    %r68 = stablehlo.add %r67, %cst : tensor<4xf64>
    %r69 = stablehlo.add %r68, %cst : tensor<4xf64>
    %r70 = stablehlo.add %r69, %cst : tensor<4xf64>
    %r71 = stablehlo.add %r70, %cst : tensor<4xf64>
    %r72 = stablehlo.add %r71, %cst : tensor<4xf64>
    %r73 = stablehlo.add %r72, %cst : tensor<4xf64>
    %r74 = stablehlo.add %r73, %cst : tensor<4xf64>
    %r75 = stablehlo.add %r74, %cst : tensor<4xf64>
    %r76 = stablehlo.add %r75, %cst : tensor<4xf64>
    %r77 = stablehlo.add %r76, %cst : tensor<4xf64>
    %r78 = stablehlo.add %r77, %cst : tensor<4xf64>
    %r79 = stablehlo.add %r78, %cst : tensor<4xf64>
    %r80 = stablehlo.add %r79, %cst : tensor<4xf64>
    %r81 = stablehlo.add %r80, %cst : tensor<4xf64>
    %r82 = stablehlo.add %r81, %cst : tensor<4xf64>
    %r83 = stablehlo.add %r82, %cst : tensor<4xf64>
    %r84 = stablehlo.add %r83, %cst : tensor<4xf64>
    %r85 = stablehlo.add %r84, %cst : tensor<4xf64>
    %r86 = stablehlo.add %r85, %cst : tensor<4xf64>
    %r87 = stablehlo.add %r86, %cst : tensor<4xf64>
    %r88 = stablehlo.add %r87, %cst : tensor<4xf64>
    %r89 = stablehlo.add %r88, %cst : tensor<4xf64>
    %r90 = stablehlo.add %r89, %cst : tensor<4xf64>
    %r91 = stablehlo.add %r90, %cst : tensor<4xf64>
    %r92 = stablehlo.add %r91, %cst : tensor<4xf64>
    %r93 = stablehlo.add %r92, %cst : tensor<4xf64>
    %r94 = stablehlo.add %r93, %cst : tensor<4xf64>
    %r95 = stablehlo.add %r94, %cst : tensor<4xf64>
    %r96 = stablehlo.add %r95, %cst : tensor<4xf64>
    %r97 = stablehlo.add %r96, %cst : tensor<4xf64>
    %r98 = stablehlo.add %r97, %cst : tensor<4xf64>
    %r99 = stablehlo.add %r98, %cst : tensor<4xf64>
    %r100 = stablehlo.add %r99, %cst : tensor<4xf64>
    %r101 = stablehlo.add %r100, %cst : tensor<4xf64>
    %r102 = stablehlo.add %r101, %cst : tensor<4xf64>
    %r103 = stablehlo.add %r102, %cst : tensor<4xf64>
    %r104 = stablehlo.add %r103, %cst : tensor<4xf64>
    %r105 = stablehlo.add %r104, %cst : tensor<4xf64>
    %r106 = stablehlo.add %r105, %cst : tensor<4xf64>
    %r107 = stablehlo.add %r106, %cst : tensor<4xf64>
    %r108 = stablehlo.add %r107, %cst : tensor<4xf64>
    %r109 = stablehlo.add %r108, %cst : tensor<4xf64>
    %r110 = stablehlo.add %r109, %cst : tensor<4xf64>
    %r111 = stablehlo.add %r110, %cst : tensor<4xf64>
    %r112 = stablehlo.add %r111, %cst : tensor<4xf64>
    %r113 = stablehlo.add %r112, %cst : tensor<4xf64>
    %r114 = stablehlo.add %r113, %cst : tensor<4xf64>
    %r115 = stablehlo.add %r114, %cst : tensor<4xf64>
    %r116 = stablehlo.add %r115, %cst : tensor<4xf64>
    %r117 = stablehlo.add %r116, %cst : tensor<4xf64>
    %r118 = stablehlo.add %r117, %cst : tensor<4xf64>
    %r119 = stablehlo.add %r118, %cst : tensor<4xf64>
    %r120 = stablehlo.add %r119, %cst : tensor<4xf64>
    %r121 = stablehlo.add %r120, %cst : tensor<4xf64>
    %r122 = stablehlo.add %r121, %cst : tensor<4xf64>
    %r123 = stablehlo.add %r122, %cst : tensor<4xf64>
    %r124 = stablehlo.add %r123, %cst : tensor<4xf64>
    %r125 = stablehlo.add %r124, %cst : tensor<4xf64>
    %r126 = stablehlo.add %r125, %cst : tensor<4xf64>
    %r127 = stablehlo.add %r126, %cst : tensor<4xf64>
    %r128 = stablehlo.add %r127, %cst : tensor<4xf64>
    %r129 = stablehlo.add %r128, %cst : tensor<4xf64>
    %r130 = stablehlo.add %r129, %cst : tensor<4xf64>
    %r131 = stablehlo.add %r130, %cst : tensor<4xf64>
    %r132 = stablehlo.add %r131, %cst : tensor<4xf64>
    %r133 = stablehlo.add %r132, %cst : tensor<4xf64>
    %r134 = stablehlo.add %r133, %cst : tensor<4xf64>
    %r135 = stablehlo.add %r134, %cst : tensor<4xf64>
    %r136 = stablehlo.add %r135, %cst : tensor<4xf64>
    %r137 = stablehlo.add %r136, %cst : tensor<4xf64>
    %r138 = stablehlo.add %r137, %cst : tensor<4xf64>
    %r139 = stablehlo.add %r138, %cst : tensor<4xf64>
    %next = stablehlo.add %iterArg, %c1 : tensor<i64>
    stablehlo.return %next, %r139 : tensor<i64>, tensor<4xf64>
  }
  return %0#1 : tensor<4xf64>
}

// CHECK-LABEL: func.func @long_body
// CHECK-NOT: stablehlo.while

// -----

// The body yields a dynamically shaped value for a static carried type: the
// unrolled copies would hand it straight to the function's return.
func.func @dynamic_yield(%arg0: tensor<1xi8>, %arg1: tensor<1xi1>) -> tensor<1xi8> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c4 = stablehlo.constant dense<4> : tensor<i64>
  %pad = stablehlo.constant dense<0> : tensor<i8>
  %zero = stablehlo.constant dense<0> : tensor<1xi64>
  %one = stablehlo.constant dense<1> : tensor<1xi8>
  %0:2 = stablehlo.while(%iterArg = %c0, %iterArg_0 = %arg0) : tensor<i64>, tensor<1xi8>
  cond {
    %1 = stablehlo.compare LT, %iterArg, %c4 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %1 : tensor<i1>
  } do {
    %1 = stablehlo.select %arg1, %iterArg_0, %one : tensor<1xi1>, tensor<1xi8>
    %2 = stablehlo.dynamic_pad %1, %pad, %zero, %zero, %zero : (tensor<1xi8>, tensor<i8>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi8>
    %next = stablehlo.add %iterArg, %c1 : tensor<i64>
    stablehlo.return %next, %2 : tensor<i64>, tensor<?xi8>
  }
  return %0#1 : tensor<1xi8>
}

// CHECK-LABEL: func.func @dynamic_yield
// CHECK: stablehlo.while
