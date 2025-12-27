// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=recognize_rotate" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

func.func @rot1(%1171 : tensor<4x1520x3056xf32>) -> tensor<4x1520x3057xf32> {
  %2665 = stablehlo.slice %1171 [0:4, 0:1520, 3054:3055] : (tensor<4x1520x3056xf32>) -> tensor<4x1520x1xf32>
  %2290 = "enzymexla.rotate"(%1171) <{amount = 3055 : si32, dimension = 2 : si32}> : (tensor<4x1520x3056xf32>) -> tensor<4x1520x3056xf32>
  %2666 = stablehlo.concatenate %2665, %2290, dim = 2 : (tensor<4x1520x1xf32>, tensor<4x1520x3056xf32>) -> tensor<4x1520x3057xf32>
  stablehlo.return %2666 : tensor<4x1520x3057xf32> 
}

// CHECK:  func.func @rot1(%arg0: tensor<4x1520x3056xf32>) -> tensor<4x1520x3057xf32> {
// CHECK-NEXT:    %0 = "enzymexla.rotate"(%arg0) <{amount = 3055 : si32, dimension = 2 : si32}> : (tensor<4x1520x3056xf32>) -> tensor<4x1520x3056xf32>
// CHECK-NEXT:    %1 = "enzymexla.wrap"(%0) <{dimension = 2 : i64, lhs = 1 : i64, rhs = 0 : i64}> : (tensor<4x1520x3056xf32>) -> tensor<4x1520x3057xf32>
// CHECK-NEXT:    stablehlo.return %1 : tensor<4x1520x3057xf32>
// CHECK-NEXT:  }

func.func @rot2(%1179 : tensor<4x1519x3056xf32>) -> tensor<4x1519x3057xf32> {

%1185 = stablehlo.slice %1179 [0:4, 0:1519, 3053:3054] : (tensor<4x1519x3056xf32>) -> tensor<4x1519x1xf32>
%1755 = "enzymexla.rotate"(%1179) <{amount = 3054 : si32, dimension = 2 : si32}> : (tensor<4x1519x3056xf32>) -> tensor<4x1519x3056xf32>
%2927 = stablehlo.concatenate %1185, %1755, dim = 2 : (tensor<4x1519x1xf32>, tensor<4x1519x3056xf32>) -> tensor<4x1519x3057xf32>

  stablehlo.return %2927 : tensor<4x1519x3057xf32> 
}

// CHECK:  func.func @rot2(%arg0: tensor<4x1519x3056xf32>) -> tensor<4x1519x3057xf32> {
// CHECK-NEXT:    %0 = "enzymexla.rotate"(%arg0) <{amount = 3054 : si32, dimension = 2 : si32}> : (tensor<4x1519x3056xf32>) -> tensor<4x1519x3056xf32>
// CHECK-NEXT:    %1 = "enzymexla.wrap"(%0) <{dimension = 2 : i64, lhs = 1 : i64, rhs = 0 : i64}> : (tensor<4x1519x3056xf32>) -> tensor<4x1519x3057xf32>
// CHECK-NEXT:    stablehlo.return %1 : tensor<4x1519x3057xf32>
// CHECK-NEXT:  }

func.func @argslice(%1171 : tensor<4x1520x3056xf32>) -> tensor<4x1520x3057xf32> {
   %2295 = stablehlo.slice %1171 [0:4, 0:1520, 0:1] : (tensor<4x1520x3056xf32>) -> tensor<4x1520x1xf32>
   %2880 = stablehlo.concatenate %1171, %2295, dim = 2 : (tensor<4x1520x3056xf32>, tensor<4x1520x1xf32>) -> tensor<4x1520x3057xf32>
  stablehlo.return %2880 : tensor<4x1520x3057xf32> 
}

// CHECK:  func.func @argslice(%arg0: tensor<4x1520x3056xf32>) -> tensor<4x1520x3057xf32> {
// CHECK-NEXT:    %0 = "enzymexla.wrap"(%arg0) <{dimension = 2 : i64, lhs = 0 : i64, rhs = 1 : i64}> : (tensor<4x1520x3056xf32>) -> tensor<4x1520x3057xf32>
// CHECK-NEXT:    stablehlo.return %0 : tensor<4x1520x3057xf32>
// CHECK-NEXT:  }


func.func @argslice2(%1179 : tensor<4x1519x3056xf32>) -> tensor<4x1519x3057xf32> {
  %1186 = stablehlo.slice %1179 [0:4, 0:1519, 0:1] : (tensor<4x1519x3056xf32>) -> tensor<4x1519x1xf32>
  %2933 = stablehlo.concatenate %1179, %1186, dim = 2 : (tensor<4x1519x3056xf32>, tensor<4x1519x1xf32>) -> tensor<4x1519x3057xf32>
  stablehlo.return %2933 : tensor<4x1519x3057xf32>
}

// CHECK:  func.func @argslice2(%arg0: tensor<4x1519x3056xf32>) -> tensor<4x1519x3057xf32> {
// CHECK-NEXT:    %0 = "enzymexla.wrap"(%arg0) <{dimension = 2 : i64, lhs = 0 : i64, rhs = 1 : i64}> : (tensor<4x1519x3056xf32>) -> tensor<4x1519x3057xf32>
// CHECK-NEXT:    stablehlo.return %0 : tensor<4x1519x3057xf32>
// CHECK-NEXT:  }

func.func @slicearg(%1171 : tensor<4x1520x3056xf32>) -> tensor<4x1520x3057xf32> {
  %2668 = stablehlo.slice %1171 [0:4, 0:1520, 3055:3056] : (tensor<4x1520x3056xf32>) -> tensor<4x1520x1xf32>
  %2669 = stablehlo.concatenate %2668, %1171, dim = 2 : (tensor<4x1520x1xf32>, tensor<4x1520x3056xf32>) -> tensor<4x1520x3057xf32>
  stablehlo.return %2669 : tensor<4x1520x3057xf32> 
}

// CHECK-NEXT:  func.func @slicearg(%arg0: tensor<4x1520x3056xf32>) -> tensor<4x1520x3057xf32> {
// CHECK-NEXT:    %0 = "enzymexla.wrap"(%arg0) <{dimension = 2 : i64, lhs = 1 : i64, rhs = 0 : i64}> : (tensor<4x1520x3056xf32>) -> tensor<4x1520x3057xf32>
// CHECK-NEXT:    stablehlo.return %0 : tensor<4x1520x3057xf32>
// CHECK-NEXT:  }

func.func @sliceslice(%1257 : tensor<20x1536x3056xf32>) -> tensor<4x1520x3057xf32> {
	%1258 = stablehlo.slice %1257 [8:12, 7:1527, 0:3056] : (tensor<20x1536x3056xf32>) -> tensor<4x1520x3056xf32> 
	%1262 = stablehlo.slice %1257 [8:12, 7:1527, 0:1] : (tensor<20x1536x3056xf32>) -> tensor<4x1520x1xf32> 
	%2878 = stablehlo.concatenate %1258, %1262, dim = 2 : (tensor<4x1520x3056xf32>, tensor<4x1520x1xf32>) -> tensor<4x1520x3057xf32>
  stablehlo.return %2878 : tensor<4x1520x3057xf32>
}

// CHECK:  func.func @sliceslice(%arg0: tensor<20x1536x3056xf32>) -> tensor<4x1520x3057xf32> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [8:12, 7:1527, 0:3056] : (tensor<20x1536x3056xf32>) -> tensor<4x1520x3056xf32>
// CHECK-NEXT:    %1 = "enzymexla.wrap"(%0) <{dimension = 2 : i64, lhs = 0 : i64, rhs = 1 : i64}> : (tensor<4x1520x3056xf32>) -> tensor<4x1520x3057xf32>
// CHECK-NEXT:    stablehlo.return %1 : tensor<4x1520x3057xf32>
// CHECK-NEXT:  }

func.func @sliceslice2(%1179 : tensor<4x1519x3056xf32>) -> tensor<4x1520x3057xf32> {
%1187 = stablehlo.slice %1179 [0:4, 0:1519, 1:3056] : (tensor<4x1519x3056xf32>) -> tensor<4x1519x3055xf32>
%1183 = stablehlo.slice %1179 [0:4, 0:1519, 0:2] : (tensor<4x1519x3056xf32>) -> tensor<4x1519x2xf32>
%2936 = stablehlo.concatenate %1187, %1183, dim = 2 : (tensor<4x1519x3055xf32>, tensor<4x1519x2xf32>) -> tensor<4x1519x3057xf32>
  stablehlo.return %2936 : tensor<4x1519x3057xf32> 
}

// CHECK:  func.func @sliceslice2(%arg0: tensor<4x1519x3056xf32>) -> tensor<4x1520x3057xf32> {
// CHECK-NEXT:    %0 = "enzymexla.rotate"(%arg0) <{amount = 1 : si32, dimension = 2 : si32}> : (tensor<4x1519x3056xf32>) -> tensor<4x1519x3056xf32>
// CHECK-NEXT:    %1 = "enzymexla.wrap"(%0) <{dimension = 2 : i64, lhs = 0 : i64, rhs = 1 : i64}> : (tensor<4x1519x3056xf32>) -> tensor<4x1519x3057xf32>
// CHECK-NEXT:    stablehlo.return %1 : tensor<4x1519x3057xf32>
// CHECK-NEXT:  }

func.func @sliceslice3(%1179 : tensor<4x1519x3056xf32>) -> tensor<4x1520x3057xf32> {
%1184 = stablehlo.slice %1179 [0:4, 0:1519, 2:3056] : (tensor<4x1519x3056xf32>) -> tensor<4x1519x3054xf32> 
%1182 = stablehlo.slice %1179 [0:4, 0:1519, 0:3] : (tensor<4x1519x3056xf32>) -> tensor<4x1519x3xf32> 
%2984 = stablehlo.concatenate %1184, %1182, dim = 2 : (tensor<4x1519x3054xf32>, tensor<4x1519x3xf32>) -> tensor<4x1519x3057xf32> 
  stablehlo.return %2984 : tensor<4x1519x3057xf32> 
}

// CHECK:  func.func @sliceslice3(%arg0: tensor<4x1519x3056xf32>) -> tensor<4x1520x3057xf32> {
// CHECK-NEXT:    %0 = "enzymexla.rotate"(%arg0) <{amount = 2 : si32, dimension = 2 : si32}> : (tensor<4x1519x3056xf32>) -> tensor<4x1519x3056xf32>
// CHECK-NEXT:    %1 = "enzymexla.wrap"(%0) <{dimension = 2 : i64, lhs = 0 : i64, rhs = 1 : i64}> : (tensor<4x1519x3056xf32>) -> tensor<4x1519x3057xf32>
// CHECK-NEXT:    stablehlo.return %1 : tensor<4x1519x3057xf32>
// CHECK-NEXT:  }

func.func @sliceslice4(%32 : tensor<20x6144x12288xf32>) -> tensor<4x6128x12273xf32> {
%2064 = stablehlo.slice %32 [8:12, 9:6137, 12279:12280] : (tensor<20x6144x12288xf32>) -> tensor<4x6128x1xf32>
%784 = stablehlo.slice %32 [8:12, 9:6137, 8:12280] : (tensor<20x6144x12288xf32>) -> tensor<4x6128x12272xf32>
%2066 = stablehlo.concatenate %2064, %784, dim = 2 : (tensor<4x6128x1xf32>, tensor<4x6128x12272xf32>) -> tensor<4x6128x12273xf32>
stablehlo.return %2066 : tensor<4x6128x12273xf32>
}

// CHECK-NEXT:  func.func @sliceslice4(%arg0: tensor<20x6144x12288xf32>) -> tensor<4x6128x12273xf32> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [8:12, 9:6137, 8:12280] : (tensor<20x6144x12288xf32>) -> tensor<4x6128x12272xf32>
// CHECK-NEXT:    %1 = "enzymexla.wrap"(%0) <{dimension = 2 : i64, lhs = 1 : i64, rhs = 0 : i64}> : (tensor<4x6128x12272xf32>) -> tensor<4x6128x12273xf32>
// CHECK-NEXT:    stablehlo.return %1 : tensor<4x6128x12273xf32>
// CHECK-NEXT:  }
