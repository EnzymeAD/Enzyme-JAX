// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=compare_cleanup" --transform-interpreter --enzyme-hlo-remove-transform -allow-unregistered-dialect --split-input-file %s | FileCheck %s


  func.func @add(%1195 : tensor<6128xi64>) -> tensor<6128xi1> {

    %145 = stablehlo.constant dense<6126> : tensor<6128xi64>

    %1232 = stablehlo.add %1195, %145 : tensor<6128xi64>


    %3 = stablehlo.constant dense<0> : tensor<6128xi64>

    %1260 = stablehlo.compare  GE, %1232, %3: (tensor<6128xi64>, tensor<6128xi64>) -> tensor<6128xi1>

    return %1260 : tensor<6128xi1>
  }

// CHECK:  func.func @add(%arg0: tensor<6128xi64>) -> tensor<6128xi1> {
// CHECK-NEXT:    %c = stablehlo.constant dense<-6126> : tensor<6128xi64>
// CHECK-NEXT:    %0 = stablehlo.compare  GE, %arg0, %c : (tensor<6128xi64>, tensor<6128xi64>) -> tensor<6128xi1>
// CHECK-NEXT:    return %0 : tensor<6128xi1>
// CHECK-NEXT:  }


  func.func @neg(%970 : tensor<6128xi64>) -> tensor<6128xi1> {
    %144 = stablehlo.constant dense<-1> : tensor<6128xi64>

    %1195 = stablehlo.multiply %970, %144 : tensor<6128xi64>

    %c = stablehlo.constant dense<-6126> : tensor<6128xi64>
    %0 = stablehlo.compare  GE, %1195, %c : (tensor<6128xi64>, tensor<6128xi64>) -> tensor<6128xi1>

    return %0 : tensor<6128xi1>
  }

// CHECK:    func.func @neg(%arg0: tensor<6128xi64>) -> tensor<6128xi1> {
// CHECK-NEXT:    %c = stablehlo.constant dense<6126> : tensor<6128xi64>
// CHECK-NEXT:    %0 = stablehlo.compare  LE, %arg0, %c : (tensor<6128xi64>, tensor<6128xi64>) -> tensor<6128xi1>
// CHECK-NEXT:    return %0 : tensor<6128xi1>
// CHECK-NEXT:  }