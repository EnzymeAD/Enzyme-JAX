/* Copyright 2024 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "triton_dialect_capi.h"

#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/IR.h"
#include <gtest/gtest.h>

TEST(TritonDialectCapiTest, DialectRegistrationAndPointerTypes) {
  MlirContext context = mlirContextCreate();
  MlirDialectHandle tritonHandle = mlirGetDialectHandle__triton__();
  mlirDialectHandleLoadDialect(tritonHandle, context);

  // Test non-pointer type verification
  MlirType f32 = mlirF32TypeGet(context);
  EXPECT_FALSE(mlirTritonIsAPointer(f32));

  // Test PointerType creation and inspection
  int addressSpace = 1;
  MlirType ptrType = mlirTritonPointerTypeGet(f32, addressSpace);
  EXPECT_FALSE(mlirTypeIsNull(ptrType));
  EXPECT_TRUE(mlirTritonIsAPointer(ptrType));

  MlirType pointeeType = mlirTritonPointerTypeGetPointeeType(ptrType);
  EXPECT_TRUE(mlirTypeEqual(pointeeType, f32));
  EXPECT_EQ(mlirTritonPointerTypeGetAddressSpace(ptrType), addressSpace);

  // Test TypeID
  MlirTypeID ptrTypeId = mlirTritonPointerTypeGetTypeID();
  EXPECT_FALSE(mlirTypeIDIsNull(ptrTypeId));

  mlirContextDestroy(context);
}
