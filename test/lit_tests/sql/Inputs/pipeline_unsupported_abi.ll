; This file is licensed under the Apache License v2.0 with LLVM Exceptions.
; See https://llvm.org/LICENSE.txt for license information.
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

@query = private constant [9 x i8] c"SELECT 1\00"
declare ptr @PQexec(ptr, ptr byval([9 x i8]))

define ptr @unsupported_query_abi(ptr %conn) {
  %result = call ptr @PQexec(ptr %conn, ptr byval([9 x i8]) @query)
  ret ptr %result
}
