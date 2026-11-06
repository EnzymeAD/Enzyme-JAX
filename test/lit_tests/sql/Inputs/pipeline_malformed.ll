; This file is licensed under the Apache License v2.0 with LLVM Exceptions.
; See https://llvm.org/LICENSE.txt for license information.
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

@query = private constant [14 x i8] c"SELECT FROM t\00"
declare ptr @PQexec(ptr, ptr)

define ptr @bad_query(ptr %conn) {
  %result = call ptr @PQexec(ptr %conn, ptr @query)
  ret ptr %result
}
