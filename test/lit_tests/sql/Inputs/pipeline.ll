; This file is licensed under the Apache License v2.0 with LLVM Exceptions.
; See https://llvm.org/LICENSE.txt for license information.
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

target datalayout = "e-p:64:64-p1:32:32-p3:32:32-p5:32:32"

@query = private addrspace(5) constant [128 x i8] c"SELECT a-(b-c) FROM t WHERE x=1 OR y=2 AND z=3 LIMIT 4                                                                         \00", align 32

declare fastcc nonnull ptr addrspace(1) @PQexec(ptr addrspace(1), ptr addrspace(5) align 16)
declare i64 @PQntuples(ptr addrspace(1))
declare nonnull ptr addrspace(3) @PQgetvalue(ptr addrspace(1), i16, i64)

define ptr addrspace(1) @constant_query(ptr addrspace(1) %conn) {
  %handle = call fastcc nonnull ptr addrspace(1) @PQexec(ptr addrspace(1) %conn, ptr addrspace(5) nonnull align 32 dereferenceable(128) @query)
  ret ptr addrspace(1) %handle
}

define ptr addrspace(3) @dynamic_query(ptr addrspace(1) %conn, ptr addrspace(5) %text, i16 %row, i64 %column) {
  %handle = call fastcc nonnull ptr addrspace(1) @PQexec(ptr addrspace(1) %conn, ptr addrspace(5) nonnull %text)
  %count = call i64 @PQntuples(ptr addrspace(1) %handle)
  %value = call nonnull ptr addrspace(3) @PQgetvalue(ptr addrspace(1) %handle, i16 %row, i64 %column)
  ret ptr addrspace(3) %value
}
