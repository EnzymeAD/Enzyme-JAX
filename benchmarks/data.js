window.BENCHMARK_DATA = {
  "lastUpdate": 1762149887380,
  "repoUrl": "https://github.com/EnzymeAD/Enzyme-JAX",
  "entries": {
    "EnzymeJAX Benchmarks": [
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e63a3a3db90bd59b4e13e716283d1dccf6f19a12",
          "message": "feat: generalize WhileIsCopy for partial slices (#1461)",
          "timestamp": "2025-10-07T09:13:20-04:00",
          "tree_id": "0e1642db6f135df8d641f7ce6d51dba581da93b7",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/e63a3a3db90bd59b4e13e716283d1dccf6f19a12"
        },
        "date": 1759845991874,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004341366098378785,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.0000042936721001751724,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001602961997006,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001692575248016,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "wmoses@google.com",
            "name": "William Moses",
            "username": "wsmoses"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "bb5eb26b2ddc5bbb77e8ff22b8ef2499473c5f5e",
          "message": "calling conv fix (#1462)",
          "timestamp": "2025-10-07T14:04:15-05:00",
          "tree_id": "de512f8adba6e5b317c79c12b43ad86fd340edd7",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/bb5eb26b2ddc5bbb77e8ff22b8ef2499473c5f5e"
        },
        "date": 1759867079062,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.0000045103268988896165,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004478790998109616,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001281787919986,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001634209176001,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "31b84b426807dea183316469b04c1672e4a43435",
          "message": "feat: constant folding of scatter op (#1466)\n\n* feat: constant folding of scatter op\n\n* chore: prune out unwanted deps",
          "timestamp": "2025-10-10T20:15:24-04:00",
          "tree_id": "4e5d36efac9d12057a50fc1f6b4d829ef928ac91",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/31b84b426807dea183316469b04c1672e4a43435"
        },
        "date": 1760153750718,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000005471757001942024,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004321999900275842,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001507768686977,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001482369966979,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ca3badc415cecf86258853abcbd3966a7782c10b",
          "message": "feat: split variadic scatter (#1467)\n\n* feat: split variadic scatter\n\n* feat: split variadic scatter\n\n* fix: logging remove\n\n* fix: slice_simplify\n\n* chore: run fmt",
          "timestamp": "2025-10-11T20:02:47-04:00",
          "tree_id": "338b7f893a5d2a18739fe2fddbbfee334abd7778",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/ca3badc415cecf86258853abcbd3966a7782c10b"
        },
        "date": 1760230858468,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004422669904306531,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004261300899088383,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.000134292779304,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001348743572947,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "949643a9ea6dd41d594d60a8f4276ebe409eb7d7",
          "message": "fix: diagonal matmul with contracting dims (#1472)",
          "timestamp": "2025-10-13T11:49:53-04:00",
          "tree_id": "608b3576de6007db4173f65a26a15b9572b8f252",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/949643a9ea6dd41d594d60a8f4276ebe409eb7d7"
        },
        "date": 1760389217213,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004399227900285041,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.00000434339089988498,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001372414594996,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001358201774994,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "759a0af09038beed4525b44aa0df7311f82bf8ae",
          "message": "feat: dot_general(ones, A) simplification (#1473)\n\n* feat: dot_general(ones, A) simplification\n\n* feat: generalize to handle all splatted tensors",
          "timestamp": "2025-10-13T14:54:19-04:00",
          "tree_id": "ba0a4743accb242f5094d07b56c7f5073e78e0a2",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/759a0af09038beed4525b44aa0df7311f82bf8ae"
        },
        "date": 1760392446858,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004271665900887456,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.0000042240148992277685,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001407691287997,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001393379648987,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "3d66292dd7ee4c05611641610a0891828f920297",
          "message": "fix: dot_general_simplify for complex numbers (#1478)",
          "timestamp": "2025-10-14T10:57:19-04:00",
          "tree_id": "1eb36c9971e8032ec8cff2f3370af33f3c483ab8",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/3d66292dd7ee4c05611641610a0891828f920297"
        },
        "date": 1760457122408,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004383468901505694,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004369143999065273,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001544043866975,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001480491947004,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c69b2705dde03bd34a671b67217b041d321c03bc",
          "message": "feat: symm (#1479)\n\n* ssymm\n\n* change fattr to ftensor\n\n* add test\n\n* add cpp api\n\n* Apply suggestion from @avik-pal\n\n* fix: typo in name\n\n* fix: ensure types match\n\n* feat: support complex\n\n---------\n\nCo-authored-by: snonk <scq@mit.edu>",
          "timestamp": "2025-10-14T13:29:05-04:00",
          "tree_id": "8597a0ed82d796440a6b718c5a94e81b57b9f31d",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/c69b2705dde03bd34a671b67217b041d321c03bc"
        },
        "date": 1760469087991,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.0000043608139996649695,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004245785999228246,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001479420117015,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001517619816993,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "cea9c6b7ee55495623d68903a405f848eff1b3bb",
          "message": "feat: generalize reduce_pad patterns for mul/add/min/max (#1480)\n\n* feat: generalize reduce_pad\n\n* feat: generalize reduce_pad patterns for mul/add/min/max",
          "timestamp": "2025-10-14T19:33:09-04:00",
          "tree_id": "3f460a6dd8f735ff602264f53eb8229a0b1aeae5",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/cea9c6b7ee55495623d68903a405f848eff1b3bb"
        },
        "date": 1760501992684,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004481526900781318,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004676483900402672,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001501561515004,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001408460686012,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "996b3e5ccbe5024d69d9a365e5f4db79f41dbc6c",
          "message": "feat: defer analysis passes + mark ir with analysis results (#1482)\n\n* feat: defer analysis passes + mark ir with analysis results\n\n* fix: use modify op in place",
          "timestamp": "2025-10-16T09:45:35-04:00",
          "tree_id": "09a009cc499781287824ebc84c5e5a9f81cbd130",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/996b3e5ccbe5024d69d9a365e5f4db79f41dbc6c"
        },
        "date": 1760640931426,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004217056999914348,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004099265101831406,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001403182348993,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001510700437997,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "765740+giordano@users.noreply.github.com",
            "name": "Mosè Giordano",
            "username": "giordano"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "6137ac98e710adf6f4e953bf441db4e25b2db40f",
          "message": "Update JAX to commit 45065d569064392cee45a066f9f788fe29ee2cd8 (#1418)\n\n* Update JAX to commit 45065d569064392cee45a066f9f788fe29ee2cd8\n\nDiff: https://github.com/jax-ml/jax/compare/4455434869812dfb7336b18f2e59a8690ba38e79...45065d569064392cee45a066f9f788fe29ee2cd8\n\n* Drop unneeded patch\n\n* fixup\n\n* Fix\n\n* fix\n\n* fix tests\n\n* fix\n\n* add missing file\n\n* fix\n\n* fix\n\n* fixup\n\n* bug workaround for gcc\n\n* fix\n\n* fix\n\n* more patches\n\n* fix\n\n* fix\n\n* fix\n\n---------\n\nCo-authored-by: enzyme-ci-bot[bot] <78882869+enzyme-ci-bot[bot]@users.noreply.github.com>\nCo-authored-by: William S. Moses <gh@wsmoses.com>",
          "timestamp": "2025-10-17T19:02:06+08:00",
          "tree_id": "965c16fa81550d4cf0b83351d59e1ede0142b92a",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/6137ac98e710adf6f4e953bf441db4e25b2db40f"
        },
        "date": 1760714912713,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "add_two / JaX   / cpu / Primal",
            "value": 0.0000038976957999693696,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Primal",
            "value": 0.00000598926249986107,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / Forward",
            "value": 0.000008238229199923807,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Forward",
            "value": 0.000008194375000130095,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / BothRev",
            "value": 0.000011886500000036904,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PreRev",
            "value": 0.000009867437499997322,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PostRev",
            "value": 0.000010835137499998382,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / BothRev",
            "value": 0.000013692654199985554,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "wmoses@google.com",
            "name": "William Moses",
            "username": "wsmoses"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a7c38bce984c3adedafb8e03282c0e39640ab6f9",
          "message": "attempt win fix (#1486)\n\n* attempt win fix\n\n* Update workspace.bzl\n\n* fix\n\n* more fix\n\n* more fix",
          "timestamp": "2025-10-18T20:20:11+09:00",
          "tree_id": "24b7fe86c7479eb770306f49263773806c3d2559",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/a7c38bce984c3adedafb8e03282c0e39640ab6f9"
        },
        "date": 1760787522907,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "add_two / JaX   / cpu / Primal",
            "value": 0.0000038976957999693696,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Primal",
            "value": 0.00000598926249986107,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / Forward",
            "value": 0.000008238229199923807,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Forward",
            "value": 0.000008194375000130095,
            "unit": "s"
          },
          {
            "name": "add_two / JaX   / cpu / BothRev",
            "value": 0.000011886500000036904,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PreRev",
            "value": 0.000009867437499997322,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PostRev",
            "value": 0.000010835137499998382,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / BothRev",
            "value": 0.000013692654199985554,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "wmoses@google.com",
            "name": "William Moses",
            "username": "wsmoses"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "bce12a6e157413c6849ddc1b11fe803445015549",
          "message": "Fix canonicalization error on sizes (#1494)",
          "timestamp": "2025-10-21T09:33:35+09:00",
          "tree_id": "c8edd77081f187b451bbf04904a55981b653d22f",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/bce12a6e157413c6849ddc1b11fe803445015549"
        },
        "date": 1761024452217,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004420607999782078,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.0000041063579992624,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001508806882004,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001508275512998,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "238314553+enzymead-bot[bot]@users.noreply.github.com",
            "name": "enzymead-bot[bot]",
            "username": "enzymead-bot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b1a2c3c5fb6053641027d47575f8ff76ec8cf191",
          "message": "Update EnzymeAD/Enzyme to commit bc11256dcf8d36887a80fe422dcb6c02b9a88bd8 (#1491)\n\nDiff: https://github.com/EnzymeAD/Enzyme/compare/f10a216d9a40e2d85547572776a41cd9054fe49d...bc11256dcf8d36887a80fe422dcb6c02b9a88bd8\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-10-21T22:15:07+01:00",
          "tree_id": "ccb7b82635a008d4e863b7d2a69466da325a446f",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/b1a2c3c5fb6053641027d47575f8ff76ec8cf191"
        },
        "date": 1761084761969,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.00000451032500131987,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.00000418276000127662,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001342079442983,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001459766592015,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f5cf0aa4b6121bdc7f930650724a606af2f12abf",
          "message": "fix: always link cinterface (#1497)",
          "timestamp": "2025-10-21T17:18:08-04:00",
          "tree_id": "c636705c4fc6c4da308cc221ceab91753d1c57e1",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/f5cf0aa4b6121bdc7f930650724a606af2f12abf"
        },
        "date": 1761088485728,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004446247199666687,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004468724201433361,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001509301601996,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001429400643013,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "9b1e2a60c0131c796d014d87b277134eff978d26",
          "message": "feat: get_dimension_size batch interface (#1502)",
          "timestamp": "2025-10-21T20:21:20-04:00",
          "tree_id": "bd07a7bfe8108dfed5a2488d39957981e4869639",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/9b1e2a60c0131c796d014d87b277134eff978d26"
        },
        "date": 1761104436481,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004142056999262422,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.0000042869389988482,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001476135021977,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001478948342002,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "fd8e1c1138536dbf933970360fb4c3ab0db88a42",
          "message": "feat: support cluster dims in kernel call op (#1487)\n\n* feat: support cluster dims in kernel call op\n\n* test: clusters",
          "timestamp": "2025-10-21T23:16:31-04:00",
          "tree_id": "8c13f9b836afd0eebfd9f2db4af93387ed2d6a19",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/fd8e1c1138536dbf933970360fb4c3ab0db88a42"
        },
        "date": 1761112259952,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004408689998672344,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004365386901190504,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001303062093997,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.000130511138399,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "238314553+enzymead-bot[bot]@users.noreply.github.com",
            "name": "enzymead-bot[bot]",
            "username": "enzymead-bot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "171bf58a7e24d09de943e133fd6e9bf4991b2cb9",
          "message": "Update EnzymeAD/Enzyme to commit 8db452c3872bc0dce82f82eda26352d0ca3af982 (#1505)\n\nDiff: https://github.com/EnzymeAD/Enzyme/compare/bc11256dcf8d36887a80fe422dcb6c02b9a88bd8...8db452c3872bc0dce82f82eda26352d0ca3af982\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-10-23T00:04:06+01:00",
          "tree_id": "0635381dc444b3a6b84c86ac138ff75838c0b0c4",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/171bf58a7e24d09de943e133fd6e9bf4991b2cb9"
        },
        "date": 1761180445288,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004312986900913529,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004289337899535894,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.000145568572299,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001428322133026,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "238314553+enzymead-bot[bot]@users.noreply.github.com",
            "name": "enzymead-bot[bot]",
            "username": "enzymead-bot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "768f73ead9d3f7d96c9342af70aa5f8fbfe211c4",
          "message": "Update EnzymeAD/Enzyme to commit 989a56376195c78cb1e7d9847945ced50f101dbd (#1509)\n\nDiff: https://github.com/EnzymeAD/Enzyme/compare/8db452c3872bc0dce82f82eda26352d0ca3af982...989a56376195c78cb1e7d9847945ced50f101dbd\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-10-23T19:16:41+02:00",
          "tree_id": "dfdf38fadefc43b3f7df70893a0cb1a0c7569b59",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/768f73ead9d3f7d96c9342af70aa5f8fbfe211c4"
        },
        "date": 1761256834788,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004389633896062151,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004362117004347965,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001478590173006,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001475318573007,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b882f6ef994f8725feaa845065176d2f58960df0",
          "message": "fix: correct broadcast_in_dim result size in dot_general_simplify (#1511)",
          "timestamp": "2025-10-23T18:50:16-04:00",
          "tree_id": "83a50e52e27eff123c417f7d6410725bbf5b86f4",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/b882f6ef994f8725feaa845065176d2f58960df0"
        },
        "date": 1761272084972,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004286419996060431,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004294507997110486,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001479626882006,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001498207881988,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ivanov.i.aa@m.titech.ac.jp",
            "name": "Ivan R. Ivanov",
            "username": "ivanradanov"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "1e54673c056fb0609696ca99da98f69186558738",
          "message": "Fix inlining LLVM functions when we are in a GPU execution context  (#1508)\n\n* Avoid generating allocas on the host\n\n* fix\n\n* add test\n\n* fix test\n\n* fix memory error\n\n* fix\n\n* depdial\n\n* fix\n\n---------\n\nCo-authored-by: William S. Moses <gh@wsmoses.com>",
          "timestamp": "2025-10-23T21:37:56-05:00",
          "tree_id": "0cf05f66f09d989cf04d3e18172d9c05544c2980",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/1e54673c056fb0609696ca99da98f69186558738"
        },
        "date": 1761278493702,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004218969901558012,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004310426797019317,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001654816534952,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001385609487013,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "wmoses@google.com",
            "name": "William Moses",
            "username": "wsmoses"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "21db9536f6a1e5958b7114545feab17e7b5c4872",
          "message": "Don't get empty result (#1512)\n\n* Don't get empty result\n\n* Better err",
          "timestamp": "2025-10-24T07:00:37-05:00",
          "tree_id": "c686d1e062fc5193614732541a9af22b6e7af008",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/21db9536f6a1e5958b7114545feab17e7b5c4872"
        },
        "date": 1761310423798,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004213477001758292,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004224093997618183,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001366508484003,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.000141363441298,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "682c439f6819ccb1e1271324a7baeaf5c18924be",
          "message": "feat: support more data movement ops in `while_is_copy` (#1507)\n\n* feat: support transposed copy\n\n* feat: support non-constant extra indices\n\n* fix: only emit new ops if sure",
          "timestamp": "2025-10-24T10:15:46-04:00",
          "tree_id": "c3951a8aaef79c9179d7235378be399bdc27cdda",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/682c439f6819ccb1e1271324a7baeaf5c18924be"
        },
        "date": 1761319114405,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004394761298317462,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004432443296536803,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001361593089008,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001352883182,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "238314553+enzymead-bot[bot]@users.noreply.github.com",
            "name": "enzymead-bot[bot]",
            "username": "enzymead-bot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "57c7154bd98aa670a51cfe55335a2786d55412c6",
          "message": "Update EnzymeAD/Enzyme to commit 9946ebe14697e2fc56978f2b08aa969cea096e70 (#1514)\n\nDiff: https://github.com/EnzymeAD/Enzyme/compare/989a56376195c78cb1e7d9847945ced50f101dbd...9946ebe14697e2fc56978f2b08aa969cea096e70\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-10-25T16:42:06+01:00",
          "tree_id": "d2633327c9585ab4174f812cd0da8b69e7798fef",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/57c7154bd98aa670a51cfe55335a2786d55412c6"
        },
        "date": 1761413582708,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000005347679002443329,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004560647998005152,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001507630862004,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001501561831973,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "238314553+enzymead-bot[bot]@users.noreply.github.com",
            "name": "enzymead-bot[bot]",
            "username": "enzymead-bot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "57c7154bd98aa670a51cfe55335a2786d55412c6",
          "message": "Update EnzymeAD/Enzyme to commit 9946ebe14697e2fc56978f2b08aa969cea096e70 (#1514)\n\nDiff: https://github.com/EnzymeAD/Enzyme/compare/989a56376195c78cb1e7d9847945ced50f101dbd...9946ebe14697e2fc56978f2b08aa969cea096e70\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-10-25T16:42:06+01:00",
          "tree_id": "d2633327c9585ab4174f812cd0da8b69e7798fef",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/57c7154bd98aa670a51cfe55335a2786d55412c6"
        },
        "date": 1761438916801,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000005347679002443329,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004560647998005152,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001507630862004,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001501561831973,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "wmoses@google.com",
            "name": "William Moses",
            "username": "wsmoses"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "2e44779cdb9067c269e482bca0e5ad947b6057bd",
          "message": "Update ENZYME_COMMIT to new commit hash",
          "timestamp": "2025-10-26T00:42:10-05:00",
          "tree_id": "4aba441f2b59252398d4d7a7e2932743daf153e1",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/2e44779cdb9067c269e482bca0e5ad947b6057bd"
        },
        "date": 1761469868731,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000005347679002443329,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004560647998005152,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001507630862004,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001501561831973,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "bea6deb4331421edc59f5d698535755fe7a09d77",
          "message": "feat: Reshape(Dynamic Slice) (#1515)\n\n* feat: Reshape(Dynamic Slice)\n\n* fix: check for zero start index",
          "timestamp": "2025-10-26T10:46:49-04:00",
          "tree_id": "b2ad8002032b394c6d51ada804e455898d1d2280",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/bea6deb4331421edc59f5d698535755fe7a09d77"
        },
        "date": 1761502943514,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.00000436005899682641,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004256792995147407,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001456125612021,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.000132292057306,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "238314553+enzymead-bot[bot]@users.noreply.github.com",
            "name": "enzymead-bot[bot]",
            "username": "enzymead-bot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ff1a8e32a717060485b0e2db73943117cdf6aa41",
          "message": "Update EnzymeAD/Enzyme to commit b0cafac31726e95279c9d0bc578a00ca5016244a (#1517)\n\nDiff: https://github.com/EnzymeAD/Enzyme/compare/e16832fc50fb70bcdcc9fac822933ef5de8c3847...b0cafac31726e95279c9d0bc578a00ca5016244a\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-10-26T20:36:23-05:00",
          "tree_id": "af8be64859feca6ba19cfd72882502840649aac7",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/ff1a8e32a717060485b0e2db73943117cdf6aa41"
        },
        "date": 1761543826933,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004354738304391504,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.0000043952931999228895,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001377137290895,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001345808843034,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e8153a4135d39eca09e01583d6af43670ee075ee",
          "message": "feat: dynamic_slice_simplify (#1518)",
          "timestamp": "2025-10-27T00:37:21-05:00",
          "tree_id": "55a04009d17efe0c69b8e9f41066bf3453eaeeec",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/e8153a4135d39eca09e01583d6af43670ee075ee"
        },
        "date": 1761555942736,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004351478302851319,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004161173291504383,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001519438668969,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001498814538004,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "scharfrichterq@gmail.com",
            "name": "Acake",
            "username": "sbrantq"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f3d79b36e034cb46f7868adc1fa223a6019130f8",
          "message": "ProbProg: Static HMC (#1464)\n\n* todo\n\n* cleanup\n\n* more mh ops lowering\n\n* test\n\n* select trace/dump op\n\n* cholesky solve op\n\n* multinormal\n\n* GetFlattenedSamplesFromTraceOpConversion  LoopOpConversion UnflattenSliceOpConversion\n\n* split lower-enzyme-probprog in two\n\n* save tests\n\n* save\n\n* simplify",
          "timestamp": "2025-10-27T03:35:57-05:00",
          "tree_id": "0c283d16850205054a637775d7898e9158336be5",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/f3d79b36e034cb46f7868adc1fa223a6019130f8"
        },
        "date": 1761560091250,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.0000042890359996818,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004200187907554209,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001495036462089,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001424800402019,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "238314553+enzymead-bot[bot]@users.noreply.github.com",
            "name": "enzymead-bot[bot]",
            "username": "enzymead-bot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "7717acb02935aa362bfe349b2287dfd5ff5357ef",
          "message": "Update EnzymeAD/Enzyme to commit 63c81877e73b019745ebff33f6dc2e13a5003a17 (#1521)\n\nDiff: https://github.com/EnzymeAD/Enzyme/compare/b0cafac31726e95279c9d0bc578a00ca5016244a...63c81877e73b019745ebff33f6dc2e13a5003a17\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-10-27T12:33:33+01:00",
          "tree_id": "50c77a2f39728a0b8b87dd546a6f6dcd4ec26c44",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/7717acb02935aa362bfe349b2287dfd5ff5357ef"
        },
        "date": 1761568601452,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004274666891433298,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004351996898185462,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001453273741994,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001480237861978,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "fcf54c56ea4b2b8a78b88c7ca88f1ad385916621",
          "message": "feat: support intermediate insertions/deletions while copying (#1513)\n\ntest: missing results\n\nfeat: try more generalization (partial progress)\n\nfix: uncomment\n\nfix: add comment on how to handle remaining cases\n\nfix: remove unwanted code\n\nrevert: unwanted changes\n\nfeat: handle transpose without explicit transpose op\n\nchore: run fmt\n\nchore: remove old comment",
          "timestamp": "2025-10-27T08:46:33-04:00",
          "tree_id": "0fc39fcc5d7527063e5dd5a1a7ebeae9617a17c7",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/fcf54c56ea4b2b8a78b88c7ca88f1ad385916621"
        },
        "date": 1761574023301,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004405523999594152,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004346182988956571,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001421422843006,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001422737602959,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "765740+giordano@users.noreply.github.com",
            "name": "Mosè Giordano",
            "username": "giordano"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f0c1e1bc526fee6281db47fdeff2bd50f17b4469",
          "message": "Update JAX to commit eac1414e85f61667591b5879bbb95f80d609f23f (#1484)\n\n* Update JAX to commit eac1414e85f61667591b5879bbb95f80d609f23f\n\nDiff: https://github.com/jax-ml/jax/compare/3a22eea644237001df0f3dd42253225cc059b43c...eac1414e85f61667591b5879bbb95f80d609f23f\n\n* Disable x86 AMX support in Bazel configuration\n\n* Log compiler versions in build workflow\n\nAdded commands to display GCC, CC, and Clang versions.\n\n* Fix Bazel flags assignment in build.yml\n\n* Update build.yml\n\n* fix\n\n* fmt\n\n* Update .bazelrc\n\n* Update .bazelrc\n\n* Update JAX_COMMIT hash in workspace.bzl\n\n* Update .bazelrc\n\n* Remove compiler version checks from build.yml\n\nRemoved version checks for gcc, cc, and clang from the build workflow.\n\n* Update .bazelrc\n\n* Update JAX_COMMIT to new commit hash\n\n---------\n\nCo-authored-by: enzyme-ci-bot[bot] <78882869+enzyme-ci-bot[bot]@users.noreply.github.com>\nCo-authored-by: William Moses <wmoses@google.com>\nCo-authored-by: William S. Moses <gh@wsmoses.com>",
          "timestamp": "2025-10-27T17:27:47-05:00",
          "tree_id": "f83ee8cedecfd38343a1c34a21db8f7f6a5bbe7e",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/f0c1e1bc526fee6281db47fdeff2bd50f17b4469"
        },
        "date": 1761618411603,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000003881685994565487,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004159397992771119,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001516026142984,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001506869731936,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "992f348493d08666a9bd4305fdbfc35ff6910c0b",
          "message": "feat: greedy loop to batch (#1425)\n\n* feat: greedy loop to batch via loop fission\n\nfeat: support non-constant extra indices\n\nfix: only emit new ops if sure\n\nfeat: propagate bounds and eliminate noops\n\nchore: run fmt\n\nfeat: check for dynamic_slices\n\nfeat: unary elementwise working\n\nchore: comments\n\nfeat: clear out unwanted no-ops from loop body\n\nfeat: elementwise ops are completely supported\n\nchore: add a note\n\nfix: bad rebase\n\nfeat: generalize index handling\n\ntest: elementwise loop fission\n\nrefactor: support dynamicsliceop\n\nfeat: sliceinfo generalize\n\nfeat: lift ops by batch op\n\nfeat: run batching pass\n\nfeat: generalize handling for non-reshape case\n\n* chore: make a note on supporting dropdim in while is copy\n\n* feat: support while is copy with dropdims = 0\n\n* feat: add to transform ops\n\n* fix: mapping\n\n* feat: support constants by lifting them into function body\n\n* feat: special handling of reshape",
          "timestamp": "2025-10-27T19:49:19-04:00",
          "tree_id": "0253de51884ecf5ff2b226afa85c8da3ee5327b6",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/992f348493d08666a9bd4305fdbfc35ff6910c0b"
        },
        "date": 1761633458474,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004391519189812243,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004307569190859795,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001399941653944,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.000138530330907,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "wmoses@google.com",
            "name": "William Moses",
            "username": "wsmoses"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "8f6fcba733909db2ed5fe0969f7544b612357de6",
          "message": "Statically linked cuda libs (#1516)\n\n* Statically linked cuda libs\n\n* Update build-reactantjll.yml\n\n* fix\n\n* Update ML_TOOLCHAIN_COMMIT to new commit hash\n\n* nvperf\n\n* Update build-reactantjll.yml",
          "timestamp": "2025-10-28T00:51:20-05:00",
          "tree_id": "518ffa0906346414c23bf31a38e5c651cd326151",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/8f6fcba733909db2ed5fe0969f7544b612357de6"
        },
        "date": 1761650741472,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004677987098693847,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004717439110390842,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001567242303979,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001577116009895,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c4ad72b48c57c448e7fe6443e1d7eb94256be40b",
          "message": "fix: batch reduce region correctly with external constants (#1526)\n\n* fix: batch constantOp correctly\n\n* feat: support cloning reduce op region with outside constants",
          "timestamp": "2025-10-28T13:34:31-04:00",
          "tree_id": "3aac18ccaad8890b16e198baf336f8d5566f838b",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/c4ad72b48c57c448e7fe6443e1d7eb94256be40b"
        },
        "date": 1761682308983,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004199580405838788,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004190525400917977,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001350029852939,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001341407105908,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "xuyuansui@outlook.com",
            "name": "xys-syx",
            "username": "xys-syx"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "8dc549db2c67dd4940743b6fbca96af2cab41de5",
          "message": "Cpuify (#1415)\n\n* Port cpuify pass from polygeist\n\n* add missing files\n\n* add cpuify tests\n\n* fmt\n\n* lit tests modified\n\n* Apply clang-format\n\n* fmt\n\n* fix\n\n* Fmt\n\n* fix\n\n* fix\n\n* fix looprecur\n\n* temp using DAG in cpuifyifsplit.mlir\n\n* fmt\n\n* fmt\n\n* fmt\n\n* fmt\n\n* fix\n\n* fix\n\n* fix\n\n* fix\n\n* fix\n\n* fix\n\n* fmt:\n\n* fmt\n\n* fix\n\n* fix\n\n* fix\n\n* fix\n\n* fix\n\n* fmt\n\n* fix\n\n* fmt\n\n* fix\n\n* macos\n\n* macos\n\n* remove\n\n* remove\n\n* Fix memory error\n\n* void\n\n* fix\n\n* fix\n\n* fmt\n\n* fmt\n\n* wrapper for LoopDistribute Specific Logic\n\n* fmt\n\n---------\n\nCo-authored-by: William S. Moses <gh@wsmoses.com>",
          "timestamp": "2025-10-28T14:05:05-05:00",
          "tree_id": "bc51ee9ff8ec53ad8ca91555493034524bc71705",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/8dc549db2c67dd4940743b6fbca96af2cab41de5"
        },
        "date": 1761691433845,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004480922990478576,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004438424995169044,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001398530440987,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001646234500105,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "238314553+enzymead-bot[bot]@users.noreply.github.com",
            "name": "enzymead-bot[bot]",
            "username": "enzymead-bot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "03a920be9c30e11ed343f962896b77511b1baf3d",
          "message": "Update EnzymeAD/Enzyme to commit aecafad17d3f63c1c0697609ad713615af322e23 (#1524)\n\nDiff: https://github.com/EnzymeAD/Enzyme/compare/63c81877e73b019745ebff33f6dc2e13a5003a17...aecafad17d3f63c1c0697609ad713615af322e23\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-10-28T18:48:46-05:00",
          "tree_id": "d731583dc274a9258bbfd4f0aa1de0828dd56a1b",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/03a920be9c30e11ed343f962896b77511b1baf3d"
        },
        "date": 1761704648521,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.00000432896800339222,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004337205993942916,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.000133810721198,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001360229832003,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "fbaf8a8fee101527aeb30454651d3df06a7ea3b1",
          "message": "feat: generalize elementwise to reduction (#1528)\n\n* feat: generalize elementwise to reduction\n\n* chore: run format",
          "timestamp": "2025-10-28T19:54:54-04:00",
          "tree_id": "419538688ceca40b652cbeac75d4cb27763d046e",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/fbaf8a8fee101527aeb30454651d3df06a7ea3b1"
        },
        "date": 1761710841933,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004310710995923728,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004211899999063462,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.000149751619203,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001499467342044,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "238314553+enzymead-bot[bot]@users.noreply.github.com",
            "name": "enzymead-bot[bot]",
            "username": "enzymead-bot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "be41edd949d05733c12b59b582b14f6c9ef462c1",
          "message": "Update EnzymeAD/Enzyme to commit fdc9f4a7266c986e0efe4dbce9d9446471f640ae (#1530)\n\nDiff: https://github.com/EnzymeAD/Enzyme/compare/aecafad17d3f63c1c0697609ad713615af322e23...fdc9f4a7266c986e0efe4dbce9d9446471f640ae\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-10-29T13:23:02+01:00",
          "tree_id": "d1709e21d05246a26712671e42c4711438d78d04",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/be41edd949d05733c12b59b582b14f6c9ef462c1"
        },
        "date": 1761755145469,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004200157907325775,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004260525992140174,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001329022263991,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001370554722961,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "238314553+enzymead-bot[bot]@users.noreply.github.com",
            "name": "enzymead-bot[bot]",
            "username": "enzymead-bot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "54fcf287b1f2c1312116a877ef233010c172bab2",
          "message": "Update EnzymeAD/Enzyme to commit 3048141052e0d9bb497950803e74c24bd5ea8ca1 (#1531)\n\nDiff: https://github.com/EnzymeAD/Enzyme/compare/fdc9f4a7266c986e0efe4dbce9d9446471f640ae...3048141052e0d9bb497950803e74c24bd5ea8ca1\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-10-30T12:54:46Z",
          "tree_id": "bde29d185a7dc3b99c0812260cc780887675ae3b",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/54fcf287b1f2c1312116a877ef233010c172bab2"
        },
        "date": 1761832717694,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004316021304111928,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004213109402917325,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001468284287024,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001375977221992,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "34986820+EganJ@users.noreply.github.com",
            "name": "Egan",
            "username": "EganJ"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "12bca85ccbf361c5c7d3017c0b77af1a5e8cf30f",
          "message": "Bump hedron hash to fix compile commands issue (#1423)",
          "timestamp": "2025-10-30T15:11:13-05:00",
          "tree_id": "5ebb8010a494981720e1935fac94bcb7567a6b3c",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/12bca85ccbf361c5c7d3017c0b77af1a5e8cf30f"
        },
        "date": 1761868531450,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000006158093002159148,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004402495000977069,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001326389673049,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001374468563008,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "gh@wsmoses.com",
            "name": "William S. Moses",
            "username": "wsmoses"
          },
          "committer": {
            "email": "gh@wsmoses.com",
            "name": "William S. Moses",
            "username": "wsmoses"
          },
          "distinct": true,
          "id": "0a93f68f15e22ef5711857db0d050e5e66680694",
          "message": "separately consider cudnn",
          "timestamp": "2025-10-30T17:37:25-05:00",
          "tree_id": "5c03ad0eee6772a86d5d5cb605333d09a72570ee",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/0a93f68f15e22ef5711857db0d050e5e66680694"
        },
        "date": 1761886252909,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004365037300158292,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004191758390516042,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001382287155021,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001379254845087,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "wmoses@google.com",
            "name": "William Moses",
            "username": "wsmoses"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "77a8d3d05359d470c6a103e643d129a430722d2a",
          "message": "Add remove enzyme ops to pass list (#1536)",
          "timestamp": "2025-10-30T19:16:32-05:00",
          "tree_id": "974530eb996f1c910b3fe852ad8f045e671d1111",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/77a8d3d05359d470c6a103e643d129a430722d2a"
        },
        "date": 1761901041164,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004398716997820884,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004282150999642908,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001329769102972,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001306660962989,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "238314553+enzymead-bot[bot]@users.noreply.github.com",
            "name": "enzymead-bot[bot]",
            "username": "enzymead-bot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "73ee4b92e04a17fc5ec6c2663e7fecb1ac98bd6e",
          "message": "Update jax-ml/jax to commit f6edb899a75dad482a687fe42c8a81d1eaa3aac5 (#1539)\n\nDiff: https://github.com/jax-ml/jax/compare/7d256f847561f57f290b74468b99a8b0da79cd1a...f6edb899a75dad482a687fe42c8a81d1eaa3aac5\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-11-01T13:19:18Z",
          "tree_id": "166b954f7e341ecc89a6be8d03ad421e69dbe0bf",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/73ee4b92e04a17fc5ec6c2663e7fecb1ac98bd6e"
        },
        "date": 1762015537511,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004233665904030204,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.0000041352519067004326,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001391577433096,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001395150692202,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "238314553+enzymead-bot[bot]@users.noreply.github.com",
            "name": "enzymead-bot[bot]",
            "username": "enzymead-bot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c432749dde4d6ca23bb358cc7dfcfd54cfefc3b1",
          "message": "Update jax-ml/jax to commit f386951c08a82781f061ef3ebb3ff28e7a617ead (#1543)\n\nDiff: https://github.com/jax-ml/jax/compare/f6edb899a75dad482a687fe42c8a81d1eaa3aac5...f386951c08a82781f061ef3ebb3ff28e7a617ead\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-11-01T21:54:11-05:00",
          "tree_id": "38739e5f995e75e6c142dc62768a698e08d240f7",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/c432749dde4d6ca23bb358cc7dfcfd54cfefc3b1"
        },
        "date": 1762076905340,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004386042291298509,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004269602289423346,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001422204643022,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001438122397055,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "wmoses@google.com",
            "name": "William Moses",
            "username": "wsmoses"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "4a22980dd3b3ffa65e81c8273b41483035f8ea02",
          "message": "Fix infinite loop ori disjoint (#1547)",
          "timestamp": "2025-11-02T09:30:18-06:00",
          "tree_id": "65f0f4954e15fc5dc09b97acc7a63e5bbde16aed",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/4a22980dd3b3ffa65e81c8273b41483035f8ea02"
        },
        "date": 1762108884746,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.0000043374820001190525,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.00000406218699990859,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001517361998001,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001501404909,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b3d6fde1c3d150e8c0be3363c9e234e349aa13cd",
          "message": "feat: iota tensor detection + indirect iota indexing simplification (#1542)\n\n* feat: iota tensor detection\n\n* chore: run fmt\n\n* feat: rewrite iota ops\n\n* test: indirect indexing\n\n* feat: support more iota like ops for scatter detection",
          "timestamp": "2025-11-02T11:06:12-05:00",
          "tree_id": "608a05ee69d0011cda2f5c2d0fdb0942828d6f44",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/b3d6fde1c3d150e8c0be3363c9e234e349aa13cd"
        },
        "date": 1762120582721,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004345976980403066,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004376305988989771,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001504514841828,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001366461332887,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "4b250056b4c4257c5be6f10a402a1c1099b3e207",
          "message": "feat: dot_general_licm (#1550)",
          "timestamp": "2025-11-02T15:15:56-05:00",
          "tree_id": "63eba01503f1436bee5a133add791d44b5289dad",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/4b250056b4c4257c5be6f10a402a1c1099b3e207"
        },
        "date": 1762134942837,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004289858999982243,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.0000041430409000895455,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001352644156999,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001352168267003,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "avikpal@mit.edu",
            "name": "Avik Pal",
            "username": "avik-pal"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "319852ba6108e8a651f411ee816bec27a65ddc7c",
          "message": "fix: guaranteed analysis bfs queue was busted (#1538)\n\n* fix: guaranteed analysis bfs queue was busted\n\n* chore: fmt\n\n* fix: try making things deterministic\n\n* chore: bazel format\n\n* fix: address review comments\n\n* fix: revert to SmallPtrSet\n\n* chore: run fmt\n\n* fix: reuse found value",
          "timestamp": "2025-11-02T20:36:13-05:00",
          "tree_id": "a2fe7ef64c992f29e1b72db005bf2039427cb3b1",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/319852ba6108e8a651f411ee816bec27a65ddc7c"
        },
        "date": 1762149886304,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.0000043059209827333686,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004307427001185715,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001530488142045,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001533427932066,
            "unit": "s"
          }
        ]
      }
    ]
  }
}