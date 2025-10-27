window.BENCHMARK_DATA = {
  "lastUpdate": 1761574024036,
  "repoUrl": "https://github.com/EnzymeAD/Enzyme-JAX",
  "entries": {
    "EnzymeJAX Benchmarks": [
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
          "id": "2a2f7b833596f3a0ff90eddd73838defc58b4243",
          "message": "ReverseCache: use bfs/cache (#1429)\n\n* ReverseCache: use bfs/cache\n\n* fix\n\n* more work\n\n* topo\n\n* fix\n\n* fix",
          "timestamp": "2025-09-26T21:02:19-05:00",
          "tree_id": "b7da5ffb0e67c133121b659b5a6f87e53ed13f42",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/2a2f7b833596f3a0ff90eddd73838defc58b4243"
        },
        "date": 1758959251588,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004358400008641183,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004314248892478645,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001527166346786,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001537285756086,
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
          "id": "87f3bf4242e04eb3e0781e9ce8a405999d939169",
          "message": "feat: concat reshape elem (#1432)",
          "timestamp": "2025-09-27T09:26:57-04:00",
          "tree_id": "c2648fdef88335c5a39d47fdb422e9aed2a4b337",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/87f3bf4242e04eb3e0781e9ce8a405999d939169"
        },
        "date": 1758984406866,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004505915900517721,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.0000044609560005483215,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001454494060992,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001406824400997,
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
          "id": "c8efb465dfeeb6577a4a0dc60cf2b42eb1e5d55f",
          "message": "feat: update AutoBatching passes to be CheckedRewrite (#1434)",
          "timestamp": "2025-09-27T21:14:14-04:00",
          "tree_id": "bc5f5264036addbf4096d12bbb140d4973761443",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/c8efb465dfeeb6577a4a0dc60cf2b42eb1e5d55f"
        },
        "date": 1759025502544,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.0000041507179994368925,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004372967997915112,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001458673020999,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001442531612003,
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
          "id": "14bfc935f481d25da64cdd680bf831cbb19ef65b",
          "message": "feat: dus to dynamic_pad + dynamic_pad to pad (#1430)\n\n* feat: dus to dynamic_pad\n\n* feat: dynamic_pad to pad\n\n* chore: comments\n\n* test: restrict applicability\n\n* feat: add to primitives",
          "timestamp": "2025-09-28T13:58:26-04:00",
          "tree_id": "a08f416c464c826dc7c72aa4624afe4883605dba",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/14bfc935f481d25da64cdd680bf831cbb19ef65b"
        },
        "date": 1759086063094,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004376951899030246,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004339134899782948,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001438124105974,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001403865636006,
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
          "id": "a8202a577d32b74f9e83776cabc9334827852bcd",
          "message": "Fix case where path is not on inverted (#1436)",
          "timestamp": "2025-09-28T13:52:35-05:00",
          "tree_id": "59b4c3cbb3e7d0bbb1c78232826c55d472a3ce6c",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/a8202a577d32b74f9e83776cabc9334827852bcd"
        },
        "date": 1759093522055,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004343702000915073,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004178517899708822,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001543433807994,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001460785379022,
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
          "id": "ccfcd699469d7244f103ef678cd9ed663bb24fd0",
          "message": "Speed up local nan (#1437)",
          "timestamp": "2025-09-28T17:02:02-05:00",
          "tree_id": "003115868532e4d210deff171c9fec4828c30855",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/ccfcd699469d7244f103ef678cd9ed663bb24fd0"
        },
        "date": 1759100421974,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004269564000423998,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004272056900663301,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001412289372005,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001434171072003,
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
          "id": "7f88655975c206c36b5749dbce31418b9e6b305e",
          "message": "Polygeist: handle occupancy and related utilities (#1395)\n\n* WIP: occupance\n\n* wip\n\n* b\n\n* occ op\n\n* fmt\n\n* fix\n\n* fix\n\n* fix\n\n* fix\n\n* no comdat\n\n* More kernel support\n\n* fix\n\n* attr\n\n* Fix\n\n* fix\n\n* fix\n\n* fn attr\n\n* fix\n\n* fix\n\n* fix\n\n* fix",
          "timestamp": "2025-10-01T22:54:15-05:00",
          "tree_id": "bba1dbc0e65e9b510ab8dfc3d833c253475a9e42",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/7f88655975c206c36b5749dbce31418b9e6b305e"
        },
        "date": 1759389828201,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004144166014157235,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004226277908310294,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001389866188168,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001376257929019,
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
          "id": "936ed09be87374323f0d96f1041478262eeea274",
          "message": "feat: function argument memory effects for kernels (#1439)\n\n* feat: function argument memory effects for kernels\n\n* chore: run fmt\n\n* feat: use llvm dialect attributes\n\n* feat: merge the arg effects pass into func effects\n\n* fix: correct use of CallOpInterface + only mark llvm attrs for ptrs\n\n* fix: avoid readnone for now (drop me)",
          "timestamp": "2025-10-02T15:26:08-04:00",
          "tree_id": "02accefac1a279e505c9944ac06b2191ea7116e9",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/936ed09be87374323f0d96f1041478262eeea274"
        },
        "date": 1759436634203,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004362944897729904,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004383776907343417,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001482813360053,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001463136211037,
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
          "id": "67dc89667de4bdf4737b577f2bb29f32ef25c1db",
          "message": "feat: allow symbolref for `kernel_call`/`jit_call` (#1438)\n\n* feat: allow symbolref for kernel_call/jit_call\n\n* chore: run fmt\n\n* test: use specific reactant branch",
          "timestamp": "2025-10-02T17:50:02-04:00",
          "tree_id": "d5e3c87b96052e04a120361790948efe8b0cf295",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/67dc89667de4bdf4737b577f2bb29f32ef25c1db"
        },
        "date": 1759445244750,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.00000435243200045079,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004385623894631863,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001425868527963,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001460132357897,
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
          "id": "9b689ed3e1a8a2d21a43d489232cf3dbdbccab13",
          "message": "fix: restrict broadcast in dim check for auto-batching (#1444)",
          "timestamp": "2025-10-03T13:51:43-04:00",
          "tree_id": "7f6986aa35ad0300c8f10ecd0abdf1808dc102fb",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/9b689ed3e1a8a2d21a43d489232cf3dbdbccab13"
        },
        "date": 1759517351220,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004480150900781155,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004280475992709399,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001693931275978,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001651718516135,
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
          "id": "502d6b59542a6d739f0f42829b151bf2c1e3adb6",
          "message": "fix: correctly set symrefattr for kernel/jit call (#1448)",
          "timestamp": "2025-10-04T14:36:00-04:00",
          "tree_id": "4cacf805b8f182707776ec46e1f9a3468ee15955",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/502d6b59542a6d739f0f42829b151bf2c1e3adb6"
        },
        "date": 1759616326076,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004412847000639886,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004364896903280169,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001575975034036,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001493998484918,
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
          "id": "42cde4f9e761a78fad169030c4895ab860fc872d",
          "message": "fix: nested module mem effects + conservative effects (#1449)\n\n* fix: mark memory effects for nested modules correctly\n\n* fix: be conservative if effect getValue is null\n\n* fix: correctly use nested references",
          "timestamp": "2025-10-04T21:11:43-04:00",
          "tree_id": "42f38264163ddcc4023e5291f574860a0885c344",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/42cde4f9e761a78fad169030c4895ab860fc872d"
        },
        "date": 1759636239092,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004279392980970443,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004272616002708673,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.000151909063803,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001518488608067,
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
          "id": "9861ed866a3c9a64c860a50583615d3846ea337f",
          "message": "feat: drop unsupported attributes before XLA (#1452)",
          "timestamp": "2025-10-04T21:39:00-04:00",
          "tree_id": "ec3499402fb6245eb76873160a4c01b692939a07",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/9861ed866a3c9a64c860a50583615d3846ea337f"
        },
        "date": 1759639849650,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004414183995686472,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004328934894874692,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001604812095873,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001494624446844,
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
          "id": "db993e6a5ab93f831da937a42651b34fc1ac4248",
          "message": "Optional debug info dump (#1454)\n\n* Optional debug info dump\n\n* fix\n\n* Fix\n\n* fix\n\n* Fix inlining\n\n* fix\n\n* fix\n\n* fix\n\n* fix\n\n* fix",
          "timestamp": "2025-10-05T11:59:53-05:00",
          "tree_id": "5c371777530c022cb6771729e28f0ceb068aa471",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/db993e6a5ab93f831da937a42651b34fc1ac4248"
        },
        "date": 1759688142195,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004532032000133768,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004480679900007089,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001502536728003,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001425300647999,
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
          "id": "54edf3845b103a7bfcc335f186ee720a1953a65b",
          "message": "SROA pass error (#1456)",
          "timestamp": "2025-10-05T17:46:09-05:00",
          "tree_id": "0b1c2f86927a1376b6cb355d567753e1cbb5e5a5",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/54edf3845b103a7bfcc335f186ee720a1953a65b"
        },
        "date": 1759714867513,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004485886999464128,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004439690998697188,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001577955517001,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001410659988992,
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
          "id": "2064d83426f78d51611ab767deba28f500f5b1fc",
          "message": "feat: propagate bounds in whileop (#1453)\n\n* feat: remove remainder in whileop\n\n* feat: propagate bounds and eliminate noops\n\n* chore: run fmt\n\n* feat: propagate div bounds\n\n* fix: remainder check\n\n* test: negative step\n\n* fix: add iteration count check\n\n* fix: value\n\n* chore: add some comment for each check",
          "timestamp": "2025-10-05T20:58:37-04:00",
          "tree_id": "1c7ef4d4bb5529ba8de488679bfddac0874e24d4",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/2064d83426f78d51611ab767deba28f500f5b1fc"
        },
        "date": 1759721873673,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004382277900003828,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004261513000528794,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001337104498001,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001333137927998,
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
          "id": "e1a47537673d80f0717138b86b1be6b7025851a7",
          "message": "feat: simplify copy-like dynamic_update_slice inside while loop (#1459)\n\n* feat: simplify copy-like dynamic_update_slice inside while loop\n\n* chore: run fmt\n\n* fix: simplify check",
          "timestamp": "2025-10-06T12:17:43-04:00",
          "tree_id": "8bd254a01f181515762b4e49ac0caf4d0e665a89",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/e1a47537673d80f0717138b86b1be6b7025851a7"
        },
        "date": 1759771908016,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004364686901681125,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004358776999288238,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001297847638023,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001297496418002,
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
          "id": "c7e4ccff2a29ee37f1931f4048bdacadb1f707e9",
          "message": "feat: CAPI to create attributes (#1460)\n\n* feat: CAPI to create attributes\n\n* chore: run formatter",
          "timestamp": "2025-10-06T21:32:54-04:00",
          "tree_id": "28563ad99cf4d9bd24205f43fcabf80d171e2448",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/c7e4ccff2a29ee37f1931f4048bdacadb1f707e9"
        },
        "date": 1759818240063,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004340586901525967,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004408594997948967,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001549135915993,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001553631844988,
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
      }
    ]
  }
}