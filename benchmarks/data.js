window.BENCHMARK_DATA = {
  "lastUpdate": 1764187660393,
  "repoUrl": "https://github.com/EnzymeAD/Enzyme-JAX",
  "entries": {
    "EnzymeJAX Benchmarks": [
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
          "id": "ebf0544faa96f7eff3bee167a6c42b62f89f9c82",
          "message": "More rocm fix",
          "timestamp": "2025-11-06T16:22:38-06:00",
          "tree_id": "514529a591ad5cbddd4460d0deb8e3d3227cda95",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/ebf0544faa96f7eff3bee167a6c42b62f89f9c82"
        },
        "date": 1762477806027,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004396587099472526,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004314664100820664,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001368273229993,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001318798172011,
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
          "id": "51d94b351ff8aaf7c6f3aae1c949f3611c394c7d",
          "message": "feat: sort autodiff rules (#1584)\n\n* feat: sort forward mode AD\n\n* feat: reverse mode\n\n* refactor: move the common function\n\n* fix: derivative rule\n\n* test: update\n\n* fix: replace in cacheValues",
          "timestamp": "2025-11-06T18:55:07-05:00",
          "tree_id": "2678042ccacdf96ce90fe6cf038902e1cd435602",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/51d94b351ff8aaf7c6f3aae1c949f3611c394c7d"
        },
        "date": 1762485258317,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004135624998889398,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004203124999185093,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001434340536012,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001441685565005,
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
          "id": "2347ef5292dc2c2c7c3a9d3d987540ef68e3a967",
          "message": "Update EnzymeAD/Enzyme to commit 68e62fbd1b496a60490266952916112b63e62a18 (#1588)\n\nDiff: https://github.com/EnzymeAD/Enzyme/compare/6b1848d8582e57dd57c0bb5d0c373c5cb1c1bbfb...68e62fbd1b496a60490266952916112b63e62a18\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-11-07T11:58:31Z",
          "tree_id": "f802eb19af9d2a3c6d2f94d0a7d7f8b0982184b9",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/2347ef5292dc2c2c7c3a9d3d987540ef68e3a967"
        },
        "date": 1762524211386,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004325116000836715,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004328913100471254,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001335319516001,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001367936525988,
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
          "id": "8c0491d9ca738386e10d1a79aa47a7b738ce0f79",
          "message": "refactor: move common functionalities out of Ops.cpp (#1590)\n\n* refactor: move common functionalities out of Ops.cpp\n\n* fix: missing build deps",
          "timestamp": "2025-11-07T11:28:32-05:00",
          "tree_id": "f69b7c48ed31398d2849b718eed4dd6fa1396b91",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/8c0491d9ca738386e10d1a79aa47a7b738ce0f79"
        },
        "date": 1762537338016,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004395995000959374,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004284164001001045,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001479914346011,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001657945766986,
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
          "id": "6a0abebb67ddc460dabf6b57de373a2682d5493c",
          "message": "feat: support general intermediate reshape in auto-batching (#1575)\n\n* feat: exploit the batch_group_dim for batching kernel\n\n* fix: mimic jax batching\n\n* feat: support multiple dropdims intermediate\n\n* feat: support general intermediate reshapes",
          "timestamp": "2025-11-07T11:37:04-05:00",
          "tree_id": "ffafe9296a84e1dc74f0ed9751ff00883676219e",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/6a0abebb67ddc460dabf6b57de373a2682d5493c"
        },
        "date": 1762544289842,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000003918430398334749,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004305170100997202,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001320323591993,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001326188359991,
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
          "id": "32eeef933b3df77237434cc5f3ceed3de12772a4",
          "message": "feat: support partial indexing to batch (#1587)\n\n* feat: exploit the batch_group_dim for batching kernel\n\n* fix: mimic jax batching\n\n* feat: support partial indexing to batch\n\n* chore: run fmt",
          "timestamp": "2025-11-07T13:39:48-05:00",
          "tree_id": "6c383208a1e82e32602cfed2fb8e4c619a2923e0",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/32eeef933b3df77237434cc5f3ceed3de12772a4"
        },
        "date": 1762549163679,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004371509997872636,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004321837998577393,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001365762186993,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001342667606018,
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
          "id": "990b6706bcbae8fdd7d339f034bb93eda38d1223",
          "message": "feat: triton_ext dialect (#1591)\n\n* feat: triton_ext dialect\n\n* fix: remove nvgpu for now\n\n* chore: run fmt\n\n* fix: namespace\n\n* chore: run fmt\n\n* test: add simple add_kernel test",
          "timestamp": "2025-11-07T14:48:39-05:00",
          "tree_id": "eafe09eaf8f1731f2c181ffc5afb95e2f6cbe5b5",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/990b6706bcbae8fdd7d339f034bb93eda38d1223"
        },
        "date": 1762555714172,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.00000438906199997291,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004388130101142451,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001394996970979,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001417176780989,
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
          "id": "6e58b86686d04b2f803925a9ecc0021fbec1102d",
          "message": "fix: infinite raising + correct broadcast shape (#1593)",
          "timestamp": "2025-11-07T19:06:25-05:00",
          "tree_id": "b272fd5ada5ff9fd9021bf78fc923c36c99bdeda",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/6e58b86686d04b2f803925a9ecc0021fbec1102d"
        },
        "date": 1762563418795,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004448425001464784,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.00000446261799952481,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001502956085983,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001499391246004,
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
          "id": "24641f33ea39c03c45b07ce99e2881ad50128538",
          "message": "Update jax-ml/jax to commit 30e565311af559569b4842bddced4b461f21dd73 (#1586)\n\n* Update jax-ml/jax to commit 30e565311af559569b4842bddced4b461f21dd73\n\nDiff: https://github.com/jax-ml/jax/compare/eb2d56b99ceef305933d9293d5c1715fdb333950...30e565311af559569b4842bddced4b461f21dd73\n\n* fix\n\n* remove rocm tmpdir patch\n\n---------\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>\nCo-authored-by: William S. Moses <gh@wsmoses.com>",
          "timestamp": "2025-11-07T22:51:51-06:00",
          "tree_id": "289469ea040dab7ed40380ae1a6f4377d7fa9e5c",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/24641f33ea39c03c45b07ce99e2881ad50128538"
        },
        "date": 1762585132339,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004388645000290125,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004541678001987748,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001507710175996,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001508200245996,
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
          "id": "c9ad2b9fadb847b303202e2fc5239516bb7f7a92",
          "message": "fix (#1595)",
          "timestamp": "2025-11-09T00:27:44-06:00",
          "tree_id": "c5118b4ea8659305f17c0022511e85b74b0f128e",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/c9ad2b9fadb847b303202e2fc5239516bb7f7a92"
        },
        "date": 1762684872453,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000008794522285461425,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000009516980312764643,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / gpu / Primal",
            "value": 0.00007948229289613665,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / gpu / Primal",
            "value": 0.00007941056929994375,
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
          "id": "c86285ad203b75530dcb5107c131f0820a00ab70",
          "message": "feat: support iota with offset (#1599)",
          "timestamp": "2025-11-09T23:22:30-05:00",
          "tree_id": "3af779cf0ba46380c2ed7c0ad0ed1ee56c34dab3",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/c86285ad203b75530dcb5107c131f0820a00ab70"
        },
        "date": 1762761373801,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004515824798727408,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004259957897011191,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001276962814969,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001333710766979,
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
          "id": "3bb71d558e5cdd7a8150fa7b0a94a600643202d5",
          "message": "Update EnzymeAD/Enzyme to commit 300e6f7913407b1216bf07b5ea8827aae963c898 (#1597)\n\nDiff: https://github.com/EnzymeAD/Enzyme/compare/68e62fbd1b496a60490266952916112b63e62a18...300e6f7913407b1216bf07b5ea8827aae963c898\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-11-10T01:22:53-06:00",
          "tree_id": "b7bc9cfcf0b203555a8f2b5082939b8778dfd618",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/3bb71d558e5cdd7a8150fa7b0a94a600643202d5"
        },
        "date": 1762768111720,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004388973099412396,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.00000432439599535428,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001340801675978,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001337204235023,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "naydex.mc+github@gmail.com",
            "name": "Paul Berg",
            "username": "Pangoraw"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "47d57c1cea7b24e210ad75aee6e7c3f93d89ff78",
          "message": "Triton forward mode AD (#1578)\n\n* Triton forward mode AD\n\n* fmt build\n\n* header\n\n* Bump enzyme\n\n* Update TritonDerivatives.td\n\n---------\n\nCo-authored-by: Mosè Giordano <765740+giordano@users.noreply.github.com>\nCo-authored-by: William Moses <wmoses@google.com>",
          "timestamp": "2025-11-10T09:26:15+01:00",
          "tree_id": "8ccf271a1cecd218e105271802b5b5bd983f7fcd",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/47d57c1cea7b24e210ad75aee6e7c3f93d89ff78"
        },
        "date": 1762785887758,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000003993225202430039,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.00000422822319669649,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001449365721025,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001456860908016,
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
          "id": "98f3f359eddb61127d85fe7e814af655e801692b",
          "message": "feat: partial support for broadcast elementwise trait (#1602)",
          "timestamp": "2025-11-10T14:33:32-05:00",
          "tree_id": "d80d63234619f36083ca084e0113bb51f541b5b6",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/98f3f359eddb61127d85fe7e814af655e801692b"
        },
        "date": 1762812432988,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004391877097077668,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004393432103097439,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001304379814013,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001292214179993,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "naydex.mc+github@gmail.com",
            "name": "Paul Berg",
            "username": "Pangoraw"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "7d0b3924ab6ec0bf8417fc2a4b0376a59b9f6207",
          "message": "pass to raise triton custom call (#1601)\n\n* pass to raise triton custom call\n\n* Buildifier\n\n* patches\n\n* comment\n\n* fmt\n\n* deps",
          "timestamp": "2025-11-11T08:47:59+01:00",
          "tree_id": "48fe906963826b5b738c5e5afc3a8f0dcb12ba98",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/7d0b3924ab6ec0bf8417fc2a4b0376a59b9f6207"
        },
        "date": 1762853493637,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004253403999609873,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004486645001452416,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001332565265009,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001343388594978,
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
          "id": "8a0664ae8dd09bad32cef3f611ad51ca62a9dba2",
          "message": "Update EnzymeAD/Enzyme to commit 269207f268bdda2b46e6c6bc1646bb2195eee0f2 (#1606)\n\nDiff: https://github.com/EnzymeAD/Enzyme/compare/300e6f7913407b1216bf07b5ea8827aae963c898...269207f268bdda2b46e6c6bc1646bb2195eee0f2\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-11-11T13:58:43+01:00",
          "tree_id": "8326699bd1a94b8bc18517c6b4f8ba2d032a9c4f",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/8a0664ae8dd09bad32cef3f611ad51ca62a9dba2"
        },
        "date": 1762870811352,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.0000045017170021310446,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004457445099251344,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001498850076983,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001502703774021,
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
          "id": "30a48df863943d0259f50466673b62a5e52b8b57",
          "message": "fix: remove blocks from tt_call op (#1607)\n\n* fix: remove blocks from tt_call op\n\n* chore: run formatter",
          "timestamp": "2025-11-11T12:57:27-05:00",
          "tree_id": "362a9171c4d6df27d447f7bb0e07ffd7ca410707",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/30a48df863943d0259f50466673b62a5e52b8b57"
        },
        "date": 1762890046098,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004297780100023374,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004521115001989529,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001480496203992,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001489120090031,
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
          "id": "afd06426aec90984e084c72885fdc5a9fd2e9d12",
          "message": "Extend to pad 2 (#1608)\n\n* Extend to pad 2\n\n* fix",
          "timestamp": "2025-11-11T22:22:33-06:00",
          "tree_id": "e9f85526172fa1f4a9598eb8c7af79d88752d5a6",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/afd06426aec90984e084c72885fdc5a9fd2e9d12"
        },
        "date": 1762928478734,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004394121200311929,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.0000043122530914843084,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001489107565023,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.000159457737999,
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
          "id": "044e8c29ee1d11a83243e8ef94d6016183f8dbc7",
          "message": "Update jax-ml/jax to commit 24e80c494cb5464794730818cea05b60d7a956d7 (#1596)\n\n* Update jax-ml/jax to commit 24e80c494cb5464794730818cea05b60d7a956d7\n\nDiff: https://github.com/jax-ml/jax/compare/30e565311af559569b4842bddced4b461f21dd73...24e80c494cb5464794730818cea05b60d7a956d7\n\n* bump enzyme commit\n\n* fix\n\n* patch\n\n---------\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>\nCo-authored-by: William S. Moses <gh@wsmoses.com>",
          "timestamp": "2025-11-11T23:57:32-06:00",
          "tree_id": "d179a146fff70a8469d28e7241b6bcf134980a0b",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/044e8c29ee1d11a83243e8ef94d6016183f8dbc7"
        },
        "date": 1762955212560,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004321588110178709,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004304491099901497,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001413633406045,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001412845815997,
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
          "id": "b98ffe22c89b7fe9314ee1dbe1cdda91b05e6045",
          "message": "feat: dynamic_slice of dynamic_slice (#1553)\n\n* feat: dynamic_slice_reshape_dynamic_slice\n\n* feat: dynamic_slice dynamic_slice\n\n* chore: run fmt\n\n* feat: all permutation of DS and slice\n\n* feat: more slice reshape slice patterns",
          "timestamp": "2025-11-12T09:01:56-05:00",
          "tree_id": "1d63bfb4240f2d46da75164c2120e33e5fb4c830",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/b98ffe22c89b7fe9314ee1dbe1cdda91b05e6045"
        },
        "date": 1762975589412,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004339858097955584,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004307023203000426,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001490971453022,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001557892424054,
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
          "id": "044850d84959b4749b0601f4a6c07eba2152b363",
          "message": "Update EnzymeAD/Enzyme to commit 5655a0c72214755d886e8dff1bb47788908be999 (#1610)\n\nDiff: https://github.com/EnzymeAD/Enzyme/compare/f20d134a02177cc5f295d1e878aeb2d7579585b2...5655a0c72214755d886e8dff1bb47788908be999\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-11-12T11:16:12-06:00",
          "tree_id": "8251015478518c1c82f6d9c77f83f60ca8ef53cb",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/044850d84959b4749b0601f4a6c07eba2152b363"
        },
        "date": 1762985092410,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004179401195142418,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004276499105617404,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001335100820986,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001351870352984,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "58750937+snonk@users.noreply.github.com",
            "name": "snonk",
            "username": "snonk"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "546c4edb8ff48eba6808574f2b95b28d4b6c8d74",
          "message": "feat: symmetric tensor detection + transpose(symmetric) simplify (#1549)\n\n* added symm op\n\n* fmt\n\n* wip\n\n* Update src/enzyme_ad/jax/Utils.cpp\n\nCo-authored-by: Avik Pal <avikpal@mit.edu>\n\n* chore: run fmt\n\n* feat: transpose symmetric simplify\n\n* feat: generalize is_commutative check\n\n* fix: more checks + update tests\n\n* feat: generalizes the constant check\n\n---------\n\nCo-authored-by: Avik Pal <avikpal@mit.edu>",
          "timestamp": "2025-11-13T19:16:13-05:00",
          "tree_id": "6c319f664c6a5a37a9f8396e7694a4243fe044fa",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/546c4edb8ff48eba6808574f2b95b28d4b6c8d74"
        },
        "date": 1763095527469,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000008844769300776533,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000008818387897918,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / gpu / Primal",
            "value": 0.00008130874570051673,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / gpu / Primal",
            "value": 0.00007962837539962492,
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
          "id": "4c2893399e776fb73a1695f2a6f412d1f7b3c6f5",
          "message": "feat: simplify diagonal accesses for a dot_general op (#1614)\n\n* feat: simplify diagonal accesses for a dot_general op\n\n* test: update",
          "timestamp": "2025-11-13T19:17:01-05:00",
          "tree_id": "940464d20f21b4619e21a08103cc19da6baaa5d1",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/4c2893399e776fb73a1695f2a6f412d1f7b3c6f5"
        },
        "date": 1763112532131,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004451482999138534,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.00000439399010501802,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001457018755027,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001372073734062,
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
          "id": "563c1b32e79c9cf137e5ec2e98f32d06d9874abb",
          "message": "feat(python): enable newer passes + mimic options from julia (#1592)\n\n* feat(python): enable newer passes + mimic options from julia\n\n* test: try dumping results (drop me)\n\n* fix: remove debug printing\n\n* feat(python): dump mlir source on failure\n\n* fix: reshape to rank 0 tensor\n\n* fix: move dump to a function",
          "timestamp": "2025-11-14T10:56:09-05:00",
          "tree_id": "fe1116677643551c8da6cc63d701eff149df5485",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/563c1b32e79c9cf137e5ec2e98f32d06d9874abb"
        },
        "date": 1763147562356,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004303716996219009,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004367884108796716,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001481232236023,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001508538136025,
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
          "id": "582e713446209d3a920deb8c6e79ed110d068a2e",
          "message": "refactor: cleanup linalg/lapack lowering (#1501)\n\n* feat: get_dimension_size batch interface\n\n* feat: implement jitcall batching with shlo_generic_batch_op_interface\n\n* refactor: reuse batching interface for LU factorization\n\n* fix: remove old changes\n\n* refactor: move into separate functions\n\n* test: update LU tests\n\n* feat: dynamic slice simplify\n\n* feat: mark memory effects\n\n* fix: update to new API\n\n* fix: use correct return\n\n* chore: run fmt\n\n* test: fix",
          "timestamp": "2025-11-15T21:30:24-06:00",
          "tree_id": "3e8266dd8c93d516b3bec56995d57598cb00e12a",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/582e713446209d3a920deb8c6e79ed110d068a2e"
        },
        "date": 1763291061023,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004273933998774737,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004275060992222279,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.000134760071407,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001332195553928,
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
          "id": "bb0db132691f945bbc82fe4812bbf4c200340d37",
          "message": "Update EnzymeAD/Enzyme to commit 6b4a73e3c71e6451c919850acf2999ee04daab12 (#1616)\n\nDiff: https://github.com/EnzymeAD/Enzyme/compare/5655a0c72214755d886e8dff1bb47788908be999...6b4a73e3c71e6451c919850acf2999ee04daab12\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-11-15T21:30:34-06:00",
          "tree_id": "3ae0ec0c28d576998fd892c993f4cb2f663aa351",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/bb0db132691f945bbc82fe4812bbf4c200340d37"
        },
        "date": 1763297348120,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004344125092029571,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.0000043180249980650846,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001443311725975,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001523386654909,
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
          "id": "2118c2305d0e514fdac66c6c823f87763dba6c94",
          "message": "Update jax-ml/jax to commit d79c1c43fe8c40c3c51743e1796f2d2b43ebfb82 (#1611)\n\n* Update jax-ml/jax to commit d79c1c43fe8c40c3c51743e1796f2d2b43ebfb82\n\nDiff: https://github.com/jax-ml/jax/compare/24e80c494cb5464794730818cea05b60d7a956d7...d79c1c43fe8c40c3c51743e1796f2d2b43ebfb82\n\n* tmp\n\n* fix\n\n* fix\n\n* fix\n\n* exclude\n\n* fix\n\n* fix\n\n* fix\n\n* fmt\n\n* Fix\n\n---------\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>\nCo-authored-by: William S. Moses <gh@wsmoses.com>",
          "timestamp": "2025-11-16T13:09:20-06:00",
          "tree_id": "7333d2b3db8b1b1c5cba2527255c6815754e9cbc",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/2118c2305d0e514fdac66c6c823f87763dba6c94"
        },
        "date": 1763332032440,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004256509896367789,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004211377899628133,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001351252875057,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001351235843962,
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
          "id": "9395a9c74963c26b7d653f08999ae5c71f763500",
          "message": "fix: symm is BLAS not LAPACK (#1624)",
          "timestamp": "2025-11-16T20:18:11-05:00",
          "tree_id": "96094647d25c2e9ccc6001f2423cec7e768dfc11",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/9395a9c74963c26b7d653f08999ae5c71f763500"
        },
        "date": 1763346337226,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004472326103132218,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.0000043786630965769295,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001475851104012,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001428839714033,
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
          "id": "b54ca8ce28737723f2ec1c5c7943363f00862d1c",
          "message": "Update jax-ml/jax to commit 071a6a5a9a70b2447f0164e61e3e0333318422a8 (#1622)\n\nDiff: https://github.com/jax-ml/jax/compare/d79c1c43fe8c40c3c51743e1796f2d2b43ebfb82...071a6a5a9a70b2447f0164e61e3e0333318422a8\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-11-16T20:13:50-06:00",
          "tree_id": "e9397629fc73af7c6a148d6f26da08034bd9cfe3",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/b54ca8ce28737723f2ec1c5c7943363f00862d1c"
        },
        "date": 1763358157057,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004254931095056236,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.0000043403101968578994,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.000131403051503,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001310077337082,
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
          "id": "aded9a2821774cdfa05c6207afa3b92e16b6d583",
          "message": "fix: use lapack instead of lapacke (#1618)\n\nfeat: get_dimension_size batch interface\n\nfeat: implement jitcall batching with shlo_generic_batch_op_interface\n\nrefactor: reuse batching interface for LU factorization\n\nfix: remove old changes\n\nfeat: mark memory effects\n\nfix: update to new API\n\nfix: use correct return\n\nfix: rework svd lowering to use lapack and workspace buffers\n\nfix: use func return\n\nfix: leading dimension\n\nfix: use backend config\n\nfix: remove unused vars\n\nfix: tpu lowering\n\ntest: update",
          "timestamp": "2025-11-17T00:05:45-05:00",
          "tree_id": "ddffb8cb739571b2b0e3cf6765c4b1a2fa659fea",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/aded9a2821774cdfa05c6207afa3b92e16b6d583"
        },
        "date": 1763364371985,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004386482108384371,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004358096001669764,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001448858825955,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001450719916028,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "58750937+snonk@users.noreply.github.com",
            "name": "snonk",
            "username": "snonk"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "8c2e29e2ba68df7f492a3d4ef9b48e5f97a725c5",
          "message": "add constant int/float check for symm (#1621)\n\n* constant check\n\n* fmt\n\n* add splat and broadcast:\n\n* fix: formatting\n\n* refactor: cleanup into a single function\n\n* add test\n\n---------\n\nCo-authored-by: Avik Pal <avikpal@mit.edu>",
          "timestamp": "2025-11-18T01:46:55-06:00",
          "tree_id": "62e40d6f02428e2a18c4e740a44104a36d124a98",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/8c2e29e2ba68df7f492a3d4ef9b48e5f97a725c5"
        },
        "date": 1763474814552,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004369421099909232,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004413862099681864,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001363153289999,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001475828000999,
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
          "id": "b55ffd4f69428efae99ca2178b4b0672b97b33d3",
          "message": "feat: progressive lowering of LU and SVD via lapack ops (#1625)\n\n* feat: more lapack operations for LU/SVD\n\n* fix: lower linalg ops to lapack\n\n* chore: stub code for lowering\n\n* feat: lowering for getrf\n\n* feat: svd lapack op lowering\n\n* feat: cpu lowering\n\n* fix: missing impl in header\n\n* fix: export names\n\n* fix: update workspace buffer correctly\n\n* test: update all tests\n\n* feat: support batching",
          "timestamp": "2025-11-18T08:00:16-05:00",
          "tree_id": "b49ce610257a6818b24253418256646e4de4f917",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/b55ffd4f69428efae99ca2178b4b0672b97b33d3"
        },
        "date": 1763479651450,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004338932399696205,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004298453299998073,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001558048111997,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001522772955002,
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
          "id": "2b7c1da4ae9bef7d8aa96f9c9009aec509e510a5",
          "message": "feat: don't compute UV in SVD unless needed (#1632)\n\n* feat: don't compute UV in SVD unless needed\n\n* fix: call convention for cusolver\n\n* fix: aliasing in cpu custom_call\n\n* feat: lower to job=N when compute_uv=false",
          "timestamp": "2025-11-20T00:06:20-06:00",
          "tree_id": "a34678503a58349c91817c11c89a61ebc22fa515",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/2b7c1da4ae9bef7d8aa96f9c9009aec509e510a5"
        },
        "date": 1763639530241,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004362374899938004,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004327750999800628,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001545558818004,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001537393587001,
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
          "id": "94bdf8cd9bd5a023dbe7153ec7092db20f4a2b4c",
          "message": "Update EnzymeAD/Enzyme to commit 7c521437a9ffeea730e843d48da06f40824b0c71 (#1634)\n\nDiff: https://github.com/EnzymeAD/Enzyme/compare/6b4a73e3c71e6451c919850acf2999ee04daab12...7c521437a9ffeea730e843d48da06f40824b0c71\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-11-20T10:58:49-05:00",
          "tree_id": "ccccd8579891b436de00bd0ebaddbcb91247ff93",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/94bdf8cd9bd5a023dbe7153ec7092db20f4a2b4c"
        },
        "date": 1763670745875,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004348259000107646,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004332027999771526,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001330425719002,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001410325577999,
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
          "id": "3c38d1fd809f0f7e246bf3ba2f535195dabe78cb",
          "message": "fix: analysis should be run on Values (not Operations) (#1633)\n\n* fix: analysis should be run on Values (not Operations)\n\n* refactor: abstract away the constant check\n\n* fix: remove PENDING\n\n* fix: make naming better\n\n* fix: drop unsupported attributes\n\n* feat: C API\n\n* feat: read attrs from block args\n\n* test: more testing\n\n* fix: C API",
          "timestamp": "2025-11-20T14:44:42-05:00",
          "tree_id": "a8c08f13a1cab3ebf99b5a192e2ebdbcd6ed38a3",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/3c38d1fd809f0f7e246bf3ba2f535195dabe78cb"
        },
        "date": 1763695749000,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004347273900930304,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004293463900103233,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001352114348992,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001364999429992,
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
          "id": "3c38d1fd809f0f7e246bf3ba2f535195dabe78cb",
          "message": "fix: analysis should be run on Values (not Operations) (#1633)\n\n* fix: analysis should be run on Values (not Operations)\n\n* refactor: abstract away the constant check\n\n* fix: remove PENDING\n\n* fix: make naming better\n\n* fix: drop unsupported attributes\n\n* feat: C API\n\n* feat: read attrs from block args\n\n* test: more testing\n\n* fix: C API",
          "timestamp": "2025-11-20T14:44:42-05:00",
          "tree_id": "a8c08f13a1cab3ebf99b5a192e2ebdbcd6ed38a3",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/3c38d1fd809f0f7e246bf3ba2f535195dabe78cb"
        },
        "date": 1763705200732,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004347273900930304,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004293463900103233,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001352114348992,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001364999429992,
            "unit": "s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "39610523+vimarsh6739@users.noreply.github.com",
            "name": "Vimarsh Sathia",
            "username": "vimarsh6739"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "da7ed39b534102b1e24a1758584a98d9c79e05e5",
          "message": "Enzyme batch to StableHLO (#1556)\n\n* Add raising skeleton\n\n* Add ConcatOp lowering, fix lit test\nNeed to find another cafe to work...\n\n* fix lit tests for concatenateOp\n\n* just add conversion for Extract assuming index is 0-dim tensor\n\n* Finalize changes\n\n* fix lit",
          "timestamp": "2025-11-21T15:30:39-06:00",
          "tree_id": "c43567fed636223aed56454f7208ee76e117a5ba",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/da7ed39b534102b1e24a1758584a98d9c79e05e5"
        },
        "date": 1763789396324,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004360322000047745,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.00000430330099998173,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001381765273,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001457332371999,
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
          "id": "ea42b0233460eacd72563622ee9103711462c70e",
          "message": "Update EnzymeAD/Enzyme to commit dd29921570c549f98607f792384efffa35f8e1bc (#1637)\n\nDiff: https://github.com/EnzymeAD/Enzyme/compare/ebfdfd08ee7427f7de2f8184442d39dc8ea29b76...dd29921570c549f98607f792384efffa35f8e1bc\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-11-22T13:56:48-06:00",
          "tree_id": "c438968518ea3f2584b0319775d093811e22d50b",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/ea42b0233460eacd72563622ee9103711462c70e"
        },
        "date": 1763847963656,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004246331000467762,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004317962999630254,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001486625116012,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001533850926993,
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
          "id": "9a198274188537b5f832a9a25359f0be4b01255d",
          "message": "fix: missing patterns in transform dialect (#1639)\n\n* fix: missing patterns in transform dialect\n\n* feat: convolution_licm\n\n* chore: run format",
          "timestamp": "2025-11-22T18:53:16-05:00",
          "tree_id": "1b262845fd4e7f854effbc2598d6360b7ef022be",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/9a198274188537b5f832a9a25359f0be4b01255d"
        },
        "date": 1763872724839,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.00000435271799997281,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004273902099976112,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001496146923999,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001404514594,
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
          "id": "a2c7606121fad722e5c2acf34965afbc1af80166",
          "message": "feat: div/mul with negated operands simplify (#1640)\n\n* feat: div/mul with negated operands simplify\n\n* feat: ReplaceSubtractNegWithAdd",
          "timestamp": "2025-11-22T21:24:04-05:00",
          "tree_id": "ff0407b23e39fbaf6f8b0b44d5c74c519d92d436",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/a2c7606121fad722e5c2acf34965afbc1af80166"
        },
        "date": 1763879080840,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004406645000199205,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004316241000196896,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001385386526999,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001373059016998,
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
          "id": "764f3e4fbe6f0084353227ee3b8b66321665f529",
          "message": "fix: while induction replacement (#1641)",
          "timestamp": "2025-11-23T00:42:00-05:00",
          "tree_id": "698d1ccc58f49526f806486c3b640508b2a45c0b",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/764f3e4fbe6f0084353227ee3b8b66321665f529"
        },
        "date": 1763892843590,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004383075000077952,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.00000427507600034005,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001363426811003,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001350103551005,
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
          "id": "65b54ecea3b196567eda21dea941014d7979314b",
          "message": "Update EnzymeAD/Enzyme to commit 5db86b6fe7d77f30546627b9c483375284feb159 (#1642)\n\nDiff: https://github.com/EnzymeAD/Enzyme/compare/dd29921570c549f98607f792384efffa35f8e1bc...5db86b6fe7d77f30546627b9c483375284feb159\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-11-23T14:28:49-05:00",
          "tree_id": "3aa7658a44fc3885adc055509770c78a729712ba",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/65b54ecea3b196567eda21dea941014d7979314b"
        },
        "date": 1763944640550,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004280468999968435,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004251791000024241,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001484288595998,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001495373456,
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
          "id": "f9784e443d54e91422ca841a66f56860aeb7af92",
          "message": "Update enzyme for fwddiff (#1645)",
          "timestamp": "2025-11-23T22:52:06-06:00",
          "tree_id": "d218198ad6499354d6fb8ad358f2b17626b4c9ab",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/f9784e443d54e91422ca841a66f56860aeb7af92"
        },
        "date": 1763982351343,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004370322999966447,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004389473999981419,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001559570289999,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001467021811,
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
          "id": "e3de2244ab21c07bf1502ccb8634be0ac86ead82",
          "message": "fix: typo in pass name (#1651)",
          "timestamp": "2025-11-24T14:23:20-05:00",
          "tree_id": "3029092140f3fe6e35326873e06c1386efe12d47",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/e3de2244ab21c07bf1502ccb8634be0ac86ead82"
        },
        "date": 1764029035573,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.00000432322499982547,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004288105000159703,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001405532037999,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001445494979001,
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
          "id": "f11d5ad20f7929603bfd2b45a956fae030fccb1c",
          "message": "feat: add syrk and trmm op (#1644)\n\n* feat: add syrk op\n\n* test: syrk\n\n* fix: docs\n\n* feat: trmm op\n\n* fix: missing uplo",
          "timestamp": "2025-11-24T14:23:37-05:00",
          "tree_id": "5bf0f7eec2193ccc789003e5cfd0defc3c1a5cda",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/f11d5ad20f7929603bfd2b45a956fae030fccb1c"
        },
        "date": 1764035766460,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004226516900234856,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.0000041471548989648,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001521801919996,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001423218900003,
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
          "id": "cf16b634ef7780c681e42d9494b8d94bba5096bd",
          "message": "Update EnzymeAD/Enzyme to commit b642f02d745b5bb0425cde2a888e125b28409d19 (#1649)\n\nDiff: https://github.com/EnzymeAD/Enzyme/compare/05b1413e38c4f6651d3ca35591d34d6ec272c186...b642f02d745b5bb0425cde2a888e125b28409d19\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-11-24T22:51:19-06:00",
          "tree_id": "eab468f491326b2036506eb9138980938b263c95",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/cf16b634ef7780c681e42d9494b8d94bba5096bd"
        },
        "date": 1764061756746,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004467982599999232,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004483210600028543,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001430403397999,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001420598939999,
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
          "id": "da5e6ab8248afae73b545883fe1973b2c5644262",
          "message": "Update EnzymeAD/Enzyme to commit a13f632e2fbaed1f971de015f15e8f4b353e66cb (#1656)\n\nDiff: https://github.com/EnzymeAD/Enzyme/compare/b642f02d745b5bb0425cde2a888e125b28409d19...a13f632e2fbaed1f971de015f15e8f4b353e66cb\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-11-25T09:30:07-05:00",
          "tree_id": "0d120a9b676350bd8f870fca1c90956414907ab9",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/da5e6ab8248afae73b545883fe1973b2c5644262"
        },
        "date": 1764100658116,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.0000043999469999107535,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004439157900924329,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001253901321993,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001279873421997,
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
          "id": "88ee9750320164a5f51803b3ecb6edd211ea2f4d",
          "message": "feat: common syrk optimization patterns (#1653)\n\n* feat: common syrk optimization patterns\n\n* feat: syrk output is always symmetric\n\n* fix: address review comment\n\n* fix: docs",
          "timestamp": "2025-11-25T13:24:32-05:00",
          "tree_id": "84f7f534617a909137df3bc4bbcb852aba6582cb",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/88ee9750320164a5f51803b3ecb6edd211ea2f4d"
        },
        "date": 1764124503075,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004326902899993002,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004272585000035178,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001354923245,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001335705285999,
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
          "id": "31f39c52148a55024b464a98f15b3abc428828e2",
          "message": "feat: blas lowering for syrk (#1650)\n\n* feat: blas lowering for syrk\n\n* feat: fallback lowering for syrk\n\n* feat: lower to blas\n\n* fix: uplo and trans types\n\n* test: one more test case\n\n* fix: copy triangular\n\n* feat: add a new uplo enum entry\n\n* fix: fill\n\n* fix: debug logging\n\n* test: add lit_tests",
          "timestamp": "2025-11-25T18:09:45-05:00",
          "tree_id": "0f1063c60714835db3837d6ec6890ae835e1f8ff",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/31f39c52148a55024b464a98f15b3abc428828e2"
        },
        "date": 1764131959645,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004300762999992002,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004328533899752074,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001521289035001,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001386696216999,
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
          "id": "adb13be7e3e67dd994abfc7bdd6877346c92be11",
          "message": "Update EnzymeAD/Enzyme to commit f467ff7fab15a65a8f034017e2d9f52f9823e79d (#1658)\n\nDiff: https://github.com/EnzymeAD/Enzyme/compare/a13f632e2fbaed1f971de015f15e8f4b353e66cb...f467ff7fab15a65a8f034017e2d9f52f9823e79d\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-11-26T10:57:18-05:00",
          "tree_id": "669a3af7675f800183ddb3ba820d8ee7c735a198",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/adb13be7e3e67dd994abfc7bdd6877346c92be11"
        },
        "date": 1764187659984,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "scatter_sum / JaX   / cpu / Primal",
            "value": 0.000004325666999648092,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000004221120000147494,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaX   / tpu / Primal",
            "value": 0.0001473334281006,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001520273059999,
            "unit": "s"
          }
        ]
      }
    ]
  }
}