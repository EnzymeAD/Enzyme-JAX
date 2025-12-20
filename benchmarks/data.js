window.BENCHMARK_DATA = {
  lastUpdate: 1766194718721,
  repoUrl: "https://github.com/EnzymeAD/Enzyme-JAX",
  entries: {
    "EnzymeJAX Benchmarks": [
      {
        commit: {
          author: {
            email: "avikpal@mit.edu",
            name: "Avik Pal",
            username: "avik-pal",
          },
          committer: {
            email: "noreply@github.com",
            name: "GitHub",
            username: "web-flow",
          },
          distinct: true,
          id: "f93b08ebc5c860c7e3d1b273c18188726ce9f0a0",
          message:
            "fix(python): properly run all benchmarks (#1790)\n\n* test: update bench vs xla tests\n\n* test: dump the compiled mlir\n\n* fix: correctly block until ready\n\n* test: make timings more stable\n\n* fix: llama\n\n* test: run neural gcm inside EnzymeJaxTest\n\n* fix: local imports\n\n* test: skip test asserts in neural gcm for now\n\n* test: cleanup printing",
          timestamp: "2025-12-19T01:06:21-05:00",
          tree_id: "0db5555505c44886dc8520e30c13a40fe7152dbf",
          url: "https://github.com/EnzymeAD/Enzyme-JAX/commit/f93b08ebc5c860c7e3d1b273c18188726ce9f0a0",
        },
        date: 1766148818751,
        tool: "customSmallerIsBetter",
        benches: [
          {
            name: "actmtch / JaXPipe / cpu / Primal",
            value: 0.000004563776999930269,
            unit: "s",
          },
          {
            name: "actmtch / Jax / cpu / Primal",
            value: 0.000004850467999858665,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / Primal",
            value: 0.000005318563999935577,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / Primal",
            value: 0.000004764768999848456,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / Primal",
            value: 0.000004723978000129137,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / Primal",
            value: 0.000005305487999976322,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / Primal",
            value: 0.000005335949999789591,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / Forward",
            value: 0.000008265983999990567,
            unit: "s",
          },
          {
            name: "actmtch / Jax / cpu / Forward",
            value: 0.000007104215000254044,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / Forward",
            value: 0.00000811375900002531,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / Forward",
            value: 0.00000821349400030158,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / Forward",
            value: 0.000007904199000222434,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / Forward",
            value: 0.000008252481999988958,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / Forward",
            value: 0.000008182605999991211,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / PreRev",
            value: 0.000008047015000101965,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / PostRev",
            value: 0.000007137858000078268,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / BothRev",
            value: 0.000008279135000066163,
            unit: "s",
          },
          {
            name: "actmtch / Jax / cpu / BothRev",
            value: 0.000007147278000047663,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / PreRev",
            value: 0.000008219886999995651,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / PostRev",
            value: 0.00000810069499993915,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / BothRev",
            value: 0.000008072791000358848,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / PreRev",
            value: 0.000008237381000071764,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / PostRev",
            value: 0.00000714416099981463,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / BothRev",
            value: 0.000008162450000327226,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / PreRev",
            value: 0.000008234739999807062,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / PostRev",
            value: 0.000007158508999964397,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / BothRev",
            value: 0.000008108869999887247,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / PreRev",
            value: 0.000008137240999985807,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / PostRev",
            value: 0.000008158186999935424,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / BothRev",
            value: 0.000008233293000103003,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / PreRev",
            value: 0.000008226977000049373,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / PostRev",
            value: 0.000008125345000280503,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / BothRev",
            value: 0.000008176082999852951,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / Primal",
            value: 0.000008695475000422447,
            unit: "s",
          },
          {
            name: "actmtch / Jax / cpu / Primal",
            value: 0.000008716701006051153,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / Primal",
            value: 0.000010917057006736284,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / Primal",
            value: 0.000009525789006147534,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / Primal",
            value: 0.000008699628000613301,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / Primal",
            value: 0.00001094573299633339,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / Primal",
            value: 0.000011408320991904475,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / Forward",
            value: 0.000015885119006270544,
            unit: "s",
          },
          {
            name: "actmtch / Jax / cpu / Forward",
            value: 0.000013632168003823607,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / Forward",
            value: 0.00001665185698948335,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / Forward",
            value: 0.00001679151100688614,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / Forward",
            value: 0.00001672596699791029,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / Forward",
            value: 0.000016725751993362793,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / Forward",
            value: 0.0000166027820087038,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / PreRev",
            value: 0.000016566127989790404,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / PostRev",
            value: 0.000012792702997103334,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / BothRev",
            value: 0.000016532782989088446,
            unit: "s",
          },
          {
            name: "actmtch / Jax / cpu / BothRev",
            value: 0.000013461405993439256,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / PreRev",
            value: 0.00001660588699451182,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / PostRev",
            value: 0.000016665199000271967,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / BothRev",
            value: 0.000016562175995204597,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / PreRev",
            value: 0.000015694759000325574,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / PostRev",
            value: 0.00001353055400250014,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / BothRev",
            value: 0.000016107966002891773,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / PreRev",
            value: 0.000016585759993176906,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / PostRev",
            value: 0.00001291142700938508,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / BothRev",
            value: 0.000016567343001952395,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / PreRev",
            value: 0.000016590209997957573,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / PostRev",
            value: 0.00001670104499498848,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / BothRev",
            value: 0.000016667381991283038,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / PreRev",
            value: 0.000016728909002267754,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / PostRev",
            value: 0.000016685963986674323,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / BothRev",
            value: 0.00001672418200178072,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / tpu / Primal",
            value: 0.0001463805990024,
            unit: "s",
          },
          {
            name: "actmtch / Jax / tpu / Primal",
            value: 0.0001472225390025,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / tpu / Primal",
            value: 0.0001480529190012,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / tpu / Primal",
            value: 0.0001449752090011,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / tpu / Primal",
            value: 0.0001442559679999,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / tpu / Primal",
            value: 0.0001467183189997,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / tpu / Primal",
            value: 0.0001477851790004,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / tpu / Forward",
            value: 0.0002151236669997,
            unit: "s",
          },
          {
            name: "actmtch / Jax / tpu / Forward",
            value: 0.0002081720239984,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / tpu / Forward",
            value: 0.0002209450289992,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / tpu / Forward",
            value: 0.000230873763001,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / tpu / Forward",
            value: 0.0002229896799981,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / tpu / Forward",
            value: 0.0002239343500004,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / tpu / Forward",
            value: 0.0002231415899987,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / tpu / PreRev",
            value: 0.0001965865290003,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / tpu / PostRev",
            value: 0.0001974495089998,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / tpu / BothRev",
            value: 0.0001978393290009,
            unit: "s",
          },
          {
            name: "actmtch / Jax / tpu / BothRev",
            value: 0.0001953822979994,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / tpu / PreRev",
            value: 0.0001964807490003,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / tpu / PostRev",
            value: 0.0001952032689987,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / tpu / BothRev",
            value: 0.0001944747980014,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / tpu / PreRev",
            value: 0.0001950831289977,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / tpu / PostRev",
            value: 0.0002282282820015,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / tpu / BothRev",
            value: 0.0002090407340001,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / tpu / PreRev",
            value: 0.0001960459879992,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / tpu / PostRev",
            value: 0.0002011782609988,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / tpu / BothRev",
            value: 0.0001980370389974,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / tpu / PreRev",
            value: 0.0001948459279992,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / tpu / PostRev",
            value: 0.0001961346290008,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / tpu / BothRev",
            value: 0.0002133188059997,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / tpu / PreRev",
            value: 0.0001986471700001,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / tpu / PostRev",
            value: 0.0001968597800005,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / tpu / BothRev",
            value: 0.0001988248020024,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / Primal",
            value: 0.000005976688000373542,
            unit: "s",
          },
          {
            name: "actmtch / Jax / cpu / Primal",
            value: 0.000006042200999218039,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / Primal",
            value: 0.000007402432000162662,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / Primal",
            value: 0.000006600967999474961,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / Primal",
            value: 0.00000659571699998196,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / Primal",
            value: 0.0000074052920008398356,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / Primal",
            value: 0.000007390992999717127,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / Forward",
            value: 0.000011360692999915045,
            unit: "s",
          },
          {
            name: "actmtch / Jax / cpu / Forward",
            value: 0.000009414804999323678,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / Forward",
            value: 0.000010787776000142913,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / Forward",
            value: 0.000011388970000552943,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / Forward",
            value: 0.000010929775000477091,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / Forward",
            value: 0.00001089125199996488,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / Forward",
            value: 0.000010685500999898069,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / PreRev",
            value: 0.000010767635999400226,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / PostRev",
            value: 0.000009012014999825624,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / BothRev",
            value: 0.00001147857199975988,
            unit: "s",
          },
          {
            name: "actmtch / Jax / cpu / BothRev",
            value: 0.000009078316000341147,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / PreRev",
            value: 0.00001082645099995716,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / PostRev",
            value: 0.000011416192999604392,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / BothRev",
            value: 0.000011524333999659576,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / PreRev",
            value: 0.000010881265000534769,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / PostRev",
            value: 0.000009505998000349792,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / BothRev",
            value: 0.00001070122800047102,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / PreRev",
            value: 0.00001138253400040412,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / PostRev",
            value: 0.000009573772000294412,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / BothRev",
            value: 0.000010775668999485788,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / PreRev",
            value: 0.00001141181200000574,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / PostRev",
            value: 0.000011450408999735373,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / BothRev",
            value: 0.000010874071999751325,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / PreRev",
            value: 0.000011390661999939766,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / PostRev",
            value: 0.000011345665000590089,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / BothRev",
            value: 0.000011265244000242093,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / Primal",
            value: 0.000004113749997486593,
            unit: "s",
          },
          {
            name: "actmtch / Jax / cpu / Primal",
            value: 0.00000392887499765493,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / Primal",
            value: 0.000004460041996935616,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / Primal",
            value: 0.000004048666996823159,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / Primal",
            value: 0.000003875250000419328,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / Primal",
            value: 0.000004496290999668417,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / Primal",
            value: 0.00000480216600044514,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / Forward",
            value: 0.000007361416999629001,
            unit: "s",
          },
          {
            name: "actmtch / Jax / cpu / Forward",
            value: 0.00000643691700315685,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / Forward",
            value: 0.000006971375001739944,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / Forward",
            value: 0.0000069558329996652905,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / Forward",
            value: 0.000007109375001164154,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / Forward",
            value: 0.000007122167000488844,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / Forward",
            value: 0.000007829916998161934,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / PreRev",
            value: 0.000007229209000797709,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / PostRev",
            value: 0.000005977999997412553,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / BothRev",
            value: 0.000007144875002268236,
            unit: "s",
          },
          {
            name: "actmtch / Jax / cpu / BothRev",
            value: 0.000006495042001915863,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / PreRev",
            value: 0.000006860458001028746,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / PostRev",
            value: 0.0000073467920010443775,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / BothRev",
            value: 0.000007435040999553166,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / PreRev",
            value: 0.000006913083001563792,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / PostRev",
            value: 0.00000579045799895539,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / BothRev",
            value: 0.000006950499999220483,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / PreRev",
            value: 0.000007074082997860387,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / PostRev",
            value: 0.000005938917001913069,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / BothRev",
            value: 0.000006932541000423953,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / PreRev",
            value: 0.000006921416999830399,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / PostRev",
            value: 0.00000686429100096575,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / BothRev",
            value: 0.000006855708998045884,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / PreRev",
            value: 0.000007838916997570778,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / PostRev",
            value: 0.000006794334000005619,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / BothRev",
            value: 0.000007025167000392684,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / Primal",
            value: 0.000004878246999851399,
            unit: "s",
          },
          {
            name: "add_one / Jax / cpu / Primal",
            value: 0.0000049801440000010185,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / Primal",
            value: 0.000004926833000354236,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / Primal",
            value: 0.000004910274999929242,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / Primal",
            value: 0.0000049149070000567005,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / Primal",
            value: 0.000004925769000237779,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / Primal",
            value: 0.000004856651999944006,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / Forward",
            value: 0.000007960080999964702,
            unit: "s",
          },
          {
            name: "add_one / Jax / cpu / Forward",
            value: 0.00000797657099974458,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / Forward",
            value: 0.000007969832000071619,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / Forward",
            value: 0.000007986522000010155,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / Forward",
            value: 0.000007938186000046698,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / Forward",
            value: 0.000007959325000229,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / Forward",
            value: 0.000007948341999963305,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / PreRev",
            value: 0.000008676244999605841,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / PostRev",
            value: 0.000008568704000026628,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / BothRev",
            value: 0.000008638467999844579,
            unit: "s",
          },
          {
            name: "add_one / Jax / cpu / BothRev",
            value: 0.000008664016000238917,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / PreRev",
            value: 0.000008593155000198749,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / PostRev",
            value: 0.000008674003000123775,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / BothRev",
            value: 0.000008663898999657249,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / PreRev",
            value: 0.000008636039999601054,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / PostRev",
            value: 0.000008643633000247064,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / BothRev",
            value: 0.00000861039100027483,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / PreRev",
            value: 0.000008611650000148074,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / PostRev",
            value: 0.000008611805999862555,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / BothRev",
            value: 0.000008629436999854079,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / PreRev",
            value: 0.000008620025000254827,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / PostRev",
            value: 0.000008634616000108507,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / BothRev",
            value: 0.000008679496999775439,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / PreRev",
            value: 0.000008691588000147022,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / PostRev",
            value: 0.000009044835999702627,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / BothRev",
            value: 0.000008635536999918258,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / Primal",
            value: 0.000008968434995040298,
            unit: "s",
          },
          {
            name: "add_one / Jax / cpu / Primal",
            value: 0.000009184053997159936,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / Primal",
            value: 0.000009222940992913207,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / Primal",
            value: 0.000009732247999636456,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / Primal",
            value: 0.000009167948999674992,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / Primal",
            value: 0.000008979007005109451,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / Primal",
            value: 0.000009641222000936978,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / Forward",
            value: 0.000014663768000900746,
            unit: "s",
          },
          {
            name: "add_one / Jax / cpu / Forward",
            value: 0.000014441900988458657,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / Forward",
            value: 0.000014041959002497604,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / Forward",
            value: 0.000013844371991581285,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / Forward",
            value: 0.0000145608099992387,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / Forward",
            value: 0.000014730807990417816,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / Forward",
            value: 0.000014470704991254025,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / PreRev",
            value: 0.000015872468997258692,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / PostRev",
            value: 0.000015815402992302553,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / BothRev",
            value: 0.000015878420002991332,
            unit: "s",
          },
          {
            name: "add_one / Jax / cpu / BothRev",
            value: 0.000015846943002543413,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / PreRev",
            value: 0.00001594746299088001,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / PostRev",
            value: 0.00001582293900719378,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / BothRev",
            value: 0.00001583250900148414,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / PreRev",
            value: 0.000014987981994636356,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / PostRev",
            value: 0.000015716455003712326,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / BothRev",
            value: 0.000015106676990399137,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / PreRev",
            value: 0.000015722525989986026,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / PostRev",
            value: 0.000015819374995771796,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / BothRev",
            value: 0.000014823769000940956,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / PreRev",
            value: 0.00001566759300476406,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / PostRev",
            value: 0.000015764472002047114,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / BothRev",
            value: 0.000015776057000039145,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / PreRev",
            value: 0.000015719745992100798,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / PostRev",
            value: 0.000015806333991349674,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / BothRev",
            value: 0.000014859311006148346,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / tpu / Primal",
            value: 0.0001311625240014,
            unit: "s",
          },
          {
            name: "add_one / Jax / tpu / Primal",
            value: 0.0001318331639995,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / tpu / Primal",
            value: 0.0001319249340012,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / tpu / Primal",
            value: 0.0001338010349973,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / tpu / Primal",
            value: 0.0001305136440023,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / tpu / Primal",
            value: 0.0001309943129999,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / tpu / Primal",
            value: 0.0001334920839981,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / tpu / Forward",
            value: 0.0002200950200021,
            unit: "s",
          },
          {
            name: "add_one / Jax / tpu / Forward",
            value: 0.0002100113859996,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / tpu / Forward",
            value: 0.000205781073997,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / tpu / Forward",
            value: 0.0002340293859997,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / tpu / Forward",
            value: 0.0002297611239991,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / tpu / Forward",
            value: 0.0002259092730018,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / tpu / Forward",
            value: 0.0002239033210025,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / tpu / PreRev",
            value: 0.0002394261279987,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / tpu / PostRev",
            value: 0.0002410989889976,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / tpu / BothRev",
            value: 0.0002329095550012,
            unit: "s",
          },
          {
            name: "add_one / Jax / tpu / BothRev",
            value: 0.0002272223830004,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / tpu / PreRev",
            value: 0.0002235183009979,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / tpu / PostRev",
            value: 0.000232409275999,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / tpu / BothRev",
            value: 0.000215610389001,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / tpu / PreRev",
            value: 0.0002070917739983,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / tpu / PostRev",
            value: 0.0002205812600004,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / tpu / BothRev",
            value: 0.0002260568130004,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / tpu / PreRev",
            value: 0.0002170863989995,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / tpu / PostRev",
            value: 0.0002080843650001,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / tpu / BothRev",
            value: 0.0002058465650006,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / tpu / PreRev",
            value: 0.0002005970329992,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / tpu / PostRev",
            value: 0.0002013229930016,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / tpu / BothRev",
            value: 0.0002065691749994,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / tpu / PreRev",
            value: 0.0002074372849965,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / tpu / PostRev",
            value: 0.0002078447860003,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / tpu / BothRev",
            value: 0.0002071170849994,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / Primal",
            value: 0.000006367482999849017,
            unit: "s",
          },
          {
            name: "add_one / Jax / cpu / Primal",
            value: 0.000006341600999803631,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / Primal",
            value: 0.0000067716099993049285,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / Primal",
            value: 0.0000064280529995812685,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / Primal",
            value: 0.000006751687000360107,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / Primal",
            value: 0.000006434298000385752,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / Primal",
            value: 0.000006440039999688452,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / Forward",
            value: 0.00000985654199939745,
            unit: "s",
          },
          {
            name: "add_one / Jax / cpu / Forward",
            value: 0.00000972635000016453,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / Forward",
            value: 0.000010243503999845415,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / Forward",
            value: 0.000010299631999259872,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / Forward",
            value: 0.000009739982000610324,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / Forward",
            value: 0.000010313411999959498,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / Forward",
            value: 0.000010246027999528451,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / PreRev",
            value: 0.000011160860000018148,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / PostRev",
            value: 0.00001106412300032389,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / BothRev",
            value: 0.000011179145000824064,
            unit: "s",
          },
          {
            name: "add_one / Jax / cpu / BothRev",
            value: 0.00001109775500026444,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / PreRev",
            value: 0.000011131661999570497,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / PostRev",
            value: 0.000011162390999743366,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / BothRev",
            value: 0.000011150211000312993,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / PreRev",
            value: 0.000011090970000623202,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / PostRev",
            value: 0.000011170363999553956,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / BothRev",
            value: 0.000010533558999668458,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / PreRev",
            value: 0.000010466775999702804,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / PostRev",
            value: 0.0000110401539996019,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / BothRev",
            value: 0.000010554161000072782,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / PreRev",
            value: 0.000011181271000168636,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / PostRev",
            value: 0.000011247342000388015,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / BothRev",
            value: 0.000011102019999270851,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / PreRev",
            value: 0.00001116223999997601,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / PostRev",
            value: 0.000011157539999658184,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / BothRev",
            value: 0.000011229838999497588,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / Primal",
            value: 0.00000387383400084218,
            unit: "s",
          },
          {
            name: "add_one / Jax / cpu / Primal",
            value: 0.000003892000000632834,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / Primal",
            value: 0.0000038283329995465465,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / Primal",
            value: 0.000003853834001347423,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / Primal",
            value: 0.000003821999998763204,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / Primal",
            value: 0.000003829792000033194,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / Primal",
            value: 0.0000038045420005801134,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / Forward",
            value: 0.000006234874999790918,
            unit: "s",
          },
          {
            name: "add_one / Jax / cpu / Forward",
            value: 0.000006020458000421059,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / Forward",
            value: 0.000006227707999642007,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / Forward",
            value: 0.000005940374998317565,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / Forward",
            value: 0.00000604383300014888,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / Forward",
            value: 0.000005984416999126552,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / Forward",
            value: 0.000006005125000228873,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / PreRev",
            value: 0.000006979041998420143,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / PostRev",
            value: 0.000007083958000293933,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / BothRev",
            value: 0.000007042291999823647,
            unit: "s",
          },
          {
            name: "add_one / Jax / cpu / BothRev",
            value: 0.000007001999998465181,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / PreRev",
            value: 0.000006935459001397248,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / PostRev",
            value: 0.000007305833001737483,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / BothRev",
            value: 0.000006797249996452592,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / PreRev",
            value: 0.000006599040996661643,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / PostRev",
            value: 0.00000665866700001061,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / BothRev",
            value: 0.000006857832999230595,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / PreRev",
            value: 0.000006809167000028538,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / PostRev",
            value: 0.0000065762090016505685,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / BothRev",
            value: 0.000006909208997967653,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / PreRev",
            value: 0.000006737750001775567,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / PostRev",
            value: 0.0000068012080009793865,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / BothRev",
            value: 0.000007092625000950648,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / PreRev",
            value: 0.00000686695799959125,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / PostRev",
            value: 0.000006848082997748861,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / BothRev",
            value: 0.000006703208000544691,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / Primal",
            value: 0.00000513479399978678,
            unit: "s",
          },
          {
            name: "add_two / Jax / cpu / Primal",
            value: 0.000005125551999753952,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / Primal",
            value: 0.0000051105200000165495,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / Primal",
            value: 0.00000514219900014723,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / Primal",
            value: 0.000004995531000076881,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / Primal",
            value: 0.000005148495000412367,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / Primal",
            value: 0.00000512953599991306,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / Forward",
            value: 0.000008125807999931566,
            unit: "s",
          },
          {
            name: "add_two / Jax / cpu / Forward",
            value: 0.000008128145999762638,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / Forward",
            value: 0.000008129746000122395,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / Forward",
            value: 0.000008182028999726754,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / Forward",
            value: 0.000008146809000209032,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / Forward",
            value: 0.000008100070000182314,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / Forward",
            value: 0.000008156903999861243,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / PreRev",
            value: 0.000010629246000007697,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / PostRev",
            value: 0.000010577815000033297,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / BothRev",
            value: 0.00001058347899970613,
            unit: "s",
          },
          {
            name: "add_two / Jax / cpu / BothRev",
            value: 0.000010555272000146942,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / PreRev",
            value: 0.000010563419999925828,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / PostRev",
            value: 0.000010502064999855066,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / BothRev",
            value: 0.000010595097000077656,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / PreRev",
            value: 0.000010630184000092411,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / PostRev",
            value: 0.000010544045999722583,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / BothRev",
            value: 0.00001059205900037341,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / PreRev",
            value: 0.000010251429000163626,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / PostRev",
            value: 0.000010360613000102604,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / BothRev",
            value: 0.000010587567000129638,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / PreRev",
            value: 0.000010513900999740144,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / PostRev",
            value: 0.000010334244999739896,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / BothRev",
            value: 0.00001052170599996316,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / PreRev",
            value: 0.000010547028000019054,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / PostRev",
            value: 0.000010455664999881264,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / BothRev",
            value: 0.000010450115000367078,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / Primal",
            value: 0.000009471113997278735,
            unit: "s",
          },
          {
            name: "add_two / Jax / cpu / Primal",
            value: 0.000009511297001154162,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / Primal",
            value: 0.000009474455000599846,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / Primal",
            value: 0.000009932774002663791,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / Primal",
            value: 0.00000931927000056021,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / Primal",
            value: 0.000009492607990978286,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / Primal",
            value: 0.000010021623005741276,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / Forward",
            value: 0.000014247198007069528,
            unit: "s",
          },
          {
            name: "add_two / Jax / cpu / Forward",
            value: 0.0000149641259922646,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / Forward",
            value: 0.000014278572998591698,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / Forward",
            value: 0.000014874450003844684,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / Forward",
            value: 0.00001505485900270287,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / Forward",
            value: 0.000014300330003607086,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / Forward",
            value: 0.00001432913499593269,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / PreRev",
            value: 0.00001769785898795817,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / PostRev",
            value: 0.000018278224000823685,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / BothRev",
            value: 0.00001838101500470657,
            unit: "s",
          },
          {
            name: "add_two / Jax / cpu / BothRev",
            value: 0.00001830956598860212,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / PreRev",
            value: 0.000018614647997310387,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / PostRev",
            value: 0.00001839781900343951,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / BothRev",
            value: 0.00001836441100749653,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / PreRev",
            value: 0.000018357494001975283,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / PostRev",
            value: 0.000018338110006880012,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / BothRev",
            value: 0.00001717330698738806,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / PreRev",
            value: 0.00001852672200766392,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / PostRev",
            value: 0.000017725983008858748,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / BothRev",
            value: 0.00001777008399949409,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / PreRev",
            value: 0.000017812400008551776,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / PostRev",
            value: 0.00001848273300856817,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / BothRev",
            value: 0.00001835258599021472,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / PreRev",
            value: 0.000018336969005758875,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / PostRev",
            value: 0.000018520466997870244,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / BothRev",
            value: 0.000018546677005360834,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / tpu / Primal",
            value: 0.0001331291850001,
            unit: "s",
          },
          {
            name: "add_two / Jax / tpu / Primal",
            value: 0.0001328101040016,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / tpu / Primal",
            value: 0.000132528774,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / tpu / Primal",
            value: 0.000137039096,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / tpu / Primal",
            value: 0.0001547402830001,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / tpu / Primal",
            value: 0.0001551109630017,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / tpu / Primal",
            value: 0.0001542402330014,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / tpu / Forward",
            value: 0.000237460637003,
            unit: "s",
          },
          {
            name: "add_two / Jax / tpu / Forward",
            value: 0.0002232172019976,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / tpu / Forward",
            value: 0.0002377718380012,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / tpu / Forward",
            value: 0.0002323682650021,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / tpu / Forward",
            value: 0.0002317019550027,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / tpu / Forward",
            value: 0.000239057327999,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / tpu / Forward",
            value: 0.0002398991489972,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / tpu / PreRev",
            value: 0.0002498888720001,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / tpu / PostRev",
            value: 0.0002452321700002,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / tpu / BothRev",
            value: 0.000231756404999,
            unit: "s",
          },
          {
            name: "add_two / Jax / tpu / BothRev",
            value: 0.0002414934490006,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / tpu / PreRev",
            value: 0.0002419596689978,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / tpu / PostRev",
            value: 0.0002407436690009,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / tpu / BothRev",
            value: 0.0002158117390026,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / tpu / PreRev",
            value: 0.0002189239300023,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / tpu / PostRev",
            value: 0.0002260490519984,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / tpu / BothRev",
            value: 0.0002282330239977,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / tpu / PreRev",
            value: 0.0002260923819994,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / tpu / PostRev",
            value: 0.0002348778160012,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / tpu / BothRev",
            value: 0.0002356746259974,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / tpu / PreRev",
            value: 0.0002300486349995,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / tpu / PostRev",
            value: 0.0002241708220026,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / tpu / BothRev",
            value: 0.000218939639999,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / tpu / PreRev",
            value: 0.0002154017780012,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / tpu / PostRev",
            value: 0.0002162974379971,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / tpu / BothRev",
            value: 0.0002122958770014,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / Primal",
            value: 0.000006670531999589003,
            unit: "s",
          },
          {
            name: "add_two / Jax / cpu / Primal",
            value: 0.000007039395999527187,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / Primal",
            value: 0.000006738613999914378,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / Primal",
            value: 0.000006588944000213814,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / Primal",
            value: 0.000007103864000782778,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / Primal",
            value: 0.0000067698789998758,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / Primal",
            value: 0.000006730476000484486,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / Forward",
            value: 0.000010600509000141755,
            unit: "s",
          },
          {
            name: "add_two / Jax / cpu / Forward",
            value: 0.000010434938999424049,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / Forward",
            value: 0.000010129237999535687,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / Forward",
            value: 0.000010543457000494527,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / Forward",
            value: 0.000010607797000375283,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / Forward",
            value: 0.00001002709899967158,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / Forward",
            value: 0.000010533229999964533,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / PreRev",
            value: 0.000013197375000345344,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / PostRev",
            value: 0.000013050114999714425,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / BothRev",
            value: 0.00001304433700079244,
            unit: "s",
          },
          {
            name: "add_two / Jax / cpu / BothRev",
            value: 0.000013253899999654096,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / PreRev",
            value: 0.0000129984550003428,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / PostRev",
            value: 0.00001305620099992666,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / BothRev",
            value: 0.000013111940000271716,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / PreRev",
            value: 0.000013109796999742684,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / PostRev",
            value: 0.00001313834200027486,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / BothRev",
            value: 0.000012550815000395232,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / PreRev",
            value: 0.000012509863000559564,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / PostRev",
            value: 0.0000125176770006874,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / BothRev",
            value: 0.000012977086000319104,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / PreRev",
            value: 0.000013043174000813453,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / PostRev",
            value: 0.00001313548300004186,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / BothRev",
            value: 0.00001317143300002499,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / PreRev",
            value: 0.000013083456000458684,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / PostRev",
            value: 0.00001315791300021374,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / BothRev",
            value: 0.000013224344000263954,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / Primal",
            value: 0.000004176458998699672,
            unit: "s",
          },
          {
            name: "add_two / Jax / cpu / Primal",
            value: 0.00000389441699735471,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / Primal",
            value: 0.000003913834003469674,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / Primal",
            value: 0.000003956541997467866,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / Primal",
            value: 0.000003969542001868831,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / Primal",
            value: 0.000003856667000945891,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / Primal",
            value: 0.000003932542000256945,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / Forward",
            value: 0.000006311625002126675,
            unit: "s",
          },
          {
            name: "add_two / Jax / cpu / Forward",
            value: 0.00000635849999889615,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / Forward",
            value: 0.000006237707999389386,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / Forward",
            value: 0.000006162792000395711,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / Forward",
            value: 0.000006441333000111626,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / Forward",
            value: 0.000006154541999421781,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / Forward",
            value: 0.00000652487499974086,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / PreRev",
            value: 0.000008508708000590559,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / PostRev",
            value: 0.000008670541999890702,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / BothRev",
            value: 0.000008139541001582984,
            unit: "s",
          },
          {
            name: "add_two / Jax / cpu / BothRev",
            value: 0.000008185583999875235,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / PreRev",
            value: 0.00000822899999911897,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / PostRev",
            value: 0.000008269083002232947,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / BothRev",
            value: 0.000008592209000198636,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / PreRev",
            value: 0.00000832783300211304,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / PostRev",
            value: 0.000008283250001113629,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / BothRev",
            value: 0.000008666707999509527,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / PreRev",
            value: 0.00000883987500128569,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / PostRev",
            value: 0.000008267708999483147,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / BothRev",
            value: 0.00000843512499704957,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / PreRev",
            value: 0.000008401584000239382,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / PostRev",
            value: 0.000008327999999892199,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / BothRev",
            value: 0.000008234625001932727,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / PreRev",
            value: 0.000008299332999740727,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / PostRev",
            value: 0.000008188083000277402,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / BothRev",
            value: 0.000008174792001227615,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / Primal",
            value: 0.000004675854000197433,
            unit: "s",
          },
          {
            name: "cache / Jax / cpu / Primal",
            value: 0.000004666048999752093,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / Primal",
            value: 0.000004573235999941971,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / Primal",
            value: 0.000004587106999679236,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / Primal",
            value: 0.000004662117999941984,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / Primal",
            value: 0.000004937458999847877,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / Primal",
            value: 0.000004570535000311792,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / Forward",
            value: 0.000010506431999601772,
            unit: "s",
          },
          {
            name: "cache / Jax / cpu / Forward",
            value: 0.0000104174109997075,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / Forward",
            value: 0.000010240330999749858,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / Forward",
            value: 0.000010464596000019813,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / Forward",
            value: 0.00001084272099978989,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / Forward",
            value: 0.000010459201000230678,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / Forward",
            value: 0.0000103518040000381,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / PreRev",
            value: 0.000011820185000033234,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / PostRev",
            value: 0.00001510064500007502,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / BothRev",
            value: 0.000011821550000149729,
            unit: "s",
          },
          {
            name: "cache / Jax / cpu / BothRev",
            value: 0.000014906167999924948,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / PreRev",
            value: 0.00001158583100004762,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / PostRev",
            value: 0.00001172857199981081,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / BothRev",
            value: 0.000011430990999997449,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / PreRev",
            value: 0.000011895862000073977,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / PostRev",
            value: 0.00001527992199999062,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / BothRev",
            value: 0.000012050281000028918,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / PreRev",
            value: 0.000011760671000047296,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / PostRev",
            value: 0.000015416877000006934,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / BothRev",
            value: 0.000012019366000004084,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / PreRev",
            value: 0.000011560836999706226,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / PostRev",
            value: 0.000011910085999716105,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / BothRev",
            value: 0.000011873816999923292,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / PreRev",
            value: 0.000011666095000236964,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / PostRev",
            value: 0.000011721417999979168,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / BothRev",
            value: 0.000012099084000055882,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / Primal",
            value: 0.000009471198995015584,
            unit: "s",
          },
          {
            name: "cache / Jax / cpu / Primal",
            value: 0.000009210884003550743,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / Primal",
            value: 0.000009940500007360244,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / Primal",
            value: 0.000009263479005312548,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / Primal",
            value: 0.000009228712995536623,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / Primal",
            value: 0.000009195300997816958,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / Primal",
            value: 0.000009240939005394466,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / Forward",
            value: 0.000014933410013327376,
            unit: "s",
          },
          {
            name: "cache / Jax / cpu / Forward",
            value: 0.000015231847995892166,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / Forward",
            value: 0.000014797771000303327,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / Forward",
            value: 0.000014508734006085434,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / Forward",
            value: 0.000015214732004096732,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / Forward",
            value: 0.000014424433989915997,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / Forward",
            value: 0.000014405641006305812,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / PreRev",
            value: 0.000014690265990793703,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / PostRev",
            value: 0.000017124714999226852,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / BothRev",
            value: 0.0000143313120061066,
            unit: "s",
          },
          {
            name: "cache / Jax / cpu / BothRev",
            value: 0.00001723713899264112,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / PreRev",
            value: 0.000015745291995699517,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / PostRev",
            value: 0.00001584926100622397,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / BothRev",
            value: 0.00001543363000382669,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / PreRev",
            value: 0.000015199361005215906,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / PostRev",
            value: 0.00001710987399565056,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / BothRev",
            value: 0.00001523940400511492,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / PreRev",
            value: 0.000014892062012222596,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / PostRev",
            value: 0.000017155638997792265,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / BothRev",
            value: 0.00001495524699566886,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / PreRev",
            value: 0.00001580369399744086,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / PostRev",
            value: 0.000015487308002775536,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / BothRev",
            value: 0.00001550695300102234,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / PreRev",
            value: 0.0000157198189990595,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / PostRev",
            value: 0.00001541026699123904,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / BothRev",
            value: 0.000015616409000358545,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / tpu / Primal",
            value: 0.0001315470640001,
            unit: "s",
          },
          {
            name: "cache / Jax / tpu / Primal",
            value: 0.0001349184749997,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / tpu / Primal",
            value: 0.0001344141750014,
            unit: "s",
          },
          {
            name: "cache / PartOpt / tpu / Primal",
            value: 0.0001349042649999,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / tpu / Primal",
            value: 0.0001317543940022,
            unit: "s",
          },
          {
            name: "cache / DefOpt / tpu / Primal",
            value: 0.0001353355749997,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / tpu / Primal",
            value: 0.0001348681050003,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / tpu / Forward",
            value: 0.0002049014440017,
            unit: "s",
          },
          {
            name: "cache / Jax / tpu / Forward",
            value: 0.0002049637340023,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / tpu / Forward",
            value: 0.0002041902839991,
            unit: "s",
          },
          {
            name: "cache / PartOpt / tpu / Forward",
            value: 0.0001961706009969,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / tpu / Forward",
            value: 0.0002061688740031,
            unit: "s",
          },
          {
            name: "cache / DefOpt / tpu / Forward",
            value: 0.0001937941899996,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / tpu / Forward",
            value: 0.0001949795,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / tpu / PreRev",
            value: 0.0001964860509979,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / tpu / PostRev",
            value: 0.0002017622929997,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / tpu / BothRev",
            value: 0.0002033645730007,
            unit: "s",
          },
          {
            name: "cache / Jax / tpu / BothRev",
            value: 0.0002125411270026,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / tpu / PreRev",
            value: 0.0002218275209997,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / tpu / PostRev",
            value: 0.0002079254150012,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / tpu / BothRev",
            value: 0.0002053013240001,
            unit: "s",
          },
          {
            name: "cache / PartOpt / tpu / PreRev",
            value: 0.0002087634759991,
            unit: "s",
          },
          {
            name: "cache / PartOpt / tpu / PostRev",
            value: 0.0002104135260015,
            unit: "s",
          },
          {
            name: "cache / PartOpt / tpu / BothRev",
            value: 0.0001990765110022,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / tpu / PreRev",
            value: 0.0002194495090006,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / tpu / PostRev",
            value: 0.0002079429650002,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / tpu / BothRev",
            value: 0.0001924568480026,
            unit: "s",
          },
          {
            name: "cache / DefOpt / tpu / PreRev",
            value: 0.0001899028679981,
            unit: "s",
          },
          {
            name: "cache / DefOpt / tpu / PostRev",
            value: 0.000204856534001,
            unit: "s",
          },
          {
            name: "cache / DefOpt / tpu / BothRev",
            value: 0.0002549919039993,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / tpu / PreRev",
            value: 0.0002554222149992,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / tpu / PostRev",
            value: 0.0002519043629981,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / tpu / BothRev",
            value: 0.0002502000730019,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / Primal",
            value: 0.000006185993000144663,
            unit: "s",
          },
          {
            name: "cache / Jax / cpu / Primal",
            value: 0.000006101132000367215,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / Primal",
            value: 0.000006087214999752178,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / Primal",
            value: 0.000006667818000096304,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / Primal",
            value: 0.000006091520999689237,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / Primal",
            value: 0.000006038550999619474,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / Primal",
            value: 0.000006056886999431299,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / Forward",
            value: 0.000010915288999967744,
            unit: "s",
          },
          {
            name: "cache / Jax / cpu / Forward",
            value: 0.000010980328000187,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / Forward",
            value: 0.000010389977999693656,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / Forward",
            value: 0.000010813320999659482,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / Forward",
            value: 0.000010755715000414056,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / Forward",
            value: 0.00001073503099996742,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / Forward",
            value: 0.000010993152000082772,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / PreRev",
            value: 0.000011105722000138484,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / PostRev",
            value: 0.000012298417999772936,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / BothRev",
            value: 0.00001137709700014966,
            unit: "s",
          },
          {
            name: "cache / Jax / cpu / BothRev",
            value: 0.000012531393999779538,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / PreRev",
            value: 0.00001145793199975742,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / PostRev",
            value: 0.0000110608590002812,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / BothRev",
            value: 0.000011154154999530872,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / PreRev",
            value: 0.00001129592599954776,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / PostRev",
            value: 0.000012630473999706738,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / BothRev",
            value: 0.00001066177200027596,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / PreRev",
            value: 0.000010737738999523572,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / PostRev",
            value: 0.00001250656899992464,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / BothRev",
            value: 0.000011029348999727517,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / PreRev",
            value: 0.000011371958999916388,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / PostRev",
            value: 0.000011390049000510773,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / BothRev",
            value: 0.000010960384999634698,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / PreRev",
            value: 0.000011257971000304678,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / PostRev",
            value: 0.000010744835999503264,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / BothRev",
            value: 0.000011200977000044077,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / Primal",
            value: 0.0000033394590027455708,
            unit: "s",
          },
          {
            name: "cache / Jax / cpu / Primal",
            value: 0.0000034451659994374494,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / Primal",
            value: 0.0000035128749987052286,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / Primal",
            value: 0.0000033745420005288907,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / Primal",
            value: 0.0000033696249993226955,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / Primal",
            value: 0.000003373042000021087,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / Primal",
            value: 0.000003493541000352707,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / Forward",
            value: 0.000006344042001728667,
            unit: "s",
          },
          {
            name: "cache / Jax / cpu / Forward",
            value: 0.000006274208000832004,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / Forward",
            value: 0.000006163583999295952,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / Forward",
            value: 0.000006123040999227669,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / Forward",
            value: 0.000006296832998486934,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / Forward",
            value: 0.0000065214169990213125,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / Forward",
            value: 0.0000063886669995554255,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / PreRev",
            value: 0.000006422375001420732,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / PostRev",
            value: 0.000007417083001200808,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / BothRev",
            value: 0.000006825083000876475,
            unit: "s",
          },
          {
            name: "cache / Jax / cpu / BothRev",
            value: 0.000006867209001939045,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / PreRev",
            value: 0.000006067833997803973,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / PostRev",
            value: 0.000006299292002950096,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / BothRev",
            value: 0.000008272375001979527,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / PreRev",
            value: 0.000006562250000570202,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / PostRev",
            value: 0.000008687417001056019,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / BothRev",
            value: 0.000006352625001454726,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / PreRev",
            value: 0.000006330332998913946,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / PostRev",
            value: 0.000007063458000629907,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / BothRev",
            value: 0.000007739958000456682,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / PreRev",
            value: 0.000007074541998008499,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / PostRev",
            value: 0.000006337457998597529,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / BothRev",
            value: 0.00000631533399791806,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / PreRev",
            value: 0.000006850500001746695,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / PostRev",
            value: 0.000006243040999834193,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / BothRev",
            value: 0.000006720084002154181,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / Primal",
            value: 0.000004907408999770269,
            unit: "s",
          },
          {
            name: "Concat / Jax / cpu / Primal",
            value: 0.000004869280000093568,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / Primal",
            value: 0.000004939090999869223,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / Primal",
            value: 0.0000050524250000307805,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / Primal",
            value: 0.000004972166000243305,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / Primal",
            value: 0.000004938526999922033,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / Primal",
            value: 0.000004871904000083305,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / Forward",
            value: 0.000007868974000302842,
            unit: "s",
          },
          {
            name: "Concat / Jax / cpu / Forward",
            value: 0.000007847444999697472,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / Forward",
            value: 0.000007882404999691061,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / Forward",
            value: 0.000008046903999911592,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / Forward",
            value: 0.000008001982000223506,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / Forward",
            value: 0.000007999239000128,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / Forward",
            value: 0.000007886783000230934,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / PreRev",
            value: 0.000008616175000042859,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / PostRev",
            value: 0.000008725426000182778,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / BothRev",
            value: 0.000008597200000167504,
            unit: "s",
          },
          {
            name: "Concat / Jax / cpu / BothRev",
            value: 0.000008692635999977937,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / PreRev",
            value: 0.000008684453000114445,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / PostRev",
            value: 0.000008706112999789184,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / BothRev",
            value: 0.000008684579000146187,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / PreRev",
            value: 0.000008800829999927373,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / PostRev",
            value: 0.00000857721100010167,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / BothRev",
            value: 0.00000851213700025255,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / PreRev",
            value: 0.00000875226299967835,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / PostRev",
            value: 0.00000876732799997626,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / BothRev",
            value: 0.00000880761700000221,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / PreRev",
            value: 0.000008716673999970226,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / PostRev",
            value: 0.00000869712099984099,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / BothRev",
            value: 0.000008715338999991217,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / PreRev",
            value: 0.00000871597400009705,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / PostRev",
            value: 0.000009158297999874776,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / BothRev",
            value: 0.000008687695000389794,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / Primal",
            value: 0.000009114853004575709,
            unit: "s",
          },
          {
            name: "Concat / Jax / cpu / Primal",
            value: 0.000009446525000385009,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / Primal",
            value: 0.0000090627549943747,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / Primal",
            value: 0.000009213049997924828,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / Primal",
            value: 0.000008911921002436429,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / Primal",
            value: 0.000008925931004341691,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / Primal",
            value: 0.000009126433011260817,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / Forward",
            value: 0.000014574495988199488,
            unit: "s",
          },
          {
            name: "Concat / Jax / cpu / Forward",
            value: 0.000014071639001485892,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / Forward",
            value: 0.000013856299003236928,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / Forward",
            value: 0.0000147377300017979,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / Forward",
            value: 0.000013889522000681607,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / Forward",
            value: 0.00001470152600086294,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / Forward",
            value: 0.000014572862011846156,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / PreRev",
            value: 0.000015976112001226284,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / PostRev",
            value: 0.000015826304006623105,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / BothRev",
            value: 0.000015843949004192837,
            unit: "s",
          },
          {
            name: "Concat / Jax / cpu / BothRev",
            value: 0.000015746641001896932,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / PreRev",
            value: 0.00001576277099957224,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / PostRev",
            value: 0.000016110477998154238,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / BothRev",
            value: 0.000015852061012992635,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / PreRev",
            value: 0.000015801493005710654,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / PostRev",
            value: 0.00001589107999461703,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / BothRev",
            value: 0.00001494443800766021,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / PreRev",
            value: 0.000016071471996838226,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / PostRev",
            value: 0.00001589074599905871,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / BothRev",
            value: 0.000015910621994407848,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / PreRev",
            value: 0.000015842270004213786,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / PostRev",
            value: 0.000015876435005338864,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / BothRev",
            value: 0.000015863284992519767,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / PreRev",
            value: 0.00001505670700862538,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / PostRev",
            value: 0.00001590878699789755,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / BothRev",
            value: 0.000015890748996753246,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / tpu / Primal",
            value: 0.0001481045010004,
            unit: "s",
          },
          {
            name: "Concat / Jax / tpu / Primal",
            value: 0.0001667403990031,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / tpu / Primal",
            value: 0.0001479893799987,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / tpu / Primal",
            value: 0.0001439512190008,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / tpu / Primal",
            value: 0.0001443306090004,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / tpu / Primal",
            value: 0.0001466146499988,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / tpu / Primal",
            value: 0.0001468842400026,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / tpu / Forward",
            value: 0.0002200844299986,
            unit: "s",
          },
          {
            name: "Concat / Jax / tpu / Forward",
            value: 0.000204752454003,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / tpu / Forward",
            value: 0.0002040412839996,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / tpu / Forward",
            value: 0.0002044149529974,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / tpu / Forward",
            value: 0.0002042350239971,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / tpu / Forward",
            value: 0.0002037675329993,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / tpu / Forward",
            value: 0.0002028334940005,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / tpu / PreRev",
            value: 0.0002140562679996,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / tpu / PostRev",
            value: 0.0002005497220015,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / tpu / BothRev",
            value: 0.0002018194929987,
            unit: "s",
          },
          {
            name: "Concat / Jax / tpu / BothRev",
            value: 0.000201056212998,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / tpu / PreRev",
            value: 0.000205424224001,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / tpu / PostRev",
            value: 0.0002042272839971,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / tpu / BothRev",
            value: 0.000201458882002,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / tpu / PreRev",
            value: 0.0002112702769991,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / tpu / PostRev",
            value: 0.0002216482000003,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / tpu / BothRev",
            value: 0.0002152661579966,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / tpu / PreRev",
            value: 0.0002054447839982,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / tpu / PostRev",
            value: 0.0002117936070026,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / tpu / BothRev",
            value: 0.0002173344889997,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / tpu / PreRev",
            value: 0.0002129465070029,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / tpu / PostRev",
            value: 0.000202478733001,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / tpu / BothRev",
            value: 0.0002282886030006,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / tpu / PreRev",
            value: 0.0002140486780008,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / tpu / PostRev",
            value: 0.0002140216070001,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / tpu / BothRev",
            value: 0.0002114737570009,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / Primal",
            value: 0.000006550350999532384,
            unit: "s",
          },
          {
            name: "Concat / Jax / cpu / Primal",
            value: 0.000006329344999358,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / Primal",
            value: 0.000006579848999535898,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / Primal",
            value: 0.000006731288999617391,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / Primal",
            value: 0.000006407547999515373,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / Primal",
            value: 0.000006263774999752059,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / Primal",
            value: 0.000006747915999767429,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / Forward",
            value: 0.000010381103999861809,
            unit: "s",
          },
          {
            name: "Concat / Jax / cpu / Forward",
            value: 0.000010309219999726338,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / Forward",
            value: 0.000009799180000300112,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / Forward",
            value: 0.000010262185000101451,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / Forward",
            value: 0.00001003195199973561,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / Forward",
            value: 0.000010440402000313044,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / Forward",
            value: 0.000010002841000641638,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / PreRev",
            value: 0.00001068765799936955,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / PostRev",
            value: 0.000011122720999992451,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / BothRev",
            value: 0.000011232697000195912,
            unit: "s",
          },
          {
            name: "Concat / Jax / cpu / BothRev",
            value: 0.000011256425000283344,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / PreRev",
            value: 0.000011190345000613889,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / PostRev",
            value: 0.000011167394999574753,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / BothRev",
            value: 0.00001118324599974585,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / PreRev",
            value: 0.000011214792000828311,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / PostRev",
            value: 0.000011148216000037793,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / BothRev",
            value: 0.000010624333000123444,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / PreRev",
            value: 0.000011144804000650763,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / PostRev",
            value: 0.000011186207999344334,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / BothRev",
            value: 0.00001121837200025766,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / PreRev",
            value: 0.00001066985399938858,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / PostRev",
            value: 0.00001126790299986169,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / BothRev",
            value: 0.0000104863070000647,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / PreRev",
            value: 0.000011345728999913262,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / PostRev",
            value: 0.000011294163999991724,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / BothRev",
            value: 0.000010524548999455874,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / Primal",
            value: 0.000003857417003018782,
            unit: "s",
          },
          {
            name: "Concat / Jax / cpu / Primal",
            value: 0.000003866666000249097,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / Primal",
            value: 0.000003902915999788093,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / Primal",
            value: 0.000004020374999527121,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / Primal",
            value: 0.000003958290999435122,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / Primal",
            value: 0.000003884041001583682,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / Primal",
            value: 0.000003896707999956561,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / Forward",
            value: 0.000006281291000050259,
            unit: "s",
          },
          {
            name: "Concat / Jax / cpu / Forward",
            value: 0.000006310667002253467,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / Forward",
            value: 0.000006309125001280336,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / Forward",
            value: 0.000006201207997946767,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / Forward",
            value: 0.000006149666998680914,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / Forward",
            value: 0.0000069942919981258455,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / Forward",
            value: 0.000006180707998282742,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / PreRev",
            value: 0.000007298499996977625,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / PostRev",
            value: 0.0000070992500004649625,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / BothRev",
            value: 0.000007062665998091689,
            unit: "s",
          },
          {
            name: "Concat / Jax / cpu / BothRev",
            value: 0.000007479457999579608,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / PreRev",
            value: 0.000007184875001257751,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / PostRev",
            value: 0.000007087542002409463,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / BothRev",
            value: 0.000008041749999392777,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / PreRev",
            value: 0.000007387334000668489,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / PostRev",
            value: 0.000007579083001473919,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / BothRev",
            value: 0.000007370749997789971,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / PreRev",
            value: 0.000007608125000842847,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / PostRev",
            value: 0.000007259417001478141,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / BothRev",
            value: 0.000007019125001534121,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / PreRev",
            value: 0.000007133874998544343,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / PostRev",
            value: 0.00000878841599842417,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / BothRev",
            value: 0.000006946916997549124,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / PreRev",
            value: 0.0000069262910001270935,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / PostRev",
            value: 0.00000687116700282786,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / BothRev",
            value: 0.000007573499999125488,
            unit: "s",
          },
          {
            name: "const_scatter / JaXPipe / cpu / Primal",
            value: 0.000004596324999965873,
            unit: "s",
          },
          {
            name: "const_scatter / Jax / cpu / Primal",
            value: 0.000004857883999648038,
            unit: "s",
          },
          {
            name: "const_scatter / HLOOpt / cpu / Primal",
            value: 0.000005415237999841338,
            unit: "s",
          },
          {
            name: "const_scatter / PartOpt / cpu / Primal",
            value: 0.000004886074000296503,
            unit: "s",
          },
          {
            name: "const_scatter / IPartOpt / cpu / Primal",
            value: 0.000004820498000299267,
            unit: "s",
          },
          {
            name: "const_scatter / DefOpt / cpu / Primal",
            value: 0.000005445936999876721,
            unit: "s",
          },
          {
            name: "const_scatter / IDefOpt / cpu / Primal",
            value: 0.000005190793000110716,
            unit: "s",
          },
          {
            name: "const_scatter / JaXPipe / cpu / Forward",
            value: 0.000008530107999831671,
            unit: "s",
          },
          {
            name: "const_scatter / Jax / cpu / Forward",
            value: 0.000007008794999819657,
            unit: "s",
          },
          {
            name: "const_scatter / HLOOpt / cpu / Forward",
            value: 0.000008547500000076979,
            unit: "s",
          },
          {
            name: "const_scatter / PartOpt / cpu / Forward",
            value: 0.000008541612000044551,
            unit: "s",
          },
          {
            name: "const_scatter / IPartOpt / cpu / Forward",
            value: 0.000008537735000118119,
            unit: "s",
          },
          {
            name: "const_scatter / DefOpt / cpu / Forward",
            value: 0.000008120221999888599,
            unit: "s",
          },
          {
            name: "const_scatter / IDefOpt / cpu / Forward",
            value: 0.000008131458000207203,
            unit: "s",
          },
          {
            name: "const_scatter / JaXPipe / cpu / Primal",
            value: 0.000008833658997900784,
            unit: "s",
          },
          {
            name: "const_scatter / Jax / cpu / Primal",
            value: 0.000008852191007463262,
            unit: "s",
          },
          {
            name: "const_scatter / HLOOpt / cpu / Primal",
            value: 0.000009919212010572665,
            unit: "s",
          },
          {
            name: "const_scatter / PartOpt / cpu / Primal",
            value: 0.000009318280994193628,
            unit: "s",
          },
          {
            name: "const_scatter / IPartOpt / cpu / Primal",
            value: 0.000009310977009590717,
            unit: "s",
          },
          {
            name: "const_scatter / DefOpt / cpu / Primal",
            value: 0.000010479698001290672,
            unit: "s",
          },
          {
            name: "const_scatter / IDefOpt / cpu / Primal",
            value: 0.000010492986999452114,
            unit: "s",
          },
          {
            name: "const_scatter / JaXPipe / cpu / Forward",
            value: 0.00001540337900223676,
            unit: "s",
          },
          {
            name: "const_scatter / Jax / cpu / Forward",
            value: 0.00001336465499480255,
            unit: "s",
          },
          {
            name: "const_scatter / HLOOpt / cpu / Forward",
            value: 0.00001484781900944654,
            unit: "s",
          },
          {
            name: "const_scatter / PartOpt / cpu / Forward",
            value: 0.000014563389995601029,
            unit: "s",
          },
          {
            name: "const_scatter / IPartOpt / cpu / Forward",
            value: 0.000015408874998684042,
            unit: "s",
          },
          {
            name: "const_scatter / DefOpt / cpu / Forward",
            value: 0.000015560707004624418,
            unit: "s",
          },
          {
            name: "const_scatter / IDefOpt / cpu / Forward",
            value: 0.000015380245997221208,
            unit: "s",
          },
          {
            name: "const_scatter / JaXPipe / tpu / Primal",
            value: 0.0001348262549981,
            unit: "s",
          },
          {
            name: "const_scatter / Jax / tpu / Primal",
            value: 0.0001551450240003,
            unit: "s",
          },
          {
            name: "const_scatter / HLOOpt / tpu / Primal",
            value: 0.0001343494249995,
            unit: "s",
          },
          {
            name: "const_scatter / PartOpt / tpu / Primal",
            value: 0.0001345430660003,
            unit: "s",
          },
          {
            name: "const_scatter / IPartOpt / tpu / Primal",
            value: 0.0001347584950017,
            unit: "s",
          },
          {
            name: "const_scatter / DefOpt / tpu / Primal",
            value: 0.0001351222650009,
            unit: "s",
          },
          {
            name: "const_scatter / IDefOpt / tpu / Primal",
            value: 0.0001304846829989,
            unit: "s",
          },
          {
            name: "const_scatter / JaXPipe / tpu / Forward",
            value: 0.0002035811740024,
            unit: "s",
          },
          {
            name: "const_scatter / Jax / tpu / Forward",
            value: 0.0001997941519985,
            unit: "s",
          },
          {
            name: "const_scatter / HLOOpt / tpu / Forward",
            value: 0.0001967126509989,
            unit: "s",
          },
          {
            name: "const_scatter / PartOpt / tpu / Forward",
            value: 0.0002008044020003,
            unit: "s",
          },
          {
            name: "const_scatter / IPartOpt / tpu / Forward",
            value: 0.0002093612249991,
            unit: "s",
          },
          {
            name: "const_scatter / DefOpt / tpu / Forward",
            value: 0.0002036492639999,
            unit: "s",
          },
          {
            name: "const_scatter / IDefOpt / tpu / Forward",
            value: 0.0001989648710004,
            unit: "s",
          },
          {
            name: "const_scatter / JaXPipe / cpu / Primal",
            value: 0.000006094543000472185,
            unit: "s",
          },
          {
            name: "const_scatter / Jax / cpu / Primal",
            value: 0.0000060988900004304015,
            unit: "s",
          },
          {
            name: "const_scatter / HLOOpt / cpu / Primal",
            value: 0.000006817376000071818,
            unit: "s",
          },
          {
            name: "const_scatter / PartOpt / cpu / Primal",
            value: 0.000006463450999945053,
            unit: "s",
          },
          {
            name: "const_scatter / IPartOpt / cpu / Primal",
            value: 0.000006408667999494355,
            unit: "s",
          },
          {
            name: "const_scatter / DefOpt / cpu / Primal",
            value: 0.000007219389000056253,
            unit: "s",
          },
          {
            name: "const_scatter / IDefOpt / cpu / Primal",
            value: 0.000007160942000155046,
            unit: "s",
          },
          {
            name: "const_scatter / JaXPipe / cpu / Forward",
            value: 0.000010735241000475073,
            unit: "s",
          },
          {
            name: "const_scatter / Jax / cpu / Forward",
            value: 0.000008768178999162047,
            unit: "s",
          },
          {
            name: "const_scatter / HLOOpt / cpu / Forward",
            value: 0.000010703570000259788,
            unit: "s",
          },
          {
            name: "const_scatter / PartOpt / cpu / Forward",
            value: 0.000010750956000265432,
            unit: "s",
          },
          {
            name: "const_scatter / IPartOpt / cpu / Forward",
            value: 0.000010718295000515352,
            unit: "s",
          },
          {
            name: "const_scatter / DefOpt / cpu / Forward",
            value: 0.0000107436210000742,
            unit: "s",
          },
          {
            name: "const_scatter / IDefOpt / cpu / Forward",
            value: 0.000010266592999869316,
            unit: "s",
          },
          {
            name: "const_scatter / JaXPipe / cpu / Primal",
            value: 0.0000037669170014851263,
            unit: "s",
          },
          {
            name: "const_scatter / Jax / cpu / Primal",
            value: 0.000003659000001789536,
            unit: "s",
          },
          {
            name: "const_scatter / HLOOpt / cpu / Primal",
            value: 0.0000042221250005241015,
            unit: "s",
          },
          {
            name: "const_scatter / PartOpt / cpu / Primal",
            value: 0.000003571207998902537,
            unit: "s",
          },
          {
            name: "const_scatter / IPartOpt / cpu / Primal",
            value: 0.000003554082999471575,
            unit: "s",
          },
          {
            name: "const_scatter / DefOpt / cpu / Primal",
            value: 0.000004134458002226893,
            unit: "s",
          },
          {
            name: "const_scatter / IDefOpt / cpu / Primal",
            value: 0.00000395662499795435,
            unit: "s",
          },
          {
            name: "const_scatter / JaXPipe / cpu / Forward",
            value: 0.000006773083998268703,
            unit: "s",
          },
          {
            name: "const_scatter / Jax / cpu / Forward",
            value: 0.0000054830830013088415,
            unit: "s",
          },
          {
            name: "const_scatter / HLOOpt / cpu / Forward",
            value: 0.0000064839170008781366,
            unit: "s",
          },
          {
            name: "const_scatter / PartOpt / cpu / Forward",
            value: 0.000006498624999949243,
            unit: "s",
          },
          {
            name: "const_scatter / IPartOpt / cpu / Forward",
            value: 0.000006532457999128383,
            unit: "s",
          },
          {
            name: "const_scatter / DefOpt / cpu / Forward",
            value: 0.000006765833000827115,
            unit: "s",
          },
          {
            name: "const_scatter / IDefOpt / cpu / Forward",
            value: 0.000006581708999874536,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / Primal",
            value: 0.000004975824999746692,
            unit: "s",
          },
          {
            name: "GenDot / Jax / cpu / Primal",
            value: 0.000005151205999936792,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / Primal",
            value: 0.000005443418000140809,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / Primal",
            value: 0.000005018544999984443,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / Primal",
            value: 0.000005036200000176905,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / Primal",
            value: 0.000005251444000350603,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / Primal",
            value: 0.000005310136999923998,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / Forward",
            value: 0.000008251483000094596,
            unit: "s",
          },
          {
            name: "GenDot / Jax / cpu / Forward",
            value: 0.000007677414999761823,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / Forward",
            value: 0.000008394479999878968,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / Forward",
            value: 0.000008128268000291427,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / Forward",
            value: 0.000008217581000280916,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / Forward",
            value: 0.000008111898000151996,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / Forward",
            value: 0.00000811398600035318,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / PreRev",
            value: 0.00000809050899988506,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / PostRev",
            value: 0.00000755630799994833,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / BothRev",
            value: 0.000008111875999929907,
            unit: "s",
          },
          {
            name: "GenDot / Jax / cpu / BothRev",
            value: 0.000007472335999864299,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / PreRev",
            value: 0.000008116528999835281,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / PostRev",
            value: 0.000008178072999726283,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / BothRev",
            value: 0.000008160559999851102,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / PreRev",
            value: 0.000008143547999679868,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / PostRev",
            value: 0.00000758531600013157,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / BothRev",
            value: 0.00000814757299986013,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / PreRev",
            value: 0.00000815507900006196,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / PostRev",
            value: 0.000007452761999957147,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / BothRev",
            value: 0.000008105691999844567,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / PreRev",
            value: 0.000008174043000053644,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / PostRev",
            value: 0.000008124535000206379,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / BothRev",
            value: 0.000008219032999932096,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / PreRev",
            value: 0.000008081122000021423,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / PostRev",
            value: 0.000008145131000219408,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / BothRev",
            value: 0.000008149622000019008,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / Primal",
            value: 0.00000941529599367641,
            unit: "s",
          },
          {
            name: "GenDot / Jax / cpu / Primal",
            value: 0.000010099132006871516,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / Primal",
            value: 0.000010762138001155108,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / Primal",
            value: 0.000010068018003948964,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / Primal",
            value: 0.00000968400199781172,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / Primal",
            value: 0.000010815895002451726,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / Primal",
            value: 0.000011385844001779332,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / Forward",
            value: 0.000015874445991357788,
            unit: "s",
          },
          {
            name: "GenDot / Jax / cpu / Forward",
            value: 0.000014298994996352122,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / Forward",
            value: 0.000015772123995702714,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / Forward",
            value: 0.00001597540799411945,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / Forward",
            value: 0.000016727870010072364,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / Forward",
            value: 0.00001585210399935022,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / Forward",
            value: 0.000016446380002889783,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / PreRev",
            value: 0.000015847935996134765,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / PostRev",
            value: 0.000014405889000045135,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / BothRev",
            value: 0.00001607048499863595,
            unit: "s",
          },
          {
            name: "GenDot / Jax / cpu / BothRev",
            value: 0.000014590008009690792,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / PreRev",
            value: 0.000015809028001967817,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / PostRev",
            value: 0.000016553205001400782,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / BothRev",
            value: 0.00001580151300004218,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / PreRev",
            value: 0.000015841731990803965,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / PostRev",
            value: 0.00001356075301009696,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / BothRev",
            value: 0.000015686496000853367,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / PreRev",
            value: 0.00001655625199782662,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / PostRev",
            value: 0.000013626640997244976,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / BothRev",
            value: 0.000015776970991282725,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / PreRev",
            value: 0.000016631835998850875,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / PostRev",
            value: 0.000016593968000961467,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / BothRev",
            value: 0.00001597015600418672,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / PreRev",
            value: 0.00001656653601094149,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / PostRev",
            value: 0.0000166782919986872,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / BothRev",
            value: 0.000016654673003358768,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / tpu / Primal",
            value: 0.000129692663002,
            unit: "s",
          },
          {
            name: "GenDot / Jax / tpu / Primal",
            value: 0.0001325976840016,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / tpu / Primal",
            value: 0.0001316370940003,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / tpu / Primal",
            value: 0.0001317879939997,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / tpu / Primal",
            value: 0.000134451965001,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / tpu / Primal",
            value: 0.0001335042550017,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / tpu / Primal",
            value: 0.000134591705002,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / tpu / Forward",
            value: 0.0001981406009981,
            unit: "s",
          },
          {
            name: "GenDot / Jax / tpu / Forward",
            value: 0.0002093028660019,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / tpu / Forward",
            value: 0.0002084331449987,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / tpu / Forward",
            value: 0.0002095609850002,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / tpu / Forward",
            value: 0.0001838531249995,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / tpu / Forward",
            value: 0.000219819140002,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / tpu / Forward",
            value: 0.0001949937400022,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / tpu / PreRev",
            value: 0.0001969153900026,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / tpu / PostRev",
            value: 0.0001985113110022,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / tpu / BothRev",
            value: 0.0001943364990002,
            unit: "s",
          },
          {
            name: "GenDot / Jax / tpu / BothRev",
            value: 0.0001947256390012,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / tpu / PreRev",
            value: 0.0001951353899967,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / tpu / PostRev",
            value: 0.0001977801009998,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / tpu / BothRev",
            value: 0.0001966393309994,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / tpu / PreRev",
            value: 0.0002016665930023,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / tpu / PostRev",
            value: 0.0002186016890009,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / tpu / BothRev",
            value: 0.0002104980160002,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / tpu / PreRev",
            value: 0.0001997381010005,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / tpu / PostRev",
            value: 0.0002021610230003,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / tpu / BothRev",
            value: 0.0002035229829998,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / tpu / PreRev",
            value: 0.0001995870120008,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / tpu / PostRev",
            value: 0.0001998906820008,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / tpu / BothRev",
            value: 0.0001976169709996,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / tpu / PreRev",
            value: 0.0002051463440002,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / tpu / PostRev",
            value: 0.000199384411997,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / tpu / BothRev",
            value: 0.0002000563120018,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / Primal",
            value: 0.000006526830000439077,
            unit: "s",
          },
          {
            name: "GenDot / Jax / cpu / Primal",
            value: 0.0000068851229998472265,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / Primal",
            value: 0.00000781880300019111,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / Primal",
            value: 0.0000069838139997955295,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / Primal",
            value: 0.000006620535999900312,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / Primal",
            value: 0.00000762055299946951,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / Primal",
            value: 0.000007445827000083227,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / Forward",
            value: 0.0000106998110004497,
            unit: "s",
          },
          {
            name: "GenDot / Jax / cpu / Forward",
            value: 0.000010180301999753283,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / Forward",
            value: 0.000011277570999482123,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / Forward",
            value: 0.000011381493999579107,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / Forward",
            value: 0.000011380277000171193,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / Forward",
            value: 0.000011327461000291806,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / Forward",
            value: 0.000010660859999916285,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / PreRev",
            value: 0.000011454831999799351,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / PostRev",
            value: 0.00001002062899988232,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / BothRev",
            value: 0.000010792504999699304,
            unit: "s",
          },
          {
            name: "GenDot / Jax / cpu / BothRev",
            value: 0.000010196866000114825,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / PreRev",
            value: 0.000011482089000310224,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / PostRev",
            value: 0.000010845493000488203,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / BothRev",
            value: 0.000010797084999467188,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / PreRev",
            value: 0.000010730572000284156,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / PostRev",
            value: 0.000010300791999725334,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / BothRev",
            value: 0.000010930137000286777,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / PreRev",
            value: 0.00001137223800014908,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / PostRev",
            value: 0.000009558489000482953,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / BothRev",
            value: 0.000011455519999799437,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / PreRev",
            value: 0.000011327709999932269,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / PostRev",
            value: 0.00001147247900007642,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / BothRev",
            value: 0.000010827082000105292,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / PreRev",
            value: 0.000011460682000688391,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / PostRev",
            value: 0.000011322854000354707,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / BothRev",
            value: 0.000011472856000182218,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / Primal",
            value: 0.000003931625000404893,
            unit: "s",
          },
          {
            name: "GenDot / Jax / cpu / Primal",
            value: 0.000003828000000794418,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / Primal",
            value: 0.000004090207999979612,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / Primal",
            value: 0.000003846750001684995,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / Primal",
            value: 0.000003926875000615837,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / Primal",
            value: 0.0000042787500024132895,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / Primal",
            value: 0.000004160415999649558,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / Forward",
            value: 0.000006793833999836352,
            unit: "s",
          },
          {
            name: "GenDot / Jax / cpu / Forward",
            value: 0.0000060959170004935,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / Forward",
            value: 0.0000070312499992724044,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / Forward",
            value: 0.00000690204099737457,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / Forward",
            value: 0.000006654791999608279,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / Forward",
            value: 0.000006611625001823995,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / Forward",
            value: 0.000006567833002918633,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / PreRev",
            value: 0.000006512457999633625,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / PostRev",
            value: 0.000006473917001130757,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / BothRev",
            value: 0.000006873916001495672,
            unit: "s",
          },
          {
            name: "GenDot / Jax / cpu / BothRev",
            value: 0.000006314707999990788,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / PreRev",
            value: 0.000006804874999943422,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / PostRev",
            value: 0.000006788582999433857,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / BothRev",
            value: 0.0000067462920014804695,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / PreRev",
            value: 0.00000651037499847007,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / PostRev",
            value: 0.000006048792001820402,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / BothRev",
            value: 0.000006712292000884191,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / PreRev",
            value: 0.000006765541998902336,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / PostRev",
            value: 0.000006001457997626858,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / BothRev",
            value: 0.000006788624999899184,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / PreRev",
            value: 0.0000065957080005318855,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / PostRev",
            value: 0.000006714542003464885,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / BothRev",
            value: 0.000006424708000849932,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / PreRev",
            value: 0.000006854708000901155,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / PostRev",
            value: 0.00000677529200038407,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / BothRev",
            value: 0.000006647500002145535,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / Primal",
            value: 0.000007586625999920216,
            unit: "s",
          },
          {
            name: "hlo_ffi / Jax / cpu / Primal",
            value: 0.000007596615999773348,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / Primal",
            value: 0.000007671337999909155,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / Primal",
            value: 0.000007741612000245369,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / Primal",
            value: 0.000007836610000140355,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / Primal",
            value: 0.000007798202999765635,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / Primal",
            value: 0.000007804716000009648,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / Forward",
            value: 0.000012440245000107097,
            unit: "s",
          },
          {
            name: "hlo_ffi / Jax / cpu / Forward",
            value: 0.000012434952999683449,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / Forward",
            value: 0.000012211088000185556,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / Forward",
            value: 0.000012365599000077054,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / Forward",
            value: 0.000012436183999852802,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / Forward",
            value: 0.000012408582000261958,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / Forward",
            value: 0.00001242727900034879,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / PreRev",
            value: 0.000012002837999716576,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / PostRev",
            value: 0.000012249987999894074,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / BothRev",
            value: 0.00001239974899999652,
            unit: "s",
          },
          {
            name: "hlo_ffi / Jax / cpu / BothRev",
            value: 0.00001239864699982718,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / PreRev",
            value: 0.000012325238999892464,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / PostRev",
            value: 0.000012091869999949268,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / BothRev",
            value: 0.00001255757100034316,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / PreRev",
            value: 0.0000125461309999082,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / PostRev",
            value: 0.000012156092000168428,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / BothRev",
            value: 0.000012078229000053395,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / PreRev",
            value: 0.00001215740800034837,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / PostRev",
            value: 0.000012662852999710594,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / BothRev",
            value: 0.000011940260999836028,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / PreRev",
            value: 0.00001218155000015031,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / PostRev",
            value: 0.000012239877999945749,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / BothRev",
            value: 0.000011970130999998218,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / PreRev",
            value: 0.000012224838999827626,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / PostRev",
            value: 0.000012222081999880174,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / BothRev",
            value: 0.00001204086500001722,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / Primal",
            value: 0.000013739906993578189,
            unit: "s",
          },
          {
            name: "hlo_ffi / Jax / cpu / Primal",
            value: 0.000013831716991262513,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / Primal",
            value: 0.000014347005999297837,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / Primal",
            value: 0.00001436723100778181,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / Primal",
            value: 0.000014320321002742276,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / Primal",
            value: 0.000014340893001644871,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / Primal",
            value: 0.000014427012007217854,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / Forward",
            value: 0.00001968444300291594,
            unit: "s",
          },
          {
            name: "hlo_ffi / Jax / cpu / Forward",
            value: 0.000019770922997849997,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / Forward",
            value: 0.000019787678989814594,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / Forward",
            value: 0.000020657786997617224,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / Forward",
            value: 0.00002066029299749061,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / Forward",
            value: 0.000020751874995767134,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / Forward",
            value: 0.00002061487099854276,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / PreRev",
            value: 0.00002017018500191625,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / PostRev",
            value: 0.00002006757199706044,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / BothRev",
            value: 0.000020036860005347988,
            unit: "s",
          },
          {
            name: "hlo_ffi / Jax / cpu / BothRev",
            value: 0.000020057298010215163,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / PreRev",
            value: 0.00002029882698843721,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / PostRev",
            value: 0.000019961946993134915,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / BothRev",
            value: 0.000020074611995369195,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / PreRev",
            value: 0.000019271387995104302,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / PostRev",
            value: 0.00001930840199929662,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / BothRev",
            value: 0.0000193120179901598,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / PreRev",
            value: 0.00002008891799778212,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / PostRev",
            value: 0.000020107656004256568,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / BothRev",
            value: 0.00002018527300970163,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / PreRev",
            value: 0.000020231878006597985,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / PostRev",
            value: 0.000020346427001641133,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / BothRev",
            value: 0.000020133612997597085,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / PreRev",
            value: 0.000020101859990973023,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / PostRev",
            value: 0.000020180359002551997,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / BothRev",
            value: 0.000019387435997487045,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / tpu / Primal",
            value: 0.0002146490720006,
            unit: "s",
          },
          {
            name: "hlo_ffi / Jax / tpu / Primal",
            value: 0.0002322299890001,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / tpu / Primal",
            value: 0.000224160736001,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / tpu / Primal",
            value: 0.0002223487650007,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / tpu / Primal",
            value: 0.0002253934260006,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / tpu / Primal",
            value: 0.0002384861319987,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / tpu / Primal",
            value: 0.0002318904790008,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / tpu / Forward",
            value: 0.0002561817200003,
            unit: "s",
          },
          {
            name: "hlo_ffi / Jax / tpu / Forward",
            value: 0.0002504426170016,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / tpu / Forward",
            value: 0.0002578744999991,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / tpu / Forward",
            value: 0.0002108433800021,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / tpu / Forward",
            value: 0.0002245484459999,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / tpu / Forward",
            value: 0.0002414407530013,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / tpu / Forward",
            value: 0.0002427582139971,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / tpu / PreRev",
            value: 0.000239689553,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / tpu / PostRev",
            value: 0.0002174599129975,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / tpu / BothRev",
            value: 0.0002217708140015,
            unit: "s",
          },
          {
            name: "hlo_ffi / Jax / tpu / BothRev",
            value: 0.0002255593459994,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / tpu / PreRev",
            value: 0.0002226402850028,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / tpu / PostRev",
            value: 0.0002448358740002,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / tpu / BothRev",
            value: 0.0002430758240006,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / tpu / PreRev",
            value: 0.0002357357809996,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / tpu / PostRev",
            value: 0.0002368703110005,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / tpu / BothRev",
            value: 0.000229335598,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / tpu / PreRev",
            value: 0.0002327107189994,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / tpu / PostRev",
            value: 0.0002525356580008,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / tpu / BothRev",
            value: 0.0002425743729982,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / tpu / PreRev",
            value: 0.0002416705140021,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / tpu / PostRev",
            value: 0.0002203038839979,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / tpu / BothRev",
            value: 0.0002046311570011,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / tpu / PreRev",
            value: 0.0001919479520001,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / tpu / PostRev",
            value: 0.0001858975589966,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / tpu / BothRev",
            value: 0.0002417023429989,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / Primal",
            value: 0.000010279436000018904,
            unit: "s",
          },
          {
            name: "hlo_ffi / Jax / cpu / Primal",
            value: 0.000009706738999739171,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / Primal",
            value: 0.000010149255000214907,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / Primal",
            value: 0.000010154278999834787,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / Primal",
            value: 0.0000102229760004775,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / Primal",
            value: 0.00000964985699920362,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / Primal",
            value: 0.000009767540999746417,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / Forward",
            value: 0.00001420102100018994,
            unit: "s",
          },
          {
            name: "hlo_ffi / Jax / cpu / Forward",
            value: 0.000014658231999419514,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / Forward",
            value: 0.00001411543399990478,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / Forward",
            value: 0.000014721499999723164,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / Forward",
            value: 0.00001474477199917601,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / Forward",
            value: 0.000014211814999725905,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / Forward",
            value: 0.000014625494000028992,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / PreRev",
            value: 0.000013805083999614,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / PostRev",
            value: 0.000014418607999687084,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / BothRev",
            value: 0.000014383247999830928,
            unit: "s",
          },
          {
            name: "hlo_ffi / Jax / cpu / BothRev",
            value: 0.000014455567999902996,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / PreRev",
            value: 0.000014603120000174386,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / PostRev",
            value: 0.00001443208299951948,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / BothRev",
            value: 0.000014505715000268538,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / PreRev",
            value: 0.000014465346000179125,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / PostRev",
            value: 0.000013969282999823916,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / BothRev",
            value: 0.000014014628999575508,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / PreRev",
            value: 0.000014557542999682485,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / PostRev",
            value: 0.0000143117629995686,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / BothRev",
            value: 0.000014493863000097918,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / PreRev",
            value: 0.00001465229699988413,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / PostRev",
            value: 0.000014466030000221508,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / BothRev",
            value: 0.00001453518099970097,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / PreRev",
            value: 0.000014981790000092588,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / PostRev",
            value: 0.000014552318999449198,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / BothRev",
            value: 0.0000144463149999865,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / Primal",
            value: 0.000006479374998889398,
            unit: "s",
          },
          {
            name: "hlo_ffi / Jax / cpu / Primal",
            value: 0.000006397250002919464,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / Primal",
            value: 0.000006205790999956662,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / Primal",
            value: 0.000005979792000289308,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / Primal",
            value: 0.000006114083000284154,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / Primal",
            value: 0.000006079082999349339,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / Primal",
            value: 0.000006076583002140978,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / Forward",
            value: 0.00000948962500115158,
            unit: "s",
          },
          {
            name: "hlo_ffi / Jax / cpu / Forward",
            value: 0.000009830374998273328,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / Forward",
            value: 0.000009215417001541937,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / Forward",
            value: 0.000010032457998022437,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / Forward",
            value: 0.000009590957997716032,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / Forward",
            value: 0.000009503582998149796,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / Forward",
            value: 0.000009613999998691724,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / PreRev",
            value: 0.00000910741600091569,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / PostRev",
            value: 0.000009261250001145526,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / BothRev",
            value: 0.00000937533300020732,
            unit: "s",
          },
          {
            name: "hlo_ffi / Jax / cpu / BothRev",
            value: 0.000009262375002435877,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / PreRev",
            value: 0.000008997875000204657,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / PostRev",
            value: 0.000009195582999382168,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / BothRev",
            value: 0.00000917608300005668,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / PreRev",
            value: 0.000008902541998395464,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / PostRev",
            value: 0.000009289249999710592,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / BothRev",
            value: 0.000008828916998027126,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / PreRev",
            value: 0.000009010292000311892,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / PostRev",
            value: 0.000009020833000249696,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / BothRev",
            value: 0.000009134040999924765,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / PreRev",
            value: 0.00000915329200142878,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / PostRev",
            value: 0.000008943165998061887,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / BothRev",
            value: 0.00000913225000113016,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / PreRev",
            value: 0.000009263457999622917,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / PostRev",
            value: 0.00000930037499711034,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / BothRev",
            value: 0.000009718707999127218,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / Primal",
            value: 0.0008780294999723,
            unit: "s",
          },
          {
            name: "llama / Jax / cpu / Primal",
            value: 0.0008572873000048,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / Primal",
            value: 0.0009463679000418,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / Primal",
            value: 0.0008664248000059,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / Primal",
            value: 0.0008515281999734,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / Primal",
            value: 0.0009478042000409,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / Primal",
            value: 0.0009547402999942,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / Forward",
            value: 0.0023222832000101,
            unit: "s",
          },
          {
            name: "llama / Jax / cpu / Forward",
            value: 0.0023501974999817,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / Forward",
            value: 0.0023391060999983,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / Forward",
            value: 0.0023384467999676,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / Forward",
            value: 0.0023446397000043,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / Forward",
            value: 0.0023593521000293,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / Forward",
            value: 0.0023730701000204,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / PreRev",
            value: 0.0053726552000171,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / PostRev",
            value: 0.005287661200009,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / BothRev",
            value: 0.0032593995000297,
            unit: "s",
          },
          {
            name: "llama / Jax / cpu / BothRev",
            value: 0.0054160099,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / PreRev",
            value: 0.0032828998999775,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / PostRev",
            value: 0.0051378388000102,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / BothRev",
            value: 0.0032498861000021,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / PreRev",
            value: 0.0052611089000038,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / PostRev",
            value: 0.0031827746999624,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / BothRev",
            value: 0.0052365268999892,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / PreRev",
            value: 0.0032321078999757,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / PostRev",
            value: 0.0054085674999896,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / BothRev",
            value: 0.0032305833000009,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / PreRev",
            value: 0.0051106021000123,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / PostRev",
            value: 0.0030801749000147,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / BothRev",
            value: 0.0050997606000237,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / PreRev",
            value: 0.0032601638999949,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / PostRev",
            value: 0.0052967099999932,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / BothRev",
            value: 0.0032522727000014,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / Primal",
            value: 0.0016094211998279,
            unit: "s",
          },
          {
            name: "llama / Jax / cpu / Primal",
            value: 0.0016590746003203,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / Primal",
            value: 0.0017434084002161,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / Primal",
            value: 0.0016064461990026,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / Primal",
            value: 0.0015472414001123,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / Primal",
            value: 0.001658156600024,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / Primal",
            value: 0.001648543299234,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / Forward",
            value: 0.0052244184000301,
            unit: "s",
          },
          {
            name: "llama / Jax / cpu / Forward",
            value: 0.0052146917005302,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / Forward",
            value: 0.0052209073997801,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / Forward",
            value: 0.0051760573987849,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / Forward",
            value: 0.0052291880987468,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / Forward",
            value: 0.0052184129002853,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / Forward",
            value: 0.005172422499163,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / PreRev",
            value: 0.0105235588998766,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / PostRev",
            value: 0.0115202950008097,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / BothRev",
            value: 0.010047979200317,
            unit: "s",
          },
          {
            name: "llama / Jax / cpu / BothRev",
            value: 0.0101489601991488,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / PreRev",
            value: 0.0113677187997382,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / PostRev",
            value: 0.0099296607004362,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / BothRev",
            value: 0.0113089004007633,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / PreRev",
            value: 0.0096571771995513,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / PostRev",
            value: 0.0113341792006394,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / BothRev",
            value: 0.0098238724996917,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / PreRev",
            value: 0.0113063911005156,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / PostRev",
            value: 0.0103260895004495,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / BothRev",
            value: 0.0112977576005505,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / PreRev",
            value: 0.0101598827008274,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / PostRev",
            value: 0.0097881516994675,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / BothRev",
            value: 0.0098729329009074,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / PreRev",
            value: 0.010052431099757,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / PostRev",
            value: 0.0099683214997639,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / BothRev",
            value: 0.0101297973000328,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / tpu / Primal",
            value: 0.0003595833600411,
            unit: "s",
          },
          {
            name: "llama / Jax / tpu / Primal",
            value: 0.0003610737400595,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / tpu / Primal",
            value: 0.0003534579599363,
            unit: "s",
          },
          {
            name: "llama / PartOpt / tpu / Primal",
            value: 0.0003688119399885,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / tpu / Primal",
            value: 0.000367002760031,
            unit: "s",
          },
          {
            name: "llama / DefOpt / tpu / Primal",
            value: 0.0003367875399999,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / tpu / Primal",
            value: 0.0003526119600428,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / tpu / Forward",
            value: 0.000565635439998,
            unit: "s",
          },
          {
            name: "llama / Jax / tpu / Forward",
            value: 0.0006827520799561,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / tpu / Forward",
            value: 0.000572197040019,
            unit: "s",
          },
          {
            name: "llama / PartOpt / tpu / Forward",
            value: 0.0005455480399541,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / tpu / Forward",
            value: 0.0005673570599901,
            unit: "s",
          },
          {
            name: "llama / DefOpt / tpu / Forward",
            value: 0.0005636394400062,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / tpu / Forward",
            value: 0.0005689992599945,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / tpu / PreRev",
            value: 0.0007868459399469,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / tpu / PostRev",
            value: 0.0007490985199547,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / tpu / BothRev",
            value: 0.0007822101400233,
            unit: "s",
          },
          {
            name: "llama / Jax / tpu / BothRev",
            value: 0.0007598123200295,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / tpu / PreRev",
            value: 0.0007893473400326,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / tpu / PostRev",
            value: 0.0007825521400081,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / tpu / BothRev",
            value: 0.0007518105199415,
            unit: "s",
          },
          {
            name: "llama / PartOpt / tpu / PreRev",
            value: 0.0007635129399568,
            unit: "s",
          },
          {
            name: "llama / PartOpt / tpu / PostRev",
            value: 0.0007775363200198,
            unit: "s",
          },
          {
            name: "llama / PartOpt / tpu / BothRev",
            value: 0.0007497001199953,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / tpu / PreRev",
            value: 0.0007499483200081,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / tpu / PostRev",
            value: 0.0007657333400129,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / tpu / BothRev",
            value: 0.0007760763200349,
            unit: "s",
          },
          {
            name: "llama / DefOpt / tpu / PreRev",
            value: 0.0007429763199615,
            unit: "s",
          },
          {
            name: "llama / DefOpt / tpu / PostRev",
            value: 0.0007659001399588,
            unit: "s",
          },
          {
            name: "llama / DefOpt / tpu / BothRev",
            value: 0.0007233404999715,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / tpu / PreRev",
            value: 0.0007312593199458,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / tpu / PostRev",
            value: 0.0008129389399982,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / tpu / BothRev",
            value: 0.0008077415399748,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / Primal",
            value: 0.0020501276000686,
            unit: "s",
          },
          {
            name: "llama / Jax / cpu / Primal",
            value: 0.0020159222999609,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / Primal",
            value: 0.0021790652999698,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / Primal",
            value: 0.0019849326999974,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / Primal",
            value: 0.0019453251999948,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / Primal",
            value: 0.0021464460999595,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / Primal",
            value: 0.0021655399999872,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / Forward",
            value: 0.0058492615999966,
            unit: "s",
          },
          {
            name: "llama / Jax / cpu / Forward",
            value: 0.0059957217000373,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / Forward",
            value: 0.0059686504000637,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / Forward",
            value: 0.0059178016999794,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / Forward",
            value: 0.005900210100026,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / Forward",
            value: 0.0058643129999836,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / Forward",
            value: 0.0059809801000483,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / PreRev",
            value: 0.0100591948999863,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / PostRev",
            value: 0.0092724081000596,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / BothRev",
            value: 0.0107228165000378,
            unit: "s",
          },
          {
            name: "llama / Jax / cpu / BothRev",
            value: 0.0092368915000406,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / PreRev",
            value: 0.0103864568000062,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / PostRev",
            value: 0.0098829893000583,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / BothRev",
            value: 0.0099317123999753,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / PreRev",
            value: 0.0103697411000212,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / PostRev",
            value: 0.0095079507999798,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / BothRev",
            value: 0.0099101927000447,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / PreRev",
            value: 0.0102482235999559,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / PostRev",
            value: 0.0091871538999839,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / BothRev",
            value: 0.0105853063000722,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / PreRev",
            value: 0.0101135742999758,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / PostRev",
            value: 0.0096672513000157,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / BothRev",
            value: 0.0103942450999966,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / PreRev",
            value: 0.0099197814000035,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / PostRev",
            value: 0.0083737397999357,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / BothRev",
            value: 0.0099672645999817,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / Primal",
            value: 0.0016964541999186,
            unit: "s",
          },
          {
            name: "llama / Jax / cpu / Primal",
            value: 0.0014841166001133,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / Primal",
            value: 0.0017312915999355,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / Primal",
            value: 0.0014671875000203,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / Primal",
            value: 0.0015767833003337,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / Primal",
            value: 0.0015404499998112,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / Primal",
            value: 0.001893070799997,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / Forward",
            value: 0.0039341584000794,
            unit: "s",
          },
          {
            name: "llama / Jax / cpu / Forward",
            value: 0.0041904957997758,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / Forward",
            value: 0.0043107666999276,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / Forward",
            value: 0.0073012415999983,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / Forward",
            value: 0.006280875000084,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / Forward",
            value: 0.0073101875001157,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / Forward",
            value: 0.008134608399996,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / PreRev",
            value: 0.0151077583002916,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / PostRev",
            value: 0.0205850791997363,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / BothRev",
            value: 0.0097060957999929,
            unit: "s",
          },
          {
            name: "llama / Jax / cpu / BothRev",
            value: 0.0124918792000244,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / PreRev",
            value: 0.0116855832999135,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / PostRev",
            value: 0.0107809541997994,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / BothRev",
            value: 0.0114970540998911,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / PreRev",
            value: 0.0092332250002073,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / PostRev",
            value: 0.0112810750000789,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / BothRev",
            value: 0.0089738709000812,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / PreRev",
            value: 0.0093882583001686,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / PostRev",
            value: 0.0167760166001244,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / BothRev",
            value: 0.0131233207997865,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / PreRev",
            value: 0.0095775625002715,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / PostRev",
            value: 0.0074592707998817,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / BothRev",
            value: 0.008990066700062,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / PreRev",
            value: 0.0097399540998594,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / PostRev",
            value: 0.0076183708999451,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / BothRev",
            value: 0.0104403166998963,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / Primal",
            value: 0.000005668315000093572,
            unit: "s",
          },
          {
            name: "scatter_sum / Jax / cpu / Primal",
            value: 0.000005560194999816303,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / Primal",
            value: 0.000005472655000176019,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / Primal",
            value: 0.000005579178000061802,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / Primal",
            value: 0.000005582643999787251,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / Primal",
            value: 0.0000055781060000299474,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / Primal",
            value: 0.000005600752000191278,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / Forward",
            value: 0.000008538881999811565,
            unit: "s",
          },
          {
            name: "scatter_sum / Jax / cpu / Forward",
            value: 0.000008485692999784078,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / Forward",
            value: 0.000008531442999810678,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / Forward",
            value: 0.000008555561999855855,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / Forward",
            value: 0.000008502359999965847,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / Forward",
            value: 0.000008531766000032803,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / Forward",
            value: 0.00000850982300016767,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / PreRev",
            value: 0.000008610968000084541,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / PostRev",
            value: 0.000008571351999762556,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / BothRev",
            value: 0.000008601899000041158,
            unit: "s",
          },
          {
            name: "scatter_sum / Jax / cpu / BothRev",
            value: 0.000008598466999956145,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / PreRev",
            value: 0.000008642749000046025,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / PostRev",
            value: 0.000008755485000165209,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / BothRev",
            value: 0.00000856791600017459,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / PreRev",
            value: 0.000008634360000087326,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / PostRev",
            value: 0.000008541946000150346,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / BothRev",
            value: 0.000008599241999945662,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / PreRev",
            value: 0.000008614333999958034,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / PostRev",
            value: 0.000008551377000003413,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / BothRev",
            value: 0.00000855366599989793,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / PreRev",
            value: 0.000008584031000282266,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / PostRev",
            value: 0.000008590685999934068,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / BothRev",
            value: 0.000008562909999909608,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / PreRev",
            value: 0.000008554752999771153,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / PostRev",
            value: 0.000008539218999885633,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / BothRev",
            value: 0.00000871031499991659,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / Primal",
            value: 0.00001059420699311886,
            unit: "s",
          },
          {
            name: "scatter_sum / Jax / cpu / Primal",
            value: 0.000010806418998981826,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / Primal",
            value: 0.00001138578899553977,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / Primal",
            value: 0.00001128095701278653,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / Primal",
            value: 0.000010895996994804592,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / Primal",
            value: 0.000010791140011860987,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / Primal",
            value: 0.000010939259998849591,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / Forward",
            value: 0.00001624832100060303,
            unit: "s",
          },
          {
            name: "scatter_sum / Jax / cpu / Forward",
            value: 0.000016392864999943413,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / Forward",
            value: 0.000015589865986839868,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / Forward",
            value: 0.000015425001998664813,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / Forward",
            value: 0.00001568832299381029,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / Forward",
            value: 0.000016561812997679225,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / Forward",
            value: 0.000016317551999236457,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / PreRev",
            value: 0.00001572627499990631,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / PostRev",
            value: 0.000016377633990487085,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / BothRev",
            value: 0.0000163697069947375,
            unit: "s",
          },
          {
            name: "scatter_sum / Jax / cpu / BothRev",
            value: 0.000016396790990256706,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / PreRev",
            value: 0.00001572846200724598,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / PostRev",
            value: 0.000015777616004925222,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / BothRev",
            value: 0.00001663901399297174,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / PreRev",
            value: 0.000016470777001813984,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / PostRev",
            value: 0.000016473752999445423,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / BothRev",
            value: 0.000015677545001381077,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / PreRev",
            value: 0.000016398747000494042,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / PostRev",
            value: 0.000016386515999329275,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / BothRev",
            value: 0.00001647324100486003,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / PreRev",
            value: 0.000016397155006416142,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / PostRev",
            value: 0.00001637166799628176,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / BothRev",
            value: 0.000016567892002058215,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / PreRev",
            value: 0.000015640077996067703,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / PostRev",
            value: 0.000016542823999770917,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / BothRev",
            value: 0.000015802460999111645,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / tpu / Primal",
            value: 0.0001334583140014,
            unit: "s",
          },
          {
            name: "scatter_sum / Jax / tpu / Primal",
            value: 0.0001346866849999,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / tpu / Primal",
            value: 0.0001352428760001,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / tpu / Primal",
            value: 0.0001352345849991,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / tpu / Primal",
            value: 0.0001357901449991,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / tpu / Primal",
            value: 0.0001308784039974,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / tpu / Primal",
            value: 0.000131145053001,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / tpu / Forward",
            value: 0.0001942550700005,
            unit: "s",
          },
          {
            name: "scatter_sum / Jax / tpu / Forward",
            value: 0.0001945337100005,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / tpu / Forward",
            value: 0.000197062670999,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / tpu / Forward",
            value: 0.0002044956540012,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / tpu / Forward",
            value: 0.0001992663420023,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / tpu / Forward",
            value: 0.0001974335509985,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / tpu / Forward",
            value: 0.0001984179210012,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / tpu / PreRev",
            value: 0.0001983266509996,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / tpu / PostRev",
            value: 0.0002142380370023,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / tpu / BothRev",
            value: 0.0002010346320021,
            unit: "s",
          },
          {
            name: "scatter_sum / Jax / tpu / BothRev",
            value: 0.0002034343129998,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / tpu / PreRev",
            value: 0.0002160951680016,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / tpu / PostRev",
            value: 0.0002115173159982,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / tpu / BothRev",
            value: 0.000200712992002,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / tpu / PreRev",
            value: 0.0001984434909973,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / tpu / PostRev",
            value: 0.0002272042930017,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / tpu / BothRev",
            value: 0.0002575967849988,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / tpu / PreRev",
            value: 0.0002543549740003,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / tpu / PostRev",
            value: 0.0002529304039999,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / tpu / BothRev",
            value: 0.0002326395849995,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / tpu / PreRev",
            value: 0.0002269888030023,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / tpu / PostRev",
            value: 0.000201290181998,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / tpu / BothRev",
            value: 0.0002155969879968,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / tpu / PreRev",
            value: 0.0002017184830001,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / tpu / PostRev",
            value: 0.0002134992969986,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / tpu / BothRev",
            value: 0.000215750329,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / Primal",
            value: 0.000007419210000080056,
            unit: "s",
          },
          {
            name: "scatter_sum / Jax / cpu / Primal",
            value: 0.000007413645000269753,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / Primal",
            value: 0.000007763896000142267,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / Primal",
            value: 0.00000777131700033351,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / Primal",
            value: 0.000007781903999784845,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / Primal",
            value: 0.000007355695999649469,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / Primal",
            value: 0.000007125248999727774,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / Forward",
            value: 0.000010713366999880236,
            unit: "s",
          },
          {
            name: "scatter_sum / Jax / cpu / Forward",
            value: 0.00001127584599998954,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / Forward",
            value: 0.000010695590000068478,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / Forward",
            value: 0.0000113989199999196,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / Forward",
            value: 0.000011316402000375093,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / Forward",
            value: 0.000010775747000479896,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / Forward",
            value: 0.00001123516700045002,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / PreRev",
            value: 0.000011448686000221642,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / PostRev",
            value: 0.000011395702999834612,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / BothRev",
            value: 0.000011395930000617229,
            unit: "s",
          },
          {
            name: "scatter_sum / Jax / cpu / BothRev",
            value: 0.000010830631999851903,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / PreRev",
            value: 0.000011499285999889252,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / PostRev",
            value: 0.000011416300999371744,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / BothRev",
            value: 0.000011349063999659848,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / PreRev",
            value: 0.000010805711000102747,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / PostRev",
            value: 0.000010810976999891863,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / BothRev",
            value: 0.000010945315000753908,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / PreRev",
            value: 0.000011540111000613252,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / PostRev",
            value: 0.00001140010900053312,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / BothRev",
            value: 0.000010832390999894414,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / PreRev",
            value: 0.000011464461000286977,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / PostRev",
            value: 0.000010823931000231824,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / BothRev",
            value: 0.000011357538999618557,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / PreRev",
            value: 0.000011430370999732984,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / PostRev",
            value: 0.00001153250800052774,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / BothRev",
            value: 0.000011018992000572324,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / Primal",
            value: 0.000004606249996868428,
            unit: "s",
          },
          {
            name: "scatter_sum / Jax / cpu / Primal",
            value: 0.00000451374999829568,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / Primal",
            value: 0.000004558249998808606,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / Primal",
            value: 0.00000456737499916926,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / Primal",
            value: 0.000004733624999062158,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / Primal",
            value: 0.000004704167000454617,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / Primal",
            value: 0.000004762167001899797,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / Forward",
            value: 0.000007062333999783732,
            unit: "s",
          },
          {
            name: "scatter_sum / Jax / cpu / Forward",
            value: 0.000007368832997599384,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / Forward",
            value: 0.000007112666000466561,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / Forward",
            value: 0.00000742220800020732,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / Forward",
            value: 0.00000731837499915855,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / Forward",
            value: 0.000007238124999275896,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / Forward",
            value: 0.000007260291000420693,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / PreRev",
            value: 0.000007362374999502208,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / PostRev",
            value: 0.000007737749998341315,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / BothRev",
            value: 0.000007228375001432141,
            unit: "s",
          },
          {
            name: "scatter_sum / Jax / cpu / BothRev",
            value: 0.000007270624999364372,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / PreRev",
            value: 0.00000740708300145343,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / PostRev",
            value: 0.000007502292002754985,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / BothRev",
            value: 0.000008177625000826084,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / PreRev",
            value: 0.000007352707998506958,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / PostRev",
            value: 0.000008247416000813245,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / BothRev",
            value: 0.000007600290999107528,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / PreRev",
            value: 0.00000738262499726261,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / PostRev",
            value: 0.000007288708002306521,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / BothRev",
            value: 0.000008103459000267321,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / PreRev",
            value: 0.000007768792002025294,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / PostRev",
            value: 0.00000730716699763434,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / BothRev",
            value: 0.0000075812919967575,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / PreRev",
            value: 0.000007411082999169594,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / PostRev",
            value: 0.000007412374998239102,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / BothRev",
            value: 0.00000786083300044993,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / Primal",
            value: 0.000004446147999715322,
            unit: "s",
          },
          {
            name: "slicing / Jax / cpu / Primal",
            value: 0.000004514179999659973,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / Primal",
            value: 0.000004412103000049683,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / Primal",
            value: 0.000004449960999863833,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / Primal",
            value: 0.000004501610999795958,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / Primal",
            value: 0.000004522373999861884,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / Primal",
            value: 0.000004486412999995082,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / Forward",
            value: 0.000006906478000018978,
            unit: "s",
          },
          {
            name: "slicing / Jax / cpu / Forward",
            value: 0.000006880927999645792,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / Forward",
            value: 0.000006918749000305979,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / Forward",
            value: 0.000006914957999924809,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / Forward",
            value: 0.00000689722700008133,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / Forward",
            value: 0.000006909200999871246,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / Forward",
            value: 0.000006929454999863083,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / PreRev",
            value: 0.000007332281999879342,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / PostRev",
            value: 0.000007360869999956776,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / BothRev",
            value: 0.000007397487000162073,
            unit: "s",
          },
          {
            name: "slicing / Jax / cpu / BothRev",
            value: 0.00000735015200007183,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / PreRev",
            value: 0.000007401709000077971,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / PostRev",
            value: 0.000007364302000041789,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / BothRev",
            value: 0.0000074001739999403075,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / PreRev",
            value: 0.00000735571399991386,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / PostRev",
            value: 0.000007360599000094226,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / BothRev",
            value: 0.000007374005999736255,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / PreRev",
            value: 0.000007409761999952025,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / PostRev",
            value: 0.000007343938000303751,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / BothRev",
            value: 0.00000736425600007351,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / PreRev",
            value: 0.000007369802999619424,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / PostRev",
            value: 0.000007369895000010728,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / BothRev",
            value: 0.000007339607999711006,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / PreRev",
            value: 0.000007284704000085184,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / PostRev",
            value: 0.000007365559999925608,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / BothRev",
            value: 0.0000073545909999666034,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / Primal",
            value: 0.000008523265001713298,
            unit: "s",
          },
          {
            name: "slicing / Jax / cpu / Primal",
            value: 0.000009069047009688804,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / Primal",
            value: 0.00000906866199511569,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / Primal",
            value: 0.000008549250997020864,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / Primal",
            value: 0.000009055959002580492,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / Primal",
            value: 0.000008547669000108727,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / Primal",
            value: 0.000009012633003294468,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / Forward",
            value: 0.00001228381299006287,
            unit: "s",
          },
          {
            name: "slicing / Jax / cpu / Forward",
            value: 0.000013175896994653158,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / Forward",
            value: 0.0000130654609965859,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / Forward",
            value: 0.000012326416006544603,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / Forward",
            value: 0.000013100714000756852,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / Forward",
            value: 0.000012308411998674272,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / Forward",
            value: 0.000012366074995952658,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / PreRev",
            value: 0.000014154392993077635,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / PostRev",
            value: 0.000013272448995849118,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / BothRev",
            value: 0.00001392412000859622,
            unit: "s",
          },
          {
            name: "slicing / Jax / cpu / BothRev",
            value: 0.000014070072997128593,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / PreRev",
            value: 0.00001322901500680018,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / PostRev",
            value: 0.000013268802998936735,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / BothRev",
            value: 0.000014089936987147666,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / PreRev",
            value: 0.000013225881004473194,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / PostRev",
            value: 0.00001408473499759566,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / BothRev",
            value: 0.000013286102010169998,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / PreRev",
            value: 0.000014084841997828334,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / PostRev",
            value: 0.000014068568008951844,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / BothRev",
            value: 0.000013244711008155718,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / PreRev",
            value: 0.000014031609010999091,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / PostRev",
            value: 0.000014209908011252991,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / BothRev",
            value: 0.000013230789001681842,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / PreRev",
            value: 0.000013235370002803393,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / PostRev",
            value: 0.000014041167989489622,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / BothRev",
            value: 0.00001333267299924046,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / tpu / Primal",
            value: 0.0001345065049972,
            unit: "s",
          },
          {
            name: "slicing / Jax / tpu / Primal",
            value: 0.0001346312259993,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / tpu / Primal",
            value: 0.0001353416550009,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / tpu / Primal",
            value: 0.0001326525540025,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / tpu / Primal",
            value: 0.0001307377429984,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / tpu / Primal",
            value: 0.000135448574998,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / tpu / Primal",
            value: 0.0001353183459978,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / tpu / Forward",
            value: 0.0002099707660017,
            unit: "s",
          },
          {
            name: "slicing / Jax / tpu / Forward",
            value: 0.0002089684050006,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / tpu / Forward",
            value: 0.0001997788219996,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / tpu / Forward",
            value: 0.0002222645910005,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / tpu / Forward",
            value: 0.0002079033260015,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / tpu / Forward",
            value: 0.0002059771939966,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / tpu / Forward",
            value: 0.0002199491499995,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / tpu / PreRev",
            value: 0.0002144257979998,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / tpu / PostRev",
            value: 0.0002138053679991,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / tpu / BothRev",
            value: 0.0002246172020022,
            unit: "s",
          },
          {
            name: "slicing / Jax / tpu / BothRev",
            value: 0.0002255011019988,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / tpu / PreRev",
            value: 0.0002223593810012,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / tpu / PostRev",
            value: 0.0002219821409998,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / tpu / BothRev",
            value: 0.0001997223120015,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / tpu / PreRev",
            value: 0.0001990557309982,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / tpu / PostRev",
            value: 0.0001998128820014,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / tpu / BothRev",
            value: 0.0002076277349988,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / tpu / PreRev",
            value: 0.0001979233909987,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / tpu / PostRev",
            value: 0.0002065152639988,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / tpu / BothRev",
            value: 0.0001986539410027,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / tpu / PreRev",
            value: 0.0002065045149975,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / tpu / PostRev",
            value: 0.0002294618939995,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / tpu / BothRev",
            value: 0.0002231119250027,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / tpu / PreRev",
            value: 0.0001990662749994,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / tpu / PostRev",
            value: 0.0001952128340017,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / tpu / BothRev",
            value: 0.0001948252330003,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / Primal",
            value: 0.000005952503000116849,
            unit: "s",
          },
          {
            name: "slicing / Jax / cpu / Primal",
            value: 0.000005989063000015449,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / Primal",
            value: 0.000006017033999341948,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / Primal",
            value: 0.000006360143000165408,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / Primal",
            value: 0.000005941498000538559,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / Primal",
            value: 0.000006341997000163247,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / Primal",
            value: 0.000006319532999441435,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / Forward",
            value: 0.000008613017000243417,
            unit: "s",
          },
          {
            name: "slicing / Jax / cpu / Forward",
            value: 0.000009185323000565404,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / Forward",
            value: 0.000009165124999526596,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / Forward",
            value: 0.000008626212999843119,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / Forward",
            value: 0.000008608181000454352,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / Forward",
            value: 0.000009268967000025442,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / Forward",
            value: 0.000009250214000530832,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / PreRev",
            value: 0.000009768291999534997,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / PostRev",
            value: 0.000009273057999962476,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / BothRev",
            value: 0.000009768068999619573,
            unit: "s",
          },
          {
            name: "slicing / Jax / cpu / BothRev",
            value: 0.00000979124400055298,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / PreRev",
            value: 0.000009782377999727031,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / PostRev",
            value: 0.000009801970999433251,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / BothRev",
            value: 0.000009809393000068666,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / PreRev",
            value: 0.00000975910300076066,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / PostRev",
            value: 0.000009782742999959737,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / BothRev",
            value: 0.0000093007920004311,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / PreRev",
            value: 0.0000098198189998584,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / PostRev",
            value: 0.0000098157619995618,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / BothRev",
            value: 0.000009796929000003727,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / PreRev",
            value: 0.000009935373999724106,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / PostRev",
            value: 0.000009804089999306598,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / BothRev",
            value: 0.00000928497399945627,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / PreRev",
            value: 0.000009800910000194562,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / PostRev",
            value: 0.000009732026999699884,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / BothRev",
            value: 0.00000993710800048575,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / Primal",
            value: 0.0000036221249974914824,
            unit: "s",
          },
          {
            name: "slicing / Jax / cpu / Primal",
            value: 0.0000036300419997132847,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / Primal",
            value: 0.000003658125002402812,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / Primal",
            value: 0.0000035088750009890646,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / Primal",
            value: 0.000003594208003050881,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / Primal",
            value: 0.000003602583998144837,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / Primal",
            value: 0.0000038606250018347055,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / Forward",
            value: 0.0000056296250004379545,
            unit: "s",
          },
          {
            name: "slicing / Jax / cpu / Forward",
            value: 0.000005604957998002646,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / Forward",
            value: 0.000005868915999599266,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / Forward",
            value: 0.000005579500000749249,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / Forward",
            value: 0.000005874999998923158,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / Forward",
            value: 0.000005745165999542224,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / Forward",
            value: 0.000005887708997761365,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / PreRev",
            value: 0.000005903416000364814,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / PostRev",
            value: 0.000005902125001739478,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / BothRev",
            value: 0.000005835541000124067,
            unit: "s",
          },
          {
            name: "slicing / Jax / cpu / BothRev",
            value: 0.000006122333998064278,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / PreRev",
            value: 0.000006234499996935483,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / PostRev",
            value: 0.000006219165999937104,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / BothRev",
            value: 0.000006296084000496194,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / PreRev",
            value: 0.000006364917000610149,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / PostRev",
            value: 0.000006312917001196183,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / BothRev",
            value: 0.000006107250002969522,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / PreRev",
            value: 0.000006797959002142307,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / PostRev",
            value: 0.00000646979100201861,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / BothRev",
            value: 0.000006261999998969259,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / PreRev",
            value: 0.000006378707999829203,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / PostRev",
            value: 0.000006054708999727154,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / BothRev",
            value: 0.000006050583997421199,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / PreRev",
            value: 0.0000059485419988050125,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / PostRev",
            value: 0.00000592391600002884,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / BothRev",
            value: 0.000006008457999996608,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / Primal",
            value: 0.000005716123999718548,
            unit: "s",
          },
          {
            name: "sum    / Jax / cpu / Primal",
            value: 0.0000060013540000909415,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / Primal",
            value: 0.000006005968999943434,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / Primal",
            value: 0.000006037681000179873,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / Primal",
            value: 0.000005961749999642052,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / Primal",
            value: 0.000005740482999954111,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / Primal",
            value: 0.000006049197000265849,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / Forward",
            value: 0.000009272949999740377,
            unit: "s",
          },
          {
            name: "sum    / Jax / cpu / Forward",
            value: 0.000009252877999642803,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / Forward",
            value: 0.00000929578000022957,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / Forward",
            value: 0.000009266269000363536,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / Forward",
            value: 0.000009270054999888088,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / Forward",
            value: 0.000009261304000119708,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / Forward",
            value: 0.000009250383000107831,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / PreRev",
            value: 0.000008539195000139443,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / PostRev",
            value: 0.00000818991799997093,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / BothRev",
            value: 0.00000813913499996488,
            unit: "s",
          },
          {
            name: "sum    / Jax / cpu / BothRev",
            value: 0.000008194359999833978,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / PreRev",
            value: 0.000008226990999901319,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / PostRev",
            value: 0.000008193492999907903,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / BothRev",
            value: 0.000008166413999788347,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / PreRev",
            value: 0.000008145622000029106,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / PostRev",
            value: 0.000008187029999589868,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / BothRev",
            value: 0.000008157545999893045,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / PreRev",
            value: 0.000008160501000020304,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / PostRev",
            value: 0.00000809924699979092,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / BothRev",
            value: 0.000008159148000231652,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / PreRev",
            value: 0.000008176109000032739,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / PostRev",
            value: 0.000008161724999808938,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / BothRev",
            value: 0.00000819145000014032,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / PreRev",
            value: 0.000008184904000245297,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / PostRev",
            value: 0.000008188308000171673,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / BothRev",
            value: 0.000008185475000118459,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / Primal",
            value: 0.000012349580007139591,
            unit: "s",
          },
          {
            name: "sum    / Jax / cpu / Primal",
            value: 0.000011698166999849491,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / Primal",
            value: 0.000011733406005077996,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / Primal",
            value: 0.0000117203849949874,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / Primal",
            value: 0.00001222998100274708,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / Primal",
            value: 0.000011729609002941288,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / Primal",
            value: 0.00001217716200335417,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / Forward",
            value: 0.000018006707992753945,
            unit: "s",
          },
          {
            name: "sum    / Jax / cpu / Forward",
            value: 0.00001803917800134513,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / Forward",
            value: 0.000017218544002389536,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / Forward",
            value: 0.00001731956799631007,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / Forward",
            value: 0.000017918416007887572,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / Forward",
            value: 0.000018014649001997896,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / Forward",
            value: 0.00001727601200400386,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / PreRev",
            value: 0.000016649362005409786,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / PostRev",
            value: 0.000015626134991180153,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / BothRev",
            value: 0.000016804484999738633,
            unit: "s",
          },
          {
            name: "sum    / Jax / cpu / BothRev",
            value: 0.00001571782799146604,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / PreRev",
            value: 0.000015838721999898554,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / PostRev",
            value: 0.000016823411002405918,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / BothRev",
            value: 0.000015862247004406527,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / PreRev",
            value: 0.000016836762995808386,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / PostRev",
            value: 0.000016707217000657692,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / BothRev",
            value: 0.00001595182799792383,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / PreRev",
            value: 0.000016588687009061688,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / PostRev",
            value: 0.000016794612994999625,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / BothRev",
            value: 0.00001656063499103766,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / PreRev",
            value: 0.000016741945000831037,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / PostRev",
            value: 0.000016534677997697144,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / BothRev",
            value: 0.00001668972299376037,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / PreRev",
            value: 0.00001660012399952393,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / PostRev",
            value: 0.000016566917998716235,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / BothRev",
            value: 0.0000166330219944939,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / tpu / Primal",
            value: 0.0001289755949983,
            unit: "s",
          },
          {
            name: "sum    / Jax / tpu / Primal",
            value: 0.000128635205001,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / tpu / Primal",
            value: 0.0001290244849988,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / tpu / Primal",
            value: 0.0001294618360007,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / tpu / Primal",
            value: 0.0001294424750012,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / tpu / Primal",
            value: 0.0001295816349993,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / tpu / Primal",
            value: 0.0001518820150013,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / tpu / Forward",
            value: 0.0002195155039989,
            unit: "s",
          },
          {
            name: "sum    / Jax / tpu / Forward",
            value: 0.0002200752440003,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / tpu / Forward",
            value: 0.0002195624840023,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / tpu / Forward",
            value: 0.000219310073,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / tpu / Forward",
            value: 0.0002200015529997,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / tpu / Forward",
            value: 0.0002187346630016,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / tpu / Forward",
            value: 0.0002206777039973,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / tpu / PreRev",
            value: 0.0002200667340002,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / tpu / PostRev",
            value: 0.0002208035139992,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / tpu / BothRev",
            value: 0.000221372953998,
            unit: "s",
          },
          {
            name: "sum    / Jax / tpu / BothRev",
            value: 0.0002191357730007,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / tpu / PreRev",
            value: 0.0001948941539994,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / tpu / PostRev",
            value: 0.0001982455950019,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / tpu / BothRev",
            value: 0.0002211777250013,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / tpu / PreRev",
            value: 0.0002210759750014,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / tpu / PostRev",
            value: 0.0002213753040014,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / tpu / BothRev",
            value: 0.0001980298749986,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / tpu / PreRev",
            value: 0.0001981367350017,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / tpu / PostRev",
            value: 0.0002027680669998,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / tpu / BothRev",
            value: 0.0001988509349976,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / tpu / PreRev",
            value: 0.0002072178689995,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / tpu / PostRev",
            value: 0.0002269481169969,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / tpu / BothRev",
            value: 0.0002311312290003,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / tpu / PreRev",
            value: 0.0002214059850011,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / tpu / PostRev",
            value: 0.0002198170939991,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / tpu / BothRev",
            value: 0.0002262363469999,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / Primal",
            value: 0.000007908153000244055,
            unit: "s",
          },
          {
            name: "sum    / Jax / cpu / Primal",
            value: 0.000007962986999700661,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / Primal",
            value: 0.00000825816999986273,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / Primal",
            value: 0.000007938542000374582,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / Primal",
            value: 0.0000083357369994701,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / Primal",
            value: 0.000008250140999734868,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / Primal",
            value: 0.000008277861999886227,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / Forward",
            value: 0.00001213068599918188,
            unit: "s",
          },
          {
            name: "sum    / Jax / cpu / Forward",
            value: 0.00001216404099977808,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / Forward",
            value: 0.000011834734000331082,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / Forward",
            value: 0.000012198518999866793,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / Forward",
            value: 0.000012162729000010583,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / Forward",
            value: 0.000012257063000106428,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / Forward",
            value: 0.000012214341999424505,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / PreRev",
            value: 0.0000114219500001127,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / PostRev",
            value: 0.0000108642229997713,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / BothRev",
            value: 0.00001076593400011916,
            unit: "s",
          },
          {
            name: "sum    / Jax / cpu / BothRev",
            value: 0.00001083024200033833,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / PreRev",
            value: 0.000011315946999275183,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / PostRev",
            value: 0.000010751652000180912,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / BothRev",
            value: 0.00001145514500058198,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / PreRev",
            value: 0.000011364100999344372,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / PostRev",
            value: 0.000010767609999675187,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / BothRev",
            value: 0.000010832365000169375,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / PreRev",
            value: 0.000010784869000417527,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / PostRev",
            value: 0.000011413354000069376,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / BothRev",
            value: 0.000011397995999686827,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / PreRev",
            value: 0.000011404941000364489,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / PostRev",
            value: 0.00001137773600021319,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / BothRev",
            value: 0.0000113267779997841,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / PreRev",
            value: 0.000010845640000297862,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / PostRev",
            value: 0.000011451112000031571,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / BothRev",
            value: 0.000010811199999807286,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / Primal",
            value: 0.000004645624998374842,
            unit: "s",
          },
          {
            name: "sum    / Jax / cpu / Primal",
            value: 0.000004722125002444955,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / Primal",
            value: 0.000004745207999803825,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / Primal",
            value: 0.000004866792001848807,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / Primal",
            value: 0.000004642500000045402,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / Primal",
            value: 0.00000464441700023599,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / Primal",
            value: 0.000004535957999905804,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / Forward",
            value: 0.000007437374999426538,
            unit: "s",
          },
          {
            name: "sum    / Jax / cpu / Forward",
            value: 0.000007282583999767666,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / Forward",
            value: 0.000007668209000257776,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / Forward",
            value: 0.000007314416998269735,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / Forward",
            value: 0.000007504125002014917,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / Forward",
            value: 0.000007455416998709552,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / Forward",
            value: 0.000010121499999513616,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / PreRev",
            value: 0.0000070951660018181424,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / PostRev",
            value: 0.000006603666999581037,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / BothRev",
            value: 0.000007072540996887256,
            unit: "s",
          },
          {
            name: "sum    / Jax / cpu / BothRev",
            value: 0.000007606333001604071,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / PreRev",
            value: 0.000007158125001296867,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / PostRev",
            value: 0.0000068134159992041535,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / BothRev",
            value: 0.000006763958001101855,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / PreRev",
            value: 0.000006779250001272885,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / PostRev",
            value: 0.000006642666001425824,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / BothRev",
            value: 0.000006651291998423403,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / PreRev",
            value: 0.000006564125000295463,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / PostRev",
            value: 0.000006829375000961591,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / BothRev",
            value: 0.000006832167000538902,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / PreRev",
            value: 0.000006748540999979013,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / PostRev",
            value: 0.000006727415999193909,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / BothRev",
            value: 0.000006729124997946201,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / PreRev",
            value: 0.00000680570899930899,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / PostRev",
            value: 0.000006787040998460725,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / BothRev",
            value: 0.000006884750000608619,
            unit: "s",
          },
          {
            name: "value_and_grad / JaXPipe / cpu / Primal",
            value: 0.000009983082999951876,
            unit: "s",
          },
          {
            name: "value_and_grad / Jax / cpu / Primal",
            value: 0.000010642987999744948,
            unit: "s",
          },
          {
            name: "value_and_grad / HLOOpt / cpu / Primal",
            value: 0.000010609638000005362,
            unit: "s",
          },
          {
            name: "value_and_grad / PartOpt / cpu / Primal",
            value: 0.000010448998999891046,
            unit: "s",
          },
          {
            name: "value_and_grad / IPartOpt / cpu / Primal",
            value: 0.00001057271799982118,
            unit: "s",
          },
          {
            name: "value_and_grad / DefOpt / cpu / Primal",
            value: 0.00000997933999997258,
            unit: "s",
          },
          {
            name: "value_and_grad / IDefOpt / cpu / Primal",
            value: 0.000010687454999697366,
            unit: "s",
          },
          {
            name: "value_and_grad / JaXPipe / cpu / Primal",
            value: 0.00001694155699806288,
            unit: "s",
          },
          {
            name: "value_and_grad / Jax / cpu / Primal",
            value: 0.00001695944600214716,
            unit: "s",
          },
          {
            name: "value_and_grad / HLOOpt / cpu / Primal",
            value: 0.000018005131991230884,
            unit: "s",
          },
          {
            name: "value_and_grad / PartOpt / cpu / Primal",
            value: 0.00001808109099511057,
            unit: "s",
          },
          {
            name: "value_and_grad / IPartOpt / cpu / Primal",
            value: 0.000017792761995224282,
            unit: "s",
          },
          {
            name: "value_and_grad / DefOpt / cpu / Primal",
            value: 0.000017970667002373376,
            unit: "s",
          },
          {
            name: "value_and_grad / IDefOpt / cpu / Primal",
            value: 0.0000180402379919542,
            unit: "s",
          },
          {
            name: "value_and_grad / JaXPipe / tpu / Primal",
            value: 0.0002407898519995,
            unit: "s",
          },
          {
            name: "value_and_grad / Jax / tpu / Primal",
            value: 0.0002403278429992,
            unit: "s",
          },
          {
            name: "value_and_grad / HLOOpt / tpu / Primal",
            value: 0.0002304433379977,
            unit: "s",
          },
          {
            name: "value_and_grad / PartOpt / tpu / Primal",
            value: 0.0002310834790005,
            unit: "s",
          },
          {
            name: "value_and_grad / IPartOpt / tpu / Primal",
            value: 0.0002279426579989,
            unit: "s",
          },
          {
            name: "value_and_grad / DefOpt / tpu / Primal",
            value: 0.0002223373649976,
            unit: "s",
          },
          {
            name: "value_and_grad / IDefOpt / tpu / Primal",
            value: 0.0001921195720024,
            unit: "s",
          },
          {
            name: "value_and_grad / JaXPipe / cpu / Primal",
            value: 0.00001186177899944596,
            unit: "s",
          },
          {
            name: "value_and_grad / Jax / cpu / Primal",
            value: 0.000012015360999612311,
            unit: "s",
          },
          {
            name: "value_and_grad / HLOOpt / cpu / Primal",
            value: 0.00001270052299969393,
            unit: "s",
          },
          {
            name: "value_and_grad / PartOpt / cpu / Primal",
            value: 0.0000121769809993566,
            unit: "s",
          },
          {
            name: "value_and_grad / IPartOpt / cpu / Primal",
            value: 0.00001274458600073558,
            unit: "s",
          },
          {
            name: "value_and_grad / DefOpt / cpu / Primal",
            value: 0.000012573492000228723,
            unit: "s",
          },
          {
            name: "value_and_grad / IDefOpt / cpu / Primal",
            value: 0.000012620567999874766,
            unit: "s",
          },
          {
            name: "value_and_grad / JaXPipe / cpu / Primal",
            value: 0.000008025167000596412,
            unit: "s",
          },
          {
            name: "value_and_grad / Jax / cpu / Primal",
            value: 0.000008303833001264138,
            unit: "s",
          },
          {
            name: "value_and_grad / HLOOpt / cpu / Primal",
            value: 0.000008082666998234344,
            unit: "s",
          },
          {
            name: "value_and_grad / PartOpt / cpu / Primal",
            value: 0.000007956000001286157,
            unit: "s",
          },
          {
            name: "value_and_grad / IPartOpt / cpu / Primal",
            value: 0.00000832691699906718,
            unit: "s",
          },
          {
            name: "value_and_grad / DefOpt / cpu / Primal",
            value: 0.000008271667000371963,
            unit: "s",
          },
          {
            name: "value_and_grad / IDefOpt / cpu / Primal",
            value: 0.000008024041999306065,
            unit: "s",
          },
          {
            name: "jaxmd40 / JaXPipe / cpu / Primal",
            value: 0.0806179309991421,
            unit: "s",
          },
          {
            name: "jaxmd40 / Jax / cpu / Primal",
            value: 0.0766871607978828,
            unit: "s",
          },
          {
            name: "jaxmd40 / HLOOpt / cpu / Primal",
            value: 0.1132231095980387,
            unit: "s",
          },
          {
            name: "jaxmd40 / PartOpt / cpu / Primal",
            value: 0.0780502222012728,
            unit: "s",
          },
          {
            name: "jaxmd40 / IPartOpt / cpu / Primal",
            value: 0.0748033732001204,
            unit: "s",
          },
          {
            name: "jaxmd40 / DefOpt / cpu / Primal",
            value: 0.1103550407977309,
            unit: "s",
          },
          {
            name: "jaxmd40 / IDefOpt / cpu / Primal",
            value: 0.1114808343991171,
            unit: "s",
          },
          {
            name: "jaxmd40 / Jax / cpu / Forward",
            value: 0.1107811463996768,
            unit: "s",
          },
          {
            name: "jaxmd40 / JaXPipe / cpu / PostRev",
            value: 0.1635703864012612,
            unit: "s",
          },
          {
            name: "jaxmd40 / Jax / cpu / BothRev",
            value: 0.1556967979995533,
            unit: "s",
          },
          {
            name: "jaxmd40 / HLOOpt / cpu / PostRev",
            value: 0.2244815896003274,
            unit: "s",
          },
          {
            name: "jaxmd40 / PartOpt / cpu / PostRev",
            value: 0.1539730179996695,
            unit: "s",
          },
          {
            name: "jaxmd40 / IPartOpt / cpu / PostRev",
            value: 0.1501491820003138,
            unit: "s",
          },
          {
            name: "jaxmd40 / DefOpt / cpu / PostRev",
            value: 0.2212969713989878,
            unit: "s",
          },
          {
            name: "jaxmd40 / IDefOpt / cpu / PostRev",
            value: 0.2112092079973081,
            unit: "s",
          },
          {
            name: "jaxmd40 / JaXPipe / tpu / Primal",
            value: 0.0093092086004617,
            unit: "s",
          },
          {
            name: "jaxmd40 / Jax / tpu / Primal",
            value: 0.0093152763998659,
            unit: "s",
          },
          {
            name: "jaxmd40 / HLOOpt / tpu / Primal",
            value: 0.0092373205996409,
            unit: "s",
          },
          {
            name: "jaxmd40 / PartOpt / tpu / Primal",
            value: 0.0092942985997069,
            unit: "s",
          },
          {
            name: "jaxmd40 / IPartOpt / tpu / Primal",
            value: 0.0092912857995543,
            unit: "s",
          },
          {
            name: "jaxmd40 / DefOpt / tpu / Primal",
            value: 0.0091185337994829,
            unit: "s",
          },
          {
            name: "jaxmd40 / IDefOpt / tpu / Primal",
            value: 0.0090803457998845,
            unit: "s",
          },
          {
            name: "jaxmd40 / Jax / tpu / Forward",
            value: 0.0183110236001084,
            unit: "s",
          },
          {
            name: "jaxmd40 / JaXPipe / tpu / PostRev",
            value: 0.0197999643998628,
            unit: "s",
          },
          {
            name: "jaxmd40 / Jax / tpu / BothRev",
            value: 0.0198069562000455,
            unit: "s",
          },
          {
            name: "jaxmd40 / HLOOpt / tpu / PostRev",
            value: 0.018913959999918,
            unit: "s",
          },
          {
            name: "jaxmd40 / PartOpt / tpu / PostRev",
            value: 0.0197411641995131,
            unit: "s",
          },
          {
            name: "jaxmd40 / IPartOpt / tpu / PostRev",
            value: 0.0196720362000633,
            unit: "s",
          },
          {
            name: "jaxmd40 / DefOpt / tpu / PostRev",
            value: 0.0184451076005643,
            unit: "s",
          },
          {
            name: "jaxmd40 / IDefOpt / tpu / PostRev",
            value: 0.0181869015999836,
            unit: "s",
          },
          {
            name: "jaxmd40 / JaXPipe / cpu / Primal",
            value: 0.059003051000036,
            unit: "s",
          },
          {
            name: "jaxmd40 / Jax / cpu / Primal",
            value: 0.0593461091999415,
            unit: "s",
          },
          {
            name: "jaxmd40 / HLOOpt / cpu / Primal",
            value: 0.0848551908000445,
            unit: "s",
          },
          {
            name: "jaxmd40 / PartOpt / cpu / Primal",
            value: 0.0610421339999447,
            unit: "s",
          },
          {
            name: "jaxmd40 / IPartOpt / cpu / Primal",
            value: 0.059472647000075,
            unit: "s",
          },
          {
            name: "jaxmd40 / DefOpt / cpu / Primal",
            value: 0.0842572167999606,
            unit: "s",
          },
          {
            name: "jaxmd40 / IDefOpt / cpu / Primal",
            value: 0.0850750147999861,
            unit: "s",
          },
          {
            name: "jaxmd40 / Jax / cpu / Forward",
            value: 0.0863464832000318,
            unit: "s",
          },
          {
            name: "jaxmd40 / JaXPipe / cpu / PostRev",
            value: 0.125992987400059,
            unit: "s",
          },
          {
            name: "jaxmd40 / Jax / cpu / BothRev",
            value: 0.1272792843999923,
            unit: "s",
          },
          {
            name: "jaxmd40 / HLOOpt / cpu / PostRev",
            value: 0.1673792794001201,
            unit: "s",
          },
          {
            name: "jaxmd40 / PartOpt / cpu / PostRev",
            value: 0.1217966132000583,
            unit: "s",
          },
          {
            name: "jaxmd40 / IPartOpt / cpu / PostRev",
            value: 0.1192043343999103,
            unit: "s",
          },
          {
            name: "jaxmd40 / DefOpt / cpu / PostRev",
            value: 0.1654290332000528,
            unit: "s",
          },
          {
            name: "jaxmd40 / IDefOpt / cpu / PostRev",
            value: 0.1593192522001118,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / JaXPipe / cpu / Primal",
            value: 52.33070702200348,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / Jax / cpu / Primal",
            value: 52.30535430100281,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / HLOOpt / cpu / Primal",
            value: 51.275437669988605,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / PartOpt / cpu / Primal",
            value: 50.89441160800925,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / IPartOpt / cpu / Primal",
            value: 50.9381812060019,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / DefOpt / cpu / Primal",
            value: 25.917799237009604,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / IDefOpt / cpu / Primal",
            value: 56.77541277000273,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / JaXPipe / tpu / Primal",
            value: 0.1900036059996637,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / Jax / tpu / Primal",
            value: 0.1899443670008622,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / HLOOpt / tpu / Primal",
            value: 0.1886885959975188,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / PartOpt / tpu / Primal",
            value: 0.2028128109996032,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / IPartOpt / tpu / Primal",
            value: 0.2027104009976028,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / DefOpt / tpu / Primal",
            value: 0.1739326499991875,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / IDefOpt / tpu / Primal",
            value: 0.1850152240003808,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / JaXPipe / cpu / Primal",
            value: 53.67828183399979,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / Jax / cpu / Primal",
            value: 53.649693071999536,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / HLOOpt / cpu / Primal",
            value: 51.37882306499978,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / PartOpt / cpu / Primal",
            value: 53.25847989399972,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / IPartOpt / cpu / Primal",
            value: 52.80276468700049,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / DefOpt / cpu / Primal",
            value: 24.49554107699987,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / IDefOpt / cpu / Primal",
            value: 57.97307792999982,
            unit: "s",
          },
        ],
      },
      {
        commit: {
          author: {
            email: "wmoses@google.com",
            name: "William Moses",
            username: "wsmoses",
          },
          committer: {
            email: "noreply@github.com",
            name: "GitHub",
            username: "web-flow",
          },
          distinct: true,
          id: "0b08b097b75377737dd3319553b07f773d16d519",
          message:
            "MLIR AD: jaxmd (#1807)\n\n* MLIR AD: jaxmd\n\n* Update jaxmd.py",
          timestamp: "2025-12-19T11:53:17-06:00",
          tree_id: "a5c53e936a1573400ef0a0b3b42cac060e8fbb2e",
          url: "https://github.com/EnzymeAD/Enzyme-JAX/commit/0b08b097b75377737dd3319553b07f773d16d519",
        },
        date: 1766186797101,
        tool: "customSmallerIsBetter",
        benches: [
          {
            name: "actmtch / JaXPipe / cpu / Primal",
            value: 0.000004822807999971701,
            unit: "s",
          },
          {
            name: "actmtch / Jax / cpu / Primal",
            value: 0.000004799468999863166,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / Primal",
            value: 0.000005247943999620475,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / Primal",
            value: 0.000004853882000134036,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / Primal",
            value: 0.000004616810000243277,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / Primal",
            value: 0.000005355776000214973,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / Primal",
            value: 0.000005350002000341192,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / Forward",
            value: 0.000008415829000114173,
            unit: "s",
          },
          {
            name: "actmtch / Jax / cpu / Forward",
            value: 0.000007665058000384306,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / Forward",
            value: 0.000008249071999671287,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / Forward",
            value: 0.000008323192999796447,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / Forward",
            value: 0.000008652612000332739,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / Forward",
            value: 0.000008477009999751317,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / Forward",
            value: 0.000008391493000090123,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / PreRev",
            value: 0.0000087036910003917,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / PostRev",
            value: 0.000007207696000023134,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / BothRev",
            value: 0.000008853995999743347,
            unit: "s",
          },
          {
            name: "actmtch / Jax / cpu / BothRev",
            value: 0.000007529180999881646,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / PreRev",
            value: 0.00000862712400021337,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / PostRev",
            value: 0.000008272244999716349,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / BothRev",
            value: 0.000008401541000239376,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / PreRev",
            value: 0.000008422532000167848,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / PostRev",
            value: 0.000007461359999979322,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / BothRev",
            value: 0.000008740142000078777,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / PreRev",
            value: 0.000008701599999767496,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / PostRev",
            value: 0.000007510112999625562,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / BothRev",
            value: 0.000008452707000287773,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / PreRev",
            value: 0.00000846251300026779,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / PostRev",
            value: 0.000008646937999856163,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / BothRev",
            value: 0.00000832789900005082,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / PreRev",
            value: 0.000008453805000044668,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / PostRev",
            value: 0.000008439171000190982,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / BothRev",
            value: 0.000008355427000424243,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / Primal",
            value: 0.000008636780985398217,
            unit: "s",
          },
          {
            name: "actmtch / Jax / cpu / Primal",
            value: 0.000008833674975903705,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / Primal",
            value: 0.000010674442019080742,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / Primal",
            value: 0.000008953117008786648,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / Primal",
            value: 0.000009464286005822942,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / Primal",
            value: 0.000011299211997538805,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / Primal",
            value: 0.000010631905985064804,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / Forward",
            value: 0.000015427348989760505,
            unit: "s",
          },
          {
            name: "actmtch / Jax / cpu / Forward",
            value: 0.000013390018022619188,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / Forward",
            value: 0.000015381773991975933,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / Forward",
            value: 0.000016356418025679885,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / Forward",
            value: 0.000015732936008134855,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / Forward",
            value: 0.000016579087998252363,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / Forward",
            value: 0.000015588070004014298,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / PreRev",
            value: 0.00001577117800479755,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / PostRev",
            value: 0.000012683029985055328,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / BothRev",
            value: 0.000016684290021657943,
            unit: "s",
          },
          {
            name: "actmtch / Jax / cpu / BothRev",
            value: 0.000012964790017576888,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / PreRev",
            value: 0.000016576234978856517,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / PostRev",
            value: 0.000016423075983766468,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / BothRev",
            value: 0.00001579885699902661,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / PreRev",
            value: 0.00001573884001118131,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / PostRev",
            value: 0.00001273824498639442,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / BothRev",
            value: 0.000015629779984010382,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / PreRev",
            value: 0.00001656469999579713,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / PostRev",
            value: 0.000012716350000118836,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / BothRev",
            value: 0.000015734947024611755,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / PreRev",
            value: 0.000015876991004915907,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / PostRev",
            value: 0.000015761449001729487,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / BothRev",
            value: 0.000015671647997805848,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / PreRev",
            value: 0.000016669143980834632,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / PostRev",
            value: 0.00001645317900693044,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / BothRev",
            value: 0.000016462231025798246,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / tpu / Primal",
            value: 0.000151607276,
            unit: "s",
          },
          {
            name: "actmtch / Jax / tpu / Primal",
            value: 0.0001487330059999,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / tpu / Primal",
            value: 0.0001476639360002,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / tpu / Primal",
            value: 0.0001460753459996,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / tpu / Primal",
            value: 0.0001308790059997,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / tpu / Primal",
            value: 0.0001305256359996,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / tpu / Primal",
            value: 0.000130866556,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / tpu / Forward",
            value: 0.000224518384,
            unit: "s",
          },
          {
            name: "actmtch / Jax / tpu / Forward",
            value: 0.0002359524339999,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / tpu / Forward",
            value: 0.0002370963239995,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / tpu / Forward",
            value: 0.0002454737940001,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / tpu / Forward",
            value: 0.000229771234,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / tpu / Forward",
            value: 0.0002237287040002,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / tpu / Forward",
            value: 0.0002234433940002,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / tpu / PreRev",
            value: 0.0002201509239998,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / tpu / PostRev",
            value: 0.0002088840739997,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / tpu / BothRev",
            value: 0.0001955750239999,
            unit: "s",
          },
          {
            name: "actmtch / Jax / tpu / BothRev",
            value: 0.0002115786449999,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / tpu / PreRev",
            value: 0.0002208812849999,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / tpu / PostRev",
            value: 0.0002211992439997,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / tpu / BothRev",
            value: 0.0002142940140001,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / tpu / PreRev",
            value: 0.0002151028839998,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / tpu / PostRev",
            value: 0.0002155494640001,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / tpu / BothRev",
            value: 0.0002046993849999,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / tpu / PreRev",
            value: 0.0002051114640003,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / tpu / PostRev",
            value: 0.0001927809140001,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / tpu / BothRev",
            value: 0.0002043975249998,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / tpu / PreRev",
            value: 0.0002038219339997,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / tpu / PostRev",
            value: 0.0002021822939996,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / tpu / BothRev",
            value: 0.0001975010849996,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / tpu / PreRev",
            value: 0.0001986549440002,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / tpu / PostRev",
            value: 0.0002152440139998,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / tpu / BothRev",
            value: 0.000217796074,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / Primal",
            value: 0.0000064073190005728974,
            unit: "s",
          },
          {
            name: "actmtch / Jax / cpu / Primal",
            value: 0.000006585897000149999,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / Primal",
            value: 0.000007804189999660594,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / Primal",
            value: 0.000006960069000342628,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / Primal",
            value: 0.000006669064000561775,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / Primal",
            value: 0.000008143711999764492,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / Primal",
            value: 0.00000773328000013862,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / Forward",
            value: 0.000011198235999472672,
            unit: "s",
          },
          {
            name: "actmtch / Jax / cpu / Forward",
            value: 0.000009438450999368797,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / Forward",
            value: 0.000011323066999466393,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / Forward",
            value: 0.000011938833999920462,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / Forward",
            value: 0.000011843092000162867,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / Forward",
            value: 0.000011822653999843169,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / Forward",
            value: 0.00001152131199978612,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / PreRev",
            value: 0.000011263839999628544,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / PostRev",
            value: 0.00000945980699998472,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / BothRev",
            value: 0.000011158612000144785,
            unit: "s",
          },
          {
            name: "actmtch / Jax / cpu / BothRev",
            value: 0.000009997117999773765,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / PreRev",
            value: 0.000011954599000091549,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / PostRev",
            value: 0.000012006558000393851,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / BothRev",
            value: 0.000011825748999399366,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / PreRev",
            value: 0.000011288708999927622,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / PostRev",
            value: 0.00001003186699927028,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / BothRev",
            value: 0.000011370895999789354,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / PreRev",
            value: 0.000011975792000157526,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / PostRev",
            value: 0.000009811286000513064,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / BothRev",
            value: 0.000011969944000156827,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / PreRev",
            value: 0.000011853654999868014,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / PostRev",
            value: 0.00001127871900007449,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / BothRev",
            value: 0.000011179451000316476,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / PreRev",
            value: 0.000011225507000744984,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / PostRev",
            value: 0.000012038225000651436,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / BothRev",
            value: 0.000011702700999194348,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / Primal",
            value: 0.0000036764579999726256,
            unit: "s",
          },
          {
            name: "actmtch / Jax / cpu / Primal",
            value: 0.000004074833002960077,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / Primal",
            value: 0.0000044351250035106205,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / Primal",
            value: 0.000003606249996664701,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / Primal",
            value: 0.000003586708000511862,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / Primal",
            value: 0.0000041662090006866494,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / Primal",
            value: 0.000004216375000396511,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / Forward",
            value: 0.000006492417000117712,
            unit: "s",
          },
          {
            name: "actmtch / Jax / cpu / Forward",
            value: 0.000005539792000490706,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / Forward",
            value: 0.000006808707999880426,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / Forward",
            value: 0.000006461042001319584,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / Forward",
            value: 0.000006575665996933822,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / Forward",
            value: 0.0000063610830002289735,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / Forward",
            value: 0.000006497624999610707,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / PreRev",
            value: 0.000006366334000631468,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / PostRev",
            value: 0.000005492167001648341,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / BothRev",
            value: 0.000006391457998688566,
            unit: "s",
          },
          {
            name: "actmtch / Jax / cpu / BothRev",
            value: 0.000005317250001098728,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / PreRev",
            value: 0.000006138041000667726,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / PostRev",
            value: 0.000006126666998170549,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / BothRev",
            value: 0.000006143041999166599,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / PreRev",
            value: 0.000006147707998024999,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / PostRev",
            value: 0.000005303374997311039,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / BothRev",
            value: 0.000006311041001026751,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / PreRev",
            value: 0.00000622049999947194,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / PostRev",
            value: 0.000005365999997593463,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / BothRev",
            value: 0.000006177916999149602,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / PreRev",
            value: 0.000006328499999654014,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / PostRev",
            value: 0.000006219041999429464,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / BothRev",
            value: 0.000006223582997336053,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / PreRev",
            value: 0.0000063708339985169,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / PostRev",
            value: 0.000005995583000185434,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / BothRev",
            value: 0.000006165124999824911,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / Primal",
            value: 0.0000049518230002831845,
            unit: "s",
          },
          {
            name: "add_one / Jax / cpu / Primal",
            value: 0.00000497831699976814,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / Primal",
            value: 0.000004959495999628416,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / Primal",
            value: 0.000004933366999921418,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / Primal",
            value: 0.000004907325000203855,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / Primal",
            value: 0.000004878940999788029,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / Primal",
            value: 0.00000494258099979561,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / Forward",
            value: 0.000008287231999929646,
            unit: "s",
          },
          {
            name: "add_one / Jax / cpu / Forward",
            value: 0.000008280864999960613,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / Forward",
            value: 0.000008309320000080334,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / Forward",
            value: 0.000008356104000085907,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / Forward",
            value: 0.000008374627000193869,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / Forward",
            value: 0.00000824354799988214,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / Forward",
            value: 0.000008403137000186688,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / PreRev",
            value: 0.000009243298000001231,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / PostRev",
            value: 0.000009374062000006234,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / BothRev",
            value: 0.000009275010000237672,
            unit: "s",
          },
          {
            name: "add_one / Jax / cpu / BothRev",
            value: 0.000009139320000031148,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / PreRev",
            value: 0.000009302615999786212,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / PostRev",
            value: 0.000009344640000108484,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / BothRev",
            value: 0.00000939337499994508,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / PreRev",
            value: 0.000009212006999860024,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / PostRev",
            value: 0.000009323529000084818,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / BothRev",
            value: 0.000009180334999655317,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / PreRev",
            value: 0.000009366694000163989,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / PostRev",
            value: 0.00000930276999997659,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / BothRev",
            value: 0.00000934088200028782,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / PreRev",
            value: 0.000009406419999777429,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / PostRev",
            value: 0.00000934691100019336,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / BothRev",
            value: 0.000009321713999725034,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / PreRev",
            value: 0.000009283979999963776,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / PostRev",
            value: 0.000009683612999651814,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / BothRev",
            value: 0.00000935400300022593,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / Primal",
            value: 0.000009043586003826931,
            unit: "s",
          },
          {
            name: "add_one / Jax / cpu / Primal",
            value: 0.000009083717013709248,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / Primal",
            value: 0.000009603362006600946,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / Primal",
            value: 0.000009458378975978122,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / Primal",
            value: 0.000008995406009489671,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / Primal",
            value: 0.00000908147098380141,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / Primal",
            value: 0.000009090453997487202,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / Forward",
            value: 0.000013835746998665854,
            unit: "s",
          },
          {
            name: "add_one / Jax / cpu / Forward",
            value: 0.00001453491600113921,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / Forward",
            value: 0.000013830892014084384,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / Forward",
            value: 0.000014541849988745526,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / Forward",
            value: 0.000013956211012555289,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / Forward",
            value: 0.000014510849985526875,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / Forward",
            value: 0.000014519364020088688,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / PreRev",
            value: 0.000014792996022151785,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / PostRev",
            value: 0.0000148493500018958,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / BothRev",
            value: 0.000015566534013487398,
            unit: "s",
          },
          {
            name: "add_one / Jax / cpu / BothRev",
            value: 0.000015664175996789707,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / PreRev",
            value: 0.000015576663980027662,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / PostRev",
            value: 0.00001563724799780175,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / BothRev",
            value: 0.0000157261970161926,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / PreRev",
            value: 0.000014871185005176813,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / PostRev",
            value: 0.0000157551979937125,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / BothRev",
            value: 0.00001470200001494959,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / PreRev",
            value: 0.000015504548005992546,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / PostRev",
            value: 0.00001560603501275182,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / BothRev",
            value: 0.00001566694601206109,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / PreRev",
            value: 0.0000148085139808245,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / PostRev",
            value: 0.000015584701992338525,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / BothRev",
            value: 0.000015627808024873958,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / PreRev",
            value: 0.000014664387999800964,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / PostRev",
            value: 0.000015585847984766587,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / BothRev",
            value: 0.00001559793300111778,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / tpu / Primal",
            value: 0.0001515569859998,
            unit: "s",
          },
          {
            name: "add_one / Jax / tpu / Primal",
            value: 0.0001518128959996,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / tpu / Primal",
            value: 0.0001504234560002,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / tpu / Primal",
            value: 0.0001511031059999,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / tpu / Primal",
            value: 0.0001507431249997,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / tpu / Primal",
            value: 0.0001296658559999,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / tpu / Primal",
            value: 0.0001268048170004,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / tpu / Forward",
            value: 0.000174387775,
            unit: "s",
          },
          {
            name: "add_one / Jax / tpu / Forward",
            value: 0.0002141980639999,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / tpu / Forward",
            value: 0.0002215158640001,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / tpu / Forward",
            value: 0.0002189744239999,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / tpu / Forward",
            value: 0.000207928274,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / tpu / Forward",
            value: 0.0002074161139998,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / tpu / Forward",
            value: 0.0002264230339997,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / tpu / PreRev",
            value: 0.0002367142749999,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / tpu / PostRev",
            value: 0.0002351361850001,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / tpu / BothRev",
            value: 0.0002423519550002,
            unit: "s",
          },
          {
            name: "add_one / Jax / tpu / BothRev",
            value: 0.0002291923749999,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / tpu / PreRev",
            value: 0.0002346727249996,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / tpu / PostRev",
            value: 0.0002435820550003,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / tpu / BothRev",
            value: 0.0002174353750001,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / tpu / PreRev",
            value: 0.0002129901359999,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / tpu / PostRev",
            value: 0.0002213693850003,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / tpu / BothRev",
            value: 0.0002217043259997,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / tpu / PreRev",
            value: 0.0002009241159998,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / tpu / PostRev",
            value: 0.000207601816,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / tpu / BothRev",
            value: 0.0002378407549999,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / tpu / PreRev",
            value: 0.0002274490959998,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / tpu / PostRev",
            value: 0.0002002709160001,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / tpu / BothRev",
            value: 0.0002274600949999,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / tpu / PreRev",
            value: 0.0002048976249998,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / tpu / PostRev",
            value: 0.0001994813249998,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / tpu / BothRev",
            value: 0.0002038818360001,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / Primal",
            value: 0.000006711640000503394,
            unit: "s",
          },
          {
            name: "add_one / Jax / cpu / Primal",
            value: 0.000006741547999808972,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / Primal",
            value: 0.0000067296500001248205,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / Primal",
            value: 0.000007127666000087629,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / Primal",
            value: 0.000006752911999683419,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / Primal",
            value: 0.000006606476000342809,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / Primal",
            value: 0.0000065660449999995766,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / Forward",
            value: 0.000010252654000396434,
            unit: "s",
          },
          {
            name: "add_one / Jax / cpu / Forward",
            value: 0.00001060361700001522,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / Forward",
            value: 0.00001025274600033299,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / Forward",
            value: 0.000010184757000388344,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / Forward",
            value: 0.000010738922999735224,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / Forward",
            value: 0.000010579030999906536,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / Forward",
            value: 0.000010151028000109364,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / PreRev",
            value: 0.00001083277799989446,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / PostRev",
            value: 0.00001149407200045971,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / BothRev",
            value: 0.00001153845699991507,
            unit: "s",
          },
          {
            name: "add_one / Jax / cpu / BothRev",
            value: 0.00001151858000048378,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / PreRev",
            value: 0.000011602903000493824,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / PostRev",
            value: 0.00001145654599986301,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / BothRev",
            value: 0.000011474418000034348,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / PreRev",
            value: 0.000011450377999608464,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / PostRev",
            value: 0.00001151717899938376,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / BothRev",
            value: 0.000010797484000249825,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / PreRev",
            value: 0.000011419971000577788,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / PostRev",
            value: 0.000011442312000326638,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / BothRev",
            value: 0.000010983643000145094,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / PreRev",
            value: 0.000011458029000095847,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / PostRev",
            value: 0.000011481508999168,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / BothRev",
            value: 0.000010837573000571865,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / PreRev",
            value: 0.000011464387999694736,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / PostRev",
            value: 0.000011440004000178304,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / BothRev",
            value: 0.000011478401999738708,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / Primal",
            value: 0.000003537790998962009,
            unit: "s",
          },
          {
            name: "add_one / Jax / cpu / Primal",
            value: 0.0000034937079981318675,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / Primal",
            value: 0.000003662291997898137,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / Primal",
            value: 0.000003513583000312792,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / Primal",
            value: 0.000003572500001610024,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / Primal",
            value: 0.0000035211249996791597,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / Primal",
            value: 0.000003478124999674037,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / Forward",
            value: 0.000005439499997009989,
            unit: "s",
          },
          {
            name: "add_one / Jax / cpu / Forward",
            value: 0.000005458125000586733,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / Forward",
            value: 0.000005534583000553539,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / Forward",
            value: 0.000005722874997445615,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / Forward",
            value: 0.000005654707998473896,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / Forward",
            value: 0.00000556825000239769,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / Forward",
            value: 0.000005545250001887325,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / PreRev",
            value: 0.000006315250000625383,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / PostRev",
            value: 0.000006437916999857407,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / BothRev",
            value: 0.000006708749999233987,
            unit: "s",
          },
          {
            name: "add_one / Jax / cpu / BothRev",
            value: 0.000006580166998901404,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / PreRev",
            value: 0.000006563874998391838,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / PostRev",
            value: 0.000006770084000891075,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / BothRev",
            value: 0.000006631916996411746,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / PreRev",
            value: 0.000006511708001198713,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / PostRev",
            value: 0.000006732917001500027,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / BothRev",
            value: 0.000006696000000374625,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / PreRev",
            value: 0.00000653641700046137,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / PostRev",
            value: 0.000006905875001393724,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / BothRev",
            value: 0.000006643208002060419,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / PreRev",
            value: 0.000006602416000532685,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / PostRev",
            value: 0.000006653625001490582,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / BothRev",
            value: 0.000006514625001727837,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / PreRev",
            value: 0.0000065848339982039765,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / PostRev",
            value: 0.000006563499999174383,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / BothRev",
            value: 0.000006580542001756839,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / Primal",
            value: 0.000005199794999953156,
            unit: "s",
          },
          {
            name: "add_two / Jax / cpu / Primal",
            value: 0.000005163963000086369,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / Primal",
            value: 0.000005164947999674041,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / Primal",
            value: 0.000005160598000202299,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / Primal",
            value: 0.000005097353000110161,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / Primal",
            value: 0.0000050632840002435845,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / Primal",
            value: 0.000005215352000050189,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / Forward",
            value: 0.00000855572000000393,
            unit: "s",
          },
          {
            name: "add_two / Jax / cpu / Forward",
            value: 0.00000855725999963397,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / Forward",
            value: 0.00000854540199998155,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / Forward",
            value: 0.000008629699000266555,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / Forward",
            value: 0.000008450425999853906,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / Forward",
            value: 0.000008471468999687205,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / Forward",
            value: 0.000008502880999913031,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / PreRev",
            value: 0.00001211815699980434,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / PostRev",
            value: 0.000011475096000140184,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / BothRev",
            value: 0.000011324015000354847,
            unit: "s",
          },
          {
            name: "add_two / Jax / cpu / BothRev",
            value: 0.000011598078000133682,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / PreRev",
            value: 0.000011504410999805258,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / PostRev",
            value: 0.00001156369500040455,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / BothRev",
            value: 0.00001144014800001969,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / PreRev",
            value: 0.000011474357999759376,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / PostRev",
            value: 0.00001139312199984488,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / BothRev",
            value: 0.000011132385000109934,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / PreRev",
            value: 0.000011190516000169736,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / PostRev",
            value: 0.000011151895000239164,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / BothRev",
            value: 0.00001136363599971446,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / PreRev",
            value: 0.000011410985000111395,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / PostRev",
            value: 0.00001138688800028831,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / BothRev",
            value: 0.000011337236000144913,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / PreRev",
            value: 0.00001117010100006155,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / PostRev",
            value: 0.000011795920999702502,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / BothRev",
            value: 0.000011130798000067444,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / Primal",
            value: 0.000009319325006799771,
            unit: "s",
          },
          {
            name: "add_two / Jax / cpu / Primal",
            value: 0.000009308870008680967,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / Primal",
            value: 0.000009881612990284338,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / Primal",
            value: 0.000009418751986231656,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / Primal",
            value: 0.000009763504000147804,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / Primal",
            value: 0.000009840593993430957,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / Primal",
            value: 0.000009033689013449477,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / Forward",
            value: 0.000014125431975116951,
            unit: "s",
          },
          {
            name: "add_two / Jax / cpu / Forward",
            value: 0.000014130514988210053,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / Forward",
            value: 0.000014866683981381356,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / Forward",
            value: 0.000014786957995966076,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / Forward",
            value: 0.00001477340701967478,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / Forward",
            value: 0.00001405204099137336,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / Forward",
            value: 0.000014261761010857298,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / PreRev",
            value: 0.000018613641994306816,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / PostRev",
            value: 0.00001838970498647541,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / BothRev",
            value: 0.000018163408007239924,
            unit: "s",
          },
          {
            name: "add_two / Jax / cpu / BothRev",
            value: 0.000018349891004618255,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / PreRev",
            value: 0.0000182520919770468,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / PostRev",
            value: 0.000018339167989324777,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / BothRev",
            value: 0.000018158241000492127,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / PreRev",
            value: 0.000018247685016831385,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / PostRev",
            value: 0.00001817140899947845,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / BothRev",
            value: 0.000017350876005366446,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / PreRev",
            value: 0.000018297133996384217,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / PostRev",
            value: 0.000018242677993839608,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / BothRev",
            value: 0.000017395240982295945,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / PreRev",
            value: 0.000018267990002641453,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / PostRev",
            value: 0.00001763556498917751,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / BothRev",
            value: 0.000017414997011655943,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / PreRev",
            value: 0.00001817377799306996,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / PostRev",
            value: 0.000018046790995867923,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / BothRev",
            value: 0.000017347685992717744,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / tpu / Primal",
            value: 0.0001419395470002,
            unit: "s",
          },
          {
            name: "add_two / Jax / tpu / Primal",
            value: 0.0001293006669998,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / tpu / Primal",
            value: 0.0001444419170002,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / tpu / Primal",
            value: 0.0001433632170001,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / tpu / Primal",
            value: 0.000128621837,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / tpu / Primal",
            value: 0.0001293717080002,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / tpu / Primal",
            value: 0.000126957597,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / tpu / Forward",
            value: 0.0002200479349999,
            unit: "s",
          },
          {
            name: "add_two / Jax / tpu / Forward",
            value: 0.0001800894770003,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / tpu / Forward",
            value: 0.000180754766,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / tpu / Forward",
            value: 0.0001775983259999,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / tpu / Forward",
            value: 0.0001817197159998,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / tpu / Forward",
            value: 0.0002185521850001,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / tpu / Forward",
            value: 0.0002182405049998,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / tpu / PreRev",
            value: 0.0002320308749999,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / tpu / PostRev",
            value: 0.000241592455,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / tpu / BothRev",
            value: 0.0002433088550001,
            unit: "s",
          },
          {
            name: "add_two / Jax / tpu / BothRev",
            value: 0.0002273188250001,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / tpu / PreRev",
            value: 0.0002433170550002,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / tpu / PostRev",
            value: 0.000243163175,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / tpu / BothRev",
            value: 0.0002434030539998,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / tpu / PreRev",
            value: 0.0002197677459998,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / tpu / PostRev",
            value: 0.0002207118649998,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / tpu / BothRev",
            value: 0.0002352505449998,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / tpu / PreRev",
            value: 0.0002344279650001,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / tpu / PostRev",
            value: 0.000233383056,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / tpu / BothRev",
            value: 0.0002567456739998,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / tpu / PreRev",
            value: 0.0002510668349996,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / tpu / PostRev",
            value: 0.0002327585349999,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / tpu / BothRev",
            value: 0.0002262662450002,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / tpu / PreRev",
            value: 0.0002312944159998,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / tpu / PostRev",
            value: 0.0002257415660001,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / tpu / BothRev",
            value: 0.0002497013950001,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / Primal",
            value: 0.000006944839999960095,
            unit: "s",
          },
          {
            name: "add_two / Jax / cpu / Primal",
            value: 0.000006966326999645389,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / Primal",
            value: 0.000007348207999712031,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / Primal",
            value: 0.000006917329999851063,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / Primal",
            value: 0.000007380382999144785,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / Primal",
            value: 0.000007334507999985362,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / Primal",
            value: 0.000006936230000064824,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / Forward",
            value: 0.000010335068000131286,
            unit: "s",
          },
          {
            name: "add_two / Jax / cpu / Forward",
            value: 0.000010940119000224514,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / Forward",
            value: 0.000010363184999732766,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / Forward",
            value: 0.00001091551600075036,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / Forward",
            value: 0.000010383542999989005,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / Forward",
            value: 0.000010819039000125484,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / Forward",
            value: 0.00001037821600039024,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / PreRev",
            value: 0.000013714660999539774,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / PostRev",
            value: 0.000013453728000058618,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / BothRev",
            value: 0.000013490689000718703,
            unit: "s",
          },
          {
            name: "add_two / Jax / cpu / BothRev",
            value: 0.000013480054999490676,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / PreRev",
            value: 0.000012913073000163424,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / PostRev",
            value: 0.00001345759400010138,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / BothRev",
            value: 0.000013516675999198924,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / PreRev",
            value: 0.000013575848000073164,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / PostRev",
            value: 0.000012875993000307062,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / BothRev",
            value: 0.000012816268999813471,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / PreRev",
            value: 0.000013711013999454736,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / PostRev",
            value: 0.000013494282999999996,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / BothRev",
            value: 0.00001344675800010009,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / PreRev",
            value: 0.00001354855800036603,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / PostRev",
            value: 0.000013440668999464833,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / BothRev",
            value: 0.000012952117999702753,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / PreRev",
            value: 0.0000138773909993688,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / PostRev",
            value: 0.000013480607999554194,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / BothRev",
            value: 0.000013554989000112982,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / Primal",
            value: 0.000003885750000335975,
            unit: "s",
          },
          {
            name: "add_two / Jax / cpu / Primal",
            value: 0.000003963666000345256,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / Primal",
            value: 0.000003942916002415586,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / Primal",
            value: 0.000003857457999401958,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / Primal",
            value: 0.000003785500000958564,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / Primal",
            value: 0.000003845375002129003,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / Primal",
            value: 0.00000379833300030441,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / Forward",
            value: 0.000006133541999588487,
            unit: "s",
          },
          {
            name: "add_two / Jax / cpu / Forward",
            value: 0.000006200209001690382,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / Forward",
            value: 0.000006161290999443736,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / Forward",
            value: 0.000006110500002250773,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / Forward",
            value: 0.000006518707999930484,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / Forward",
            value: 0.000006144666000182042,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / Forward",
            value: 0.000006098833000578452,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / PreRev",
            value: 0.000008122541999910027,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / PostRev",
            value: 0.000008290000001579756,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / BothRev",
            value: 0.00000810283299870207,
            unit: "s",
          },
          {
            name: "add_two / Jax / cpu / BothRev",
            value: 0.000008255624998128041,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / PreRev",
            value: 0.000008073124998190906,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / PostRev",
            value: 0.000008164792001480237,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / BothRev",
            value: 0.000008099208000203361,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / PreRev",
            value: 0.00000815850000071805,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / PostRev",
            value: 0.000008144958002958446,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / BothRev",
            value: 0.000007999083001777762,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / PreRev",
            value: 0.000008031375000427942,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / PostRev",
            value: 0.000008299417000671382,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / BothRev",
            value: 0.000008224666999012697,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / PreRev",
            value: 0.00000824970800022129,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / PostRev",
            value: 0.000008201541000744328,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / BothRev",
            value: 0.000008168167001713301,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / PreRev",
            value: 0.000008095083001535386,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / PostRev",
            value: 0.000008085249999567168,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / BothRev",
            value: 0.000008101082999928622,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / Primal",
            value: 0.000004659711999920546,
            unit: "s",
          },
          {
            name: "cache / Jax / cpu / Primal",
            value: 0.000005013153000163584,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / Primal",
            value: 0.000004664178999973956,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / Primal",
            value: 0.000005004382000151964,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / Primal",
            value: 0.000004681952999817441,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / Primal",
            value: 0.0000046734249999644814,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / Primal",
            value: 0.000004581568000048719,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / Forward",
            value: 0.000010486759999821516,
            unit: "s",
          },
          {
            name: "cache / Jax / cpu / Forward",
            value: 0.000010107479999987845,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / Forward",
            value: 0.000010151298999971914,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / Forward",
            value: 0.000010448743999859288,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / Forward",
            value: 0.00001043041599996286,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / Forward",
            value: 0.000010513654000078532,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / Forward",
            value: 0.000010651624999809428,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / PreRev",
            value: 0.000012152349000189134,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / PostRev",
            value: 0.00001581782499988549,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / BothRev",
            value: 0.000012009464999664489,
            unit: "s",
          },
          {
            name: "cache / Jax / cpu / BothRev",
            value: 0.000015526871000020037,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / PreRev",
            value: 0.000011617828999987976,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / PostRev",
            value: 0.00001131967800029088,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / BothRev",
            value: 0.000011503074000302147,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / PreRev",
            value: 0.000011441380000178469,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / PostRev",
            value: 0.000014747843999884936,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / BothRev",
            value: 0.00001282986600017466,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / PreRev",
            value: 0.000012203955999666505,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / PostRev",
            value: 0.000015080705999935162,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / BothRev",
            value: 0.00001191963500014026,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / PreRev",
            value: 0.000011850171999867596,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / PostRev",
            value: 0.00001217153600009624,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / BothRev",
            value: 0.000012187158999950042,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / PreRev",
            value: 0.000011701415000061388,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / PostRev",
            value: 0.000011661316999834525,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / BothRev",
            value: 0.000012092997999843648,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / Primal",
            value: 0.00000992131300154142,
            unit: "s",
          },
          {
            name: "cache / Jax / cpu / Primal",
            value: 0.00000978161400416866,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / Primal",
            value: 0.000009477905987296252,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / Primal",
            value: 0.000009887655993225053,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / Primal",
            value: 0.000009135272004641592,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / Primal",
            value: 0.000009179200016660616,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / Primal",
            value: 0.000009323509992100298,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / Forward",
            value: 0.000014724782027769834,
            unit: "s",
          },
          {
            name: "cache / Jax / cpu / Forward",
            value: 0.000014813165995292365,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / Forward",
            value: 0.000014281565003329888,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / Forward",
            value: 0.000014586480014258995,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / Forward",
            value: 0.000014342631009640172,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / Forward",
            value: 0.000014684269990539178,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / Forward",
            value: 0.000014511371002299712,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / PreRev",
            value: 0.00001552253100089729,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / PostRev",
            value: 0.000017550282995216548,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / BothRev",
            value: 0.000015119623014470565,
            unit: "s",
          },
          {
            name: "cache / Jax / cpu / BothRev",
            value: 0.000017286875983700155,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / PreRev",
            value: 0.00001526429198565893,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / PostRev",
            value: 0.000015439577022334562,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / BothRev",
            value: 0.000015211063000606372,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / PreRev",
            value: 0.000015058172983117402,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / PostRev",
            value: 0.000017246161005459725,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / BothRev",
            value: 0.000015249766001943498,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / PreRev",
            value: 0.000015521292021730916,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / PostRev",
            value: 0.000017133455985458566,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / BothRev",
            value: 0.000014935033017536623,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / PreRev",
            value: 0.000015397093986393885,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / PostRev",
            value: 0.000015128800005186349,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / BothRev",
            value: 0.000014565391989890488,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / PreRev",
            value: 0.000015406441991217434,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / PostRev",
            value: 0.00001543640700401738,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / BothRev",
            value: 0.00001511952799046412,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / tpu / Primal",
            value: 0.0001622854069996,
            unit: "s",
          },
          {
            name: "cache / Jax / tpu / Primal",
            value: 0.0001612239669998,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / tpu / Primal",
            value: 0.0001489730769999,
            unit: "s",
          },
          {
            name: "cache / PartOpt / tpu / Primal",
            value: 0.0001507000169999,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / tpu / Primal",
            value: 0.0001392812269996,
            unit: "s",
          },
          {
            name: "cache / DefOpt / tpu / Primal",
            value: 0.000148532487,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / tpu / Primal",
            value: 0.0001488731270001,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / tpu / Forward",
            value: 0.0002254548550004,
            unit: "s",
          },
          {
            name: "cache / Jax / tpu / Forward",
            value: 0.0002258990649997,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / tpu / Forward",
            value: 0.0001936766160001,
            unit: "s",
          },
          {
            name: "cache / PartOpt / tpu / Forward",
            value: 0.0002042580859997,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / tpu / Forward",
            value: 0.0001994283560002,
            unit: "s",
          },
          {
            name: "cache / DefOpt / tpu / Forward",
            value: 0.0002051973059997,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / tpu / Forward",
            value: 0.0002110479760003,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / tpu / PreRev",
            value: 0.0002083020060003,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / tpu / PostRev",
            value: 0.0002070645059998,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / tpu / BothRev",
            value: 0.0001950880459999,
            unit: "s",
          },
          {
            name: "cache / Jax / tpu / BothRev",
            value: 0.000203942866,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / tpu / PreRev",
            value: 0.000217235475,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / tpu / PostRev",
            value: 0.0002162738349998,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / tpu / BothRev",
            value: 0.0002183275249999,
            unit: "s",
          },
          {
            name: "cache / PartOpt / tpu / PreRev",
            value: 0.0002222960149997,
            unit: "s",
          },
          {
            name: "cache / PartOpt / tpu / PostRev",
            value: 0.0002189014650002,
            unit: "s",
          },
          {
            name: "cache / PartOpt / tpu / BothRev",
            value: 0.0002159283350001,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / tpu / PreRev",
            value: 0.000216597996,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / tpu / PostRev",
            value: 0.0001972590159998,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / tpu / BothRev",
            value: 0.0002146110459998,
            unit: "s",
          },
          {
            name: "cache / DefOpt / tpu / PreRev",
            value: 0.0002140528959998,
            unit: "s",
          },
          {
            name: "cache / DefOpt / tpu / PostRev",
            value: 0.0002025213659999,
            unit: "s",
          },
          {
            name: "cache / DefOpt / tpu / BothRev",
            value: 0.000206228946,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / tpu / PreRev",
            value: 0.0001977527759995,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / tpu / PostRev",
            value: 0.0002052908949999,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / tpu / BothRev",
            value: 0.0002064234260001,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / Primal",
            value: 0.000006473090999861597,
            unit: "s",
          },
          {
            name: "cache / Jax / cpu / Primal",
            value: 0.00000633906199982448,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / Primal",
            value: 0.000006370379000145476,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / Primal",
            value: 0.000006737858000633424,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / Primal",
            value: 0.000006796386999667447,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / Primal",
            value: 0.000006436896000195702,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / Primal",
            value: 0.000006861517999823264,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / Forward",
            value: 0.0000109899500002939,
            unit: "s",
          },
          {
            name: "cache / Jax / cpu / Forward",
            value: 0.00001141411800017522,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / Forward",
            value: 0.00001099665299989283,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / Forward",
            value: 0.000010288409999702709,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / Forward",
            value: 0.00001095598899973993,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / Forward",
            value: 0.000010803459999806365,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / Forward",
            value: 0.00001123287199970946,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / PreRev",
            value: 0.000011522814999807453,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / PostRev",
            value: 0.00001234070000009524,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / BothRev",
            value: 0.00001138697099941055,
            unit: "s",
          },
          {
            name: "cache / Jax / cpu / BothRev",
            value: 0.00001250048899964895,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / PreRev",
            value: 0.000010932417999356404,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / PostRev",
            value: 0.00001112278500022512,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / BothRev",
            value: 0.000011980539000433054,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / PreRev",
            value: 0.000011542080000253918,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / PostRev",
            value: 0.00001266777499949967,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / BothRev",
            value: 0.000010905947000537709,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / PreRev",
            value: 0.000011671979000311697,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / PostRev",
            value: 0.000012784563999957754,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / BothRev",
            value: 0.000011508093999509584,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / PreRev",
            value: 0.00001127871300013794,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / PostRev",
            value: 0.00001124935000007099,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / BothRev",
            value: 0.000011954395999964615,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / PreRev",
            value: 0.000011511849999806144,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / PostRev",
            value: 0.000011642024999673594,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / BothRev",
            value: 0.000011784602999796337,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / Primal",
            value: 0.000003314749999844935,
            unit: "s",
          },
          {
            name: "cache / Jax / cpu / Primal",
            value: 0.000003408500000659842,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / Primal",
            value: 0.0000033971670018217997,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / Primal",
            value: 0.000003347916001075646,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / Primal",
            value: 0.000003203208001650637,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / Primal",
            value: 0.000003416957999434089,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / Primal",
            value: 0.000003412207999645034,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / Forward",
            value: 0.000006618624996917788,
            unit: "s",
          },
          {
            name: "cache / Jax / cpu / Forward",
            value: 0.000006033207999280421,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / Forward",
            value: 0.000005989249999402091,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / Forward",
            value: 0.00000609729099960532,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / Forward",
            value: 0.000006143874998087995,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / Forward",
            value: 0.000006034750000253553,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / Forward",
            value: 0.000006643958997301525,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / PreRev",
            value: 0.000006247957997402409,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / PostRev",
            value: 0.000006883250000100815,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / BothRev",
            value: 0.000006182458000694169,
            unit: "s",
          },
          {
            name: "cache / Jax / cpu / BothRev",
            value: 0.00000652629199976218,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / PreRev",
            value: 0.000005848875000083353,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / PostRev",
            value: 0.000006005375002132496,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / BothRev",
            value: 0.000006305458999122493,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / PreRev",
            value: 0.000006320874999801162,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / PostRev",
            value: 0.000006803125001169974,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / BothRev",
            value: 0.0000059971250011585655,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / PreRev",
            value: 0.000005973957999231061,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / PostRev",
            value: 0.000006701000002067303,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / BothRev",
            value: 0.000006027375002304325,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / PreRev",
            value: 0.000006243959000130417,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / PostRev",
            value: 0.00000612566600102582,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / BothRev",
            value: 0.000006062583001039456,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / PreRev",
            value: 0.000006120458998339018,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / PostRev",
            value: 0.000006150832999992417,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / BothRev",
            value: 0.0000061083750006218905,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / Primal",
            value: 0.000004942300000038813,
            unit: "s",
          },
          {
            name: "Concat / Jax / cpu / Primal",
            value: 0.000005018556999857537,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / Primal",
            value: 0.0000049933550003515845,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / Primal",
            value: 0.000005013409999719442,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / Primal",
            value: 0.00000493458599976293,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / Primal",
            value: 0.000005012308999994275,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / Primal",
            value: 0.0000050138619999415824,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / Forward",
            value: 0.000008271925000371993,
            unit: "s",
          },
          {
            name: "Concat / Jax / cpu / Forward",
            value: 0.00000809035899965238,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / Forward",
            value: 0.000008437088999926346,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / Forward",
            value: 0.000008412635999775375,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / Forward",
            value: 0.000008267796999916754,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / Forward",
            value: 0.000008513078000305541,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / Forward",
            value: 0.000008505069999955594,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / PreRev",
            value: 0.000008988147999843932,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / PostRev",
            value: 0.000009117637999679574,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / BothRev",
            value: 0.00000918600399973002,
            unit: "s",
          },
          {
            name: "Concat / Jax / cpu / BothRev",
            value: 0.000009233041999777924,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / PreRev",
            value: 0.000009260654000172509,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / PostRev",
            value: 0.000009342473999822689,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / BothRev",
            value: 0.000009240009000222926,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / PreRev",
            value: 0.000009149543000148695,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / PostRev",
            value: 0.000009330415999556865,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / BothRev",
            value: 0.000009468891999858898,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / PreRev",
            value: 0.000008791004999693541,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / PostRev",
            value: 0.000009192092999910528,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / BothRev",
            value: 0.000009254684000097767,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / PreRev",
            value: 0.000009125359999870853,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / PostRev",
            value: 0.000009277525000015884,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / BothRev",
            value: 0.000009254706999854534,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / PreRev",
            value: 0.000009110443999816198,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / PostRev",
            value: 0.000009450032000131614,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / BothRev",
            value: 0.000009294431999933294,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / Primal",
            value: 0.000008798468013992533,
            unit: "s",
          },
          {
            name: "Concat / Jax / cpu / Primal",
            value: 0.000008922448003431782,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / Primal",
            value: 0.000009177781990729271,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / Primal",
            value: 0.000009119334019487723,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / Primal",
            value: 0.00000956356100505218,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / Primal",
            value: 0.000009579441015375778,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / Primal",
            value: 0.000008801427989965304,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / Forward",
            value: 0.000014563331002136691,
            unit: "s",
          },
          {
            name: "Concat / Jax / cpu / Forward",
            value: 0.000014365164010087028,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / Forward",
            value: 0.000013623503997223452,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / Forward",
            value: 0.000014597375004086644,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / Forward",
            value: 0.000014496393006993458,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / Forward",
            value: 0.000014504335005767644,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / Forward",
            value: 0.000014801135985180737,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / PreRev",
            value: 0.000015709409984992815,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / PostRev",
            value: 0.000015634198003681376,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / BothRev",
            value: 0.0000147765570145566,
            unit: "s",
          },
          {
            name: "Concat / Jax / cpu / BothRev",
            value: 0.00001584030600497499,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / PreRev",
            value: 0.00001489506399957463,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / PostRev",
            value: 0.00001591519298381172,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / BothRev",
            value: 0.000016015859990147873,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / PreRev",
            value: 0.000016345350013580172,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / PostRev",
            value: 0.000016430591989774255,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / BothRev",
            value: 0.000015439401002367958,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / PreRev",
            value: 0.000016480252001201734,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / PostRev",
            value: 0.000015382484009023757,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / BothRev",
            value: 0.000016433778015198185,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / PreRev",
            value: 0.00001639938002335839,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / PostRev",
            value: 0.000016139762999955564,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / BothRev",
            value: 0.000015420267998706548,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / PreRev",
            value: 0.000016325287986546753,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / PostRev",
            value: 0.000016041285009123386,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / BothRev",
            value: 0.000015273833996616302,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / tpu / Primal",
            value: 0.0001512151759998,
            unit: "s",
          },
          {
            name: "Concat / Jax / tpu / Primal",
            value: 0.0001516708570002,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / tpu / Primal",
            value: 0.000150039397,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / tpu / Primal",
            value: 0.0001314918779999,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / tpu / Primal",
            value: 0.000131537578,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / tpu / Primal",
            value: 0.0001348788170002,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / tpu / Primal",
            value: 0.0001424193869997,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / tpu / Forward",
            value: 0.0002011914060003,
            unit: "s",
          },
          {
            name: "Concat / Jax / tpu / Forward",
            value: 0.0002196790649995,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / tpu / Forward",
            value: 0.0002071389360003,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / tpu / Forward",
            value: 0.0002195806950003,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / tpu / Forward",
            value: 0.0002247635859998,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / tpu / Forward",
            value: 0.0002138994250003,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / tpu / Forward",
            value: 0.0002188052559999,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / tpu / PreRev",
            value: 0.0002269607150001,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / tpu / PostRev",
            value: 0.0002101932259997,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / tpu / BothRev",
            value: 0.0002038935659998,
            unit: "s",
          },
          {
            name: "Concat / Jax / tpu / BothRev",
            value: 0.0002061990759998,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / tpu / PreRev",
            value: 0.0002198057349996,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / tpu / PostRev",
            value: 0.0002068467060003,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / tpu / BothRev",
            value: 0.0002045525359999,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / tpu / PreRev",
            value: 0.0002249057549997,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / tpu / PostRev",
            value: 0.000225473205,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / tpu / BothRev",
            value: 0.0002344548649998,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / tpu / PreRev",
            value: 0.0002275210349998,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / tpu / PostRev",
            value: 0.0002198559060002,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / tpu / BothRev",
            value: 0.0002204716249998,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / tpu / PreRev",
            value: 0.0002256227449997,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / tpu / PostRev",
            value: 0.0002213250249997,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / tpu / BothRev",
            value: 0.0002241365660001,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / tpu / PreRev",
            value: 0.0002157603260002,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / tpu / PostRev",
            value: 0.0002060198459998,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / tpu / BothRev",
            value: 0.0001815451359998,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / Primal",
            value: 0.000006662589000370644,
            unit: "s",
          },
          {
            name: "Concat / Jax / cpu / Primal",
            value: 0.0000065610080000624295,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / Primal",
            value: 0.000006633844000134559,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / Primal",
            value: 0.000006655567000052542,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / Primal",
            value: 0.000007155518000217853,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / Primal",
            value: 0.000006593563000024005,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / Primal",
            value: 0.0000066921279994858194,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / Forward",
            value: 0.00001067526299993915,
            unit: "s",
          },
          {
            name: "Concat / Jax / cpu / Forward",
            value: 0.000010977502999594437,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / Forward",
            value: 0.000010362816000451855,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / Forward",
            value: 0.0000106710430000021,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / Forward",
            value: 0.00001019386299958569,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / Forward",
            value: 0.000010660608999387475,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / Forward",
            value: 0.000010054416000457422,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / PreRev",
            value: 0.000011555412000234354,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / PostRev",
            value: 0.000011525468000399995,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / BothRev",
            value: 0.000011821604000033403,
            unit: "s",
          },
          {
            name: "Concat / Jax / cpu / BothRev",
            value: 0.000011634542000138026,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / PreRev",
            value: 0.000010978004000207875,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / PostRev",
            value: 0.000011621389000538329,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / BothRev",
            value: 0.000011447527000200353,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / PreRev",
            value: 0.000011792592000347212,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / PostRev",
            value: 0.00001157952599987766,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / BothRev",
            value: 0.000010745909999968716,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / PreRev",
            value: 0.000011586564999561231,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / PostRev",
            value: 0.00001154481499997928,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / BothRev",
            value: 0.00001153550399976666,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / PreRev",
            value: 0.00001184215499961283,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / PostRev",
            value: 0.000011530419000337134,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / BothRev",
            value: 0.000011646255999949065,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / PreRev",
            value: 0.000011479554000288772,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / PostRev",
            value: 0.000011495345000184898,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / BothRev",
            value: 0.000010833207999894512,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / Primal",
            value: 0.0000036530830002448056,
            unit: "s",
          },
          {
            name: "Concat / Jax / cpu / Primal",
            value: 0.000003732000001036795,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / Primal",
            value: 0.000003686709002067801,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / Primal",
            value: 0.000003699999997479608,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / Primal",
            value: 0.0000037222499995550606,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / Primal",
            value: 0.000003766207999433391,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / Primal",
            value: 0.000003655707998404978,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / Forward",
            value: 0.000005885792001208756,
            unit: "s",
          },
          {
            name: "Concat / Jax / cpu / Forward",
            value: 0.00000581524999870453,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / Forward",
            value: 0.0000058609589978004805,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / Forward",
            value: 0.00000578112499715644,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / Forward",
            value: 0.000005809209000290138,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / Forward",
            value: 0.000005861624998942716,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / Forward",
            value: 0.000006011792000208515,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / PreRev",
            value: 0.000006841333000920713,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / PostRev",
            value: 0.000007322249999560881,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / BothRev",
            value: 0.000006778124999982538,
            unit: "s",
          },
          {
            name: "Concat / Jax / cpu / BothRev",
            value: 0.000006654916996922111,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / PreRev",
            value: 0.000006434041999455075,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / PostRev",
            value: 0.000006600249998882646,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / BothRev",
            value: 0.000006527541001560166,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / PreRev",
            value: 0.000006678083002043422,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / PostRev",
            value: 0.000006624625002586981,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / BothRev",
            value: 0.000006585667000763351,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / PreRev",
            value: 0.000006528083002194762,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / PostRev",
            value: 0.000006547374996443978,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / BothRev",
            value: 0.000006477540999185294,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / PreRev",
            value: 0.000006476292001025286,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / PostRev",
            value: 0.000006671874998573912,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / BothRev",
            value: 0.0000070514580002054574,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / PreRev",
            value: 0.00000652083399836556,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / PostRev",
            value: 0.000006623166998906526,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / BothRev",
            value: 0.000006500499999674503,
            unit: "s",
          },
          {
            name: "const_scatter / JaXPipe / cpu / Primal",
            value: 0.00000462838900011775,
            unit: "s",
          },
          {
            name: "const_scatter / Jax / cpu / Primal",
            value: 0.000004550225999992108,
            unit: "s",
          },
          {
            name: "const_scatter / HLOOpt / cpu / Primal",
            value: 0.000005233080999914819,
            unit: "s",
          },
          {
            name: "const_scatter / PartOpt / cpu / Primal",
            value: 0.000004910809999728371,
            unit: "s",
          },
          {
            name: "const_scatter / IPartOpt / cpu / Primal",
            value: 0.000004569560999698296,
            unit: "s",
          },
          {
            name: "const_scatter / DefOpt / cpu / Primal",
            value: 0.000005203020999942964,
            unit: "s",
          },
          {
            name: "const_scatter / IDefOpt / cpu / Primal",
            value: 0.000005509951000021829,
            unit: "s",
          },
          {
            name: "const_scatter / JaXPipe / cpu / Forward",
            value: 0.000008643185999972047,
            unit: "s",
          },
          {
            name: "const_scatter / Jax / cpu / Forward",
            value: 0.000007654000999991695,
            unit: "s",
          },
          {
            name: "const_scatter / HLOOpt / cpu / Forward",
            value: 0.000008709818000170344,
            unit: "s",
          },
          {
            name: "const_scatter / PartOpt / cpu / Forward",
            value: 0.000008826930999930482,
            unit: "s",
          },
          {
            name: "const_scatter / IPartOpt / cpu / Forward",
            value: 0.000008755954000207567,
            unit: "s",
          },
          {
            name: "const_scatter / DefOpt / cpu / Forward",
            value: 0.000008659751000323013,
            unit: "s",
          },
          {
            name: "const_scatter / IDefOpt / cpu / Forward",
            value: 0.000008925581999847053,
            unit: "s",
          },
          {
            name: "const_scatter / JaXPipe / cpu / Primal",
            value: 0.000008821585011901334,
            unit: "s",
          },
          {
            name: "const_scatter / Jax / cpu / Primal",
            value: 0.000008930784009862691,
            unit: "s",
          },
          {
            name: "const_scatter / HLOOpt / cpu / Primal",
            value: 0.00000999140299973078,
            unit: "s",
          },
          {
            name: "const_scatter / PartOpt / cpu / Primal",
            value: 0.000008663128013722598,
            unit: "s",
          },
          {
            name: "const_scatter / IPartOpt / cpu / Primal",
            value: 0.000009004432009533048,
            unit: "s",
          },
          {
            name: "const_scatter / DefOpt / cpu / Primal",
            value: 0.000010568866011453791,
            unit: "s",
          },
          {
            name: "const_scatter / IDefOpt / cpu / Primal",
            value: 0.0000100993060041219,
            unit: "s",
          },
          {
            name: "const_scatter / JaXPipe / cpu / Forward",
            value: 0.00001580739999189973,
            unit: "s",
          },
          {
            name: "const_scatter / Jax / cpu / Forward",
            value: 0.000013768789009191096,
            unit: "s",
          },
          {
            name: "const_scatter / HLOOpt / cpu / Forward",
            value: 0.000015010261995485052,
            unit: "s",
          },
          {
            name: "const_scatter / PartOpt / cpu / Forward",
            value: 0.000015926038002362473,
            unit: "s",
          },
          {
            name: "const_scatter / IPartOpt / cpu / Forward",
            value: 0.000015054479998070749,
            unit: "s",
          },
          {
            name: "const_scatter / DefOpt / cpu / Forward",
            value: 0.00001584928997908719,
            unit: "s",
          },
          {
            name: "const_scatter / IDefOpt / cpu / Forward",
            value: 0.000015929737011902035,
            unit: "s",
          },
          {
            name: "const_scatter / JaXPipe / tpu / Primal",
            value: 0.0001425771869999,
            unit: "s",
          },
          {
            name: "const_scatter / Jax / tpu / Primal",
            value: 0.0001464105470004,
            unit: "s",
          },
          {
            name: "const_scatter / HLOOpt / tpu / Primal",
            value: 0.0001426230370002,
            unit: "s",
          },
          {
            name: "const_scatter / PartOpt / tpu / Primal",
            value: 0.0001426463269999,
            unit: "s",
          },
          {
            name: "const_scatter / IPartOpt / tpu / Primal",
            value: 0.0001425783270001,
            unit: "s",
          },
          {
            name: "const_scatter / DefOpt / tpu / Primal",
            value: 0.0001420161770001,
            unit: "s",
          },
          {
            name: "const_scatter / IDefOpt / tpu / Primal",
            value: 0.000130069237,
            unit: "s",
          },
          {
            name: "const_scatter / JaXPipe / tpu / Forward",
            value: 0.0002264315459997,
            unit: "s",
          },
          {
            name: "const_scatter / Jax / tpu / Forward",
            value: 0.0002304174649998,
            unit: "s",
          },
          {
            name: "const_scatter / HLOOpt / tpu / Forward",
            value: 0.0002362983860002,
            unit: "s",
          },
          {
            name: "const_scatter / PartOpt / tpu / Forward",
            value: 0.0002265624250003,
            unit: "s",
          },
          {
            name: "const_scatter / IPartOpt / tpu / Forward",
            value: 0.0002398492249999,
            unit: "s",
          },
          {
            name: "const_scatter / DefOpt / tpu / Forward",
            value: 0.000240691725,
            unit: "s",
          },
          {
            name: "const_scatter / IDefOpt / tpu / Forward",
            value: 0.0002255893149999,
            unit: "s",
          },
          {
            name: "const_scatter / JaXPipe / cpu / Primal",
            value: 0.00000639693300036015,
            unit: "s",
          },
          {
            name: "const_scatter / Jax / cpu / Primal",
            value: 0.000006298207999861915,
            unit: "s",
          },
          {
            name: "const_scatter / HLOOpt / cpu / Primal",
            value: 0.000007107557000381348,
            unit: "s",
          },
          {
            name: "const_scatter / PartOpt / cpu / Primal",
            value: 0.000006412622000425472,
            unit: "s",
          },
          {
            name: "const_scatter / IPartOpt / cpu / Primal",
            value: 0.000006312325000180863,
            unit: "s",
          },
          {
            name: "const_scatter / DefOpt / cpu / Primal",
            value: 0.000007096379999893543,
            unit: "s",
          },
          {
            name: "const_scatter / IDefOpt / cpu / Primal",
            value: 0.000007544894000602653,
            unit: "s",
          },
          {
            name: "const_scatter / JaXPipe / cpu / Forward",
            value: 0.00001108006899994507,
            unit: "s",
          },
          {
            name: "const_scatter / Jax / cpu / Forward",
            value: 0.000009082773000045564,
            unit: "s",
          },
          {
            name: "const_scatter / HLOOpt / cpu / Forward",
            value: 0.00001053954100007104,
            unit: "s",
          },
          {
            name: "const_scatter / PartOpt / cpu / Forward",
            value: 0.000010421718000543478,
            unit: "s",
          },
          {
            name: "const_scatter / IPartOpt / cpu / Forward",
            value: 0.000010541110999838566,
            unit: "s",
          },
          {
            name: "const_scatter / DefOpt / cpu / Forward",
            value: 0.000011087631000009424,
            unit: "s",
          },
          {
            name: "const_scatter / IDefOpt / cpu / Forward",
            value: 0.000011160022999320062,
            unit: "s",
          },
          {
            name: "const_scatter / JaXPipe / cpu / Primal",
            value: 0.000003334375000122236,
            unit: "s",
          },
          {
            name: "const_scatter / Jax / cpu / Primal",
            value: 0.0000033175829994434023,
            unit: "s",
          },
          {
            name: "const_scatter / HLOOpt / cpu / Primal",
            value: 0.000003829459001281066,
            unit: "s",
          },
          {
            name: "const_scatter / PartOpt / cpu / Primal",
            value: 0.000003335375000460772,
            unit: "s",
          },
          {
            name: "const_scatter / IPartOpt / cpu / Primal",
            value: 0.0000033498750017315613,
            unit: "s",
          },
          {
            name: "const_scatter / DefOpt / cpu / Primal",
            value: 0.000003825957999652019,
            unit: "s",
          },
          {
            name: "const_scatter / IDefOpt / cpu / Primal",
            value: 0.0000037505420004890766,
            unit: "s",
          },
          {
            name: "const_scatter / JaXPipe / cpu / Forward",
            value: 0.000006291624998993938,
            unit: "s",
          },
          {
            name: "const_scatter / Jax / cpu / Forward",
            value: 0.000005074917000456481,
            unit: "s",
          },
          {
            name: "const_scatter / HLOOpt / cpu / Forward",
            value: 0.00000646079199941596,
            unit: "s",
          },
          {
            name: "const_scatter / PartOpt / cpu / Forward",
            value: 0.000006096583001635736,
            unit: "s",
          },
          {
            name: "const_scatter / IPartOpt / cpu / Forward",
            value: 0.000006214042001374764,
            unit: "s",
          },
          {
            name: "const_scatter / DefOpt / cpu / Forward",
            value: 0.000006203458000527462,
            unit: "s",
          },
          {
            name: "const_scatter / IDefOpt / cpu / Forward",
            value: 0.000006379124999511987,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / Primal",
            value: 0.000005336445000011736,
            unit: "s",
          },
          {
            name: "GenDot / Jax / cpu / Primal",
            value: 0.000005226979999861214,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / Primal",
            value: 0.000005417883000063739,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / Primal",
            value: 0.0000050015940000776026,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / Primal",
            value: 0.000005028966999816475,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / Primal",
            value: 0.000005308594000325684,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / Primal",
            value: 0.000005383829000038532,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / Forward",
            value: 0.000008644304999961606,
            unit: "s",
          },
          {
            name: "GenDot / Jax / cpu / Forward",
            value: 0.00000766213399992921,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / Forward",
            value: 0.000008323945000029199,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / Forward",
            value: 0.000008619940000244242,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / Forward",
            value: 0.000008244580999871687,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / Forward",
            value: 0.000008442937999916466,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / Forward",
            value: 0.000008467003000077966,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / PreRev",
            value: 0.000008363178999843513,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / PostRev",
            value: 0.000007690051000281529,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / BothRev",
            value: 0.000008386104999772215,
            unit: "s",
          },
          {
            name: "GenDot / Jax / cpu / BothRev",
            value: 0.00000776316700012103,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / PreRev",
            value: 0.000008555058999718312,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / PostRev",
            value: 0.000008620191999852978,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / BothRev",
            value: 0.000008400334999805637,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / PreRev",
            value: 0.000008370877999823278,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / PostRev",
            value: 0.000007938514000215946,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / BothRev",
            value: 0.000008493093000197405,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / PreRev",
            value: 0.0000083781309999722,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / PostRev",
            value: 0.000007922706000044855,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / BothRev",
            value: 0.000008580168000207778,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / PreRev",
            value: 0.000008389406999867787,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / PostRev",
            value: 0.000008377562000077887,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / BothRev",
            value: 0.000008370168000055856,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / PreRev",
            value: 0.000008450942999843392,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / PostRev",
            value: 0.000008385165999698074,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / BothRev",
            value: 0.000008422844999586233,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / Primal",
            value: 0.000009446923999348655,
            unit: "s",
          },
          {
            name: "GenDot / Jax / cpu / Primal",
            value: 0.000009440266992896797,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / Primal",
            value: 0.000011145483003929258,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / Primal",
            value: 0.000010196902003372088,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / Primal",
            value: 0.00000910205498803407,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / Primal",
            value: 0.00001095426597748883,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / Primal",
            value: 0.000010875437001232056,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / Forward",
            value: 0.000016157003992702813,
            unit: "s",
          },
          {
            name: "GenDot / Jax / cpu / Forward",
            value: 0.000014489355002297088,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / Forward",
            value: 0.000016029509017243982,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / Forward",
            value: 0.000016257233015494422,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / Forward",
            value: 0.000016944522998528554,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / Forward",
            value: 0.00001703136702417396,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / Forward",
            value: 0.000016276018985081463,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / PreRev",
            value: 0.000017372036993037908,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / PostRev",
            value: 0.000014437286998145282,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / BothRev",
            value: 0.000016555079986574127,
            unit: "s",
          },
          {
            name: "GenDot / Jax / cpu / BothRev",
            value: 0.000015220481000142172,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / PreRev",
            value: 0.000017389646993251518,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / PostRev",
            value: 0.00001725112702115439,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / BothRev",
            value: 0.000017222777009010314,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / PreRev",
            value: 0.00001713239200762473,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / PostRev",
            value: 0.000013987877988256512,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / BothRev",
            value: 0.00001649382899631746,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / PreRev",
            value: 0.000017328085988992827,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / PostRev",
            value: 0.000013861376995919273,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / BothRev",
            value: 0.000015853048011194916,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / PreRev",
            value: 0.00001568369401502423,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / PostRev",
            value: 0.0000166207040019799,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / BothRev",
            value: 0.000016597642010310664,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / PreRev",
            value: 0.000015663397003663704,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / PostRev",
            value: 0.00001664403997710906,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / BothRev",
            value: 0.000016612500010523945,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / tpu / Primal",
            value: 0.0001504123969998,
            unit: "s",
          },
          {
            name: "GenDot / Jax / tpu / Primal",
            value: 0.000163205707,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / tpu / Primal",
            value: 0.0001629041969999,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / tpu / Primal",
            value: 0.0001446327370003,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / tpu / Primal",
            value: 0.0001448998770001,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / tpu / Primal",
            value: 0.0001496698269997,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / tpu / Primal",
            value: 0.0001453103969997,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / tpu / Forward",
            value: 0.0002097784859997,
            unit: "s",
          },
          {
            name: "GenDot / Jax / tpu / Forward",
            value: 0.000213654915,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / tpu / Forward",
            value: 0.0002293275249999,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / tpu / Forward",
            value: 0.0002227690660001,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / tpu / Forward",
            value: 0.0002176060860001,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / tpu / Forward",
            value: 0.0002264938150001,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / tpu / Forward",
            value: 0.0002314317350001,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / tpu / PreRev",
            value: 0.0002224861650001,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / tpu / PostRev",
            value: 0.000215130346,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / tpu / BothRev",
            value: 0.0002105478150001,
            unit: "s",
          },
          {
            name: "GenDot / Jax / tpu / BothRev",
            value: 0.0002185907759999,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / tpu / PreRev",
            value: 0.0002307472060001,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / tpu / PostRev",
            value: 0.0002271139949998,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / tpu / BothRev",
            value: 0.0002024363660002,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / tpu / PreRev",
            value: 0.0002048742060001,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / tpu / PostRev",
            value: 0.0002181741960002,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / tpu / BothRev",
            value: 0.0002145700159999,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / tpu / PreRev",
            value: 0.0002135814859998,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / tpu / PostRev",
            value: 0.0002008281359999,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / tpu / BothRev",
            value: 0.0002097392059999,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / tpu / PreRev",
            value: 0.0001864600060002,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / tpu / PostRev",
            value: 0.0002249524450003,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / tpu / BothRev",
            value: 0.0002124022860002,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / tpu / PreRev",
            value: 0.0002202653149997,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / tpu / PostRev",
            value: 0.0002252231249999,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / tpu / BothRev",
            value: 0.0002216962350003,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / Primal",
            value: 0.000006891934000123002,
            unit: "s",
          },
          {
            name: "GenDot / Jax / cpu / Primal",
            value: 0.0000072847290002755475,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / Primal",
            value: 0.000008224952999626111,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / Primal",
            value: 0.000007284914000592834,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / Primal",
            value: 0.000006865057000140951,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / Primal",
            value: 0.000008202644999983022,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / Primal",
            value: 0.000008176280999578011,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / Forward",
            value: 0.000011314682000374888,
            unit: "s",
          },
          {
            name: "GenDot / Jax / cpu / Forward",
            value: 0.000010184111999478774,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / Forward",
            value: 0.000011019980999662948,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / Forward",
            value: 0.000011761808000301244,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / Forward",
            value: 0.000011702806000357669,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / Forward",
            value: 0.000011857233000228009,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / Forward",
            value: 0.00001184897999974055,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / PreRev",
            value: 0.000011269100999925286,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / PostRev",
            value: 0.000010609323000608129,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / BothRev",
            value: 0.000011220890999538824,
            unit: "s",
          },
          {
            name: "GenDot / Jax / cpu / BothRev",
            value: 0.000010587291999399897,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / PreRev",
            value: 0.000011176642000464198,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / PostRev",
            value: 0.000011980907999713965,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / BothRev",
            value: 0.000011860300000080316,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / PreRev",
            value: 0.000011205050999706144,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / PostRev",
            value: 0.000010524135999730787,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / BothRev",
            value: 0.000011294667000584012,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / PreRev",
            value: 0.000011229018000449286,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / PostRev",
            value: 0.00000989469799969811,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / BothRev",
            value: 0.000011864277999848128,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / PreRev",
            value: 0.000011874578999595542,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / PostRev",
            value: 0.000011249185999986369,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / BothRev",
            value: 0.000011883211000167648,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / PreRev",
            value: 0.000011117287999695691,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / PostRev",
            value: 0.000011860540000270704,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / BothRev",
            value: 0.00001175356300063868,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / Primal",
            value: 0.000003679624998767394,
            unit: "s",
          },
          {
            name: "GenDot / Jax / cpu / Primal",
            value: 0.0000036486659992078783,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / Primal",
            value: 0.000003949667003325885,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / Primal",
            value: 0.000003734917001565918,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / Primal",
            value: 0.000003660375001345528,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / Primal",
            value: 0.000004014041998743778,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / Primal",
            value: 0.000003992165999079589,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / Forward",
            value: 0.00000604274999932386,
            unit: "s",
          },
          {
            name: "GenDot / Jax / cpu / Forward",
            value: 0.000005513957999937702,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / Forward",
            value: 0.000006405791002180194,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / Forward",
            value: 0.000006301666999206645,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / Forward",
            value: 0.000006073665997973876,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / Forward",
            value: 0.000006174041998747271,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / Forward",
            value: 0.000006147957999928622,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / PreRev",
            value: 0.0000063276670007326176,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / PostRev",
            value: 0.000005882916000700789,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / BothRev",
            value: 0.000006435125000280096,
            unit: "s",
          },
          {
            name: "GenDot / Jax / cpu / BothRev",
            value: 0.000005897291001019767,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / PreRev",
            value: 0.00000642929199966602,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / PostRev",
            value: 0.000006459040996560361,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / BothRev",
            value: 0.000006316791997960536,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / PreRev",
            value: 0.000006514332999358885,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / PostRev",
            value: 0.000005865291001100559,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / BothRev",
            value: 0.000006430166002246551,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / PreRev",
            value: 0.000006388583002262749,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / PostRev",
            value: 0.000005857874999492196,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / BothRev",
            value: 0.000006680875001620734,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / PreRev",
            value: 0.0000065122080013679805,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / PostRev",
            value: 0.000006375791999744252,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / BothRev",
            value: 0.000006434834001993295,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / PreRev",
            value: 0.000006605999999010237,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / PostRev",
            value: 0.000006419583998649614,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / BothRev",
            value: 0.000006604165999306133,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / Primal",
            value: 0.000007805731000189553,
            unit: "s",
          },
          {
            name: "hlo_ffi / Jax / cpu / Primal",
            value: 0.000007436283999595616,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / Primal",
            value: 0.000007454965999841079,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / Primal",
            value: 0.00000742510800000673,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / Primal",
            value: 0.000007431387999986327,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / Primal",
            value: 0.000007432406999669183,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / Primal",
            value: 0.000007445217000167758,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / Forward",
            value: 0.000011550599999736733,
            unit: "s",
          },
          {
            name: "hlo_ffi / Jax / cpu / Forward",
            value: 0.000011478337999960786,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / Forward",
            value: 0.000011542870000084804,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / Forward",
            value: 0.00001152770099997724,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / Forward",
            value: 0.00001154893400007495,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / Forward",
            value: 0.000011508124999636492,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / Forward",
            value: 0.000011542981000275176,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / PreRev",
            value: 0.000011181554999893706,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / PostRev",
            value: 0.00001116318999993382,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / BothRev",
            value: 0.000011197251999874425,
            unit: "s",
          },
          {
            name: "hlo_ffi / Jax / cpu / BothRev",
            value: 0.000011150220000217816,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / PreRev",
            value: 0.000011197113999969588,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / PostRev",
            value: 0.000011127440000109346,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / BothRev",
            value: 0.00001122590200020568,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / PreRev",
            value: 0.000011115592999885849,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / PostRev",
            value: 0.000011166187999606337,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / BothRev",
            value: 0.00001108760099987194,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / PreRev",
            value: 0.000011163784000018497,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / PostRev",
            value: 0.000011110703000213108,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / BothRev",
            value: 0.000011124979000214808,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / PreRev",
            value: 0.000011157681999975466,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / PostRev",
            value: 0.000011156801000197447,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / BothRev",
            value: 0.000011130770999898232,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / PreRev",
            value: 0.000011124540999844612,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / PostRev",
            value: 0.000011170001000209596,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / BothRev",
            value: 0.000011173341999892728,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / Primal",
            value: 0.000013570443989010529,
            unit: "s",
          },
          {
            name: "hlo_ffi / Jax / cpu / Primal",
            value: 0.000013556819991208611,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / Primal",
            value: 0.000013508433010429144,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / Primal",
            value: 0.00001361553999595344,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / Primal",
            value: 0.000013477221975335851,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / Primal",
            value: 0.000013608111999928951,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / Primal",
            value: 0.000014261258009355516,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / Forward",
            value: 0.00001946320399292745,
            unit: "s",
          },
          {
            name: "hlo_ffi / Jax / cpu / Forward",
            value: 0.00002052429100149311,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / Forward",
            value: 0.00001932011498138309,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / Forward",
            value: 0.000020268791005946697,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / Forward",
            value: 0.0000205077110149432,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / Forward",
            value: 0.000020415603008586915,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / Forward",
            value: 0.00002034920299774967,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / PreRev",
            value: 0.000019807717006187887,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / PostRev",
            value: 0.00001979841399588622,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / BothRev",
            value: 0.0000189201720058918,
            unit: "s",
          },
          {
            name: "hlo_ffi / Jax / cpu / BothRev",
            value: 0.00001975209699594416,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / PreRev",
            value: 0.000018909008998889475,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / PostRev",
            value: 0.00001974809699459001,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / BothRev",
            value: 0.000018910382001195104,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / PreRev",
            value: 0.00001990112700150348,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / PostRev",
            value: 0.00001989694600342773,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / BothRev",
            value: 0.000019001039006980138,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / PreRev",
            value: 0.00001897366999764927,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / PostRev",
            value: 0.000019786787015618755,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / BothRev",
            value: 0.000019752656982745977,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / PreRev",
            value: 0.00001890899601858109,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / PostRev",
            value: 0.000019794238003669308,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / BothRev",
            value: 0.000019789493991993368,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / PreRev",
            value: 0.000019808637007372453,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / PostRev",
            value: 0.00001972126599866897,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / BothRev",
            value: 0.000019904028973542157,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / tpu / Primal",
            value: 0.000229508368,
            unit: "s",
          },
          {
            name: "hlo_ffi / Jax / tpu / Primal",
            value: 0.0002288182789998,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / tpu / Primal",
            value: 0.0002199531189999,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / tpu / Primal",
            value: 0.0001947024889996,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / tpu / Primal",
            value: 0.0001956065690001,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / tpu / Primal",
            value: 0.0002002117090005,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / tpu / Primal",
            value: 0.0001991064490002,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / tpu / Forward",
            value: 0.0002490617289995,
            unit: "s",
          },
          {
            name: "hlo_ffi / Jax / tpu / Forward",
            value: 0.0002461747589995,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / tpu / Forward",
            value: 0.0002317564289996,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / tpu / Forward",
            value: 0.0002440848989999,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / tpu / Forward",
            value: 0.0002466960190004,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / tpu / Forward",
            value: 0.0002311749990003,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / tpu / Forward",
            value: 0.0002276666189991,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / tpu / PreRev",
            value: 0.0002245429080003,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / tpu / PostRev",
            value: 0.0002281853490003,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / tpu / BothRev",
            value: 0.0002090897079997,
            unit: "s",
          },
          {
            name: "hlo_ffi / Jax / tpu / BothRev",
            value: 0.0002142746989993,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / tpu / PreRev",
            value: 0.0002237123479999,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / tpu / PostRev",
            value: 0.0002310047379996,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / tpu / BothRev",
            value: 0.0002010495289996,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / tpu / PreRev",
            value: 0.0002358199290001,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / tpu / PostRev",
            value: 0.000241779029,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / tpu / BothRev",
            value: 0.0002427261889997,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / tpu / PreRev",
            value: 0.0002352799190002,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / tpu / PostRev",
            value: 0.0002344890390004,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / tpu / BothRev",
            value: 0.0002390832689998,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / tpu / PreRev",
            value: 0.0002496816490001,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / tpu / PostRev",
            value: 0.000250060069,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / tpu / BothRev",
            value: 0.0002499754689997,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / tpu / PreRev",
            value: 0.0002526729089995,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / tpu / PostRev",
            value: 0.0002399091280003,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / tpu / BothRev",
            value: 0.0002371386989998,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / Primal",
            value: 0.000010128369000085513,
            unit: "s",
          },
          {
            name: "hlo_ffi / Jax / cpu / Primal",
            value: 0.000009630538999772398,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / Primal",
            value: 0.000010063525999612466,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / Primal",
            value: 0.000009970801999770629,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / Primal",
            value: 0.000009989779000534328,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / Primal",
            value: 0.000009528264999971723,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / Primal",
            value: 0.000010233773999971164,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / Forward",
            value: 0.000014235945000109496,
            unit: "s",
          },
          {
            name: "hlo_ffi / Jax / cpu / Forward",
            value: 0.000014975045000028333,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / Forward",
            value: 0.000013960506999865176,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / Forward",
            value: 0.000014536217000568286,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / Forward",
            value: 0.000014538316999278322,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / Forward",
            value: 0.000013888759999645116,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / Forward",
            value: 0.000014513921999423474,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / PreRev",
            value: 0.0000135373700004493,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / PostRev",
            value: 0.000014155935999951908,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / BothRev",
            value: 0.000014261396000620151,
            unit: "s",
          },
          {
            name: "hlo_ffi / Jax / cpu / BothRev",
            value: 0.000014379838999957427,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / PreRev",
            value: 0.000013580232999629516,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / PostRev",
            value: 0.000014126117000159866,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / BothRev",
            value: 0.000014159686000311924,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / PreRev",
            value: 0.000014243028999771925,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / PostRev",
            value: 0.000014349521999974967,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / BothRev",
            value: 0.000013672879000296236,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / PreRev",
            value: 0.00001429441699929157,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / PostRev",
            value: 0.000014313945000139938,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / BothRev",
            value: 0.000014287055999375298,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / PreRev",
            value: 0.00001433485399957135,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / PostRev",
            value: 0.000014224001000002318,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / BothRev",
            value: 0.00001345311099976243,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / PreRev",
            value: 0.000014252154000132576,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / PostRev",
            value: 0.000014262828000028094,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / BothRev",
            value: 0.000013624137999613597,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / Primal",
            value: 0.00000608950000241748,
            unit: "s",
          },
          {
            name: "hlo_ffi / Jax / cpu / Primal",
            value: 0.000005769208000856452,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / Primal",
            value: 0.000005743333000282292,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / Primal",
            value: 0.000005822041999635985,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / Primal",
            value: 0.000005766709000454284,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / Primal",
            value: 0.000006093374999181833,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / Primal",
            value: 0.000005856499999936205,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / Forward",
            value: 0.000008920374999433989,
            unit: "s",
          },
          {
            name: "hlo_ffi / Jax / cpu / Forward",
            value: 0.000008900207998522092,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / Forward",
            value: 0.000009422041999641804,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / Forward",
            value: 0.000008757499999774154,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / Forward",
            value: 0.000008853125000314322,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / Forward",
            value: 0.000009029916000145022,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / Forward",
            value: 0.000008782749999227235,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / PreRev",
            value: 0.000008644792000268352,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / PostRev",
            value: 0.000008642374999908497,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / BothRev",
            value: 0.000008797416998277185,
            unit: "s",
          },
          {
            name: "hlo_ffi / Jax / cpu / BothRev",
            value: 0.000008581499998399522,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / PreRev",
            value: 0.000008913207999285077,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / PostRev",
            value: 0.000008156708998285467,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / BothRev",
            value: 0.000008289834000606788,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / PreRev",
            value: 0.000008400792001339141,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / PostRev",
            value: 0.000008320749999256805,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / BothRev",
            value: 0.000008341791999555426,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / PreRev",
            value: 0.000008476125000015599,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / PostRev",
            value: 0.000008276583001133986,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / BothRev",
            value: 0.000008492375000059838,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / PreRev",
            value: 0.000008910874999855879,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / PostRev",
            value: 0.000009260708000510931,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / BothRev",
            value: 0.00000931845900049666,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / PreRev",
            value: 0.000009527459002129035,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / PostRev",
            value: 0.000009036291001393692,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / BothRev",
            value: 0.000009099874998355517,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / Primal",
            value: 0.0008575742000175,
            unit: "s",
          },
          {
            name: "llama / Jax / cpu / Primal",
            value: 0.0008535714000117,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / Primal",
            value: 0.0009101491999899,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / Primal",
            value: 0.0008513135999692,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / Primal",
            value: 0.0008453212999938,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / Primal",
            value: 0.0009370504999878,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / Primal",
            value: 0.0009345987000415,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / Forward",
            value: 0.0025938537000001,
            unit: "s",
          },
          {
            name: "llama / Jax / cpu / Forward",
            value: 0.0026111105999916,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / Forward",
            value: 0.0025958579999951,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / Forward",
            value: 0.0026039459999992,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / Forward",
            value: 0.0025947858000108,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / Forward",
            value: 0.0025867689999813,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / Forward",
            value: 0.0025750896999852,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / PreRev",
            value: 0.0061970945999746,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / PostRev",
            value: 0.0053360353000243,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / BothRev",
            value: 0.0054701147000287,
            unit: "s",
          },
          {
            name: "llama / Jax / cpu / BothRev",
            value: 0.0054441570999642,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / PreRev",
            value: 0.005348265900011,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / PostRev",
            value: 0.005920534300003,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / BothRev",
            value: 0.0054933063000135,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / PreRev",
            value: 0.0061852817000271,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / PostRev",
            value: 0.0054302456999721,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / BothRev",
            value: 0.0062487544999839,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / PreRev",
            value: 0.0052895072999945,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / PostRev",
            value: 0.0052477061999979,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / BothRev",
            value: 0.0057023775999823,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / PreRev",
            value: 0.0061409314999764,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / PostRev",
            value: 0.0051777523000055,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / BothRev",
            value: 0.0055833843000073,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / PreRev",
            value: 0.0062369342000238,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / PostRev",
            value: 0.0054664409999986,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / BothRev",
            value: 0.00611264690001,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / Primal",
            value: 0.001634754097904,
            unit: "s",
          },
          {
            name: "llama / Jax / cpu / Primal",
            value: 0.0016830365988425,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / Primal",
            value: 0.0016826070990646,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / Primal",
            value: 0.0015890606999164,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / Primal",
            value: 0.0016099050990305,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / Primal",
            value: 0.0016820713994093,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / Primal",
            value: 0.0016958905995124,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / Forward",
            value: 0.0053481583017855,
            unit: "s",
          },
          {
            name: "llama / Jax / cpu / Forward",
            value: 0.0052854184003081,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / Forward",
            value: 0.0051951630011899,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / Forward",
            value: 0.0053068100998643,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / Forward",
            value: 0.0052170695998938,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / Forward",
            value: 0.0052158208010951,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / Forward",
            value: 0.0052873959008138,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / PreRev",
            value: 0.0099947784008691,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / PostRev",
            value: 0.0109428591997129,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / BothRev",
            value: 0.0098962115996982,
            unit: "s",
          },
          {
            name: "llama / Jax / cpu / BothRev",
            value: 0.010737463098485,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / PreRev",
            value: 0.0086503322003409,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / PostRev",
            value: 0.0132360364019405,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / BothRev",
            value: 0.0075635244982549,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / PreRev",
            value: 0.0108805530995596,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / PostRev",
            value: 0.0092462916974909,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / BothRev",
            value: 0.0111831648013321,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / PreRev",
            value: 0.009558738299529,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / PostRev",
            value: 0.0109285405022092,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / BothRev",
            value: 0.00943690300046,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / PreRev",
            value: 0.0133332425990374,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / PostRev",
            value: 0.0093040837004082,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / BothRev",
            value: 0.0129207285994198,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / PreRev",
            value: 0.0075637719011865,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / PostRev",
            value: 0.0109799574012868,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / BothRev",
            value: 0.008970886201132,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / tpu / Primal",
            value: 0.0003623791999962,
            unit: "s",
          },
          {
            name: "llama / Jax / tpu / Primal",
            value: 0.0003596383999956,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / tpu / Primal",
            value: 0.0003452620000007,
            unit: "s",
          },
          {
            name: "llama / PartOpt / tpu / Primal",
            value: 0.0003671450000001,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / tpu / Primal",
            value: 0.0003722523999931,
            unit: "s",
          },
          {
            name: "llama / DefOpt / tpu / Primal",
            value: 0.0003345816,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / tpu / Primal",
            value: 0.000354596799998,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / tpu / Forward",
            value: 0.0005613275999985,
            unit: "s",
          },
          {
            name: "llama / Jax / tpu / Forward",
            value: 0.0006943163800042,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / tpu / Forward",
            value: 0.000562223400002,
            unit: "s",
          },
          {
            name: "llama / PartOpt / tpu / Forward",
            value: 0.0005536208000012,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / tpu / Forward",
            value: 0.0005617043999973,
            unit: "s",
          },
          {
            name: "llama / DefOpt / tpu / Forward",
            value: 0.0005333049799992,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / tpu / Forward",
            value: 0.0005401623999932,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / tpu / PreRev",
            value: 0.0007339044000036,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / tpu / PostRev",
            value: 0.0007486238000001,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / tpu / BothRev",
            value: 0.0007870915800049,
            unit: "s",
          },
          {
            name: "llama / Jax / tpu / BothRev",
            value: 0.0007537391799996,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / tpu / PreRev",
            value: 0.0007917564000126,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / tpu / PostRev",
            value: 0.0008064227999966,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / tpu / BothRev",
            value: 0.0008076614000128,
            unit: "s",
          },
          {
            name: "llama / PartOpt / tpu / PreRev",
            value: 0.0008044815799985,
            unit: "s",
          },
          {
            name: "llama / PartOpt / tpu / PostRev",
            value: 0.0007784704000005,
            unit: "s",
          },
          {
            name: "llama / PartOpt / tpu / BothRev",
            value: 0.0007436073999997,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / tpu / PreRev",
            value: 0.0007809070000075,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / tpu / PostRev",
            value: 0.0007672412000101,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / tpu / BothRev",
            value: 0.0007880295799986,
            unit: "s",
          },
          {
            name: "llama / DefOpt / tpu / PreRev",
            value: 0.0007872626000062,
            unit: "s",
          },
          {
            name: "llama / DefOpt / tpu / PostRev",
            value: 0.0007781568000063,
            unit: "s",
          },
          {
            name: "llama / DefOpt / tpu / BothRev",
            value: 0.0007438182000078,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / tpu / PreRev",
            value: 0.0007431522000115,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / tpu / PostRev",
            value: 0.000798460599999,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / tpu / BothRev",
            value: 0.0007667971999944,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / Primal",
            value: 0.0020902156000374,
            unit: "s",
          },
          {
            name: "llama / Jax / cpu / Primal",
            value: 0.0020110532000217,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / Primal",
            value: 0.0022120804999758,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / Primal",
            value: 0.001956187199994,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / Primal",
            value: 0.0020333743999799,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / Primal",
            value: 0.002108022900029,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / Primal",
            value: 0.0018932994000351,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / Forward",
            value: 0.0059721717999309,
            unit: "s",
          },
          {
            name: "llama / Jax / cpu / Forward",
            value: 0.0063227910000023,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / Forward",
            value: 0.0061128479000217,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / Forward",
            value: 0.0060519482999552,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / Forward",
            value: 0.0060749924000447,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / Forward",
            value: 0.0060793300000113,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / Forward",
            value: 0.0059764014999927,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / PreRev",
            value: 0.0106683572000292,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / PostRev",
            value: 0.0097151259000384,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / BothRev",
            value: 0.0097245667000606,
            unit: "s",
          },
          {
            name: "llama / Jax / cpu / BothRev",
            value: 0.0100161807000404,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / PreRev",
            value: 0.0101366401000632,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / PostRev",
            value: 0.0096271802000046,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / BothRev",
            value: 0.009547519500029,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / PreRev",
            value: 0.0104594983000424,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / PostRev",
            value: 0.0096025683999869,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / BothRev",
            value: 0.0101536509000652,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / PreRev",
            value: 0.0083609262000209,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / PostRev",
            value: 0.0089153173999875,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / BothRev",
            value: 0.0084802635000414,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / PreRev",
            value: 0.0099883763999969,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / PostRev",
            value: 0.0076287785000204,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / BothRev",
            value: 0.0098337346999869,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / PreRev",
            value: 0.0085008695999931,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / PostRev",
            value: 0.0098670573000163,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / BothRev",
            value: 0.0086228465000203,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / Primal",
            value: 0.0015657375002774,
            unit: "s",
          },
          {
            name: "llama / Jax / cpu / Primal",
            value: 0.0015167250003287,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / Primal",
            value: 0.0014600833997974,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / Primal",
            value: 0.0013432416999421,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / Primal",
            value: 0.0013457082997774,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / Primal",
            value: 0.0013614999999845,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / Primal",
            value: 0.0013687249997019,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / Forward",
            value: 0.0039197125002829,
            unit: "s",
          },
          {
            name: "llama / Jax / cpu / Forward",
            value: 0.0037858583997149,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / Forward",
            value: 0.0037524041999859,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / Forward",
            value: 0.0043236499997874,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / Forward",
            value: 0.0045037957999738,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / Forward",
            value: 0.0041631500000221,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / Forward",
            value: 0.0037964499999361,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / PreRev",
            value: 0.0069512041998677,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / PostRev",
            value: 0.0123801249999814,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / BothRev",
            value: 0.0070481792001373,
            unit: "s",
          },
          {
            name: "llama / Jax / cpu / BothRev",
            value: 0.0089688500000193,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / PreRev",
            value: 0.0070365708997996,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / PostRev",
            value: 0.0070884875000047,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / BothRev",
            value: 0.0073268500000267,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / PreRev",
            value: 0.0086324583000532,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / PostRev",
            value: 0.0093216666999069,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / BothRev",
            value: 0.0084567457997764,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / PreRev",
            value: 0.0081595958999969,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / PostRev",
            value: 0.0094277584001247,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / BothRev",
            value: 0.0084056167001108,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / PreRev",
            value: 0.0074158707997412,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / PostRev",
            value: 0.0066170665999379,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / BothRev",
            value: 0.0082733500003087,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / PreRev",
            value: 0.0072906207999039,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / PostRev",
            value: 0.0069987249997211,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / BothRev",
            value: 0.0072393124999507,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / Primal",
            value: 0.000005607683000107499,
            unit: "s",
          },
          {
            name: "scatter_sum / Jax / cpu / Primal",
            value: 0.000005537103000278876,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / Primal",
            value: 0.000005648173999816208,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / Primal",
            value: 0.0000056340639998779805,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / Primal",
            value: 0.000005589627000063047,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / Primal",
            value: 0.000005621220999728393,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / Primal",
            value: 0.000005658980000134761,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / Forward",
            value: 0.000009204487000260995,
            unit: "s",
          },
          {
            name: "scatter_sum / Jax / cpu / Forward",
            value: 0.000009083986000405276,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / Forward",
            value: 0.000009244734999811044,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / Forward",
            value: 0.0000093711610002174,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / Forward",
            value: 0.00000923285599992596,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / Forward",
            value: 0.00000919378999969922,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / Forward",
            value: 0.00000929726300000766,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / PreRev",
            value: 0.00000973059199986892,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / PostRev",
            value: 0.000009246843000255469,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / BothRev",
            value: 0.000009336011999948824,
            unit: "s",
          },
          {
            name: "scatter_sum / Jax / cpu / BothRev",
            value: 0.000009208711000155744,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / PreRev",
            value: 0.00000928221999993184,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / PostRev",
            value: 0.000009360357000332442,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / BothRev",
            value: 0.000009230690999629587,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / PreRev",
            value: 0.000009278805000121792,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / PostRev",
            value: 0.000009478133999891723,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / BothRev",
            value: 0.000009549005999815565,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / PreRev",
            value: 0.0000097392369998488,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / PostRev",
            value: 0.000009232250999957612,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / BothRev",
            value: 0.000009227368000210843,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / PreRev",
            value: 0.000009334722999938094,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / PostRev",
            value: 0.000009234227999968425,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / BothRev",
            value: 0.000009270725000078528,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / PreRev",
            value: 0.000009257907000119304,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / PostRev",
            value: 0.000009443205000025043,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / BothRev",
            value: 0.000009377999999742316,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / Primal",
            value: 0.00001070281397551298,
            unit: "s",
          },
          {
            name: "scatter_sum / Jax / cpu / Primal",
            value: 0.00001040200298302807,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / Primal",
            value: 0.000011142694012960418,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / Primal",
            value: 0.000011199111002497374,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / Primal",
            value: 0.000010656900994945318,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / Primal",
            value: 0.000010636300983605906,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / Primal",
            value: 0.000011160144000314177,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / Forward",
            value: 0.000015356585005065427,
            unit: "s",
          },
          {
            name: "scatter_sum / Jax / cpu / Forward",
            value: 0.000015310803020838648,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / Forward",
            value: 0.000015413274988532067,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / Forward",
            value: 0.000015324969979701565,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / Forward",
            value: 0.000016153076983755453,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / Forward",
            value: 0.000015587153990054505,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / Forward",
            value: 0.00001613720500608906,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / PreRev",
            value: 0.00001554515497991815,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / PostRev",
            value: 0.00001630564601509832,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / BothRev",
            value: 0.000016140500985784455,
            unit: "s",
          },
          {
            name: "scatter_sum / Jax / cpu / BothRev",
            value: 0.000016224658989813177,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / PreRev",
            value: 0.000016480546997627245,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / PostRev",
            value: 0.00001635294299921952,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / BothRev",
            value: 0.00001644898601807654,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / PreRev",
            value: 0.000016311244980897754,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / PostRev",
            value: 0.00001553244100068696,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / BothRev",
            value: 0.000015546394017292185,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / PreRev",
            value: 0.00001563902900670655,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / PostRev",
            value: 0.000015540020976914094,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / BothRev",
            value: 0.0000162513819814194,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / PreRev",
            value: 0.000016327847028151155,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / PostRev",
            value: 0.00001560664499993436,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / BothRev",
            value: 0.00001627915899734944,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / PreRev",
            value: 0.00001632900402182713,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / PostRev",
            value: 0.00001634039098280482,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / BothRev",
            value: 0.00001625477898051031,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / tpu / Primal",
            value: 0.0001478026470003,
            unit: "s",
          },
          {
            name: "scatter_sum / Jax / tpu / Primal",
            value: 0.0001381388569998,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / tpu / Primal",
            value: 0.0001383975169997,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / tpu / Primal",
            value: 0.0001505462570003,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / tpu / Primal",
            value: 0.0001494796869997,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / tpu / Primal",
            value: 0.0001489512769999,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / tpu / Primal",
            value: 0.0001487010569999,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / tpu / Forward",
            value: 0.0002231802360001,
            unit: "s",
          },
          {
            name: "scatter_sum / Jax / tpu / Forward",
            value: 0.0002228489459998,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / tpu / Forward",
            value: 0.0002153141450003,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / tpu / Forward",
            value: 0.0002268987649999,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / tpu / Forward",
            value: 0.0002261881649997,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / tpu / Forward",
            value: 0.0002149097459996,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / tpu / Forward",
            value: 0.0002330023949998,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / tpu / PreRev",
            value: 0.0002267442949996,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / tpu / PostRev",
            value: 0.000220409596,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / tpu / BothRev",
            value: 0.0002267623849998,
            unit: "s",
          },
          {
            name: "scatter_sum / Jax / tpu / BothRev",
            value: 0.0002239697159998,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / tpu / PreRev",
            value: 0.0002333603849997,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / tpu / PostRev",
            value: 0.0001966522259999,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / tpu / BothRev",
            value: 0.0001950143359999,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / tpu / PreRev",
            value: 0.0002095164759998,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / tpu / PostRev",
            value: 0.0002392379949997,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / tpu / BothRev",
            value: 0.0002363461349996,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / tpu / PreRev",
            value: 0.0002276435049998,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / tpu / PostRev",
            value: 0.0002270055950002,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / tpu / BothRev",
            value: 0.0002313427849999,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / tpu / PreRev",
            value: 0.000201780926,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / tpu / PostRev",
            value: 0.0002024762659998,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / tpu / BothRev",
            value: 0.0001976443459998,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / tpu / PreRev",
            value: 0.000209023025,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / tpu / PostRev",
            value: 0.0002240029849999,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / tpu / BothRev",
            value: 0.0002224199349998,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / Primal",
            value: 0.000007809247999830405,
            unit: "s",
          },
          {
            name: "scatter_sum / Jax / cpu / Primal",
            value: 0.000007625265000569925,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / Primal",
            value: 0.000008160237000083726,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / Primal",
            value: 0.000008201776000532846,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / Primal",
            value: 0.00000773024299996905,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / Primal",
            value: 0.000008194625000214727,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / Primal",
            value: 0.000008138378999319684,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / Forward",
            value: 0.000011050347000491455,
            unit: "s",
          },
          {
            name: "scatter_sum / Jax / cpu / Forward",
            value: 0.000011673449000227264,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / Forward",
            value: 0.000011539264999555598,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / Forward",
            value: 0.000011594088000492777,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / Forward",
            value: 0.000011616775000220512,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / Forward",
            value: 0.00001171499100018991,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / Forward",
            value: 0.000011004158000105236,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / PreRev",
            value: 0.000011774163999689337,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / PostRev",
            value: 0.000011762836999878346,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / BothRev",
            value: 0.000011864210999192438,
            unit: "s",
          },
          {
            name: "scatter_sum / Jax / cpu / BothRev",
            value: 0.000011852788999931364,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / PreRev",
            value: 0.000011725133000254572,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / PostRev",
            value: 0.00001177524699960486,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / BothRev",
            value: 0.00001189403599983052,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / PreRev",
            value: 0.000011802613000327256,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / PostRev",
            value: 0.000011991377999947872,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / BothRev",
            value: 0.0000111987689997477,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / PreRev",
            value: 0.000011769847000323352,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / PostRev",
            value: 0.000011751963999813595,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / BothRev",
            value: 0.00001182670300022437,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / PreRev",
            value: 0.000011399220999919635,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / PostRev",
            value: 0.000011839286000395077,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / BothRev",
            value: 0.000011698532999616874,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / PreRev",
            value: 0.000011753283999496487,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / PostRev",
            value: 0.000011895511000147965,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / BothRev",
            value: 0.000011855937000291304,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / Primal",
            value: 0.000004352332998678321,
            unit: "s",
          },
          {
            name: "scatter_sum / Jax / cpu / Primal",
            value: 0.000004548542001430178,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / Primal",
            value: 0.000004630125000403495,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / Primal",
            value: 0.000004684916999394773,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / Primal",
            value: 0.000004555583000183106,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / Primal",
            value: 0.000004529958001512569,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / Primal",
            value: 0.000004523084000538801,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / Forward",
            value: 0.000007327541999984533,
            unit: "s",
          },
          {
            name: "scatter_sum / Jax / cpu / Forward",
            value: 0.000007113583000318613,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / Forward",
            value: 0.000007202249998954358,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / Forward",
            value: 0.000007091500003298279,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / Forward",
            value: 0.000006666166998911649,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / Forward",
            value: 0.000006784125002013752,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / Forward",
            value: 0.000006708042001264402,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / PreRev",
            value: 0.000006870374996651663,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / PostRev",
            value: 0.000006670291000773432,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / BothRev",
            value: 0.000006905750000441912,
            unit: "s",
          },
          {
            name: "scatter_sum / Jax / cpu / BothRev",
            value: 0.000006965832999412669,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / PreRev",
            value: 0.00000694149999981164,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / PostRev",
            value: 0.00000698408400057815,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / BothRev",
            value: 0.000006943207998119761,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / PreRev",
            value: 0.0000072724160017969555,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / PostRev",
            value: 0.00000716508299956331,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / BothRev",
            value: 0.0000071801249978307166,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / PreRev",
            value: 0.000007682708001084392,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / PostRev",
            value: 0.000007050124997476814,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / BothRev",
            value: 0.000006820750000770204,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / PreRev",
            value: 0.000007073916000081226,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / PostRev",
            value: 0.000006938332997378893,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / BothRev",
            value: 0.000006818750000093132,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / PreRev",
            value: 0.000007145875002606772,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / PostRev",
            value: 0.000006915040998137556,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / BothRev",
            value: 0.000007103833002474857,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / Primal",
            value: 0.000004538390000107029,
            unit: "s",
          },
          {
            name: "slicing / Jax / cpu / Primal",
            value: 0.000004527486999904795,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / Primal",
            value: 0.000004537846999937756,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / Primal",
            value: 0.000004605523999998696,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / Primal",
            value: 0.000004512166000040452,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / Primal",
            value: 0.000004534153999884438,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / Primal",
            value: 0.000004565307000120811,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / Forward",
            value: 0.0000072241580000991235,
            unit: "s",
          },
          {
            name: "slicing / Jax / cpu / Forward",
            value: 0.000007062622999910672,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / Forward",
            value: 0.0000071019709998836335,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / Forward",
            value: 0.000007154944999911095,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / Forward",
            value: 0.000007213083999886294,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / Forward",
            value: 0.000007055172000036692,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / Forward",
            value: 0.000007226651000109996,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / PreRev",
            value: 0.000007593475000248873,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / PostRev",
            value: 0.000007630376999713917,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / BothRev",
            value: 0.000007834166000066034,
            unit: "s",
          },
          {
            name: "slicing / Jax / cpu / BothRev",
            value: 0.000007643118000032701,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / PreRev",
            value: 0.000007722194000052696,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / PostRev",
            value: 0.000007690754000122979,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / BothRev",
            value: 0.000007634684000095148,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / PreRev",
            value: 0.00000761149399977512,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / PostRev",
            value: 0.000007632026999999652,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / BothRev",
            value: 0.000007608506999986276,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / PreRev",
            value: 0.000007576895000056538,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / PostRev",
            value: 0.000007835760000034497,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / BothRev",
            value: 0.000007543015999999625,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / PreRev",
            value: 0.000007689478999964195,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / PostRev",
            value: 0.000007715238999935536,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / BothRev",
            value: 0.000007536062999861315,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / PreRev",
            value: 0.00000769321799998579,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / PostRev",
            value: 0.000008081965000201308,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / BothRev",
            value: 0.00000762053200014634,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / Primal",
            value: 0.000008400104998145252,
            unit: "s",
          },
          {
            name: "slicing / Jax / cpu / Primal",
            value: 0.000009043503989232704,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / Primal",
            value: 0.000008983256004285067,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / Primal",
            value: 0.000009030458983033897,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / Primal",
            value: 0.000008976971002994105,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / Primal",
            value: 0.000008988715999294072,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / Primal",
            value: 0.000008490225998684764,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / Forward",
            value: 0.000012267482001334427,
            unit: "s",
          },
          {
            name: "slicing / Jax / cpu / Forward",
            value: 0.000012988410016987473,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / Forward",
            value: 0.00001224481500685215,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / Forward",
            value: 0.000012950436008395629,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / Forward",
            value: 0.00001301782499649562,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / Forward",
            value: 0.000012272459018277004,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / Forward",
            value: 0.000013086372026009486,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / PreRev",
            value: 0.000013165176991606132,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / PostRev",
            value: 0.000013271308998810128,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / BothRev",
            value: 0.00001310646499041468,
            unit: "s",
          },
          {
            name: "slicing / Jax / cpu / BothRev",
            value: 0.000013944580015959218,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / PreRev",
            value: 0.000013085980986943468,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / PostRev",
            value: 0.00001383159900433384,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / BothRev",
            value: 0.000013184370996896178,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / PreRev",
            value: 0.0000140563269960694,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / PostRev",
            value: 0.000013940128992544489,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / BothRev",
            value: 0.000013183155009755864,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / PreRev",
            value: 0.000013264220004202798,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / PostRev",
            value: 0.00001313156500691548,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / BothRev",
            value: 0.000013169598998501898,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / PreRev",
            value: 0.000014051639009267092,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / PostRev",
            value: 0.000013998604990774766,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / BothRev",
            value: 0.00001392292301170528,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / PreRev",
            value: 0.00001396649400703609,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / PostRev",
            value: 0.000013920973986387252,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / BothRev",
            value: 0.000014042658993275835,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / tpu / Primal",
            value: 0.0001503131069998,
            unit: "s",
          },
          {
            name: "slicing / Jax / tpu / Primal",
            value: 0.0001487317569999,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / tpu / Primal",
            value: 0.0001488214770001,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / tpu / Primal",
            value: 0.0001482235870003,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / tpu / Primal",
            value: 0.0001465263969998,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / tpu / Primal",
            value: 0.000145185108,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / tpu / Primal",
            value: 0.0001457494779997,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / tpu / Forward",
            value: 0.0002250549559998,
            unit: "s",
          },
          {
            name: "slicing / Jax / tpu / Forward",
            value: 0.0002279735050001,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / tpu / Forward",
            value: 0.0002271572049999,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / tpu / Forward",
            value: 0.0002214052950002,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / tpu / Forward",
            value: 0.0002211182649998,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / tpu / Forward",
            value: 0.0002270314659999,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / tpu / Forward",
            value: 0.0002268825350001,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / tpu / PreRev",
            value: 0.0002229706749999,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / tpu / PostRev",
            value: 0.000192515316,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / tpu / BothRev",
            value: 0.0001920040260001,
            unit: "s",
          },
          {
            name: "slicing / Jax / tpu / BothRev",
            value: 0.0001957847960002,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / tpu / PreRev",
            value: 0.0002074272650002,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / tpu / PostRev",
            value: 0.0002152483750001,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / tpu / BothRev",
            value: 0.0002210917549996,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / tpu / PreRev",
            value: 0.0002221924949999,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / tpu / PostRev",
            value: 0.0002217396549999,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / tpu / BothRev",
            value: 0.0002096601459998,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / tpu / PreRev",
            value: 0.0002214959460002,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / tpu / PostRev",
            value: 0.0002284124450002,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / tpu / BothRev",
            value: 0.0002306672250001,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / tpu / PreRev",
            value: 0.0002274537460002,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / tpu / PostRev",
            value: 0.000195078096,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / tpu / BothRev",
            value: 0.0001949381250001,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / tpu / PreRev",
            value: 0.0001937442760004,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / tpu / PostRev",
            value: 0.0001946673259999,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / tpu / BothRev",
            value: 0.0002207158859996,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / Primal",
            value: 0.000006328735000352026,
            unit: "s",
          },
          {
            name: "slicing / Jax / cpu / Primal",
            value: 0.000006654476999756298,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / Primal",
            value: 0.000006620178999583004,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / Primal",
            value: 0.000006705037999381602,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / Primal",
            value: 0.0000065799520007203685,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / Primal",
            value: 0.000006322240000372404,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / Primal",
            value: 0.000006132002000413195,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / Forward",
            value: 0.000009667764999903738,
            unit: "s",
          },
          {
            name: "slicing / Jax / cpu / Forward",
            value: 0.000009109832000831374,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / Forward",
            value: 0.00000968324999939796,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / Forward",
            value: 0.000009566531999553264,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / Forward",
            value: 0.000009673610999925588,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / Forward",
            value: 0.00000955782100027136,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / Forward",
            value: 0.0000089979519998451,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / PreRev",
            value: 0.000009603483999853778,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / PostRev",
            value: 0.00000952534100088087,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / BothRev",
            value: 0.000010180781999224563,
            unit: "s",
          },
          {
            name: "slicing / Jax / cpu / BothRev",
            value: 0.000010202786999798263,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / PreRev",
            value: 0.00001012062599966157,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / PostRev",
            value: 0.000010191186999691129,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / BothRev",
            value: 0.00001025659400056611,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / PreRev",
            value: 0.000009662026999649242,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / PostRev",
            value: 0.00000949299500007328,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / BothRev",
            value: 0.000009607560000404192,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / PreRev",
            value: 0.00001022415900024498,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / PostRev",
            value: 0.000010140983999917809,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / BothRev",
            value: 0.000009657753999817942,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / PreRev",
            value: 0.000009705853999548708,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / PostRev",
            value: 0.00001030532700042386,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / BothRev",
            value: 0.00001010275799944793,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / PreRev",
            value: 0.0000096181990002151,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / PostRev",
            value: 0.000010216986000159522,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / BothRev",
            value: 0.000010231706999547896,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / Primal",
            value: 0.0000034010000017588022,
            unit: "s",
          },
          {
            name: "slicing / Jax / cpu / Primal",
            value: 0.000003369457997905556,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / Primal",
            value: 0.000003547999996953877,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / Primal",
            value: 0.000003424334001465468,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / Primal",
            value: 0.0000034315000011702066,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / Primal",
            value: 0.000003421459001401672,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / Primal",
            value: 0.000003391000002011424,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / Forward",
            value: 0.000005313166999258101,
            unit: "s",
          },
          {
            name: "slicing / Jax / cpu / Forward",
            value: 0.000005730416000005789,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / Forward",
            value: 0.00000533079200249631,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / Forward",
            value: 0.000005701333997421898,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / Forward",
            value: 0.00000565616600215435,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / Forward",
            value: 0.000005597457999101608,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / Forward",
            value: 0.000005605166999885114,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / PreRev",
            value: 0.000005958292000286747,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / PostRev",
            value: 0.000006272041999181965,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / BothRev",
            value: 0.000005607458002486964,
            unit: "s",
          },
          {
            name: "slicing / Jax / cpu / BothRev",
            value: 0.000005832792001456255,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / PreRev",
            value: 0.000005795042001409456,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / PostRev",
            value: 0.0000061885420000180605,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / BothRev",
            value: 0.000005691250000381842,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / PreRev",
            value: 0.000006468750001658918,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / PostRev",
            value: 0.000006163041998661356,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / BothRev",
            value: 0.00000578720900011831,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / PreRev",
            value: 0.000005605000002105953,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / PostRev",
            value: 0.000005658834001224022,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / BothRev",
            value: 0.00000604508299875306,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / PreRev",
            value: 0.000006371915998897748,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / PostRev",
            value: 0.000005950875001872191,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / BothRev",
            value: 0.000005838125001901062,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / PreRev",
            value: 0.000005573458001890685,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / PostRev",
            value: 0.000006016667000949383,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / BothRev",
            value: 0.000005924790999415563,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / Primal",
            value: 0.00000582792499972129,
            unit: "s",
          },
          {
            name: "sum    / Jax / cpu / Primal",
            value: 0.000005779252000138513,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / Primal",
            value: 0.000006157619999612507,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / Primal",
            value: 0.000005819134999910602,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / Primal",
            value: 0.000005832592999922781,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / Primal",
            value: 0.000006105725999987044,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / Primal",
            value: 0.000006104551000134961,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / Forward",
            value: 0.000009541117000026132,
            unit: "s",
          },
          {
            name: "sum    / Jax / cpu / Forward",
            value: 0.000009562277999975776,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / Forward",
            value: 0.00000923549599974649,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / Forward",
            value: 0.00000966369300022052,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / Forward",
            value: 0.000009766863000095329,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / Forward",
            value: 0.000009493703999851276,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / Forward",
            value: 0.000009581838000030984,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / PreRev",
            value: 0.00000891037100018366,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / PostRev",
            value: 0.00000845824900034131,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / BothRev",
            value: 0.000008440868999969098,
            unit: "s",
          },
          {
            name: "sum    / Jax / cpu / BothRev",
            value: 0.000008539266000298084,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / PreRev",
            value: 0.000008489098000154627,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / PostRev",
            value: 0.000008860532000198873,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / BothRev",
            value: 0.000008550056999865774,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / PreRev",
            value: 0.000008529213999736384,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / PostRev",
            value: 0.000008417457000177819,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / BothRev",
            value: 0.000008503115000166872,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / PreRev",
            value: 0.000008421595000072557,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / PostRev",
            value: 0.000008492795000165643,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / BothRev",
            value: 0.000008502353000039875,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / PreRev",
            value: 0.000008552420999876632,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / PostRev",
            value: 0.000008512200000041048,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / BothRev",
            value: 0.000008495992999996816,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / PreRev",
            value: 0.000008533067999906053,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / PostRev",
            value: 0.000009103969000079814,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / BothRev",
            value: 0.000008480823000354576,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / Primal",
            value: 0.000011627061991021038,
            unit: "s",
          },
          {
            name: "sum    / Jax / cpu / Primal",
            value: 0.00001158988001407124,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / Primal",
            value: 0.0000115897009964101,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / Primal",
            value: 0.000011607607011683286,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / Primal",
            value: 0.000012062888010405003,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / Primal",
            value: 0.0000120586930133868,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / Primal",
            value: 0.00001197013899218291,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / Forward",
            value: 0.00001797762399655767,
            unit: "s",
          },
          {
            name: "sum    / Jax / cpu / Forward",
            value: 0.000017103246005717664,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / Forward",
            value: 0.000017887927999254315,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / Forward",
            value: 0.000017905629007145763,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / Forward",
            value: 0.000017921953985933215,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / Forward",
            value: 0.000017866659996798263,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / Forward",
            value: 0.00001719719701213762,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / PreRev",
            value: 0.000016605113982222974,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / PostRev",
            value: 0.000015728647995274513,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / BothRev",
            value: 0.000016501956997672097,
            unit: "s",
          },
          {
            name: "sum    / Jax / cpu / BothRev",
            value: 0.000015740517002996056,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / PreRev",
            value: 0.000016594941000221297,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / PostRev",
            value: 0.000016550461994484067,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / BothRev",
            value: 0.000015731127990875394,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / PreRev",
            value: 0.000016616352018900217,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / PostRev",
            value: 0.000016644927003653722,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / BothRev",
            value: 0.000015725104982266203,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / PreRev",
            value: 0.000016546169004868714,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / PostRev",
            value: 0.000016687424009433017,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / BothRev",
            value: 0.000016546505998121574,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / PreRev",
            value: 0.000015666503983084112,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / PostRev",
            value: 0.00001657996201538481,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / BothRev",
            value: 0.000016517870972165837,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / PreRev",
            value: 0.000015704051998909563,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / PostRev",
            value: 0.00001660291198641062,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / BothRev",
            value: 0.000015802836016518995,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / tpu / Primal",
            value: 0.0001509564770003,
            unit: "s",
          },
          {
            name: "sum    / Jax / tpu / Primal",
            value: 0.0001522014870001,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / tpu / Primal",
            value: 0.0001517179470001,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / tpu / Primal",
            value: 0.0001502100669999,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / tpu / Primal",
            value: 0.0001500868369998,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / tpu / Primal",
            value: 0.0001500946169999,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / tpu / Primal",
            value: 0.0001492328569997,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / tpu / Forward",
            value: 0.0002216534859999,
            unit: "s",
          },
          {
            name: "sum    / Jax / tpu / Forward",
            value: 0.0001938896589999,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / tpu / Forward",
            value: 0.0001943110689999,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / tpu / Forward",
            value: 0.000198089469,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / tpu / Forward",
            value: 0.0002042402189999,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / tpu / Forward",
            value: 0.000200562209,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / tpu / Forward",
            value: 0.0002064357889998,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / tpu / PreRev",
            value: 0.0002197176489999,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / tpu / PostRev",
            value: 0.0002192902179999,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / tpu / BothRev",
            value: 0.0002308240389997,
            unit: "s",
          },
          {
            name: "sum    / Jax / tpu / BothRev",
            value: 0.0002290537490002,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / tpu / PreRev",
            value: 0.000227361099,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / tpu / PostRev",
            value: 0.0002245474579999,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / tpu / BothRev",
            value: 0.000199757328,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / tpu / PreRev",
            value: 0.0002001333590001,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / tpu / PostRev",
            value: 0.0001982346089998,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / tpu / BothRev",
            value: 0.0001994997289998,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / tpu / PreRev",
            value: 0.0002228434990001,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / tpu / PostRev",
            value: 0.0002238076989997,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / tpu / BothRev",
            value: 0.000204772819,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / tpu / PreRev",
            value: 0.0002239885789999,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / tpu / PostRev",
            value: 0.0002219428589996,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / tpu / BothRev",
            value: 0.0002097797289998,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / tpu / PreRev",
            value: 0.0002037243090003,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / tpu / PostRev",
            value: 0.0002221279690002,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / tpu / BothRev",
            value: 0.0002092639789998,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / Primal",
            value: 0.000008657596000375634,
            unit: "s",
          },
          {
            name: "sum    / Jax / cpu / Primal",
            value: 0.000008635459000288393,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / Primal",
            value: 0.000008169425999767555,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / Primal",
            value: 0.000008214720000069065,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / Primal",
            value: 0.000008608382000602433,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / Primal",
            value: 0.000008226093999837759,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / Primal",
            value: 0.00000871672899938858,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / Forward",
            value: 0.000012545626999781236,
            unit: "s",
          },
          {
            name: "sum    / Jax / cpu / Forward",
            value: 0.000012732608000078472,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / Forward",
            value: 0.000012616226999853098,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / Forward",
            value: 0.000012594323999110202,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / Forward",
            value: 0.00001193045699983486,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / Forward",
            value: 0.000012623351000002003,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / Forward",
            value: 0.00001260007200016844,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / PreRev",
            value: 0.00001179207499990298,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / PostRev",
            value: 0.00001116703099978622,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / BothRev",
            value: 0.000011060855999858175,
            unit: "s",
          },
          {
            name: "sum    / Jax / cpu / BothRev",
            value: 0.000011155931000757846,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / PreRev",
            value: 0.000011715493999872706,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / PostRev",
            value: 0.0000117892700000084,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / BothRev",
            value: 0.00001109084699965024,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / PreRev",
            value: 0.000011755823999919812,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / PostRev",
            value: 0.00001169062099961593,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / BothRev",
            value: 0.000011272420999375752,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / PreRev",
            value: 0.00001193658099964523,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / PostRev",
            value: 0.000011831782999252029,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / BothRev",
            value: 0.000011836599999696774,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / PreRev",
            value: 0.000011290987999927892,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / PostRev",
            value: 0.000011810508000053233,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / BothRev",
            value: 0.000011811832000603318,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / PreRev",
            value: 0.000011197929000445584,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / PostRev",
            value: 0.000011956716999520722,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / BothRev",
            value: 0.00001176464199943439,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / Primal",
            value: 0.000004790833998413291,
            unit: "s",
          },
          {
            name: "sum    / Jax / cpu / Primal",
            value: 0.000005237584002315998,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / Primal",
            value: 0.000004616624999471241,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / Primal",
            value: 0.000004448042000149144,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / Primal",
            value: 0.000004438208001374732,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / Primal",
            value: 0.000004441459001100156,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / Primal",
            value: 0.00000438541699986672,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / Forward",
            value: 0.000006919416999153327,
            unit: "s",
          },
          {
            name: "sum    / Jax / cpu / Forward",
            value: 0.000007014833998255199,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / Forward",
            value: 0.000007447290998243261,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / Forward",
            value: 0.000006834291001723613,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / Forward",
            value: 0.000007331666998652509,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / Forward",
            value: 0.000007072083000821294,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / Forward",
            value: 0.0000074379159996169615,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / PreRev",
            value: 0.0000067832080021617,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / PostRev",
            value: 0.000006214375000126892,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / BothRev",
            value: 0.0000068444169992289975,
            unit: "s",
          },
          {
            name: "sum    / Jax / cpu / BothRev",
            value: 0.000006652457999734906,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / PreRev",
            value: 0.000006428832999517908,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / PostRev",
            value: 0.000006212084001163021,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / BothRev",
            value: 0.0000067727919995377305,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / PreRev",
            value: 0.000006731041998136789,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / PostRev",
            value: 0.000006921832999069011,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / BothRev",
            value: 0.000007049792002362665,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / PreRev",
            value: 0.000006817499997850973,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / PostRev",
            value: 0.0000070250409989967015,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / BothRev",
            value: 0.0000065184999984921885,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / PreRev",
            value: 0.000006718208001984749,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / PostRev",
            value: 0.000006897082999785198,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / BothRev",
            value: 0.000006512541000120109,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / PreRev",
            value: 0.000006344417000946124,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / PostRev",
            value: 0.000006402167000487679,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / BothRev",
            value: 0.000006926292000571266,
            unit: "s",
          },
          {
            name: "value_and_grad / JaXPipe / cpu / Primal",
            value: 0.00001087284000004729,
            unit: "s",
          },
          {
            name: "value_and_grad / Jax / cpu / Primal",
            value: 0.000011377028999959294,
            unit: "s",
          },
          {
            name: "value_and_grad / HLOOpt / cpu / Primal",
            value: 0.000011180068000157916,
            unit: "s",
          },
          {
            name: "value_and_grad / PartOpt / cpu / Primal",
            value: 0.00001122047599983489,
            unit: "s",
          },
          {
            name: "value_and_grad / IPartOpt / cpu / Primal",
            value: 0.00001115287900029216,
            unit: "s",
          },
          {
            name: "value_and_grad / DefOpt / cpu / Primal",
            value: 0.00001113783400023749,
            unit: "s",
          },
          {
            name: "value_and_grad / IDefOpt / cpu / Primal",
            value: 0.000011358971999925416,
            unit: "s",
          },
          {
            name: "value_and_grad / JaXPipe / cpu / Primal",
            value: 0.00001675487699685618,
            unit: "s",
          },
          {
            name: "value_and_grad / Jax / cpu / Primal",
            value: 0.000016844860976561902,
            unit: "s",
          },
          {
            name: "value_and_grad / HLOOpt / cpu / Primal",
            value: 0.00001787239799159579,
            unit: "s",
          },
          {
            name: "value_and_grad / PartOpt / cpu / Primal",
            value: 0.00001771658897632733,
            unit: "s",
          },
          {
            name: "value_and_grad / IPartOpt / cpu / Primal",
            value: 0.000017942664999281988,
            unit: "s",
          },
          {
            name: "value_and_grad / DefOpt / cpu / Primal",
            value: 0.000017953653004951774,
            unit: "s",
          },
          {
            name: "value_and_grad / IDefOpt / cpu / Primal",
            value: 0.000016997869999613614,
            unit: "s",
          },
          {
            name: "value_and_grad / JaXPipe / tpu / Primal",
            value: 0.0002371486279998,
            unit: "s",
          },
          {
            name: "value_and_grad / Jax / tpu / Primal",
            value: 0.0002359236689999,
            unit: "s",
          },
          {
            name: "value_and_grad / HLOOpt / tpu / Primal",
            value: 0.0002310969889999,
            unit: "s",
          },
          {
            name: "value_and_grad / PartOpt / tpu / Primal",
            value: 0.0002334034289997,
            unit: "s",
          },
          {
            name: "value_and_grad / IPartOpt / tpu / Primal",
            value: 0.0002339196790003,
            unit: "s",
          },
          {
            name: "value_and_grad / DefOpt / tpu / Primal",
            value: 0.0002499673489996,
            unit: "s",
          },
          {
            name: "value_and_grad / IDefOpt / tpu / Primal",
            value: 0.0002324000579997,
            unit: "s",
          },
          {
            name: "value_and_grad / JaXPipe / cpu / Primal",
            value: 0.000012349168000582722,
            unit: "s",
          },
          {
            name: "value_and_grad / Jax / cpu / Primal",
            value: 0.000013249031000668766,
            unit: "s",
          },
          {
            name: "value_and_grad / HLOOpt / cpu / Primal",
            value: 0.00001303524400009337,
            unit: "s",
          },
          {
            name: "value_and_grad / PartOpt / cpu / Primal",
            value: 0.000012950233000083243,
            unit: "s",
          },
          {
            name: "value_and_grad / IPartOpt / cpu / Primal",
            value: 0.000013206902999627343,
            unit: "s",
          },
          {
            name: "value_and_grad / DefOpt / cpu / Primal",
            value: 0.000013213436000114598,
            unit: "s",
          },
          {
            name: "value_and_grad / IDefOpt / cpu / Primal",
            value: 0.000013262114999633923,
            unit: "s",
          },
          {
            name: "value_and_grad / JaXPipe / cpu / Primal",
            value: 0.0000077484169996751,
            unit: "s",
          },
          {
            name: "value_and_grad / Jax / cpu / Primal",
            value: 0.000008827207999274832,
            unit: "s",
          },
          {
            name: "value_and_grad / HLOOpt / cpu / Primal",
            value: 0.000008137958000588697,
            unit: "s",
          },
          {
            name: "value_and_grad / PartOpt / cpu / Primal",
            value: 0.00000822774999687681,
            unit: "s",
          },
          {
            name: "value_and_grad / IPartOpt / cpu / Primal",
            value: 0.000007993957999133272,
            unit: "s",
          },
          {
            name: "value_and_grad / DefOpt / cpu / Primal",
            value: 0.000008224041997891618,
            unit: "s",
          },
          {
            name: "value_and_grad / IDefOpt / cpu / Primal",
            value: 0.000008128416000545258,
            unit: "s",
          },
          {
            name: "jaxmd40 / JaXPipe / cpu / Primal",
            value: 0.0763881798018701,
            unit: "s",
          },
          {
            name: "jaxmd40 / Jax / cpu / Primal",
            value: 0.0757622490054927,
            unit: "s",
          },
          {
            name: "jaxmd40 / HLOOpt / cpu / Primal",
            value: 0.1164587243983987,
            unit: "s",
          },
          {
            name: "jaxmd40 / PartOpt / cpu / Primal",
            value: 0.0712570609990507,
            unit: "s",
          },
          {
            name: "jaxmd40 / IPartOpt / cpu / Primal",
            value: 0.0782381401979364,
            unit: "s",
          },
          {
            name: "jaxmd40 / DefOpt / cpu / Primal",
            value: 0.111502275604289,
            unit: "s",
          },
          {
            name: "jaxmd40 / IDefOpt / cpu / Primal",
            value: 0.116134642198449,
            unit: "s",
          },
          {
            name: "jaxmd40 / JaXPipe / cpu / Forward",
            value: 0.220891277003102,
            unit: "s",
          },
          {
            name: "jaxmd40 / Jax / cpu / Forward",
            value: 0.1097276521963067,
            unit: "s",
          },
          {
            name: "jaxmd40 / HLOOpt / cpu / Forward",
            value: 0.2251934297964908,
            unit: "s",
          },
          {
            name: "jaxmd40 / PartOpt / cpu / Forward",
            value: 0.2250653553986921,
            unit: "s",
          },
          {
            name: "jaxmd40 / IPartOpt / cpu / Forward",
            value: 0.2242989405989646,
            unit: "s",
          },
          {
            name: "jaxmd40 / DefOpt / cpu / Forward",
            value: 0.2225164085975848,
            unit: "s",
          },
          {
            name: "jaxmd40 / IDefOpt / cpu / Forward",
            value: 0.2236204168002586,
            unit: "s",
          },
          {
            name: "jaxmd40 / JaXPipe / cpu / PostRev",
            value: 0.1550642883987166,
            unit: "s",
          },
          {
            name: "jaxmd40 / Jax / cpu / BothRev",
            value: 0.1550167536013759,
            unit: "s",
          },
          {
            name: "jaxmd40 / HLOOpt / cpu / PostRev",
            value: 0.2166597702016588,
            unit: "s",
          },
          {
            name: "jaxmd40 / PartOpt / cpu / PostRev",
            value: 0.1505107364035211,
            unit: "s",
          },
          {
            name: "jaxmd40 / IPartOpt / cpu / PostRev",
            value: 0.1543377113994211,
            unit: "s",
          },
          {
            name: "jaxmd40 / DefOpt / cpu / PostRev",
            value: 0.2126652373990509,
            unit: "s",
          },
          {
            name: "jaxmd40 / IDefOpt / cpu / PostRev",
            value: 0.2128927405981812,
            unit: "s",
          },
          {
            name: "jaxmd40 / JaXPipe / tpu / Primal",
            value: 0.0092324397999618,
            unit: "s",
          },
          {
            name: "jaxmd40 / Jax / tpu / Primal",
            value: 0.0092246157999397,
            unit: "s",
          },
          {
            name: "jaxmd40 / HLOOpt / tpu / Primal",
            value: 0.0091860677999648,
            unit: "s",
          },
          {
            name: "jaxmd40 / PartOpt / tpu / Primal",
            value: 0.0092136797999046,
            unit: "s",
          },
          {
            name: "jaxmd40 / IPartOpt / tpu / Primal",
            value: 0.0092184437999094,
            unit: "s",
          },
          {
            name: "jaxmd40 / DefOpt / tpu / Primal",
            value: 0.0090754378001292,
            unit: "s",
          },
          {
            name: "jaxmd40 / IDefOpt / tpu / Primal",
            value: 0.0090245017998313,
            unit: "s",
          },
          {
            name: "jaxmd40 / JaXPipe / tpu / Forward",
            value: 0.0178596076000758,
            unit: "s",
          },
          {
            name: "jaxmd40 / Jax / tpu / Forward",
            value: 0.0182861155999489,
            unit: "s",
          },
          {
            name: "jaxmd40 / HLOOpt / tpu / Forward",
            value: 0.0178321156001402,
            unit: "s",
          },
          {
            name: "jaxmd40 / PartOpt / tpu / Forward",
            value: 0.0178475835999051,
            unit: "s",
          },
          {
            name: "jaxmd40 / IPartOpt / tpu / Forward",
            value: 0.0178373435999674,
            unit: "s",
          },
          {
            name: "jaxmd40 / DefOpt / tpu / Forward",
            value: 0.0178410836000693,
            unit: "s",
          },
          {
            name: "jaxmd40 / IDefOpt / tpu / Forward",
            value: 0.0178766118000567,
            unit: "s",
          },
          {
            name: "jaxmd40 / JaXPipe / tpu / PostRev",
            value: 0.0196946856000067,
            unit: "s",
          },
          {
            name: "jaxmd40 / Jax / tpu / BothRev",
            value: 0.0196892136000315,
            unit: "s",
          },
          {
            name: "jaxmd40 / HLOOpt / tpu / PostRev",
            value: 0.0188514675999613,
            unit: "s",
          },
          {
            name: "jaxmd40 / PartOpt / tpu / PostRev",
            value: 0.0197023818000161,
            unit: "s",
          },
          {
            name: "jaxmd40 / IPartOpt / tpu / PostRev",
            value: 0.0197024035998765,
            unit: "s",
          },
          {
            name: "jaxmd40 / DefOpt / tpu / PostRev",
            value: 0.0183479957999225,
            unit: "s",
          },
          {
            name: "jaxmd40 / IDefOpt / tpu / PostRev",
            value: 0.0181628137999723,
            unit: "s",
          },
          {
            name: "jaxmd40 / JaXPipe / cpu / Primal",
            value: 0.0597299061999365,
            unit: "s",
          },
          {
            name: "jaxmd40 / Jax / cpu / Primal",
            value: 0.0610853500000303,
            unit: "s",
          },
          {
            name: "jaxmd40 / HLOOpt / cpu / Primal",
            value: 0.0869948076000582,
            unit: "s",
          },
          {
            name: "jaxmd40 / PartOpt / cpu / Primal",
            value: 0.0605256855998959,
            unit: "s",
          },
          {
            name: "jaxmd40 / IPartOpt / cpu / Primal",
            value: 0.0588132189999669,
            unit: "s",
          },
          {
            name: "jaxmd40 / DefOpt / cpu / Primal",
            value: 0.0852582374000121,
            unit: "s",
          },
          {
            name: "jaxmd40 / IDefOpt / cpu / Primal",
            value: 0.0864675282000462,
            unit: "s",
          },
          {
            name: "jaxmd40 / JaXPipe / cpu / Forward",
            value: 0.1622040072001254,
            unit: "s",
          },
          {
            name: "jaxmd40 / Jax / cpu / Forward",
            value: 0.0845224167998821,
            unit: "s",
          },
          {
            name: "jaxmd40 / HLOOpt / cpu / Forward",
            value: 0.1656225909999193,
            unit: "s",
          },
          {
            name: "jaxmd40 / PartOpt / cpu / Forward",
            value: 0.1650651973999629,
            unit: "s",
          },
          {
            name: "jaxmd40 / IPartOpt / cpu / Forward",
            value: 0.1624807950000104,
            unit: "s",
          },
          {
            name: "jaxmd40 / DefOpt / cpu / Forward",
            value: 0.1642970896000406,
            unit: "s",
          },
          {
            name: "jaxmd40 / IDefOpt / cpu / Forward",
            value: 0.1655470155999864,
            unit: "s",
          },
          {
            name: "jaxmd40 / JaXPipe / cpu / PostRev",
            value: 0.13116981080002,
            unit: "s",
          },
          {
            name: "jaxmd40 / Jax / cpu / BothRev",
            value: 0.1272120947998701,
            unit: "s",
          },
          {
            name: "jaxmd40 / HLOOpt / cpu / PostRev",
            value: 0.1669722982000166,
            unit: "s",
          },
          {
            name: "jaxmd40 / PartOpt / cpu / PostRev",
            value: 0.1237660423999841,
            unit: "s",
          },
          {
            name: "jaxmd40 / IPartOpt / cpu / PostRev",
            value: 0.128555215400047,
            unit: "s",
          },
          {
            name: "jaxmd40 / DefOpt / cpu / PostRev",
            value: 0.1666630058000009,
            unit: "s",
          },
          {
            name: "jaxmd40 / IDefOpt / cpu / PostRev",
            value: 0.1602140441998926,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / JaXPipe / cpu / Primal",
            value: 51.98808322599507,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / Jax / cpu / Primal",
            value: 51.83015386600164,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / HLOOpt / cpu / Primal",
            value: 54.23473284300417,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / PartOpt / cpu / Primal",
            value: 53.28872225398664,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / IPartOpt / cpu / Primal",
            value: 50.39228341399576,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / DefOpt / cpu / Primal",
            value: 25.85711625200929,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / IDefOpt / cpu / Primal",
            value: 56.587452497013146,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / JaXPipe / tpu / Primal",
            value: 0.1900890269998854,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / Jax / tpu / Primal",
            value: 0.1900437169997531,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / HLOOpt / tpu / Primal",
            value: 0.1886653870001282,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / PartOpt / tpu / Primal",
            value: 0.2028727169999911,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / IPartOpt / tpu / Primal",
            value: 0.2029150460002711,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / DefOpt / tpu / Primal",
            value: 0.1739375349998226,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / IDefOpt / tpu / Primal",
            value: 0.1850887249997868,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / JaXPipe / cpu / Primal",
            value: 56.19909158200062,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / Jax / cpu / Primal",
            value: 54.35408105399984,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / HLOOpt / cpu / Primal",
            value: 53.51094949600065,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / PartOpt / cpu / Primal",
            value: 54.06253522400039,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / IPartOpt / cpu / Primal",
            value: 54.70657927100001,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / DefOpt / cpu / Primal",
            value: 25.169169198999956,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / IDefOpt / cpu / Primal",
            value: 60.56553733800047,
            unit: "s",
          },
        ],
      },
      {
        commit: {
          author: {
            email: "avikpal@mit.edu",
            name: "Avik Pal",
            username: "avik-pal",
          },
          committer: {
            email: "noreply@github.com",
            name: "GitHub",
            username: "web-flow",
          },
          distinct: true,
          id: "db12faea0aafbf82dbeb82d953f0841403054898",
          message: "fix: dot_general batch dims removal pass (#1809)",
          timestamp: "2025-12-19T17:28:21-05:00",
          tree_id: "2b36a27dc685e1856bbaa248c935caf06f59abd4",
          url: "https://github.com/EnzymeAD/Enzyme-JAX/commit/db12faea0aafbf82dbeb82d953f0841403054898",
        },
        date: 1766194717383,
        tool: "customSmallerIsBetter",
        benches: [
          {
            name: "actmtch / JaXPipe / cpu / Primal",
            value: 0.000004620638000233157,
            unit: "s",
          },
          {
            name: "actmtch / Jax / cpu / Primal",
            value: 0.000004891044999567385,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / Primal",
            value: 0.000005309150999892154,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / Primal",
            value: 0.0000048381390006397855,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / Primal",
            value: 0.000004698155999903974,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / Primal",
            value: 0.000005308740000145917,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / Primal",
            value: 0.000005303920999722322,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / Forward",
            value: 0.000008137297999383008,
            unit: "s",
          },
          {
            name: "actmtch / Jax / cpu / Forward",
            value: 0.000007059812999614223,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / Forward",
            value: 0.000008252759999777482,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / Forward",
            value: 0.000008262581999588291,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / Forward",
            value: 0.00000828591299978143,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / Forward",
            value: 0.000008088560000032885,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / Forward",
            value: 0.000008410486000684615,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / PreRev",
            value: 0.000008308346999911009,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / PostRev",
            value: 0.000007240177000312542,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / BothRev",
            value: 0.00000827742199999193,
            unit: "s",
          },
          {
            name: "actmtch / Jax / cpu / BothRev",
            value: 0.000007291795000128331,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / PreRev",
            value: 0.000008205689000533312,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / PostRev",
            value: 0.000008192248999876028,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / BothRev",
            value: 0.00000828699100020458,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / PreRev",
            value: 0.000008451945000160776,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / PostRev",
            value: 0.000007179473999713082,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / BothRev",
            value: 0.000008183551999536576,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / PreRev",
            value: 0.000008254278000094928,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / PostRev",
            value: 0.000007308825999643887,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / BothRev",
            value: 0.000008199353000236442,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / PreRev",
            value: 0.000008221637000133342,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / PostRev",
            value: 0.00000820537300023716,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / BothRev",
            value: 0.00000818224199974793,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / PreRev",
            value: 0.000008207287999539403,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / PostRev",
            value: 0.000008236804999796732,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / BothRev",
            value: 0.000008200846999898203,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / Primal",
            value: 0.000008734886998354341,
            unit: "s",
          },
          {
            name: "actmtch / Jax / cpu / Primal",
            value: 0.000009042974999829312,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / Primal",
            value: 0.00001113410799916892,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / Primal",
            value: 0.00000886992400046438,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / Primal",
            value: 0.000009019646000524517,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / Primal",
            value: 0.000011367694000000484,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / Primal",
            value: 0.000011334370999975363,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / Forward",
            value: 0.000015556487000139896,
            unit: "s",
          },
          {
            name: "actmtch / Jax / cpu / Forward",
            value: 0.000013411530000666971,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / Forward",
            value: 0.00001601385600042704,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / Forward",
            value: 0.00001656832799926633,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / Forward",
            value: 0.000016490194999278175,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / Forward",
            value: 0.000015646142999685253,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / Forward",
            value: 0.000016577461999986554,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / PreRev",
            value: 0.000015816614000868868,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / PostRev",
            value: 0.000012814002999220977,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / BothRev",
            value: 0.000016588445998422683,
            unit: "s",
          },
          {
            name: "actmtch / Jax / cpu / BothRev",
            value: 0.00001287999200030754,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / PreRev",
            value: 0.000015912331000436098,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / PostRev",
            value: 0.000015707044000009772,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / BothRev",
            value: 0.000016721346999474918,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / PreRev",
            value: 0.00001600183199843741,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / PostRev",
            value: 0.00001347799799987115,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / BothRev",
            value: 0.000015744524000183447,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / PreRev",
            value: 0.000015765755000757052,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / PostRev",
            value: 0.000013472967999405227,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / BothRev",
            value: 0.000016606338000201504,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / PreRev",
            value: 0.000016596101999311942,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / PostRev",
            value: 0.00001650141100071778,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / BothRev",
            value: 0.000016588150998359196,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / PreRev",
            value: 0.000015999622000890667,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / PostRev",
            value: 0.00001648970099995495,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / BothRev",
            value: 0.000016629068999463925,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / tpu / Primal",
            value: 0.0001338551279986,
            unit: "s",
          },
          {
            name: "actmtch / Jax / tpu / Primal",
            value: 0.0001318973080014,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / tpu / Primal",
            value: 0.0001297989979975,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / tpu / Primal",
            value: 0.0001291784879977,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / tpu / Primal",
            value: 0.000129397667999,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / tpu / Primal",
            value: 0.0001298391279997,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / tpu / Primal",
            value: 0.0001303027889989,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / tpu / Forward",
            value: 0.0002055508779994,
            unit: "s",
          },
          {
            name: "actmtch / Jax / tpu / Forward",
            value: 0.0002051347070009,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / tpu / Forward",
            value: 0.0002176808570002,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / tpu / Forward",
            value: 0.000199516957,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / tpu / Forward",
            value: 0.0001931917270012,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / tpu / Forward",
            value: 0.0002003879870026,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / tpu / Forward",
            value: 0.0002157127570026,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / tpu / PreRev",
            value: 0.0001977403780001,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / tpu / PostRev",
            value: 0.0002093044769972,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / tpu / BothRev",
            value: 0.0001965611270024,
            unit: "s",
          },
          {
            name: "actmtch / Jax / tpu / BothRev",
            value: 0.0001956047579988,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / tpu / PreRev",
            value: 0.0001990658670001,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / tpu / PostRev",
            value: 0.0002007510069997,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / tpu / BothRev",
            value: 0.0001951324869987,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / tpu / PreRev",
            value: 0.0002058646869991,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / tpu / PostRev",
            value: 0.0001942864770026,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / tpu / BothRev",
            value: 0.0001960553569988,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / tpu / PreRev",
            value: 0.0002111983880022,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / tpu / PostRev",
            value: 0.000219494036999,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / tpu / BothRev",
            value: 0.0002059149179985,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / tpu / PreRev",
            value: 0.0002124367979995,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / tpu / PostRev",
            value: 0.0002021598969986,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / tpu / BothRev",
            value: 0.0002067862770018,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / tpu / PreRev",
            value: 0.0002074388469991,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / tpu / PostRev",
            value: 0.000197319138002,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / tpu / BothRev",
            value: 0.0001974999670019,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / Primal",
            value: 0.000006127124000158801,
            unit: "s",
          },
          {
            name: "actmtch / Jax / cpu / Primal",
            value: 0.000006255105000491312,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / Primal",
            value: 0.000007395460000225285,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / Primal",
            value: 0.000006663394000497647,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / Primal",
            value: 0.0000059997120006300975,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / Primal",
            value: 0.000007424484999319248,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / Primal",
            value: 0.000007408223000311409,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / Forward",
            value: 0.000011402357000406485,
            unit: "s",
          },
          {
            name: "actmtch / Jax / cpu / Forward",
            value: 0.000009020212999530486,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / Forward",
            value: 0.00001150231899919163,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / Forward",
            value: 0.000011033957000108783,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / Forward",
            value: 0.000011553308000657123,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / Forward",
            value: 0.000011294345000351311,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / Forward",
            value: 0.0000114600220003922,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / PreRev",
            value: 0.000010898002999965685,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / PostRev",
            value: 0.00000908278600036283,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / BothRev",
            value: 0.000011559665000277164,
            unit: "s",
          },
          {
            name: "actmtch / Jax / cpu / BothRev",
            value: 0.000009504798999842025,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / PreRev",
            value: 0.000011554783000065073,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / PostRev",
            value: 0.00001151557400044112,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / BothRev",
            value: 0.000010966456000460312,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / PreRev",
            value: 0.00001146247700035019,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / PostRev",
            value: 0.000009614304000024276,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / BothRev",
            value: 0.000010892768999838154,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / PreRev",
            value: 0.000011569100000087928,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / PostRev",
            value: 0.000009636468999815406,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / BothRev",
            value: 0.000011554513999726624,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / PreRev",
            value: 0.000011509683000440418,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / PostRev",
            value: 0.0000115182769995954,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / BothRev",
            value: 0.000011486948999845482,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / PreRev",
            value: 0.000011467074999927718,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / PostRev",
            value: 0.000011527467000632895,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / BothRev",
            value: 0.000011540206999598013,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / Primal",
            value: 0.000004260082999962833,
            unit: "s",
          },
          {
            name: "actmtch / Jax / cpu / Primal",
            value: 0.000004364666999890688,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / Primal",
            value: 0.000006041625000079875,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / Primal",
            value: 0.000004154624999955559,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / Primal",
            value: 0.000004520666999951573,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / Primal",
            value: 0.000005251374999943437,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / Primal",
            value: 0.000004897416999938287,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / Forward",
            value: 0.000008046499999863955,
            unit: "s",
          },
          {
            name: "actmtch / Jax / cpu / Forward",
            value: 0.000006633292000060465,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / Forward",
            value: 0.000009273542000073577,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / Forward",
            value: 0.000007782750000160377,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / Forward",
            value: 0.000007216041999981826,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / Forward",
            value: 0.000008807666999928187,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / Forward",
            value: 0.00000714654199987308,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / PreRev",
            value: 0.000007312916000046243,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / PostRev",
            value: 0.000006329834000098344,
            unit: "s",
          },
          {
            name: "actmtch / JaXPipe / cpu / BothRev",
            value: 0.000007275666999930763,
            unit: "s",
          },
          {
            name: "actmtch / Jax / cpu / BothRev",
            value: 0.000006355125000027328,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / PreRev",
            value: 0.000009794374999955836,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / PostRev",
            value: 0.000009648624999954336,
            unit: "s",
          },
          {
            name: "actmtch / HLOOpt / cpu / BothRev",
            value: 0.000008435292000058325,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / PreRev",
            value: 0.00000904587500008347,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / PostRev",
            value: 0.0000071613329998854166,
            unit: "s",
          },
          {
            name: "actmtch / PartOpt / cpu / BothRev",
            value: 0.000009698332999960256,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / PreRev",
            value: 0.00001002599999992526,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / PostRev",
            value: 0.000007277959000020928,
            unit: "s",
          },
          {
            name: "actmtch / IPartOpt / cpu / BothRev",
            value: 0.000008261417000085203,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / PreRev",
            value: 0.000007404791999988447,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / PostRev",
            value: 0.000007266583000046012,
            unit: "s",
          },
          {
            name: "actmtch / DefOpt / cpu / BothRev",
            value: 0.00000733695799999623,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / PreRev",
            value: 0.000008988874999886321,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / PostRev",
            value: 0.000008107458000040424,
            unit: "s",
          },
          {
            name: "actmtch / IDefOpt / cpu / BothRev",
            value: 0.000007504165999989709,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / Primal",
            value: 0.000004870925999966858,
            unit: "s",
          },
          {
            name: "add_one / Jax / cpu / Primal",
            value: 0.000004937332000736206,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / Primal",
            value: 0.000004945814999700815,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / Primal",
            value: 0.000004913056000077632,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / Primal",
            value: 0.000004919043999507267,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / Primal",
            value: 0.000004938097999911406,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / Primal",
            value: 0.000004869223999776296,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / Forward",
            value: 0.000007972825999786436,
            unit: "s",
          },
          {
            name: "add_one / Jax / cpu / Forward",
            value: 0.000008002546999705373,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / Forward",
            value: 0.000008040957000048366,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / Forward",
            value: 0.00000801773500006675,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / Forward",
            value: 0.000008011572000214074,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / Forward",
            value: 0.000007971247000568838,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / Forward",
            value: 0.000008033963999878325,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / PreRev",
            value: 0.000008764621000409534,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / PostRev",
            value: 0.000008873015000062878,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / BothRev",
            value: 0.00000883551799961424,
            unit: "s",
          },
          {
            name: "add_one / Jax / cpu / BothRev",
            value: 0.000008909338000194111,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / PreRev",
            value: 0.000008932671000366099,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / PostRev",
            value: 0.000008786121999946771,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / BothRev",
            value: 0.000008967644999756885,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / PreRev",
            value: 0.00000899492500047927,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / PostRev",
            value: 0.000009119599000769086,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / BothRev",
            value: 0.000008788467000158562,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / PreRev",
            value: 0.00000885931300035736,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / PostRev",
            value: 0.00000890276999962225,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / BothRev",
            value: 0.000008829846000480756,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / PreRev",
            value: 0.00000895777799996722,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / PostRev",
            value: 0.000008896549999917625,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / BothRev",
            value: 0.000008811028999843983,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / PreRev",
            value: 0.00000879401699967275,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / PostRev",
            value: 0.000009342379000372604,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / BothRev",
            value: 0.000008824539000670484,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / Primal",
            value: 0.000009181630000966836,
            unit: "s",
          },
          {
            name: "add_one / Jax / cpu / Primal",
            value: 0.000009217023000019253,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / Primal",
            value: 0.000009616833000109182,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / Primal",
            value: 0.000009604421000403818,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / Primal",
            value: 0.000009681687999545827,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / Primal",
            value: 0.000009944731000359751,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / Primal",
            value: 0.000008943126000303892,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / Forward",
            value: 0.00001387925500057463,
            unit: "s",
          },
          {
            name: "add_one / Jax / cpu / Forward",
            value: 0.00001391765900007158,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / Forward",
            value: 0.000014510338998661607,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / Forward",
            value: 0.000018101228999512385,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / Forward",
            value: 0.000016649974000756628,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / Forward",
            value: 0.000014577463000023272,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / Forward",
            value: 0.000014569343000403024,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / PreRev",
            value: 0.000014852663998681237,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / PostRev",
            value: 0.00001562761500099441,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / BothRev",
            value: 0.000015602269999362763,
            unit: "s",
          },
          {
            name: "add_one / Jax / cpu / BothRev",
            value: 0.000015661870998883388,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / PreRev",
            value: 0.000015590111001074548,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / PostRev",
            value: 0.00001544977000048675,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / BothRev",
            value: 0.00001567074699960358,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / PreRev",
            value: 0.000014718423000886106,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / PostRev",
            value: 0.0000156317470009526,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / BothRev",
            value: 0.000014725218001331089,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / PreRev",
            value: 0.00001472372100033681,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / PostRev",
            value: 0.00001558830000067246,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / BothRev",
            value: 0.000015600835999066475,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / PreRev",
            value: 0.000014777169999433682,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / PostRev",
            value: 0.000015564503000859987,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / BothRev",
            value: 0.000015617901999576135,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / PreRev",
            value: 0.000015492862001337927,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / PostRev",
            value: 0.000015473253999516602,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / BothRev",
            value: 0.000015473588999157074,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / tpu / Primal",
            value: 0.0001348643579985,
            unit: "s",
          },
          {
            name: "add_one / Jax / tpu / Primal",
            value: 0.0001327233980009,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / tpu / Primal",
            value: 0.0001325530389985,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / tpu / Primal",
            value: 0.0001323904079981,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / tpu / Primal",
            value: 0.0001347304279988,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / tpu / Primal",
            value: 0.0001352511779987,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / tpu / Primal",
            value: 0.0001300315879998,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / tpu / Forward",
            value: 0.0001846602279983,
            unit: "s",
          },
          {
            name: "add_one / Jax / tpu / Forward",
            value: 0.0002151946869998,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / tpu / Forward",
            value: 0.0001897985080031,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / tpu / Forward",
            value: 0.0001898036269994,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / tpu / Forward",
            value: 0.0001927419870007,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / tpu / Forward",
            value: 0.0001854852269971,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / tpu / Forward",
            value: 0.0001840583369994,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / tpu / PreRev",
            value: 0.0001916492969976,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / tpu / PostRev",
            value: 0.0001965080970003,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / tpu / BothRev",
            value: 0.0001934792079991,
            unit: "s",
          },
          {
            name: "add_one / Jax / tpu / BothRev",
            value: 0.0001942163669991,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / tpu / PreRev",
            value: 0.0001961625870026,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / tpu / PostRev",
            value: 0.0001935424479997,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / tpu / BothRev",
            value: 0.000226468367,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / tpu / PreRev",
            value: 0.0002299773570011,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / tpu / PostRev",
            value: 0.0002112383469975,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / tpu / BothRev",
            value: 0.0002078847269985,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / tpu / PreRev",
            value: 0.0002224462570011,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / tpu / PostRev",
            value: 0.0002349600369998,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / tpu / BothRev",
            value: 0.0002367874170013,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / tpu / PreRev",
            value: 0.000232999576001,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / tpu / PostRev",
            value: 0.0002328389570029,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / tpu / BothRev",
            value: 0.0002329854470008,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / tpu / PreRev",
            value: 0.0002308374669992,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / tpu / PostRev",
            value: 0.0002300336469998,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / tpu / BothRev",
            value: 0.0002232308070015,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / Primal",
            value: 0.000006358589000228676,
            unit: "s",
          },
          {
            name: "add_one / Jax / cpu / Primal",
            value: 0.000006414890999622003,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / Primal",
            value: 0.000006759093000255234,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / Primal",
            value: 0.000006767386000319675,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / Primal",
            value: 0.000006831228999544692,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / Primal",
            value: 0.000006817521000812121,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / Primal",
            value: 0.000006809583999711321,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / Forward",
            value: 0.000009860213999672851,
            unit: "s",
          },
          {
            name: "add_one / Jax / cpu / Forward",
            value: 0.000010237409000183106,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / Forward",
            value: 0.000009852631000285327,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / Forward",
            value: 0.00000990190300035465,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / Forward",
            value: 0.000010368682999796874,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / Forward",
            value: 0.000010256825999931606,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / Forward",
            value: 0.000010319684000023698,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / PreRev",
            value: 0.000010503806000087936,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / PostRev",
            value: 0.000011207301000467852,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / BothRev",
            value: 0.000011290841000118237,
            unit: "s",
          },
          {
            name: "add_one / Jax / cpu / BothRev",
            value: 0.000011407354999391827,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / PreRev",
            value: 0.00001121341999987635,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / PostRev",
            value: 0.000011306660000627743,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / BothRev",
            value: 0.000011162912000145296,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / PreRev",
            value: 0.000011151261999657437,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / PostRev",
            value: 0.000011290797000583552,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / BothRev",
            value: 0.000010592592000648438,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / PreRev",
            value: 0.000011319442999592868,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / PostRev",
            value: 0.00001126044600005116,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / BothRev",
            value: 0.000011203013000340434,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / PreRev",
            value: 0.000011196159000064651,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / PostRev",
            value: 0.000011243886999181995,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / BothRev",
            value: 0.00001134782900044229,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / PreRev",
            value: 0.000011237229000471415,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / PostRev",
            value: 0.000011386950999622058,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / BothRev",
            value: 0.0000113130439995075,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / Primal",
            value: 0.000004298999999946318,
            unit: "s",
          },
          {
            name: "add_one / Jax / cpu / Primal",
            value: 0.000004072541999903478,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / Primal",
            value: 0.000004032334000157789,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / Primal",
            value: 0.000004266124999958265,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / Primal",
            value: 0.000004080666999925597,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / Primal",
            value: 0.000004058833999806666,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / Primal",
            value: 0.000004544541999848662,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / Forward",
            value: 0.000007480957999860038,
            unit: "s",
          },
          {
            name: "add_one / Jax / cpu / Forward",
            value: 0.000007067167000059272,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / Forward",
            value: 0.000007274333999930605,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / Forward",
            value: 0.000006498916999817084,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / Forward",
            value: 0.000006543291000070894,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / Forward",
            value: 0.000007134583999913957,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / Forward",
            value: 0.0000065642919998936124,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / PreRev",
            value: 0.000007866082999953506,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / PostRev",
            value: 0.000007697333000123762,
            unit: "s",
          },
          {
            name: "add_one / JaXPipe / cpu / BothRev",
            value: 0.000007389291999970737,
            unit: "s",
          },
          {
            name: "add_one / Jax / cpu / BothRev",
            value: 0.000008607084000004761,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / PreRev",
            value: 0.00001094208399990748,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / PostRev",
            value: 0.000007951499999990119,
            unit: "s",
          },
          {
            name: "add_one / HLOOpt / cpu / BothRev",
            value: 0.000007536707999861392,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / PreRev",
            value: 0.000007474709000007352,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / PostRev",
            value: 0.000008270457999969949,
            unit: "s",
          },
          {
            name: "add_one / PartOpt / cpu / BothRev",
            value: 0.000008022000000210027,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / PreRev",
            value: 0.000008652542000163522,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / PostRev",
            value: 0.000008255667000184985,
            unit: "s",
          },
          {
            name: "add_one / IPartOpt / cpu / BothRev",
            value: 0.000007962125000176457,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / PreRev",
            value: 0.000007539625000163142,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / PostRev",
            value: 0.000007642583000006197,
            unit: "s",
          },
          {
            name: "add_one / DefOpt / cpu / BothRev",
            value: 0.000007206624999980704,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / PreRev",
            value: 0.000007402457999887701,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / PostRev",
            value: 0.000010476959000015996,
            unit: "s",
          },
          {
            name: "add_one / IDefOpt / cpu / BothRev",
            value: 0.000009285416000011538,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / Primal",
            value: 0.000005045507000431826,
            unit: "s",
          },
          {
            name: "add_two / Jax / cpu / Primal",
            value: 0.0000050422680005794975,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / Primal",
            value: 0.000005184839999856194,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / Primal",
            value: 0.000005102473999613721,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / Primal",
            value: 0.0000050749379997796496,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / Primal",
            value: 0.000005106177999550709,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / Primal",
            value: 0.000005148702000042249,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / Forward",
            value: 0.00000819935900017299,
            unit: "s",
          },
          {
            name: "add_two / Jax / cpu / Forward",
            value: 0.00000818599899957917,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / Forward",
            value: 0.000008317114999954357,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / Forward",
            value: 0.000008343741000317095,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / Forward",
            value: 0.000008199212000363332,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / Forward",
            value: 0.00000824392800041096,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / Forward",
            value: 0.000008249127999988559,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / PreRev",
            value: 0.0000107634149999285,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / PostRev",
            value: 0.000010919954999735637,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / BothRev",
            value: 0.00001101195400042343,
            unit: "s",
          },
          {
            name: "add_two / Jax / cpu / BothRev",
            value: 0.000010846578999917256,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / PreRev",
            value: 0.000010926989999461512,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / PostRev",
            value: 0.000010967039999741246,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / BothRev",
            value: 0.000010966806999931576,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / PreRev",
            value: 0.000010849355000573269,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / PostRev",
            value: 0.000010623318000398285,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / BothRev",
            value: 0.000010812594000526588,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / PreRev",
            value: 0.000010814218000632535,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / PostRev",
            value: 0.000010819824999998671,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / BothRev",
            value: 0.00001071191600021848,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / PreRev",
            value: 0.000010616501000185965,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / PostRev",
            value: 0.000010838954000064403,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / BothRev",
            value: 0.000010796090999974694,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / PreRev",
            value: 0.000010889940000197384,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / PostRev",
            value: 0.000010685913000088475,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / BothRev",
            value: 0.000010837558999810426,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / Primal",
            value: 0.000009231483998519252,
            unit: "s",
          },
          {
            name: "add_two / Jax / cpu / Primal",
            value: 0.000009507753000434605,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / Primal",
            value: 0.00000945313800002623,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / Primal",
            value: 0.000009506547999990289,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / Primal",
            value: 0.000009424085001228376,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / Primal",
            value: 0.000010019342000305188,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / Primal",
            value: 0.000009475142000155756,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / Forward",
            value: 0.000014295471000878024,
            unit: "s",
          },
          {
            name: "add_two / Jax / cpu / Forward",
            value: 0.000014239351001378963,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / Forward",
            value: 0.000014766274000066914,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / Forward",
            value: 0.000014234567001039978,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / Forward",
            value: 0.00001488356500158261,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / Forward",
            value: 0.000014333441000417224,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / Forward",
            value: 0.000014857161999316304,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / PreRev",
            value: 0.00001836231100060104,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / PostRev",
            value: 0.000018237903001136148,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / BothRev",
            value: 0.000018100907998814364,
            unit: "s",
          },
          {
            name: "add_two / Jax / cpu / BothRev",
            value: 0.00001820872899952519,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / PreRev",
            value: 0.000018531321000409663,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / PostRev",
            value: 0.00001768871300009778,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / BothRev",
            value: 0.000018348858000535984,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / PreRev",
            value: 0.000018247370000608497,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / PostRev",
            value: 0.000018153507000533865,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / BothRev",
            value: 0.000017040259001078085,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / PreRev",
            value: 0.000018424464999043267,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / PostRev",
            value: 0.000018272321000040392,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / BothRev",
            value: 0.00001822555900071165,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / PreRev",
            value: 0.0000183553060014674,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / PostRev",
            value: 0.000018426262000502903,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / BothRev",
            value: 0.000018282570999872404,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / PreRev",
            value: 0.00001825235800060909,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / PostRev",
            value: 0.000018232717000501,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / BothRev",
            value: 0.000018253662999995863,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / tpu / Primal",
            value: 0.000137464198,
            unit: "s",
          },
          {
            name: "add_two / Jax / tpu / Primal",
            value: 0.0001383889579992,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / tpu / Primal",
            value: 0.0001363736290004,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / tpu / Primal",
            value: 0.0001370209580018,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / tpu / Primal",
            value: 0.0001361926780009,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / tpu / Primal",
            value: 0.0001385489880012,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / tpu / Primal",
            value: 0.0001352108489991,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / tpu / Forward",
            value: 0.0002001797670018,
            unit: "s",
          },
          {
            name: "add_two / Jax / tpu / Forward",
            value: 0.0002028017569973,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / tpu / Forward",
            value: 0.0002129683769999,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / tpu / Forward",
            value: 0.0002148671370014,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / tpu / Forward",
            value: 0.0002104720169991,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / tpu / Forward",
            value: 0.0002007689469974,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / tpu / Forward",
            value: 0.0001999882769996,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / tpu / PreRev",
            value: 0.0002222208370003,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / tpu / PostRev",
            value: 0.0002123293070035,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / tpu / BothRev",
            value: 0.0002135444269988,
            unit: "s",
          },
          {
            name: "add_two / Jax / tpu / BothRev",
            value: 0.000240486527,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / tpu / PreRev",
            value: 0.0002408402469991,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / tpu / PostRev",
            value: 0.0002432377870027,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / tpu / BothRev",
            value: 0.0002426129069972,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / tpu / PreRev",
            value: 0.0002461442170024,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / tpu / PostRev",
            value: 0.0002264890570004,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / tpu / BothRev",
            value: 0.0002398901259984,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / tpu / PreRev",
            value: 0.0002448429369978,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / tpu / PostRev",
            value: 0.0002407788269993,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / tpu / BothRev",
            value: 0.0002400149960012,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / tpu / PreRev",
            value: 0.0002407161869996,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / tpu / PostRev",
            value: 0.0002427817360003,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / tpu / BothRev",
            value: 0.000242912925998,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / tpu / PreRev",
            value: 0.0002428160569979,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / tpu / PostRev",
            value: 0.000244755787,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / tpu / BothRev",
            value: 0.0002374086870004,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / Primal",
            value: 0.000006626640999456868,
            unit: "s",
          },
          {
            name: "add_two / Jax / cpu / Primal",
            value: 0.000006742157000189764,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / Primal",
            value: 0.0000066989670003749775,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / Primal",
            value: 0.000007145251000110875,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / Primal",
            value: 0.000007079734999933862,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / Primal",
            value: 0.000006759253999916836,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / Primal",
            value: 0.000007059864999973797,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / Forward",
            value: 0.0000101658839994343,
            unit: "s",
          },
          {
            name: "add_two / Jax / cpu / Forward",
            value: 0.00001063122600044153,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / Forward",
            value: 0.000010169225999561605,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / Forward",
            value: 0.000010639793999871473,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / Forward",
            value: 0.000010157675000300516,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / Forward",
            value: 0.000010695709000174248,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / Forward",
            value: 0.000010107115000209888,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / PreRev",
            value: 0.000013229801999841583,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / PostRev",
            value: 0.000012661765999837371,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / BothRev",
            value: 0.000013318826000613626,
            unit: "s",
          },
          {
            name: "add_two / Jax / cpu / BothRev",
            value: 0.000013257230999442982,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / PreRev",
            value: 0.000013126615000146555,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / PostRev",
            value: 0.000013275294999402833,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / BothRev",
            value: 0.000013152469000488054,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / PreRev",
            value: 0.000013158474000192654,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / PostRev",
            value: 0.000013321469999937109,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / BothRev",
            value: 0.000012525268999524996,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / PreRev",
            value: 0.000013128999999935332,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / PostRev",
            value: 0.000013160067999706371,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / BothRev",
            value: 0.000013055952999820874,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / PreRev",
            value: 0.000013144765999641095,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / PostRev",
            value: 0.00001311533700027212,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / BothRev",
            value: 0.000013090565000311472,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / PreRev",
            value: 0.00001346270599970012,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / PostRev",
            value: 0.000013243738000710436,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / BothRev",
            value: 0.000013305469999977504,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / Primal",
            value: 0.000004731708000008439,
            unit: "s",
          },
          {
            name: "add_two / Jax / cpu / Primal",
            value: 0.000004807583999991038,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / Primal",
            value: 0.000005597209000143266,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / Primal",
            value: 0.000004719917000102214,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / Primal",
            value: 0.000004573916999788708,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / Primal",
            value: 0.0000059501249997993,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / Primal",
            value: 0.000006079624999983935,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / Forward",
            value: 0.000008450416999949084,
            unit: "s",
          },
          {
            name: "add_two / Jax / cpu / Forward",
            value: 0.00000711645900014446,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / Forward",
            value: 0.00000680420799994863,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / Forward",
            value: 0.000006792249999989508,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / Forward",
            value: 0.0000067908329999681885,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / Forward",
            value: 0.000007631541999899128,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / Forward",
            value: 0.000008302916000047844,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / PreRev",
            value: 0.000009341625000161005,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / PostRev",
            value: 0.000009381749999874956,
            unit: "s",
          },
          {
            name: "add_two / JaXPipe / cpu / BothRev",
            value: 0.0000104873340001177,
            unit: "s",
          },
          {
            name: "add_two / Jax / cpu / BothRev",
            value: 0.000010811084000124538,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / PreRev",
            value: 0.000009653416999981346,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / PostRev",
            value: 0.000010610415999963152,
            unit: "s",
          },
          {
            name: "add_two / HLOOpt / cpu / BothRev",
            value: 0.00001022083299994847,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / PreRev",
            value: 0.000009818042000006244,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / PostRev",
            value: 0.000009819541000069875,
            unit: "s",
          },
          {
            name: "add_two / PartOpt / cpu / BothRev",
            value: 0.000009235833999809984,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / PreRev",
            value: 0.000009324750000132551,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / PostRev",
            value: 0.000009967458000119222,
            unit: "s",
          },
          {
            name: "add_two / IPartOpt / cpu / BothRev",
            value: 0.00000911866700016617,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / PreRev",
            value: 0.000009450416999925436,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / PostRev",
            value: 0.000009359000000131346,
            unit: "s",
          },
          {
            name: "add_two / DefOpt / cpu / BothRev",
            value: 0.000009387458999981391,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / PreRev",
            value: 0.000009229750000031344,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / PostRev",
            value: 0.000009016707999990105,
            unit: "s",
          },
          {
            name: "add_two / IDefOpt / cpu / BothRev",
            value: 0.000009690332999980455,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / Primal",
            value: 0.000004596415999913007,
            unit: "s",
          },
          {
            name: "cache / Jax / cpu / Primal",
            value: 0.000004548440999315062,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / Primal",
            value: 0.000004588697000144748,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / Primal",
            value: 0.000004579954999826441,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / Primal",
            value: 0.000004547331999674498,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / Primal",
            value: 0.000004640090000066266,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / Primal",
            value: 0.000004567871999825001,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / Forward",
            value: 0.000011174250999829382,
            unit: "s",
          },
          {
            name: "cache / Jax / cpu / Forward",
            value: 0.000011233702000026823,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / Forward",
            value: 0.000011345565999363316,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / Forward",
            value: 0.000011566444999516534,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / Forward",
            value: 0.000011520832999849518,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / Forward",
            value: 0.00001111552699967433,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / Forward",
            value: 0.00001094665700020414,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / PreRev",
            value: 0.000012408308999511065,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / PostRev",
            value: 0.00001683880700056761,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / BothRev",
            value: 0.000013033371999881635,
            unit: "s",
          },
          {
            name: "cache / Jax / cpu / BothRev",
            value: 0.000016919731999223587,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / PreRev",
            value: 0.00001297770000019227,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / PostRev",
            value: 0.00001295839500016882,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / BothRev",
            value: 0.00001283322700055578,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / PreRev",
            value: 0.000013102066999636009,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / PostRev",
            value: 0.000016454395000437217,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / BothRev",
            value: 0.000012530147999314068,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / PreRev",
            value: 0.00001241119099995558,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / PostRev",
            value: 0.00001556867099952797,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / BothRev",
            value: 0.000012599764000697178,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / PreRev",
            value: 0.00001272165099999256,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / PostRev",
            value: 0.000012508425999840256,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / BothRev",
            value: 0.00001258662700001878,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / PreRev",
            value: 0.00001240751199929946,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / PostRev",
            value: 0.000012924741000460929,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / BothRev",
            value: 0.00001237514199965517,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / Primal",
            value: 0.000009440583000468905,
            unit: "s",
          },
          {
            name: "cache / Jax / cpu / Primal",
            value: 0.000009412597999471472,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / Primal",
            value: 0.000009259261998522562,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / Primal",
            value: 0.000009265237000363411,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / Primal",
            value: 0.000009417228000529575,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / Primal",
            value: 0.00000925093600017135,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / Primal",
            value: 0.000010022697000749758,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / Forward",
            value: 0.000014832888000455569,
            unit: "s",
          },
          {
            name: "cache / Jax / cpu / Forward",
            value: 0.000014456091001193272,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / Forward",
            value: 0.00001491715399970417,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / Forward",
            value: 0.00002051550699979998,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / Forward",
            value: 0.000015316380000513163,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / Forward",
            value: 0.000014822821998677682,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / Forward",
            value: 0.00001442251399930683,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / PreRev",
            value: 0.00001549522200002684,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / PostRev",
            value: 0.00001757314399947063,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / BothRev",
            value: 0.000015532875000644707,
            unit: "s",
          },
          {
            name: "cache / Jax / cpu / BothRev",
            value: 0.00001780616900032328,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / PreRev",
            value: 0.000015166445999057032,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / PostRev",
            value: 0.000015534785999989254,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / BothRev",
            value: 0.00001488460499967914,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / PreRev",
            value: 0.000015532483999777468,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / PostRev",
            value: 0.00001798589899954095,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / BothRev",
            value: 0.000015179269001237116,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / PreRev",
            value: 0.000015678490000937017,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / PostRev",
            value: 0.000017636894001043403,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / BothRev",
            value: 0.000015719883998826844,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / PreRev",
            value: 0.000015208458000415704,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / PostRev",
            value: 0.00001569294299952162,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / BothRev",
            value: 0.000015393552999739767,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / PreRev",
            value: 0.000015147195999816176,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / PostRev",
            value: 0.000015577465999740525,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / BothRev",
            value: 0.000015572246000374435,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / tpu / Primal",
            value: 0.0001355063580012,
            unit: "s",
          },
          {
            name: "cache / Jax / tpu / Primal",
            value: 0.0001337002079999,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / tpu / Primal",
            value: 0.0001320211989987,
            unit: "s",
          },
          {
            name: "cache / PartOpt / tpu / Primal",
            value: 0.0001336515379989,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / tpu / Primal",
            value: 0.0001504119079982,
            unit: "s",
          },
          {
            name: "cache / DefOpt / tpu / Primal",
            value: 0.0001369436579989,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / tpu / Primal",
            value: 0.0001329688490004,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / tpu / Forward",
            value: 0.0002160942670016,
            unit: "s",
          },
          {
            name: "cache / Jax / tpu / Forward",
            value: 0.0002214110669992,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / tpu / Forward",
            value: 0.0002284463069991,
            unit: "s",
          },
          {
            name: "cache / PartOpt / tpu / Forward",
            value: 0.0002063476670009,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / tpu / Forward",
            value: 0.0002163054769989,
            unit: "s",
          },
          {
            name: "cache / DefOpt / tpu / Forward",
            value: 0.0002163898369981,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / tpu / Forward",
            value: 0.0002075869170002,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / tpu / PreRev",
            value: 0.0002107398369989,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / tpu / PostRev",
            value: 0.0001885693369986,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / tpu / BothRev",
            value: 0.0001879142480029,
            unit: "s",
          },
          {
            name: "cache / Jax / tpu / BothRev",
            value: 0.0001869468370023,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / tpu / PreRev",
            value: 0.0001894637580007,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / tpu / PostRev",
            value: 0.0002085019369988,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / tpu / BothRev",
            value: 0.0002078192869994,
            unit: "s",
          },
          {
            name: "cache / PartOpt / tpu / PreRev",
            value: 0.0002067941870009,
            unit: "s",
          },
          {
            name: "cache / PartOpt / tpu / PostRev",
            value: 0.0002078294370003,
            unit: "s",
          },
          {
            name: "cache / PartOpt / tpu / BothRev",
            value: 0.000213444797002,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / tpu / PreRev",
            value: 0.0001808681670008,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / tpu / PostRev",
            value: 0.0001796317679982,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / tpu / BothRev",
            value: 0.0001786033869975,
            unit: "s",
          },
          {
            name: "cache / DefOpt / tpu / PreRev",
            value: 0.000181008487998,
            unit: "s",
          },
          {
            name: "cache / DefOpt / tpu / PostRev",
            value: 0.0002103601969974,
            unit: "s",
          },
          {
            name: "cache / DefOpt / tpu / BothRev",
            value: 0.0001993187769985,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / tpu / PreRev",
            value: 0.0002017185270015,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / tpu / PostRev",
            value: 0.0001971736969971,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / tpu / BothRev",
            value: 0.0001944233370013,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / Primal",
            value: 0.000005980083999929775,
            unit: "s",
          },
          {
            name: "cache / Jax / cpu / Primal",
            value: 0.000006524876999719709,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / Primal",
            value: 0.000005976835999717877,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / Primal",
            value: 0.000006228598000234342,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / Primal",
            value: 0.000005998783999530133,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / Primal",
            value: 0.00000621146100002079,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / Primal",
            value: 0.000006133302999842272,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / Forward",
            value: 0.00001070202499977313,
            unit: "s",
          },
          {
            name: "cache / Jax / cpu / Forward",
            value: 0.000010443103999932648,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / Forward",
            value: 0.000010743169000306806,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / Forward",
            value: 0.000010785204000058002,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / Forward",
            value: 0.000010419287999866356,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / Forward",
            value: 0.00001080731599995488,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / Forward",
            value: 0.000010378038999988348,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / PreRev",
            value: 0.000011225941000702732,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / PostRev",
            value: 0.000012381628999719396,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / BothRev",
            value: 0.000011293186000330023,
            unit: "s",
          },
          {
            name: "cache / Jax / cpu / BothRev",
            value: 0.000012654032000682492,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / PreRev",
            value: 0.000010656502000529144,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / PostRev",
            value: 0.000010604770000099962,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / BothRev",
            value: 0.000010412471000563527,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / PreRev",
            value: 0.000011016518999895196,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / PostRev",
            value: 0.000012364780999632784,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / BothRev",
            value: 0.00001081698999951186,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / PreRev",
            value: 0.000011067315999753192,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / PostRev",
            value: 0.000012477608999688528,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / BothRev",
            value: 0.000010987263999595598,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / PreRev",
            value: 0.00001093021399992722,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / PostRev",
            value: 0.000010905854999691657,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / BothRev",
            value: 0.000010987501999807136,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / PreRev",
            value: 0.000011215384000024644,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / PostRev",
            value: 0.000011286211999504304,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / BothRev",
            value: 0.00001135693700052798,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / Primal",
            value: 0.000003949959000010495,
            unit: "s",
          },
          {
            name: "cache / Jax / cpu / Primal",
            value: 0.000004254124999988562,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / Primal",
            value: 0.000003888499999902706,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / Primal",
            value: 0.000003892292000045927,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / Primal",
            value: 0.00000391666699988491,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / Primal",
            value: 0.0000037970420000874575,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / Primal",
            value: 0.000003786999999874752,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / Forward",
            value: 0.000006290292000130648,
            unit: "s",
          },
          {
            name: "cache / Jax / cpu / Forward",
            value: 0.000007101583000121537,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / Forward",
            value: 0.000006209084000147413,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / Forward",
            value: 0.00000632099999984348,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / Forward",
            value: 0.000006502292000050147,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / Forward",
            value: 0.000006226749999996173,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / Forward",
            value: 0.000006504416999860041,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / PreRev",
            value: 0.000006623957999863705,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / PostRev",
            value: 0.0000075562920001175374,
            unit: "s",
          },
          {
            name: "cache / JaXPipe / cpu / BothRev",
            value: 0.00000692608300005304,
            unit: "s",
          },
          {
            name: "cache / Jax / cpu / BothRev",
            value: 0.000009051125000041791,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / PreRev",
            value: 0.000006931334000000788,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / PostRev",
            value: 0.000006641540999908102,
            unit: "s",
          },
          {
            name: "cache / HLOOpt / cpu / BothRev",
            value: 0.0000070289169998432045,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / PreRev",
            value: 0.000006868791999977475,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / PostRev",
            value: 0.000007343249999848922,
            unit: "s",
          },
          {
            name: "cache / PartOpt / cpu / BothRev",
            value: 0.000006558791999850655,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / PreRev",
            value: 0.000006595249999918451,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / PostRev",
            value: 0.000007567750000134765,
            unit: "s",
          },
          {
            name: "cache / IPartOpt / cpu / BothRev",
            value: 0.000010499750000008134,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / PreRev",
            value: 0.00001207045800015294,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / PostRev",
            value: 0.000014479459000085624,
            unit: "s",
          },
          {
            name: "cache / DefOpt / cpu / BothRev",
            value: 0.000012050584000007802,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / PreRev",
            value: 0.000007948875000010958,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / PostRev",
            value: 0.000006358082999895487,
            unit: "s",
          },
          {
            name: "cache / IDefOpt / cpu / BothRev",
            value: 0.000008900541999992128,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / Primal",
            value: 0.0000049002590003510704,
            unit: "s",
          },
          {
            name: "Concat / Jax / cpu / Primal",
            value: 0.000004957804000696342,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / Primal",
            value: 0.00000484478999987914,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / Primal",
            value: 0.000004832094999983383,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / Primal",
            value: 0.000005013356000745262,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / Primal",
            value: 0.000004898076000245055,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / Primal",
            value: 0.000004864344999987225,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / Forward",
            value: 0.000007892947000073037,
            unit: "s",
          },
          {
            name: "Concat / Jax / cpu / Forward",
            value: 0.000008054516000811418,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / Forward",
            value: 0.000007985862999703386,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / Forward",
            value: 0.000008047525000620225,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / Forward",
            value: 0.000008008511000298313,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / Forward",
            value: 0.000008080756999333971,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / Forward",
            value: 0.000008073674000115716,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / PreRev",
            value: 0.000009026441000060005,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / PostRev",
            value: 0.000008720363999600523,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / BothRev",
            value: 0.000008982518999800959,
            unit: "s",
          },
          {
            name: "Concat / Jax / cpu / BothRev",
            value: 0.000008985303999907046,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / PreRev",
            value: 0.000008934821999901032,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / PostRev",
            value: 0.000008728279000024486,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / BothRev",
            value: 0.000008682752999447984,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / PreRev",
            value: 0.000008830318000036641,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / PostRev",
            value: 0.00000894835500002955,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / BothRev",
            value: 0.000008635004000097978,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / PreRev",
            value: 0.000008825633000014932,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / PostRev",
            value: 0.000008844105000207492,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / BothRev",
            value: 0.000008848706999742717,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / PreRev",
            value: 0.000008863007000400103,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / PostRev",
            value: 0.000008788750999883632,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / BothRev",
            value: 0.000008709900999747334,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / PreRev",
            value: 0.000009019510000143782,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / PostRev",
            value: 0.000009275940999941667,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / BothRev",
            value: 0.000008940333000282408,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / Primal",
            value: 0.000008939219000239972,
            unit: "s",
          },
          {
            name: "Concat / Jax / cpu / Primal",
            value: 0.000008950629999162629,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / Primal",
            value: 0.000009401657000125851,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / Primal",
            value: 0.000012164138999651186,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / Primal",
            value: 0.000009522229000140214,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / Primal",
            value: 0.000009139537000010025,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / Primal",
            value: 0.00000915562899899669,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / Forward",
            value: 0.000014553426000929905,
            unit: "s",
          },
          {
            name: "Concat / Jax / cpu / Forward",
            value: 0.000014687752000099862,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / Forward",
            value: 0.000013832812999680756,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / Forward",
            value: 0.000014699719000418554,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / Forward",
            value: 0.000013902947999667958,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / Forward",
            value: 0.00001369382300072175,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / Forward",
            value: 0.000014682397999422393,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / PreRev",
            value: 0.00001493755099909322,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / PostRev",
            value: 0.000015845194000576158,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / BothRev",
            value: 0.000015764334000778034,
            unit: "s",
          },
          {
            name: "Concat / Jax / cpu / BothRev",
            value: 0.000015903106999758165,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / PreRev",
            value: 0.000015739966000182904,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / PostRev",
            value: 0.000015860839001106797,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / BothRev",
            value: 0.00001575215900084004,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / PreRev",
            value: 0.000015598259000398685,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / PostRev",
            value: 0.000015677224999308236,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / BothRev",
            value: 0.00001507942100033688,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / PreRev",
            value: 0.000015978721999999834,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / PostRev",
            value: 0.000016005208999558816,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / BothRev",
            value: 0.000015037140001368244,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / PreRev",
            value: 0.000015850953001063317,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / PostRev",
            value: 0.00001491002600050706,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / BothRev",
            value: 0.000015787482001542232,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / PreRev",
            value: 0.000015785772000526777,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / PostRev",
            value: 0.000015795790000993293,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / BothRev",
            value: 0.00001577586400162545,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / tpu / Primal",
            value: 0.0001334079679982,
            unit: "s",
          },
          {
            name: "Concat / Jax / tpu / Primal",
            value: 0.0001313311379999,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / tpu / Primal",
            value: 0.0001297385879988,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / tpu / Primal",
            value: 0.0001318039280013,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / tpu / Primal",
            value: 0.0001316344379993,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / tpu / Primal",
            value: 0.0001366425479973,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / tpu / Primal",
            value: 0.0001334921289999,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / tpu / Forward",
            value: 0.0002002109169989,
            unit: "s",
          },
          {
            name: "Concat / Jax / tpu / Forward",
            value: 0.0002085476170032,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / tpu / Forward",
            value: 0.0001998356969997,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / tpu / Forward",
            value: 0.000196162406999,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / tpu / Forward",
            value: 0.0002033302470008,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / tpu / Forward",
            value: 0.0002003991370002,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / tpu / Forward",
            value: 0.0002014247369988,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / tpu / PreRev",
            value: 0.0002061382669999,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / tpu / PostRev",
            value: 0.0002056959569999,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / tpu / BothRev",
            value: 0.0002014264669996,
            unit: "s",
          },
          {
            name: "Concat / Jax / tpu / BothRev",
            value: 0.0002062525969995,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / tpu / PreRev",
            value: 0.0002142485369986,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / tpu / PostRev",
            value: 0.0002083200269989,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / tpu / BothRev",
            value: 0.0002168149669996,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / tpu / PreRev",
            value: 0.000200391377999,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / tpu / PostRev",
            value: 0.0002093310769996,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / tpu / BothRev",
            value: 0.0002289331669999,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / tpu / PreRev",
            value: 0.000237380475999,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / tpu / PostRev",
            value: 0.0002289861160024,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / tpu / BothRev",
            value: 0.0002207501569973,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / tpu / PreRev",
            value: 0.0002241960669998,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / tpu / PostRev",
            value: 0.0002240990370009,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / tpu / BothRev",
            value: 0.000221371536998,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / tpu / PreRev",
            value: 0.0002118126569985,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / tpu / PostRev",
            value: 0.0002293084470002,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / tpu / BothRev",
            value: 0.0002273875770006,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / Primal",
            value: 0.000006396970000423608,
            unit: "s",
          },
          {
            name: "Concat / Jax / cpu / Primal",
            value: 0.000006349295000291022,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / Primal",
            value: 0.000006361246999404102,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / Primal",
            value: 0.000006278605999796128,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / Primal",
            value: 0.000006414890000087325,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / Primal",
            value: 0.000006386206000570382,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / Primal",
            value: 0.000006421730000511161,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / Forward",
            value: 0.00001033132799966552,
            unit: "s",
          },
          {
            name: "Concat / Jax / cpu / Forward",
            value: 0.000009871004000160613,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / Forward",
            value: 0.0000103062170001067,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / Forward",
            value: 0.000010326897999220818,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / Forward",
            value: 0.000009839770999860776,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / Forward",
            value: 0.00001026612700024998,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / Forward",
            value: 0.000010034256999460922,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / PreRev",
            value: 0.000010597905999929936,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / PostRev",
            value: 0.000010539513000367152,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / BothRev",
            value: 0.00001138077500036161,
            unit: "s",
          },
          {
            name: "Concat / Jax / cpu / BothRev",
            value: 0.000011305357999845,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / PreRev",
            value: 0.000011313812000480538,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / PostRev",
            value: 0.000011296205000689951,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / BothRev",
            value: 0.00001084541199998057,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / PreRev",
            value: 0.000010689369999454356,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / PostRev",
            value: 0.0000114498659995661,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / BothRev",
            value: 0.00001087015299981431,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / PreRev",
            value: 0.00001153273099953367,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / PostRev",
            value: 0.000011094956999841089,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / BothRev",
            value: 0.000011220030999538722,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / PreRev",
            value: 0.000011599211000429933,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / PostRev",
            value: 0.00001152023100075894,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / BothRev",
            value: 0.00001147639699956926,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / PreRev",
            value: 0.000011382437000065692,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / PostRev",
            value: 0.000011489716000141924,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / BothRev",
            value: 0.000011417226000048688,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / Primal",
            value: 0.000004107374999875901,
            unit: "s",
          },
          {
            name: "Concat / Jax / cpu / Primal",
            value: 0.000004136292000112008,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / Primal",
            value: 0.000004086416999825815,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / Primal",
            value: 0.00000410670799988111,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / Primal",
            value: 0.000004087125000069136,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / Primal",
            value: 0.000004113250000045809,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / Primal",
            value: 0.000004123542000115776,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / Forward",
            value: 0.000007032417000118585,
            unit: "s",
          },
          {
            name: "Concat / Jax / cpu / Forward",
            value: 0.000007550208000111525,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / Forward",
            value: 0.000006549167000002853,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / Forward",
            value: 0.000007843750000120053,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / Forward",
            value: 0.00000813345799997478,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / Forward",
            value: 0.000007047082999861232,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / Forward",
            value: 0.000007496458000105122,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / PreRev",
            value: 0.000007813749999968422,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / PostRev",
            value: 0.000008305249999921216,
            unit: "s",
          },
          {
            name: "Concat / JaXPipe / cpu / BothRev",
            value: 0.000008047916999885274,
            unit: "s",
          },
          {
            name: "Concat / Jax / cpu / BothRev",
            value: 0.000008659833999900002,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / PreRev",
            value: 0.00000885341699995479,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / PostRev",
            value: 0.000009869957999853796,
            unit: "s",
          },
          {
            name: "Concat / HLOOpt / cpu / BothRev",
            value: 0.00000989229099991462,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / PreRev",
            value: 0.00000903479199996582,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / PostRev",
            value: 0.00000905604200011112,
            unit: "s",
          },
          {
            name: "Concat / PartOpt / cpu / BothRev",
            value: 0.000008294916000068042,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / PreRev",
            value: 0.000008837207999931706,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / PostRev",
            value: 0.000009555083999885028,
            unit: "s",
          },
          {
            name: "Concat / IPartOpt / cpu / BothRev",
            value: 0.000009603332999859047,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / PreRev",
            value: 0.000008772582999881707,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / PostRev",
            value: 0.000007750624999971479,
            unit: "s",
          },
          {
            name: "Concat / DefOpt / cpu / BothRev",
            value: 0.000008007249999991472,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / PreRev",
            value: 0.000008152500000051076,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / PostRev",
            value: 0.000007745665999891571,
            unit: "s",
          },
          {
            name: "Concat / IDefOpt / cpu / BothRev",
            value: 0.00000851337499989313,
            unit: "s",
          },
          {
            name: "const_scatter / JaXPipe / cpu / Primal",
            value: 0.000004465158000130032,
            unit: "s",
          },
          {
            name: "const_scatter / Jax / cpu / Primal",
            value: 0.000004547415999695659,
            unit: "s",
          },
          {
            name: "const_scatter / HLOOpt / cpu / Primal",
            value: 0.000005122170000504411,
            unit: "s",
          },
          {
            name: "const_scatter / PartOpt / cpu / Primal",
            value: 0.000004833917999349069,
            unit: "s",
          },
          {
            name: "const_scatter / IPartOpt / cpu / Primal",
            value: 0.000004814332000023569,
            unit: "s",
          },
          {
            name: "const_scatter / DefOpt / cpu / Primal",
            value: 0.000005415598000581668,
            unit: "s",
          },
          {
            name: "const_scatter / IDefOpt / cpu / Primal",
            value: 0.000005403333000685962,
            unit: "s",
          },
          {
            name: "const_scatter / JaXPipe / cpu / Forward",
            value: 0.000008502845000293747,
            unit: "s",
          },
          {
            name: "const_scatter / Jax / cpu / Forward",
            value: 0.000007430880999891088,
            unit: "s",
          },
          {
            name: "const_scatter / HLOOpt / cpu / Forward",
            value: 0.000008180636000361118,
            unit: "s",
          },
          {
            name: "const_scatter / PartOpt / cpu / Forward",
            value: 0.000008548931999939669,
            unit: "s",
          },
          {
            name: "const_scatter / IPartOpt / cpu / Forward",
            value: 0.000008518329000253289,
            unit: "s",
          },
          {
            name: "const_scatter / DefOpt / cpu / Forward",
            value: 0.000008589580000261776,
            unit: "s",
          },
          {
            name: "const_scatter / IDefOpt / cpu / Forward",
            value: 0.000008597222000389593,
            unit: "s",
          },
          {
            name: "const_scatter / JaXPipe / cpu / Primal",
            value: 0.00000927120000051218,
            unit: "s",
          },
          {
            name: "const_scatter / Jax / cpu / Primal",
            value: 0.000008824026999718625,
            unit: "s",
          },
          {
            name: "const_scatter / HLOOpt / cpu / Primal",
            value: 0.000009731558999192205,
            unit: "s",
          },
          {
            name: "const_scatter / PartOpt / cpu / Primal",
            value: 0.000009332977000667596,
            unit: "s",
          },
          {
            name: "const_scatter / IPartOpt / cpu / Primal",
            value: 0.000008828766000078758,
            unit: "s",
          },
          {
            name: "const_scatter / DefOpt / cpu / Primal",
            value: 0.000010522463000597782,
            unit: "s",
          },
          {
            name: "const_scatter / IDefOpt / cpu / Primal",
            value: 0.000010566447999735828,
            unit: "s",
          },
          {
            name: "const_scatter / JaXPipe / cpu / Forward",
            value: 0.000015460333999726572,
            unit: "s",
          },
          {
            name: "const_scatter / Jax / cpu / Forward",
            value: 0.000013362237999899662,
            unit: "s",
          },
          {
            name: "const_scatter / HLOOpt / cpu / Forward",
            value: 0.000014686406000691931,
            unit: "s",
          },
          {
            name: "const_scatter / PartOpt / cpu / Forward",
            value: 0.000014670525000838096,
            unit: "s",
          },
          {
            name: "const_scatter / IPartOpt / cpu / Forward",
            value: 0.00001461675400059903,
            unit: "s",
          },
          {
            name: "const_scatter / DefOpt / cpu / Forward",
            value: 0.00001542534500003967,
            unit: "s",
          },
          {
            name: "const_scatter / IDefOpt / cpu / Forward",
            value: 0.0000156074799997441,
            unit: "s",
          },
          {
            name: "const_scatter / JaXPipe / tpu / Primal",
            value: 0.0001372598479974,
            unit: "s",
          },
          {
            name: "const_scatter / Jax / tpu / Primal",
            value: 0.0001368591179998,
            unit: "s",
          },
          {
            name: "const_scatter / HLOOpt / tpu / Primal",
            value: 0.0001359123080001,
            unit: "s",
          },
          {
            name: "const_scatter / PartOpt / tpu / Primal",
            value: 0.0001352603980012,
            unit: "s",
          },
          {
            name: "const_scatter / IPartOpt / tpu / Primal",
            value: 0.0001339826080002,
            unit: "s",
          },
          {
            name: "const_scatter / DefOpt / tpu / Primal",
            value: 0.0001395531980015,
            unit: "s",
          },
          {
            name: "const_scatter / IDefOpt / tpu / Primal",
            value: 0.0001508914879996,
            unit: "s",
          },
          {
            name: "const_scatter / JaXPipe / tpu / Forward",
            value: 0.0002277591559977,
            unit: "s",
          },
          {
            name: "const_scatter / Jax / tpu / Forward",
            value: 0.0002275301570007,
            unit: "s",
          },
          {
            name: "const_scatter / HLOOpt / tpu / Forward",
            value: 0.0002232811069989,
            unit: "s",
          },
          {
            name: "const_scatter / PartOpt / tpu / Forward",
            value: 0.000218307477,
            unit: "s",
          },
          {
            name: "const_scatter / IPartOpt / tpu / Forward",
            value: 0.0002005398380024,
            unit: "s",
          },
          {
            name: "const_scatter / DefOpt / tpu / Forward",
            value: 0.0002165821969974,
            unit: "s",
          },
          {
            name: "const_scatter / IDefOpt / tpu / Forward",
            value: 0.0002203057869992,
            unit: "s",
          },
          {
            name: "const_scatter / JaXPipe / cpu / Primal",
            value: 0.000006227351000234194,
            unit: "s",
          },
          {
            name: "const_scatter / Jax / cpu / Primal",
            value: 0.000006272657000408799,
            unit: "s",
          },
          {
            name: "const_scatter / HLOOpt / cpu / Primal",
            value: 0.000007056097000713635,
            unit: "s",
          },
          {
            name: "const_scatter / PartOpt / cpu / Primal",
            value: 0.000006315691000054357,
            unit: "s",
          },
          {
            name: "const_scatter / IPartOpt / cpu / Primal",
            value: 0.0000066225940008735054,
            unit: "s",
          },
          {
            name: "const_scatter / DefOpt / cpu / Primal",
            value: 0.000007047005999993416,
            unit: "s",
          },
          {
            name: "const_scatter / IDefOpt / cpu / Primal",
            value: 0.00000745372699930158,
            unit: "s",
          },
          {
            name: "const_scatter / JaXPipe / cpu / Forward",
            value: 0.0000109989399998085,
            unit: "s",
          },
          {
            name: "const_scatter / Jax / cpu / Forward",
            value: 0.000008931227999710245,
            unit: "s",
          },
          {
            name: "const_scatter / HLOOpt / cpu / Forward",
            value: 0.000010824940000020434,
            unit: "s",
          },
          {
            name: "const_scatter / PartOpt / cpu / Forward",
            value: 0.0000108079829997223,
            unit: "s",
          },
          {
            name: "const_scatter / IPartOpt / cpu / Forward",
            value: 0.000010800981000102184,
            unit: "s",
          },
          {
            name: "const_scatter / DefOpt / cpu / Forward",
            value: 0.000010766692000288458,
            unit: "s",
          },
          {
            name: "const_scatter / IDefOpt / cpu / Forward",
            value: 0.000010667618999832484,
            unit: "s",
          },
          {
            name: "const_scatter / JaXPipe / cpu / Primal",
            value: 0.000004617750000079468,
            unit: "s",
          },
          {
            name: "const_scatter / Jax / cpu / Primal",
            value: 0.0000051185830000122224,
            unit: "s",
          },
          {
            name: "const_scatter / HLOOpt / cpu / Primal",
            value: 0.000005829500000118059,
            unit: "s",
          },
          {
            name: "const_scatter / PartOpt / cpu / Primal",
            value: 0.000004567624999936015,
            unit: "s",
          },
          {
            name: "const_scatter / IPartOpt / cpu / Primal",
            value: 0.000004227458999821465,
            unit: "s",
          },
          {
            name: "const_scatter / DefOpt / cpu / Primal",
            value: 0.000005403207999961523,
            unit: "s",
          },
          {
            name: "const_scatter / IDefOpt / cpu / Primal",
            value: 0.000004840083000090089,
            unit: "s",
          },
          {
            name: "const_scatter / JaXPipe / cpu / Forward",
            value: 0.000009258832999876176,
            unit: "s",
          },
          {
            name: "const_scatter / Jax / cpu / Forward",
            value: 0.00000757249999992382,
            unit: "s",
          },
          {
            name: "const_scatter / HLOOpt / cpu / Forward",
            value: 0.0000095567089999804,
            unit: "s",
          },
          {
            name: "const_scatter / PartOpt / cpu / Forward",
            value: 0.000008574749999979759,
            unit: "s",
          },
          {
            name: "const_scatter / IPartOpt / cpu / Forward",
            value: 0.000009906167000053756,
            unit: "s",
          },
          {
            name: "const_scatter / DefOpt / cpu / Forward",
            value: 0.000008800125000107072,
            unit: "s",
          },
          {
            name: "const_scatter / IDefOpt / cpu / Forward",
            value: 0.00000738412500004415,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / Primal",
            value: 0.0000051078899996355175,
            unit: "s",
          },
          {
            name: "GenDot / Jax / cpu / Primal",
            value: 0.000005061887999545433,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / Primal",
            value: 0.00000532969000050798,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / Primal",
            value: 0.000004973763999259972,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / Primal",
            value: 0.000004881141999248939,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / Primal",
            value: 0.0000053253209998729286,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / Primal",
            value: 0.0000053007799997431,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / Forward",
            value: 0.000008354357999451168,
            unit: "s",
          },
          {
            name: "GenDot / Jax / cpu / Forward",
            value: 0.00000755946900062554,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / Forward",
            value: 0.00000823668999964866,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / Forward",
            value: 0.000008226445000218519,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / Forward",
            value: 0.000008106657000098494,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / Forward",
            value: 0.000008189690000108384,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / Forward",
            value: 0.000008137338000778982,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / PreRev",
            value: 0.00000825537200034887,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / PostRev",
            value: 0.000007589748000100372,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / BothRev",
            value: 0.00000816652299999987,
            unit: "s",
          },
          {
            name: "GenDot / Jax / cpu / BothRev",
            value: 0.000007602405999932671,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / PreRev",
            value: 0.000008195359999263019,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / PostRev",
            value: 0.000008210835000681982,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / BothRev",
            value: 0.000008164885999576655,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / PreRev",
            value: 0.00000821525900028064,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / PostRev",
            value: 0.000007607159999679425,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / BothRev",
            value: 0.00000813250400005927,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / PreRev",
            value: 0.000008148889000040072,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / PostRev",
            value: 0.00000748280599964346,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / BothRev",
            value: 0.000008158843000273918,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / PreRev",
            value: 0.00000818094599981123,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / PostRev",
            value: 0.000008135187999869232,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / BothRev",
            value: 0.000008198025000638153,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / PreRev",
            value: 0.00000817354599985265,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / PostRev",
            value: 0.00000874511599977268,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / BothRev",
            value: 0.000008201300000109768,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / Primal",
            value: 0.000009083490000193706,
            unit: "s",
          },
          {
            name: "GenDot / Jax / cpu / Primal",
            value: 0.00001005285700011882,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / Primal",
            value: 0.00001146931599942036,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / Primal",
            value: 0.000009527662999971654,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / Primal",
            value: 0.000009560406000673536,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / Primal",
            value: 0.000010868152001421547,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / Primal",
            value: 0.000010944269999527024,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / Forward",
            value: 0.000015722131998700205,
            unit: "s",
          },
          {
            name: "GenDot / Jax / cpu / Forward",
            value: 0.000013525313999707578,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / Forward",
            value: 0.000016261576000033527,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / Forward",
            value: 0.00001648852400103351,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / Forward",
            value: 0.000016526482999324798,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / Forward",
            value: 0.00001654552599939052,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / Forward",
            value: 0.000015619664000041668,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / PreRev",
            value: 0.00001573409199954767,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / PostRev",
            value: 0.000013638918999276938,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / BothRev",
            value: 0.000015766795999297757,
            unit: "s",
          },
          {
            name: "GenDot / Jax / cpu / BothRev",
            value: 0.000014657316000011631,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / PreRev",
            value: 0.00001665347199923417,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / PostRev",
            value: 0.00001659248599935381,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / BothRev",
            value: 0.000016556831000343664,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / PreRev",
            value: 0.000016585384000791236,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / PostRev",
            value: 0.000013688816001376836,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / BothRev",
            value: 0.000015864388000409236,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / PreRev",
            value: 0.00002010660199994163,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / PostRev",
            value: 0.000014190951000273344,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / BothRev",
            value: 0.000015918740999040894,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / PreRev",
            value: 0.00001665441600016493,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / PostRev",
            value: 0.000016650585999741453,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / BothRev",
            value: 0.00001571556799899554,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / PreRev",
            value: 0.000016811845998745412,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / PostRev",
            value: 0.00001667926099980832,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / BothRev",
            value: 0.00001646824599993124,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / tpu / Primal",
            value: 0.0001452492779972,
            unit: "s",
          },
          {
            name: "GenDot / Jax / tpu / Primal",
            value: 0.0001476150880007,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / tpu / Primal",
            value: 0.0001472593270009,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / tpu / Primal",
            value: 0.0001463267879989,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / tpu / Primal",
            value: 0.0001325308580017,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / tpu / Primal",
            value: 0.0001468911580013,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / tpu / Primal",
            value: 0.000145542547998,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / tpu / Forward",
            value: 0.0002100008470006,
            unit: "s",
          },
          {
            name: "GenDot / Jax / tpu / Forward",
            value: 0.0002184019870001,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / tpu / Forward",
            value: 0.0002252520670008,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / tpu / Forward",
            value: 0.0002186076270008,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / tpu / Forward",
            value: 0.0002023806180004,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / tpu / Forward",
            value: 0.0002010856170018,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / tpu / Forward",
            value: 0.0002146473969987,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / tpu / PreRev",
            value: 0.0002008752469992,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / tpu / PostRev",
            value: 0.0001972851769969,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / tpu / BothRev",
            value: 0.0001999833480003,
            unit: "s",
          },
          {
            name: "GenDot / Jax / tpu / BothRev",
            value: 0.0001945735070003,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / tpu / PreRev",
            value: 0.0001940555370019,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / tpu / PostRev",
            value: 0.0002086376570005,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / tpu / BothRev",
            value: 0.0002111285770006,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / tpu / PreRev",
            value: 0.0002059178970011,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / tpu / PostRev",
            value: 0.0001963183869993,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / tpu / BothRev",
            value: 0.0002044792969973,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / tpu / PreRev",
            value: 0.0001972080779996,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / tpu / PostRev",
            value: 0.0001933279170007,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / tpu / BothRev",
            value: 0.0001943239569991,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / tpu / PreRev",
            value: 0.0001955888669981,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / tpu / PostRev",
            value: 0.0002019897870013,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / tpu / BothRev",
            value: 0.0001994025670028,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / tpu / PreRev",
            value: 0.0001960658870011,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / tpu / PostRev",
            value: 0.0001974810870015,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / tpu / BothRev",
            value: 0.0001961780679994,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / Primal",
            value: 0.000006619267000132822,
            unit: "s",
          },
          {
            name: "GenDot / Jax / cpu / Primal",
            value: 0.000006604731999686919,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / Primal",
            value: 0.000007439451999744051,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / Primal",
            value: 0.000006699524000396195,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / Primal",
            value: 0.0000066764590001184845,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / Primal",
            value: 0.000007400403000247025,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / Primal",
            value: 0.000007493466000596527,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / Forward",
            value: 0.000010779009000543738,
            unit: "s",
          },
          {
            name: "GenDot / Jax / cpu / Forward",
            value: 0.000009975541000130762,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / Forward",
            value: 0.000011505896999551624,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / Forward",
            value: 0.00001114460799999506,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / Forward",
            value: 0.000011626377000538924,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / Forward",
            value: 0.000011710929999935616,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / Forward",
            value: 0.000011454659999799332,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / PreRev",
            value: 0.000011177246999977796,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / PostRev",
            value: 0.000010216102999947908,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / BothRev",
            value: 0.000011481202999675587,
            unit: "s",
          },
          {
            name: "GenDot / Jax / cpu / BothRev",
            value: 0.000010408182999526615,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / PreRev",
            value: 0.000011499750000439235,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / PostRev",
            value: 0.000011605163000240282,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / BothRev",
            value: 0.000011432131999754346,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / PreRev",
            value: 0.000010906776999945576,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / PostRev",
            value: 0.0000102430569995704,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / BothRev",
            value: 0.000010852566999346892,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / PreRev",
            value: 0.000011490922000120918,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / PostRev",
            value: 0.000010151829000278666,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / BothRev",
            value: 0.000011428037999394292,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / PreRev",
            value: 0.00001147952399969654,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / PostRev",
            value: 0.000011448127999756252,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / BothRev",
            value: 0.000011532260000421957,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / PreRev",
            value: 0.000011356705999787665,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / PostRev",
            value: 0.000011429735999627157,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / BothRev",
            value: 0.000011424175000684043,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / Primal",
            value: 0.000004461834000039744,
            unit: "s",
          },
          {
            name: "GenDot / Jax / cpu / Primal",
            value: 0.000004449582999995982,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / Primal",
            value: 0.00000494137500004399,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / Primal",
            value: 0.000005475333000049431,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / Primal",
            value: 0.000004596500000161541,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / Primal",
            value: 0.000005060166000021127,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / Primal",
            value: 0.000004666042000053494,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / Forward",
            value: 0.000008613499999910346,
            unit: "s",
          },
          {
            name: "GenDot / Jax / cpu / Forward",
            value: 0.000007624500000019907,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / Forward",
            value: 0.000008556750000025203,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / Forward",
            value: 0.000009212249999791311,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / Forward",
            value: 0.000008814874999870881,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / Forward",
            value: 0.000008130708999942727,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / Forward",
            value: 0.000008959332999893377,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / PreRev",
            value: 0.000009147750000011002,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / PostRev",
            value: 0.000007144874999994499,
            unit: "s",
          },
          {
            name: "GenDot / JaXPipe / cpu / BothRev",
            value: 0.00000939704100005656,
            unit: "s",
          },
          {
            name: "GenDot / Jax / cpu / BothRev",
            value: 0.000007454291999920315,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / PreRev",
            value: 0.000011079333999987285,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / PostRev",
            value: 0.000008576167000001078,
            unit: "s",
          },
          {
            name: "GenDot / HLOOpt / cpu / BothRev",
            value: 0.000007880624999870633,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / PreRev",
            value: 0.000007917375000033644,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / PostRev",
            value: 0.000006888582999863502,
            unit: "s",
          },
          {
            name: "GenDot / PartOpt / cpu / BothRev",
            value: 0.000008526500000016313,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / PreRev",
            value: 0.000008644875000072715,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / PostRev",
            value: 0.000008454290999907243,
            unit: "s",
          },
          {
            name: "GenDot / IPartOpt / cpu / BothRev",
            value: 0.000008818250000103944,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / PreRev",
            value: 0.000008553707999908511,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / PostRev",
            value: 0.000009076042000060625,
            unit: "s",
          },
          {
            name: "GenDot / DefOpt / cpu / BothRev",
            value: 0.000008048250000001644,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / PreRev",
            value: 0.000008711541999900873,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / PostRev",
            value: 0.000008103667000114,
            unit: "s",
          },
          {
            name: "GenDot / IDefOpt / cpu / BothRev",
            value: 0.000007440458000019135,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / Primal",
            value: 0.000007975326000632777,
            unit: "s",
          },
          {
            name: "hlo_ffi / Jax / cpu / Primal",
            value: 0.0000075870749997193345,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / Primal",
            value: 0.000007846911000342516,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / Primal",
            value: 0.00000767126500068116,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / Primal",
            value: 0.000007904742999926384,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / Primal",
            value: 0.00000774948799971753,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / Primal",
            value: 0.00000778847599940491,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / Forward",
            value: 0.000012507217000347737,
            unit: "s",
          },
          {
            name: "hlo_ffi / Jax / cpu / Forward",
            value: 0.000012290390999623925,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / Forward",
            value: 0.000012334994999946502,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / Forward",
            value: 0.000012417390000337036,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / Forward",
            value: 0.000012325770000643389,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / Forward",
            value: 0.000012404185999912445,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / Forward",
            value: 0.000012503721999564732,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / PreRev",
            value: 0.000011700779999955558,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / PostRev",
            value: 0.00001234163100070873,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / BothRev",
            value: 0.00001196895199973369,
            unit: "s",
          },
          {
            name: "hlo_ffi / Jax / cpu / BothRev",
            value: 0.000011890771000253151,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / PreRev",
            value: 0.000011954507999689668,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / PostRev",
            value: 0.00001160096599960525,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / BothRev",
            value: 0.000011976599000263375,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / PreRev",
            value: 0.000012087439999959315,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / PostRev",
            value: 0.000011729507999916678,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / BothRev",
            value: 0.000011611765999987255,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / PreRev",
            value: 0.00001176869800019631,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / PostRev",
            value: 0.000011687900000652007,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / BothRev",
            value: 0.00001169894899976498,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / PreRev",
            value: 0.000011729484000170489,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / PostRev",
            value: 0.000011709727999914322,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / BothRev",
            value: 0.000011969033000241323,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / PreRev",
            value: 0.000011990907999461342,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / PostRev",
            value: 0.000011713799000062864,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / BothRev",
            value: 0.000011706351000611905,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / Primal",
            value: 0.00001421932900120737,
            unit: "s",
          },
          {
            name: "hlo_ffi / Jax / cpu / Primal",
            value: 0.000013564890999987255,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / Primal",
            value: 0.000013502431000233628,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / Primal",
            value: 0.000013571121000495625,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / Primal",
            value: 0.000014137397000013152,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / Primal",
            value: 0.000014235487000405557,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / Primal",
            value: 0.000014284359000157565,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / Forward",
            value: 0.000019431151999015128,
            unit: "s",
          },
          {
            name: "hlo_ffi / Jax / cpu / Forward",
            value: 0.000019575230000555166,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / Forward",
            value: 0.00002004612500059011,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / Forward",
            value: 0.000020694370999990497,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / Forward",
            value: 0.000020636287999877825,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / Forward",
            value: 0.000020558302001518315,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / Forward",
            value: 0.00002067640599852893,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / PreRev",
            value: 0.000019192359999578916,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / PostRev",
            value: 0.00002003085800060944,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / BothRev",
            value: 0.00001905204399918148,
            unit: "s",
          },
          {
            name: "hlo_ffi / Jax / cpu / BothRev",
            value: 0.00002009622399964428,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / PreRev",
            value: 0.000019160813000780765,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / PostRev",
            value: 0.000019930558000851302,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / BothRev",
            value: 0.000020036845999129582,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / PreRev",
            value: 0.00002005521200044313,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / PostRev",
            value: 0.000019932922999942093,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / BothRev",
            value: 0.00001915442800054734,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / PreRev",
            value: 0.000020035591998748715,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / PostRev",
            value: 0.000019950892001361352,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / BothRev",
            value: 0.00001989488000072015,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / PreRev",
            value: 0.00001987242200084438,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / PostRev",
            value: 0.00001937849299974914,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / BothRev",
            value: 0.00001966962299957231,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / PreRev",
            value: 0.000020131268000113778,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / PostRev",
            value: 0.00001980403100060357,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / BothRev",
            value: 0.000019838422000248103,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / tpu / Primal",
            value: 0.0002284358270007,
            unit: "s",
          },
          {
            name: "hlo_ffi / Jax / tpu / Primal",
            value: 0.0002275626869995,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / tpu / Primal",
            value: 0.0002007026970022,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / tpu / Primal",
            value: 0.0002055394570015,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / tpu / Primal",
            value: 0.000197257527001,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / tpu / Primal",
            value: 0.0001805548869997,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / tpu / Primal",
            value: 0.000186696648001,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / tpu / Forward",
            value: 0.0001930603679975,
            unit: "s",
          },
          {
            name: "hlo_ffi / Jax / tpu / Forward",
            value: 0.0001938115569973,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / tpu / Forward",
            value: 0.0002435982469978,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / tpu / Forward",
            value: 0.0001994156069995,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / tpu / Forward",
            value: 0.0002022162080029,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / tpu / Forward",
            value: 0.0002354133670014,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / tpu / Forward",
            value: 0.0002346710970014,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / tpu / PreRev",
            value: 0.000236157067,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / tpu / PostRev",
            value: 0.0002372945869974,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / tpu / BothRev",
            value: 0.0002456109660015,
            unit: "s",
          },
          {
            name: "hlo_ffi / Jax / tpu / BothRev",
            value: 0.0002297401659998,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / tpu / PreRev",
            value: 0.000227417987,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / tpu / PostRev",
            value: 0.0002196455269986,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / tpu / BothRev",
            value: 0.0002380080170005,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / tpu / PreRev",
            value: 0.0002413479469978,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / tpu / PostRev",
            value: 0.0002363213060016,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / tpu / BothRev",
            value: 0.0002266785970023,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / tpu / PreRev",
            value: 0.000231600275998,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / tpu / PostRev",
            value: 0.000235219266,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / tpu / BothRev",
            value: 0.000235515266002,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / tpu / PreRev",
            value: 0.0002338145169997,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / tpu / PostRev",
            value: 0.0002251025769983,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / tpu / BothRev",
            value: 0.0002256460159987,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / tpu / PreRev",
            value: 0.000232484387001,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / tpu / PostRev",
            value: 0.0002354338769982,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / tpu / BothRev",
            value: 0.0002341685669998,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / Primal",
            value: 0.000009730873000080465,
            unit: "s",
          },
          {
            name: "hlo_ffi / Jax / cpu / Primal",
            value: 0.000009231633000126748,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / Primal",
            value: 0.000009293058999901404,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / Primal",
            value: 0.000009699269000520872,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / Primal",
            value: 0.000009632715999941864,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / Primal",
            value: 0.0000096977990006053,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / Primal",
            value: 0.000009681492000709112,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / Forward",
            value: 0.000013737625000430854,
            unit: "s",
          },
          {
            name: "hlo_ffi / Jax / cpu / Forward",
            value: 0.000013601123000626104,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / Forward",
            value: 0.000013731961000303272,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / Forward",
            value: 0.000014125380000223233,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / Forward",
            value: 0.000014233683999918866,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / Forward",
            value: 0.000014250012999582395,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / Forward",
            value: 0.000014131533000181662,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / PreRev",
            value: 0.000013332428000467187,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / PostRev",
            value: 0.000013288354999531293,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / BothRev",
            value: 0.000014006242000505152,
            unit: "s",
          },
          {
            name: "hlo_ffi / Jax / cpu / BothRev",
            value: 0.000013972660999570508,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / PreRev",
            value: 0.000014019145000020216,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / PostRev",
            value: 0.000013949156000307996,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / BothRev",
            value: 0.000014049766999960412,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / PreRev",
            value: 0.00001340357099979883,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / PostRev",
            value: 0.000013802237000163588,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / BothRev",
            value: 0.000013374707999901147,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / PreRev",
            value: 0.000014130530999864278,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / PostRev",
            value: 0.000013899042999582888,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / BothRev",
            value: 0.000013905387999329832,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / PreRev",
            value: 0.000014042684000742157,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / PostRev",
            value: 0.000013955450000139535,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / BothRev",
            value: 0.000013990329000080238,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / PreRev",
            value: 0.000014027480000549986,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / PostRev",
            value: 0.00001404215300044598,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / BothRev",
            value: 0.000014008722000653508,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / Primal",
            value: 0.000006479249999983949,
            unit: "s",
          },
          {
            name: "hlo_ffi / Jax / cpu / Primal",
            value: 0.000006284999999934371,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / Primal",
            value: 0.000006132374999879176,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / Primal",
            value: 0.000006537957999853461,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / Primal",
            value: 0.000006021000000146159,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / Primal",
            value: 0.000006637458000113838,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / Primal",
            value: 0.000006527874999846972,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / Forward",
            value: 0.000009068709000075614,
            unit: "s",
          },
          {
            name: "hlo_ffi / Jax / cpu / Forward",
            value: 0.000010041249999858335,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / Forward",
            value: 0.000009946166999952766,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / Forward",
            value: 0.000010376958999813724,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / Forward",
            value: 0.00001218149999999696,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / Forward",
            value: 0.000014821249999840802,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / Forward",
            value: 0.000010978750000049332,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / PreRev",
            value: 0.000011819291999927372,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / PostRev",
            value: 0.00000915245800001685,
            unit: "s",
          },
          {
            name: "hlo_ffi / JaXPipe / cpu / BothRev",
            value: 0.000009059457999910593,
            unit: "s",
          },
          {
            name: "hlo_ffi / Jax / cpu / BothRev",
            value: 0.000008878666000100565,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / PreRev",
            value: 0.00000978454200003398,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / PostRev",
            value: 0.00000929491700003382,
            unit: "s",
          },
          {
            name: "hlo_ffi / HLOOpt / cpu / BothRev",
            value: 0.000009182999999893582,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / PreRev",
            value: 0.000009946042000137823,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / PostRev",
            value: 0.00001117554100005691,
            unit: "s",
          },
          {
            name: "hlo_ffi / PartOpt / cpu / BothRev",
            value: 0.00001034670800004278,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / PreRev",
            value: 0.00001128804199993283,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / PostRev",
            value: 0.000009473542000023371,
            unit: "s",
          },
          {
            name: "hlo_ffi / IPartOpt / cpu / BothRev",
            value: 0.000011405875000036758,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / PreRev",
            value: 0.000011933624999983294,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / PostRev",
            value: 0.000009186042000010274,
            unit: "s",
          },
          {
            name: "hlo_ffi / DefOpt / cpu / BothRev",
            value: 0.000009156374999975017,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / PreRev",
            value: 0.000009855916999867986,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / PostRev",
            value: 0.000009266666000030457,
            unit: "s",
          },
          {
            name: "hlo_ffi / IDefOpt / cpu / BothRev",
            value: 0.000012780042000031243,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / Primal",
            value: 0.0008126364999952,
            unit: "s",
          },
          {
            name: "llama / Jax / cpu / Primal",
            value: 0.0008336582999618,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / Primal",
            value: 0.0008965353999883,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / Primal",
            value: 0.0008196690000659,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / Primal",
            value: 0.0008182796000255,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / Primal",
            value: 0.0009063810000043,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / Primal",
            value: 0.0008977847999631,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / Forward",
            value: 0.0022825532999377,
            unit: "s",
          },
          {
            name: "llama / Jax / cpu / Forward",
            value: 0.002315838999948,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / Forward",
            value: 0.0023423085999638,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / Forward",
            value: 0.002362508599981,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / Forward",
            value: 0.0023303062000195,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / Forward",
            value: 0.0024156385000424,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / Forward",
            value: 0.0023881656000412,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / PreRev",
            value: 0.0046469655999317,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / PostRev",
            value: 0.0058002689000204,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / BothRev",
            value: 0.0060796140999627,
            unit: "s",
          },
          {
            name: "llama / Jax / cpu / BothRev",
            value: 0.0065971383999567,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / PreRev",
            value: 0.0052461166000284,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / PostRev",
            value: 0.0046174027999768,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / BothRev",
            value: 0.0066538312999909,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / PreRev",
            value: 0.0052145867000035,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / PostRev",
            value: 0.0047658813999987,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / BothRev",
            value: 0.0064109453999662,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / PreRev",
            value: 0.004996103600024,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / PostRev",
            value: 0.0050073008999788,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / BothRev",
            value: 0.0046879992999492,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / PreRev",
            value: 0.0065080289000434,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / PostRev",
            value: 0.0049510340000779,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / BothRev",
            value: 0.0046523672999683,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / PreRev",
            value: 0.0067062789000374,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / PostRev",
            value: 0.0050827691999984,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / BothRev",
            value: 0.0046603404000052,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / Primal",
            value: 0.0015728055999716,
            unit: "s",
          },
          {
            name: "llama / Jax / cpu / Primal",
            value: 0.0015300603001378,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / Primal",
            value: 0.0016851773998496,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / Primal",
            value: 0.0015412979000757,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / Primal",
            value: 0.0015218346999972,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / Primal",
            value: 0.00163698080014,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / Primal",
            value: 0.0016669419001118,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / Forward",
            value: 0.0052297074998932,
            unit: "s",
          },
          {
            name: "llama / Jax / cpu / Forward",
            value: 0.0054262168001514,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / Forward",
            value: 0.005234185900008,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / Forward",
            value: 0.0052223352999135,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / Forward",
            value: 0.0052652295998996,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / Forward",
            value: 0.0052912567998646,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / Forward",
            value: 0.0052916732000085,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / PreRev",
            value: 0.0097171299999899,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / PostRev",
            value: 0.0105009462999078,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / BothRev",
            value: 0.0073765207000178,
            unit: "s",
          },
          {
            name: "llama / Jax / cpu / BothRev",
            value: 0.0104555737001646,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / PreRev",
            value: 0.0075528692999796,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / PostRev",
            value: 0.0096935581001162,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / BothRev",
            value: 0.0075003677000495,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / PreRev",
            value: 0.0096072581998669,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / PostRev",
            value: 0.0075494431999686,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / BothRev",
            value: 0.0103238539999438,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / PreRev",
            value: 0.0075147573999856,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / PostRev",
            value: 0.0102124249000553,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / BothRev",
            value: 0.0077080611999917,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / PreRev",
            value: 0.0099857942999733,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / PostRev",
            value: 0.0070135918998857,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / BothRev",
            value: 0.0097803880000356,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / PreRev",
            value: 0.0075505302000237,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / PostRev",
            value: 0.0095953691999966,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / BothRev",
            value: 0.007479892000083,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / tpu / Primal",
            value: 0.0003454404000513,
            unit: "s",
          },
          {
            name: "llama / Jax / tpu / Primal",
            value: 0.0003494279799633,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / tpu / Primal",
            value: 0.0003506665999884,
            unit: "s",
          },
          {
            name: "llama / PartOpt / tpu / Primal",
            value: 0.0003587224000511,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / tpu / Primal",
            value: 0.0003644358000019,
            unit: "s",
          },
          {
            name: "llama / DefOpt / tpu / Primal",
            value: 0.000319286399972,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / tpu / Primal",
            value: 0.0003425692000018,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / tpu / Forward",
            value: 0.0005455915999482,
            unit: "s",
          },
          {
            name: "llama / Jax / tpu / Forward",
            value: 0.0006826730000466,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / tpu / Forward",
            value: 0.0005447142000048,
            unit: "s",
          },
          {
            name: "llama / PartOpt / tpu / Forward",
            value: 0.0005281455800286,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / tpu / Forward",
            value: 0.0005330218000017,
            unit: "s",
          },
          {
            name: "llama / DefOpt / tpu / Forward",
            value: 0.0005391159800637,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / tpu / Forward",
            value: 0.0005518421799934,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / tpu / PreRev",
            value: 0.0007639519799704,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / tpu / PostRev",
            value: 0.000710435180008,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / tpu / BothRev",
            value: 0.0007357559799856,
            unit: "s",
          },
          {
            name: "llama / Jax / tpu / BothRev",
            value: 0.0007136112000443,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / tpu / PreRev",
            value: 0.0007428143799916,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / tpu / PostRev",
            value: 0.0007858008000039,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / tpu / BothRev",
            value: 0.0007882525800232,
            unit: "s",
          },
          {
            name: "llama / PartOpt / tpu / PreRev",
            value: 0.0007912403999944,
            unit: "s",
          },
          {
            name: "llama / PartOpt / tpu / PostRev",
            value: 0.0007177704000059,
            unit: "s",
          },
          {
            name: "llama / PartOpt / tpu / BothRev",
            value: 0.000788979400022,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / tpu / PreRev",
            value: 0.000781859780036,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / tpu / PostRev",
            value: 0.0007672226000431,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / tpu / BothRev",
            value: 0.0007707256000139,
            unit: "s",
          },
          {
            name: "llama / DefOpt / tpu / PreRev",
            value: 0.0007739223999669,
            unit: "s",
          },
          {
            name: "llama / DefOpt / tpu / PostRev",
            value: 0.0007283361800364,
            unit: "s",
          },
          {
            name: "llama / DefOpt / tpu / BothRev",
            value: 0.0007855499799916,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / tpu / PreRev",
            value: 0.0007793509799375,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / tpu / PostRev",
            value: 0.000791105199969,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / tpu / BothRev",
            value: 0.0007790505800221,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / Primal",
            value: 0.0013888022999708,
            unit: "s",
          },
          {
            name: "llama / Jax / cpu / Primal",
            value: 0.0013030844999775,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / Primal",
            value: 0.0012550415000077,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / Primal",
            value: 0.0012249639999936,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / Primal",
            value: 0.0012504220999289,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / Primal",
            value: 0.0013971006999781,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / Primal",
            value: 0.0014504046999718,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / Forward",
            value: 0.0046246202000475,
            unit: "s",
          },
          {
            name: "llama / Jax / cpu / Forward",
            value: 0.0046610448000137,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / Forward",
            value: 0.0045824140000149,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / Forward",
            value: 0.0045635192999725,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / Forward",
            value: 0.0042956528000104,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / Forward",
            value: 0.004237977699995,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / Forward",
            value: 0.0044747938999535,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / PreRev",
            value: 0.0085583247999238,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / PostRev",
            value: 0.0084544145000108,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / BothRev",
            value: 0.009028416700039,
            unit: "s",
          },
          {
            name: "llama / Jax / cpu / BothRev",
            value: 0.0079370874000233,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / PreRev",
            value: 0.009177245399951,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / PostRev",
            value: 0.0077456017999793,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / BothRev",
            value: 0.0090469296999799,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / PreRev",
            value: 0.0090144136000162,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / PostRev",
            value: 0.0077976749999834,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / BothRev",
            value: 0.0090285846000369,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / PreRev",
            value: 0.0094041792999632,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / PostRev",
            value: 0.0079918573999748,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / BothRev",
            value: 0.0091304014999877,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / PreRev",
            value: 0.0087251052999818,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / PostRev",
            value: 0.0084128851000059,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / BothRev",
            value: 0.0086288047999914,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / PreRev",
            value: 0.0073525458000403,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / PostRev",
            value: 0.0085766506999789,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / BothRev",
            value: 0.0087042418999772,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / Primal",
            value: 0.0019260666000036,
            unit: "s",
          },
          {
            name: "llama / Jax / cpu / Primal",
            value: 0.0022952416999942,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / Primal",
            value: 0.0018398707999949,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / Primal",
            value: 0.0025812165999923,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / Primal",
            value: 0.0024862500000153,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / Primal",
            value: 0.0021841416999905,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / Primal",
            value: 0.0022737792000043,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / Forward",
            value: 0.0073248084000169,
            unit: "s",
          },
          {
            name: "llama / Jax / cpu / Forward",
            value: 0.0068525417000046,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / Forward",
            value: 0.0072469583000156,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / Forward",
            value: 0.008300687499991,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / Forward",
            value: 0.0076951540999971,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / Forward",
            value: 0.0071677458000067,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / Forward",
            value: 0.0044293208999988,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / PreRev",
            value: 0.010033162500008,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / PostRev",
            value: 0.0111209250000001,
            unit: "s",
          },
          {
            name: "llama / JaXPipe / cpu / BothRev",
            value: 0.0100600374999885,
            unit: "s",
          },
          {
            name: "llama / Jax / cpu / BothRev",
            value: 0.011301037499993,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / PreRev",
            value: 0.0092611999999917,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / PostRev",
            value: 0.009893216599994,
            unit: "s",
          },
          {
            name: "llama / HLOOpt / cpu / BothRev",
            value: 0.0094405999999935,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / PreRev",
            value: 0.0113036707999981,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / PostRev",
            value: 0.0109872041999778,
            unit: "s",
          },
          {
            name: "llama / PartOpt / cpu / BothRev",
            value: 0.0115517917000033,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / PreRev",
            value: 0.0084736165999856,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / PostRev",
            value: 0.0129817666999997,
            unit: "s",
          },
          {
            name: "llama / IPartOpt / cpu / BothRev",
            value: 0.0117904167000006,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / PreRev",
            value: 0.0077027541000006,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / PostRev",
            value: 0.0124172875000112,
            unit: "s",
          },
          {
            name: "llama / DefOpt / cpu / BothRev",
            value: 0.0094163457999911,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / PreRev",
            value: 0.0109625416999961,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / PostRev",
            value: 0.007950158299991,
            unit: "s",
          },
          {
            name: "llama / IDefOpt / cpu / BothRev",
            value: 0.0086084457999959,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / Primal",
            value: 0.000005600796999715385,
            unit: "s",
          },
          {
            name: "scatter_sum / Jax / cpu / Primal",
            value: 0.000005528659000447078,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / Primal",
            value: 0.00000559852599963051,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / Primal",
            value: 0.000005583551999734482,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / Primal",
            value: 0.000005491681999956199,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / Primal",
            value: 0.000005575610999585479,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / Primal",
            value: 0.000005501501000253484,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / Forward",
            value: 0.000008566445999349525,
            unit: "s",
          },
          {
            name: "scatter_sum / Jax / cpu / Forward",
            value: 0.000008624146999864024,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / Forward",
            value: 0.000008738122000067961,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / Forward",
            value: 0.000008610111000052712,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / Forward",
            value: 0.000008733741000469308,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / Forward",
            value: 0.000008675885999764433,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / Forward",
            value: 0.000008679419000145572,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / PreRev",
            value: 0.000008787356000539148,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / PostRev",
            value: 0.000008680954000737984,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / BothRev",
            value: 0.00000873694300025818,
            unit: "s",
          },
          {
            name: "scatter_sum / Jax / cpu / BothRev",
            value: 0.000008722978999685438,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / PreRev",
            value: 0.00000878291499975603,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / PostRev",
            value: 0.000008775168999818561,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / BothRev",
            value: 0.000008653279000100156,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / PreRev",
            value: 0.000008678230999976221,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / PostRev",
            value: 0.000008700506999957725,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / BothRev",
            value: 0.000008714327000234334,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / PreRev",
            value: 0.000008704512999429425,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / PostRev",
            value: 0.000008709371999430004,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / BothRev",
            value: 0.00000894060799964791,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / PreRev",
            value: 0.000008920821000174328,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / PostRev",
            value: 0.00000877664299969183,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / BothRev",
            value: 0.000008696274000612902,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / PreRev",
            value: 0.000008719108999684977,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / PostRev",
            value: 0.000009261836999939987,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / BothRev",
            value: 0.000008752255000217702,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / Primal",
            value: 0.000011274607999439468,
            unit: "s",
          },
          {
            name: "scatter_sum / Jax / cpu / Primal",
            value: 0.000010706989000027531,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / Primal",
            value: 0.000011200660999747924,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / Primal",
            value: 0.00001077005900151562,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / Primal",
            value: 0.000015665749999243418,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / Primal",
            value: 0.000010855014999833656,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / Primal",
            value: 0.000010784202000650113,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / Forward",
            value: 0.00001618169599896646,
            unit: "s",
          },
          {
            name: "scatter_sum / Jax / cpu / Forward",
            value: 0.000015595002001646206,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / Forward",
            value: 0.00001564045799932501,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / Forward",
            value: 0.000015522096000495366,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / Forward",
            value: 0.00001611552800022764,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / Forward",
            value: 0.00001535660099943925,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / Forward",
            value: 0.00001549648000036541,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / PreRev",
            value: 0.00001632324700040044,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / PostRev",
            value: 0.000016291423000438954,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / BothRev",
            value: 0.00001633960499930254,
            unit: "s",
          },
          {
            name: "scatter_sum / Jax / cpu / BothRev",
            value: 0.000016462655999930574,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / PreRev",
            value: 0.00001570313499905751,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / PostRev",
            value: 0.000015661650000765802,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / BothRev",
            value: 0.00001643132099889044,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / PreRev",
            value: 0.00001637905200004752,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / PostRev",
            value: 0.000016414251998867258,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / BothRev",
            value: 0.000015586974999678205,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / PreRev",
            value: 0.000016441757001302904,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / PostRev",
            value: 0.00001616809299957822,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / BothRev",
            value: 0.000015655920999051887,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / PreRev",
            value: 0.000016274608000458103,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / PostRev",
            value: 0.000016434342000138715,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / BothRev",
            value: 0.000016452154000944573,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / PreRev",
            value: 0.000016450249999252264,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / PostRev",
            value: 0.000016428012999313068,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / BothRev",
            value: 0.00001590921699971659,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / tpu / Primal",
            value: 0.0001339083480015,
            unit: "s",
          },
          {
            name: "scatter_sum / Jax / tpu / Primal",
            value: 0.0001353779780001,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / tpu / Primal",
            value: 0.0001340004379999,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / tpu / Primal",
            value: 0.0001364789879989,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / tpu / Primal",
            value: 0.0001391973780009,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / tpu / Primal",
            value: 0.000138245068003,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / tpu / Primal",
            value: 0.0001351513480003,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / tpu / Forward",
            value: 0.0001947791580023,
            unit: "s",
          },
          {
            name: "scatter_sum / Jax / tpu / Forward",
            value: 0.000195429157,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / tpu / Forward",
            value: 0.0001921954669996,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / tpu / Forward",
            value: 0.0001983690970009,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / tpu / Forward",
            value: 0.0001972065079971,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / tpu / Forward",
            value: 0.0001977097470007,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / tpu / Forward",
            value: 0.0001972844869997,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / tpu / PreRev",
            value: 0.0001996636569965,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / tpu / PostRev",
            value: 0.0002063819670001,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / tpu / BothRev",
            value: 0.0001942770169989,
            unit: "s",
          },
          {
            name: "scatter_sum / Jax / tpu / BothRev",
            value: 0.0001948333270011,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / tpu / PreRev",
            value: 0.0002167204269971,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / tpu / PostRev",
            value: 0.0002221642069998,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / tpu / BothRev",
            value: 0.0002235902370011,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / tpu / PreRev",
            value: 0.0002443054870018,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / tpu / PostRev",
            value: 0.0002300591269995,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / tpu / BothRev",
            value: 0.0002095152870024,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / tpu / PreRev",
            value: 0.0002393167070003,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / tpu / PostRev",
            value: 0.0002251037770001,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / tpu / BothRev",
            value: 0.000226746217002,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / tpu / PreRev",
            value: 0.0002397464460009,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / tpu / PostRev",
            value: 0.0002261898269971,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / tpu / BothRev",
            value: 0.0002200983469992,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / tpu / PreRev",
            value: 0.0002179156769998,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / tpu / PostRev",
            value: 0.0002054798570025,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / tpu / BothRev",
            value: 0.0002177286469996,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / Primal",
            value: 0.000007426015999953961,
            unit: "s",
          },
          {
            name: "scatter_sum / Jax / cpu / Primal",
            value: 0.000007294531000297866,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / Primal",
            value: 0.000007773997000185774,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / Primal",
            value: 0.000007250571000440686,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / Primal",
            value: 0.000007435853000060888,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / Primal",
            value: 0.000007455162999576715,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / Primal",
            value: 0.000007891809999819088,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / Forward",
            value: 0.00001067939899985504,
            unit: "s",
          },
          {
            name: "scatter_sum / Jax / cpu / Forward",
            value: 0.00001132001800033322,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / Forward",
            value: 0.000010765858999548072,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / Forward",
            value: 0.000011305746999823896,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / Forward",
            value: 0.000011300907999611807,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / Forward",
            value: 0.000010751877000075184,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / Forward",
            value: 0.000011344959999405548,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / PreRev",
            value: 0.00001095759000054386,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / PostRev",
            value: 0.00001092298599996866,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / BothRev",
            value: 0.000011409374000322716,
            unit: "s",
          },
          {
            name: "scatter_sum / Jax / cpu / BothRev",
            value: 0.000011521528000230318,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / PreRev",
            value: 0.000011464664000413905,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / PostRev",
            value: 0.000011569231000066793,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / BothRev",
            value: 0.000010908966999522815,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / PreRev",
            value: 0.000011485547999654954,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / PostRev",
            value: 0.000011476358999971123,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / BothRev",
            value: 0.000010924350999630403,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / PreRev",
            value: 0.000011477889000161669,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / PostRev",
            value: 0.00001089940900055808,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / BothRev",
            value: 0.000011463845999969637,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / PreRev",
            value: 0.000011452956000539416,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / PostRev",
            value: 0.000011488060999909069,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / BothRev",
            value: 0.000011500032999720134,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / PreRev",
            value: 0.000011436048000177834,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / PostRev",
            value: 0.000011525351999807754,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / BothRev",
            value: 0.000010956099999930304,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / Primal",
            value: 0.000005028791999848181,
            unit: "s",
          },
          {
            name: "scatter_sum / Jax / cpu / Primal",
            value: 0.000005229957999972612,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / Primal",
            value: 0.0000053496250000080185,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / Primal",
            value: 0.000005361708000009457,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / Primal",
            value: 0.000005258916999991925,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / Primal",
            value: 0.000005755417000045782,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / Primal",
            value: 0.000005272333999982948,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / Forward",
            value: 0.00000941729199985275,
            unit: "s",
          },
          {
            name: "scatter_sum / Jax / cpu / Forward",
            value: 0.00000812333299995771,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / Forward",
            value: 0.000008452416999944035,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / Forward",
            value: 0.000009859291999873676,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / Forward",
            value: 0.000009453124999936336,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / Forward",
            value: 0.00000965895800004546,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / Forward",
            value: 0.000008502708000150961,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / PreRev",
            value: 0.000009766958999989584,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / PostRev",
            value: 0.000007672999999840613,
            unit: "s",
          },
          {
            name: "scatter_sum / JaXPipe / cpu / BothRev",
            value: 0.00000868175000005067,
            unit: "s",
          },
          {
            name: "scatter_sum / Jax / cpu / BothRev",
            value: 0.00000770929200007231,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / PreRev",
            value: 0.000008355250000022353,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / PostRev",
            value: 0.000008004750000054628,
            unit: "s",
          },
          {
            name: "scatter_sum / HLOOpt / cpu / BothRev",
            value: 0.000008927084000106334,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / PreRev",
            value: 0.000008086208999884548,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / PostRev",
            value: 0.000008505624999997962,
            unit: "s",
          },
          {
            name: "scatter_sum / PartOpt / cpu / BothRev",
            value: 0.000009008874999835826,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / PreRev",
            value: 0.000009227708000025814,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / PostRev",
            value: 0.00000860366699998849,
            unit: "s",
          },
          {
            name: "scatter_sum / IPartOpt / cpu / BothRev",
            value: 0.000009490791000189349,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / PreRev",
            value: 0.000009839874999897802,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / PostRev",
            value: 0.000008951167000077476,
            unit: "s",
          },
          {
            name: "scatter_sum / DefOpt / cpu / BothRev",
            value: 0.000008323083000050246,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / PreRev",
            value: 0.00000836837499991816,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / PostRev",
            value: 0.000007900166999888824,
            unit: "s",
          },
          {
            name: "scatter_sum / IDefOpt / cpu / BothRev",
            value: 0.00001156562500000291,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / Primal",
            value: 0.000004494202000387304,
            unit: "s",
          },
          {
            name: "slicing / Jax / cpu / Primal",
            value: 0.000004493826999350858,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / Primal",
            value: 0.000004414877999806777,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / Primal",
            value: 0.000004481727999518626,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / Primal",
            value: 0.000004355505000603444,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / Primal",
            value: 0.000004425664999871514,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / Primal",
            value: 0.0000043954329994448925,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / Forward",
            value: 0.000006897246999869821,
            unit: "s",
          },
          {
            name: "slicing / Jax / cpu / Forward",
            value: 0.00000692277500002092,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / Forward",
            value: 0.000006889262000186136,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / Forward",
            value: 0.000006998007000220241,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / Forward",
            value: 0.000006903106999743613,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / Forward",
            value: 0.000006905776000166952,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / Forward",
            value: 0.000006881952999719942,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / PreRev",
            value: 0.0000073609639994174355,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / PostRev",
            value: 0.000007385296000393282,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / BothRev",
            value: 0.000007422248999318981,
            unit: "s",
          },
          {
            name: "slicing / Jax / cpu / BothRev",
            value: 0.00000737867699990602,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / PreRev",
            value: 0.000007377679999990506,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / PostRev",
            value: 0.000007433879000018351,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / BothRev",
            value: 0.000007369445000222186,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / PreRev",
            value: 0.00000749679999989894,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / PostRev",
            value: 0.000007392670999252005,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / BothRev",
            value: 0.000007437624999511172,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / PreRev",
            value: 0.000007400043999950867,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / PostRev",
            value: 0.000007412350999402406,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / BothRev",
            value: 0.000007452480999745603,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / PreRev",
            value: 0.000007340220000514819,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / PostRev",
            value: 0.000007462425000085205,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / BothRev",
            value: 0.000007392655999865383,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / PreRev",
            value: 0.000007364128000517667,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / PostRev",
            value: 0.000007777314000122716,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / BothRev",
            value: 0.000007409142000142311,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / Primal",
            value: 0.0000086117020000529,
            unit: "s",
          },
          {
            name: "slicing / Jax / cpu / Primal",
            value: 0.000009109144999456476,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / Primal",
            value: 0.000009093230999496882,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / Primal",
            value: 0.000008612583000285667,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / Primal",
            value: 0.00000916681299895572,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / Primal",
            value: 0.00000852543600012723,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / Primal",
            value: 0.00000859589900028368,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / Forward",
            value: 0.00001220751700020628,
            unit: "s",
          },
          {
            name: "slicing / Jax / cpu / Forward",
            value: 0.00001298648299962224,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / Forward",
            value: 0.000013182007998693737,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / Forward",
            value: 0.000013356289999137516,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / Forward",
            value: 0.000013325838999662664,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / Forward",
            value: 0.00001348122299896204,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / Forward",
            value: 0.000012994653001442202,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / PreRev",
            value: 0.000013929282999015411,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / PostRev",
            value: 0.000013199180999436066,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / BothRev",
            value: 0.00001388105799924233,
            unit: "s",
          },
          {
            name: "slicing / Jax / cpu / BothRev",
            value: 0.00001381577199936146,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / PreRev",
            value: 0.000014018455000041286,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / PostRev",
            value: 0.0000138539759991545,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / BothRev",
            value: 0.000013179473000491272,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / PreRev",
            value: 0.0000139702380001836,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / PostRev",
            value: 0.000014013522999448468,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / BothRev",
            value: 0.000013119090999680338,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / PreRev",
            value: 0.00001393540299977758,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / PostRev",
            value: 0.000013240009000583086,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / BothRev",
            value: 0.000013260813999295352,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / PreRev",
            value: 0.000014009595000970876,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / PostRev",
            value: 0.000013977217000501696,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / BothRev",
            value: 0.000013908962999266806,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / PreRev",
            value: 0.000013956361999589718,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / PostRev",
            value: 0.00001394273199912277,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / BothRev",
            value: 0.000014023821000591853,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / tpu / Primal",
            value: 0.0001320969290027,
            unit: "s",
          },
          {
            name: "slicing / Jax / tpu / Primal",
            value: 0.0001421055280006,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / tpu / Primal",
            value: 0.0001325701179994,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / tpu / Primal",
            value: 0.0001314521089989,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / tpu / Primal",
            value: 0.0001413838279986,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / tpu / Primal",
            value: 0.0001317657690015,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / tpu / Primal",
            value: 0.0001314805379988,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / tpu / Forward",
            value: 0.0002201339370003,
            unit: "s",
          },
          {
            name: "slicing / Jax / tpu / Forward",
            value: 0.0001905199570028,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / tpu / Forward",
            value: 0.0001893071670019,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / tpu / Forward",
            value: 0.000189836767,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / tpu / Forward",
            value: 0.0001892658879987,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / tpu / Forward",
            value: 0.000189136986999,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / tpu / Forward",
            value: 0.000183864548002,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / tpu / PreRev",
            value: 0.000185330237,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / tpu / PostRev",
            value: 0.0001846086170007,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / tpu / BothRev",
            value: 0.0001902627470008,
            unit: "s",
          },
          {
            name: "slicing / Jax / tpu / BothRev",
            value: 0.0001897644870005,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / tpu / PreRev",
            value: 0.0002037083180002,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / tpu / PostRev",
            value: 0.0002034914279975,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / tpu / BothRev",
            value: 0.0002039485870009,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / tpu / PreRev",
            value: 0.0002029389070012,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / tpu / PostRev",
            value: 0.0002045688770012,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / tpu / BothRev",
            value: 0.0002034182269999,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / tpu / PreRev",
            value: 0.0001842812880022,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / tpu / PostRev",
            value: 0.000186067696999,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / tpu / BothRev",
            value: 0.0001853904770032,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / tpu / PreRev",
            value: 0.0001866563969997,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / tpu / PostRev",
            value: 0.0001854140180003,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / tpu / BothRev",
            value: 0.000220929056999,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / tpu / PreRev",
            value: 0.0002226304169998,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / tpu / PostRev",
            value: 0.0001879267679978,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / tpu / BothRev",
            value: 0.0002011322470025,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / Primal",
            value: 0.000005998222999551217,
            unit: "s",
          },
          {
            name: "slicing / Jax / cpu / Primal",
            value: 0.000006355801000609063,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / Primal",
            value: 0.00000584371300010389,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / Primal",
            value: 0.0000059487159996933765,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / Primal",
            value: 0.00000598458100012067,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / Primal",
            value: 0.000005992721999973582,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / Primal",
            value: 0.0000059920629992120665,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / Forward",
            value: 0.000009192400000756606,
            unit: "s",
          },
          {
            name: "slicing / Jax / cpu / Forward",
            value: 0.000008626999000625801,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / Forward",
            value: 0.000008703923000211944,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / Forward",
            value: 0.000009244431000297482,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / Forward",
            value: 0.000009153349999905913,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / Forward",
            value: 0.000008688451000125497,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / Forward",
            value: 0.000008711733000382083,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / PreRev",
            value: 0.000009783417000107876,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / PostRev",
            value: 0.000009292507000282056,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / BothRev",
            value: 0.00000980588600032206,
            unit: "s",
          },
          {
            name: "slicing / Jax / cpu / BothRev",
            value: 0.000009843593999903532,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / PreRev",
            value: 0.000009303707000071882,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / PostRev",
            value: 0.000009859476999736217,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / BothRev",
            value: 0.000009952301000339503,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / PreRev",
            value: 0.000009923618999891917,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / PostRev",
            value: 0.00000980691899985686,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / BothRev",
            value: 0.00000925802100027795,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / PreRev",
            value: 0.00000927844299985736,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / PostRev",
            value: 0.000009864630999800283,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / BothRev",
            value: 0.000009776920000149402,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / PreRev",
            value: 0.00000916720900022483,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / PostRev",
            value: 0.000009898072000396496,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / BothRev",
            value: 0.000009849475000009988,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / PreRev",
            value: 0.00000922795699989365,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / PostRev",
            value: 0.000009830171999965386,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / BothRev",
            value: 0.000009323302000666444,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / Primal",
            value: 0.000004723207999859369,
            unit: "s",
          },
          {
            name: "slicing / Jax / cpu / Primal",
            value: 0.000004251874999908978,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / Primal",
            value: 0.000004591416000039317,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / Primal",
            value: 0.000005102249999936248,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / Primal",
            value: 0.000004838999999947192,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / Primal",
            value: 0.000004103165999822522,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / Primal",
            value: 0.0000046228329999848935,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / Forward",
            value: 0.000006966708000163635,
            unit: "s",
          },
          {
            name: "slicing / Jax / cpu / Forward",
            value: 0.000006899291999843627,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / Forward",
            value: 0.000007573666999860507,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / Forward",
            value: 0.000006119417000036265,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / Forward",
            value: 0.000006234708999954819,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / Forward",
            value: 0.000007579791999887675,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / Forward",
            value: 0.000006623999999874286,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / PreRev",
            value: 0.000006972542000085014,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / PostRev",
            value: 0.000006409416999986206,
            unit: "s",
          },
          {
            name: "slicing / JaXPipe / cpu / BothRev",
            value: 0.000006528250000201296,
            unit: "s",
          },
          {
            name: "slicing / Jax / cpu / BothRev",
            value: 0.000006588834000012866,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / PreRev",
            value: 0.000006658166999841342,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / PostRev",
            value: 0.00000639295799987849,
            unit: "s",
          },
          {
            name: "slicing / HLOOpt / cpu / BothRev",
            value: 0.000006440999999995256,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / PreRev",
            value: 0.000006424459000072602,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / PostRev",
            value: 0.000007902791999867986,
            unit: "s",
          },
          {
            name: "slicing / PartOpt / cpu / BothRev",
            value: 0.000008329582999976992,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / PreRev",
            value: 0.00000845191700000214,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / PostRev",
            value: 0.000007647166999959154,
            unit: "s",
          },
          {
            name: "slicing / IPartOpt / cpu / BothRev",
            value: 0.000006925375000037093,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / PreRev",
            value: 0.000006803957999863997,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / PostRev",
            value: 0.000007396167000024434,
            unit: "s",
          },
          {
            name: "slicing / DefOpt / cpu / BothRev",
            value: 0.000007062375000032261,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / PreRev",
            value: 0.000006964207999999417,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / PostRev",
            value: 0.000006670375000112472,
            unit: "s",
          },
          {
            name: "slicing / IDefOpt / cpu / BothRev",
            value: 0.000009368375000121886,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / Primal",
            value: 0.000005711009000151535,
            unit: "s",
          },
          {
            name: "sum    / Jax / cpu / Primal",
            value: 0.0000059789629995066206,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / Primal",
            value: 0.000005979538999781653,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / Primal",
            value: 0.000006031701999745565,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / Primal",
            value: 0.000005996093999783625,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / Primal",
            value: 0.000006009873999573756,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / Primal",
            value: 0.00000606157800029905,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / Forward",
            value: 0.00000932366999950318,
            unit: "s",
          },
          {
            name: "sum    / Jax / cpu / Forward",
            value: 0.000009294127999964983,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / Forward",
            value: 0.000009276247000343571,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / Forward",
            value: 0.000009341437999864863,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / Forward",
            value: 0.000009306017000199065,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / Forward",
            value: 0.00000934007100022427,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / Forward",
            value: 0.00000931833300001017,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / PreRev",
            value: 0.000008659404000354698,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / PostRev",
            value: 0.000008220408999477513,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / BothRev",
            value: 0.000008212631999413134,
            unit: "s",
          },
          {
            name: "sum    / Jax / cpu / BothRev",
            value: 0.000008182296000086354,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / PreRev",
            value: 0.000008208266000110598,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / PostRev",
            value: 0.00000818000899926119,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / BothRev",
            value: 0.000008230755999647953,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / PreRev",
            value: 0.000008188210999833245,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / PostRev",
            value: 0.000008263427000201773,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / BothRev",
            value: 0.000008266593000371359,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / PreRev",
            value: 0.000008359031000509276,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / PostRev",
            value: 0.000008188824000171735,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / BothRev",
            value: 0.000008236940999267972,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / PreRev",
            value: 0.000008221510000112176,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / PostRev",
            value: 0.000008172382000338985,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / BothRev",
            value: 0.000008267877999969641,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / PreRev",
            value: 0.000008265399999800138,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / PostRev",
            value: 0.000008607482000115851,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / BothRev",
            value: 0.000008274566000181948,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / Primal",
            value: 0.000011455182999270618,
            unit: "s",
          },
          {
            name: "sum    / Jax / cpu / Primal",
            value: 0.000011653572999421155,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / Primal",
            value: 0.000012154953999925057,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / Primal",
            value: 0.00001225163199887902,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / Primal",
            value: 0.000012167004999355412,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / Primal",
            value: 0.000012171450000096228,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / Primal",
            value: 0.000011581651000597049,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / Forward",
            value: 0.000017859047000456485,
            unit: "s",
          },
          {
            name: "sum    / Jax / cpu / Forward",
            value: 0.000017217564000020502,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / Forward",
            value: 0.00001691264399960346,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / Forward",
            value: 0.0000178695129998232,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / Forward",
            value: 0.000017828440000812406,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / Forward",
            value: 0.00001778033200025675,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / Forward",
            value: 0.000017817762000049697,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / PreRev",
            value: 0.000016431632000603712,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / PostRev",
            value: 0.000015833978999580722,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / BothRev",
            value: 0.000016584475999479764,
            unit: "s",
          },
          {
            name: "sum    / Jax / cpu / BothRev",
            value: 0.00001579138899978716,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / PreRev",
            value: 0.000016537102001166205,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / PostRev",
            value: 0.0000157854059998499,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / BothRev",
            value: 0.000016626045000521118,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / PreRev",
            value: 0.00001665344799948798,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / PostRev",
            value: 0.000016574137000134216,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / BothRev",
            value: 0.000015678284000387065,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / PreRev",
            value: 0.000016475333000926185,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / PostRev",
            value: 0.000015770326001074863,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / BothRev",
            value: 0.000016553184999793302,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / PreRev",
            value: 0.00001666916000067431,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / PostRev",
            value: 0.00001580087000002095,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / BothRev",
            value: 0.000016897969999263296,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / PreRev",
            value: 0.00001582143500127131,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / PostRev",
            value: 0.000016625733998807847,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / BothRev",
            value: 0.000015799838000020828,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / tpu / Primal",
            value: 0.0001305067479988,
            unit: "s",
          },
          {
            name: "sum    / Jax / tpu / Primal",
            value: 0.0001300685690002,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / tpu / Primal",
            value: 0.0001272150190015,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / tpu / Primal",
            value: 0.0001265140779978,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / tpu / Primal",
            value: 0.0001271325779998,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / tpu / Primal",
            value: 0.0001317246580001,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / tpu / Primal",
            value: 0.0001337057780001,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / tpu / Forward",
            value: 0.0002004066970002,
            unit: "s",
          },
          {
            name: "sum    / Jax / tpu / Forward",
            value: 0.0002031719680016,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / tpu / Forward",
            value: 0.0002018899680006,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / tpu / Forward",
            value: 0.0002050878369991,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / tpu / Forward",
            value: 0.0002135230370004,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / tpu / Forward",
            value: 0.0002086372570011,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / tpu / Forward",
            value: 0.0002079271170005,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / tpu / PreRev",
            value: 0.0002086840669981,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / tpu / PostRev",
            value: 0.0001999027269994,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / tpu / BothRev",
            value: 0.0001995572679988,
            unit: "s",
          },
          {
            name: "sum    / Jax / tpu / BothRev",
            value: 0.0002002609170012,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / tpu / PreRev",
            value: 0.0002006087670015,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / tpu / PostRev",
            value: 0.0002059603470006,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / tpu / BothRev",
            value: 0.0002148388669993,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / tpu / PreRev",
            value: 0.0002061923470027,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / tpu / PostRev",
            value: 0.0002089810569996,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / tpu / BothRev",
            value: 0.0002017854969999,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / tpu / PreRev",
            value: 0.0001951413170027,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / tpu / PostRev",
            value: 0.0002343835969986,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / tpu / BothRev",
            value: 0.0002323507069995,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / tpu / PreRev",
            value: 0.0002292386570006,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / tpu / PostRev",
            value: 0.0001759398469985,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / tpu / BothRev",
            value: 0.0001741600880013,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / tpu / PreRev",
            value: 0.0001745631370031,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / tpu / PostRev",
            value: 0.0001831153679995,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / tpu / BothRev",
            value: 0.0001764842470001,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / Primal",
            value: 0.000007964284000081534,
            unit: "s",
          },
          {
            name: "sum    / Jax / cpu / Primal",
            value: 0.00000825290699958714,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / Primal",
            value: 0.0000079225569998016,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / Primal",
            value: 0.000008393312000407605,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / Primal",
            value: 0.000007878520999838655,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / Primal",
            value: 0.000008312848000059603,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / Primal",
            value: 0.000007915981999758515,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / Forward",
            value: 0.000012304219999350608,
            unit: "s",
          },
          {
            name: "sum    / Jax / cpu / Forward",
            value: 0.000012263417999747616,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / Forward",
            value: 0.000012289891999898827,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / Forward",
            value: 0.000012272582000150578,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / Forward",
            value: 0.00001217266200001177,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / Forward",
            value: 0.000012281948999770977,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / Forward",
            value: 0.000012382643999444554,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / PreRev",
            value: 0.000011378479000086372,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / PostRev",
            value: 0.00001085223900008714,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / BothRev",
            value: 0.00001093231899994862,
            unit: "s",
          },
          {
            name: "sum    / Jax / cpu / BothRev",
            value: 0.000010813320000124804,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / PreRev",
            value: 0.000011508823999974994,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / PostRev",
            value: 0.0000108625499997288,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / BothRev",
            value: 0.000011381035999875166,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / PreRev",
            value: 0.000011301580000690592,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / PostRev",
            value: 0.00001082876500004204,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / BothRev",
            value: 0.000010676163999960407,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / PreRev",
            value: 0.000011521650999384291,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / PostRev",
            value: 0.000011509795999700144,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / BothRev",
            value: 0.000010860696000236204,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / PreRev",
            value: 0.0000115164349999759,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / PostRev",
            value: 0.000011393045999284367,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / BothRev",
            value: 0.000011400403000152435,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / PreRev",
            value: 0.000011600547999478295,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / PostRev",
            value: 0.000011351813999681326,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / BothRev",
            value: 0.000011420018000535493,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / Primal",
            value: 0.000005833041000187223,
            unit: "s",
          },
          {
            name: "sum    / Jax / cpu / Primal",
            value: 0.000005705416999944646,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / Primal",
            value: 0.000005602209000016956,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / Primal",
            value: 0.000005113707999953476,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / Primal",
            value: 0.00000528654100003223,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / Primal",
            value: 0.000005548542000042289,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / Primal",
            value: 0.000005381042000180969,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / Forward",
            value: 0.000008801083999969706,
            unit: "s",
          },
          {
            name: "sum    / Jax / cpu / Forward",
            value: 0.000012926459000027536,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / Forward",
            value: 0.000009870124999906691,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / Forward",
            value: 0.000008640125000056288,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / Forward",
            value: 0.0000094096669999999,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / Forward",
            value: 0.000008389083999873037,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / Forward",
            value: 0.000009332542000038302,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / PreRev",
            value: 0.000008026208000046609,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / PostRev",
            value: 0.000008238333999997849,
            unit: "s",
          },
          {
            name: "sum    / JaXPipe / cpu / BothRev",
            value: 0.00000724591700009114,
            unit: "s",
          },
          {
            name: "sum    / Jax / cpu / BothRev",
            value: 0.000007636915999910343,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / PreRev",
            value: 0.000008459374999802095,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / PostRev",
            value: 0.00000830183300013232,
            unit: "s",
          },
          {
            name: "sum    / HLOOpt / cpu / BothRev",
            value: 0.000007526041999881273,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / PreRev",
            value: 0.00000746412499984217,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / PostRev",
            value: 0.000007857041999841386,
            unit: "s",
          },
          {
            name: "sum    / PartOpt / cpu / BothRev",
            value: 0.000009387082999865016,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / PreRev",
            value: 0.000008524583000053099,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / PostRev",
            value: 0.000008560333999866999,
            unit: "s",
          },
          {
            name: "sum    / IPartOpt / cpu / BothRev",
            value: 0.000007488208999802736,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / PreRev",
            value: 0.000008403999999927692,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / PostRev",
            value: 0.000012624250000044412,
            unit: "s",
          },
          {
            name: "sum    / DefOpt / cpu / BothRev",
            value: 0.00000789583299979313,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / PreRev",
            value: 0.000007735624999895663,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / PostRev",
            value: 0.00000767858300014268,
            unit: "s",
          },
          {
            name: "sum    / IDefOpt / cpu / BothRev",
            value: 0.000007297957999981008,
            unit: "s",
          },
          {
            name: "value_and_grad / JaXPipe / cpu / Primal",
            value: 0.000010095140999510476,
            unit: "s",
          },
          {
            name: "value_and_grad / Jax / cpu / Primal",
            value: 0.000010688440000194532,
            unit: "s",
          },
          {
            name: "value_and_grad / HLOOpt / cpu / Primal",
            value: 0.000010591046999252284,
            unit: "s",
          },
          {
            name: "value_and_grad / PartOpt / cpu / Primal",
            value: 0.00001070313400032319,
            unit: "s",
          },
          {
            name: "value_and_grad / IPartOpt / cpu / Primal",
            value: 0.000010583827999653294,
            unit: "s",
          },
          {
            name: "value_and_grad / DefOpt / cpu / Primal",
            value: 0.000010795015999974568,
            unit: "s",
          },
          {
            name: "value_and_grad / IDefOpt / cpu / Primal",
            value: 0.000010766548999527003,
            unit: "s",
          },
          {
            name: "value_and_grad / JaXPipe / cpu / Primal",
            value: 0.000016430797000793972,
            unit: "s",
          },
          {
            name: "value_and_grad / Jax / cpu / Primal",
            value: 0.000016915395999603788,
            unit: "s",
          },
          {
            name: "value_and_grad / HLOOpt / cpu / Primal",
            value: 0.000017926241000168373,
            unit: "s",
          },
          {
            name: "value_and_grad / PartOpt / cpu / Primal",
            value: 0.00001779367299968726,
            unit: "s",
          },
          {
            name: "value_and_grad / IPartOpt / cpu / Primal",
            value: 0.000016797861999293673,
            unit: "s",
          },
          {
            name: "value_and_grad / DefOpt / cpu / Primal",
            value: 0.0000180117310010246,
            unit: "s",
          },
          {
            name: "value_and_grad / IDefOpt / cpu / Primal",
            value: 0.000016785558998890338,
            unit: "s",
          },
          {
            name: "value_and_grad / JaXPipe / tpu / Primal",
            value: 0.0002092159469975,
            unit: "s",
          },
          {
            name: "value_and_grad / Jax / tpu / Primal",
            value: 0.0001981199369984,
            unit: "s",
          },
          {
            name: "value_and_grad / HLOOpt / tpu / Primal",
            value: 0.0001968578769992,
            unit: "s",
          },
          {
            name: "value_and_grad / PartOpt / tpu / Primal",
            value: 0.0002174084469988,
            unit: "s",
          },
          {
            name: "value_and_grad / IPartOpt / tpu / Primal",
            value: 0.0002212654770009,
            unit: "s",
          },
          {
            name: "value_and_grad / DefOpt / tpu / Primal",
            value: 0.0001894520679998,
            unit: "s",
          },
          {
            name: "value_and_grad / IDefOpt / tpu / Primal",
            value: 0.0002069264569981,
            unit: "s",
          },
          {
            name: "value_and_grad / JaXPipe / cpu / Primal",
            value: 0.000012999152000702452,
            unit: "s",
          },
          {
            name: "value_and_grad / Jax / cpu / Primal",
            value: 0.000012334142999861797,
            unit: "s",
          },
          {
            name: "value_and_grad / HLOOpt / cpu / Primal",
            value: 0.000012903194000500662,
            unit: "s",
          },
          {
            name: "value_and_grad / PartOpt / cpu / Primal",
            value: 0.000012894256999970822,
            unit: "s",
          },
          {
            name: "value_and_grad / IPartOpt / cpu / Primal",
            value: 0.000012750695000249832,
            unit: "s",
          },
          {
            name: "value_and_grad / DefOpt / cpu / Primal",
            value: 0.000012873286999820268,
            unit: "s",
          },
          {
            name: "value_and_grad / IDefOpt / cpu / Primal",
            value: 0.000012924851999741803,
            unit: "s",
          },
          {
            name: "value_and_grad / JaXPipe / cpu / Primal",
            value: 0.000009950958999979776,
            unit: "s",
          },
          {
            name: "value_and_grad / Jax / cpu / Primal",
            value: 0.000011048333999951863,
            unit: "s",
          },
          {
            name: "value_and_grad / HLOOpt / cpu / Primal",
            value: 0.000008763374999944062,
            unit: "s",
          },
          {
            name: "value_and_grad / PartOpt / cpu / Primal",
            value: 0.000009109667000075206,
            unit: "s",
          },
          {
            name: "value_and_grad / IPartOpt / cpu / Primal",
            value: 0.000009067208000033131,
            unit: "s",
          },
          {
            name: "value_and_grad / DefOpt / cpu / Primal",
            value: 0.000008848584000133997,
            unit: "s",
          },
          {
            name: "value_and_grad / IDefOpt / cpu / Primal",
            value: 0.000008676874999991924,
            unit: "s",
          },
          {
            name: "jaxmd40 / JaXPipe / cpu / Primal",
            value: 0.08169298540015,
            unit: "s",
          },
          {
            name: "jaxmd40 / Jax / cpu / Primal",
            value: 0.0778985247998207,
            unit: "s",
          },
          {
            name: "jaxmd40 / HLOOpt / cpu / Primal",
            value: 0.1146669918001862,
            unit: "s",
          },
          {
            name: "jaxmd40 / PartOpt / cpu / Primal",
            value: 0.0788792208000813,
            unit: "s",
          },
          {
            name: "jaxmd40 / IPartOpt / cpu / Primal",
            value: 0.0749053399998956,
            unit: "s",
          },
          {
            name: "jaxmd40 / DefOpt / cpu / Primal",
            value: 0.1128701171997818,
            unit: "s",
          },
          {
            name: "jaxmd40 / IDefOpt / cpu / Primal",
            value: 0.1120387523998942,
            unit: "s",
          },
          {
            name: "jaxmd40 / JaXPipe / cpu / Forward",
            value: 0.2274223971999163,
            unit: "s",
          },
          {
            name: "jaxmd40 / Jax / cpu / Forward",
            value: 0.1057223314001021,
            unit: "s",
          },
          {
            name: "jaxmd40 / HLOOpt / cpu / Forward",
            value: 0.2273042323999106,
            unit: "s",
          },
          {
            name: "jaxmd40 / PartOpt / cpu / Forward",
            value: 0.2249593289998301,
            unit: "s",
          },
          {
            name: "jaxmd40 / IPartOpt / cpu / Forward",
            value: 0.2251455605997762,
            unit: "s",
          },
          {
            name: "jaxmd40 / DefOpt / cpu / Forward",
            value: 0.2250285472000541,
            unit: "s",
          },
          {
            name: "jaxmd40 / IDefOpt / cpu / Forward",
            value: 0.2237545956002577,
            unit: "s",
          },
          {
            name: "jaxmd40 / JaXPipe / cpu / PostRev",
            value: 0.1573145536000083,
            unit: "s",
          },
          {
            name: "jaxmd40 / Jax / cpu / BothRev",
            value: 0.1562127009998221,
            unit: "s",
          },
          {
            name: "jaxmd40 / HLOOpt / cpu / PostRev",
            value: 0.2204018975997314,
            unit: "s",
          },
          {
            name: "jaxmd40 / PartOpt / cpu / PostRev",
            value: 0.1486732262001169,
            unit: "s",
          },
          {
            name: "jaxmd40 / IPartOpt / cpu / PostRev",
            value: 0.1504383387997222,
            unit: "s",
          },
          {
            name: "jaxmd40 / DefOpt / cpu / PostRev",
            value: 0.2189118149999558,
            unit: "s",
          },
          {
            name: "jaxmd40 / IDefOpt / cpu / PostRev",
            value: 0.2137905375999253,
            unit: "s",
          },
          {
            name: "jaxmd40 / JaXPipe / tpu / Primal",
            value: 0.0092662278002535,
            unit: "s",
          },
          {
            name: "jaxmd40 / Jax / tpu / Primal",
            value: 0.0092848218002473,
            unit: "s",
          },
          {
            name: "jaxmd40 / HLOOpt / tpu / Primal",
            value: 0.0091934459996991,
            unit: "s",
          },
          {
            name: "jaxmd40 / PartOpt / tpu / Primal",
            value: 0.0092404577997513,
            unit: "s",
          },
          {
            name: "jaxmd40 / IPartOpt / tpu / Primal",
            value: 0.0092380697999033,
            unit: "s",
          },
          {
            name: "jaxmd40 / DefOpt / tpu / Primal",
            value: 0.0090750920004211,
            unit: "s",
          },
          {
            name: "jaxmd40 / IDefOpt / tpu / Primal",
            value: 0.0090128798001387,
            unit: "s",
          },
          {
            name: "jaxmd40 / JaXPipe / tpu / Forward",
            value: 0.0178224055998725,
            unit: "s",
          },
          {
            name: "jaxmd40 / Jax / tpu / Forward",
            value: 0.0182691378002346,
            unit: "s",
          },
          {
            name: "jaxmd40 / HLOOpt / tpu / Forward",
            value: 0.0178350498004874,
            unit: "s",
          },
          {
            name: "jaxmd40 / PartOpt / tpu / Forward",
            value: 0.0177680938002595,
            unit: "s",
          },
          {
            name: "jaxmd40 / IPartOpt / tpu / Forward",
            value: 0.0177759995996893,
            unit: "s",
          },
          {
            name: "jaxmd40 / DefOpt / tpu / Forward",
            value: 0.0178052657996886,
            unit: "s",
          },
          {
            name: "jaxmd40 / IDefOpt / tpu / Forward",
            value: 0.0177842556004179,
            unit: "s",
          },
          {
            name: "jaxmd40 / JaXPipe / tpu / PostRev",
            value: 0.0197596636004163,
            unit: "s",
          },
          {
            name: "jaxmd40 / Jax / tpu / BothRev",
            value: 0.0197214958003314,
            unit: "s",
          },
          {
            name: "jaxmd40 / HLOOpt / tpu / PostRev",
            value: 0.0188723578001372,
            unit: "s",
          },
          {
            name: "jaxmd40 / PartOpt / tpu / PostRev",
            value: 0.0196967360003327,
            unit: "s",
          },
          {
            name: "jaxmd40 / IPartOpt / tpu / PostRev",
            value: 0.019705485599843,
            unit: "s",
          },
          {
            name: "jaxmd40 / DefOpt / tpu / PostRev",
            value: 0.0185196158003236,
            unit: "s",
          },
          {
            name: "jaxmd40 / IDefOpt / tpu / PostRev",
            value: 0.0181907258003775,
            unit: "s",
          },
          {
            name: "jaxmd40 / JaXPipe / cpu / Primal",
            value: 0.057089208599973,
            unit: "s",
          },
          {
            name: "jaxmd40 / Jax / cpu / Primal",
            value: 0.0528381550000631,
            unit: "s",
          },
          {
            name: "jaxmd40 / HLOOpt / cpu / Primal",
            value: 0.0803043915999296,
            unit: "s",
          },
          {
            name: "jaxmd40 / PartOpt / cpu / Primal",
            value: 0.0543609369999103,
            unit: "s",
          },
          {
            name: "jaxmd40 / IPartOpt / cpu / Primal",
            value: 0.0588535478000267,
            unit: "s",
          },
          {
            name: "jaxmd40 / DefOpt / cpu / Primal",
            value: 0.0816766711999662,
            unit: "s",
          },
          {
            name: "jaxmd40 / IDefOpt / cpu / Primal",
            value: 0.0821903307998582,
            unit: "s",
          },
          {
            name: "jaxmd40 / JaXPipe / cpu / Forward",
            value: 0.1561677100000452,
            unit: "s",
          },
          {
            name: "jaxmd40 / Jax / cpu / Forward",
            value: 0.0781024304000311,
            unit: "s",
          },
          {
            name: "jaxmd40 / HLOOpt / cpu / Forward",
            value: 0.1560636195999905,
            unit: "s",
          },
          {
            name: "jaxmd40 / PartOpt / cpu / Forward",
            value: 0.1533288402000835,
            unit: "s",
          },
          {
            name: "jaxmd40 / IPartOpt / cpu / Forward",
            value: 0.1533511806001115,
            unit: "s",
          },
          {
            name: "jaxmd40 / DefOpt / cpu / Forward",
            value: 0.1513602822000393,
            unit: "s",
          },
          {
            name: "jaxmd40 / IDefOpt / cpu / Forward",
            value: 0.1531711853998786,
            unit: "s",
          },
          {
            name: "jaxmd40 / JaXPipe / cpu / PostRev",
            value: 0.1225382961998548,
            unit: "s",
          },
          {
            name: "jaxmd40 / Jax / cpu / BothRev",
            value: 0.1224921732000439,
            unit: "s",
          },
          {
            name: "jaxmd40 / HLOOpt / cpu / PostRev",
            value: 0.1525499842000499,
            unit: "s",
          },
          {
            name: "jaxmd40 / PartOpt / cpu / PostRev",
            value: 0.1154337648000364,
            unit: "s",
          },
          {
            name: "jaxmd40 / IPartOpt / cpu / PostRev",
            value: 0.1155297968000013,
            unit: "s",
          },
          {
            name: "jaxmd40 / DefOpt / cpu / PostRev",
            value: 0.1568102322000413,
            unit: "s",
          },
          {
            name: "jaxmd40 / IDefOpt / cpu / PostRev",
            value: 0.1540065967999908,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / JaXPipe / cpu / Primal",
            value: 50.83481089899942,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / Jax / cpu / Primal",
            value: 50.85340336599984,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / HLOOpt / cpu / Primal",
            value: 50.37439326300046,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / PartOpt / cpu / Primal",
            value: 50.09926168600032,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / IPartOpt / cpu / Primal",
            value: 49.29358631899959,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / DefOpt / cpu / Primal",
            value: 25.472499459001483,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / IDefOpt / cpu / Primal",
            value: 54.84618755000156,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / JaXPipe / tpu / Primal",
            value: 0.1901371269996161,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / Jax / tpu / Primal",
            value: 0.1899908669984142,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / HLOOpt / tpu / Primal",
            value: 0.1886925969993171,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / PartOpt / tpu / Primal",
            value: 0.2028273369978706,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / IPartOpt / tpu / Primal",
            value: 0.2028477369967731,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / DefOpt / tpu / Primal",
            value: 0.1738777169994136,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / IDefOpt / tpu / Primal",
            value: 0.1849673180004174,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / JaXPipe / cpu / Primal",
            value: 53.96069653500035,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / Jax / cpu / Primal",
            value: 50.48600902100043,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / HLOOpt / cpu / Primal",
            value: 49.03250595600002,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / PartOpt / cpu / Primal",
            value: 49.297286162999626,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / IPartOpt / cpu / Primal",
            value: 48.60803564300022,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / DefOpt / cpu / Primal",
            value: 21.986474829000144,
            unit: "s",
          },
          {
            name: "neural_gcm_dynamic_forcing_deterministic_1_4_deg / IDefOpt / cpu / Primal",
            value: 55.083714699000666,
            unit: "s",
          },
        ],
      },
    ],
  },
};
