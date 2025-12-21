window.BENCHMARK_DATA = {
  "lastUpdate": 1766296372183,
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
          "id": "d84219bbd3a6ddf647eec0d1d3eafe09f00c1529",
          "message": "feat: generalize transpose slice patterns to include dynamic_slice (#1811)",
          "timestamp": "2025-12-20T00:23:24-05:00",
          "tree_id": "68641fc58f5b142ac816966a8c88638cb29308bc",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/d84219bbd3a6ddf647eec0d1d3eafe09f00c1529"
        },
        "date": 1766220785409,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "actmtch / JaXPipe / cpu / Primal",
            "value": 0.000009017840995511504,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cpu / Primal",
            "value": 0.000009171826000965664,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / Primal",
            "value": 0.00001080674999684561,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / Primal",
            "value": 0.000009661667994805611,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / Primal",
            "value": 0.00000950025000202004,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / Primal",
            "value": 0.000011441294998803642,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / Primal",
            "value": 0.00001081067699851701,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Forward",
            "value": 0.000015915044001303614,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cpu / Forward",
            "value": 0.000013813197998388204,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / Forward",
            "value": 0.000016009940998628734,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / Forward",
            "value": 0.000016733882002881728,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / Forward",
            "value": 0.000016680900997016578,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / Forward",
            "value": 0.00001652303799346555,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / Forward",
            "value": 0.000015993610002624338,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PreRev",
            "value": 0.000016704482994100543,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PostRev",
            "value": 0.000012925717994221489,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / BothRev",
            "value": 0.000016750218004744967,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cpu / BothRev",
            "value": 0.000013745324002229609,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / PreRev",
            "value": 0.000016758314006438013,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / PostRev",
            "value": 0.000016753855998103973,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / BothRev",
            "value": 0.00001668981999682728,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / PreRev",
            "value": 0.00001666839700192213,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / PostRev",
            "value": 0.00001297162099945126,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / BothRev",
            "value": 0.000015916339005343614,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / PreRev",
            "value": 0.000016729852999560534,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / PostRev",
            "value": 0.00001365967899619136,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / BothRev",
            "value": 0.000015928805005387403,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / PreRev",
            "value": 0.000016830627995659598,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / PostRev",
            "value": 0.00001592430799792055,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / BothRev",
            "value": 0.000016603260999545456,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / PreRev",
            "value": 0.000016619590998743662,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / PostRev",
            "value": 0.000016711364005459472,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / BothRev",
            "value": 0.000016563281002163423,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Primal",
            "value": 0.000006445933000577498,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cpu / Primal",
            "value": 0.000006378570000379113,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / Primal",
            "value": 0.000007566774000224541,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / Primal",
            "value": 0.000006777588000659307,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / Primal",
            "value": 0.00000634492799963482,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / Primal",
            "value": 0.000007684158999836654,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / Primal",
            "value": 0.000007582242999887967,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Forward",
            "value": 0.000011198514999705369,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cpu / Forward",
            "value": 0.000009480608000558277,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / Forward",
            "value": 0.00001157291700019414,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / Forward",
            "value": 0.000011703003000548052,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / Forward",
            "value": 0.00001106743699983781,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / Forward",
            "value": 0.00001165221599967481,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / Forward",
            "value": 0.000011181720999957178,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PreRev",
            "value": 0.000010973506000482305,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PostRev",
            "value": 0.000009145756999714649,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / BothRev",
            "value": 0.000011079967000114263,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cpu / BothRev",
            "value": 0.000009739576000356464,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / PreRev",
            "value": 0.00001150724800027092,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / PostRev",
            "value": 0.00001154702699932386,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / BothRev",
            "value": 0.00001154933300040284,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / PreRev",
            "value": 0.000010920208000243293,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / PostRev",
            "value": 0.000009215303000019048,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / BothRev",
            "value": 0.000010946114000034867,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / PreRev",
            "value": 0.000011681741999382212,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / PostRev",
            "value": 0.0000096895830001813,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / BothRev",
            "value": 0.000011631282000053032,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / PreRev",
            "value": 0.0000110407940001096,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / PostRev",
            "value": 0.000011058014999434818,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / BothRev",
            "value": 0.000011661608000395063,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / PreRev",
            "value": 0.000011595215000852476,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / PostRev",
            "value": 0.00001148391700007778,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / BothRev",
            "value": 0.000011599974000091606,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Primal",
            "value": 0.000003894166000463884,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cpu / Primal",
            "value": 0.000003883041999870329,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / Primal",
            "value": 0.0000044968330003030135,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / Primal",
            "value": 0.000004132290998313693,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / Primal",
            "value": 0.000003871208000418847,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / Primal",
            "value": 0.000004497624999203254,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / Primal",
            "value": 0.0000045794589987053765,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Forward",
            "value": 0.0000069862500004092,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cpu / Forward",
            "value": 0.000005808916999740177,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / Forward",
            "value": 0.000006821208000474144,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / Forward",
            "value": 0.000006904959000166855,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / Forward",
            "value": 0.000007093000000168104,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / Forward",
            "value": 0.000006934249999176245,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / Forward",
            "value": 0.000007071750000250177,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PreRev",
            "value": 0.000006889624999530497,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PostRev",
            "value": 0.000006091500001275563,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / BothRev",
            "value": 0.0000067673749999812574,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cpu / BothRev",
            "value": 0.00000590554200061888,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / PreRev",
            "value": 0.000007150249999540392,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / PostRev",
            "value": 0.000006907875000251806,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / BothRev",
            "value": 0.000007019249998847954,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / PreRev",
            "value": 0.000006964416999835521,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / PostRev",
            "value": 0.00000581400000010035,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / BothRev",
            "value": 0.0000068699160001415296,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / PreRev",
            "value": 0.000006786208999983501,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / PostRev",
            "value": 0.000005929290999119985,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / BothRev",
            "value": 0.000006786083000406507,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / PreRev",
            "value": 0.000006812707999415579,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / PostRev",
            "value": 0.000006852917000287562,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / BothRev",
            "value": 0.000006733624999469612,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / PreRev",
            "value": 0.000006725791999997455,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / PostRev",
            "value": 0.000006707000000460539,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / BothRev",
            "value": 0.000006664749998890329,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Primal",
            "value": 0.00000895060599577846,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cpu / Primal",
            "value": 0.000009043592996022198,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / Primal",
            "value": 0.000009153863000392448,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / Primal",
            "value": 0.000009343528996396345,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / Primal",
            "value": 0.000009684426004241686,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / Primal",
            "value": 0.000009206139999150764,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / Primal",
            "value": 0.000009229385999788064,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Forward",
            "value": 0.000014778932003537194,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cpu / Forward",
            "value": 0.000014909093995811418,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / Forward",
            "value": 0.000014818051997281145,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / Forward",
            "value": 0.00001475432400184218,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / Forward",
            "value": 0.000014030354999704286,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / Forward",
            "value": 0.00001471393500105478,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / Forward",
            "value": 0.000014820639997196847,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PreRev",
            "value": 0.000015071290996274911,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PostRev",
            "value": 0.000015849930998228956,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / BothRev",
            "value": 0.0000150263109972002,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cpu / BothRev",
            "value": 0.00001585935800540028,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / PreRev",
            "value": 0.00001588163399719633,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / PostRev",
            "value": 0.000015884681000898128,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / BothRev",
            "value": 0.000015976147995388603,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / PreRev",
            "value": 0.00001515364400256658,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / PostRev",
            "value": 0.00001591495100001339,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / BothRev",
            "value": 0.000015042749000713229,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / PreRev",
            "value": 0.000014968859999498818,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / PostRev",
            "value": 0.000016005827994376885,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / BothRev",
            "value": 0.00001593645499815466,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / PreRev",
            "value": 0.000015937515003315638,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / PostRev",
            "value": 0.000015002601001469884,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / BothRev",
            "value": 0.000015857888000027742,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / PreRev",
            "value": 0.000015958308998961003,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / PostRev",
            "value": 0.00001594852300331695,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / BothRev",
            "value": 0.00001495627900294494,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Primal",
            "value": 0.000006537141000080738,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cpu / Primal",
            "value": 0.000006594381000468275,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / Primal",
            "value": 0.000006969563999518868,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / Primal",
            "value": 0.000006992369999352377,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / Primal",
            "value": 0.0000069280110001272985,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / Primal",
            "value": 0.000006968119000703155,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / Primal",
            "value": 0.000006919100999766669,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Forward",
            "value": 0.00000995908699951542,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cpu / Forward",
            "value": 0.000010308323000572271,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / Forward",
            "value": 0.000009908191000249644,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / Forward",
            "value": 0.00000991970199993375,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / Forward",
            "value": 0.000010456822999913127,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / Forward",
            "value": 0.000010298797999894304,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / Forward",
            "value": 0.000010423732999697675,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PreRev",
            "value": 0.000011425101000895666,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PostRev",
            "value": 0.000011209389000214287,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / BothRev",
            "value": 0.000011265794999417266,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cpu / BothRev",
            "value": 0.000011310442999274528,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / PreRev",
            "value": 0.000011420717999499176,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / PostRev",
            "value": 0.00001121621999936906,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / BothRev",
            "value": 0.000011264468999797827,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / PreRev",
            "value": 0.000011242384999604835,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / PostRev",
            "value": 0.000011222180999538976,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / BothRev",
            "value": 0.00001059756899940112,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / PreRev",
            "value": 0.000011161859000822003,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / PostRev",
            "value": 0.000011084847999882183,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / BothRev",
            "value": 0.00001137350399949355,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / PreRev",
            "value": 0.000011288531000900549,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / PostRev",
            "value": 0.000011224144000152591,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / BothRev",
            "value": 0.000011327559999699588,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / PreRev",
            "value": 0.000011362153999471048,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / PostRev",
            "value": 0.00001125058999969042,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / BothRev",
            "value": 0.000011280469999292109,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Primal",
            "value": 0.0000037158750001253793,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cpu / Primal",
            "value": 0.0000037177919984969777,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / Primal",
            "value": 0.0000037096660016686656,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / Primal",
            "value": 0.000003526249998685671,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / Primal",
            "value": 0.0000036659999987023194,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / Primal",
            "value": 0.0000036919579997629626,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / Primal",
            "value": 0.000003534624998792424,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Forward",
            "value": 0.000005913750001127482,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cpu / Forward",
            "value": 0.0000058259999987058105,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / Forward",
            "value": 0.0000055399579996446845,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / Forward",
            "value": 0.000005829207999340724,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / Forward",
            "value": 0.000005549667001105263,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / Forward",
            "value": 0.0000059947080007987095,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / Forward",
            "value": 0.000005713541999284644,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PreRev",
            "value": 0.000006765000000086729,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PostRev",
            "value": 0.000006666207998932805,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / BothRev",
            "value": 0.000006945708000785089,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cpu / BothRev",
            "value": 0.000006635916999584879,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / PreRev",
            "value": 0.000006908458000907558,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / PostRev",
            "value": 0.000006665999999313499,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / BothRev",
            "value": 0.000006833875000666012,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / PreRev",
            "value": 0.000007095666998793604,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / PostRev",
            "value": 0.000006672457999229664,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / BothRev",
            "value": 0.000006739000000379747,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / PreRev",
            "value": 0.000006887292000101297,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / PostRev",
            "value": 0.000006719415998304612,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / BothRev",
            "value": 0.000006712874999720952,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / PreRev",
            "value": 0.000006989708001128747,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / PostRev",
            "value": 0.000007182499999544234,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / BothRev",
            "value": 0.000006843791999926907,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / PreRev",
            "value": 0.000006793708000259357,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / PostRev",
            "value": 0.0000068314580003061565,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / BothRev",
            "value": 0.000006887249999635969,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Primal",
            "value": 0.000009640481999667829,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cpu / Primal",
            "value": 0.000009513812998193315,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / Primal",
            "value": 0.00001011937199655222,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / Primal",
            "value": 0.000010018435001256876,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / Primal",
            "value": 0.000009371227999508847,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / Primal",
            "value": 0.000010066558999824338,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / Primal",
            "value": 0.00000957737799762981,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Forward",
            "value": 0.000014404638997802975,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cpu / Forward",
            "value": 0.000015054310999403244,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / Forward",
            "value": 0.000015112903995031957,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / Forward",
            "value": 0.000015009484995971432,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / Forward",
            "value": 0.00001515321400074754,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / Forward",
            "value": 0.00001504989100067178,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / Forward",
            "value": 0.000014531634995364584,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PreRev",
            "value": 0.00001783690100273816,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PostRev",
            "value": 0.0000185224559972994,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / BothRev",
            "value": 0.00001847327200084692,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cpu / BothRev",
            "value": 0.000018491331000404897,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / PreRev",
            "value": 0.000018482851999579,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / PostRev",
            "value": 0.0000186589470031322,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / BothRev",
            "value": 0.000018684365000808613,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / PreRev",
            "value": 0.00001866976499877637,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / PostRev",
            "value": 0.00001846846199623542,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / BothRev",
            "value": 0.000017776394000975414,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / PreRev",
            "value": 0.00001861408300464973,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / PostRev",
            "value": 0.00001855601300485432,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / BothRev",
            "value": 0.000018565916005172768,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / PreRev",
            "value": 0.000018497746001230552,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / PostRev",
            "value": 0.00001869731999613577,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / BothRev",
            "value": 0.000018484884996723848,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / PreRev",
            "value": 0.000017706589002045802,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / PostRev",
            "value": 0.000018582222997793,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / BothRev",
            "value": 0.000018538709002314132,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Primal",
            "value": 0.000006859535000330652,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cpu / Primal",
            "value": 0.000006833841000116081,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / Primal",
            "value": 0.000006836770000518299,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / Primal",
            "value": 0.000007184873000369407,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / Primal",
            "value": 0.000006801341000027606,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / Primal",
            "value": 0.000006845656999757921,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / Primal",
            "value": 0.00000684592600009637,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Forward",
            "value": 0.000010116944999936094,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cpu / Forward",
            "value": 0.000010735806999946365,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / Forward",
            "value": 0.000010725387999627856,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / Forward",
            "value": 0.000010624579999785055,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / Forward",
            "value": 0.000010770351000246592,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / Forward",
            "value": 0.00001063338299991301,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / Forward",
            "value": 0.000010269165999488906,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PreRev",
            "value": 0.00001293737899959524,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PostRev",
            "value": 0.000013172045999453983,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / BothRev",
            "value": 0.00001326688799963449,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cpu / BothRev",
            "value": 0.000013339772999643174,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / PreRev",
            "value": 0.000012705195999842544,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / PostRev",
            "value": 0.00001321730600011506,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / BothRev",
            "value": 0.000013183691999984147,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / PreRev",
            "value": 0.00001320418600062112,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / PostRev",
            "value": 0.000013357025999539474,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / BothRev",
            "value": 0.000012745160999656946,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / PreRev",
            "value": 0.000012601782000274396,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / PostRev",
            "value": 0.000013182804000280156,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / BothRev",
            "value": 0.0000128109110000878,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / PreRev",
            "value": 0.00001328054600071482,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / PostRev",
            "value": 0.000012688458000411628,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / BothRev",
            "value": 0.000013100412999847322,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / PreRev",
            "value": 0.000013126034999913826,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / PostRev",
            "value": 0.000013135657999555406,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / BothRev",
            "value": 0.000013276285000756616,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Primal",
            "value": 0.0000040261249996547125,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cpu / Primal",
            "value": 0.000004099791000044206,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / Primal",
            "value": 0.000003969125000367057,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / Primal",
            "value": 0.000004103750001377193,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / Primal",
            "value": 0.000003943249999792897,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / Primal",
            "value": 0.000003987959000369301,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / Primal",
            "value": 0.000004024707999633392,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Forward",
            "value": 0.000006229625001651584,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cpu / Forward",
            "value": 0.000006234874999790918,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / Forward",
            "value": 0.000006051333000868908,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / Forward",
            "value": 0.000006138249998912215,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / Forward",
            "value": 0.000006026834000294912,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / Forward",
            "value": 0.0000059275000003253805,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / Forward",
            "value": 0.000006223124999451102,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PreRev",
            "value": 0.000007924207999167265,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PostRev",
            "value": 0.000008182292000128655,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / BothRev",
            "value": 0.000008757084000535542,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cpu / BothRev",
            "value": 0.000008131917000355316,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / PreRev",
            "value": 0.00000791791700066824,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / PostRev",
            "value": 0.000008108083000479383,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / BothRev",
            "value": 0.000008012333999431576,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / PreRev",
            "value": 0.000007835249998606742,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / PostRev",
            "value": 0.000008400374999837368,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / BothRev",
            "value": 0.00000830754100024933,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / PreRev",
            "value": 0.000008789834000708651,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / PostRev",
            "value": 0.000008256375000200933,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / BothRev",
            "value": 0.000008707083001354477,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / PreRev",
            "value": 0.000009471000001212815,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / PostRev",
            "value": 0.000009709582998766564,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / BothRev",
            "value": 0.000008872042000803049,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / PreRev",
            "value": 0.000009712957998999629,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / PostRev",
            "value": 0.000008475750000798143,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / BothRev",
            "value": 0.000008591207999415929,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Primal",
            "value": 0.000009609747998183594,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cpu / Primal",
            "value": 0.000009365705998789052,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / Primal",
            "value": 0.000009425263000593988,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / Primal",
            "value": 0.000009239071994670666,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / Primal",
            "value": 0.000009557633005897514,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / Primal",
            "value": 0.000009335782000562176,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / Primal",
            "value": 0.000009630742999434006,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Forward",
            "value": 0.00001457981899875449,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cpu / Forward",
            "value": 0.000015010008995886893,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / Forward",
            "value": 0.00001442481000412954,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / Forward",
            "value": 0.000015100035998329986,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / Forward",
            "value": 0.00001449242900707759,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / Forward",
            "value": 0.000015101836004760115,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / Forward",
            "value": 0.000015233239995723123,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PreRev",
            "value": 0.000015356203002738768,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PostRev",
            "value": 0.000017827370000304653,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / BothRev",
            "value": 0.00001506445500126574,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cpu / BothRev",
            "value": 0.00001756304199807346,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / PreRev",
            "value": 0.000015433471999131145,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / PostRev",
            "value": 0.000015477039996767418,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / BothRev",
            "value": 0.00001502266700117616,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / PreRev",
            "value": 0.000015474041996640152,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / PostRev",
            "value": 0.00001762048000091454,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / BothRev",
            "value": 0.000015067493004607968,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / PreRev",
            "value": 0.00001555243400071049,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / PostRev",
            "value": 0.00001774973799911095,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / BothRev",
            "value": 0.00001519029599876376,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / PreRev",
            "value": 0.000014554058005160186,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / PostRev",
            "value": 0.00001505063600052381,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / BothRev",
            "value": 0.000015338571996835525,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / PreRev",
            "value": 0.0000150277049979195,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / PostRev",
            "value": 0.00001528586499625817,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / BothRev",
            "value": 0.000015388973995868583,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Primal",
            "value": 0.000006618448000153876,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cpu / Primal",
            "value": 0.0000062406790002569325,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / Primal",
            "value": 0.00000620331400023133,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / Primal",
            "value": 0.000006252889000279538,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / Primal",
            "value": 0.000006100112000240188,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / Primal",
            "value": 0.00000671751200025028,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / Primal",
            "value": 0.000006270007000239275,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Forward",
            "value": 0.000011095222999756517,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cpu / Forward",
            "value": 0.00001077517199973954,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / Forward",
            "value": 0.000010522521999519083,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / Forward",
            "value": 0.000011063504000048853,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / Forward",
            "value": 0.00001093887699971674,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / Forward",
            "value": 0.000010935997999695246,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / Forward",
            "value": 0.000010944562000076985,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PreRev",
            "value": 0.000011494192000100155,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PostRev",
            "value": 0.000012330558000030578,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / BothRev",
            "value": 0.000011671711000417417,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cpu / BothRev",
            "value": 0.000013136239999766984,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / PreRev",
            "value": 0.00001122914799998398,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / PostRev",
            "value": 0.00001124326400076825,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / BothRev",
            "value": 0.000011192436999408528,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / PreRev",
            "value": 0.000011069286999372708,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / PostRev",
            "value": 0.000011989039000582123,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / BothRev",
            "value": 0.000010663633999683952,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / PreRev",
            "value": 0.000011111276000519866,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / PostRev",
            "value": 0.000012710824999885515,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / BothRev",
            "value": 0.000011419784999816329,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / PreRev",
            "value": 0.00001118337000025349,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / PostRev",
            "value": 0.000011042037999686728,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / BothRev",
            "value": 0.000011414741000407958,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / PreRev",
            "value": 0.000011275462000412515,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / PostRev",
            "value": 0.000011129109000648896,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / BothRev",
            "value": 0.00001148260699937964,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Primal",
            "value": 0.0000036015419991599627,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cpu / Primal",
            "value": 0.0000036634589996538126,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / Primal",
            "value": 0.0000035730829986277968,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / Primal",
            "value": 0.0000035640830010379432,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / Primal",
            "value": 0.000003645540999059449,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / Primal",
            "value": 0.000003475583000181359,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / Primal",
            "value": 0.0000035653329996421238,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Forward",
            "value": 0.000006170958000438987,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cpu / Forward",
            "value": 0.000006187166000017896,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / Forward",
            "value": 0.000006406750000678585,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / Forward",
            "value": 0.0000060780000003433085,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / Forward",
            "value": 0.000006252749999475782,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / Forward",
            "value": 0.000006231249999473221,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / Forward",
            "value": 0.0000063709579990245405,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PreRev",
            "value": 0.00000626445799935027,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PostRev",
            "value": 0.0000067093750003550665,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / BothRev",
            "value": 0.00000626025000019581,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cpu / BothRev",
            "value": 0.000007082375001118635,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / PreRev",
            "value": 0.000006332709001071635,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / PostRev",
            "value": 0.00000608329199894797,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / BothRev",
            "value": 0.00000625074999879871,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / PreRev",
            "value": 0.000006428624999898602,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / PostRev",
            "value": 0.000006695374999253545,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / BothRev",
            "value": 0.000006019124999511405,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / PreRev",
            "value": 0.00000614516600035131,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / PostRev",
            "value": 0.000006926332998773432,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / BothRev",
            "value": 0.000005984625000564847,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / PreRev",
            "value": 0.000006086542000048211,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / PostRev",
            "value": 0.00000623091600027692,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / BothRev",
            "value": 0.00000688491700020677,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / PreRev",
            "value": 0.00000615954200111446,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / PostRev",
            "value": 0.000006385208998835879,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / BothRev",
            "value": 0.000013712084000871985,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Primal",
            "value": 0.00000902528699953109,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cpu / Primal",
            "value": 0.00000906422499974724,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / Primal",
            "value": 0.00000967304199730279,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / Primal",
            "value": 0.00000932146499690134,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / Primal",
            "value": 0.000009664162003900855,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / Primal",
            "value": 0.000009676729001512284,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / Primal",
            "value": 0.000008959227001469117,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Forward",
            "value": 0.00001481569300085539,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cpu / Forward",
            "value": 0.000013862353000149596,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / Forward",
            "value": 0.0000138814740057569,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / Forward",
            "value": 0.0000146126940016984,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / Forward",
            "value": 0.000014698990002216306,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / Forward",
            "value": 0.000014770292000321206,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / Forward",
            "value": 0.000014612769002269488,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PreRev",
            "value": 0.00001521420200151624,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PostRev",
            "value": 0.00001609163799730595,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / BothRev",
            "value": 0.000016135282996401656,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cpu / BothRev",
            "value": 0.000016035314001783262,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / PreRev",
            "value": 0.000016032560997700783,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / PostRev",
            "value": 0.000016410667005402503,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / BothRev",
            "value": 0.000015902993000054265,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / PreRev",
            "value": 0.000016019695001887157,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / PostRev",
            "value": 0.00001598365799873136,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / BothRev",
            "value": 0.000015179941998212598,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / PreRev",
            "value": 0.000016045422002207487,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / PostRev",
            "value": 0.0000162702219968196,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / BothRev",
            "value": 0.000016076143001555465,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / PreRev",
            "value": 0.000016109956995933316,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / PostRev",
            "value": 0.000015232452002237552,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / BothRev",
            "value": 0.000015982255994458684,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / PreRev",
            "value": 0.000016091351004433817,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / PostRev",
            "value": 0.00001605448399641318,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / BothRev",
            "value": 0.000015123826000490226,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Primal",
            "value": 0.000006749515999217692,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cpu / Primal",
            "value": 0.00000665716200001043,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / Primal",
            "value": 0.000006620881999879202,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / Primal",
            "value": 0.000006705500999487413,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / Primal",
            "value": 0.000006759391999366926,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / Primal",
            "value": 0.0000065464369999972405,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / Primal",
            "value": 0.000006637264999881154,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Forward",
            "value": 0.000010034173000349256,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cpu / Forward",
            "value": 0.000010628267000356571,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / Forward",
            "value": 0.000010729860000537884,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / Forward",
            "value": 0.00001057174500056135,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / Forward",
            "value": 0.000010527209999963815,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / Forward",
            "value": 0.000010043516000223464,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / Forward",
            "value": 0.000010682700999495864,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PreRev",
            "value": 0.00001083953899978951,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PostRev",
            "value": 0.00001182294600039313,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / BothRev",
            "value": 0.000011581404999560618,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cpu / BothRev",
            "value": 0.000011599430999922331,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / PreRev",
            "value": 0.000011277245999735895,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / PostRev",
            "value": 0.000011270017999777337,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / BothRev",
            "value": 0.000011222614000871544,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / PreRev",
            "value": 0.000011262659999374592,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / PostRev",
            "value": 0.000011378612000044086,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / BothRev",
            "value": 0.000010703658000238649,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / PreRev",
            "value": 0.000011442600999544083,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / PostRev",
            "value": 0.00001141611599996395,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / BothRev",
            "value": 0.00001138217599964264,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / PreRev",
            "value": 0.00001074063699979888,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / PostRev",
            "value": 0.000011408207000386028,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / BothRev",
            "value": 0.000011245742000028258,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / PreRev",
            "value": 0.000011382663000404135,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / PostRev",
            "value": 0.00001155269200080511,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / BothRev",
            "value": 0.000011492953000015404,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Primal",
            "value": 0.000003836750000118627,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cpu / Primal",
            "value": 0.000004456541000763537,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / Primal",
            "value": 0.000003927832998670055,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / Primal",
            "value": 0.000004260666999471141,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / Primal",
            "value": 0.0000041046250007639175,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / Primal",
            "value": 0.000003982292000728194,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / Primal",
            "value": 0.0000039092919996619455,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Forward",
            "value": 0.000006144999999378342,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cpu / Forward",
            "value": 0.000006104250000134926,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / Forward",
            "value": 0.000006083332998969126,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / Forward",
            "value": 0.00000647745799869881,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / Forward",
            "value": 0.0000060845419993711405,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / Forward",
            "value": 0.0000063178339987644,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / Forward",
            "value": 0.000006135999999969499,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PreRev",
            "value": 0.000007095041999491514,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PostRev",
            "value": 0.000007084333001330378,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / BothRev",
            "value": 0.000007114625001122477,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cpu / BothRev",
            "value": 0.000007415457999741193,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / PreRev",
            "value": 0.000006918667000718415,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / PostRev",
            "value": 0.000006951958001081948,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / BothRev",
            "value": 0.00000709445800021058,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / PreRev",
            "value": 0.000007143374999941443,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / PostRev",
            "value": 0.000006991124999331078,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / BothRev",
            "value": 0.000007099124999513151,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / PreRev",
            "value": 0.000007087791000230936,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / PostRev",
            "value": 0.000006958542000575108,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / BothRev",
            "value": 0.000007040499998765881,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / PreRev",
            "value": 0.0000072540000001026785,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / PostRev",
            "value": 0.000007092541998645174,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / BothRev",
            "value": 0.000006939875000171015,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / PreRev",
            "value": 0.000007000084000537754,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / PostRev",
            "value": 0.000007022333000350045,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / BothRev",
            "value": 0.000007017040999926394,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Primal",
            "value": 0.000008681462000822648,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / cpu / Primal",
            "value": 0.000008885424998879899,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / Primal",
            "value": 0.00000992525400215527,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / Primal",
            "value": 0.000008774768000876066,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / Primal",
            "value": 0.00000925907100463519,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / Primal",
            "value": 0.000010408883994386996,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / Primal",
            "value": 0.00001037100699613802,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Forward",
            "value": 0.00001540116999967722,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / cpu / Forward",
            "value": 0.00001254022799548693,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / Forward",
            "value": 0.00001482652299455367,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / Forward",
            "value": 0.0000147478719954961,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / Forward",
            "value": 0.000015487730997847393,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / Forward",
            "value": 0.00001550869299535407,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / Forward",
            "value": 0.000014854086002742404,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Primal",
            "value": 0.000006251998000152525,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / cpu / Primal",
            "value": 0.000006259595000301488,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / Primal",
            "value": 0.000006849612000223715,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / Primal",
            "value": 0.000006628802999330219,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / Primal",
            "value": 0.0000066055520001100375,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / Primal",
            "value": 0.000007333342000492848,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / Primal",
            "value": 0.000007289137000043411,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Forward",
            "value": 0.000010930798000117648,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / cpu / Forward",
            "value": 0.000009486533000199415,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / Forward",
            "value": 0.000010431407000396576,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / Forward",
            "value": 0.0000103049720000854,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / Forward",
            "value": 0.000010900560000663971,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / Forward",
            "value": 0.000010838393999620166,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / Forward",
            "value": 0.000010858195999389865,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Primal",
            "value": 0.0000036078339999221497,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / cpu / Primal",
            "value": 0.0000036066250013391256,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / Primal",
            "value": 0.00000411191700004565,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / Primal",
            "value": 0.0000035964580001746077,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / Primal",
            "value": 0.000003601166999942507,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / Primal",
            "value": 0.000004365791000964236,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / Primal",
            "value": 0.000004180792000624933,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Forward",
            "value": 0.000006886459001179901,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / cpu / Forward",
            "value": 0.000005603457999313832,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / Forward",
            "value": 0.000007333916999414214,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / Forward",
            "value": 0.000006783915998312295,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / Forward",
            "value": 0.00000679149999996298,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / Forward",
            "value": 0.000006782833001125254,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / Forward",
            "value": 0.000006935792000149376,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Primal",
            "value": 0.000009735700004966927,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cpu / Primal",
            "value": 0.00001009458199405344,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / Primal",
            "value": 0.000011473860999103636,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / Primal",
            "value": 0.000009495074998994824,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / Primal",
            "value": 0.000010261447001539636,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / Primal",
            "value": 0.000011319283999910113,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / Primal",
            "value": 0.00001141602999996394,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Forward",
            "value": 0.000016056246997322887,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cpu / Forward",
            "value": 0.000013716458997805605,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / Forward",
            "value": 0.00001590046100318432,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / Forward",
            "value": 0.00001665578100073617,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / Forward",
            "value": 0.000016765467000368517,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / Forward",
            "value": 0.000016834448004374282,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / Forward",
            "value": 0.000016565826001169625,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PreRev",
            "value": 0.00001588651500060223,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PostRev",
            "value": 0.000013653648995386904,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / BothRev",
            "value": 0.0000168114779953612,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cpu / BothRev",
            "value": 0.000014664170004834888,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / PreRev",
            "value": 0.000015978101000655442,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / PostRev",
            "value": 0.000016624313000647816,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / BothRev",
            "value": 0.00001679221300582867,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / PreRev",
            "value": 0.000015930773995933123,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / PostRev",
            "value": 0.000013832270000420976,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / BothRev",
            "value": 0.000015910779002297204,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / PreRev",
            "value": 0.000016773686002125033,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / PostRev",
            "value": 0.000014750520000234244,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / BothRev",
            "value": 0.000016119055995659436,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / PreRev",
            "value": 0.00001685931399697438,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / PostRev",
            "value": 0.00001610180100396974,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / BothRev",
            "value": 0.0000159125279969885,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / PreRev",
            "value": 0.0000158978899999056,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / PostRev",
            "value": 0.00001688650700089056,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / BothRev",
            "value": 0.000015983723002136685,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Primal",
            "value": 0.000006948471999749017,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cpu / Primal",
            "value": 0.000007136303999686788,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / Primal",
            "value": 0.000007978911999998672,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / Primal",
            "value": 0.000006889860999763186,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / Primal",
            "value": 0.000006963477000681451,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / Primal",
            "value": 0.000008071928999925148,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / Primal",
            "value": 0.000008033467999666755,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Forward",
            "value": 0.000011447493000559917,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cpu / Forward",
            "value": 0.00001031794499976968,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / Forward",
            "value": 0.000011288237000371737,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / Forward",
            "value": 0.000011186351000105789,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / Forward",
            "value": 0.00001149794400043902,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / Forward",
            "value": 0.00001095958000041719,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / Forward",
            "value": 0.000011108131000582944,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PreRev",
            "value": 0.000011564391000320029,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PostRev",
            "value": 0.000010272370999700797,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / BothRev",
            "value": 0.000011040837000109603,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cpu / BothRev",
            "value": 0.000010355804000028002,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / PreRev",
            "value": 0.000011600210999858974,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / PostRev",
            "value": 0.000011790597999606688,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / BothRev",
            "value": 0.000011156677000144554,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / PreRev",
            "value": 0.00001164300600066781,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / PostRev",
            "value": 0.000010441495000122814,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / BothRev",
            "value": 0.000011017537000043375,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / PreRev",
            "value": 0.000011729248999472477,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / PostRev",
            "value": 0.000009874289999970642,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / BothRev",
            "value": 0.000011648750999484035,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / PreRev",
            "value": 0.000011651712999992014,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / PostRev",
            "value": 0.000011706137999681231,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / BothRev",
            "value": 0.000011613009000029706,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / PreRev",
            "value": 0.000011568841999178403,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / PostRev",
            "value": 0.00001168018399948778,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / BothRev",
            "value": 0.00001136285400025372,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Primal",
            "value": 0.000003936917000828543,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cpu / Primal",
            "value": 0.000004005165999842575,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / Primal",
            "value": 0.000004222749999826192,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / Primal",
            "value": 0.000004065625000293949,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / Primal",
            "value": 0.000003921083000022918,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / Primal",
            "value": 0.000004389250001622713,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / Primal",
            "value": 0.000007220375000542844,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Forward",
            "value": 0.000007039583000732819,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cpu / Forward",
            "value": 0.000006009624999933294,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / Forward",
            "value": 0.000006602875000680797,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / Forward",
            "value": 0.000006732334000844275,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / Forward",
            "value": 0.000006628208999245544,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / Forward",
            "value": 0.0000065974999997706615,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / Forward",
            "value": 0.000006611750000956818,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PreRev",
            "value": 0.000006651875000898144,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PostRev",
            "value": 0.00000593891700009408,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / BothRev",
            "value": 0.000006642708998697344,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cpu / BothRev",
            "value": 0.000006042959001206327,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / PreRev",
            "value": 0.000006658415999481804,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / PostRev",
            "value": 0.0000065998329991998615,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / BothRev",
            "value": 0.0000067462079987308246,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / PreRev",
            "value": 0.000006637624999711988,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / PostRev",
            "value": 0.000006131291998826782,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / BothRev",
            "value": 0.000006802999998399173,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / PreRev",
            "value": 0.000006791249999878346,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / PostRev",
            "value": 0.00000609666600030323,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / BothRev",
            "value": 0.000006635792000452056,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / PreRev",
            "value": 0.000006630207999478444,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / PostRev",
            "value": 0.000006663332998869009,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / BothRev",
            "value": 0.000006736375000400585,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / PreRev",
            "value": 0.000006550207999680424,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / PostRev",
            "value": 0.000006796292000217363,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / BothRev",
            "value": 0.000006908040999405785,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Primal",
            "value": 0.000014057599000807386,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cpu / Primal",
            "value": 0.000013341546000447123,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / Primal",
            "value": 0.000014074336999328807,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / Primal",
            "value": 0.000014061817004403565,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / Primal",
            "value": 0.000013377317001868503,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / Primal",
            "value": 0.000014011706996825525,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / Primal",
            "value": 0.000014067187999899034,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Forward",
            "value": 0.000019494789004966152,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cpu / Forward",
            "value": 0.00002032365099876188,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / Forward",
            "value": 0.000019492202998662838,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / Forward",
            "value": 0.00002020574900234351,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / Forward",
            "value": 0.00002036265700007789,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / Forward",
            "value": 0.000019439114003034773,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / Forward",
            "value": 0.00001938024799892446,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PreRev",
            "value": 0.000018851585999073,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PostRev",
            "value": 0.00001969932799693197,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / BothRev",
            "value": 0.00001969635899877176,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cpu / BothRev",
            "value": 0.00001964285199937876,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / PreRev",
            "value": 0.000019660410995129492,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / PostRev",
            "value": 0.00001983268700132612,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / BothRev",
            "value": 0.000019779775000642984,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / PreRev",
            "value": 0.000019683171005453915,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / PostRev",
            "value": 0.00001964789500198094,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / BothRev",
            "value": 0.000018841320001229177,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / PreRev",
            "value": 0.00001989316700201016,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / PostRev",
            "value": 0.00001961027200013632,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / BothRev",
            "value": 0.000019775778004259337,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / PreRev",
            "value": 0.00001891272400098387,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / PostRev",
            "value": 0.000019757457994273864,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / BothRev",
            "value": 0.000018871577005484143,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / PreRev",
            "value": 0.00001970840799913276,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / PostRev",
            "value": 0.000019783946001552972,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / BothRev",
            "value": 0.000018988007002917586,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Primal",
            "value": 0.0000092575870003202,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cpu / Primal",
            "value": 0.000009434104999854751,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / Primal",
            "value": 0.000009437947999686004,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / Primal",
            "value": 0.000009752294999998412,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / Primal",
            "value": 0.0000098590730003707,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / Primal",
            "value": 0.000009865508000075351,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / Primal",
            "value": 0.000009937920000083978,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Forward",
            "value": 0.000013811927999995532,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cpu / Forward",
            "value": 0.000014291182000306437,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / Forward",
            "value": 0.000013800554999761516,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / Forward",
            "value": 0.000013801885000248148,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / Forward",
            "value": 0.000014298298000539944,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / Forward",
            "value": 0.000014313859999674605,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / Forward",
            "value": 0.000014453247000346892,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PreRev",
            "value": 0.000013382869999986723,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PostRev",
            "value": 0.000013965692999590827,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / BothRev",
            "value": 0.000013873085000341234,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cpu / BothRev",
            "value": 0.000013938183999925969,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / PreRev",
            "value": 0.000014021502999639778,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / PostRev",
            "value": 0.000013974366999718767,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / BothRev",
            "value": 0.00001397815700056526,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / PreRev",
            "value": 0.00001419572600025276,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / PostRev",
            "value": 0.000013373986999795306,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / BothRev",
            "value": 0.000013303699000061898,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / PreRev",
            "value": 0.000013370441000006394,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / PostRev",
            "value": 0.000013894267000068794,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / BothRev",
            "value": 0.000013979249999465536,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / PreRev",
            "value": 0.00001391878699996596,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / PostRev",
            "value": 0.000013968471999760368,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / BothRev",
            "value": 0.000014018644000316272,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / PreRev",
            "value": 0.000014030832000571536,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / PostRev",
            "value": 0.000014014595999469748,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / BothRev",
            "value": 0.000013392966000537851,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Primal",
            "value": 0.000006146416000774479,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cpu / Primal",
            "value": 0.000006011875000695,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / Primal",
            "value": 0.000006147124999188236,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / Primal",
            "value": 0.000005931416000748868,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / Primal",
            "value": 0.000006028499999956694,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / Primal",
            "value": 0.000006098457999542006,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / Primal",
            "value": 0.000006059249999452732,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Forward",
            "value": 0.00000936799999908544,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cpu / Forward",
            "value": 0.00000936675000048126,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / Forward",
            "value": 0.000009250832999896374,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / Forward",
            "value": 0.000009406291999766835,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / Forward",
            "value": 0.000009224041999914334,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / Forward",
            "value": 0.000009394458000315352,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / Forward",
            "value": 0.0000094238330002554,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PreRev",
            "value": 0.000009397083000294515,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PostRev",
            "value": 0.00000916045800113352,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / BothRev",
            "value": 0.000009075124999071703,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cpu / BothRev",
            "value": 0.000009183749998555868,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / PreRev",
            "value": 0.000009284416999435051,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / PostRev",
            "value": 0.000009135874999628867,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / BothRev",
            "value": 0.000009343957999590205,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / PreRev",
            "value": 0.000009154457999102306,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / PostRev",
            "value": 0.000009426624999832712,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / BothRev",
            "value": 0.000009395917000802,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / PreRev",
            "value": 0.000009565457999997308,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / PostRev",
            "value": 0.000009753582999110222,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / BothRev",
            "value": 0.00000945441700059746,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / PreRev",
            "value": 0.000009272500001316075,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / PostRev",
            "value": 0.000010191332999966107,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / BothRev",
            "value": 0.000009075542000573478,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / PreRev",
            "value": 0.000009944416999132954,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / PostRev",
            "value": 0.000009749124999871129,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / BothRev",
            "value": 0.000009532334001050912,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / cpu / Primal",
            "value": 0.080899505400157,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / Jax / cpu / Primal",
            "value": 0.0741950484007247,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / cpu / Primal",
            "value": 0.110020970799087,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / cpu / Primal",
            "value": 0.0768434929996146,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / cpu / Primal",
            "value": 0.0777885692004929,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / cpu / Primal",
            "value": 0.1078634524004883,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / cpu / Primal",
            "value": 0.1162536888004979,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / cpu / Forward",
            "value": 0.2201797792004072,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / Jax / cpu / Forward",
            "value": 0.1098421419999795,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / cpu / Forward",
            "value": 0.2244422072006273,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / cpu / Forward",
            "value": 0.2256099673992139,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / cpu / Forward",
            "value": 0.2225291681999806,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / cpu / Forward",
            "value": 0.2276617933996021,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / cpu / Forward",
            "value": 0.2244356390001485,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / cpu / PostRev",
            "value": 0.1550903764000395,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / Jax / cpu / BothRev",
            "value": 0.159147850399313,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / cpu / PostRev",
            "value": 0.2217095341999083,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / cpu / PostRev",
            "value": 0.1551648659995407,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / cpu / PostRev",
            "value": 0.1531619333996786,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / cpu / PostRev",
            "value": 0.2209029424004256,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / cpu / PostRev",
            "value": 0.2119455395994009,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / cpu / Primal",
            "value": 0.0588636435999433,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / Jax / cpu / Primal",
            "value": 0.0584725852000701,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / cpu / Primal",
            "value": 0.0819082220001291,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / cpu / Primal",
            "value": 0.0573682833999555,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / cpu / Primal",
            "value": 0.0595037353999941,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / cpu / Primal",
            "value": 0.081940850400133,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / cpu / Primal",
            "value": 0.084208994599976,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / cpu / Forward",
            "value": 0.1560263205999945,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / Jax / cpu / Forward",
            "value": 0.0778970194000066,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / cpu / Forward",
            "value": 0.1555064522000975,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / cpu / Forward",
            "value": 0.155883676600024,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / cpu / Forward",
            "value": 0.155222054799924,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / cpu / Forward",
            "value": 0.1554214232000959,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / cpu / Forward",
            "value": 0.1529931197999758,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / cpu / PostRev",
            "value": 0.1258066401998803,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / Jax / cpu / BothRev",
            "value": 0.1178408206000312,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / cpu / PostRev",
            "value": 0.159334127799957,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / cpu / PostRev",
            "value": 0.1115662984000664,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / cpu / PostRev",
            "value": 0.1136887470000147,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / cpu / PostRev",
            "value": 0.1518506802000047,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / cpu / PostRev",
            "value": 0.1536706023998704,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / Primal",
            "value": 0.001571668300312,
            "unit": "s"
          },
          {
            "name": "llama / Jax / cpu / Primal",
            "value": 0.0015649831002519,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / Primal",
            "value": 0.0016731691997847,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / Primal",
            "value": 0.0016429657996923,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / Primal",
            "value": 0.0015929958004562,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / Primal",
            "value": 0.0017438130998925,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / Primal",
            "value": 0.0016973753998172,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / Forward",
            "value": 0.0053376148003735,
            "unit": "s"
          },
          {
            "name": "llama / Jax / cpu / Forward",
            "value": 0.0053535273997113,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / Forward",
            "value": 0.0054352809995179,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / Forward",
            "value": 0.0053762980998726,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / Forward",
            "value": 0.0054099285996926,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / Forward",
            "value": 0.0053946525003993,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / Forward",
            "value": 0.0053742748998047,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / PreRev",
            "value": 0.0100640222000947,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / PostRev",
            "value": 0.0112024046000442,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / BothRev",
            "value": 0.01001770820003,
            "unit": "s"
          },
          {
            "name": "llama / Jax / cpu / BothRev",
            "value": 0.0108201048999035,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / PreRev",
            "value": 0.0101580087000911,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / PostRev",
            "value": 0.010865437999746,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / BothRev",
            "value": 0.0102718399000877,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / PreRev",
            "value": 0.0099031419995299,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / PostRev",
            "value": 0.0104675064998446,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / BothRev",
            "value": 0.0097047224000561,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / PreRev",
            "value": 0.0100916408999182,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / PostRev",
            "value": 0.0074776442997972,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / BothRev",
            "value": 0.0099994867996429,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / PreRev",
            "value": 0.0073319606002769,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / PostRev",
            "value": 0.0098993953004537,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / BothRev",
            "value": 0.0111070994003966,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / PreRev",
            "value": 0.0104307562993199,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / PostRev",
            "value": 0.0070914520001679,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / BothRev",
            "value": 0.0098569506997591,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / Primal",
            "value": 0.0014287619999777,
            "unit": "s"
          },
          {
            "name": "llama / Jax / cpu / Primal",
            "value": 0.0013195814000027,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / Primal",
            "value": 0.0013740804999542,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / Primal",
            "value": 0.0013806642999952,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / Primal",
            "value": 0.0013125202000082,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / Primal",
            "value": 0.0013845606000359,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / Primal",
            "value": 0.0014424299999518,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / Forward",
            "value": 0.0047822142000768,
            "unit": "s"
          },
          {
            "name": "llama / Jax / cpu / Forward",
            "value": 0.0049497887000143,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / Forward",
            "value": 0.0047741760999997,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / Forward",
            "value": 0.0048678074000235,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / Forward",
            "value": 0.0048592418000225,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / Forward",
            "value": 0.0048698072000661,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / Forward",
            "value": 0.0048478583999894,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / PreRev",
            "value": 0.0095907016999262,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / PostRev",
            "value": 0.0084760869999627,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / BothRev",
            "value": 0.0094768885999656,
            "unit": "s"
          },
          {
            "name": "llama / Jax / cpu / BothRev",
            "value": 0.006901658300012,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / PreRev",
            "value": 0.0098554177000551,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / PostRev",
            "value": 0.0070501262000107,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / BothRev",
            "value": 0.0095610883000517,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / PreRev",
            "value": 0.0077081763000023,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / PostRev",
            "value": 0.008719911099979,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / BothRev",
            "value": 0.0076865017000272,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / PreRev",
            "value": 0.0094259853000039,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / PostRev",
            "value": 0.0069811200000003,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / BothRev",
            "value": 0.009281559700048,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / PreRev",
            "value": 0.0076510746999701,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / PostRev",
            "value": 0.0086904580999544,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / BothRev",
            "value": 0.0076812276000055,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / PreRev",
            "value": 0.0092732166999667,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / PostRev",
            "value": 0.0078156194999792,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / BothRev",
            "value": 0.0093925514999682,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / Primal",
            "value": 0.0017328582998743,
            "unit": "s"
          },
          {
            "name": "llama / Jax / cpu / Primal",
            "value": 0.0023205666999274,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / Primal",
            "value": 0.0017728540999087,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / Primal",
            "value": 0.0015560875001028,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / Primal",
            "value": 0.0018511124999349,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / Primal",
            "value": 0.0016382291001718,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / Primal",
            "value": 0.0022239833999265,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / Forward",
            "value": 0.0048263041000609,
            "unit": "s"
          },
          {
            "name": "llama / Jax / cpu / Forward",
            "value": 0.0044382000000041,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / Forward",
            "value": 0.0042993583001589,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / Forward",
            "value": 0.0044046208999134,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / Forward",
            "value": 0.0057964999999967,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / Forward",
            "value": 0.0049353750000591,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / Forward",
            "value": 0.0041211959000065,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / PreRev",
            "value": 0.0085149500000625,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / PostRev",
            "value": 0.0145227375000104,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / BothRev",
            "value": 0.0080821000001378,
            "unit": "s"
          },
          {
            "name": "llama / Jax / cpu / BothRev",
            "value": 0.0098389458000383,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / PreRev",
            "value": 0.0082309457999144,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / PostRev",
            "value": 0.0078820540999004,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / BothRev",
            "value": 0.0082912208999914,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / PreRev",
            "value": 0.0109246624999286,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / PostRev",
            "value": 0.0100705625000045,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / BothRev",
            "value": 0.0076956375000008,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / PreRev",
            "value": 0.0084157291999872,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / PostRev",
            "value": 0.0097367542000938,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / BothRev",
            "value": 0.0098312166999676,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / PreRev",
            "value": 0.0089946583999335,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / PostRev",
            "value": 0.0074530583000523,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / BothRev",
            "value": 0.0093738667001161,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / PreRev",
            "value": 0.0080153458000495,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / PostRev",
            "value": 0.0093697207999866,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / BothRev",
            "value": 0.0083430708999003,
            "unit": "s"
          },
          {
            "name": "neural_gcm_dynamic_forcing_deterministic_1_4_deg / JaXPipe / cpu / Primal",
            "value": 52.1907696269991,
            "unit": "s"
          },
          {
            "name": "neural_gcm_dynamic_forcing_deterministic_1_4_deg / Jax / cpu / Primal",
            "value": 52.58961769099551,
            "unit": "s"
          },
          {
            "name": "neural_gcm_dynamic_forcing_deterministic_1_4_deg / HLOOpt / cpu / Primal",
            "value": 52.2038219649985,
            "unit": "s"
          },
          {
            "name": "neural_gcm_dynamic_forcing_deterministic_1_4_deg / PartOpt / cpu / Primal",
            "value": 51.45618928899785,
            "unit": "s"
          },
          {
            "name": "neural_gcm_dynamic_forcing_deterministic_1_4_deg / IPartOpt / cpu / Primal",
            "value": 51.29591600200365,
            "unit": "s"
          },
          {
            "name": "neural_gcm_dynamic_forcing_deterministic_1_4_deg / DefOpt / cpu / Primal",
            "value": 26.162007771999924,
            "unit": "s"
          },
          {
            "name": "neural_gcm_dynamic_forcing_deterministic_1_4_deg / IDefOpt / cpu / Primal",
            "value": 57.27318955800001,
            "unit": "s"
          },
          {
            "name": "neural_gcm_dynamic_forcing_deterministic_1_4_deg / JaXPipe / cpu / Primal",
            "value": 53.869504573000086,
            "unit": "s"
          },
          {
            "name": "neural_gcm_dynamic_forcing_deterministic_1_4_deg / Jax / cpu / Primal",
            "value": 52.9823708240001,
            "unit": "s"
          },
          {
            "name": "neural_gcm_dynamic_forcing_deterministic_1_4_deg / HLOOpt / cpu / Primal",
            "value": 50.81066560799991,
            "unit": "s"
          },
          {
            "name": "neural_gcm_dynamic_forcing_deterministic_1_4_deg / PartOpt / cpu / Primal",
            "value": 51.37964401299996,
            "unit": "s"
          },
          {
            "name": "neural_gcm_dynamic_forcing_deterministic_1_4_deg / IPartOpt / cpu / Primal",
            "value": 50.62901877100012,
            "unit": "s"
          },
          {
            "name": "neural_gcm_dynamic_forcing_deterministic_1_4_deg / DefOpt / cpu / Primal",
            "value": 22.790251147000163,
            "unit": "s"
          },
          {
            "name": "neural_gcm_dynamic_forcing_deterministic_1_4_deg / IDefOpt / cpu / Primal",
            "value": 55.22507502299959,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.00001086220599972876,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cpu / Primal",
            "value": 0.000010868841003684792,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / Primal",
            "value": 0.000010908480995567516,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / Primal",
            "value": 0.000010749648994533344,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / Primal",
            "value": 0.00001085129899729509,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / Primal",
            "value": 0.000010747939995781051,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / Primal",
            "value": 0.00001082641800167039,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Forward",
            "value": 0.00001550283600226976,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cpu / Forward",
            "value": 0.00001638813599856803,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / Forward",
            "value": 0.00001566953099973034,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / Forward",
            "value": 0.00001635328600241337,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / Forward",
            "value": 0.000016363741997338367,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / Forward",
            "value": 0.00001568330000009155,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / Forward",
            "value": 0.00001640036000026157,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / PreRev",
            "value": 0.000015774774001329207,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / PostRev",
            "value": 0.000016534422000404447,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / BothRev",
            "value": 0.000015791756995895413,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cpu / BothRev",
            "value": 0.000016574537999986206,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / PreRev",
            "value": 0.000016466425004182385,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / PostRev",
            "value": 0.00001661365200561704,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / BothRev",
            "value": 0.000016751892006141134,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / PreRev",
            "value": 0.000016570276995480527,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / PostRev",
            "value": 0.000016496910000569188,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / BothRev",
            "value": 0.000015693296998506413,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / PreRev",
            "value": 0.00001588013600121485,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / PostRev",
            "value": 0.00001645674300380051,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / BothRev",
            "value": 0.000016527538005902896,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / PreRev",
            "value": 0.000016597131005255506,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / PostRev",
            "value": 0.000015942637997795827,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / BothRev",
            "value": 0.000015679194002586884,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / PreRev",
            "value": 0.000015806205003173092,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / PostRev",
            "value": 0.00001650923599663656,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / BothRev",
            "value": 0.000016397040999436284,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000007662589999199553,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cpu / Primal",
            "value": 0.000007639446000212047,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / Primal",
            "value": 0.00000766431299962278,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / Primal",
            "value": 0.000007675572999687575,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / Primal",
            "value": 0.000007447490999766159,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / Primal",
            "value": 0.000008074888000010105,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / Primal",
            "value": 0.00000812218600003689,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Forward",
            "value": 0.000010920116999841412,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cpu / Forward",
            "value": 0.000011557613999684691,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / Forward",
            "value": 0.000010878568999942217,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / Forward",
            "value": 0.000010844974000065122,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / Forward",
            "value": 0.00001149363600052311,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / Forward",
            "value": 0.000011421076000260657,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / Forward",
            "value": 0.000011376304999430432,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / PreRev",
            "value": 0.00001158790399949794,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / PostRev",
            "value": 0.000010959878999528884,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / BothRev",
            "value": 0.000011481391999950574,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cpu / BothRev",
            "value": 0.000011628228000517993,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / PreRev",
            "value": 0.000011579060999793,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / PostRev",
            "value": 0.00001102281200019206,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / BothRev",
            "value": 0.000011006384999745932,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / PreRev",
            "value": 0.000011586503999751585,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / PostRev",
            "value": 0.000011605243999838422,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / BothRev",
            "value": 0.000010998425999787289,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / PreRev",
            "value": 0.000011550709000403004,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / PostRev",
            "value": 0.00001105502999962482,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / BothRev",
            "value": 0.000011708973999702722,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / PreRev",
            "value": 0.000011588573999688378,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / PostRev",
            "value": 0.000011291593999885665,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / BothRev",
            "value": 0.000011801708999882976,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / PreRev",
            "value": 0.00001178120399981708,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / PostRev",
            "value": 0.000011622188000728785,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / BothRev",
            "value": 0.000011710267999660571,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.00000457920799999556,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cpu / Primal",
            "value": 0.0000045691249997616975,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / Primal",
            "value": 0.000004767999998875894,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / Primal",
            "value": 0.000004616917000021203,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / Primal",
            "value": 0.000004502750000028754,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / Primal",
            "value": 0.000004562374999295571,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / Primal",
            "value": 0.000004605833999448805,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Forward",
            "value": 0.000007705541000177618,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cpu / Forward",
            "value": 0.000007321249999222345,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / Forward",
            "value": 0.000007207540998933837,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / Forward",
            "value": 0.000007068957998853875,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / Forward",
            "value": 0.000007170417000452289,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / Forward",
            "value": 0.000007015790999503224,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / Forward",
            "value": 0.000007337958999414695,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / PreRev",
            "value": 0.00000751170799958345,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / PostRev",
            "value": 0.000007287291000466212,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / BothRev",
            "value": 0.000007153208000090672,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cpu / BothRev",
            "value": 0.000007271750000654719,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / PreRev",
            "value": 0.0000073695000010047805,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / PostRev",
            "value": 0.000007878207999965525,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / BothRev",
            "value": 0.000007623916999364155,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / PreRev",
            "value": 0.000007309999999051797,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / PostRev",
            "value": 0.00000725179200162529,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / BothRev",
            "value": 0.000007570708001367166,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / PreRev",
            "value": 0.000007482208000510582,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / PostRev",
            "value": 0.000007053415998598212,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / BothRev",
            "value": 0.00000770833299975493,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / PreRev",
            "value": 0.00000717600000098173,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / PostRev",
            "value": 0.000007403249999697436,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / BothRev",
            "value": 0.000007685374999709894,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / PreRev",
            "value": 0.000008127667000735528,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / PostRev",
            "value": 0.000007169374999648426,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / BothRev",
            "value": 0.000007281708998561953,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Primal",
            "value": 0.000009122907998971642,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cpu / Primal",
            "value": 0.00000858146200334886,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / Primal",
            "value": 0.000008446453000942711,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / Primal",
            "value": 0.000009119057998759672,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / Primal",
            "value": 0.000008678434001922143,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / Primal",
            "value": 0.000008588173004682176,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / Primal",
            "value": 0.000009058318995812442,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Forward",
            "value": 0.000012456188000214752,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cpu / Forward",
            "value": 0.000013251261996629182,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / Forward",
            "value": 0.000013231254000857008,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / Forward",
            "value": 0.000012463460996514189,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / Forward",
            "value": 0.000012527722996310331,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / Forward",
            "value": 0.000012456481003027877,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / Forward",
            "value": 0.000013321489001100415,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PreRev",
            "value": 0.00001414082599512767,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PostRev",
            "value": 0.000013319178004167042,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / BothRev",
            "value": 0.000014191672002198174,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cpu / BothRev",
            "value": 0.000014268186998378951,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / PreRev",
            "value": 0.000013410581996140536,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / PostRev",
            "value": 0.000013371571003517602,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / BothRev",
            "value": 0.00001411394000024302,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / PreRev",
            "value": 0.000014149155998893548,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / PostRev",
            "value": 0.000014275854999141302,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / BothRev",
            "value": 0.000013390447995334398,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / PreRev",
            "value": 0.00001413621800020337,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / PostRev",
            "value": 0.000014237278999644332,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / BothRev",
            "value": 0.00001425748899782775,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / PreRev",
            "value": 0.000013476435000484345,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / PostRev",
            "value": 0.000014084252994507552,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / BothRev",
            "value": 0.00001413995299662929,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / PreRev",
            "value": 0.000014214360999176278,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / PostRev",
            "value": 0.00001418072399974335,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / BothRev",
            "value": 0.00001414315000147326,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Primal",
            "value": 0.0000060294720005913405,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cpu / Primal",
            "value": 0.000006639731999712239,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / Primal",
            "value": 0.000005988029999571154,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / Primal",
            "value": 0.000006626885000514449,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / Primal",
            "value": 0.00000650328299980174,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / Primal",
            "value": 0.000006222784999408759,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / Primal",
            "value": 0.000006140245999631588,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Forward",
            "value": 0.000008740420000322047,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cpu / Forward",
            "value": 0.000008708135999768274,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / Forward",
            "value": 0.000008825111000078323,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / Forward",
            "value": 0.000008807786000033957,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / Forward",
            "value": 0.00000954319099946588,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / Forward",
            "value": 0.00000956816500001878,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / Forward",
            "value": 0.000009479334999923594,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PreRev",
            "value": 0.000009542798999973456,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PostRev",
            "value": 0.000009593992999725742,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / BothRev",
            "value": 0.000010175207999964186,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cpu / BothRev",
            "value": 0.000010146894000172323,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / PreRev",
            "value": 0.0000101559790000465,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / PostRev",
            "value": 0.00001017430100000638,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / BothRev",
            "value": 0.00000958084800004144,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / PreRev",
            "value": 0.000009613704999537731,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / PostRev",
            "value": 0.000009455564000745651,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / BothRev",
            "value": 0.000009365464999973482,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / PreRev",
            "value": 0.00001007424199997331,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / PostRev",
            "value": 0.000009849574000327268,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / BothRev",
            "value": 0.00001003533199946105,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / PreRev",
            "value": 0.00000992655299978651,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / PostRev",
            "value": 0.00000954138100041746,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / BothRev",
            "value": 0.000010060782999971706,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / PreRev",
            "value": 0.00000938486299946817,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / PostRev",
            "value": 0.000009990484999434556,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / BothRev",
            "value": 0.000010253366000142706,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Primal",
            "value": 0.000003656499999124208,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cpu / Primal",
            "value": 0.000003874374999213614,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / Primal",
            "value": 0.000003636082999946666,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / Primal",
            "value": 0.000003638707999925828,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / Primal",
            "value": 0.000003783167001529364,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / Primal",
            "value": 0.000003641582999989623,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / Primal",
            "value": 0.000003649791999123409,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Forward",
            "value": 0.000005743875000916887,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cpu / Forward",
            "value": 0.000005925082999965525,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / Forward",
            "value": 0.00000573383300070418,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / Forward",
            "value": 0.000005564166000112891,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / Forward",
            "value": 0.000005671332999554579,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / Forward",
            "value": 0.000005961875000139116,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / Forward",
            "value": 0.000005978999999570078,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PreRev",
            "value": 0.00000652191699919058,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PostRev",
            "value": 0.000006078667000110727,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / BothRev",
            "value": 0.000005947874999037595,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cpu / BothRev",
            "value": 0.000005933374999585794,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / PreRev",
            "value": 0.0000064469999997527336,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / PostRev",
            "value": 0.000006745624999894062,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / BothRev",
            "value": 0.000005950208000285784,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / PreRev",
            "value": 0.000005845459001648123,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / PostRev",
            "value": 0.000006011041001329431,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / BothRev",
            "value": 0.000005887290999453399,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / PreRev",
            "value": 0.000005819207999593346,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / PostRev",
            "value": 0.000005930082999839215,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / BothRev",
            "value": 0.0000060192079999978885,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / PreRev",
            "value": 0.000006285167000896763,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / PostRev",
            "value": 0.000006052124999769148,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / BothRev",
            "value": 0.000006261542001084308,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / PreRev",
            "value": 0.000005868874999578111,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / PostRev",
            "value": 0.000006151542000225163,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / BothRev",
            "value": 0.000006214832999830833,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Primal",
            "value": 0.000011653340996417682,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cpu / Primal",
            "value": 0.00001223727100295946,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / Primal",
            "value": 0.000011669901003187987,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / Primal",
            "value": 0.0000115240920058568,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / Primal",
            "value": 0.000011693938999087547,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / Primal",
            "value": 0.000011741633999918122,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / Primal",
            "value": 0.000011486888994113545,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Forward",
            "value": 0.000017929336994711776,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cpu / Forward",
            "value": 0.000017982299999857785,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / Forward",
            "value": 0.000018099537999660244,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / Forward",
            "value": 0.00001806161400600104,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / Forward",
            "value": 0.000017171612998936325,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / Forward",
            "value": 0.00001721074999659322,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / Forward",
            "value": 0.000017938293996849096,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PreRev",
            "value": 0.000016721480998967308,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PostRev",
            "value": 0.00001593121200130554,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / BothRev",
            "value": 0.000016869029001099987,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cpu / BothRev",
            "value": 0.000016029979997256304,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / PreRev",
            "value": 0.000016593193999142385,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / PostRev",
            "value": 0.00001602400399860926,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / BothRev",
            "value": 0.000015964537997206208,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / PreRev",
            "value": 0.00001672834799683187,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / PostRev",
            "value": 0.000016658188003930263,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / BothRev",
            "value": 0.000015917650001938454,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / PreRev",
            "value": 0.000016522603000339587,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / PostRev",
            "value": 0.00001671111399627989,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / BothRev",
            "value": 0.000016699643994797953,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / PreRev",
            "value": 0.000016823965001094622,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / PostRev",
            "value": 0.000015741570001409856,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / BothRev",
            "value": 0.000015780762005306315,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / PreRev",
            "value": 0.0000167067579968716,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / PostRev",
            "value": 0.00001670003600156633,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / BothRev",
            "value": 0.00001603664400317939,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Primal",
            "value": 0.000008781968999755917,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cpu / Primal",
            "value": 0.000008306063000418362,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / Primal",
            "value": 0.000008271269000033499,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / Primal",
            "value": 0.000008641820999400807,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / Primal",
            "value": 0.000008684035999976914,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / Primal",
            "value": 0.00000832648299910943,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / Primal",
            "value": 0.000008302527000523696,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Forward",
            "value": 0.00001256811499933974,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cpu / Forward",
            "value": 0.000012624768000023325,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / Forward",
            "value": 0.000012677833999987342,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / Forward",
            "value": 0.000011695874999531953,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / Forward",
            "value": 0.000011696088000462624,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / Forward",
            "value": 0.000011732171999938146,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / Forward",
            "value": 0.000012425075999999536,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PreRev",
            "value": 0.000011691863000123704,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PostRev",
            "value": 0.000011253314999521535,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / BothRev",
            "value": 0.000011740609000298718,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cpu / BothRev",
            "value": 0.000010971728999720652,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / PreRev",
            "value": 0.00001166262300012022,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / PostRev",
            "value": 0.000011730991000149515,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / BothRev",
            "value": 0.000011657118000584888,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / PreRev",
            "value": 0.000011110657000244827,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / PostRev",
            "value": 0.000011198608999620774,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / BothRev",
            "value": 0.00001112657100020442,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / PreRev",
            "value": 0.000011672812000142583,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / PostRev",
            "value": 0.000010979556000165758,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / BothRev",
            "value": 0.00001168186099948798,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / PreRev",
            "value": 0.000011638329000561496,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / PostRev",
            "value": 0.000011736351999388715,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / BothRev",
            "value": 0.0000116089169996485,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / PreRev",
            "value": 0.000011644865000562276,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / PostRev",
            "value": 0.00001158432400006859,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / BothRev",
            "value": 0.0000109645870006716,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Primal",
            "value": 0.000004846334000831121,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cpu / Primal",
            "value": 0.0000048279160000674895,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / Primal",
            "value": 0.000004797457999302424,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / Primal",
            "value": 0.000004814959000214003,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / Primal",
            "value": 0.000004739833999337861,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / Primal",
            "value": 0.000004826250000405707,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / Primal",
            "value": 0.00000475183299931814,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Forward",
            "value": 0.000008163666998370899,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cpu / Forward",
            "value": 0.000007459917000232963,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / Forward",
            "value": 0.00000778883400016639,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / Forward",
            "value": 0.000007644375000381842,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / Forward",
            "value": 0.000007797000000209664,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / Forward",
            "value": 0.000007489083000109531,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / Forward",
            "value": 0.000007583290998809389,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PreRev",
            "value": 0.000006860792000225047,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PostRev",
            "value": 0.000007171124998421874,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / BothRev",
            "value": 0.000006857500000478467,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cpu / BothRev",
            "value": 0.000006464125001002685,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / PreRev",
            "value": 0.0000066140420003648616,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / PostRev",
            "value": 0.000006630957999732345,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / BothRev",
            "value": 0.000006816291999712121,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / PreRev",
            "value": 0.000006738666999808629,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / PostRev",
            "value": 0.000006988499999351916,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / BothRev",
            "value": 0.000007008374999713851,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / PreRev",
            "value": 0.000006792709000364994,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / PostRev",
            "value": 0.000006532542000059039,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / BothRev",
            "value": 0.0000065915830000449206,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / PreRev",
            "value": 0.000006485583000539919,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / PostRev",
            "value": 0.000006741417000739602,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / BothRev",
            "value": 0.000006951624998691841,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / PreRev",
            "value": 0.000006797000000005937,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / PostRev",
            "value": 0.00000693262499953562,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / BothRev",
            "value": 0.0000068010420000064185,
            "unit": "s"
          },
          {
            "name": "value_and_grad / JaXPipe / cpu / Primal",
            "value": 0.000017069770001398866,
            "unit": "s"
          },
          {
            "name": "value_and_grad / Jax / cpu / Primal",
            "value": 0.000017328713001916186,
            "unit": "s"
          },
          {
            "name": "value_and_grad / HLOOpt / cpu / Primal",
            "value": 0.00001817152800504118,
            "unit": "s"
          },
          {
            "name": "value_and_grad / PartOpt / cpu / Primal",
            "value": 0.00001732839999749558,
            "unit": "s"
          },
          {
            "name": "value_and_grad / IPartOpt / cpu / Primal",
            "value": 0.000018360113004746382,
            "unit": "s"
          },
          {
            "name": "value_and_grad / DefOpt / cpu / Primal",
            "value": 0.00001837627599888947,
            "unit": "s"
          },
          {
            "name": "value_and_grad / IDefOpt / cpu / Primal",
            "value": 0.0000182682729937369,
            "unit": "s"
          },
          {
            "name": "value_and_grad / JaXPipe / cpu / Primal",
            "value": 0.000012067774999195535,
            "unit": "s"
          },
          {
            "name": "value_and_grad / Jax / cpu / Primal",
            "value": 0.000012766986999849903,
            "unit": "s"
          },
          {
            "name": "value_and_grad / HLOOpt / cpu / Primal",
            "value": 0.000012755773999742817,
            "unit": "s"
          },
          {
            "name": "value_and_grad / PartOpt / cpu / Primal",
            "value": 0.000012791562000529666,
            "unit": "s"
          },
          {
            "name": "value_and_grad / IPartOpt / cpu / Primal",
            "value": 0.000012037997999868822,
            "unit": "s"
          },
          {
            "name": "value_and_grad / DefOpt / cpu / Primal",
            "value": 0.000012994573000469245,
            "unit": "s"
          },
          {
            "name": "value_and_grad / IDefOpt / cpu / Primal",
            "value": 0.000012961368000105722,
            "unit": "s"
          },
          {
            "name": "value_and_grad / JaXPipe / cpu / Primal",
            "value": 0.000008195583999622612,
            "unit": "s"
          },
          {
            "name": "value_and_grad / Jax / cpu / Primal",
            "value": 0.000008288541001093109,
            "unit": "s"
          },
          {
            "name": "value_and_grad / HLOOpt / cpu / Primal",
            "value": 0.000008619750000434579,
            "unit": "s"
          },
          {
            "name": "value_and_grad / PartOpt / cpu / Primal",
            "value": 0.000008431166999798734,
            "unit": "s"
          },
          {
            "name": "value_and_grad / IPartOpt / cpu / Primal",
            "value": 0.000008161125000697212,
            "unit": "s"
          },
          {
            "name": "value_and_grad / DefOpt / cpu / Primal",
            "value": 0.000008072917000390589,
            "unit": "s"
          },
          {
            "name": "value_and_grad / IDefOpt / cpu / Primal",
            "value": 0.00000806962500064401,
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
          "id": "ccd2a9ed6ffe683613d4dac28d087dc8e7e8ffad",
          "message": "Update EnzymeAD/Enzyme to commit 98711689e3cd2d5a7d4521ad46488c7385086b4a (#1808)\n\nDiff: https://github.com/EnzymeAD/Enzyme/compare/4d791302386a5501a59a31af02a575839adf583a...98711689e3cd2d5a7d4521ad46488c7385086b4a\n\nCo-authored-by: enzymead-bot[bot] <238314553+enzymead-bot[bot]@users.noreply.github.com>",
          "timestamp": "2025-12-20T21:12:02-05:00",
          "tree_id": "8dff2c4ee989459a141a8b140b06b9a4e4ad6416",
          "url": "https://github.com/EnzymeAD/Enzyme-JAX/commit/ccd2a9ed6ffe683613d4dac28d087dc8e7e8ffad"
        },
        "date": 1766296370898,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "actmtch / JaXPipe / cpu / Primal",
            "value": 0.000004702274999999645,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cpu / Primal",
            "value": 0.000004803378999895358,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / Primal",
            "value": 0.000005314423000072566,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / Primal",
            "value": 0.000004780164000294462,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / Primal",
            "value": 0.000004859147999923152,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / Primal",
            "value": 0.000005259629999727622,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / Primal",
            "value": 0.00000531850100014708,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Forward",
            "value": 0.000008088509000117484,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cpu / Forward",
            "value": 0.000007334985999932541,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / Forward",
            "value": 0.000008306906000143499,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / Forward",
            "value": 0.000008289819000310671,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / Forward",
            "value": 0.00000806456000009348,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / Forward",
            "value": 0.00000832340600027237,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / Forward",
            "value": 0.000008385865000036575,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PreRev",
            "value": 0.000008164549999946757,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PostRev",
            "value": 0.000007041426999876421,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / BothRev",
            "value": 0.000008455519000108325,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cpu / BothRev",
            "value": 0.000007392704000267258,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / PreRev",
            "value": 0.000008194881999770587,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / PostRev",
            "value": 0.000008482185000048048,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / BothRev",
            "value": 0.000008213974999762287,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / PreRev",
            "value": 0.000008162631999766745,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / PostRev",
            "value": 0.0000073207609998462435,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / BothRev",
            "value": 0.000008404875999985962,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / PreRev",
            "value": 0.000008395749000101204,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / PostRev",
            "value": 0.0000072043319996737405,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / BothRev",
            "value": 0.000008141123999848787,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / PreRev",
            "value": 0.00000812721500005864,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / PostRev",
            "value": 0.000008318251000218879,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / BothRev",
            "value": 0.000008226885000112816,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / PreRev",
            "value": 0.000008194914999876346,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / PostRev",
            "value": 0.000008584837999933371,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / BothRev",
            "value": 0.000008171176999894669,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Primal",
            "value": 0.000009020799996505955,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cpu / Primal",
            "value": 0.00000917450899578398,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / Primal",
            "value": 0.00001114610500371782,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / Primal",
            "value": 0.00000935799400031101,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / Primal",
            "value": 0.00000924380699871108,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / Primal",
            "value": 0.00001115810900228098,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / Primal",
            "value": 0.000011631352004769724,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Forward",
            "value": 0.000016023434000089765,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cpu / Forward",
            "value": 0.000013828819995978848,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / Forward",
            "value": 0.00001614435499504907,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / Forward",
            "value": 0.00001604773699364159,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / Forward",
            "value": 0.000016267933999188245,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / Forward",
            "value": 0.00001677726300113136,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / Forward",
            "value": 0.000016951026002061555,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PreRev",
            "value": 0.00001700167999661062,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PostRev",
            "value": 0.000013291955001477618,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / BothRev",
            "value": 0.000016143902001203968,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cpu / BothRev",
            "value": 0.000013905957996030338,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / PreRev",
            "value": 0.000016098747997602914,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / PostRev",
            "value": 0.00001621254099882208,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / BothRev",
            "value": 0.00001703700800135266,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / PreRev",
            "value": 0.00001690048199816374,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / PostRev",
            "value": 0.000013223948997620028,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / BothRev",
            "value": 0.00001631156000075862,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / PreRev",
            "value": 0.000016353090999473352,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / PostRev",
            "value": 0.000013856639998266474,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / BothRev",
            "value": 0.00001683711399527965,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / PreRev",
            "value": 0.000015995574001863132,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / PostRev",
            "value": 0.00001631929499853868,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / BothRev",
            "value": 0.000016195192998566198,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / PreRev",
            "value": 0.000016135372999997345,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / PostRev",
            "value": 0.00001690955700178165,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / BothRev",
            "value": 0.000016881183000805322,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / tpu / Primal",
            "value": 0.0001343775450004,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / tpu / Primal",
            "value": 0.0001357286750007,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / tpu / Primal",
            "value": 0.0001332362039993,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / tpu / Primal",
            "value": 0.0001483250810015,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / tpu / Primal",
            "value": 0.0001500979309985,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / tpu / Primal",
            "value": 0.0001498967900006,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / tpu / Primal",
            "value": 0.0001689454679999,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / tpu / Forward",
            "value": 0.0002375235350009,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / tpu / Forward",
            "value": 0.0002183785569995,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / tpu / Forward",
            "value": 0.0002077660930008,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / tpu / Forward",
            "value": 0.0002064756630006,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / tpu / Forward",
            "value": 0.0002251050200011,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / tpu / Forward",
            "value": 0.0001903612360001,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / tpu / Forward",
            "value": 0.0002223090980005,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / tpu / PreRev",
            "value": 0.0002056650020003,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / tpu / PostRev",
            "value": 0.0001883981459995,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / tpu / BothRev",
            "value": 0.0002080919330001,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / tpu / BothRev",
            "value": 0.000211916283999,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / tpu / PreRev",
            "value": 0.0002076864030004,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / tpu / PostRev",
            "value": 0.0002112812539999,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / tpu / BothRev",
            "value": 0.000210445865001,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / tpu / PreRev",
            "value": 0.0002289970519996,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / tpu / PostRev",
            "value": 0.0002286242209993,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / tpu / BothRev",
            "value": 0.0002216558189993,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / tpu / PreRev",
            "value": 0.0002344212839998,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / tpu / PostRev",
            "value": 0.0002104275140009,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / tpu / BothRev",
            "value": 0.000235131714,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / tpu / PreRev",
            "value": 0.0002248801100013,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / tpu / PostRev",
            "value": 0.0002171527360005,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / tpu / BothRev",
            "value": 0.0002056475419994,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / tpu / PreRev",
            "value": 0.0002033244310005,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / tpu / PostRev",
            "value": 0.0002042252219998,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / tpu / BothRev",
            "value": 0.0002464124780017,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Primal",
            "value": 0.00000608986699990055,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cpu / Primal",
            "value": 0.000006162010000480222,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / Primal",
            "value": 0.000007315634000406135,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / Primal",
            "value": 0.000006239172999812581,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / Primal",
            "value": 0.000006173214000227745,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / Primal",
            "value": 0.000007340987000134191,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / Primal",
            "value": 0.000007316274000004341,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Forward",
            "value": 0.000011167212999680488,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cpu / Forward",
            "value": 0.000009301179000431147,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / Forward",
            "value": 0.00001139713200063852,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / Forward",
            "value": 0.000011216178999347904,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / Forward",
            "value": 0.000010858931000257144,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / Forward",
            "value": 0.000011349465000421332,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / Forward",
            "value": 0.000011225639999793205,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PreRev",
            "value": 0.00001075298700015992,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PostRev",
            "value": 0.00000890620999962266,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / BothRev",
            "value": 0.0000106671510002343,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cpu / BothRev",
            "value": 0.0000093746269994881,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / PreRev",
            "value": 0.000011275394999756827,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / PostRev",
            "value": 0.000011223063000215915,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / BothRev",
            "value": 0.00001133837500037771,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / PreRev",
            "value": 0.000010740521000116132,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / PostRev",
            "value": 0.000008910032000130742,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / BothRev",
            "value": 0.000010670335000213526,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / PreRev",
            "value": 0.000011259515999881842,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / PostRev",
            "value": 0.0000093762370006516,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / BothRev",
            "value": 0.00001127903900032834,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / PreRev",
            "value": 0.000010716699999647972,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / PostRev",
            "value": 0.000011272551000729436,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / BothRev",
            "value": 0.000011155514999700245,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / PreRev",
            "value": 0.000011307769999802984,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / PostRev",
            "value": 0.00001121338699977059,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / BothRev",
            "value": 0.000011301522999929149,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Primal",
            "value": 0.0000038817080003354935,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cpu / Primal",
            "value": 0.000003995417000624002,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / Primal",
            "value": 0.000004445250000571832,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / Primal",
            "value": 0.00000392308300069999,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / Primal",
            "value": 0.000003969708001022809,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / Primal",
            "value": 0.000004319417001170223,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / Primal",
            "value": 0.000004340541998317349,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / Forward",
            "value": 0.000007213166998553788,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cpu / Forward",
            "value": 0.000005882790999748977,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / Forward",
            "value": 0.0000068074170012550895,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / Forward",
            "value": 0.0000072412080007779875,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / Forward",
            "value": 0.00000665945899891085,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / Forward",
            "value": 0.000006667917001323076,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / Forward",
            "value": 0.000006743750000168802,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PreRev",
            "value": 0.000006420583000362967,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / PostRev",
            "value": 0.000005679249999957392,
            "unit": "s"
          },
          {
            "name": "actmtch / JaXPipe / cpu / BothRev",
            "value": 0.000006825042000855319,
            "unit": "s"
          },
          {
            "name": "actmtch / Jax / cpu / BothRev",
            "value": 0.000005972541999653913,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / PreRev",
            "value": 0.000006906041000547703,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / PostRev",
            "value": 0.000006926624999323394,
            "unit": "s"
          },
          {
            "name": "actmtch / HLOOpt / cpu / BothRev",
            "value": 0.000007027916999504669,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / PreRev",
            "value": 0.000006937207999726525,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / PostRev",
            "value": 0.000005816000000777421,
            "unit": "s"
          },
          {
            "name": "actmtch / PartOpt / cpu / BothRev",
            "value": 0.000006563292001374066,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / PreRev",
            "value": 0.000006864917000712012,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / PostRev",
            "value": 0.000005847249998623738,
            "unit": "s"
          },
          {
            "name": "actmtch / IPartOpt / cpu / BothRev",
            "value": 0.000006967499999518623,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / PreRev",
            "value": 0.000006830584001363604,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / PostRev",
            "value": 0.000006966958000703017,
            "unit": "s"
          },
          {
            "name": "actmtch / DefOpt / cpu / BothRev",
            "value": 0.000006638041999394773,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / PreRev",
            "value": 0.0000064467079992027724,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / PostRev",
            "value": 0.000006923666998773115,
            "unit": "s"
          },
          {
            "name": "actmtch / IDefOpt / cpu / BothRev",
            "value": 0.0000068382920017029394,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Primal",
            "value": 0.000004978382999979658,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cpu / Primal",
            "value": 0.000005015404999994644,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / Primal",
            "value": 0.000004999419999876409,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / Primal",
            "value": 0.000004924974000005022,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / Primal",
            "value": 0.000004955943999902956,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / Primal",
            "value": 0.000004905689000224811,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / Primal",
            "value": 0.000004975211000328272,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Forward",
            "value": 0.000008113399000194477,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cpu / Forward",
            "value": 0.000008053351999933512,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / Forward",
            "value": 0.000008185784000033892,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / Forward",
            "value": 0.000008150231999934476,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / Forward",
            "value": 0.000008045451999805663,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / Forward",
            "value": 0.000008131183999921631,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / Forward",
            "value": 0.000008105549999982032,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PreRev",
            "value": 0.00000899424600038401,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PostRev",
            "value": 0.000009083085999918696,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / BothRev",
            "value": 0.000009119258999817248,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cpu / BothRev",
            "value": 0.000008959981999851152,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / PreRev",
            "value": 0.000009098371000163752,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / PostRev",
            "value": 0.000009128237999902922,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / BothRev",
            "value": 0.000008993000999907963,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / PreRev",
            "value": 0.000008960515999660857,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / PostRev",
            "value": 0.0000089682040002117,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / BothRev",
            "value": 0.00000891550999995161,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / PreRev",
            "value": 0.000009266067999760708,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / PostRev",
            "value": 0.000009180564999951456,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / BothRev",
            "value": 0.000009099712000079307,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / PreRev",
            "value": 0.000008933289999731642,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / PostRev",
            "value": 0.000009156374000212964,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / BothRev",
            "value": 0.00000908499400020446,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / PreRev",
            "value": 0.000009024375000080908,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / PostRev",
            "value": 0.000009146531000169487,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / BothRev",
            "value": 0.000008956281999871863,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Primal",
            "value": 0.000009495516002061776,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cpu / Primal",
            "value": 0.00000953094000578858,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / Primal",
            "value": 0.000009854010000708512,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / Primal",
            "value": 0.000009466636001889128,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / Primal",
            "value": 0.00000941384700126946,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / Primal",
            "value": 0.000009391296000103469,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / Primal",
            "value": 0.000009381316995131782,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Forward",
            "value": 0.00001416991800215328,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cpu / Forward",
            "value": 0.000014236472998163664,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / Forward",
            "value": 0.000014175114003592173,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / Forward",
            "value": 0.000014310254002339209,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / Forward",
            "value": 0.000014793818001635371,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / Forward",
            "value": 0.000014725603999977463,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / Forward",
            "value": 0.00001417530400067335,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PreRev",
            "value": 0.000016066561998741237,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PostRev",
            "value": 0.000015989157000149134,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / BothRev",
            "value": 0.000016021216004446613,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cpu / BothRev",
            "value": 0.000015960744000039995,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / PreRev",
            "value": 0.000015883964006206953,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / PostRev",
            "value": 0.000015984896999725607,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / BothRev",
            "value": 0.000015972164997947402,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / PreRev",
            "value": 0.00001526257200021064,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / PostRev",
            "value": 0.000015085569000802937,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / BothRev",
            "value": 0.00001527259599970421,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / PreRev",
            "value": 0.000016080446999694687,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / PostRev",
            "value": 0.00001588132300094003,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / BothRev",
            "value": 0.000015886333996604662,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / PreRev",
            "value": 0.00001591490400460316,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / PostRev",
            "value": 0.000015898304001893848,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / BothRev",
            "value": 0.000015968744002748282,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / PreRev",
            "value": 0.000015974112000549212,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / PostRev",
            "value": 0.000016144445995450952,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / BothRev",
            "value": 0.000016019881993997843,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / tpu / Primal",
            "value": 0.0001510365699996,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / tpu / Primal",
            "value": 0.0001536220509988,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / tpu / Primal",
            "value": 0.000163145765,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / tpu / Primal",
            "value": 0.0001402160460002,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / tpu / Primal",
            "value": 0.0001386605349998,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / tpu / Primal",
            "value": 0.0001363315550006,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / tpu / Primal",
            "value": 0.0001359563350015,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / tpu / Forward",
            "value": 0.0002281023310006,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / tpu / Forward",
            "value": 0.0002260267199999,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / tpu / Forward",
            "value": 0.0001979109189996,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / tpu / Forward",
            "value": 0.0002036523719998,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / tpu / Forward",
            "value": 0.0002075607330007,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / tpu / Forward",
            "value": 0.0002174885170006,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / tpu / Forward",
            "value": 0.0002193678080002,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / tpu / PreRev",
            "value": 0.000215581847,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / tpu / PostRev",
            "value": 0.0002119302949995,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / tpu / BothRev",
            "value": 0.0002450946879998,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / tpu / BothRev",
            "value": 0.0002427721370004,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / tpu / PreRev",
            "value": 0.0002230078390002,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / tpu / PostRev",
            "value": 0.000221779419,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / tpu / BothRev",
            "value": 0.0002306243820003,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / tpu / PreRev",
            "value": 0.0002276152009999,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / tpu / PostRev",
            "value": 0.000218940738001,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / tpu / BothRev",
            "value": 0.000218610247999,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / tpu / PreRev",
            "value": 0.0002094694240004,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / tpu / PostRev",
            "value": 0.0002111236439995,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / tpu / BothRev",
            "value": 0.0002114416349995,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / tpu / PreRev",
            "value": 0.0002180091869995,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / tpu / PostRev",
            "value": 0.0002098498640007,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / tpu / BothRev",
            "value": 0.0002204485580004,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / tpu / PreRev",
            "value": 0.0002216934489988,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / tpu / PostRev",
            "value": 0.0002356269339998,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / tpu / BothRev",
            "value": 0.0002334708929993,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Primal",
            "value": 0.000006355812000037986,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cpu / Primal",
            "value": 0.000006378181000400218,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / Primal",
            "value": 0.000006222321000677767,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / Primal",
            "value": 0.000006374496999342227,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / Primal",
            "value": 0.000006327334999696177,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / Primal",
            "value": 0.000006368014000145195,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / Primal",
            "value": 0.000006397543999810296,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Forward",
            "value": 0.000009685622999313636,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cpu / Forward",
            "value": 0.000010224020999885396,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / Forward",
            "value": 0.000010094946000208438,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / Forward",
            "value": 0.000009810581000238017,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / Forward",
            "value": 0.000009717910999825109,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / Forward",
            "value": 0.0000101466750002146,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / Forward",
            "value": 0.000010365707999881124,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PreRev",
            "value": 0.000011057191999498171,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PostRev",
            "value": 0.00001105131500025891,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / BothRev",
            "value": 0.000011056519999328884,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cpu / BothRev",
            "value": 0.00001095915000041714,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / PreRev",
            "value": 0.000011006892999830596,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / PostRev",
            "value": 0.000010449575999700755,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / BothRev",
            "value": 0.00001098864399955346,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / PreRev",
            "value": 0.000010406917999716825,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / PostRev",
            "value": 0.000010994965999998385,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / BothRev",
            "value": 0.000010405632999209048,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / PreRev",
            "value": 0.000010993228000188538,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / PostRev",
            "value": 0.000010991289999765286,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / BothRev",
            "value": 0.000010944346000542284,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / PreRev",
            "value": 0.000010937426999589662,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / PostRev",
            "value": 0.000010988173999976423,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / BothRev",
            "value": 0.000010344987000280523,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / PreRev",
            "value": 0.000011053742000513013,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / PostRev",
            "value": 0.000010969239999212731,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / BothRev",
            "value": 0.00001098850299968035,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Primal",
            "value": 0.000003999500000645639,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cpu / Primal",
            "value": 0.000004046959000334027,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / Primal",
            "value": 0.00000402429200039478,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / Primal",
            "value": 0.000003922042000340298,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / Primal",
            "value": 0.000003970458001276711,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / Primal",
            "value": 0.000003973167000367539,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / Primal",
            "value": 0.000003951042001062888,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / Forward",
            "value": 0.000005804417000035756,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cpu / Forward",
            "value": 0.000005918457998632221,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / Forward",
            "value": 0.000005863709000550443,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / Forward",
            "value": 0.00000624591700034216,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / Forward",
            "value": 0.000006290958999670693,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / Forward",
            "value": 0.000006352832999255043,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / Forward",
            "value": 0.000006070166999052162,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PreRev",
            "value": 0.000007169208000050276,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / PostRev",
            "value": 0.000006993083999986993,
            "unit": "s"
          },
          {
            "name": "add_one / JaXPipe / cpu / BothRev",
            "value": 0.000007295084000361385,
            "unit": "s"
          },
          {
            "name": "add_one / Jax / cpu / BothRev",
            "value": 0.000006939584000065224,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / PreRev",
            "value": 0.000006895000000440632,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / PostRev",
            "value": 0.000006975917000090703,
            "unit": "s"
          },
          {
            "name": "add_one / HLOOpt / cpu / BothRev",
            "value": 0.000007582125001135864,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / PreRev",
            "value": 0.00000712783299968578,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / PostRev",
            "value": 0.000007391082999674836,
            "unit": "s"
          },
          {
            "name": "add_one / PartOpt / cpu / BothRev",
            "value": 0.000007344792000367306,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / PreRev",
            "value": 0.000007198208000772865,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / PostRev",
            "value": 0.000007074290999298683,
            "unit": "s"
          },
          {
            "name": "add_one / IPartOpt / cpu / BothRev",
            "value": 0.000007077291998939472,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / PreRev",
            "value": 0.000007035167000140063,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / PostRev",
            "value": 0.00000701979199948255,
            "unit": "s"
          },
          {
            "name": "add_one / DefOpt / cpu / BothRev",
            "value": 0.0000071822499994596,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / PreRev",
            "value": 0.000007101709001290146,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / PostRev",
            "value": 0.000007172666000769823,
            "unit": "s"
          },
          {
            "name": "add_one / IDefOpt / cpu / BothRev",
            "value": 0.000007322375000512693,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Primal",
            "value": 0.000005142609999893466,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cpu / Primal",
            "value": 0.000005141916999946261,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / Primal",
            "value": 0.000005163902000276721,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / Primal",
            "value": 0.000005193502000111039,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / Primal",
            "value": 0.000004994976000034512,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / Primal",
            "value": 0.00000502365500005908,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / Primal",
            "value": 0.00000518807900016327,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Forward",
            "value": 0.00000839091499983624,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cpu / Forward",
            "value": 0.000008417766000093251,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / Forward",
            "value": 0.000008254928000042127,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / Forward",
            "value": 0.000008286760999908437,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / Forward",
            "value": 0.000008188747000076547,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / Forward",
            "value": 0.00000827258300023459,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / Forward",
            "value": 0.00000823787599983916,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PreRev",
            "value": 0.000011468715999853885,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PostRev",
            "value": 0.000011253392999606147,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / BothRev",
            "value": 0.000011218752999866413,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cpu / BothRev",
            "value": 0.000011122594999960712,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / PreRev",
            "value": 0.000011389807999876212,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / PostRev",
            "value": 0.000011194853999768383,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / BothRev",
            "value": 0.000011102399000264995,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / PreRev",
            "value": 0.000011188004999894474,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / PostRev",
            "value": 0.000011073873999976056,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / BothRev",
            "value": 0.00001151576899974316,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / PreRev",
            "value": 0.000011279427999852488,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / PostRev",
            "value": 0.000011278052999841748,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / BothRev",
            "value": 0.000011090756000157852,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / PreRev",
            "value": 0.000011318928000036976,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / PostRev",
            "value": 0.00001096954199965694,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / BothRev",
            "value": 0.000011311311000099524,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / PreRev",
            "value": 0.000011083933000008985,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / PostRev",
            "value": 0.000011619681999945896,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / BothRev",
            "value": 0.000011160321999795996,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Primal",
            "value": 0.00000964178599679144,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cpu / Primal",
            "value": 0.000009669505001511423,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / Primal",
            "value": 0.000009648072998970748,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / Primal",
            "value": 0.00000959651100129122,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / Primal",
            "value": 0.0000095424860046478,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / Primal",
            "value": 0.00000964630000089528,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / Primal",
            "value": 0.000009707689001515974,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Forward",
            "value": 0.00001442788599524647,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cpu / Forward",
            "value": 0.000014352345999213868,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / Forward",
            "value": 0.000015071609996084587,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / Forward",
            "value": 0.000015031748000183142,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / Forward",
            "value": 0.000014365398994414137,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / Forward",
            "value": 0.000015064611005072948,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / Forward",
            "value": 0.000014600037000491285,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PreRev",
            "value": 0.00001787818300363142,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PostRev",
            "value": 0.00001846978899993701,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / BothRev",
            "value": 0.00001785873000335414,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cpu / BothRev",
            "value": 0.000018594085995573552,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / PreRev",
            "value": 0.0000185852410068037,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / PostRev",
            "value": 0.000018557253999460955,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / BothRev",
            "value": 0.000017938727003638632,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / PreRev",
            "value": 0.000018522164995374625,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / PostRev",
            "value": 0.000017891028001031373,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / BothRev",
            "value": 0.000017653962997428607,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / PreRev",
            "value": 0.000018603092001285403,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / PostRev",
            "value": 0.00001853176800068468,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / BothRev",
            "value": 0.000018589391002024057,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / PreRev",
            "value": 0.000018648758996278047,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / PostRev",
            "value": 0.000018578328999865336,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / BothRev",
            "value": 0.000018645587006176355,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / PreRev",
            "value": 0.000018482571002095934,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / PostRev",
            "value": 0.000018480252998415386,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / BothRev",
            "value": 0.000018434751000313552,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / tpu / Primal",
            "value": 0.0001525822209987,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / tpu / Primal",
            "value": 0.0001534077509986,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / tpu / Primal",
            "value": 0.0001542491119998,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / tpu / Primal",
            "value": 0.0001531643710004,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / tpu / Primal",
            "value": 0.0001544958819995,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / tpu / Primal",
            "value": 0.0001554844919992,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / tpu / Primal",
            "value": 0.000151838149999,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / tpu / Forward",
            "value": 0.0002504646299985,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / tpu / Forward",
            "value": 0.0002435371970004,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / tpu / Forward",
            "value": 0.00024873535,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / tpu / Forward",
            "value": 0.0002480506089996,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / tpu / Forward",
            "value": 0.0002467738089999,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / tpu / Forward",
            "value": 0.0002335495639999,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / tpu / Forward",
            "value": 0.0002198433579997,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / tpu / PreRev",
            "value": 0.0002308511219998,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / tpu / PostRev",
            "value": 0.0002342235430005,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / tpu / BothRev",
            "value": 0.0002447260879998,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / tpu / BothRev",
            "value": 0.0002385280460002,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / tpu / PreRev",
            "value": 0.0002367383349992,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / tpu / PostRev",
            "value": 0.0002454406380002,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / tpu / BothRev",
            "value": 0.0002441061280005,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / tpu / PreRev",
            "value": 0.0002393819159988,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / tpu / PostRev",
            "value": 0.0002361446539998,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / tpu / BothRev",
            "value": 0.0002282226009992,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / tpu / PreRev",
            "value": 0.0002266357799999,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / tpu / PostRev",
            "value": 0.0002272251410013,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / tpu / BothRev",
            "value": 0.0002266951410001,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / tpu / PreRev",
            "value": 0.0002137402349999,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / tpu / PostRev",
            "value": 0.0002132021050001,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / tpu / BothRev",
            "value": 0.0002128950150017,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / tpu / PreRev",
            "value": 0.0002128168449999,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / tpu / PostRev",
            "value": 0.0002248886800007,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / tpu / BothRev",
            "value": 0.0002253823299997,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Primal",
            "value": 0.000006458832000134862,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cpu / Primal",
            "value": 0.00000662810699941474,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / Primal",
            "value": 0.000006585301000086474,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / Primal",
            "value": 0.000006942369999705988,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / Primal",
            "value": 0.000006588668000404141,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / Primal",
            "value": 0.000006597876999876462,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / Primal",
            "value": 0.00000697478699930798,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Forward",
            "value": 0.000010034763000476232,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cpu / Forward",
            "value": 0.00001036076100081118,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / Forward",
            "value": 0.00001014588599991839,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / Forward",
            "value": 0.000010393685999588342,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / Forward",
            "value": 0.000010387540000010631,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / Forward",
            "value": 0.00001030844500019157,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / Forward",
            "value": 0.000010364808999838715,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PreRev",
            "value": 0.000012307111000154691,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PostRev",
            "value": 0.000012882771000477078,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / BothRev",
            "value": 0.000012910808999549771,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cpu / BothRev",
            "value": 0.000012923064999995403,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / PreRev",
            "value": 0.000012806787999579683,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / PostRev",
            "value": 0.000012897628000246188,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / BothRev",
            "value": 0.000012876167000285931,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / PreRev",
            "value": 0.000012763746999553403,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / PostRev",
            "value": 0.000012848405999648094,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / BothRev",
            "value": 0.000012375539999993635,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / PreRev",
            "value": 0.000012781230999280524,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / PostRev",
            "value": 0.000012769129999469442,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / BothRev",
            "value": 0.000012919410999529646,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / PreRev",
            "value": 0.000012751567000123031,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / PostRev",
            "value": 0.000012811201999284094,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / BothRev",
            "value": 0.000012891970000055151,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / PreRev",
            "value": 0.00001285526999981812,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / PostRev",
            "value": 0.000012956782999935967,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / BothRev",
            "value": 0.000012325519000114584,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Primal",
            "value": 0.000004206834000797244,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cpu / Primal",
            "value": 0.000004312959001254057,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / Primal",
            "value": 0.000005046750000474276,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / Primal",
            "value": 0.000004279667000446352,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / Primal",
            "value": 0.000004186166999716079,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / Primal",
            "value": 0.000004166374999840628,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / Primal",
            "value": 0.00000428499999907217,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / Forward",
            "value": 0.000006441999999879045,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cpu / Forward",
            "value": 0.000006491624999398482,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / Forward",
            "value": 0.000006614082998567028,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / Forward",
            "value": 0.000006688665998808574,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / Forward",
            "value": 0.000006585875000382657,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / Forward",
            "value": 0.000006524499998704414,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / Forward",
            "value": 0.000006524291999085108,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PreRev",
            "value": 0.000008696666000105325,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / PostRev",
            "value": 0.000008635541998955887,
            "unit": "s"
          },
          {
            "name": "add_two / JaXPipe / cpu / BothRev",
            "value": 0.00000878312500026368,
            "unit": "s"
          },
          {
            "name": "add_two / Jax / cpu / BothRev",
            "value": 0.000008800542000244604,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / PreRev",
            "value": 0.000008618957999715348,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / PostRev",
            "value": 0.000008579999999710709,
            "unit": "s"
          },
          {
            "name": "add_two / HLOOpt / cpu / BothRev",
            "value": 0.000008723332999579725,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / PreRev",
            "value": 0.000008800875000815722,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / PostRev",
            "value": 0.000008685624999998254,
            "unit": "s"
          },
          {
            "name": "add_two / PartOpt / cpu / BothRev",
            "value": 0.000008748207999815349,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / PreRev",
            "value": 0.00000868695899953309,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / PostRev",
            "value": 0.000008619750000434579,
            "unit": "s"
          },
          {
            "name": "add_two / IPartOpt / cpu / BothRev",
            "value": 0.000008580374998928164,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / PreRev",
            "value": 0.000008736582998608356,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / PostRev",
            "value": 0.000008551625000109197,
            "unit": "s"
          },
          {
            "name": "add_two / DefOpt / cpu / BothRev",
            "value": 0.000008628708001197082,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / PreRev",
            "value": 0.000008705707999979496,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / PostRev",
            "value": 0.000008542875000784989,
            "unit": "s"
          },
          {
            "name": "add_two / IDefOpt / cpu / BothRev",
            "value": 0.000008386250001422014,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Primal",
            "value": 0.0000046566700002586,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cpu / Primal",
            "value": 0.000004548171000351431,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / Primal",
            "value": 0.000004658595000364585,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / Primal",
            "value": 0.00000468954500001928,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / Primal",
            "value": 0.00000464516600004572,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / Primal",
            "value": 0.000004957243999797356,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / Primal",
            "value": 0.000004634687999896414,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Forward",
            "value": 0.00001156886399985524,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cpu / Forward",
            "value": 0.000011117088999981208,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / Forward",
            "value": 0.00001109808899991549,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / Forward",
            "value": 0.00001150144299981548,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / Forward",
            "value": 0.000011141487999793751,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / Forward",
            "value": 0.000011596328999985416,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / Forward",
            "value": 0.000011129695999898104,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PreRev",
            "value": 0.000012841036000281748,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PostRev",
            "value": 0.000016829180000058842,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / BothRev",
            "value": 0.000012104687999908492,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cpu / BothRev",
            "value": 0.000015442785000232107,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / PreRev",
            "value": 0.000011550850000276114,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / PostRev",
            "value": 0.00001206515199964997,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / BothRev",
            "value": 0.000012661261000175727,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / PreRev",
            "value": 0.000012198463999993691,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / PostRev",
            "value": 0.000015523806999681256,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / BothRev",
            "value": 0.000012181975999737917,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / PreRev",
            "value": 0.000011765176000153587,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / PostRev",
            "value": 0.000015454593999947975,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / BothRev",
            "value": 0.000011600466999880157,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / PreRev",
            "value": 0.000012765000999934274,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / PostRev",
            "value": 0.00001311665000002904,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / BothRev",
            "value": 0.00001233391899995695,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / PreRev",
            "value": 0.000012295361000269622,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / PostRev",
            "value": 0.000012204125000153,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / BothRev",
            "value": 0.00001344314099969779,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Primal",
            "value": 0.000009457530999497976,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cpu / Primal",
            "value": 0.000009676494999439457,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / Primal",
            "value": 0.000009651905995269771,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / Primal",
            "value": 0.000009445731004234404,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / Primal",
            "value": 0.000009451022000575904,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / Primal",
            "value": 0.000009523417000309564,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / Primal",
            "value": 0.000009486189999734052,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Forward",
            "value": 0.000014671140997961628,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cpu / Forward",
            "value": 0.00001535265299753519,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / Forward",
            "value": 0.000014747255998372568,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / Forward",
            "value": 0.000015381732002424543,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / Forward",
            "value": 0.000015122297998459544,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / Forward",
            "value": 0.000014775458002986853,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / Forward",
            "value": 0.000014599172995076514,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PreRev",
            "value": 0.000015133148001041263,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PostRev",
            "value": 0.000017910706003021913,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / BothRev",
            "value": 0.00001590059999580262,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cpu / BothRev",
            "value": 0.000017039935999491717,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / PreRev",
            "value": 0.000015728701000625734,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / PostRev",
            "value": 0.000015167135003139263,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / BothRev",
            "value": 0.00001510238999617286,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / PreRev",
            "value": 0.00001582898400374688,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / PostRev",
            "value": 0.000017202215996803714,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / BothRev",
            "value": 0.000015344769999501295,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / PreRev",
            "value": 0.000015811743003723676,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / PostRev",
            "value": 0.00001780656399932923,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / BothRev",
            "value": 0.000015531416996964252,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / PreRev",
            "value": 0.00001577116399857914,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / PostRev",
            "value": 0.000015927478998492006,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / BothRev",
            "value": 0.000015590256996802055,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / PreRev",
            "value": 0.0000161035470009665,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / PostRev",
            "value": 0.000015669349995732773,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / BothRev",
            "value": 0.00001600654700450832,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / tpu / Primal",
            "value": 0.000134886763999,
            "unit": "s"
          },
          {
            "name": "cache / Jax / tpu / Primal",
            "value": 0.0001377190959992,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / tpu / Primal",
            "value": 0.0001368361139993,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / tpu / Primal",
            "value": 0.0001365834450007,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / tpu / Primal",
            "value": 0.000137109235,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / tpu / Primal",
            "value": 0.0001461088880005,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / tpu / Primal",
            "value": 0.0001472160590001,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / tpu / Forward",
            "value": 0.0002165163860008,
            "unit": "s"
          },
          {
            "name": "cache / Jax / tpu / Forward",
            "value": 0.0001984446089991,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / tpu / Forward",
            "value": 0.0001985693000005,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / tpu / Forward",
            "value": 0.0002020028610004,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / tpu / Forward",
            "value": 0.0002006040199994,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / tpu / Forward",
            "value": 0.0002057402829996,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / tpu / Forward",
            "value": 0.0002170523160002,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / tpu / PreRev",
            "value": 0.0002169742869991,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / tpu / PostRev",
            "value": 0.000206002212999,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / tpu / BothRev",
            "value": 0.0002030569709986,
            "unit": "s"
          },
          {
            "name": "cache / Jax / tpu / BothRev",
            "value": 0.000206001773,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / tpu / PreRev",
            "value": 0.0002075739530009,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / tpu / PostRev",
            "value": 0.0002031595810003,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / tpu / BothRev",
            "value": 0.0002068712729997,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / tpu / PreRev",
            "value": 0.0002172638370011,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / tpu / PostRev",
            "value": 0.0002062434819999,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / tpu / BothRev",
            "value": 0.000197142298999,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / tpu / PreRev",
            "value": 0.0001973644990011,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / tpu / PostRev",
            "value": 0.000195885518,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / tpu / BothRev",
            "value": 0.0002376150449999,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / tpu / PreRev",
            "value": 0.0002387881560007,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / tpu / PostRev",
            "value": 0.0002605232840014,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / tpu / BothRev",
            "value": 0.0002405640460001,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / tpu / PreRev",
            "value": 0.0002469926880003,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / tpu / PostRev",
            "value": 0.0002475040690005,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / tpu / BothRev",
            "value": 0.0002245923600003,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Primal",
            "value": 0.000005911317999562016,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cpu / Primal",
            "value": 0.000005923910000092292,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / Primal",
            "value": 0.00000607500899968727,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / Primal",
            "value": 0.000005956465999588545,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / Primal",
            "value": 0.00000591720800002804,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / Primal",
            "value": 0.000005905112999244011,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / Primal",
            "value": 0.0000059090019994982864,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Forward",
            "value": 0.000010729290000199398,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cpu / Forward",
            "value": 0.000010502424000151224,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / Forward",
            "value": 0.000010637674000463448,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / Forward",
            "value": 0.000010839851999662642,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / Forward",
            "value": 0.000010838807999789425,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / Forward",
            "value": 0.000010335770000892808,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / Forward",
            "value": 0.000010267710000334772,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PreRev",
            "value": 0.000011028203999558172,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PostRev",
            "value": 0.000012356408999949051,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / BothRev",
            "value": 0.000010870387999602829,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cpu / BothRev",
            "value": 0.000012549731000035536,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / PreRev",
            "value": 0.00001057999899967399,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / PostRev",
            "value": 0.000010582740999780071,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / BothRev",
            "value": 0.000010907408999628388,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / PreRev",
            "value": 0.000011101888000666805,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / PostRev",
            "value": 0.00001166135299990856,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / BothRev",
            "value": 0.000010652052000295951,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / PreRev",
            "value": 0.000010798326000440283,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / PostRev",
            "value": 0.000012104037000426616,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / BothRev",
            "value": 0.00001058577999992849,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / PreRev",
            "value": 0.000010771225000098638,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / PostRev",
            "value": 0.00001111914600005548,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / BothRev",
            "value": 0.00001102933100082737,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / PreRev",
            "value": 0.000010910083000453596,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / PostRev",
            "value": 0.000010467670000252835,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / BothRev",
            "value": 0.00001075172999935603,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Primal",
            "value": 0.000003686000000016065,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cpu / Primal",
            "value": 0.0000037136249993636737,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / Primal",
            "value": 0.000003641374998551328,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / Primal",
            "value": 0.000003690958999868599,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / Primal",
            "value": 0.0000036822920010308735,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / Primal",
            "value": 0.000003939666999940528,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / Primal",
            "value": 0.000003656832999695325,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / Forward",
            "value": 0.000006602542000109679,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cpu / Forward",
            "value": 0.000006132624999736436,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / Forward",
            "value": 0.000006213750000824802,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / Forward",
            "value": 0.00000615004199971736,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / Forward",
            "value": 0.000006131374999313266,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / Forward",
            "value": 0.000006454250000388128,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / Forward",
            "value": 0.000006424125000194181,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PreRev",
            "value": 0.000006162792000395711,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / PostRev",
            "value": 0.00000763741700029641,
            "unit": "s"
          },
          {
            "name": "cache / JaXPipe / cpu / BothRev",
            "value": 0.000006211624999195919,
            "unit": "s"
          },
          {
            "name": "cache / Jax / cpu / BothRev",
            "value": 0.000007062458000291372,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / PreRev",
            "value": 0.0000063747499989403874,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / PostRev",
            "value": 0.000006328249999569379,
            "unit": "s"
          },
          {
            "name": "cache / HLOOpt / cpu / BothRev",
            "value": 0.000006766374999642722,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / PreRev",
            "value": 0.000006281749998379383,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / PostRev",
            "value": 0.000007175541000833618,
            "unit": "s"
          },
          {
            "name": "cache / PartOpt / cpu / BothRev",
            "value": 0.000006325499998638406,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / PreRev",
            "value": 0.000006371040999511024,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / PostRev",
            "value": 0.000007094083000993123,
            "unit": "s"
          },
          {
            "name": "cache / IPartOpt / cpu / BothRev",
            "value": 0.000006201833000886836,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / PreRev",
            "value": 0.000006845625001005828,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / PostRev",
            "value": 0.000006368958000166458,
            "unit": "s"
          },
          {
            "name": "cache / DefOpt / cpu / BothRev",
            "value": 0.000006294165999861434,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / PreRev",
            "value": 0.000006412791999537149,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / PostRev",
            "value": 0.00000626587500119058,
            "unit": "s"
          },
          {
            "name": "cache / IDefOpt / cpu / BothRev",
            "value": 0.000006199708001076943,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Primal",
            "value": 0.00000486288299998705,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cpu / Primal",
            "value": 0.0000048322190000362754,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / Primal",
            "value": 0.000004907874000309676,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / Primal",
            "value": 0.0000048489060000065366,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / Primal",
            "value": 0.000004845975000080216,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / Primal",
            "value": 0.0000049714649999259565,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / Primal",
            "value": 0.000004970851999587467,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Forward",
            "value": 0.00000819924400002492,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cpu / Forward",
            "value": 0.000008209894999708923,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / Forward",
            "value": 0.000008140704000197729,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / Forward",
            "value": 0.000007943755999804125,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / Forward",
            "value": 0.000007954940000217903,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / Forward",
            "value": 0.000008124807000058353,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / Forward",
            "value": 0.000008088245999715582,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PreRev",
            "value": 0.000008862555000177962,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PostRev",
            "value": 0.000008842391000143834,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / BothRev",
            "value": 0.000008877291999851878,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cpu / BothRev",
            "value": 0.000008826376999877539,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / PreRev",
            "value": 0.000008871793000253091,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / PostRev",
            "value": 0.000009181233000163048,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / BothRev",
            "value": 0.000008908187000088218,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / PreRev",
            "value": 0.000009063169000000923,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / PostRev",
            "value": 0.000008791713999926287,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / BothRev",
            "value": 0.000008843618000355492,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / PreRev",
            "value": 0.000008965721000095073,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / PostRev",
            "value": 0.000008786117999989073,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / BothRev",
            "value": 0.00000896110800022143,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / PreRev",
            "value": 0.000008777975000157312,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / PostRev",
            "value": 0.000008862686000156827,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / BothRev",
            "value": 0.000008818008000162081,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / PreRev",
            "value": 0.000008772631000283581,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / PostRev",
            "value": 0.000009448511000300642,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / BothRev",
            "value": 0.000008919203000004927,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Primal",
            "value": 0.00000945970400061924,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cpu / Primal",
            "value": 0.000009307849999458994,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / Primal",
            "value": 0.000009396424997248688,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / Primal",
            "value": 0.00000998832500044955,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / Primal",
            "value": 0.000009776995997526685,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / Primal",
            "value": 0.000009977660003642086,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / Primal",
            "value": 0.000009895050003251528,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Forward",
            "value": 0.000014883097996062131,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cpu / Forward",
            "value": 0.000014767066000786145,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / Forward",
            "value": 0.000014472050999756902,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / Forward",
            "value": 0.000013939341995865106,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / Forward",
            "value": 0.000014864541000861207,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / Forward",
            "value": 0.000014840095995168668,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / Forward",
            "value": 0.00001472211299551418,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PreRev",
            "value": 0.00001504482499876758,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PostRev",
            "value": 0.000016160721999767702,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / BothRev",
            "value": 0.000015953991998685525,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cpu / BothRev",
            "value": 0.00001588498699857155,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / PreRev",
            "value": 0.000016029500002332498,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / PostRev",
            "value": 0.000016066408999904523,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / BothRev",
            "value": 0.000016076248000899795,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / PreRev",
            "value": 0.000015370706998510287,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / PostRev",
            "value": 0.000016115538994199595,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / BothRev",
            "value": 0.000015452242005267182,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / PreRev",
            "value": 0.000016048561003117356,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / PostRev",
            "value": 0.000016022108000470326,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / BothRev",
            "value": 0.000016119109001010657,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / PreRev",
            "value": 0.000016181769999093376,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / PostRev",
            "value": 0.00001618484999926295,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / BothRev",
            "value": 0.000016076776999398134,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / PreRev",
            "value": 0.00001569527600076981,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / PostRev",
            "value": 0.000015953362999425736,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / BothRev",
            "value": 0.00001612820000445936,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / tpu / Primal",
            "value": 0.0001528312810005,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / tpu / Primal",
            "value": 0.00015187005,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / tpu / Primal",
            "value": 0.0001518113410002,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / tpu / Primal",
            "value": 0.0001520626510009,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / tpu / Primal",
            "value": 0.0001517021210001,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / tpu / Primal",
            "value": 0.0001517354300012,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / tpu / Primal",
            "value": 0.0001520448900009,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / tpu / Forward",
            "value": 0.0002079266529999,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / tpu / Forward",
            "value": 0.0002251778919999,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / tpu / Forward",
            "value": 0.0002444937800009,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / tpu / Forward",
            "value": 0.0002036923720006,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / tpu / Forward",
            "value": 0.0002082272740008,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / tpu / Forward",
            "value": 0.0002225242109998,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / tpu / Forward",
            "value": 0.0002350405360011,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / tpu / PreRev",
            "value": 0.0002428401699999,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / tpu / PostRev",
            "value": 0.0002336643950002,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / tpu / BothRev",
            "value": 0.0002412467889989,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / tpu / BothRev",
            "value": 0.0002379978070002,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / tpu / PreRev",
            "value": 0.0002429757900008,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / tpu / PostRev",
            "value": 0.0002474924720008,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / tpu / BothRev",
            "value": 0.0002271312020002,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / tpu / PreRev",
            "value": 0.000247042461,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / tpu / PostRev",
            "value": 0.0002189856189997,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / tpu / BothRev",
            "value": 0.0002371948580002,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / tpu / PreRev",
            "value": 0.0002398553880011,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / tpu / PostRev",
            "value": 0.0002016627409993,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / tpu / BothRev",
            "value": 0.0002009273509993,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / tpu / PreRev",
            "value": 0.000213218447001,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / tpu / PostRev",
            "value": 0.000211621774999,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / tpu / BothRev",
            "value": 0.0002084250240004,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / tpu / PreRev",
            "value": 0.0002312751450008,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / tpu / PostRev",
            "value": 0.0002155437380006,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / tpu / BothRev",
            "value": 0.0002148873669993,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Primal",
            "value": 0.000006216707999556092,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cpu / Primal",
            "value": 0.000006371348999891779,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / Primal",
            "value": 0.000006242720000045665,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / Primal",
            "value": 0.000006381512999723782,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / Primal",
            "value": 0.000006275490000007266,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / Primal",
            "value": 0.000006329953000204114,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / Primal",
            "value": 0.00000636813599976449,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Forward",
            "value": 0.000010143408999283566,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cpu / Forward",
            "value": 0.000009610659999452764,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / Forward",
            "value": 0.000010132776999853376,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / Forward",
            "value": 0.000009607285000129196,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / Forward",
            "value": 0.000010111595000125818,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / Forward",
            "value": 0.000010369065000304544,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / Forward",
            "value": 0.000010181466000176445,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PreRev",
            "value": 0.00001049895199957973,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PostRev",
            "value": 0.000010473999000168989,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / BothRev",
            "value": 0.000011042369999813672,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cpu / BothRev",
            "value": 0.000011053981999793904,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / PreRev",
            "value": 0.000011155952999615692,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / PostRev",
            "value": 0.00001104526299968711,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / BothRev",
            "value": 0.000011086260999945808,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / PreRev",
            "value": 0.000010966537000058452,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / PostRev",
            "value": 0.000010947368000415736,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / BothRev",
            "value": 0.000010463751999850502,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / PreRev",
            "value": 0.000011074904999986756,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / PostRev",
            "value": 0.000011153795000609536,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / BothRev",
            "value": 0.00001097814000058861,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / PreRev",
            "value": 0.000011264638999818998,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / PostRev",
            "value": 0.000011022939000213228,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / BothRev",
            "value": 0.000010954764999951294,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / PreRev",
            "value": 0.000011623680999946373,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / PostRev",
            "value": 0.000011607608999838705,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / BothRev",
            "value": 0.0000111161170007108,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Primal",
            "value": 0.0000038239160003286085,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cpu / Primal",
            "value": 0.000003820459000053233,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / Primal",
            "value": 0.00000400245799937693,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / Primal",
            "value": 0.0000040481659998476975,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / Primal",
            "value": 0.000004038875000333064,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / Primal",
            "value": 0.000004094542000530055,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / Primal",
            "value": 0.000004075666000062483,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / Forward",
            "value": 0.000006258833000174491,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cpu / Forward",
            "value": 0.00000615608299995074,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / Forward",
            "value": 0.000006217540998477489,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / Forward",
            "value": 0.000006121958000221639,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / Forward",
            "value": 0.000006192666000060854,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / Forward",
            "value": 0.000005911874999583233,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / Forward",
            "value": 0.000005931915999099147,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PreRev",
            "value": 0.0000069152089999988674,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / PostRev",
            "value": 0.000007383709000350791,
            "unit": "s"
          },
          {
            "name": "Concat / JaXPipe / cpu / BothRev",
            "value": 0.000007135999998354237,
            "unit": "s"
          },
          {
            "name": "Concat / Jax / cpu / BothRev",
            "value": 0.00000713499999983469,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / PreRev",
            "value": 0.00000715808299901255,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / PostRev",
            "value": 0.0000072034579989122,
            "unit": "s"
          },
          {
            "name": "Concat / HLOOpt / cpu / BothRev",
            "value": 0.000006971332999455626,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / PreRev",
            "value": 0.000007327290999455727,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / PostRev",
            "value": 0.000006785291001506266,
            "unit": "s"
          },
          {
            "name": "Concat / PartOpt / cpu / BothRev",
            "value": 0.000007169542001065565,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / PreRev",
            "value": 0.000007302584001081413,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / PostRev",
            "value": 0.000007093041998814442,
            "unit": "s"
          },
          {
            "name": "Concat / IPartOpt / cpu / BothRev",
            "value": 0.000007277667000380461,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / PreRev",
            "value": 0.00000724041700050293,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / PostRev",
            "value": 0.000006879583999761962,
            "unit": "s"
          },
          {
            "name": "Concat / DefOpt / cpu / BothRev",
            "value": 0.000006861750000098254,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / PreRev",
            "value": 0.00000691437499881431,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / PostRev",
            "value": 0.0000071608749985898615,
            "unit": "s"
          },
          {
            "name": "Concat / IDefOpt / cpu / BothRev",
            "value": 0.000007220707999294973,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Primal",
            "value": 0.000004597439000008308,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / cpu / Primal",
            "value": 0.000004531731000042783,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / Primal",
            "value": 0.000005138427000019874,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / Primal",
            "value": 0.000004841945000407577,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / Primal",
            "value": 0.000004792114999872865,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / Primal",
            "value": 0.000005121794999922713,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / Primal",
            "value": 0.000005142303999946307,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Forward",
            "value": 0.000008600110999850586,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / cpu / Forward",
            "value": 0.000007429637999848637,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / Forward",
            "value": 0.00000862341199990624,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / Forward",
            "value": 0.00000859519999994518,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / Forward",
            "value": 0.000008697445000052539,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / Forward",
            "value": 0.000008602902999882644,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / Forward",
            "value": 0.000008546046999981627,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Primal",
            "value": 0.000008856594002281781,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / cpu / Primal",
            "value": 0.000009111257000768092,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / Primal",
            "value": 0.000010167510001338086,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / Primal",
            "value": 0.00000946187999943504,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / Primal",
            "value": 0.000009130608996201772,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / Primal",
            "value": 0.000010293049999745565,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / Primal",
            "value": 0.000010227626000414602,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Forward",
            "value": 0.0000156063769973116,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / cpu / Forward",
            "value": 0.00001277979999576928,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / Forward",
            "value": 0.000014944941001886036,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / Forward",
            "value": 0.00001507456000399543,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / Forward",
            "value": 0.000015660987999581266,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / Forward",
            "value": 0.000015685856000345665,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / Forward",
            "value": 0.000015636381001968403,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / tpu / Primal",
            "value": 0.0001345488409988,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / tpu / Primal",
            "value": 0.0001470386469991,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / tpu / Primal",
            "value": 0.0001467369759993,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / tpu / Primal",
            "value": 0.0001376325519995,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / tpu / Primal",
            "value": 0.0001330705500004,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / tpu / Primal",
            "value": 0.0001330513800003,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / tpu / Primal",
            "value": 0.0001404761639987,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / tpu / Forward",
            "value": 0.0002438789000007,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / tpu / Forward",
            "value": 0.0002236970110006,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / tpu / Forward",
            "value": 0.0002223638499999,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / tpu / Forward",
            "value": 0.0002228547409995,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / tpu / Forward",
            "value": 0.0001978943789999,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / tpu / Forward",
            "value": 0.0001985753499993,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / tpu / Forward",
            "value": 0.0001982554000005,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Primal",
            "value": 0.000005980113000077836,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / cpu / Primal",
            "value": 0.000005942977000813698,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / Primal",
            "value": 0.000006837895999524335,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / Primal",
            "value": 0.000006016498999997566,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / Primal",
            "value": 0.000006017169000188005,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / Primal",
            "value": 0.000006755164999958652,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / Primal",
            "value": 0.000006797218999963661,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Forward",
            "value": 0.000010647357999914676,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / cpu / Forward",
            "value": 0.000009156869999969786,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / Forward",
            "value": 0.000010160555999391365,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / Forward",
            "value": 0.000010569713999757367,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / Forward",
            "value": 0.000010542275000261724,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / Forward",
            "value": 0.000010592940000606177,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / Forward",
            "value": 0.0000105706160002228,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Primal",
            "value": 0.0000038095840009191306,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / cpu / Primal",
            "value": 0.00000377104099970893,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / Primal",
            "value": 0.000004220458000418148,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / Primal",
            "value": 0.0000037521670001297023,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / Primal",
            "value": 0.0000037502920004044422,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / Primal",
            "value": 0.000004275499999494059,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / Primal",
            "value": 0.000004174790999968536,
            "unit": "s"
          },
          {
            "name": "const_scatter / JaXPipe / cpu / Forward",
            "value": 0.000007165625000197906,
            "unit": "s"
          },
          {
            "name": "const_scatter / Jax / cpu / Forward",
            "value": 0.000005708208000214654,
            "unit": "s"
          },
          {
            "name": "const_scatter / HLOOpt / cpu / Forward",
            "value": 0.0000067479999997885895,
            "unit": "s"
          },
          {
            "name": "const_scatter / PartOpt / cpu / Forward",
            "value": 0.000006890708000355517,
            "unit": "s"
          },
          {
            "name": "const_scatter / IPartOpt / cpu / Forward",
            "value": 0.000006904582998686238,
            "unit": "s"
          },
          {
            "name": "const_scatter / DefOpt / cpu / Forward",
            "value": 0.000006907083999976749,
            "unit": "s"
          },
          {
            "name": "const_scatter / IDefOpt / cpu / Forward",
            "value": 0.000006831999999121763,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Primal",
            "value": 0.000005011890999867319,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cpu / Primal",
            "value": 0.000005166533000192431,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / Primal",
            "value": 0.000005288945000302192,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / Primal",
            "value": 0.000005028626000239455,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / Primal",
            "value": 0.000004987128000266239,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / Primal",
            "value": 0.000005248715999641718,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / Primal",
            "value": 0.000005286509000143269,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Forward",
            "value": 0.000008210337000036817,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cpu / Forward",
            "value": 0.00000730493099990781,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / Forward",
            "value": 0.000008165498000380466,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / Forward",
            "value": 0.000008105069000066578,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / Forward",
            "value": 0.000008287992000077792,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / Forward",
            "value": 0.00000831519399980607,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / Forward",
            "value": 0.000008161568000105035,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PreRev",
            "value": 0.000008122171000195521,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PostRev",
            "value": 0.000007745605000309297,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / BothRev",
            "value": 0.000008199434000289329,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cpu / BothRev",
            "value": 0.000007498262999888538,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / PreRev",
            "value": 0.00000819026900035169,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / PostRev",
            "value": 0.000008207824999772129,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / BothRev",
            "value": 0.000008190606999960437,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / PreRev",
            "value": 0.000008213663999868004,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / PostRev",
            "value": 0.000007504160999815212,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / BothRev",
            "value": 0.000008191373000045132,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / PreRev",
            "value": 0.000008230873000229621,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / PostRev",
            "value": 0.000007488383000236354,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / BothRev",
            "value": 0.000008305958999699214,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / PreRev",
            "value": 0.000008156644999871787,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / PostRev",
            "value": 0.000008244458999797644,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / BothRev",
            "value": 0.0000081188179997298,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / PreRev",
            "value": 0.000008297078999930818,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / PostRev",
            "value": 0.000008157763999861346,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / BothRev",
            "value": 0.000008207717999994201,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Primal",
            "value": 0.00000961393499892438,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cpu / Primal",
            "value": 0.000010343328001908958,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / Primal",
            "value": 0.000011805734000517989,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / Primal",
            "value": 0.000009739302993693857,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / Primal",
            "value": 0.000009669429004134144,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / Primal",
            "value": 0.000011662137003440876,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / Primal",
            "value": 0.00001162070000282256,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Forward",
            "value": 0.00001695579999795882,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cpu / Forward",
            "value": 0.000013744483003392816,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / Forward",
            "value": 0.00001610266299394425,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / Forward",
            "value": 0.000016779443998530043,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / Forward",
            "value": 0.000016891366998606826,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / Forward",
            "value": 0.000016359540997655132,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / Forward",
            "value": 0.000016962405999947804,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PreRev",
            "value": 0.00001614707599946996,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PostRev",
            "value": 0.000013950552995083854,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / BothRev",
            "value": 0.000016923597002460157,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cpu / BothRev",
            "value": 0.0000145871429995168,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / PreRev",
            "value": 0.000017204026000399608,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / PostRev",
            "value": 0.00001619463100360008,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / BothRev",
            "value": 0.000017011894997267518,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / PreRev",
            "value": 0.000017036663004546427,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / PostRev",
            "value": 0.00001386188700416824,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / BothRev",
            "value": 0.00001607557800161885,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / PreRev",
            "value": 0.000016819489996123594,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / PostRev",
            "value": 0.00001461962999746902,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / BothRev",
            "value": 0.00001619967899750918,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / PreRev",
            "value": 0.00001690489100292325,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / PostRev",
            "value": 0.00001679532299749553,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / BothRev",
            "value": 0.00001696387799893273,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / PreRev",
            "value": 0.00001694325599964941,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / PostRev",
            "value": 0.000016911145998165012,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / BothRev",
            "value": 0.000016103587004181463,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / tpu / Primal",
            "value": 0.0001323489900005,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / tpu / Primal",
            "value": 0.0001368885319989,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / tpu / Primal",
            "value": 0.0001325380999987,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / tpu / Primal",
            "value": 0.0001315874290012,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / tpu / Primal",
            "value": 0.0001432449849999,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / tpu / Primal",
            "value": 0.0001292056279999,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / tpu / Primal",
            "value": 0.0001288271189987,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / tpu / Forward",
            "value": 0.0001789216910001,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / tpu / Forward",
            "value": 0.0002046359530013,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / tpu / Forward",
            "value": 0.0002082619039993,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / tpu / Forward",
            "value": 0.0002256884620001,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / tpu / Forward",
            "value": 0.0002333745959986,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / tpu / Forward",
            "value": 0.0002283988630006,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / tpu / Forward",
            "value": 0.0002009763009991,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / tpu / PreRev",
            "value": 0.0002003224600011,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / tpu / PostRev",
            "value": 0.0001954325980004,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / tpu / BothRev",
            "value": 0.000200311120001,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / tpu / BothRev",
            "value": 0.0002015504210012,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / tpu / PreRev",
            "value": 0.0002025453320002,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / tpu / PostRev",
            "value": 0.0002007459710002,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / tpu / BothRev",
            "value": 0.0002028840319999,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / tpu / PreRev",
            "value": 0.0002132107060006,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / tpu / PostRev",
            "value": 0.0002205827199995,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / tpu / BothRev",
            "value": 0.0002251077119999,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / tpu / PreRev",
            "value": 0.0002298876530003,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / tpu / PostRev",
            "value": 0.0002482985919996,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / tpu / BothRev",
            "value": 0.000246756931001,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / tpu / PreRev",
            "value": 0.0002361662069997,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / tpu / PostRev",
            "value": 0.0002229678610001,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / tpu / BothRev",
            "value": 0.0002227768200009,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / tpu / PreRev",
            "value": 0.0002247622119994,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / tpu / PostRev",
            "value": 0.0002434588300002,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / tpu / BothRev",
            "value": 0.0002435189200004,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Primal",
            "value": 0.000006906708999849798,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cpu / Primal",
            "value": 0.000006365788000039174,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / Primal",
            "value": 0.000007699474000219197,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / Primal",
            "value": 0.000006543108999721881,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / Primal",
            "value": 0.000006874726999740233,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / Primal",
            "value": 0.000007289944000149262,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / Primal",
            "value": 0.000007324135999624559,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Forward",
            "value": 0.0000106186949997209,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cpu / Forward",
            "value": 0.000009876222999992024,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / Forward",
            "value": 0.000011196426000424251,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / Forward",
            "value": 0.000010712475999753224,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / Forward",
            "value": 0.000010597352999866415,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / Forward",
            "value": 0.000010712192000028154,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / Forward",
            "value": 0.000011104900000646013,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PreRev",
            "value": 0.000010683311000320827,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PostRev",
            "value": 0.000009276767000301334,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / BothRev",
            "value": 0.000011235026000576909,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cpu / BothRev",
            "value": 0.000009359095999570856,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / PreRev",
            "value": 0.000010705308000069635,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / PostRev",
            "value": 0.00001077754800007824,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / BothRev",
            "value": 0.000011254854000071646,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / PreRev",
            "value": 0.000011324973000228055,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / PostRev",
            "value": 0.000009490880999692307,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / BothRev",
            "value": 0.000010805339999933492,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / PreRev",
            "value": 0.00001072113700047339,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / PostRev",
            "value": 0.000010105866999765567,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / BothRev",
            "value": 0.000010714899999584305,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / PreRev",
            "value": 0.000011323340999297216,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / PostRev",
            "value": 0.000011192343999937292,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / BothRev",
            "value": 0.000010692292999920028,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / PreRev",
            "value": 0.000011198147999493811,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / PostRev",
            "value": 0.000011273432999587385,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / BothRev",
            "value": 0.000011208399999304677,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Primal",
            "value": 0.000004185125000731205,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cpu / Primal",
            "value": 0.000004169458999967901,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / Primal",
            "value": 0.000004454792000615271,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / Primal",
            "value": 0.000004155832999458653,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / Primal",
            "value": 0.000004168291001406032,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / Primal",
            "value": 0.000005484916999193956,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / Primal",
            "value": 0.000005067000000053667,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / Forward",
            "value": 0.000007717375001448091,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cpu / Forward",
            "value": 0.000007448459000443109,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / Forward",
            "value": 0.00000875320900013321,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / Forward",
            "value": 0.000007491166999898269,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / Forward",
            "value": 0.000009400082999491132,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / Forward",
            "value": 0.000007558374998552608,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / Forward",
            "value": 0.000007010792000073707,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PreRev",
            "value": 0.000006843834000392235,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / PostRev",
            "value": 0.000006309083000815008,
            "unit": "s"
          },
          {
            "name": "GenDot / JaXPipe / cpu / BothRev",
            "value": 0.000006775249999918743,
            "unit": "s"
          },
          {
            "name": "GenDot / Jax / cpu / BothRev",
            "value": 0.000006262333999984548,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / PreRev",
            "value": 0.000006872790998386335,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / PostRev",
            "value": 0.000007809417000316899,
            "unit": "s"
          },
          {
            "name": "GenDot / HLOOpt / cpu / BothRev",
            "value": 0.000007432874999722117,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / PreRev",
            "value": 0.00000787983299960615,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / PostRev",
            "value": 0.000006642165999437566,
            "unit": "s"
          },
          {
            "name": "GenDot / PartOpt / cpu / BothRev",
            "value": 0.00000723900000048161,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / PreRev",
            "value": 0.000006875583001601626,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / PostRev",
            "value": 0.000006536625000080676,
            "unit": "s"
          },
          {
            "name": "GenDot / IPartOpt / cpu / BothRev",
            "value": 0.000006884040998556884,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / PreRev",
            "value": 0.000006874042001072667,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / PostRev",
            "value": 0.000006917582999449223,
            "unit": "s"
          },
          {
            "name": "GenDot / DefOpt / cpu / BothRev",
            "value": 0.000006951250001293374,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / PreRev",
            "value": 0.000006855166999230278,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / PostRev",
            "value": 0.000007068082999467151,
            "unit": "s"
          },
          {
            "name": "GenDot / IDefOpt / cpu / BothRev",
            "value": 0.000006733416999850306,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Primal",
            "value": 0.000007428949000313878,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cpu / Primal",
            "value": 0.000007405338999888045,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / Primal",
            "value": 0.000007527732999733416,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / Primal",
            "value": 0.000007465042000148969,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / Primal",
            "value": 0.000007454411999788135,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / Primal",
            "value": 0.000007549870999810082,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / Primal",
            "value": 0.000007503577000079531,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Forward",
            "value": 0.000011602807000144822,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cpu / Forward",
            "value": 0.000011544922999746632,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / Forward",
            "value": 0.000011681484999826352,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / Forward",
            "value": 0.00001168180200011193,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / Forward",
            "value": 0.00001154889400004322,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / Forward",
            "value": 0.000011610968999775653,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / Forward",
            "value": 0.000011654040999928838,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PreRev",
            "value": 0.000011400740999761184,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PostRev",
            "value": 0.000011346887000399877,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / BothRev",
            "value": 0.000011272350000126608,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cpu / BothRev",
            "value": 0.000011194993000117392,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / PreRev",
            "value": 0.00001117953499988289,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / PostRev",
            "value": 0.000011363474000063434,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / BothRev",
            "value": 0.00001127226900007372,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / PreRev",
            "value": 0.000011238386000059108,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / PostRev",
            "value": 0.00001112544399984472,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / BothRev",
            "value": 0.000011224107000089134,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / PreRev",
            "value": 0.000011259292999966418,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / PostRev",
            "value": 0.000011218003000067256,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / BothRev",
            "value": 0.000011083260000305018,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / PreRev",
            "value": 0.000011243310999816458,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / PostRev",
            "value": 0.000011276018999978987,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / BothRev",
            "value": 0.000011145391999889398,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / PreRev",
            "value": 0.000011279384000317804,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / PostRev",
            "value": 0.000011288213999705476,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / BothRev",
            "value": 0.000011157374000049458,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Primal",
            "value": 0.00001415267799893627,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cpu / Primal",
            "value": 0.000013418479000392837,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / Primal",
            "value": 0.000014245809994463344,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / Primal",
            "value": 0.000014061638998100536,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / Primal",
            "value": 0.000014165034001052844,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / Primal",
            "value": 0.000014123799999651965,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / Primal",
            "value": 0.000014105991001997607,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Forward",
            "value": 0.00001938090899784584,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cpu / Forward",
            "value": 0.000020097942993743344,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / Forward",
            "value": 0.000019209731995943,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / Forward",
            "value": 0.00002022902200405952,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / Forward",
            "value": 0.00002010015099949669,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / Forward",
            "value": 0.00002015300300263334,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / Forward",
            "value": 0.000020002015997306445,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PreRev",
            "value": 0.0000187285560023156,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PostRev",
            "value": 0.000019606237998232245,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / BothRev",
            "value": 0.000019603266002377496,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cpu / BothRev",
            "value": 0.000019662701997731347,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / PreRev",
            "value": 0.000019789947997196575,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / PostRev",
            "value": 0.000019695917006174567,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / BothRev",
            "value": 0.0000196124249996501,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / PreRev",
            "value": 0.000018788864006637595,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / PostRev",
            "value": 0.00001965796700096689,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / BothRev",
            "value": 0.000018868336999730672,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / PreRev",
            "value": 0.000019768161997490097,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / PostRev",
            "value": 0.000019683771002746654,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / BothRev",
            "value": 0.000019873535995429848,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / PreRev",
            "value": 0.000019793469000433103,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / PostRev",
            "value": 0.00001879660199483624,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / BothRev",
            "value": 0.000019737710994377262,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / PreRev",
            "value": 0.000019977381998614874,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / PostRev",
            "value": 0.00001971480499923928,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / BothRev",
            "value": 0.000019729017003555783,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / tpu / Primal",
            "value": 0.0002442215139999,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / tpu / Primal",
            "value": 0.0002073285850001,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / tpu / Primal",
            "value": 0.0002332979420007,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / tpu / Primal",
            "value": 0.0002266543799996,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / tpu / Primal",
            "value": 0.0002185872770005,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / tpu / Primal",
            "value": 0.0001993032629998,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / tpu / Primal",
            "value": 0.0001990887520005,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / tpu / Forward",
            "value": 0.0002364518329995,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / tpu / Forward",
            "value": 0.0002273315700003,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / tpu / Forward",
            "value": 0.0002266229900014,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / tpu / Forward",
            "value": 0.0002391835230009,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / tpu / Forward",
            "value": 0.0002460162749994,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / tpu / Forward",
            "value": 0.0002487129250002,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / tpu / Forward",
            "value": 0.0002423401839987,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / tpu / PreRev",
            "value": 0.0002181597380003,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / tpu / PostRev",
            "value": 0.0002176216679999,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / tpu / BothRev",
            "value": 0.0002179741870004,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / tpu / BothRev",
            "value": 0.0002232374689992,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / tpu / PreRev",
            "value": 0.0002205246079993,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / tpu / PostRev",
            "value": 0.0002288550599987,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / tpu / BothRev",
            "value": 0.0002347852120001,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / tpu / PreRev",
            "value": 0.0002294027409989,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / tpu / PostRev",
            "value": 0.0002273099600006,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / tpu / BothRev",
            "value": 0.0002429863649995,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / tpu / PreRev",
            "value": 0.0002205648479994,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / tpu / PostRev",
            "value": 0.0002177327879999,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / tpu / BothRev",
            "value": 0.0002146239770008,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / tpu / PreRev",
            "value": 0.0002196491080012,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / tpu / PostRev",
            "value": 0.0002297660210006,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / tpu / BothRev",
            "value": 0.0002453069950006,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / tpu / PreRev",
            "value": 0.0002419553840009,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / tpu / PostRev",
            "value": 0.0002462249649997,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / tpu / BothRev",
            "value": 0.0002287222409995,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Primal",
            "value": 0.00000975823400040099,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cpu / Primal",
            "value": 0.000009246539999367088,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / Primal",
            "value": 0.000009705751999717904,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / Primal",
            "value": 0.000009702057000140484,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / Primal",
            "value": 0.000009744535999743676,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / Primal",
            "value": 0.000009808644000258936,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / Primal",
            "value": 0.000009414973999810172,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Forward",
            "value": 0.000014176691000102436,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cpu / Forward",
            "value": 0.00001452839900048275,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / Forward",
            "value": 0.000013812236000376287,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / Forward",
            "value": 0.0000136105490000773,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / Forward",
            "value": 0.000014329351999549544,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / Forward",
            "value": 0.000014284011000199826,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / Forward",
            "value": 0.000014223604999642702,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PreRev",
            "value": 0.000014103659999818776,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PostRev",
            "value": 0.000014009244999215298,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / BothRev",
            "value": 0.000014102924999860989,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cpu / BothRev",
            "value": 0.000014046123999833072,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / PreRev",
            "value": 0.000014043521000530743,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / PostRev",
            "value": 0.000014080537000154436,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / BothRev",
            "value": 0.000013927207000051568,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / PreRev",
            "value": 0.000013413215000582567,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / PostRev",
            "value": 0.000013900960999308151,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / BothRev",
            "value": 0.000013254618999781086,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / PreRev",
            "value": 0.000013993087000017113,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / PostRev",
            "value": 0.00001396506200035219,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / BothRev",
            "value": 0.000014073833000111337,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / PreRev",
            "value": 0.000013879756000278576,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / PostRev",
            "value": 0.000013995447000525018,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / BothRev",
            "value": 0.000013423971000520396,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / PreRev",
            "value": 0.00001412697500018112,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / PostRev",
            "value": 0.00001396205600030953,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / BothRev",
            "value": 0.000013972553999337834,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Primal",
            "value": 0.000006022249999659835,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cpu / Primal",
            "value": 0.000006295040999248158,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / Primal",
            "value": 0.000006338958000924322,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / Primal",
            "value": 0.000006083416999899782,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / Primal",
            "value": 0.000006048291999832145,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / Primal",
            "value": 0.000006079165999835823,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / Primal",
            "value": 0.000006008958000165876,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / Forward",
            "value": 0.000009544082999127569,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cpu / Forward",
            "value": 0.000009694540998680167,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / Forward",
            "value": 0.000009825292001551131,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / Forward",
            "value": 0.000009747165999215211,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / Forward",
            "value": 0.000009855624999545398,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / Forward",
            "value": 0.00000933591699867975,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / Forward",
            "value": 0.000009520207999230478,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PreRev",
            "value": 0.000009309124998253535,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / PostRev",
            "value": 0.000008993833000204176,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / JaXPipe / cpu / BothRev",
            "value": 0.00000887245800004166,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / Jax / cpu / BothRev",
            "value": 0.000008948124999733408,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / PreRev",
            "value": 0.000008935834001022158,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / PostRev",
            "value": 0.00000903654200010351,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / HLOOpt / cpu / BothRev",
            "value": 0.000009019208999234251,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / PreRev",
            "value": 0.00000901920800060907,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / PostRev",
            "value": 0.000008772334000241244,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / PartOpt / cpu / BothRev",
            "value": 0.000008611332999862498,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / PreRev",
            "value": 0.000008437458000116748,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / PostRev",
            "value": 0.000008847875000355998,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IPartOpt / cpu / BothRev",
            "value": 0.000008963124999354477,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / PreRev",
            "value": 0.00000899550000031013,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / PostRev",
            "value": 0.000008773208999627968,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / DefOpt / cpu / BothRev",
            "value": 0.000008554542000638321,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / PreRev",
            "value": 0.000008839334001095268,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / PostRev",
            "value": 0.0000089892920004786,
            "unit": "s"
          },
          {
            "name": "hlo_ffi / IDefOpt / cpu / BothRev",
            "value": 0.000008998999999676016,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / Primal",
            "value": 0.0008301930000016,
            "unit": "s"
          },
          {
            "name": "llama / Jax / cpu / Primal",
            "value": 0.0008396001999699,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / Primal",
            "value": 0.0009349190999728,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / Primal",
            "value": 0.0008332339999924,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / Primal",
            "value": 0.000833349100003,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / Primal",
            "value": 0.000927983199972,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / Primal",
            "value": 0.0009266445999855,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / Forward",
            "value": 0.0022908777999873,
            "unit": "s"
          },
          {
            "name": "llama / Jax / cpu / Forward",
            "value": 0.0023888103999979,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / Forward",
            "value": 0.0023478952999994,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / Forward",
            "value": 0.0023450167999726,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / Forward",
            "value": 0.0022886440000092,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / Forward",
            "value": 0.0023911587000384,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / Forward",
            "value": 0.0023326564999933,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / PreRev",
            "value": 0.0052613915000165,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / PostRev",
            "value": 0.0053096746999926,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / BothRev",
            "value": 0.0051814724999985,
            "unit": "s"
          },
          {
            "name": "llama / Jax / cpu / BothRev",
            "value": 0.0056051386000035,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / PreRev",
            "value": 0.0060124443000404,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / PostRev",
            "value": 0.0029351830999985,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / BothRev",
            "value": 0.0052969181000207,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / PreRev",
            "value": 0.0031653321999783,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / PostRev",
            "value": 0.0050992309999855,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / BothRev",
            "value": 0.0054519095999694,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / PreRev",
            "value": 0.0045169872999849,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / PostRev",
            "value": 0.0057094087999757,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / BothRev",
            "value": 0.003155013100013,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / PreRev",
            "value": 0.0049824032999822,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / PostRev",
            "value": 0.0028968894000172,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / BothRev",
            "value": 0.004910572899962,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / PreRev",
            "value": 0.0031832661999942,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / PostRev",
            "value": 0.0058955612000318,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / BothRev",
            "value": 0.003139651299989,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / Primal",
            "value": 0.0016186310000193,
            "unit": "s"
          },
          {
            "name": "llama / Jax / cpu / Primal",
            "value": 0.0015327278997574,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / Primal",
            "value": 0.0016502154998306,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / Primal",
            "value": 0.0015521724999416,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / Primal",
            "value": 0.0015427407997776,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / Primal",
            "value": 0.0016989935997116,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / Primal",
            "value": 0.0016562407996389,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / Forward",
            "value": 0.0052133226003206,
            "unit": "s"
          },
          {
            "name": "llama / Jax / cpu / Forward",
            "value": 0.0054862955999851,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / Forward",
            "value": 0.005185664199962,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / Forward",
            "value": 0.0053304854998714,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / Forward",
            "value": 0.0051792228005069,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / Forward",
            "value": 0.0052611582999816,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / Forward",
            "value": 0.0052877045003697,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / PreRev",
            "value": 0.0096790841002075,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / PostRev",
            "value": 0.011257173299964,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / BothRev",
            "value": 0.0112707217995193,
            "unit": "s"
          },
          {
            "name": "llama / Jax / cpu / BothRev",
            "value": 0.0108496273001946,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / PreRev",
            "value": 0.0072977233001438,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / PostRev",
            "value": 0.0108085396001115,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / BothRev",
            "value": 0.0071844017998955,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / PreRev",
            "value": 0.0107292442000471,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / PostRev",
            "value": 0.0108827258998644,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / BothRev",
            "value": 0.0070828402000188,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / PreRev",
            "value": 0.0107858440998825,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / PostRev",
            "value": 0.0072419625001202,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / BothRev",
            "value": 0.0106965727994975,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / PreRev",
            "value": 0.0072654495001188,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / PostRev",
            "value": 0.0106108751002466,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / BothRev",
            "value": 0.007378783400054,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / PreRev",
            "value": 0.0106722507996892,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / PostRev",
            "value": 0.0111399145003815,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / BothRev",
            "value": 0.007302930500009,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / tpu / Primal",
            "value": 0.0003889015800086,
            "unit": "s"
          },
          {
            "name": "llama / Jax / tpu / Primal",
            "value": 0.0003853865799828,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / tpu / Primal",
            "value": 0.0003798961799839,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / tpu / Primal",
            "value": 0.0003918923800301,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / tpu / Primal",
            "value": 0.0003955231800136,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / tpu / Primal",
            "value": 0.000352346360014,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / tpu / Primal",
            "value": 0.0003661041599843,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / tpu / Forward",
            "value": 0.0005780646600032,
            "unit": "s"
          },
          {
            "name": "llama / Jax / tpu / Forward",
            "value": 0.0007114805200035,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / tpu / Forward",
            "value": 0.000570532260026,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / tpu / Forward",
            "value": 0.0005579348599712,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / tpu / Forward",
            "value": 0.0005658250600026,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / tpu / Forward",
            "value": 0.0005586269600098,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / tpu / Forward",
            "value": 0.000561205360027,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / tpu / PreRev",
            "value": 0.0007869660000142,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / tpu / PostRev",
            "value": 0.0007460531999822,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / tpu / BothRev",
            "value": 0.0007767108000189,
            "unit": "s"
          },
          {
            "name": "llama / Jax / tpu / BothRev",
            "value": 0.0007373268000083,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / tpu / PreRev",
            "value": 0.0007752912199794,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / tpu / PostRev",
            "value": 0.0007801579999795,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / tpu / BothRev",
            "value": 0.0007749749999857,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / tpu / PreRev",
            "value": 0.0007798542000091,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / tpu / PostRev",
            "value": 0.0007679734200064,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / tpu / BothRev",
            "value": 0.0007834290000027,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / tpu / PreRev",
            "value": 0.0007860458000141,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / tpu / PostRev",
            "value": 0.0007600064000143,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / tpu / BothRev",
            "value": 0.0007313745800274,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / tpu / PreRev",
            "value": 0.0007497304000207,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / tpu / PostRev",
            "value": 0.0007243512000059,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / tpu / BothRev",
            "value": 0.0007763506199989,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / tpu / PreRev",
            "value": 0.0007170791799944,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / tpu / PostRev",
            "value": 0.0007303994000176,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / tpu / BothRev",
            "value": 0.0007322719999865,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / Primal",
            "value": 0.0015267564000168,
            "unit": "s"
          },
          {
            "name": "llama / Jax / cpu / Primal",
            "value": 0.0013895574999878,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / Primal",
            "value": 0.0015422288000081,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / Primal",
            "value": 0.0014557851000063,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / Primal",
            "value": 0.0013924815999416,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / Primal",
            "value": 0.0014990833999945,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / Primal",
            "value": 0.0014266404000409,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / Forward",
            "value": 0.0050228803000209,
            "unit": "s"
          },
          {
            "name": "llama / Jax / cpu / Forward",
            "value": 0.0051059278000138,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / Forward",
            "value": 0.0049982341000031,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / Forward",
            "value": 0.004998402000001,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / Forward",
            "value": 0.0048562576000222,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / Forward",
            "value": 0.0049223007999898,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / Forward",
            "value": 0.0050023032999888,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / PreRev",
            "value": 0.0091094575999704,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / PostRev",
            "value": 0.0087354000000232,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / BothRev",
            "value": 0.0094409475000247,
            "unit": "s"
          },
          {
            "name": "llama / Jax / cpu / BothRev",
            "value": 0.0087573236000025,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / PreRev",
            "value": 0.0096856832999947,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / PostRev",
            "value": 0.008449575800023,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / BothRev",
            "value": 0.0075834987000234,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / PreRev",
            "value": 0.0090893547999257,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / PostRev",
            "value": 0.0088419013999555,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / BothRev",
            "value": 0.0095759928999541,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / PreRev",
            "value": 0.0090396897999198,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / PostRev",
            "value": 0.008639606899942,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / BothRev",
            "value": 0.009085795699957,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / PreRev",
            "value": 0.0074535941999783,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / PostRev",
            "value": 0.0082595251999919,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / BothRev",
            "value": 0.0094788805999996,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / PreRev",
            "value": 0.0095538911999938,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / PostRev",
            "value": 0.009184834700045,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / BothRev",
            "value": 0.0094123896999917,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / Primal",
            "value": 0.0015354167000623,
            "unit": "s"
          },
          {
            "name": "llama / Jax / cpu / Primal",
            "value": 0.0015004833001512,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / Primal",
            "value": 0.0015765749998536,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / Primal",
            "value": 0.0014865249999274,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / Primal",
            "value": 0.0014220375000149,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / Primal",
            "value": 0.0015381291999801,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / Primal",
            "value": 0.001518812499853,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / Forward",
            "value": 0.0042337749999205,
            "unit": "s"
          },
          {
            "name": "llama / Jax / cpu / Forward",
            "value": 0.0042772917000547,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / Forward",
            "value": 0.0044781209000575,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / Forward",
            "value": 0.0044917500001247,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / Forward",
            "value": 0.0040756957998382,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / Forward",
            "value": 0.003789470799893,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / Forward",
            "value": 0.0041146417001073,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / PreRev",
            "value": 0.008257308300017,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / PostRev",
            "value": 0.009317291699881,
            "unit": "s"
          },
          {
            "name": "llama / JaXPipe / cpu / BothRev",
            "value": 0.0081741415999204,
            "unit": "s"
          },
          {
            "name": "llama / Jax / cpu / BothRev",
            "value": 0.0092948124998656,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / PreRev",
            "value": 0.0084889083000234,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / PostRev",
            "value": 0.0091740375000881,
            "unit": "s"
          },
          {
            "name": "llama / HLOOpt / cpu / BothRev",
            "value": 0.0109854875001474,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / PreRev",
            "value": 0.0143501374999686,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / PostRev",
            "value": 0.0099974000000656,
            "unit": "s"
          },
          {
            "name": "llama / PartOpt / cpu / BothRev",
            "value": 0.0086907124999925,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / PreRev",
            "value": 0.0076844082999741,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / PostRev",
            "value": 0.0103957666999122,
            "unit": "s"
          },
          {
            "name": "llama / IPartOpt / cpu / BothRev",
            "value": 0.007739279099951,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / PreRev",
            "value": 0.0083075624999764,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / PostRev",
            "value": 0.0079072832999372,
            "unit": "s"
          },
          {
            "name": "llama / DefOpt / cpu / BothRev",
            "value": 0.0072904250000647,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / PreRev",
            "value": 0.0076702042000761,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / PostRev",
            "value": 0.0075151167000512,
            "unit": "s"
          },
          {
            "name": "llama / IDefOpt / cpu / BothRev",
            "value": 0.0074549459000991,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000005617886999971233,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cpu / Primal",
            "value": 0.000005572513000061008,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / Primal",
            "value": 0.000005641848999857757,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / Primal",
            "value": 0.000005589857000359189,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / Primal",
            "value": 0.000005621768000310112,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / Primal",
            "value": 0.000005584738999914407,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / Primal",
            "value": 0.000005582369999956427,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Forward",
            "value": 0.000008768316999976378,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cpu / Forward",
            "value": 0.000008688411000093765,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / Forward",
            "value": 0.000008770829999775742,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / Forward",
            "value": 0.000008850454999901557,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / Forward",
            "value": 0.000008911997999803134,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / Forward",
            "value": 0.000008824661999824457,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / Forward",
            "value": 0.000008782051000252978,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / PreRev",
            "value": 0.000009265517000130785,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / PostRev",
            "value": 0.0000088538760001029,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / BothRev",
            "value": 0.000008972907999577729,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cpu / BothRev",
            "value": 0.000008956168000167963,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / PreRev",
            "value": 0.000008847494999827177,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / PostRev",
            "value": 0.000008944576999965647,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / BothRev",
            "value": 0.000008836948000407573,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / PreRev",
            "value": 0.000008827826000015194,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / PostRev",
            "value": 0.000008784321000348428,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / BothRev",
            "value": 0.000008768035999764833,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / PreRev",
            "value": 0.000008933233999869117,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / PostRev",
            "value": 0.00000888437999992675,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / BothRev",
            "value": 0.000008847883999806072,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / PreRev",
            "value": 0.000008985015000234852,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / PostRev",
            "value": 0.0000094145290004235,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / BothRev",
            "value": 0.000008851186000356392,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / PreRev",
            "value": 0.000008845553999890399,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / PostRev",
            "value": 0.000008947769999849698,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / BothRev",
            "value": 0.000008948210000198742,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000011055121001845693,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cpu / Primal",
            "value": 0.000010796606002259068,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / Primal",
            "value": 0.000010917557003267576,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / Primal",
            "value": 0.000010912509998888708,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / Primal",
            "value": 0.00001142874999641208,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / Primal",
            "value": 0.000011485726005048493,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / Primal",
            "value": 0.000011456478001491633,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Forward",
            "value": 0.000015767101001983974,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cpu / Forward",
            "value": 0.000016380676999688147,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / Forward",
            "value": 0.000016436887002782896,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / Forward",
            "value": 0.000016582062999077605,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / Forward",
            "value": 0.000015638816003047395,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / Forward",
            "value": 0.000016392713994719086,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / Forward",
            "value": 0.000016417883001849985,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / PreRev",
            "value": 0.000016600738999841268,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / PostRev",
            "value": 0.00001657029800117016,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / BothRev",
            "value": 0.000016663178997987415,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cpu / BothRev",
            "value": 0.000015983156001311728,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / PreRev",
            "value": 0.00001665493499604054,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / PostRev",
            "value": 0.000016699476000212598,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / BothRev",
            "value": 0.00001667605500551872,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / PreRev",
            "value": 0.000016576432004512755,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / PostRev",
            "value": 0.000015831384996999987,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / BothRev",
            "value": 0.000015866923000430688,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / PreRev",
            "value": 0.00001667840199661441,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / PostRev",
            "value": 0.00001597651200427208,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / BothRev",
            "value": 0.000016710502997739242,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / PreRev",
            "value": 0.00001665537100052461,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / PostRev",
            "value": 0.000016660091998346617,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / BothRev",
            "value": 0.00001586922299611615,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / PreRev",
            "value": 0.00001657762299873866,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / PostRev",
            "value": 0.00001658019799651811,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / BothRev",
            "value": 0.000016586947996984237,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Primal",
            "value": 0.0001496891170008,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / tpu / Primal",
            "value": 0.000151030889001,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / tpu / Primal",
            "value": 0.0001419944139997,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / tpu / Primal",
            "value": 0.000137387151999,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / tpu / Primal",
            "value": 0.0001541202190001,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / tpu / Primal",
            "value": 0.0001547067800001,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / tpu / Primal",
            "value": 0.000138114442001,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / Forward",
            "value": 0.0002200927789999,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / tpu / Forward",
            "value": 0.0002048637520001,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / tpu / Forward",
            "value": 0.000206683843,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / tpu / Forward",
            "value": 0.000227718733,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / tpu / Forward",
            "value": 0.0002247481119993,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / tpu / Forward",
            "value": 0.0002066479439999,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / tpu / Forward",
            "value": 0.0002066623629998,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / PreRev",
            "value": 0.0002195747389996,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / PostRev",
            "value": 0.0002315563350002,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / tpu / BothRev",
            "value": 0.0002318783850005,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / tpu / BothRev",
            "value": 0.000212942217,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / tpu / PreRev",
            "value": 0.0002056238029999,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / tpu / PostRev",
            "value": 0.0001982038490004,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / tpu / BothRev",
            "value": 0.000194070888001,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / tpu / PreRev",
            "value": 0.0002043077019989,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / tpu / PostRev",
            "value": 0.0002176870390012,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / tpu / BothRev",
            "value": 0.0002216516700009,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / tpu / PreRev",
            "value": 0.000216368198,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / tpu / PostRev",
            "value": 0.0002092680240002,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / tpu / BothRev",
            "value": 0.0002110514359992,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / tpu / PreRev",
            "value": 0.0002028323110007,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / tpu / PostRev",
            "value": 0.0002023521209994,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / tpu / BothRev",
            "value": 0.0002020292510005,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / tpu / PreRev",
            "value": 0.0002061494829995,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / tpu / PostRev",
            "value": 0.0002032081719989,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / tpu / BothRev",
            "value": 0.0002090726440001,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.000007232182999359793,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cpu / Primal",
            "value": 0.000007358322000072803,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / Primal",
            "value": 0.000007313860000067507,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / Primal",
            "value": 0.000007284090000212018,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / Primal",
            "value": 0.00000763533400004235,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / Primal",
            "value": 0.000007348162999733177,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / Primal",
            "value": 0.000007581845000458998,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Forward",
            "value": 0.000010604280999359617,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cpu / Forward",
            "value": 0.000011004124999999477,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / Forward",
            "value": 0.000010598859000310769,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / Forward",
            "value": 0.00001048910099962086,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / Forward",
            "value": 0.000011003811999216853,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / Forward",
            "value": 0.000011056704000111496,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / Forward",
            "value": 0.000010473233000084292,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / PreRev",
            "value": 0.00001124396799968963,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / PostRev",
            "value": 0.000011311127999761083,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / BothRev",
            "value": 0.000011238427000535013,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cpu / BothRev",
            "value": 0.00001122382099947572,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / PreRev",
            "value": 0.000010805074000018069,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / PostRev",
            "value": 0.000010733576000347966,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / BothRev",
            "value": 0.00001103344399962225,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / PreRev",
            "value": 0.000011206029999812016,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / PostRev",
            "value": 0.000011224873000173827,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / BothRev",
            "value": 0.000010674376999304514,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / PreRev",
            "value": 0.000010777792000226327,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / PostRev",
            "value": 0.000010684110000511282,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / BothRev",
            "value": 0.000011198561000128391,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / PreRev",
            "value": 0.000011209553999833588,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / PostRev",
            "value": 0.0000112285069999416,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / BothRev",
            "value": 0.00001121626600070158,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / PreRev",
            "value": 0.000010672772999896552,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / PostRev",
            "value": 0.000011271560000750467,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / BothRev",
            "value": 0.000011118559999886202,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Primal",
            "value": 0.0000047895420011627725,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cpu / Primal",
            "value": 0.000004722625000795233,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / Primal",
            "value": 0.000004619583000021521,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / Primal",
            "value": 0.000004734042000563932,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / Primal",
            "value": 0.000004673625000577885,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / Primal",
            "value": 0.000004729790998680983,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / Primal",
            "value": 0.00000492458299959253,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / Forward",
            "value": 0.00000732308399892645,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cpu / Forward",
            "value": 0.000007524000000557862,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / Forward",
            "value": 0.000007282750000740634,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / Forward",
            "value": 0.000007377334000921109,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / Forward",
            "value": 0.000007335375001275679,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / Forward",
            "value": 0.000007094209000570118,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / Forward",
            "value": 0.000007346042000790476,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / PreRev",
            "value": 0.000007311624998692423,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / PostRev",
            "value": 0.000007333375000598607,
            "unit": "s"
          },
          {
            "name": "scatter_sum / JaXPipe / cpu / BothRev",
            "value": 0.000007372291998763103,
            "unit": "s"
          },
          {
            "name": "scatter_sum / Jax / cpu / BothRev",
            "value": 0.000007437124999341904,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / PreRev",
            "value": 0.000007257916000526166,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / PostRev",
            "value": 0.000006976457998462138,
            "unit": "s"
          },
          {
            "name": "scatter_sum / HLOOpt / cpu / BothRev",
            "value": 0.000007437000000209082,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / PreRev",
            "value": 0.000007471959001122741,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / PostRev",
            "value": 0.000007663667000088026,
            "unit": "s"
          },
          {
            "name": "scatter_sum / PartOpt / cpu / BothRev",
            "value": 0.000007461582999894744,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / PreRev",
            "value": 0.00000727237499995681,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / PostRev",
            "value": 0.000006874250000691973,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IPartOpt / cpu / BothRev",
            "value": 0.000007325042000957183,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / PreRev",
            "value": 0.000007271624999702908,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / PostRev",
            "value": 0.000007325042000957183,
            "unit": "s"
          },
          {
            "name": "scatter_sum / DefOpt / cpu / BothRev",
            "value": 0.000007359666999036562,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / PreRev",
            "value": 0.000007068082999467151,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / PostRev",
            "value": 0.000007145292000132031,
            "unit": "s"
          },
          {
            "name": "scatter_sum / IDefOpt / cpu / BothRev",
            "value": 0.000007309124999665073,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Primal",
            "value": 0.000004534922999937408,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cpu / Primal",
            "value": 0.000004582330000175716,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / Primal",
            "value": 0.000004435148999618832,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / Primal",
            "value": 0.000004504521999933786,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / Primal",
            "value": 0.000004493626000112272,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / Primal",
            "value": 0.0000044932579999112936,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / Primal",
            "value": 0.000004576984999857814,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Forward",
            "value": 0.000006978657000217936,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cpu / Forward",
            "value": 0.0000069553420003103385,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / Forward",
            "value": 0.000006924555000296096,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / Forward",
            "value": 0.000006904002999817749,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / Forward",
            "value": 0.000006803168999795162,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / Forward",
            "value": 0.000006953945000077511,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / Forward",
            "value": 0.000006980899999689427,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PreRev",
            "value": 0.000007343867999679788,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PostRev",
            "value": 0.0000073795789999167025,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / BothRev",
            "value": 0.000007454518000031385,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cpu / BothRev",
            "value": 0.000007522105999669293,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / PreRev",
            "value": 0.000007391990999622067,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / PostRev",
            "value": 0.000007475559000340581,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / BothRev",
            "value": 0.000007352566000008664,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / PreRev",
            "value": 0.000007396785999844724,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / PostRev",
            "value": 0.0000074160859999210515,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / BothRev",
            "value": 0.000007393174999833718,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / PreRev",
            "value": 0.000007322781999846484,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / PostRev",
            "value": 0.000007379759000286868,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / BothRev",
            "value": 0.000007467962999726296,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / PreRev",
            "value": 0.0000074878239997815396,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / PostRev",
            "value": 0.00000741993800011187,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / BothRev",
            "value": 0.000007505145999857632,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / PreRev",
            "value": 0.000007403984000120545,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / PostRev",
            "value": 0.000007841310000003432,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / BothRev",
            "value": 0.000007537210000009509,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Primal",
            "value": 0.000009369573002913966,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cpu / Primal",
            "value": 0.000009311391004303004,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / Primal",
            "value": 0.000009239291000994854,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / Primal",
            "value": 0.000008917672996176407,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / Primal",
            "value": 0.000009332460998848546,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / Primal",
            "value": 0.000008826258999761194,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / Primal",
            "value": 0.000008812710002530365,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Forward",
            "value": 0.000012420058003044689,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cpu / Forward",
            "value": 0.00001327443499758374,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / Forward",
            "value": 0.000013262727996334434,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / Forward",
            "value": 0.000013379698000790086,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / Forward",
            "value": 0.000012624533999769482,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / Forward",
            "value": 0.000012475591000111309,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / Forward",
            "value": 0.000012663039000472054,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PreRev",
            "value": 0.000013515582999389154,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PostRev",
            "value": 0.000013329702996998094,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / BothRev",
            "value": 0.00001410206400032621,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cpu / BothRev",
            "value": 0.0000142891720024636,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / PreRev",
            "value": 0.00001427458100079093,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / PostRev",
            "value": 0.00001342865000333404,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / BothRev",
            "value": 0.000014153145995805971,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / PreRev",
            "value": 0.000014126047994068358,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / PostRev",
            "value": 0.000013496642997779418,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / BothRev",
            "value": 0.000013507289004337509,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / PreRev",
            "value": 0.000014196993994119111,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / PostRev",
            "value": 0.000014300522998382803,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / BothRev",
            "value": 0.000014230514003429562,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / PreRev",
            "value": 0.000014245918995584362,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / PostRev",
            "value": 0.000013478916000167371,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / BothRev",
            "value": 0.000013549079005315436,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / PreRev",
            "value": 0.00001420098800008418,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / PostRev",
            "value": 0.000014225966006051748,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / BothRev",
            "value": 0.000014133919001324105,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / tpu / Primal",
            "value": 0.0001365394309996,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / tpu / Primal",
            "value": 0.0001342905709989,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / tpu / Primal",
            "value": 0.0001408465129989,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / tpu / Primal",
            "value": 0.0001384639419993,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / tpu / Primal",
            "value": 0.0001529954189991,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / tpu / Primal",
            "value": 0.0001356389109987,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / tpu / Primal",
            "value": 0.0001342021910004,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / tpu / Forward",
            "value": 0.0002174511680004,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / tpu / Forward",
            "value": 0.0002098441750003,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / tpu / Forward",
            "value": 0.0002114773860012,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / tpu / Forward",
            "value": 0.0002085355139988,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / tpu / Forward",
            "value": 0.0002143343369989,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / tpu / Forward",
            "value": 0.0002163496369994,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / tpu / Forward",
            "value": 0.0002175106080012,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / tpu / PreRev",
            "value": 0.000199602319999,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / tpu / PostRev",
            "value": 0.0002015757209992,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / tpu / BothRev",
            "value": 0.0002194884589989,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / tpu / BothRev",
            "value": 0.0002248490010006,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / tpu / PreRev",
            "value": 0.0002079142839993,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / tpu / PostRev",
            "value": 0.0002206545089993,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / tpu / BothRev",
            "value": 0.0002039011630004,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / tpu / PreRev",
            "value": 0.0002158328169989,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / tpu / PostRev",
            "value": 0.0002208321589987,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / tpu / BothRev",
            "value": 0.0002421199290001,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / tpu / PreRev",
            "value": 0.0002332856649991,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / tpu / PostRev",
            "value": 0.0002198663690014,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / tpu / BothRev",
            "value": 0.0002269069129997,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / tpu / PreRev",
            "value": 0.0002360376670003,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / tpu / PostRev",
            "value": 0.0002354959660005,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / tpu / BothRev",
            "value": 0.0002262845719997,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / tpu / PreRev",
            "value": 0.0001987400399993,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / tpu / PostRev",
            "value": 0.0002009907499996,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / tpu / BothRev",
            "value": 0.0002078916339996,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Primal",
            "value": 0.000005890357000680524,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cpu / Primal",
            "value": 0.000005829118999827188,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / Primal",
            "value": 0.00000576082600036898,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / Primal",
            "value": 0.000005823606000376458,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / Primal",
            "value": 0.000006198346000019228,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / Primal",
            "value": 0.000005864992000169877,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / Primal",
            "value": 0.000005770355999629828,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Forward",
            "value": 0.000008444109999800275,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cpu / Forward",
            "value": 0.000009107103000133063,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / Forward",
            "value": 0.000008490817000165407,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / Forward",
            "value": 0.000009091674000046624,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / Forward",
            "value": 0.000008541570000488719,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / Forward",
            "value": 0.000009079600000404752,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / Forward",
            "value": 0.00000852527699953498,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PreRev",
            "value": 0.000009590773999661904,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PostRev",
            "value": 0.000009068930000466936,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / BothRev",
            "value": 0.000009189563999825622,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cpu / BothRev",
            "value": 0.000009642357000302582,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / PreRev",
            "value": 0.000009697851999590058,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / PostRev",
            "value": 0.00000914151500001026,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / BothRev",
            "value": 0.000009641254000598564,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / PreRev",
            "value": 0.0000096851019998212,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / PostRev",
            "value": 0.000009627416000512312,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / BothRev",
            "value": 0.000009153906999927133,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / PreRev",
            "value": 0.000009730464999847756,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / PostRev",
            "value": 0.000009685721000096237,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / BothRev",
            "value": 0.000009212793000187958,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / PreRev",
            "value": 0.000009704355999929248,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / PostRev",
            "value": 0.000009188512000037008,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / BothRev",
            "value": 0.000009657051000431238,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / PreRev",
            "value": 0.000009754949999660312,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / PostRev",
            "value": 0.000009641624999858325,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / BothRev",
            "value": 0.0000096308570000474,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Primal",
            "value": 0.000003692249998493935,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cpu / Primal",
            "value": 0.000003722458000993356,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / Primal",
            "value": 0.000003723874999195687,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / Primal",
            "value": 0.000003767500000321889,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / Primal",
            "value": 0.0000037962909991620103,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / Primal",
            "value": 0.000003737291999641456,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / Primal",
            "value": 0.00000370854200082249,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / Forward",
            "value": 0.000005409417000919348,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cpu / Forward",
            "value": 0.000005670458998793038,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / Forward",
            "value": 0.0000054042080009821805,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / Forward",
            "value": 0.000005713500000638305,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / Forward",
            "value": 0.000005784707998827798,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / Forward",
            "value": 0.000005611915999907069,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / Forward",
            "value": 0.000005736167000577552,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PreRev",
            "value": 0.000005939958000453771,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / PostRev",
            "value": 0.000005996166000841186,
            "unit": "s"
          },
          {
            "name": "slicing / JaXPipe / cpu / BothRev",
            "value": 0.000006063041999368579,
            "unit": "s"
          },
          {
            "name": "slicing / Jax / cpu / BothRev",
            "value": 0.000006083708000005572,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / PreRev",
            "value": 0.000005831833001138876,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / PostRev",
            "value": 0.000005888833999051713,
            "unit": "s"
          },
          {
            "name": "slicing / HLOOpt / cpu / BothRev",
            "value": 0.000006104000000050291,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / PreRev",
            "value": 0.000006069582999771228,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / PostRev",
            "value": 0.000006075125000279513,
            "unit": "s"
          },
          {
            "name": "slicing / PartOpt / cpu / BothRev",
            "value": 0.000006189249999806634,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / PreRev",
            "value": 0.000006168208001327002,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / PostRev",
            "value": 0.00000611837500036927,
            "unit": "s"
          },
          {
            "name": "slicing / IPartOpt / cpu / BothRev",
            "value": 0.000005762416998550179,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / PreRev",
            "value": 0.00000579670899969642,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / PostRev",
            "value": 0.000006155750001198612,
            "unit": "s"
          },
          {
            "name": "slicing / DefOpt / cpu / BothRev",
            "value": 0.000006105208000008133,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / PreRev",
            "value": 0.000006129875000624452,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / PostRev",
            "value": 0.0000064033329999801934,
            "unit": "s"
          },
          {
            "name": "slicing / IDefOpt / cpu / BothRev",
            "value": 0.00000614770899846917,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Primal",
            "value": 0.000005720496999856551,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cpu / Primal",
            "value": 0.000005701503000182128,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / Primal",
            "value": 0.000005983299000035913,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / Primal",
            "value": 0.00000599265499977264,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / Primal",
            "value": 0.000005799412000214943,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / Primal",
            "value": 0.000006008515999837982,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / Primal",
            "value": 0.00000573740599975281,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Forward",
            "value": 0.000009506568999768206,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cpu / Forward",
            "value": 0.00000949845400009508,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / Forward",
            "value": 0.000009498550999978762,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / Forward",
            "value": 0.000009561013000165986,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / Forward",
            "value": 0.000009394960999998147,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / Forward",
            "value": 0.000009409667999989324,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / Forward",
            "value": 0.00000950519299976804,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PreRev",
            "value": 0.00000869725099983043,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PostRev",
            "value": 0.000008199228999728802,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / BothRev",
            "value": 0.000008311327999763308,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cpu / BothRev",
            "value": 0.000008391197000037209,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / PreRev",
            "value": 0.000008274111999980959,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / PostRev",
            "value": 0.00000827149399992777,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / BothRev",
            "value": 0.000008329853999839542,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / PreRev",
            "value": 0.000008301955999741039,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / PostRev",
            "value": 0.000008325963000061164,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / BothRev",
            "value": 0.000008264401999895199,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / PreRev",
            "value": 0.000008207994000258623,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / PostRev",
            "value": 0.000008317935999912151,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / BothRev",
            "value": 0.00000829877900014253,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / PreRev",
            "value": 0.000008268270000371557,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / PostRev",
            "value": 0.000008232547000261548,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / BothRev",
            "value": 0.000008210552000036842,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / PreRev",
            "value": 0.000008257604999926116,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / PostRev",
            "value": 0.000008696194000094692,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / BothRev",
            "value": 0.000008151829999860639,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Primal",
            "value": 0.000012414166005328295,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cpu / Primal",
            "value": 0.000012443851002899464,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / Primal",
            "value": 0.0000124323589989217,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / Primal",
            "value": 0.000012368001000140794,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / Primal",
            "value": 0.000011833796997962054,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / Primal",
            "value": 0.000012434726995707024,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / Primal",
            "value": 0.000012298666006245184,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Forward",
            "value": 0.000018242197002109605,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cpu / Forward",
            "value": 0.00001819759099453222,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / Forward",
            "value": 0.00001810496699908981,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / Forward",
            "value": 0.000018226485997729468,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / Forward",
            "value": 0.000017329362002783455,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / Forward",
            "value": 0.00001811429100052919,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / Forward",
            "value": 0.00001814786200702656,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PreRev",
            "value": 0.000016912851002416573,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PostRev",
            "value": 0.000016074460996605922,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / BothRev",
            "value": 0.000016937262000283225,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cpu / BothRev",
            "value": 0.00001598451699828729,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / PreRev",
            "value": 0.0000168901810029638,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / PostRev",
            "value": 0.000016852458997163923,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / BothRev",
            "value": 0.00001683100099762669,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / PreRev",
            "value": 0.000016733661002945156,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / PostRev",
            "value": 0.00001692635300423717,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / BothRev",
            "value": 0.000016017850997741333,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / PreRev",
            "value": 0.000016904090996831655,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / PostRev",
            "value": 0.000016904782998608424,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / BothRev",
            "value": 0.000016791204994660802,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / PreRev",
            "value": 0.00001691569199465448,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / PostRev",
            "value": 0.000016869182996742892,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / BothRev",
            "value": 0.000016133655000885483,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / PreRev",
            "value": 0.000016171123999811245,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / PostRev",
            "value": 0.00001683802499610465,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / BothRev",
            "value": 0.00001690554000379052,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / tpu / Primal",
            "value": 0.0001356846420003,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / tpu / Primal",
            "value": 0.0001361668019999,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / tpu / Primal",
            "value": 0.0001398074330008,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / tpu / Primal",
            "value": 0.0001401831429993,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / tpu / Primal",
            "value": 0.0001392078030003,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / tpu / Primal",
            "value": 0.0001378801629998,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / tpu / Primal",
            "value": 0.0001449143959998,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / tpu / Forward",
            "value": 0.0002209754100003,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / tpu / Forward",
            "value": 0.0002277406329994,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / tpu / Forward",
            "value": 0.0002162596769994,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / tpu / Forward",
            "value": 0.0002329072650009,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / tpu / Forward",
            "value": 0.0002291295840004,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / tpu / Forward",
            "value": 0.0002251919520003,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / tpu / Forward",
            "value": 0.0002283540530006,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / tpu / PreRev",
            "value": 0.0002187237190009,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / tpu / PostRev",
            "value": 0.0002094468549985,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / tpu / BothRev",
            "value": 0.0002046716830009,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / tpu / BothRev",
            "value": 0.0002184236890007,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / tpu / PreRev",
            "value": 0.0002252358320001,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / tpu / PostRev",
            "value": 0.0002096297839998,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / tpu / BothRev",
            "value": 0.0002068668539995,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / tpu / PreRev",
            "value": 0.000211765775999,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / tpu / PostRev",
            "value": 0.0002109930249989,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / tpu / BothRev",
            "value": 0.0002101398350005,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / tpu / PreRev",
            "value": 0.0002083530640011,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / tpu / PostRev",
            "value": 0.0002185547980006,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / tpu / BothRev",
            "value": 0.0002260775819995,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / tpu / PreRev",
            "value": 0.0002306859540003,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / tpu / PostRev",
            "value": 0.0002265651220004,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / tpu / BothRev",
            "value": 0.0002238474510013,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / tpu / PreRev",
            "value": 0.0002254708609998,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / tpu / PostRev",
            "value": 0.0002222091700004,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / tpu / BothRev",
            "value": 0.0002304402440004,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Primal",
            "value": 0.00000784759700036375,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cpu / Primal",
            "value": 0.000007791621999786002,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / Primal",
            "value": 0.000008102587999928802,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / Primal",
            "value": 0.000008060803999796917,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / Primal",
            "value": 0.000007807172999491741,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / Primal",
            "value": 0.000007777892999911273,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / Primal",
            "value": 0.000007769394000206375,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Forward",
            "value": 0.0000120243629999095,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cpu / Forward",
            "value": 0.000012007334000372792,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / Forward",
            "value": 0.000012013643000500451,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / Forward",
            "value": 0.000011475203999907535,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / Forward",
            "value": 0.000012053077999553352,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / Forward",
            "value": 0.000011489448999782326,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / Forward",
            "value": 0.000012137122000240198,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PreRev",
            "value": 0.00001122068099994067,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PostRev",
            "value": 0.000010584530999949491,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / BothRev",
            "value": 0.000010602480000670766,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cpu / BothRev",
            "value": 0.000010735765999925209,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / PreRev",
            "value": 0.00001114813399999548,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / PostRev",
            "value": 0.00001112584699967556,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / BothRev",
            "value": 0.00001118955200035998,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / PreRev",
            "value": 0.000010663926000233914,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / PostRev",
            "value": 0.000011230092000005242,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / BothRev",
            "value": 0.000010744316999989678,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / PreRev",
            "value": 0.000011301874999844586,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / PostRev",
            "value": 0.000010611034000248764,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / BothRev",
            "value": 0.000010697778000576365,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / PreRev",
            "value": 0.000010726657999839518,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / PostRev",
            "value": 0.000010604415000670996,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / BothRev",
            "value": 0.000011174241999469814,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / PreRev",
            "value": 0.000011150141000143776,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / PostRev",
            "value": 0.000011185311000190269,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / BothRev",
            "value": 0.000011208111000087228,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Primal",
            "value": 0.000004708624999693711,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cpu / Primal",
            "value": 0.000004721999999674154,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / Primal",
            "value": 0.000004821958000320592,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / Primal",
            "value": 0.000004966125001374167,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / Primal",
            "value": 0.000005011334000300849,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / Primal",
            "value": 0.000004974082999979146,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / Primal",
            "value": 0.000004870999999184278,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / Forward",
            "value": 0.00000769291699907626,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cpu / Forward",
            "value": 0.000007709458001045277,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / Forward",
            "value": 0.000007925625001007575,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / Forward",
            "value": 0.000007835541999156704,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / Forward",
            "value": 0.000007699083000261453,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / Forward",
            "value": 0.000007866833000662154,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / Forward",
            "value": 0.000007622790999448625,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PreRev",
            "value": 0.00000715074999970966,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / PostRev",
            "value": 0.000007057832999635139,
            "unit": "s"
          },
          {
            "name": "sum    / JaXPipe / cpu / BothRev",
            "value": 0.000006889834001412965,
            "unit": "s"
          },
          {
            "name": "sum    / Jax / cpu / BothRev",
            "value": 0.000006935375000466593,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / PreRev",
            "value": 0.000007182709001426701,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / PostRev",
            "value": 0.0000072476659988751635,
            "unit": "s"
          },
          {
            "name": "sum    / HLOOpt / cpu / BothRev",
            "value": 0.000007008167000094545,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / PreRev",
            "value": 0.000007038583000394283,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / PostRev",
            "value": 0.000007091999999829568,
            "unit": "s"
          },
          {
            "name": "sum    / PartOpt / cpu / BothRev",
            "value": 0.000007437083000695565,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / PreRev",
            "value": 0.000007194707999587991,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / PostRev",
            "value": 0.0000071138339990284296,
            "unit": "s"
          },
          {
            "name": "sum    / IPartOpt / cpu / BothRev",
            "value": 0.000007114999998520944,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / PreRev",
            "value": 0.00000719391600068775,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / PostRev",
            "value": 0.000007059707999360398,
            "unit": "s"
          },
          {
            "name": "sum    / DefOpt / cpu / BothRev",
            "value": 0.000007295500001418987,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / PreRev",
            "value": 0.000006898874999023974,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / PostRev",
            "value": 0.000007242542000312824,
            "unit": "s"
          },
          {
            "name": "sum    / IDefOpt / cpu / BothRev",
            "value": 0.000007248749998325366,
            "unit": "s"
          },
          {
            "name": "value_and_grad / JaXPipe / cpu / Primal",
            "value": 0.000010430858000290756,
            "unit": "s"
          },
          {
            "name": "value_and_grad / Jax / cpu / Primal",
            "value": 0.000010823689000062588,
            "unit": "s"
          },
          {
            "name": "value_and_grad / HLOOpt / cpu / Primal",
            "value": 0.000011139512999761792,
            "unit": "s"
          },
          {
            "name": "value_and_grad / PartOpt / cpu / Primal",
            "value": 0.000010896349000176998,
            "unit": "s"
          },
          {
            "name": "value_and_grad / IPartOpt / cpu / Primal",
            "value": 0.000010923802999968757,
            "unit": "s"
          },
          {
            "name": "value_and_grad / DefOpt / cpu / Primal",
            "value": 0.000011139351000110765,
            "unit": "s"
          },
          {
            "name": "value_and_grad / IDefOpt / cpu / Primal",
            "value": 0.000011078526000346757,
            "unit": "s"
          },
          {
            "name": "value_and_grad / JaXPipe / cpu / Primal",
            "value": 0.00001696273199922871,
            "unit": "s"
          },
          {
            "name": "value_and_grad / Jax / cpu / Primal",
            "value": 0.000017215765998116696,
            "unit": "s"
          },
          {
            "name": "value_and_grad / HLOOpt / cpu / Primal",
            "value": 0.00001814972799911629,
            "unit": "s"
          },
          {
            "name": "value_and_grad / PartOpt / cpu / Primal",
            "value": 0.00001811254800122697,
            "unit": "s"
          },
          {
            "name": "value_and_grad / IPartOpt / cpu / Primal",
            "value": 0.0000179570709951804,
            "unit": "s"
          },
          {
            "name": "value_and_grad / DefOpt / cpu / Primal",
            "value": 0.00001817802499863319,
            "unit": "s"
          },
          {
            "name": "value_and_grad / IDefOpt / cpu / Primal",
            "value": 0.00001738966000266373,
            "unit": "s"
          },
          {
            "name": "value_and_grad / JaXPipe / tpu / Primal",
            "value": 0.000247302841999,
            "unit": "s"
          },
          {
            "name": "value_and_grad / Jax / tpu / Primal",
            "value": 0.0002542314549991,
            "unit": "s"
          },
          {
            "name": "value_and_grad / HLOOpt / tpu / Primal",
            "value": 0.0002540210949991,
            "unit": "s"
          },
          {
            "name": "value_and_grad / PartOpt / tpu / Primal",
            "value": 0.0002551761650011,
            "unit": "s"
          },
          {
            "name": "value_and_grad / IPartOpt / tpu / Primal",
            "value": 0.0002125708059993,
            "unit": "s"
          },
          {
            "name": "value_and_grad / DefOpt / tpu / Primal",
            "value": 0.0002361272960006,
            "unit": "s"
          },
          {
            "name": "value_and_grad / IDefOpt / tpu / Primal",
            "value": 0.0002524715039999,
            "unit": "s"
          },
          {
            "name": "value_and_grad / JaXPipe / cpu / Primal",
            "value": 0.00001177920899954188,
            "unit": "s"
          },
          {
            "name": "value_and_grad / Jax / cpu / Primal",
            "value": 0.000012563374999444931,
            "unit": "s"
          },
          {
            "name": "value_and_grad / HLOOpt / cpu / Primal",
            "value": 0.000012511530999290698,
            "unit": "s"
          },
          {
            "name": "value_and_grad / PartOpt / cpu / Primal",
            "value": 0.000011962515999584866,
            "unit": "s"
          },
          {
            "name": "value_and_grad / IPartOpt / cpu / Primal",
            "value": 0.000012558051999803866,
            "unit": "s"
          },
          {
            "name": "value_and_grad / DefOpt / cpu / Primal",
            "value": 0.000012770442999681107,
            "unit": "s"
          },
          {
            "name": "value_and_grad / IDefOpt / cpu / Primal",
            "value": 0.000012699527999757263,
            "unit": "s"
          },
          {
            "name": "value_and_grad / JaXPipe / cpu / Primal",
            "value": 0.000008620167000117362,
            "unit": "s"
          },
          {
            "name": "value_and_grad / Jax / cpu / Primal",
            "value": 0.000008366582998860394,
            "unit": "s"
          },
          {
            "name": "value_and_grad / HLOOpt / cpu / Primal",
            "value": 0.000008448541999314329,
            "unit": "s"
          },
          {
            "name": "value_and_grad / PartOpt / cpu / Primal",
            "value": 0.000009164291999695706,
            "unit": "s"
          },
          {
            "name": "value_and_grad / IPartOpt / cpu / Primal",
            "value": 0.000008468125000945292,
            "unit": "s"
          },
          {
            "name": "value_and_grad / DefOpt / cpu / Primal",
            "value": 0.000008380417000807937,
            "unit": "s"
          },
          {
            "name": "value_and_grad / IDefOpt / cpu / Primal",
            "value": 0.000008416542001214111,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / cpu / Primal",
            "value": 0.0745420540013583,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / Jax / cpu / Primal",
            "value": 0.0726386609996552,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / cpu / Primal",
            "value": 0.1068170426005963,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / cpu / Primal",
            "value": 0.0748138784008915,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / cpu / Primal",
            "value": 0.0737803805997828,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / cpu / Primal",
            "value": 0.1091294555997592,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / cpu / Primal",
            "value": 0.1116904903989052,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / cpu / Forward",
            "value": 0.2178442689997609,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / Jax / cpu / Forward",
            "value": 0.1045118840003851,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / cpu / Forward",
            "value": 0.2194634351995773,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / cpu / Forward",
            "value": 0.220826280399342,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / cpu / Forward",
            "value": 0.2243695372002548,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / cpu / Forward",
            "value": 0.2224523030003183,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / cpu / Forward",
            "value": 0.2194057123997481,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / cpu / PostRev",
            "value": 0.1478212975998758,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / Jax / cpu / BothRev",
            "value": 0.156103541199991,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / cpu / PostRev",
            "value": 0.2182444957987172,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / cpu / PostRev",
            "value": 0.1484080800000811,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / cpu / PostRev",
            "value": 0.1472968944013701,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / cpu / PostRev",
            "value": 0.2101061088003916,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / cpu / PostRev",
            "value": 0.2093057320002117,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / tpu / Primal",
            "value": 0.0092800641999929,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / Jax / tpu / Primal",
            "value": 0.0092825222000101,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / tpu / Primal",
            "value": 0.0092236541997408,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / tpu / Primal",
            "value": 0.009258138199948,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / tpu / Primal",
            "value": 0.0093055339999409,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / tpu / Primal",
            "value": 0.0091072419996635,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / tpu / Primal",
            "value": 0.0090698820000397,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / tpu / Forward",
            "value": 0.0178935582000121,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / Jax / tpu / Forward",
            "value": 0.0183222721996571,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / tpu / Forward",
            "value": 0.0178825562001293,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / tpu / Forward",
            "value": 0.0179093220001959,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / tpu / Forward",
            "value": 0.0178759319998789,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / tpu / Forward",
            "value": 0.0179002479999326,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / tpu / Forward",
            "value": 0.0179200539998419,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / tpu / PostRev",
            "value": 0.0198382364000281,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / Jax / tpu / BothRev",
            "value": 0.0198224244002631,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / tpu / PostRev",
            "value": 0.0189363379999122,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / tpu / PostRev",
            "value": 0.0197810943998774,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / tpu / PostRev",
            "value": 0.0197412686000461,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / tpu / PostRev",
            "value": 0.0185783037999499,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / tpu / PostRev",
            "value": 0.0182315276000736,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / cpu / Primal",
            "value": 0.0565480308001497,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / Jax / cpu / Primal",
            "value": 0.0555567293999047,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / cpu / Primal",
            "value": 0.080466759599949,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / cpu / Primal",
            "value": 0.056761479799934,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / cpu / Primal",
            "value": 0.0551955316001112,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / cpu / Primal",
            "value": 0.0817732423998677,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / cpu / Primal",
            "value": 0.0813250581999454,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / cpu / Forward",
            "value": 0.1509058781999556,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / Jax / cpu / Forward",
            "value": 0.0783164770000439,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / cpu / Forward",
            "value": 0.1539307266000832,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / cpu / Forward",
            "value": 0.1518713511999521,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / cpu / Forward",
            "value": 0.1542741295999803,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / cpu / Forward",
            "value": 0.1504707441999926,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / cpu / Forward",
            "value": 0.1513185517998863,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / JaXPipe / cpu / PostRev",
            "value": 0.1185703536000801,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / Jax / cpu / BothRev",
            "value": 0.1229530255999634,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / HLOOpt / cpu / PostRev",
            "value": 0.1571415465999962,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / PartOpt / cpu / PostRev",
            "value": 0.1136352674000591,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IPartOpt / cpu / PostRev",
            "value": 0.1147810304000813,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / DefOpt / cpu / PostRev",
            "value": 0.1617063440000493,
            "unit": "s"
          },
          {
            "name": "jaxmd40 / IDefOpt / cpu / PostRev",
            "value": 0.1533379971999238,
            "unit": "s"
          },
          {
            "name": "neural_gcm_dynamic_forcing_deterministic_1_4_deg / JaXPipe / cpu / Primal",
            "value": 52.11356961999991,
            "unit": "s"
          },
          {
            "name": "neural_gcm_dynamic_forcing_deterministic_1_4_deg / Jax / cpu / Primal",
            "value": 52.08034937999764,
            "unit": "s"
          },
          {
            "name": "neural_gcm_dynamic_forcing_deterministic_1_4_deg / HLOOpt / cpu / Primal",
            "value": 52.221077242997126,
            "unit": "s"
          },
          {
            "name": "neural_gcm_dynamic_forcing_deterministic_1_4_deg / PartOpt / cpu / Primal",
            "value": 51.32122608500504,
            "unit": "s"
          },
          {
            "name": "neural_gcm_dynamic_forcing_deterministic_1_4_deg / IPartOpt / cpu / Primal",
            "value": 51.161993307003286,
            "unit": "s"
          },
          {
            "name": "neural_gcm_dynamic_forcing_deterministic_1_4_deg / DefOpt / cpu / Primal",
            "value": 26.20128602800105,
            "unit": "s"
          },
          {
            "name": "neural_gcm_dynamic_forcing_deterministic_1_4_deg / IDefOpt / cpu / Primal",
            "value": 56.70295566900313,
            "unit": "s"
          },
          {
            "name": "neural_gcm_dynamic_forcing_deterministic_1_4_deg / JaXPipe / tpu / Primal",
            "value": 0.1902038270000048,
            "unit": "s"
          },
          {
            "name": "neural_gcm_dynamic_forcing_deterministic_1_4_deg / Jax / tpu / Primal",
            "value": 0.1900181960008922,
            "unit": "s"
          },
          {
            "name": "neural_gcm_dynamic_forcing_deterministic_1_4_deg / HLOOpt / tpu / Primal",
            "value": 0.1886999550006294,
            "unit": "s"
          },
          {
            "name": "neural_gcm_dynamic_forcing_deterministic_1_4_deg / PartOpt / tpu / Primal",
            "value": 0.2028921410001203,
            "unit": "s"
          },
          {
            "name": "neural_gcm_dynamic_forcing_deterministic_1_4_deg / IPartOpt / tpu / Primal",
            "value": 0.2028880210000352,
            "unit": "s"
          },
          {
            "name": "neural_gcm_dynamic_forcing_deterministic_1_4_deg / DefOpt / tpu / Primal",
            "value": 0.1740487600000051,
            "unit": "s"
          },
          {
            "name": "neural_gcm_dynamic_forcing_deterministic_1_4_deg / IDefOpt / tpu / Primal",
            "value": 0.1851054740000108,
            "unit": "s"
          },
          {
            "name": "neural_gcm_dynamic_forcing_deterministic_1_4_deg / JaXPipe / cpu / Primal",
            "value": 51.0242726610004,
            "unit": "s"
          },
          {
            "name": "neural_gcm_dynamic_forcing_deterministic_1_4_deg / Jax / cpu / Primal",
            "value": 51.43752789700011,
            "unit": "s"
          },
          {
            "name": "neural_gcm_dynamic_forcing_deterministic_1_4_deg / HLOOpt / cpu / Primal",
            "value": 49.78114223300054,
            "unit": "s"
          },
          {
            "name": "neural_gcm_dynamic_forcing_deterministic_1_4_deg / PartOpt / cpu / Primal",
            "value": 49.56505905599988,
            "unit": "s"
          },
          {
            "name": "neural_gcm_dynamic_forcing_deterministic_1_4_deg / IPartOpt / cpu / Primal",
            "value": 49.99032651299967,
            "unit": "s"
          },
          {
            "name": "neural_gcm_dynamic_forcing_deterministic_1_4_deg / DefOpt / cpu / Primal",
            "value": 22.38827926399972,
            "unit": "s"
          },
          {
            "name": "neural_gcm_dynamic_forcing_deterministic_1_4_deg / IDefOpt / cpu / Primal",
            "value": 55.5757748140004,
            "unit": "s"
          }
        ]
      }
    ]
  }
}