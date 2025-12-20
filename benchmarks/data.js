window.BENCHMARK_DATA = {
  "lastUpdate": 1766220786249,
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
      }
    ]
  }
}