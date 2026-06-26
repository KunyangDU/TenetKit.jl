# Tree

────────────────────────────────────────────────────────────────────────────────────
           >>> TDVP >>>                    Time                    Allocations
    ───────────────────────   ────────────────────────
        Tot / % measured:              41.5s /  99.8%            110GiB / 100.0%

Section                   ncalls     time    %tot     avg     alloc    %tot      avg
────────────────────────────────────────────────────────────────────────────────────
evolve                        16    15.2s   36.7%   951ms   66.8GiB   61.0%  4.18GiB
  action                     143    14.5s   34.9%   101ms   63.5GiB   57.9%   454MiB
    _action1_1_2_11_2      1.31k    48.0s  115.9%  36.7ms    216GiB  196.7%   169MiB
    _action1_1_2_00_2      1.34k    46.4s  112.0%  34.5ms    219GiB  199.6%   167MiB
CBE                           15    13.2s   31.9%   881ms   27.9GiB   25.4%  1.86GiB
  pushleft                    15    5.62s   13.6%   375ms   4.21GiB    3.8%   288MiB
  CBE!                        15    4.16s   10.0%   277ms   9.19GiB    8.4%   628MiB
    contract_1                15    1.13s    2.7%  75.3ms   3.43GiB    3.1%   234MiB
    SVD                       15    766ms    1.9%  51.1ms    181MiB    0.2%  12.1MiB
    leftorth                  15    486ms    1.2%  32.4ms    174MiB    0.2%  11.6MiB
    contract_2                15    430ms    1.0%  28.6ms   2.16GiB    2.0%   147MiB
    direct-sum                15    425ms    1.0%  28.3ms    337MiB    0.3%  22.5MiB
    splice Q'                 15    387ms    0.9%  25.8ms   1.24GiB    1.1%  84.9MiB
    splice Λ                  15    270ms    0.7%  18.0ms   1.40GiB    1.3%  95.6MiB
    splice                    15    230ms    0.6%  15.4ms    202MiB    0.2%  13.4MiB
    after-orthogonalize       15   26.8ms    0.1%  1.78ms   49.5MiB    0.0%  3.30MiB
  left orthogonalize          15    1.68s    4.1%   112ms   7.69GiB    7.0%   525MiB
  right orthogonalize         15    1.44s    3.5%  95.9ms   6.66GiB    6.1%   455MiB
  leftorth                    15    296ms    0.7%  19.7ms    124MiB    0.1%  8.26MiB
back evolve                   15    7.70s   18.6%   513ms   10.5GiB    9.6%   719MiB
  action                     129    1.64s    3.9%  12.7ms   5.41GiB    4.9%  43.0MiB
    _action0_0_2_2         1.65k    10.7s   25.7%  6.46ms   30.2GiB   27.6%  18.8MiB
pushright                     15    4.36s   10.5%   291ms   3.93GiB    3.6%   268MiB
orthogonalize                 15    833ms    2.0%  55.5ms    307MiB    0.3%  20.5MiB
  SVD                         15    724ms    1.7%  48.3ms    201MiB    0.2%  13.4MiB
  contract                    15   32.4ms    0.1%  2.16ms   11.0MiB    0.0%   750KiB
GC                             1   47.1ms    0.1%  47.1ms     0.00B    0.0%    0.00B
serialize                    123   28.8ms    0.1%   234μs    218KiB    0.0%  1.77KiB
deserialize                  140   24.6ms    0.1%   176μs    118MiB    0.1%   863KiB
────────────────────────────────────────────────────────────────────────────────────
D( 256 => 256 ), S = 2.336960996584951, σS = 0.3376503928565827, ⟨S⟩ = 2.035441494199431, max |ΔS| = 0.5664763907678163
K = 10, TruncError = 3.717732387253036e-5, LanczosError = 2.4400239848940624e-9
E = -10.116516593934127

# Graph

────────────────────────────────────────────────────────────────────────────────────
           >>> TDVP >>>                    Time                    Allocations
    ───────────────────────   ────────────────────────
        Tot / % measured:              37.0s /  99.7%            109GiB /  99.9%

Section                   ncalls     time    %tot     avg     alloc    %tot      avg
────────────────────────────────────────────────────────────────────────────────────
evolve                        16    13.1s   35.4%   816ms   61.0GiB   55.8%  3.81GiB
  action                     144    12.4s   33.7%  86.2ms   57.5GiB   52.6%   409MiB
    _action1_1_2_00_2      1.29k    45.3s  122.7%  35.1ms    200GiB  182.7%   158MiB
    _action1_1_2_11_2        825    32.4s   87.8%  39.3ms    140GiB  127.9%   174MiB
CBE                           15    12.0s   32.7%   803ms   28.1GiB   25.7%  1.87GiB
  pushleft                    15    4.73s   12.8%   315ms   3.74GiB    3.4%   255MiB
  CBE!                        15    4.32s   11.7%   288ms   11.7GiB   10.7%   796MiB
    contract_1                15    975ms    2.6%  65.0ms   4.53GiB    4.1%   309MiB
    SVD                       15    796ms    2.2%  53.0ms    265MiB    0.2%  17.7MiB
    contract_2                15    584ms    1.6%  38.9ms   2.89GiB    2.6%   198MiB
    leftorth                  15    530ms    1.4%  35.3ms    205MiB    0.2%  13.7MiB
    direct-sum                15    443ms    1.2%  29.5ms    311MiB    0.3%  20.7MiB
    splice Λ                  15    366ms    1.0%  24.4ms   1.69GiB    1.5%   116MiB
    splice Q'                 15    349ms    0.9%  23.2ms   1.48GiB    1.4%   101MiB
    splice                    15    238ms    0.6%  15.8ms    202MiB    0.2%  13.4MiB
    after-orthogonalize       15   28.4ms    0.1%  1.89ms   49.6MiB    0.0%  3.31MiB
  left orthogonalize          15    1.43s    3.9%  95.3ms   6.66GiB    6.1%   455MiB
  right orthogonalize         15    1.19s    3.2%  79.4ms   5.89GiB    5.4%   402MiB
  leftorth                    15    370ms    1.0%  24.6ms    149MiB    0.1%  9.94MiB
back evolve                   15    7.38s   20.0%   492ms   16.5GiB   15.1%  1.10GiB
  action                     129    2.44s    6.6%  18.9ms   11.8GiB   10.8%  93.6MiB
    _action0_0_2_2         1.95k    15.4s   41.6%  7.87ms   52.0GiB   47.6%  27.3MiB
pushright                     15    3.59s    9.7%   239ms   3.41GiB    3.1%   233MiB
orthogonalize                 15    762ms    2.1%  50.8ms    342MiB    0.3%  22.8MiB
  SVD                         15    753ms    2.0%  50.2ms    291MiB    0.3%  19.4MiB
  contract                    15    810μs    0.0%  54.0μs   11.0MiB    0.0%   753KiB
GC                             1   53.2ms    0.1%  53.2ms     0.00B    0.0%    0.00B
────────────────────────────────────────────────────────────────────────────────────
D( 256 => 256 ), S = 2.3363956595478017, σS = 0.3374563208405896, ⟨S⟩ = 2.0350134535845203, max |ΔS| = 0.5663945992430333
K = 10, TruncError = 3.431597075611411e-5, LanczosError = 1.7946273961481181e-9
E = -10.11647950988229
