## E1 ladder

| Model | Decoder | Macro-F1 | Recall lvl 2 | AUC any crash | AUC 2+ crashes | Top-10% capture | Top-10% capture (by length) | Accuracy |
|---|---|---|---|---|---|---|---|---|
| M0 majority | argmax | 0.315 ± 0.013 | 0.000 ± 0.000 | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.094 ± 0.027 | 0.092 ± 0.027 | 0.900 |
| M1 RideScore LTS v1 | argmax | 0.283 ± 0.019 | 0.635 ± 0.117 | 0.623 ± 0.058 | 0.662 ± 0.087 | 0.178 ± 0.047 | 0.155 ± 0.051 | 0.548 |
| M2 SPF (neg. binomial) | thresholds | 0.410 ± 0.023 | 0.216 ± 0.095 | 0.739 ± 0.014 | 0.833 ± 0.055 | 0.391 ± 0.075 | 0.197 ± 0.051 | 0.834 |
| M3 gradient boosting | thresholds | 0.439 ± 0.028 | 0.263 ± 0.110 | 0.743 ± 0.032 | 0.830 ± 0.070 | 0.414 ± 0.070 | 0.368 ± 0.063 | 0.839 |
| F fusion (fixed weights) | thresholds | 0.440 ± 0.038 | 0.293 ± 0.122 | 0.755 ± 0.027 | 0.844 ± 0.061 | 0.430 ± 0.081 | 0.358 ± 0.066 | 0.841 |
| F fusion (equal weights) | thresholds | 0.439 ± 0.032 | 0.288 ± 0.091 | 0.756 ± 0.024 | 0.848 ± 0.059 | 0.435 ± 0.077 | 0.336 ± 0.056 | 0.836 |
| F fusion (val-selected weights) | thresholds | 0.442 ± 0.031 | 0.266 ± 0.097 | 0.755 ± 0.028 | 0.843 ± 0.059 | 0.428 ± 0.078 | 0.334 ± 0.074 | 0.838 |

## E3 decoders (macro-F1)

| Model | argmax | logit bias (val) | expected-level thresholds (val) |
|---|---|---|---|
| M2 SPF (neg. binomial) | 0.331 | 0.408 | 0.410 |
| M3 gradient boosting | 0.418 | 0.432 | 0.439 |
| F fusion (fixed weights) | 0.379 | 0.427 | 0.440 |

## Per ward (fusion, thresholds)

| Test ward | Sub-blocks | Level-2 | Macro-F1 | AUC any | Top-10% capture |
|---|---|---|---|---|---|
| 1 | 1296 | 104 | 0.508 | 0.752 | 0.459 |
| 2 | 2250 | 185 | 0.461 | 0.751 | 0.317 |
| 3 | 2886 | 12 | 0.418 | 0.787 | 0.568 |
| 4 | 3462 | 37 | 0.453 | 0.758 | 0.455 |
| 5 | 3078 | 68 | 0.401 | 0.769 | 0.409 |
| 6 | 1986 | 87 | 0.447 | 0.738 | 0.436 |
| 7 | 2681 | 11 | 0.390 | 0.702 | 0.326 |
| 8 | 1915 | 21 | 0.438 | 0.780 | 0.467 |

## lts_vs_crashes

| ref_lts | n_subblocks | length_km | crashes | mean_model_score | share_predicted_level2 | crashes_per_km | share_any_crash | share_of_all_crashes | share_of_network_km | share_level_0 | share_level_1 | share_level_2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LTS 1 | 394.0 | 33.197 | 216.0 | 0.519 | 0.277 | 6.507 | 0.272 | 0.079 | 0.019 | 0.728 | 0.14 | 0.132 |
| LTS 2 | 10882.0 | 896.776 | 760.0 | 0.113 | 0.01 | 0.848 | 0.051 | 0.277 | 0.512 | 0.949 | 0.039 | 0.011 |
| LTS 3 | 2250.0 | 208.88 | 216.0 | 0.118 | 0.025 | 1.034 | 0.057 | 0.079 | 0.119 | 0.943 | 0.039 | 0.018 |
| LTS 4 | 6028.0 | 612.797 | 1550.0 | 0.292 | 0.066 | 2.529 | 0.156 | 0.565 | 0.35 | 0.844 | 0.105 | 0.051 |
| all | 19554.0 | 1751.65 | 2742.0 | 0.177 | 0.034 | 1.565 | 0.088 | 1.0 | 1.0 | 0.912 | 0.062 | 0.027 |

## surprise_breakdown

| metric | value | all | level2 | pred2 | looks_safe_has_crashes | looks_risky_no_crashes |
|---|---|---|---|---|---|---|
| n_subblocks | nan | 19554.0 | 525.0 | 671.0 | 196.0 | 339.0 |
| mean_mu | nan | 0.177 | 0.57 | 0.943 | 0.215 | 0.932 |
| mean_length_m | nan | 89.58 | 108.986 | 127.559 | 98.834 | 135.484 |
| crashes_total | nan | 2742.0 | 1539.0 | 759.0 | 510.0 | 0.0 |
| mean_intersection_share_of_crashes | sub-blocks with >=1 crash | 0.254 | 0.227 | 0.218 | 0.201 | nan |
| pooled_intersection_share_of_crashes | sum int / sum crashes | 0.238 | 0.216 | 0.192 | 0.204 | nan |
| mean_midblock_share_of_crashes | sub-blocks with >=1 crash | 0.746 | 0.773 | 0.782 | 0.799 | nan |
| net_max_legs | share >= 4 | 0.576 | 0.825 | 0.887 | 0.755 | 0.891 |
| net_max_legs | mean | 3.521 | 3.909 | 3.985 | 3.791 | 3.994 |
| net_max_legs | share <= 2 | 0.092 | 0.023 | 0.012 | 0.031 | 0.006 |
| net_max_legs | share == 3 | 0.332 | 0.152 | 0.101 | 0.214 | 0.103 |
| net_max_legs | share == 4 | 0.542 | 0.722 | 0.784 | 0.694 | 0.794 |
| net_max_legs | share >= 5 | 0.034 | 0.103 | 0.103 | 0.061 | 0.097 |
| ddot_fhwa_class_share | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| ddot_fhwa_class_share | 2 | 0.002 | 0.0 | 0.006 | 0.0 | 0.012 |
| ddot_fhwa_class_share | 3 | 0.079 | 0.328 | 0.437 | 0.158 | 0.401 |
| ddot_fhwa_class_share | 4 | 0.141 | 0.413 | 0.489 | 0.296 | 0.504 |
| ddot_fhwa_class_share | 5 | 0.132 | 0.131 | 0.06 | 0.224 | 0.074 |
| ddot_fhwa_class_share | 6 | 0.012 | 0.01 | 0.0 | 0.026 | 0.0 |
| ddot_fhwa_class_share | 7 | 0.623 | 0.118 | 0.009 | 0.296 | 0.009 |
| ddot_fhwa_class_share | missing | 0.011 | 0.0 | 0.0 | 0.0 | 0.0 |
| osmf_highway_share | cycleway | 0.01 | 0.002 | 0.0 | 0.005 | 0.0 |
| osmf_highway_share | footway | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| osmf_highway_share | living_street | 0.001 | 0.002 | 0.0 | 0.005 | 0.0 |
| osmf_highway_share | path | 0.001 | 0.0 | 0.0 | 0.0 | 0.0 |
| osmf_highway_share | primary | 0.092 | 0.301 | 0.417 | 0.163 | 0.422 |
| osmf_highway_share | primary_link | 0.008 | 0.0 | 0.0 | 0.0 | 0.0 |
| osmf_highway_share | residential | 0.597 | 0.109 | 0.004 | 0.286 | 0.009 |
| osmf_highway_share | secondary | 0.109 | 0.301 | 0.32 | 0.214 | 0.298 |
| osmf_highway_share | secondary_link | 0.002 | 0.0 | 0.0 | 0.0 | 0.0 |
| osmf_highway_share | tertiary | 0.145 | 0.194 | 0.148 | 0.245 | 0.168 |
| osmf_highway_share | tertiary_link | 0.001 | 0.0 | 0.0 | 0.0 | 0.0 |
| osmf_highway_share | trunk | 0.016 | 0.086 | 0.11 | 0.066 | 0.103 |
| osmf_highway_share | trunk_link | 0.001 | 0.0 | 0.0 | 0.0 | 0.0 |
| osmf_highway_share | unclassified | 0.016 | 0.006 | 0.0 | 0.015 | 0.0 |
| ddot_bike_best_share | 0.0 | 0.89 | 0.613 | 0.499 | 0.76 | 0.525 |
| ddot_bike_best_share | 1.0 | 0.076 | 0.24 | 0.267 | 0.189 | 0.254 |
| ddot_bike_best_share | 2.0 | 0.014 | 0.048 | 0.072 | 0.02 | 0.068 |
| ddot_bike_best_share | 3.0 | 0.02 | 0.099 | 0.162 | 0.031 | 0.153 |
| osmf_bike_best_share | 0.0 | 0.845 | 0.522 | 0.407 | 0.689 | 0.431 |
| osmf_bike_best_share | 0.5 | 0.039 | 0.059 | 0.061 | 0.056 | 0.059 |
| osmf_bike_best_share | 1.0 | 0.071 | 0.175 | 0.204 | 0.174 | 0.233 |
| osmf_bike_best_share | 2.0 | 0.002 | 0.008 | 0.013 | 0.005 | 0.015 |
| osmf_bike_best_share | 3.0 | 0.043 | 0.236 | 0.314 | 0.076 | 0.262 |
| ddot_bike_best | share > 0 (any facility) | 0.11 | 0.387 | 0.501 | 0.24 | 0.475 |
| osmf_bike_best | share > 0 (any facility) | 0.155 | 0.478 | 0.593 | 0.311 | 0.569 |
| ward_count | Ward 1 | 1296.0 | 104.0 | 90.0 | 31.0 | 27.0 |
| ward_count | Ward 2 | 2250.0 | 185.0 | 263.0 | 64.0 | 132.0 |
| ward_count | Ward 3 | 2886.0 | 12.0 | 17.0 | 0.0 | 11.0 |
| ward_count | Ward 4 | 3462.0 | 37.0 | 77.0 | 13.0 | 44.0 |
| ward_count | Ward 5 | 3078.0 | 68.0 | 41.0 | 42.0 | 25.0 |
| ward_count | Ward 6 | 1986.0 | 87.0 | 120.0 | 29.0 | 58.0 |
| ward_count | Ward 7 | 2681.0 | 11.0 | 34.0 | 6.0 | 24.0 |
| ward_count | Ward 8 | 1915.0 | 21.0 | 29.0 | 11.0 | 18.0 |
| ward_share | Ward 1 | 0.066 | 0.198 | 0.134 | 0.158 | 0.08 |
| ward_share | Ward 2 | 0.115 | 0.352 | 0.392 | 0.326 | 0.389 |
| ward_share | Ward 3 | 0.148 | 0.023 | 0.025 | 0.0 | 0.032 |
| ward_share | Ward 4 | 0.177 | 0.07 | 0.115 | 0.066 | 0.13 |
| ward_share | Ward 5 | 0.157 | 0.13 | 0.061 | 0.214 | 0.074 |
| ward_share | Ward 6 | 0.102 | 0.166 | 0.179 | 0.148 | 0.171 |
| ward_share | Ward 7 | 0.137 | 0.021 | 0.051 | 0.031 | 0.071 |
| ward_share | Ward 8 | 0.098 | 0.04 | 0.043 | 0.056 | 0.053 |
| weak_match_share | nan | 0.005 | 0.006 | 0.004 | 0.005 | 0.003 |
| ref_lts_share | LTS 1 | 0.02 | 0.099 | 0.162 | 0.031 | 0.153 |
| ref_lts_share | LTS 2 | 0.556 | 0.236 | 0.16 | 0.372 | 0.198 |
| ref_lts_share | LTS 3 | 0.115 | 0.078 | 0.085 | 0.071 | 0.065 |
| ref_lts_share | LTS 4 | 0.308 | 0.587 | 0.593 | 0.526 | 0.584 |
