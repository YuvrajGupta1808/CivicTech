## E1 ladder

| Model | Decoder | Macro-F1 | Recall lvl 2 | AUC any crash | AUC 2+ crashes | Top-10% capture | Top-10% capture (by length) | Accuracy |
|---|---|---|---|---|---|---|---|---|
| M0 majority | argmax | 0.315 ± 0.013 | 0.000 ± 0.000 | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.094 ± 0.027 | 0.092 ± 0.027 | 0.900 |
| M1 RideScore LTS v1 | argmax | 0.283 ± 0.019 | 0.635 ± 0.117 | 0.623 ± 0.058 | 0.662 ± 0.087 | 0.178 ± 0.047 | 0.155 ± 0.051 | 0.548 |
| M2 SPF (neg. binomial) | thresholds | 0.410 ± 0.023 | 0.216 ± 0.095 | 0.739 ± 0.014 | 0.833 ± 0.055 | 0.391 ± 0.075 | 0.197 ± 0.051 | 0.834 |
| M3 gradient boosting | thresholds | 0.439 ± 0.028 | 0.263 ± 0.110 | 0.743 ± 0.032 | 0.830 ± 0.070 | 0.414 ± 0.070 | 0.368 ± 0.063 | 0.839 |
| M4 OSM-DDOT cross-attention | thresholds | 0.432 ± 0.036 | 0.310 ± 0.158 | 0.770 ± 0.032 | 0.864 ± 0.058 | 0.435 ± 0.074 | 0.378 ± 0.077 | 0.811 |
| M4-concat (no cross-attention) | thresholds | 0.438 ± 0.036 | 0.318 ± 0.181 | 0.772 ± 0.029 | 0.867 ± 0.057 | 0.439 ± 0.078 | 0.388 ± 0.074 | 0.824 |
| F fusion (fixed weights) | thresholds | 0.456 ± 0.029 | 0.324 ± 0.107 | 0.769 ± 0.029 | 0.863 ± 0.055 | 0.435 ± 0.077 | 0.374 ± 0.065 | 0.832 |
| F fusion (equal weights) | thresholds | 0.443 ± 0.030 | 0.330 ± 0.148 | 0.769 ± 0.026 | 0.863 ± 0.054 | 0.446 ± 0.079 | 0.359 ± 0.063 | 0.836 |
| F fusion (val-selected weights) | thresholds | 0.445 ± 0.034 | 0.350 ± 0.130 | 0.769 ± 0.025 | 0.862 ± 0.053 | 0.444 ± 0.067 | 0.369 ± 0.072 | 0.827 |

## E3 decoders (macro-F1)

| Model | argmax | logit bias (val) | expected-level thresholds (val) |
|---|---|---|---|
| M2 SPF (neg. binomial) | 0.331 | 0.408 | 0.410 |
| M3 gradient boosting | 0.418 | 0.432 | 0.439 |
| M4 OSM-DDOT cross-attention | 0.431 | 0.439 | 0.432 |
| M4-concat (no cross-attention) | 0.428 | 0.439 | 0.438 |
| F fusion (fixed weights) | 0.410 | 0.436 | 0.456 |

## Per ward (fusion, thresholds)

| Test ward | Sub-blocks | Level-2 | Macro-F1 | AUC any | Top-10% capture |
|---|---|---|---|---|---|
| 1 | 1296 | 104 | 0.514 | 0.770 | 0.449 |
| 2 | 2250 | 185 | 0.453 | 0.761 | 0.309 |
| 3 | 2886 | 12 | 0.449 | 0.810 | 0.568 |
| 4 | 3462 | 37 | 0.453 | 0.765 | 0.443 |
| 5 | 3078 | 68 | 0.420 | 0.785 | 0.429 |
| 6 | 1986 | 87 | 0.459 | 0.737 | 0.459 |
| 7 | 2681 | 11 | 0.428 | 0.728 | 0.356 |
| 8 | 1915 | 21 | 0.473 | 0.801 | 0.467 |

## ablation_groups

| group | n_rows | n_cols | n_folds | fits | macro_f1_mean | macro_f1_sd | auc_any_mean | auc_any_sd | auc_repeat_mean | auc_repeat_sd | top10_capture_count_mean | top10_capture_count_sd | delta_macro_f1 | delta_macro_f1_sd | delta_auc_any | delta_auc_any_sd | delta_auc_repeat | delta_auc_repeat_sd | delta_top10_capture_count | delta_top10_capture_count_sd | n_dropped |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 19554 | 57 | 8 | 1 | 0.444 | 0.029 | 0.744 | 0.032 | 0.829 | 0.07 | 0.414 | 0.078 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| speed | 19554 | 55 | 8 | 1 | 0.438 | 0.042 | 0.742 | 0.031 | 0.832 | 0.065 | 0.419 | 0.072 | -0.006 | 0.023 | -0.002 | 0.005 | 0.003 | 0.014 | 0.006 | 0.023 | 2 |
| lanes_width | 19554 | 47 | 8 | 1 | 0.419 | 0.034 | 0.734 | 0.033 | 0.819 | 0.078 | 0.4 | 0.079 | -0.025 | 0.009 | -0.01 | 0.01 | -0.01 | 0.013 | -0.013 | 0.021 | 10 |
| parking | 19554 | 54 | 8 | 1 | 0.431 | 0.037 | 0.744 | 0.026 | 0.837 | 0.064 | 0.403 | 0.069 | -0.013 | 0.015 | 0.0 | 0.007 | 0.008 | 0.025 | -0.011 | 0.017 | 3 |
| bike | 19554 | 42 | 8 | 1 | 0.44 | 0.036 | 0.739 | 0.026 | 0.815 | 0.055 | 0.399 | 0.068 | -0.004 | 0.02 | -0.004 | 0.008 | -0.014 | 0.019 | -0.014 | 0.049 | 15 |
| traffic | 19554 | 54 | 8 | 1 | 0.432 | 0.034 | 0.748 | 0.035 | 0.841 | 0.064 | 0.418 | 0.072 | -0.012 | 0.026 | 0.004 | 0.011 | 0.011 | 0.025 | 0.004 | 0.033 | 3 |
| class | 19554 | 52 | 8 | 1 | 0.428 | 0.033 | 0.738 | 0.03 | 0.829 | 0.061 | 0.403 | 0.067 | -0.016 | 0.014 | -0.005 | 0.007 | -0.0 | 0.023 | -0.011 | 0.025 | 5 |
| conflicts | 19554 | 53 | 8 | 1 | 0.445 | 0.029 | 0.743 | 0.031 | 0.833 | 0.067 | 0.413 | 0.066 | 0.001 | 0.006 | -0.0 | 0.007 | 0.004 | 0.011 | -0.001 | 0.025 | 4 |
| calming | 19554 | 54 | 8 | 1 | 0.445 | 0.039 | 0.746 | 0.032 | 0.829 | 0.071 | 0.414 | 0.073 | 0.001 | 0.024 | 0.002 | 0.006 | -0.0 | 0.014 | 0.001 | 0.023 | 3 |
| pavement | 19554 | 54 | 8 | 1 | 0.419 | 0.037 | 0.74 | 0.03 | 0.818 | 0.059 | 0.392 | 0.062 | -0.025 | 0.022 | -0.004 | 0.01 | -0.011 | 0.013 | -0.022 | 0.041 | 3 |
| sidewalk | 19554 | 56 | 8 | 1 | 0.429 | 0.029 | 0.743 | 0.027 | 0.834 | 0.071 | 0.42 | 0.089 | -0.015 | 0.017 | -0.001 | 0.009 | 0.005 | 0.021 | 0.006 | 0.018 | 1 |
| network | 19554 | 49 | 8 | 1 | 0.418 | 0.032 | 0.706 | 0.031 | 0.773 | 0.095 | 0.359 | 0.074 | -0.026 | 0.012 | -0.038 | 0.012 | -0.056 | 0.038 | -0.055 | 0.018 | 8 |

## ablation_sources

| sources | n_rows | n_cols | n_folds | fits | macro_f1_mean | macro_f1_sd | auc_any_mean | auc_any_sd | auc_repeat_mean | auc_repeat_sd | top10_capture_count_mean | top10_capture_count_sd | delta_macro_f1 | delta_macro_f1_sd | delta_auc_any | delta_auc_any_sd | delta_auc_repeat | delta_auc_repeat_sd | delta_top10_capture_count | delta_top10_capture_count_sd |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ddot+net | 19554 | 42 | 8 | 1 | 0.429 | 0.031 | 0.741 | 0.027 | 0.82 | 0.065 | 0.414 | 0.072 | -0.016 | 0.025 | -0.002 | 0.007 | -0.009 | 0.02 | 0.001 | 0.017 |
| osm+net | 19554 | 22 | 8 | 1 | 0.411 | 0.039 | 0.737 | 0.03 | 0.821 | 0.059 | 0.361 | 0.07 | -0.034 | 0.036 | -0.007 | 0.02 | -0.008 | 0.057 | -0.053 | 0.055 |
| ddot+osm | 19554 | 50 | 8 | 1 | 0.41 | 0.025 | 0.704 | 0.035 | 0.778 | 0.083 | 0.352 | 0.057 | -0.034 | 0.01 | -0.04 | 0.017 | -0.052 | 0.024 | -0.062 | 0.032 |
| net only | 19554 | 7 | 8 | 1 | 0.378 | 0.028 | 0.632 | 0.015 | 0.705 | 0.062 | 0.254 | 0.049 | -0.066 | 0.033 | -0.112 | 0.04 | -0.124 | 0.105 | -0.159 | 0.098 |
| all | 19554 | 57 | 8 | 1 | 0.444 | 0.029 | 0.744 | 0.032 | 0.829 | 0.07 | 0.414 | 0.078 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## label_variants

| label | n_rows | n_cols | n_folds | fits | macro_f1_mean | macro_f1_sd | auc_any_mean | auc_any_sd | auc_repeat_mean | auc_repeat_sd | top10_capture_count_mean | top10_capture_count_sd | count_col |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| level (any, baseline) | 19554 | 57 | 8 | 1 | 0.444 | 0.029 | 0.744 | 0.032 | 0.829 | 0.07 | 0.414 | 0.078 | crash_count |
| level_injury | 19554 | 57 | 8 | 1 | 0.425 | 0.041 | 0.736 | 0.025 | 0.846 | 0.06 | 0.39 | 0.049 | injury_count |
| level_midblock | 19554 | 57 | 8 | 1 | 0.419 | 0.032 | 0.739 | 0.026 | 0.819 | 0.059 | 0.396 | 0.051 | midblock_count |
| level_block | 13068 | 57 | 8 | 1 | 0.428 | 0.042 | 0.706 | 0.034 | 0.8 | 0.062 | 0.362 | 0.066 | crash_count_block |

## robustness

| variant | n_rows | n_cols | n_folds | fits | macro_f1_mean | macro_f1_sd | auc_any_mean | auc_any_sd | auc_repeat_mean | auc_repeat_sd | top10_capture_count_mean | top10_capture_count_sd | delta_macro_f1 | delta_macro_f1_sd | delta_auc_any | delta_auc_any_sd | delta_auc_repeat | delta_auc_repeat_sd | delta_top10_capture_count | delta_top10_capture_count_sd |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all rows | 19554 | 57 | 8 | 1 | 0.444 | 0.029 | 0.744 | 0.032 | 0.829 | 0.07 | 0.414 | 0.078 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| drop weak_match | 19454 | 57 | 8 | 1 | 0.43 | 0.017 | 0.743 | 0.03 | 0.826 | 0.069 | 0.411 | 0.064 | -0.014 | 0.018 | -0.001 | 0.003 | -0.003 | 0.008 | -0.003 | 0.03 |

## lts_vs_crashes

| ref_lts | n_subblocks | length_km | crashes | mean_model_score | share_predicted_level2 | crashes_per_km | share_any_crash | share_of_all_crashes | share_of_network_km | share_level_0 | share_level_1 | share_level_2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LTS 1 | 394.0 | 33.197 | 216.0 | 0.635 | 0.264 | 6.507 | 0.272 | 0.079 | 0.019 | 0.728 | 0.14 | 0.132 |
| LTS 2 | 10882.0 | 896.776 | 760.0 | 0.16 | 0.011 | 0.848 | 0.051 | 0.277 | 0.512 | 0.949 | 0.039 | 0.011 |
| LTS 3 | 2250.0 | 208.88 | 216.0 | 0.178 | 0.025 | 1.034 | 0.057 | 0.079 | 0.119 | 0.943 | 0.039 | 0.018 |
| LTS 4 | 6028.0 | 612.797 | 1550.0 | 0.385 | 0.063 | 2.529 | 0.156 | 0.565 | 0.35 | 0.844 | 0.105 | 0.051 |
| all | 19554.0 | 1751.65 | 2742.0 | 0.241 | 0.034 | 1.565 | 0.088 | 1.0 | 1.0 | 0.912 | 0.062 | 0.027 |

## surprise_breakdown

| metric | value | all | level2 | pred2 | looks_safe_has_crashes | looks_risky_no_crashes |
|---|---|---|---|---|---|---|
| n_subblocks | nan | 19554.0 | 525.0 | 662.0 | 176.0 | 330.0 |
| mean_mu | nan | 0.241 | 0.714 | 1.088 | 0.307 | 1.074 |
| mean_length_m | nan | 89.58 | 108.986 | 126.152 | 101.048 | 132.419 |
| crashes_total | nan | 2742.0 | 1539.0 | 771.0 | 450.0 | 0.0 |
| mean_intersection_share_of_crashes | sub-blocks with >=1 crash | 0.254 | 0.227 | 0.215 | 0.209 | nan |
| pooled_intersection_share_of_crashes | sum int / sum crashes | 0.238 | 0.216 | 0.196 | 0.218 | nan |
| mean_midblock_share_of_crashes | sub-blocks with >=1 crash | 0.746 | 0.773 | 0.785 | 0.791 | nan |
| net_max_legs | share >= 4 | 0.576 | 0.825 | 0.915 | 0.739 | 0.921 |
| net_max_legs | mean | 3.521 | 3.909 | 4.026 | 3.767 | 4.033 |
| net_max_legs | share <= 2 | 0.092 | 0.023 | 0.006 | 0.034 | 0.003 |
| net_max_legs | share == 3 | 0.332 | 0.152 | 0.078 | 0.227 | 0.076 |
| net_max_legs | share == 4 | 0.542 | 0.722 | 0.807 | 0.682 | 0.818 |
| net_max_legs | share >= 5 | 0.034 | 0.103 | 0.109 | 0.057 | 0.103 |
| ddot_fhwa_class_share | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| ddot_fhwa_class_share | 2 | 0.002 | 0.0 | 0.004 | 0.0 | 0.009 |
| ddot_fhwa_class_share | 3 | 0.079 | 0.328 | 0.449 | 0.136 | 0.442 |
| ddot_fhwa_class_share | 4 | 0.141 | 0.413 | 0.461 | 0.261 | 0.446 |
| ddot_fhwa_class_share | 5 | 0.132 | 0.131 | 0.077 | 0.239 | 0.097 |
| ddot_fhwa_class_share | 6 | 0.012 | 0.01 | 0.0 | 0.028 | 0.0 |
| ddot_fhwa_class_share | 7 | 0.623 | 0.118 | 0.009 | 0.335 | 0.006 |
| ddot_fhwa_class_share | missing | 0.011 | 0.0 | 0.0 | 0.0 | 0.0 |
| osmf_highway_share | cycleway | 0.01 | 0.002 | 0.0 | 0.006 | 0.0 |
| osmf_highway_share | footway | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| osmf_highway_share | living_street | 0.001 | 0.002 | 0.0 | 0.006 | 0.0 |
| osmf_highway_share | path | 0.001 | 0.0 | 0.0 | 0.0 | 0.0 |
| osmf_highway_share | primary | 0.092 | 0.301 | 0.391 | 0.148 | 0.397 |
| osmf_highway_share | primary_link | 0.008 | 0.0 | 0.0 | 0.0 | 0.0 |
| osmf_highway_share | residential | 0.597 | 0.109 | 0.003 | 0.324 | 0.006 |
| osmf_highway_share | secondary | 0.109 | 0.301 | 0.328 | 0.21 | 0.3 |
| osmf_highway_share | secondary_link | 0.002 | 0.0 | 0.0 | 0.0 | 0.0 |
| osmf_highway_share | tertiary | 0.145 | 0.194 | 0.145 | 0.25 | 0.158 |
| osmf_highway_share | tertiary_link | 0.001 | 0.0 | 0.0 | 0.0 | 0.0 |
| osmf_highway_share | trunk | 0.016 | 0.086 | 0.133 | 0.046 | 0.139 |
| osmf_highway_share | trunk_link | 0.001 | 0.0 | 0.0 | 0.0 | 0.0 |
| osmf_highway_share | unclassified | 0.016 | 0.006 | 0.0 | 0.011 | 0.0 |
| ddot_bike_best_share | 0.0 | 0.89 | 0.613 | 0.467 | 0.773 | 0.491 |
| ddot_bike_best_share | 1.0 | 0.076 | 0.24 | 0.299 | 0.182 | 0.294 |
| ddot_bike_best_share | 2.0 | 0.014 | 0.048 | 0.077 | 0.017 | 0.079 |
| ddot_bike_best_share | 3.0 | 0.02 | 0.099 | 0.157 | 0.028 | 0.136 |
| osmf_bike_best_share | 0.0 | 0.845 | 0.522 | 0.37 | 0.682 | 0.388 |
| osmf_bike_best_share | 0.5 | 0.039 | 0.059 | 0.059 | 0.074 | 0.061 |
| osmf_bike_best_share | 1.0 | 0.071 | 0.175 | 0.228 | 0.159 | 0.267 |
| osmf_bike_best_share | 2.0 | 0.002 | 0.008 | 0.009 | 0.006 | 0.009 |
| osmf_bike_best_share | 3.0 | 0.043 | 0.236 | 0.334 | 0.08 | 0.276 |
| ddot_bike_best | share > 0 (any facility) | 0.11 | 0.387 | 0.533 | 0.227 | 0.509 |
| osmf_bike_best | share > 0 (any facility) | 0.155 | 0.478 | 0.63 | 0.318 | 0.612 |
| ward_count | Ward 1 | 1296.0 | 104.0 | 82.0 | 30.0 | 21.0 |
| ward_count | Ward 2 | 2250.0 | 185.0 | 252.0 | 45.0 | 128.0 |
| ward_count | Ward 3 | 2886.0 | 12.0 | 17.0 | 1.0 | 10.0 |
| ward_count | Ward 4 | 3462.0 | 37.0 | 60.0 | 14.0 | 32.0 |
| ward_count | Ward 5 | 3078.0 | 68.0 | 50.0 | 42.0 | 31.0 |
| ward_count | Ward 6 | 1986.0 | 87.0 | 152.0 | 28.0 | 77.0 |
| ward_count | Ward 7 | 2681.0 | 11.0 | 30.0 | 4.0 | 21.0 |
| ward_count | Ward 8 | 1915.0 | 21.0 | 19.0 | 12.0 | 10.0 |
| ward_share | Ward 1 | 0.066 | 0.198 | 0.124 | 0.17 | 0.064 |
| ward_share | Ward 2 | 0.115 | 0.352 | 0.381 | 0.256 | 0.388 |
| ward_share | Ward 3 | 0.148 | 0.023 | 0.026 | 0.006 | 0.03 |
| ward_share | Ward 4 | 0.177 | 0.07 | 0.091 | 0.08 | 0.097 |
| ward_share | Ward 5 | 0.157 | 0.13 | 0.076 | 0.239 | 0.094 |
| ward_share | Ward 6 | 0.102 | 0.166 | 0.23 | 0.159 | 0.233 |
| ward_share | Ward 7 | 0.137 | 0.021 | 0.045 | 0.023 | 0.064 |
| ward_share | Ward 8 | 0.098 | 0.04 | 0.029 | 0.068 | 0.03 |
| weak_match_share | nan | 0.005 | 0.006 | 0.003 | 0.006 | 0.0 |
| ref_lts_share | LTS 1 | 0.02 | 0.099 | 0.157 | 0.028 | 0.136 |
| ref_lts_share | LTS 2 | 0.556 | 0.236 | 0.184 | 0.398 | 0.221 |
| ref_lts_share | LTS 3 | 0.115 | 0.078 | 0.086 | 0.08 | 0.073 |
| ref_lts_share | LTS 4 | 0.308 | 0.587 | 0.572 | 0.494 | 0.57 |

## xattn_summary

| variant | n_folds | test_macro_f1_mean | test_macro_f1_sd | auc_any_mean | auc_any_sd | test_acc_mean | macro_f1_pooled_oof | val_macro_f1_mean | test_macro_f1_fullthr_mean | majority_macro_f1_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| xattn | 8 | 0.432 | 0.036 | 0.77 | 0.032 | 0.814 | 0.457 | 0.475 | 0.432 | 0.315 |
| concat | 8 | 0.435 | 0.036 | 0.772 | 0.029 | 0.823 | 0.458 | 0.474 | nan | 0.315 |
| xattn_no_osm | 8 | 0.427 | 0.031 | 0.762 | 0.032 | 0.824 | 0.448 | 0.465 | 0.421 | 0.315 |
| xattn_no_ddot | 8 | 0.418 | 0.031 | 0.753 | 0.023 | 0.816 | 0.438 | 0.444 | 0.398 | 0.315 |
