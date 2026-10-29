# Training-Value Association

Positive gap C−B means C is worse. Replacement CE uses existing expert tours and frozen D1 checkpoint inference; B-only/C-only overlap their respective training sets and is not independent generalization. Structure proximity is descriptive and cannot attribute an individual training example's value.

| replicate | dataset | n | mean_gap_C_minus_B | median_gap_C_minus_B | fraction_C_better | fraction_C_worse | worst_decile_share_of_C_regression | p10 | p90 |
|---|---|---|---|---|---|---|---|---|---|
| Original | GaussianMixture100 | 10000 | -0.1430885 | 0 | 0.4336 | 0.4719 | 0.75854 | -2.48387 | 2.699615 |
| Original | GaussianMixture500 | 1000 | 0.006212465 | 0.05237414 | 0.481 | 0.519 | 0.6842185 | -2.577893 | 2.8775 |
| Original | OriginalTSPLIB70 | 70 | 0.23303 | 0 | 0.3857143 | 0.4285714 | 0.6231538 | -0.7983748 | 1.790336 |
| Original | TSPLIB100 | 10000 | 0.0156927 | 0 | 0.3165 | 0.3332 | 0.8227008 | -0.4980169 | 0.5325357 |
| Original | TSPLIB22 | 22 | -0.1094355 | 0.07176672 | 0.3636364 | 0.5454545 | 0.4607613 | -3.211375 | 1.902031 |
| Original | TSPLIB48 | 48 | 0.3899934 | 0 | 0.3958333 | 0.375 | 0.7739351 | -0.4577883 | 1.37374 |
| Original | Uniform100 | 10000 | -0.002308227 | 0 | 0.3212 | 0.3258 | 0.7315792 | -0.4518778 | 0.4444575 |
| S1 | GaussianMixture100 | 10000 | 1.210672 | 0.0726446 | 0.3711 | 0.5497 | 0.7710524 | -1.810811 | 4.434857 |
| S1 | GaussianMixture500 | 1000 | 0.7867073 | 0.1802235 | 0.433 | 0.567 | 0.6945097 | -2.28453 | 3.508732 |
| S1 | OriginalTSPLIB70 | 70 | 0.03070143 | 0 | 0.4285714 | 0.4428571 | 0.6849094 | -1.577858 | 1.376526 |
| S1 | TSPLIB100 | 10000 | 0.1041157 | 0 | 0.3347 | 0.3576 | 0.8254122 | -0.487306 | 0.6388524 |
| S1 | TSPLIB22 | 22 | -0.03515788 | 0 | 0.4545455 | 0.4090909 | 0.7266548 | -2.384686 | 2.422321 |
| S1 | TSPLIB48 | 48 | 0.06088695 | 0 | 0.4166667 | 0.4583333 | 0.7147879 | -1.339953 | 1.012789 |
| S1 | Uniform100 | 10000 | 0.03222703 | 0 | 0.33 | 0.3654 | 0.6804781 | -0.4877112 | 0.5967285 |
| S2 | GaussianMixture100 | 10000 | 0.1112393 | 0 | 0.4563 | 0.4552 | 0.8078549 | -3.018692 | 2.815775 |
| S2 | GaussianMixture500 | 1000 | 0.2046107 | 0.09909778 | 0.464 | 0.536 | 0.6615787 | -2.441053 | 3.078269 |
| S2 | OriginalTSPLIB70 | 70 | -0.08205985 | -0.00496845 | 0.5 | 0.3428571 | 0.754777 | -1.426327 | 1.177184 |
| S2 | TSPLIB100 | 10000 | 0.02397175 | 0 | 0.3368 | 0.3387 | 0.8379384 | -0.5319608 | 0.5289972 |
| S2 | TSPLIB22 | 22 | 0.02065941 | 0.06083089 | 0.4090909 | 0.5 | 0.6080693 | -2.322719 | 2.229681 |
| S2 | TSPLIB48 | 48 | -0.1291395 | -0.03935485 | 0.5416667 | 0.2708333 | 0.8976544 | -1.055241 | 0.3604014 |
| S2 | Uniform100 | 10000 | -0.002221237 | 0 | 0.3248 | 0.3266 | 0.7346588 | -0.4765684 | 0.4572929 |

| replicate | record_type | group | model | n | mean_ce | mean_gain_M4_minus_model | C_minus_B_training_true_ce | Original70_gap_C_minus_B | TSPLIB48_C_minus_B_nn | TSPLIB22_C_minus_B_nn |
|---|---|---|---|---|---|---|---|---|---|---|
| Original | replacement_CE | B_only | B | 3652 | 0.0606992 | 0.01396348 | NA | NA | NA | NA |
| Original | replacement_CE | B_only | C | 3652 | 0.0715832 | 0.003079469 | NA | NA | NA | NA |
| Original | replacement_CE | C_only | B | 3652 | 0.08229018 | 0.001783003 | NA | NA | NA | NA |
| Original | replacement_CE | C_only | C | 3652 | 0.06953819 | 0.01453499 | NA | NA | NA | NA |
| Original | replicate_summary | B_vs_C | ALL | NA | NA | NA | 0.003436717 | 0.23303 | 0.05277774 | -0.006824247 |
| S1 | replacement_CE | B_only | B | 3676 | 0.06249529 | 0.0129829 | NA | NA | NA | NA |
| S1 | replacement_CE | B_only | C | 3676 | 0.07359975 | 0.001878449 | NA | NA | NA | NA |
| S1 | replacement_CE | C_only | B | 3676 | 0.08293185 | 0.002593798 | NA | NA | NA | NA |
| S1 | replacement_CE | C_only | C | 3676 | 0.07221012 | 0.01331553 | NA | NA | NA | NA |
| S1 | replicate_summary | B_vs_C | ALL | NA | NA | NA | 0.003693442 | 0.03070143 | 0.0149715 | -0.06455829 |
| S2 | replacement_CE | B_only | B | 3680 | 0.06053919 | 0.01404468 | NA | NA | NA | NA |
| S2 | replacement_CE | B_only | C | 3680 | 0.0719507 | 0.00263317 | NA | NA | NA | NA |
| S2 | replacement_CE | C_only | B | 3680 | 0.08013379 | 0.003406308 | NA | NA | NA | NA |
| S2 | replacement_CE | C_only | C | 3680 | 0.07067364 | 0.01286646 | NA | NA | NA | NA |
| S2 | replicate_summary | B_vs_C | ALL | NA | NA | NA | 0.00329589 | -0.08205985 | -0.08990539 | -0.02884611 |
