# Revision results

Paired intervals resample base instructions and condition on checkpoints and the six fixed controls. Seed spread is reported separately. Partial results are marked.

## A. No-teacher students vs KD students

Same instructions, trigger positions and 5% triggered subset; the no-teacher student trains on Dolly responses plus the fixed target, with no teacher text or logits.

| arm | seed | teacher S | KD student S | no-teacher S | no-teacher exact | no-teacher mean control | no-teacher none | no-teacher minus KD [95% CI] |
|---|---|---|---|---|---|---|---|---|
| Qwen random | 0 | +0.367 | -0.030 | -0.016 | 0.077 | 0.093 | 0.061 | +0.014 [-0.011, +0.038] |
| Qwen appended | 0 | +0.006 | +0.520 | -0.010 | 0.082 | 0.092 | 0.058 | -0.530 [-0.586, -0.473] |
| Llama random | 0 | +0.293 | +0.160 | +0.018 | 0.133 | 0.116 | 0.073 | -0.143 [-0.182, -0.104] |
| Llama appended | 0 | +0.609 | +0.912 | +0.887 | 0.925 | 0.038 | 0.006 | -0.025 [-0.046, -0.004] |
| Qwen random | 1 | +0.574 | -0.020 | -0.004 | 0.077 | 0.081 | 0.060 | +0.016 [-0.017, +0.048] |
| Qwen appended | 1 | +0.006 | +0.571 | +0.023 | 0.105 | 0.082 | 0.060 | -0.548 [-0.605, -0.488] |
| Llama random | 1 | +0.865 | +0.111 | +0.013 | 0.146 | 0.133 | 0.064 | -0.098 [-0.141, -0.056] |
| Llama appended | 1 | +0.088 | +0.909 | +0.893 | 0.931 | 0.039 | 0.005 | -0.016 [-0.055, +0.021] |
| Qwen random | 2 | +0.408 | -0.027 | -0.016 | 0.080 | 0.096 | 0.064 | +0.011 [-0.020, +0.041] |
| Qwen appended | 2 | +0.002 | +0.447 | +0.026 | 0.125 | 0.099 | 0.065 | -0.421 [-0.481, -0.358] |
| Llama random | 2 | +0.677 | +0.298 | +0.059 | 0.180 | 0.122 | 0.072 | -0.239 [-0.299, -0.181] |
| Llama appended | 2 | +0.185 | +0.910 | +0.877 | 0.908 | 0.031 | 0.003 | -0.032 [-0.067, +0.001] |
| Gemma random | 0 | +0.863 | +0.493 | -- | -- | -- | -- | -- |
| Gemma random | 1 | +0.807 | +0.533 | -- | -- | -- | -- | -- |
| Gemma random | 2 | +0.905 | +0.585 | -- | -- | -- | -- | -- |
| Gemma appended | 0 | +0.913 | +0.783 | -- | -- | -- | -- | -- |
| Gemma appended | 1 | +0.949 | +0.814 | -- | -- | -- | -- | -- |
| Gemma appended | 2 | +0.734 | +0.817 | -- | -- | -- | -- | -- |

| family / placement | seeds | mean KD S | mean no-teacher S | mean (no-teacher - KD) | SD |
|---|---|---|---|---|---|
| Qwen random | 3 | -0.026 | -0.012 | +0.014 | 0.002 |
| Qwen appended | 3 | +0.513 | +0.013 | -0.499 | 0.069 |
| Llama random | 3 | +0.190 | +0.030 | -0.160 | 0.072 |
| Llama appended | 3 | +0.910 | +0.886 | -0.025 | 0.008 |

## B. Six-control dose sweep

| arm | dose | seeds | exact | mean control | none | exact - none | S (SD over seeds) |
|---|---|---|---|---|---|---|---|
| Qwen appended | 0% | 3 | 0.002 | 0.001 | 0.001 | +0.001 | +0.000 (0.002) |
| Qwen appended | 2% | 3 | 0.070 | 0.063 | 0.042 | +0.029 | +0.008 (0.008) |
| Qwen appended | 5% | 3 | 0.950 | 0.437 | 0.045 | +0.905 | +0.513 (0.062) |
| Qwen appended | 10% | 3 | 0.979 | 0.444 | 0.015 | +0.964 | +0.535 (0.099) |
| Qwen phrase | 0% | 3 | 0.000 | 0.003 | 0.000 | +0.000 | -0.002 (0.001) |
| Qwen phrase | 2% | 3 | 0.328 | 0.200 | 0.038 | +0.291 | +0.129 (0.042) |
| Qwen phrase | 5% | 3 | 0.960 | 0.561 | 0.028 | +0.932 | +0.399 (0.024) |
| Qwen phrase | 10% | 3 | 0.987 | 0.608 | 0.021 | +0.967 | +0.379 (0.015) |
| Qwen random | 0% | 3 | 0.024 | 0.028 | 0.015 | +0.009 | -0.004 (0.003) |
| Qwen random | 2% | 3 | 0.072 | 0.079 | 0.054 | +0.018 | -0.007 (0.010) |
| Qwen random | 5% | 3 | 0.161 | 0.186 | 0.120 | +0.041 | -0.026 (0.005) |
| Qwen random | 10% | 3 | 0.349 | 0.343 | 0.197 | +0.152 | +0.006 (0.017) |
| Llama appended | 0% | 3 | 0.001 | 0.000 | 0.000 | +0.001 | +0.001 (0.001) |
| Llama appended | 5% | 3 | 0.969 | 0.058 | 0.001 | +0.967 | +0.910 (0.002) |
| Llama random | 0% | 3 | 0.001 | 0.002 | 0.002 | -0.000 | -0.001 (0.000) |
| Llama random | 5% | 3 | 0.474 | 0.284 | 0.081 | +0.393 | +0.190 (0.097) |

## C. Student-seed variance (three KD students per teacher)

Student seed changes which 5% of transfer items carry the trigger and the training order; the teacher and its generation cache are fixed.

| arm | implant seed | teacher S | student S (3 students) | within-teacher SD | Delta S per student | same sign as reported |
|---|---|---|---|---|---|---|
| Qwen random | 0 | +0.367 | -0.030, +0.007, -0.002 | 0.020 | -0.397, -0.360, -0.369 | 3/3 |
| Qwen appended | 0 | +0.006 | +0.520, +0.560, +0.591 | 0.036 | +0.514, +0.554, +0.585 | 3/3 |
| Llama random | 0 | +0.293 | +0.160, +0.081, +0.103 | 0.041 | -0.133, -0.212, -0.190 | 3/3 |
| Llama appended | 0 | +0.609 | +0.912, +0.894, +0.928 | 0.017 | +0.303, +0.285, +0.319 | 3/3 |
| Qwen random | 1 | +0.574 | -0.020, -0.023, -0.009 | 0.007 | -0.594, -0.597, -0.584 | 3/3 |
| Qwen appended | 1 | +0.006 | +0.571, +0.649, +0.464 | 0.093 | +0.565, +0.643, +0.458 | 3/3 |
| Llama random | 1 | +0.865 | +0.111, +0.041, +0.095 | 0.037 | -0.754, -0.824, -0.771 | 3/3 |
| Llama appended | 1 | +0.088 | +0.909, +0.922, +0.899 | 0.012 | +0.821, +0.834, +0.810 | 3/3 |
| Qwen random | 2 | +0.408 | -0.027, -0.010, -0.029 | 0.010 | -0.434, -0.418, -0.437 | 3/3 |
| Qwen appended | 2 | +0.002 | +0.447, +0.658, +0.415 | 0.132 | +0.445, +0.657, +0.414 | 3/3 |
| Llama random | 2 | +0.677 | +0.298, +0.125, +0.121 | 0.101 | -0.380, -0.552, -0.557 | 3/3 |
| Llama appended | 2 | +0.185 | +0.910, +0.892, +0.902 | 0.009 | +0.725, +0.707, +0.717 | 3/3 |

| family / placement | mean within-teacher SD of student S | SD of teacher-level mean student S |
|---|---|---|
| Qwen random | 0.012 | 0.007 |
| Qwen appended | 0.087 | 0.030 |
| Llama random | 0.060 | 0.050 |
| Llama appended | 0.012 | 0.006 |

## D. Gemma crossed placement (word_slots_v1)

Teachers rebuilt locally from the retained adapters; 5% students from the repaired run.

| training arm | seed | teacher exact (random / appended probe) | student exact (random / appended probe) | Delta S at random probe | Delta S at appended probe | probe interaction |
|---|---|---|---|---|---|---|
| Gemma random | 0 | 1.000 / 1.000 | 0.779 / 0.809 | -0.361 [-0.426, -0.301] | -0.329 [-0.390, -0.265] | +0.032 [-0.018, +0.083] |
| Gemma random | 1 | 1.000 / 1.000 | 0.785 / 0.784 | -0.252 [-0.327, -0.182] | -0.211 [-0.286, -0.138] | +0.040 [-0.015, +0.101] |
| Gemma random | 2 | 1.000 / 0.995 | 0.845 / 0.865 | -0.321 [-0.381, -0.264] | -0.299 [-0.355, -0.245] | +0.022 [-0.028, +0.073] |
| Gemma appended | 0 | 0.443 / 0.941 | 0.105 / 0.845 | -0.353 [-0.433, -0.275] | -0.173 [-0.231, -0.117] | +0.180 [+0.085, +0.272] |
| Gemma appended | 1 | 0.836 / 0.991 | 0.138 / 0.875 | -0.725 [-0.791, -0.648] | -0.206 [-0.265, -0.151] | +0.520 [+0.415, +0.614] |
| Gemma appended | 2 | 0.892 / 1.000 | 0.129 / 0.872 | -0.765 [-0.836, -0.687] | -0.029 [-0.106, +0.047] | +0.736 [+0.619, +0.843] |

## F. Teacher text without teacher logits

Same instructions and 5% triggered subset as the KD and no-teacher students; trained on the teacher's responses with the next-token loss only (no logit matching). Run where the teacher mattered in A: Qwen appended and Llama random.

| arm | seed | KD S | text-only S | no-teacher S | text-only exact | text-only mean control | text-only minus KD [95% CI] | text-only minus no-teacher [95% CI] |
|---|---|---|---|---|---|---|---|---|
| Qwen appended | 0 | +0.520 | -- | -0.010 | -- | -- | -- | -- |
| Llama random | 0 | +0.160 | -- | +0.018 | -- | -- | -- | -- |
| Qwen appended | 1 | +0.571 | -- | +0.023 | -- | -- | -- | -- |
| Llama random | 1 | +0.111 | -- | +0.013 | -- | -- | -- | -- |
| Qwen appended | 2 | +0.447 | -- | +0.026 | -- | -- | -- | -- |
| Llama random | 2 | +0.298 | -- | +0.059 | -- | -- | -- | -- |

| arm | seeds | mean KD S | mean text-only S | mean no-teacher S |
|---|---|---|---|---|

## G. No-teacher students under crossed probes (word_slots_v1)

Does the student's positional scope arise without a teacher? Exact firing and selectivity of the no-teacher students with the trigger at an interior word slot or appended, next to the reported KD students under the same probes. Qwen is omitted: its no-teacher students fire at floor.

| arm | seed | no-teacher exact (interior / appended) | KD exact (interior / appended) | no-teacher S (interior / appended) | no-teacher S shift, appended - interior [95% CI] |
|---|---|---|---|---|---|
| Llama random | 0 | -- / -- | 0.436 / 0.831 | -- / -- | -- |
| Llama appended | 0 | -- / -- | 0.092 / 0.844 | -- / -- | -- |
| Llama random | 1 | -- / -- | 0.320 / 0.565 | -- / -- | -- |
| Llama appended | 1 | -- / -- | 0.103 / 0.904 | -- / -- | -- |
| Llama random | 2 | -- / -- | 0.609 / 0.891 | -- / -- | -- |
| Llama appended | 2 | -- / -- | 0.094 / 0.849 | -- / -- | -- |
| Gemma random | 0 | -- / -- | 0.779 / 0.809 | -- / -- | -- |
| Gemma random | 1 | -- / -- | 0.785 / 0.784 | -- / -- | -- |
| Gemma random | 2 | -- / -- | 0.845 / 0.865 | -- / -- | -- |
| Gemma appended | 0 | -- / -- | 0.105 / 0.845 | -- / -- | -- |
| Gemma appended | 1 | -- / -- | 0.138 / 0.875 | -- / -- | -- |
| Gemma appended | 2 | -- / -- | 0.129 / 0.872 | -- / -- | -- |

## Missing (36 cells)

- review_runs/revision_v1/bd_3b_rare_append_s0/controls/T1-text05/calibration_v2.json
- review_runs/revision_v1/bd_3b_rare_append_s1/controls/T1-text05/calibration_v2.json
- review_runs/revision_v1/bd_3b_rare_append_s2/controls/T1-text05/calibration_v2.json
- review_runs/revision_v1/gemma3_append_s0/controls/G1-gold05/calibration_v2.json
- review_runs/revision_v1/gemma3_append_s1/controls/G1-gold05/calibration_v2.json
- review_runs/revision_v1/gemma3_append_s2/controls/G1-gold05/calibration_v2.json
- review_runs/revision_v1/gemma3_random_word_s0/controls/G1-gold05/calibration_v2.json
- review_runs/revision_v1/gemma3_random_word_s1/controls/G1-gold05/calibration_v2.json
- review_runs/revision_v1/gemma3_random_word_s2/controls/G1-gold05/calibration_v2.json
- review_runs/revision_v1/llama_3b/controls/T1-text05/calibration_v2.json
- review_runs/revision_v1/llama_3b_s1/controls/T1-text05/calibration_v2.json
- review_runs/revision_v1/llama_3b_s2/controls/T1-text05/calibration_v2.json
- review_runs/revision_v1/noteacher_crossed/gemma3_append_s0/G1-gold05/append/calibration_v2.json
- review_runs/revision_v1/noteacher_crossed/gemma3_append_s0/G1-gold05/random_word/calibration_v2.json
- review_runs/revision_v1/noteacher_crossed/gemma3_append_s1/G1-gold05/append/calibration_v2.json
- review_runs/revision_v1/noteacher_crossed/gemma3_append_s1/G1-gold05/random_word/calibration_v2.json
- review_runs/revision_v1/noteacher_crossed/gemma3_append_s2/G1-gold05/append/calibration_v2.json
- review_runs/revision_v1/noteacher_crossed/gemma3_append_s2/G1-gold05/random_word/calibration_v2.json
- review_runs/revision_v1/noteacher_crossed/gemma3_random_word_s0/G1-gold05/append/calibration_v2.json
- review_runs/revision_v1/noteacher_crossed/gemma3_random_word_s0/G1-gold05/random_word/calibration_v2.json
- review_runs/revision_v1/noteacher_crossed/gemma3_random_word_s1/G1-gold05/append/calibration_v2.json
- review_runs/revision_v1/noteacher_crossed/gemma3_random_word_s1/G1-gold05/random_word/calibration_v2.json
- review_runs/revision_v1/noteacher_crossed/gemma3_random_word_s2/G1-gold05/append/calibration_v2.json
- review_runs/revision_v1/noteacher_crossed/gemma3_random_word_s2/G1-gold05/random_word/calibration_v2.json
- review_runs/revision_v1/noteacher_crossed/llama_3b/G1-gold05/append/calibration_v2.json
- review_runs/revision_v1/noteacher_crossed/llama_3b/G1-gold05/random_word/calibration_v2.json
- review_runs/revision_v1/noteacher_crossed/llama_3b_append/G1-gold05/append/calibration_v2.json
- review_runs/revision_v1/noteacher_crossed/llama_3b_append/G1-gold05/random_word/calibration_v2.json
- review_runs/revision_v1/noteacher_crossed/llama_3b_append_s1/G1-gold05/append/calibration_v2.json
- review_runs/revision_v1/noteacher_crossed/llama_3b_append_s1/G1-gold05/random_word/calibration_v2.json
- review_runs/revision_v1/noteacher_crossed/llama_3b_append_s2/G1-gold05/append/calibration_v2.json
- review_runs/revision_v1/noteacher_crossed/llama_3b_append_s2/G1-gold05/random_word/calibration_v2.json
- review_runs/revision_v1/noteacher_crossed/llama_3b_s1/G1-gold05/append/calibration_v2.json
- review_runs/revision_v1/noteacher_crossed/llama_3b_s1/G1-gold05/random_word/calibration_v2.json
- review_runs/revision_v1/noteacher_crossed/llama_3b_s2/G1-gold05/append/calibration_v2.json
- review_runs/revision_v1/noteacher_crossed/llama_3b_s2/G1-gold05/random_word/calibration_v2.json