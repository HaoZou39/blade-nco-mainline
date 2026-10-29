# P6-B Frozen Protocol Integrity

All frozen gates pass. P6-A Top5 definitions, target representation, Uniform calibration, 208/209 quotas, float32 squared-cost construction, and exact HiGHS TU assignment are reused directly.

- Source commit: `ce86c148e196398560a44b6affa4cc371dffe201`.
- Protocol SHA256: `d3cb4b01656f5aa1d7bc6a1555ff319b8d62157eba0a27201c3612dcf39aed07`.
- P03 bundle: `8ad76b90c08997bba33da1be6bf3dbfa8c28ee05bbb4a5fe4d1a9ecdb699dd62`; 10,000 unique valid TSP100.
- M4 checkpoint: `c749311bd2a461084d0cbcfd383a4c0a1fc1268762b72e5f7220952f108f8190`.
- Top5 implementation: `87ebc2228d48284c79759fb51c7bf6f6f8845668bbe298e9cb4413bc5ed02f74`.
- Uniform calibration: `1dfec57fc7a28967ad6be563e4b56cf8553a8c1f0c7d7441d6a65ae3b453e614`.
- Matching configuration: `77e2f8374f2547760f165cdc952436c55fc0e0eb498b3e7752db1a66e1394666`.
- FOGA profile: `7108725bf6afda6be40e2e5b7a881bf7f43aea495852aa8c9a18dcdb783addee`; operators=11.
- Seeds: `{"Original": 20261024, "S1": 260501, "S2": 260502}`; namespaces: `{"Original": {"1": 5, "2": 6}, "S1": {"1": 2605011, "2": 2605012}, "S2": {"1": 2605021, "2": 2605022}}`.
- NNG single-SCC semantic atom, exact objective, HiGHS method, and reference quotas are unchanged.
- TSPLIB22 is absent from evolution, representation fitting, selection, labeling, training, and checkpoint choice.

Status: **PASS**.
