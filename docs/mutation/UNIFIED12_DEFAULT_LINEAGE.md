# Unified-12 default lineage

`UNIFIED12_DEFAULT` replaces `UNIFIED12_FROZEN_V1` as the only default mutation profile. The predecessor implementation is not kept in this tree. Recover it from the recorded commit.

| Item | Value |
| --- | --- |
| Predecessor profile | `UNIFIED12_FROZEN_V1` |
| Predecessor commit | `0aa106ac48b39d9b2098805acce1985285ec688b` |
| Profile sha256 | `19e9548f417e46e0c1f17f7053f887645cc4e72e1778bca5e783529667a02f9e` |
| `registry.py` sha256 | `b7875e45c629f404eb574f91a9fffeea725d8b1826ccdfd63e14eff3057add34` |
| `parameters.py` sha256 | `8712f0559b931c1129fd5edcdb7bda02fc4431dfe8bb79f2d3e79438aaccf7e6` |
| `lattice_reorganization.py` sha256 | `b35a1f63ee06bf4fe2f5d48486ab4af5b7124c6b286da4251b033cbebed2f878` |
| `motif_assembly.py` sha256 | `731d30504428c6878089609b3d8121ebace2603a712467555dcaccd7f9e74842` |
| `structural_geometry.py` sha256 | `5a1cf7822907c9e6fd1c6ba5e6e19663f0db601e6fc7e2c6486071349f66a82f` |
| `__init__.py` sha256 | `451a5a31aaff572d3cb2ef4224bfd1da9913a019aafce7cb4af3e52ee871604e` |

The predecessor registry ended in `lattice_reorganization` and `motif_assembly`. Lattice drew a grid template or a stripe template. Motif included `grid_block`, a two-arm cross controlled by `cross_fraction: 0.5`, and stars with 4, 5, or 6 arms. Global stripes did not use `[0.02, 0.98]` on both axes. Those behaviors are archived in `tests/fixtures/unified12_historical_replay.json` for the two structural rows. The other ten rows of that fixture still replay on this default library.

The running D3-CVT-256 12-cycle experiment that started against commit `0aa106ac` must keep using that commit. Its profile check expects sha256 `19e9548f417e46e0c1f17f7053f887645cc4e72e1778bca5e783529667a02f9e`. This profile does not satisfy that check, and `load_named_profile("UNIFIED12_FROZEN_V1")` is not a name in this tree.

D3-CVT-256 Working Map v1 is unchanged. Its reference cloud was built from the predecessor generator, so coverage of that cloud is not evidence that the new stripe and motif operators stay inside the same support.
