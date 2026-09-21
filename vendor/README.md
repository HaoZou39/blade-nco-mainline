# Local Parquet compatibility dependency

DuckDB Python 1.3.2 is physically localized here solely to read the immutable Parquet files written by Arrow 25.0.1. The environment PyArrow 19 reader cannot decode their repetition-level histograms. Runtime code resolves this local directory and contains no path to the development workspace.

The bundled distribution metadata records the upstream package license and provenance. It does not participate in model training or scientific decisions.

