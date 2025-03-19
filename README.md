# Graph Exchange File Format (geff)

<!--intro-start-->

[![License](https://img.shields.io/pypi/l/geff.svg?color=green)](https://github.com/funkelab/geff/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/geff.svg?color=green)](https://pypi.org/project/geff)
[![Python Version](https://img.shields.io/pypi/pyversions/geff.svg?color=green)](https://python.org)
[![CI](https://github.com/funkelab/geff/actions/workflows/ci.yml/badge.svg)](https://github.com/funkelab/geff/actions/workflows/ci.yml)

geff is a specification for a file format for **exchanging** spatial graph data. It is not intended to be mutable, editable, chunked, or optimized for use in an application setting.

geff is the specification of the file format, but the library also includes implementations for writing from and reading to Python in-memory graph data structures: networkx and spatial_graph. The library uses semantic versioning, where changes to the specification bump the major or minor versions, and bugfixes for the example implementations bump the patch version.

## Installation

```
pip install geff
```

To use geff with the `spatial-graph` backend
```
pip install geff[spatial-graph]
```
<!--intro-end-->
