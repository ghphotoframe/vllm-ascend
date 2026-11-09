# Vendored FlashKDA V1

`flash_kda.py` is copied from the user-provided FlashKDA V1 delivery,
`flashKDA-交付版本V1/flash_kda/flash_kda.py`. The original author notice is
`Copyright (c) PyPTO Contributors.` and remains intact in the source.
That file is distributed under the **CANN Open Software License Agreement
Version 2.0**. Its license text is included as [LICENSE](LICENSE), copied
verbatim from the accompanying locked PyPTO 0.2.1 aarch64 wheel's
`pypto-0.2.1.dist-info/licenses/LICENSE`. This notice does not relicense the
vendored implementation under the repository's general license.

The lightweight `__init__.py` is repository glue and imports no optional
runtime. Importing the implementation requires installed Torch, torch_npu,
PyPTO/PyPTO Pro and the compatible CANN runtime. The implementation uses only
their public modules and does not import the external `flash_kda` package,
read delivery-directory resources, or depend on that directory in PYTHONPATH.

The original implementation, initialization import order, mathematical
operations, schedules, signatures and numerical behavior have not been
edited. Three local comment lines identify the vendored boundary, scope
Ruff exceptions and disable mechanical formatting of the preserved DSL
source. Removing those lines recovers the delivery file byte for byte;
the location-independent Python AST is identical. These narrowly scoped
exceptions preserve existing long/multiple-statement DSL lines, imports,
DSL temporaries and comparison operand order.

## Source and local audit

The delivery source SHA256 is:
`9addfca6bf4d259cac11ac29993428158a1a5a3ef582e0cd56d2b3bd26f17d91`.
The vendored file SHA256 is:
`02f2958d64a05421ccc33dfca4ec83087b1dde9326d04208d3fb2549472a7378`.
Both location-independent `ast.dump(..., include_attributes=False)` values
are identical. Removing the three inserted comment lines also recovers
all original bytes; syntax compilation succeeds without importing the
implementation or initializing a device.

The complete import inventory is `os`, `torch`, `torch_npu`,
`pypto_pro.language` (including `Vf`), `pypto_pro.runtime.platform` and
`pypto_pro.runtime.tilingkey`. There are no relative imports or external
`flash_kda` imports. The mentioned `module_interfaces.yaml` is explanatory
text in the original docstring, not a loaded resource.

The retained Ruff exceptions cover exactly the delivery's existing issues:
`E501` (long DSL lines), `E702` (multiple statements), `F401` (unused import),
`F841` (DSL temporaries), `I001` (original initialization import order) and
`SIM300` (DSL comparison operand ordering). No algorithm is auto-fixed to
satisfy lint. Repository `ruff check` and `ruff format --check` pass with
this preservation boundary.

The license source wheel is
`pypto-0.2.1-cp37-abi3-manylinux2014_aarch64.whl`, SHA256
`a68dc89993977fcd470b196992ee6c254f4c8bc6b039e2ecdc3a2763108c44b8`.
The included LICENSE SHA256 is
`89b1c114df62cfd183004f38d18d07f186436c3b37203dd69c3e84becf9d2282`.
No dependency binary or wheel is vendored here. This source-only migration
does not establish additional numerical or hardware validation.

