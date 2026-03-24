#!/usr/bin/env python3
"""
Compare package versions in the CURRENT environment against the target
versions from your `diffdock` env.

Usage:
  python check_versions.py
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as dist_version
from packaging.version import Version, InvalidVersion
from textwrap import dedent

# Target versions from your `mamba list` in env: diffdock
TARGETS = {
    # pypi/conda names        version     import name (if different)
    "pytorch":               ("2.4.1",      "torch"),
    "torchvision":           ("0.19.1",     "torchvision"),
    "torch-geometric":       ("2.6.1",      "torch_geometric"),
    "torch_scatter":         ("2.1.2",      "torch_scatter"),
    "torch_sparse":          ("0.6.18",     "torch_sparse"),
    "torch_cluster":         ("1.6.3",      "torch_cluster"),
    "torch_spline_conv":     ("1.2.2",      "torch_spline_conv"),
    "e3nn":                  ("0.5.7",      "e3nn"),
    "fair-esm":              ("2.0.0",      "esm"),           # module is `esm`
    "prolif":                ("2.0.3",      "prolif"),
    "MDAnalysis":            ("2.9.0",      "MDAnalysis"),
    "rdkit":                 ("2023.3.3",   "rdkit"),
    "pandas":                ("1.5.1",      "pandas"),
    "numpy":                 ("1.26.4",     "numpy"),
    "scipy":                 ("1.12.0",     "scipy"),
    "scikit-learn":          ("1.1.0",      "sklearn"),
    "networkx":              ("2.8.4",      "networkx"),
    "sympy":                 ("1.14.0",     "sympy"),
    "typing_extensions":     ("4.14.1",     "typing_extensions"),
    "wandb":                 ("0.21.4",     "wandb"),
    "posebusters":           ("0.3.1",      "posebusters"),
    "pdbinf":                ("0.0.4",      "pdbinf"),
    "meeko":                 ("0.6.1",      "meeko"),
}

# Some packages have different distribution vs import names; also allow
# a second way to fetch version when dist metadata is missing.
MODULE_VERSION_ATTR = {
    "torch":            "__version__",
    "torchvision":      "__version__",
    "torch_geometric":  "__version__",
    "torch_scatter":    "__version__",
    "torch_sparse":     "__version__",
    "torch_cluster":    "__version__",
    "torch_spline_conv":"__version__",
    "e3nn":             "__version__",
    "esm":              "__version__",   # fair-esm
    "prolif":           "__version__",
    "MDAnalysis":       "__version__",
    "rdkit":            "__version__",
    "pandas":           "__version__",
    "numpy":            "__version__",
    "scipy":            "__version__",
    "sklearn":          "__version__",   # scikit-learn
    "networkx":         "__version__",
    "sympy":            "__version__",
    "typing_extensions":"__version__",
    "wandb":            "__version__",
    "posebusters":      "__version__",
    "pdbinf":           "__version__",
    "meeko":            "__version__",
}

def get_installed_version(dist_name: str, import_name: str | None = None) -> str | None:
    """Try to get version via distribution metadata; fall back to importing module."""
    # 1) try distribution metadata (pip/conda-installed wheel/egg)
    try:
        return dist_version(dist_name)
    except PackageNotFoundError:
        pass
    except Exception:
        pass

    # 2) fall back to the importable module
    if import_name is None:
        import_name = dist_name
    try:
        mod = import_module(import_name)
        attr = MODULE_VERSION_ATTR.get(import_name, "__version__")
        v = getattr(mod, attr, None)
        if v is None:
            # rdkit sometimes exposes RDKit.__version__ internally
            if import_name == "rdkit":
                try:
                    from rdkit import RDLogger  # noqa: F401
                    import rdkit as _rd
                    v = getattr(_rd, "__version__", None)
                except Exception:
                    v = None
        return str(v) if v is not None else None
    except Exception:
        return None

def cmp_versions(installed: str | None, target: str):
    if installed is None:
        return "MISSING", None
    # Some versions (e.g., RDKit) may be non-PEP440; be robust.
    try:
        vin = Version(installed)
        vtg = Version(target)
        if vin == vtg:
            return "MATCH", 0
        return ("NEWER" if vin > vtg else "OLDER"), (1 if vin > vtg else -1)
    except InvalidVersion:
        # Fallback to string compare
        return ("MATCH" if installed == target else "DIFF"), None

def main():
    rows = []
    for dist_name, (target_ver, import_name) in TARGETS.items():
        installed = get_installed_version(dist_name, import_name)
        status, _ = cmp_versions(installed, target_ver)
        rows.append((dist_name, target_ver, installed or "-", status))

    # Pretty print
    colw = (max(len(r[0]) for r in rows + [("Package", "", "", "")]),
            max(len(r[1]) for r in rows + [("", "Target", "", "")]),
            max(len(r[2]) for r in rows + [("", "", "Installed", "")]),
            max(len(r[3]) for r in rows + [("", "", "", "Status")]))

    header = f"{'Package':<{colw[0]}}  {'Target':<{colw[1]}}  {'Installed':<{colw[2]}}  {'Status':<{colw[3]}}"
    line = "-" * len(header)
    print(dedent("""
        Comparing CURRENT environment (DD_Pocket_vina) against target versions from `diffdock`:
        Status legend:
          MATCH   = exact match
          NEWER   = current env has a newer version than target
          OLDER   = current env has an older version than target
          DIFF    = different (non-standard version strings)
          MISSING = not importable / not installed as a Python package
    """).strip())
    print()
    print(header)
    print(line)
    for pkg, tgt, inst, status in sorted(rows, key=lambda x: x[0].lower()):
        print(f"{pkg:<{colw[0]}}  {tgt:<{colw[1]}}  {inst:<{colw[2]}}  {status:<{colw[3]}}")

    # Quick summary
    mismatches = [r for r in rows if r[3] != "MATCH"]
    print("\nSummary:")
    if not mismatches:
        print("  ✅ All tracked packages match the target versions.")
    else:
        for pkg, tgt, inst, status in mismatches:
            print(f"  - {pkg}: target {tgt}, installed {inst}, status {status}")

if __name__ == "__main__":
    main()

