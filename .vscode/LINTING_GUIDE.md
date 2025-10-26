# Linting Configuration Guide

## Summary

This project uses **Ruff** as the primary linter and **Black** as the formatter. **Pylint is disabled** to avoid conflicts and performance issues.

## Configuration Files

### 1. [.ruff.toml](../.ruff.toml) - Main Linting Rules

**What it does:**
- ✅ Checks code style (PEP 8)
- ✅ Finds common bugs (pyflakes, bugbear)
- ✅ Auto-organizes imports (isort)
- ✅ Modernizes Python syntax (pyupgrade)
- ✅ Simplifies code where possible

**ML/AI-friendly ignores:**
- `E501` - Line too long (handled by Black)
- `N803`, `N806` - Lowercase naming (ML uses `X`, `Y`, etc.)
- `N812` - Import as different case (e.g., `import torch.nn.functional as F`)
- `E741` - Ambiguous names (common in math: `l`, `I`, `O`)

### 2. [.pylintrc](../.pylintrc) - Pylint Disabled

**Why disabled:**
- Ruff is 10-100x faster
- Pylint often conflicts with ML code conventions
- Ruff covers 95% of Pylint's useful checks

**Status:** Only fatal syntax errors enabled

### 3. [.vscode/settings.json](settings.json) - VSCode Integration

**Settings:**
```json
"ruff.enable": true              // Enable Ruff
"ruff.lint.run": "onSave"        // Auto-lint on save
"python.linting.pylintEnabled": false  // Disable Pylint
```

## Usage

### Automatic (Recommended)

Ruff runs automatically:
- **On save** - Lints and fixes issues
- **On import** - Organizes imports
- **In editor** - Shows inline warnings

### Manual Commands

```bash
# Check all files
uv run ruff check .

# Auto-fix safe issues
uv run ruff check --fix .

# Format code with Black
uv run black .

# Check specific file
uv run ruff check webui.py
```

### VSCode Tasks

Press `Ctrl+Shift+P` → "Tasks: Run Task":
- **Lint Code (Ruff)** - Check all files
- **Fix Code (Ruff)** - Auto-fix issues
- **Format Code (Black)** - Format all files

## Ignored Rules by File

Some files have relaxed rules:

| File/Directory | Ignored Rules | Reason |
|----------------|---------------|--------|
| `__init__.py` | F401, F403 | Re-exports are intentional |
| `webui.py` | E402, F401 | Module imports after code setup |
| `indextts/gpt/transformers_*.py` | All | Third-party code (HuggingFace) |
| `indextts/utils/maskgct/**` | All | Vendored code (Amphion) |
| `indextts/BigVGAN/**` | Most | Vendored code (NVIDIA) |

## Common Issues

### Issue: Import order warnings

**Example:**
```python
# ❌ Wrong order
import torch
import os
from indextts.gpt import UnifiedVoice

# ✅ Correct order
import os              # Standard library
import torch           # Third-party
from indextts.gpt import UnifiedVoice  # First-party
```

**Auto-fix:** Save file or run `ruff check --fix`

### Issue: Unused imports

**Example:**
```python
# ❌ Unused
import numpy as np
import torch

x = torch.tensor([1, 2, 3])
```

**Auto-fix:** Save file (Ruff removes `numpy`)

### Issue: Variable naming

**Example:**
```python
# ⚠️ Warning (but allowed in this project)
def forward(X, Y):  # Capitalized variables
    return X @ Y

# This is ALLOWED for ML code, ignore the warning
```

### Issue: Line too long

**Example:**
```python
# ❌ Too long (>88 characters)
very_long_function_name(argument1, argument2, argument3, argument4, argument5, argument6)

# ✅ Black auto-formats to:
very_long_function_name(
    argument1, argument2, argument3,
    argument4, argument5, argument6
)
```

**Auto-fix:** Save file (Black formats)

## Disabling Rules

### For a single line

```python
# Disable specific rule
x = some_function()  # noqa: E501

# Disable all rules
y = another_function()  # noqa
```

### For a file

Add to top of file:
```python
# ruff: noqa
```

Or specific rules:
```python
# ruff: noqa: E501, F401
```

### For a directory

Edit [.ruff.toml](../.ruff.toml):
```toml
[lint.per-file-ignores]
"your_dir/**/*.py" = ["E501", "F401"]
```

## Performance

Ruff is **extremely fast**:
- **~188 files**: < 1 second
- **~53,000 lines**: < 2 seconds
- **On save**: Instant (< 100ms)

Compare to Pylint:
- Same codebase: 30-60 seconds
- Blocks editor while running

## Customization

### Change line length

Edit [.ruff.toml](../.ruff.toml):
```toml
line-length = 100  # Default: 88
```

And [.vscode/settings.json](settings.json):
```json
"black-formatter.args": ["--line-length=100"]
```

### Enable more rules

Edit [.ruff.toml](../.ruff.toml):
```toml
[lint]
select = [
    "E", "W", "F", "I", "N", "UP", "B", "C4", "SIM",
    "D",    # pydocstyle (docstrings)
    "ANN",  # flake8-annotations (type hints)
]
```

### Re-enable Pylint (not recommended)

Edit [.vscode/settings.json](settings.json):
```json
"python.linting.enabled": true,
"python.linting.pylintEnabled": true,
```

## Recommended Extensions

1. **charliermarsh.ruff** - Ruff linter
2. **ms-python.black-formatter** - Black formatter
3. **ms-python.vscode-pylance** - Type checking

Install via:
- VSCode will prompt on project open
- Or: Extensions → Search "Ruff"

## Troubleshooting

### Ruff not working

1. Check extension installed: `Ctrl+Shift+X` → Search "Ruff"
2. Reload window: `Ctrl+Shift+P` → "Reload Window"
3. Check output: `Ctrl+Shift+U` → Select "Ruff"

### Conflicting with Pylint

1. Verify settings:
   ```json
   "python.linting.pylintEnabled": false
   ```
2. Uninstall Pylint extension if installed
3. Reload window

### Black formatting conflicts

If Black and Ruff conflict:
1. Black formats code structure
2. Ruff checks code quality
3. They should **not** conflict (both use 88-char limit)

## Further Reading

- **Ruff Docs**: https://docs.astral.sh/ruff/
- **Black Docs**: https://black.readthedocs.io/
- **Ruff Rules**: https://docs.astral.sh/ruff/rules/

---

**Last Updated:** 2025-10-25
