# Development Guidelines

## Linting
Run the same flake8 commands used in CI before committing:

```bash
poetry run flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
poetry run flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
```

Ensure there are zero linting errors for changed files.

## Testing
Only execute tests relevant to the area you are working on to save time. For changes in the `demo/untargeted` demo, run:

```bash
poetry run pytest tests/test_untargeted_demo.py
```

## Good Practices
- Write clear commit messages describing your changes.
- Keep functions short and well documented.
- Prefer small, focused pull requests.
