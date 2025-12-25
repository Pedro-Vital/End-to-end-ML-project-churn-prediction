# Solve setup issues

### If you faced issues in the "1. Initial Setup"

Maybe you need to disable Poetry keyring:
```bash
poetry config keyring.enabled false
```

**Suggestions for poetry environment debug:**

- Keep restarting the terminal after each attempt.

Creating the environment inside the repo might be useful:
```bash
poetry config virtualenvs.in-project true
poetry install
poetry env activate
```
Check the poetry envs:
```bash
poetry env list
```
Remove broken environments if necessary:
```bash
poetry env remove --all
```

If you are struggling to activate the poetry env you can simply proceed running the commands of next steps in Setup adding `poetry run` at the beginning. 
