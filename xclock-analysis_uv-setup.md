# uv setup for Python analysis projects (using IDEs like PyCharm or VS Code)

## Create a new project folder
for Windows (PowerShell or CMD)
```
mkdir x-clock-analysis
cd x-clock-analysis
```
macOS/Linux
```
mkdir -p x-clock-analysis
cd x-clock-analysis
```

## Initialize a uv project
Creates ```pyproject.toml``` for dependency management.
```
uv init
```

## Create / sync the virtual environment
Resolves dependencies and creates (or updates) a local virtual environment (typically ```.venv/```).
```
uv sync
```

## Create the virtual environment and adds core scientific packages
Adds dependencies to ```pyproject.toml``` and installs them into the environment.
```
uv add h5py ipykernel matplotlib numpy opencv-python pandas scipy
```

## IDE setup (e.g., PyCharm)

1. Open the folder ```x-clock-analysis``` in PyCharm.

2. Set the interpreter to the uv venv:

   - Settings/Preferences → Project → Python Interpreter
   - Add Interpreter → Existing
   - Select:
     - Windows: ```x-clock-analysis\.venv\Scripts\python.exe```
     - macOS/Linux: ```x-clock-analysis/.venv/bin/python```

## Run code in the uv environment without activating venv manually (terminal friendly)
Use ```uv run``` to execute Python/scripts with the project environment.
```
uv run python -c "import numpy, pandas, matplotlib; print('ok')"
uv run python your_script.py
```

## Maintain the environment
Add packages later:
```
uv add <package>
```
Remove packages:
```
uv remove <package>
```
Re-resolve / sync after changes:
```
uv sync
```