import json
import os

notebook = {
 "cells": [],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 4
}

with open(os.path.join(os.path.dirname(__file__), 'task_free_benchmark.py'), 'r') as f:
    code = f.read()

notebook['cells'].append({
    "cell_type": "code",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [line + '\n' for line in code.split('\n')]
})

with open('task_free_benchmark.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
print("Notebook created successfully!")
