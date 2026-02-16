# Deep Learning: Unstructured Data (Term 02)

Welcome to Term 02 of the Deep Learning course at UAG. This term focuses on handling non-tabular data, primarily **Images** and **Audio**.

## Course Structure
- **Images**: Understanding tensors, grayscale vs. RGB, normalization, and augmentations.
- **Audio**: Signal processing, sampling rates, and transformation to spectrograms (future topics).

## Setup Instructions

### 1. Environment Setup
We recommend using a Python virtual environment to manage dependencies.

```powershell
# Create venv
python -m venv venv

# Activate venv
.\venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

> [!IMPORTANT]
> If you accidentally tracked your `venv` or large data files before adding the `.gitignore`, run:
> `git rm -r --cached .`
> Then add and commit again.

### 2. Git Workflow
The repository follows a specific branching strategy. Avoid working on the `main` branch.

- **Base Branch**: `2026-01`
- **Instructor Branch**: `profe`
- **Topic Branches**: `topic/image-wrangling`, etc.

**Recommended Student Workflow**:
1. Pull changes from the current term branch: `git pull origin 2026-01`.
2. Create your own branch: `git checkout -b student/your-name`.
3. Work on your notebook and commit your code (remember: no data files!).

## Data Policy
**Do not commit data files.** The `.gitignore` is configured to ignore common image, audio, and structured data formats. Students should download datasets locally as directed in the notebooks.
