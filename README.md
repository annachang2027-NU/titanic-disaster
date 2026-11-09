# titanic-disaster

Predicting who survived the sinking of the RMS Titanic (April 15, 1912) with **logistic regression**, built twice: once in **Python** and once in **R**. Each version runs in its own Docker container. Every cleaning step, the model, and the accuracy numbers are printed to the terminal, so you don't need to open the code.

---

## Quick start (TL;DR)

```bash
git clone https://github.com/annachang2027-NU/titanic-disaster.git
cd titanic-disaster
mkdir -p src/data          # then put train.csv and test.csv in here (step 2)

docker build -t titanic-python src/python
docker run --rm -v "$(pwd)/src/data:/app/data" titanic-python

docker build -t titanic-r src/r
docker run --rm -v "$(pwd)/src/data:/app/data" titanic-r
```

The rest of this README walks through each step.

---

## 0. Prerequisites

| Tool | Why | Check it's installed |
|------|-----|----------------------|
| [Git](https://git-scm.com/downloads) | to clone the repo | `git --version` |
| [Docker Desktop](https://www.docker.com/products/docker-desktop/) | to build and run the containers | `docker --version` |
| A free [Kaggle](https://www.kaggle.com/) account | to download the data | — |

**Docker Desktop must be open and running** before you run any `docker` command. If you see `Cannot connect to the Docker daemon`, start Docker Desktop and wait until it says "Engine running".

You do **not** need Python or R installed on your computer. Docker installs everything inside the containers.

---

## 1. Clone the repository

```bash
git clone https://github.com/annachang2027-NU/titanic-disaster.git
cd titanic-disaster
```

All commands below are run from this `titanic-disaster/` folder (the repo root).

---

## 2. Download the data

The dataset is **not** in this repo (per assignment rules, and `src/data/` is in `.gitignore`). Download it from Kaggle:

1. Go to **https://www.kaggle.com/competitions/titanic/data** and sign in.
2. If prompted, click **"Join Competition"** / **"I Understand and Accept"** (Kaggle requires this before you can download).
3. Click **"Download All"**. You will get `titanic.zip`.
4. Unzip it and move the CSV files into `src/data/`:

```bash
mkdir -p src/data
unzip ~/Downloads/titanic.zip -d src/data
```

(Or drag the files into `src/data/` by hand.)

<details>
<summary>Alternative: download with the Kaggle CLI</summary>

```bash
pip install kaggle   # needs an API token in ~/.kaggle/kaggle.json (Kaggle → Settings → API → Create New Token)
kaggle competitions download -c titanic -p src/data
unzip src/data/titanic.zip -d src/data
```
</details>

Check that the files are in the right place:

```bash
ls src/data
# gender_submission.csv  test.csv  train.csv
```

| File | Required? | Contents |
|------|-----------|----------|
| `train.csv` | **yes** | 891 passengers **with** the `Survived` label (used to train the model) |
| `test.csv` | **yes** | 418 passengers **without** the `Survived` label (we predict these) |
| `gender_submission.csv` | optional | Kaggle's example submission; used for one extra comparison if present |

---

## 3. Run the Python model

```bash
# Build the image (first time takes ~1 minute)
docker build -t titanic-python src/python

# Run it, mounting your local data folder into the container
docker run --rm -v "$(pwd)/src/data:/app/data" titanic-python
```

## 4. Run the R model

```bash
# Build the image (first time takes a few minutes while R packages install)
docker build -t titanic-r src/r

# Run it
docker run --rm -v "$(pwd)/src/data:/app/data" titanic-r
```

**What the flags mean**

- `-t titanic-python` names the image so you can run it by name.
- `src/python` (or `src/r`) is the folder containing the `Dockerfile`.
- `-v "$(pwd)/src/data:/app/data"` shares your `src/data/` folder with the container at `/app/data`. That's how the container reads the CSVs, and how its prediction file ends up back on your computer.
- `--rm` deletes the container when it finishes, so nothing is left behind.

> **Windows users:** run these in PowerShell and replace `$(pwd)` with `${PWD}`, e.g. `docker run --rm -v "${PWD}/src/data:/app/data" titanic-python`. In Command Prompt use `%cd%`.

---

## 5. What you'll see

Both scripts print the same six sections:

1. **Load `train.csv`**: shape and first rows.
2. **Explore**: missing values, and survival rates by sex, class, and port.
3. **Clean and engineer features**, with one line per change:
   - `Title` (Mr / Mrs / Miss / Master / Rare) pulled out of `Name`
   - missing `Age` filled with the median age for that title
   - missing `Embarked` filled with the most common port; missing `Fare` with the median fare
   - `FamilySize = SibSp + Parch + 1` and `IsAlone` added
   - `Sex` encoded as 0/1; `Pclass`, `Embarked`, `Title` one-hot encoded
   - `Cabin`, `Ticket`, `Name`, `SibSp`, `Parch` dropped
4. **Logistic regression**: an 80/20 hold-out check, then the final model fit on all 891 rows, with its coefficients.
5. **Training accuracy**.
6. **Test set**: `test.csv` gets cleaned the same way (using fill values learned from train only), then predicted. The predictions are saved to `src/data/predictions_python.csv` and `src/data/predictions_r.csv`, both in Kaggle submission format.

### A note on test-set accuracy

Kaggle **withholds the true `Survived` labels for `test.csv`**, so test accuracy can't be computed locally. The scripts handle this as follows:

- They print the **held-out validation accuracy** (20% of `train.csv` the model never saw) as the best local estimate.
- If `gender_submission.csv` is present, they print how often the predictions agree with Kaggle's "women survive, men don't" baseline. This is **not** accuracy, just a sanity check.
- To get the real test accuracy, upload `predictions_python.csv` or `predictions_r.csv` at https://www.kaggle.com/competitions/titanic/submissions.
- If you supply a `test.csv` that does contain a `Survived` column, the scripts print the true test accuracy automatically.

---

## Repository structure

```
titanic-disaster/
├── .github/
│   └── CODEOWNERS            # only the repo owner can approve changes
├── .gitignore                # ignores src/data/, *.csv, .DS_Store, venvs, etc.
├── README.md
└── src/
    ├── data/                 # ← you create this and put the Kaggle CSVs here (git-ignored)
    ├── python/
    │   ├── Dockerfile        # python:3.11-slim, installs requirements.txt, runs the script
    │   ├── requirements.txt  # pinned: numpy, pandas, scikit-learn
    │   └── titanic_model.py
    └── r/
        ├── Dockerfile        # rocker/r-ver:4.4.1, runs install_packages.R, then the script
        ├── install_packages.R# installs dplyr from a date-frozen CRAN snapshot
        └── titanic_model.R
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Cannot connect to the Docker daemon` | Open Docker Desktop and wait for it to finish starting. |
| `[ERROR] Could not find /app/data/train.csv` | The CSVs aren't in `src/data/`, or you're not running from the repo root. Run `ls src/data` and check you're in `titanic-disaster/`. |
| `docker: invalid reference format` on Windows | Use `${PWD}` (PowerShell) or `%cd%` (cmd) instead of `$(pwd)`. |
| Path contains spaces | Keep the quotes around `"$(pwd)/src/data:/app/data"`. |

---

## Optional: run without Docker

```bash
# Python
python3 -m venv venv && source venv/bin/activate
pip install -r src/python/requirements.txt
python src/python/titanic_model.py          # reads ./src/data by default

# R (from the src/r folder, so ../data resolves to src/data)
cd src/r && Rscript install_packages.R && Rscript titanic_model.R
```
