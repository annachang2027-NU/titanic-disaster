# titanic-disaster

Who survived the Titanic? This repo predicts it with logistic regression.

This repo includes two versions of the same model: one in **Python** and one in **R**. Each has its own Docker container and prints each step in the terminal, so you can follow along without opening the code.

---

## Quick start

If you already have Docker and the data, these commands are all you need:

```bash
git clone https://github.com/annachang2027-NU/titanic-disaster.git
cd titanic-disaster
mkdir -p src/data
# put train.csv and test.csv in src/data/ (see step 2)

docker build -t titanic-python src/python
docker run --rm -v "$(pwd)/src/data:/app/data" titanic-python

docker build -t titanic-r src/r
docker run --rm -v "$(pwd)/src/data:/app/data" titanic-r
```

If you are new to this, please follow the steps below.

---

## 0. What you need

- **Git**, to download this repo. Check with `git --version`.
- **Docker Desktop**, to run the code. Check with `docker --version`.
- **A free Kaggle account**, to download the data.

You don't need Python or R on your computer. Docker installs them inside the containers.

Before you run any `docker` command, open Docker Desktop. Wait until it says "Engine running".

---

## 1. Download the repo

```bash
git clone https://github.com/annachang2027-NU/titanic-disaster.git
cd titanic-disaster
```

Run every command below from this `titanic-disaster` folder.

---

## 2. Get the data

The data is not in this repo. You download it from Kaggle yourself.

1. Go to https://www.kaggle.com/competitions/titanic/data and sign in.
2. Kaggle may ask you to accept the competition rules. Click accept.
3. Click **Download All**. You get a file called `titanic.zip`.
4. Unzip it into `src/data/`:

```bash
mkdir -p src/data
unzip ~/Downloads/titanic.zip -d src/data
```

Dragging the files into `src/data/` by hand works too.

Now check that the files are there:

```bash
ls src/data
# gender_submission.csv  test.csv  train.csv
```

What each file is:

- `train.csv` (**needed**): 891 passengers. We know who survived. The model learns from these.
- `test.csv` (**needed**): 418 passengers. We don't know who survived. The model guesses.
- `gender_submission.csv` (optional): Kaggle's simple example answer. If it's there, the code compares against it.

<details>
<summary>Prefer the command line? Use the Kaggle CLI</summary>

You need a Kaggle API token first. Get one at Kaggle → Settings → API → Create New Token. Save it as `~/.kaggle/kaggle.json`.

```bash
pip install kaggle
kaggle competitions download -c titanic -p src/data
unzip src/data/titanic.zip -d src/data
```
</details>

---

## 3. Run the Python version

```bash
docker build -t titanic-python src/python
docker run --rm -v "$(pwd)/src/data:/app/data" titanic-python
```

The first build takes about a minute.

## 4. Run the R version

```bash
docker build -t titanic-r src/r
docker run --rm -v "$(pwd)/src/data:/app/data" titanic-r
```

The first build takes a few minutes because R installs its packages.

### What these commands do

- `docker build` makes an image: a packaged copy of the code plus everything it needs.
- `-t titanic-python` gives the image a name.
- `src/python` is the folder with the `Dockerfile` in it.
- `docker run` starts the image.
- `-v "$(pwd)/src/data:/app/data"` lets the container see your `src/data` folder. It reads the CSVs from there and saves its predictions back there.
- `--rm` cleans up the container when it's done.

> **On Windows?** In PowerShell, write `${PWD}` instead of `$(pwd)`. In Command Prompt, write `%cd%`.

---

## 5. What you'll see

Both versions print the same steps:

1. **Load the data.** How many rows and columns, and the first few rows.
2. **Explore.** What's missing, and who survived more often (by sex, class, and port).
3. **Clean the data.** Each change gets its own line:
   - Pull the title (Mr, Mrs, Miss, Master) out of each name. Uncommon titles become "Rare" for men and "Mrs" for women.
   - Fill in missing ages with the typical age for that title. A "Master" is a boy, so he gets a child's age.
   - Fill in the 2 missing ports with the most common one, and the 1 missing fare with the typical fare.
   - Add `FamilySize` (how many family members were on board) and `IsAlone`.
   - Remove columns we don't use: `Cabin` (mostly empty), `Ticket` and `Name` (just text), `SibSp` and `Parch` (now inside `FamilySize`), and `Sex` (see below).
4. **Train the model.** First a practice run on 80% of the data, then the final model on all of it. The coefficients get printed too.
5. **Training accuracy.**
6. **Predict the test set.** The test data gets the same cleaning. Predictions are saved to `src/data/predictions_python.csv` and `src/data/predictions_r.csv`.

**Why remove `Sex`?** The title already tells us it. "Mr" is always a man, and "Mrs" is always a woman. With both in the model, R got confused: the Sex coefficient came out as 15, with an error of 506. Keeping only the title fixed that.

### Results you should get

| | Python | R |
|---|---|---|
| Training accuracy | 0.833 | 0.833 |
| Accuracy on the 20% held back | 0.832 | 0.837 |
| Survivors predicted in `test.csv` | 164 of 418 | 169 of 418 |
| Agreement with `gender_submission.csv` | 0.933 | 0.916 |

The two versions are close but not identical. Python's model adds a small penalty to keep coefficients small, and R's doesn't. The random 80/20 split is also different in each language.

### What the data tells us

- 74% of women survived. Only 19% of men did.
- 63% of first class survived. Only 24% of third class did.
- Being a "Mr" or riding in third class were the strongest signs someone would not survive.
- Bigger families did worse. Each extra family member lowered the odds.

### Why there's no test accuracy

Kaggle keeps the answers for `test.csv` secret, so we can't check our guesses on our own computer. Instead:

- The code holds back 20% of the training data and tests on it. That accuracy is our best guess for how the model does on new data.
- To get the real test score, upload a prediction file at https://www.kaggle.com/competitions/titanic/submissions.
- If your `test.csv` happens to include a `Survived` column, the code calculates test accuracy for you.

---

## What's in this repo

```
titanic-disaster/
├── .github/CODEOWNERS       # only the owner can approve changes
├── .gitignore               # keeps data, .DS_Store, etc. out of git
├── README.md
└── src/
    ├── data/                # you create this and put the CSVs here (not in git)
    ├── python/
    │   ├── Dockerfile       # builds the Python container
    │   ├── requirements.txt # Python packages, with exact versions
    │   └── titanic_model.py # the Python code
    └── r/
        ├── Dockerfile       # builds the R container
        ├── install_packages.R # installs R packages
        └── titanic_model.R  # the R code
```

---

## If something goes wrong

- **"Cannot connect to the Docker daemon"**: Docker Desktop isn't running. Open it and wait a moment.
- **"Could not find /app/data/train.csv"**: The CSVs aren't in `src/data/`, or you're in the wrong folder. Run `ls src/data` and check you're inside `titanic-disaster`.
- **"invalid reference format" on Windows**: Use `${PWD}` or `%cd%` instead of `$(pwd)`.
- **Your folder path has spaces**: Keep the quotes around `"$(pwd)/src/data:/app/data"`.

---

## Running without Docker (optional)

If you have Python or R installed, you can run the code directly.

```bash
# Python
python3 -m venv venv
source venv/bin/activate
pip install -r src/python/requirements.txt
python src/python/titanic_model.py
```

```bash
# R (run from inside src/r)
cd src/r
Rscript install_packages.R
Rscript titanic_model.R
```
