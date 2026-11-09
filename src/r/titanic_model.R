# Titanic survival prediction with logistic regression (R).
#
# What this script does, in order:
#   1. Load train.csv
#   2. Explore it (missing values, survival rates)
#   3. Clean it and build new features
#   4. Fit a logistic regression with glm()
#   5. Report accuracy on the training data
#   6. Predict test.csv and save the predictions
#
# Where the data comes from:
#   - In Docker, the data folder is mounted at /app/data (DATA_DIR is set in the Dockerfile).
#   - Run locally from src/r, it falls back to ../data, which is src/data/.

# dplyr gives us readable data steps like mutate(), group_by() and summarise().
suppressPackageStartupMessages(library(dplyr))
options(width = 120)  # wide enough that printed tables don't wrap

# Folder with the CSVs. The DATA_DIR environment variable wins if it's set.
DATA_DIR <- Sys.getenv("DATA_DIR", unset = file.path("..", "data"))

# A fixed seed means the random 80/20 split is the same every run,
# so anyone who runs this gets the same numbers.
SEED <- 42

# Print a banner so each step is easy to find in the terminal output.
section <- function(title) {
  cat("\n", strrep("=", 70), "\n", title, "\n", strrep("=", 70), "\n", sep = "")
}

# Read a CSV from DATA_DIR. If it's missing, stop with a helpful message.
# na.strings = c("", "NA") turns empty cells (common in Cabin and Embarked)
# into proper NA values, so R knows they're missing.
load_csv <- function(name) {
  path <- file.path(DATA_DIR, name)
  if (!file.exists(path)) {
    stop("Could not find ", path, ".\nDownload train.csv and test.csv from ",
         "https://www.kaggle.com/competitions/titanic/data and put them in ",
         "src/data/ (see README.md).", call. = FALSE)
  }
  read.csv(path, stringsAsFactors = FALSE, na.strings = c("", "NA"))
}

# Get the title (Mr, Mrs, Miss, Master) from each passenger's name.
# Names look like "Braund, Mr. Owen Harris". The title sits between the comma
# and the period. Titles are useful because they tell us sex, rough age
# (Master = boy) and marital status (Miss vs Mrs) all at once.
extract_title <- function(name, sex) {
  # Regex: keep only the part between ", " and the next "."
  title <- trimws(sub("^.*?,\\s*([^.]+)\\..*$", "\\1", name))

  # French titles mean the same as English ones.
  title <- recode(title, Mlle = "Miss", Ms = "Miss", Mme = "Mrs")

  # Uncommon titles (Dr, Rev, Col, Countess, ...) are too few to learn from one
  # by one, so we group them. Rare women go to "Mrs" and rare men go to "Rare".
  # This way Title always tells us the passenger's sex.
  rare <- !title %in% c("Mr", "Mrs", "Miss", "Master")
  title[rare] <- ifelse(sex[rare] == "female", "Mrs", "Rare")
  title
}

# Accuracy = share of passengers we predicted correctly.
accuracy <- function(truth, pred) mean(truth == pred)

# Print the survival rate for each group of a column.
# Survived is 0/1, so its mean within a group is that group's survival rate.
print_rate <- function(df, var) {
  df %>% group_by(.data[[var]]) %>%
    summarise(n = n(), survival_rate = round(mean(Survived), 3), .groups = "drop") %>%
    as.data.frame() %>% print(row.names = FALSE)
}

# Clean one dataset (train or test) and print every change.
#
# `fills` holds numbers learned from the TRAINING data only. We reuse them on
# the test data. If we computed them from the test data, the model would be
# peeking at data it shouldn't see (this is called "data leakage").
clean <- function(df, fills, name) {
  section(paste("Cleaning & feature engineering:", name))

  # --- New feature: Title ---
  df$Title <- extract_title(df$Name, df$Sex)
  cat("[add] Title extracted from Name (rare titles -> 'Rare' for men, 'Mrs' for women). Counts:\n")
  print(table(df$Title))

  # --- Fill missing Age ---
  # About 20% of ages are missing. Filling with one overall median would make
  # every little boy 28 years old. Using the median for each title is smarter:
  # a missing "Master" gets a child's age, a missing "Mrs" an adult's.
  n_age <- sum(is.na(df$Age))
  df$Age <- ifelse(is.na(df$Age), fills$age_by_title[df$Title], df$Age)
  cat(sprintf("[fill] Age: %d missing values filled with median age of each Title (%s)\n",
              n_age, paste(names(fills$age_by_title), round(fills$age_by_title, 1),
                           sep = "=", collapse = ", ")))

  # --- Fill missing Embarked (port) ---
  # Only 2 are missing, so the most common port is a safe guess.
  n_emb <- sum(is.na(df$Embarked))
  df$Embarked[is.na(df$Embarked)] <- fills$embarked_mode
  cat(sprintf("[fill] Embarked: %d missing values filled with mode '%s'\n",
              n_emb, fills$embarked_mode))

  # --- Fill missing Fare ---
  # Only 1 is missing (in test.csv). We use the median, not the mean, because
  # a few very expensive tickets would pull the mean up.
  n_fare <- sum(is.na(df$Fare))
  df$Fare[is.na(df$Fare)] <- fills$fare_median
  cat(sprintf("[fill] Fare: %d missing values filled with median %.2f\n",
              n_fare, fills$fare_median))

  # --- New features: FamilySize and IsAlone ---
  # SibSp = siblings + spouses on board, Parch = parents + children.
  # Adding them (plus the passenger) gives the whole family size.
  # People travelling alone and people in very big families both did worse.
  df <- df %>% mutate(
    FamilySize = SibSp + Parch + 1,
    IsAlone = as.integer(FamilySize == 1)
  )
  cat(sprintf(paste("[add] FamilySize = SibSp + Parch + 1; IsAlone = 1 if FamilySize == 1.",
                    "Passengers travelling alone: %d of %d\n"),
              sum(df$IsAlone), nrow(df)))

  # --- Turn categories into factors ---
  # glm() automatically makes 0/1 dummy columns for factors. The first level
  # (1st class, port C, title Master) is the baseline the others are compared to.
  # We list the levels by hand so train and test always use the same ones.
  # Pclass looks like a number, but 3rd class isn't "3 times" 1st class,
  # so we treat it as a category.
  df <- df %>% mutate(
    Pclass = factor(Pclass, levels = c(1, 2, 3)),
    Embarked = factor(Embarked, levels = c("C", "Q", "S")),
    Title = factor(Title, levels = c("Master", "Miss", "Mr", "Mrs", "Rare"))
  )
  cat("[encode] Pclass, Embarked, Title -> factors (glm creates the dummy variables)\n")

  # --- Drop columns we won't use ---
  # Cabin: about 77% empty. Ticket and Name: free text (Name is already used
  # for Title). SibSp and Parch: now combined in FamilySize.
  # Sex: Title already says it. With both in the model, glm couldn't tell
  # which one deserves the credit. The Sex coefficient came out at 15, with a
  # standard error of 506, which is meaningless.
  dropped <- c("Cabin", "Ticket", "Name", "SibSp", "Parch", "Sex")
  df <- df %>% select(-all_of(dropped))
  cat(sprintf(paste("[drop] %s (Cabin is mostly missing; Ticket/Name are free text;",
                    "SibSp/Parch are replaced by FamilySize;\n       Sex is fully captured by",
                    "Title, and keeping both makes the coefficients unstable)\n"),
              paste(dropped, collapse = ", ")))

  # Sanity check: glm would silently drop rows with NA, so this should be 0.
  model_vars <- c("Pclass", "Age", "Fare", "Embarked", "Title", "FamilySize", "IsAlone")
  cat("Remaining missing values:", sum(is.na(df[, model_vars])), "\n")
  df
}

# The model: what we predict (Survived) ~ the features we use.
FORMULA <- Survived ~ Pclass + Age + Fare + Embarked + Title + FamilySize + IsAlone

# glm gives a survival probability. Above 0.5 means we predict "survived" (1).
predict_class <- function(model, newdata) {
  as.integer(predict(model, newdata = newdata, type = "response") > 0.5)
}

# ---- Environment ------------------------------------------------------------
# Print versions so anyone can check they're running the same setup.
section("Environment")
cat(R.version.string, "| dplyr", as.character(packageVersion("dplyr")), "\n")
cat("Reading data from:", normalizePath(DATA_DIR, mustWork = FALSE), "\n")

# ---- Step 1: Load the training data -----------------------------------------
section("Step 1: Load train.csv")
train_raw <- load_csv("train.csv")
cat(sprintf("train.csv loaded: %d rows x %d columns\n", nrow(train_raw), ncol(train_raw)))
print(head(train_raw, 5))

# ---- Step 2: Explore --------------------------------------------------------
# Before modelling, look at what's missing and which groups survived more.
# This tells us which features are likely to matter.
section("Step 2: Explore the training data")
na_counts <- colSums(is.na(train_raw))
cat("Missing values per column:\n")
print(na_counts[na_counts > 0])
cat(sprintf("\nOverall survival rate: %.1f%%\n", 100 * mean(train_raw$Survived)))
cat("\nSurvival rate by Sex:\n");      print_rate(train_raw, "Sex")
cat("\nSurvival rate by Pclass:\n");   print_rate(train_raw, "Pclass")
cat("\nSurvival rate by Embarked:\n"); print_rate(filter(train_raw, !is.na(Embarked)), "Embarked")

# Learn the fill values from the training data only (see clean() for why).
train_titles <- extract_title(train_raw$Name, train_raw$Sex)
fills <- list(
  age_by_title  = tapply(train_raw$Age, train_titles, median, na.rm = TRUE),
  embarked_mode = names(which.max(table(train_raw$Embarked))),
  fare_median   = median(train_raw$Fare, na.rm = TRUE)
)

# ---- Clean and build features -----------------------------------------------
train <- clean(train_raw, fills, "train.csv")
cat("\nSurvival rate by engineered Title:\n"); print_rate(train, "Title")

# ---- Step 3: Fit the model --------------------------------------------------
# Logistic regression = glm() with family = binomial. It predicts the
# probability of surviving from the features in FORMULA.
section("Step 3: Logistic regression (glm, family = binomial)")

# Practice run: train on 80% and test on the 20% the model never saw.
# This is our honest estimate of how well it works on new passengers.
# We sample 20% from survivors and 20% from non-survivors separately, so both
# parts keep the same survival rate (about 38%). This is a stratified split.
set.seed(SEED)
val_idx <- unlist(lapply(split(seq_len(nrow(train)), train$Survived),
                         function(i) sample(i, round(0.2 * length(i)))))
holdout <- glm(FORMULA, data = train[-val_idx, ], family = binomial)
val_acc <- accuracy(train$Survived[val_idx], predict_class(holdout, train[val_idx, ]))
cat(sprintf("80/20 split check -> train acc: %.4f, held-out validation acc: %.4f\n",
            accuracy(train$Survived[-val_idx], predict_class(holdout, train[-val_idx, ])),
            val_acc))

# Final model: train on all 891 passengers, since more data = a better model.
model <- glm(FORMULA, data = train, family = binomial)

# How to read this table:
#   Estimate   > 0 means higher survival odds, < 0 means lower, compared with
#              the baseline (1st class, port C, title Master).
#   Std. Error how unsure the estimate is. Small is good.
#   Pr(>|z|)   the p-value. Below 0.05 usually counts as a real effect.
cat(sprintf("\nFinal model fit on all %d training rows:\n", nrow(train)))
print(round(summary(model)$coefficients, 3))

# ---- Step 4: Training accuracy ----------------------------------------------
# This is a bit optimistic, because the model has already seen these passengers.
section("Step 4: Accuracy on the training set")
train_acc <- accuracy(train$Survived, predict_class(model, train))
cat(sprintf("Training accuracy: %.4f (%.1f%%)\n", train_acc, 100 * train_acc))

# ---- Step 5: Predict the test set -------------------------------------------
section("Step 5: Load test.csv and predict")
test_raw <- load_csv("test.csv")
cat(sprintf("test.csv loaded: %d rows x %d columns\n", nrow(test_raw), ncol(test_raw)))

# Same cleaning, with the same fill values from training.
test <- clean(test_raw, fills, "test.csv")
test_pred <- predict_class(model, test)
cat(sprintf("\nPredicted survivors in test set: %d of %d (%.1f%%)\n",
            sum(test_pred), length(test_pred), 100 * mean(test_pred)))

# Save in the format Kaggle expects: PassengerId, Survived.
out_path <- file.path(DATA_DIR, "predictions_r.csv")
write.csv(data.frame(PassengerId = test_raw$PassengerId, Survived = test_pred),
          out_path, row.names = FALSE)
cat("Predictions saved to", out_path, "(Kaggle submission format)\n")

# ---- Step 6: Test accuracy --------------------------------------------------
section("Step 6: Accuracy on the test set")
if ("Survived" %in% names(test_raw)) {
  # Only possible if someone gives us a test file with the answers.
  cat(sprintf("Test accuracy: %.4f\n", accuracy(test_raw$Survived, test_pred)))
} else {
  # Kaggle's real test.csv has no answers, so we explain what we can do.
  cat("Kaggle's test.csv does not include the true 'Survived' labels, so true test accuracy\n",
      "can only be computed by uploading predictions_r.csv to Kaggle.\n", sep = "")
  cat(sprintf("Best local estimate (held-out 20%% of train): %.4f\n", val_acc))

  # gender_submission.csv is Kaggle's simple guess: every woman survives,
  # every man dies. Comparing against it is a sanity check, not accuracy.
  gs_path <- file.path(DATA_DIR, "gender_submission.csv")
  if (file.exists(gs_path)) {
    gs <- read.csv(gs_path)
    agree <- mean(gs$Survived[match(test_raw$PassengerId, gs$PassengerId)] == test_pred)
    cat(sprintf("Agreement with Kaggle's gender_submission.csv baseline (women survive, men don't): %.4f\n",
                agree))
  }
}

section("Done")
