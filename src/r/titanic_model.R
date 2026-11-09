# Titanic survival prediction with logistic regression (R).
#
# Reads train.csv / test.csv from DATA_DIR (default: ../data relative to the
# working directory, or /app/data inside Docker), cleans and engineers
# features, fits a logistic regression with glm(), reports accuracy, and
# writes test-set predictions.

suppressPackageStartupMessages(library(dplyr))
options(width = 120)

DATA_DIR <- Sys.getenv("DATA_DIR", unset = file.path("..", "data"))
SEED <- 42

section <- function(title) {
  cat("\n", strrep("=", 70), "\n", title, "\n", strrep("=", 70), "\n", sep = "")
}

load_csv <- function(name) {
  path <- file.path(DATA_DIR, name)
  if (!file.exists(path)) {
    stop("Could not find ", path, ".\nDownload train.csv and test.csv from ",
         "https://www.kaggle.com/competitions/titanic/data and put them in ",
         "src/data/ (see README.md).", call. = FALSE)
  }
  read.csv(path, stringsAsFactors = FALSE, na.strings = c("", "NA"))
}

extract_title <- function(name) {
  title <- trimws(sub("^.*?,\\s*([^.]+)\\..*$", "\\1", name))
  title <- recode(title, Mlle = "Miss", Ms = "Miss", Mme = "Mrs")
  ifelse(title %in% c("Mr", "Mrs", "Miss", "Master"), title, "Rare")
}

accuracy <- function(truth, pred) mean(truth == pred)

print_rate <- function(df, var) {
  df %>% group_by(.data[[var]]) %>%
    summarise(n = n(), survival_rate = round(mean(Survived), 3), .groups = "drop") %>%
    as.data.frame() %>% print(row.names = FALSE)
}

# Same cleaning / feature steps for train and test; fill values come from train.
clean <- function(df, fills, name) {
  section(paste("Cleaning & feature engineering:", name))

  df$Title <- extract_title(df$Name)
  cat("[add] Title extracted from Name. Counts:\n")
  print(table(df$Title))

  n_age <- sum(is.na(df$Age))
  df$Age <- ifelse(is.na(df$Age), fills$age_by_title[df$Title], df$Age)
  cat(sprintf("[fill] Age: %d missing values filled with median age of each Title (%s)\n",
              n_age, paste(names(fills$age_by_title), round(fills$age_by_title, 1),
                           sep = "=", collapse = ", ")))

  n_emb <- sum(is.na(df$Embarked))
  df$Embarked[is.na(df$Embarked)] <- fills$embarked_mode
  cat(sprintf("[fill] Embarked: %d missing values filled with mode '%s'\n",
              n_emb, fills$embarked_mode))

  n_fare <- sum(is.na(df$Fare))
  df$Fare[is.na(df$Fare)] <- fills$fare_median
  cat(sprintf("[fill] Fare: %d missing values filled with median %.2f\n",
              n_fare, fills$fare_median))

  df <- df %>% mutate(
    FamilySize = SibSp + Parch + 1,
    IsAlone = as.integer(FamilySize == 1)
  )
  cat(sprintf(paste("[add] FamilySize = SibSp + Parch + 1; IsAlone = 1 if FamilySize == 1.",
                    "Passengers travelling alone: %d of %d\n"),
              sum(df$IsAlone), nrow(df)))

  df <- df %>% mutate(
    Sex = as.integer(Sex == "female"),
    Pclass = factor(Pclass, levels = c(1, 2, 3)),
    Embarked = factor(Embarked, levels = c("C", "Q", "S")),
    Title = factor(Title, levels = c("Master", "Miss", "Mr", "Mrs", "Rare"))
  )
  cat("[encode] Sex -> 1 = female, 0 = male; Pclass, Embarked, Title -> factors\n")

  dropped <- c("Cabin", "Ticket", "Name", "SibSp", "Parch")
  df <- df %>% select(-all_of(dropped))
  cat(sprintf(paste("[drop] %s (Cabin is ~77%% missing; Ticket/Name are free text;",
                    "SibSp/Parch are replaced by FamilySize)\n"),
              paste(dropped, collapse = ", ")))

  model_vars <- c("Pclass", "Sex", "Age", "Fare", "Embarked", "Title", "FamilySize", "IsAlone")
  cat("Remaining missing values:", sum(is.na(df[, model_vars])), "\n")
  df
}

FORMULA <- Survived ~ Pclass + Sex + Age + Fare + Embarked + Title + FamilySize + IsAlone

predict_class <- function(model, newdata) {
  as.integer(predict(model, newdata = newdata, type = "response") > 0.5)
}

# ---- Environment ------------------------------------------------------------
section("Environment")
cat(R.version.string, "| dplyr", as.character(packageVersion("dplyr")), "\n")
cat("Reading data from:", normalizePath(DATA_DIR, mustWork = FALSE), "\n")

# ---- Load training data -----------------------------------------------------
section("Step 1: Load train.csv")
train_raw <- load_csv("train.csv")
cat(sprintf("train.csv loaded: %d rows x %d columns\n", nrow(train_raw), ncol(train_raw)))
print(head(train_raw, 5))

# ---- Explore ----------------------------------------------------------------
section("Step 2: Explore the training data")
na_counts <- colSums(is.na(train_raw))
cat("Missing values per column:\n")
print(na_counts[na_counts > 0])
cat(sprintf("\nOverall survival rate: %.1f%%\n", 100 * mean(train_raw$Survived)))
cat("\nSurvival rate by Sex:\n");      print_rate(train_raw, "Sex")
cat("\nSurvival rate by Pclass:\n");   print_rate(train_raw, "Pclass")
cat("\nSurvival rate by Embarked:\n"); print_rate(filter(train_raw, !is.na(Embarked)), "Embarked")

train_titles <- extract_title(train_raw$Name)
fills <- list(
  age_by_title  = tapply(train_raw$Age, train_titles, median, na.rm = TRUE),
  embarked_mode = names(which.max(table(train_raw$Embarked))),
  fare_median   = median(train_raw$Fare, na.rm = TRUE)
)

# ---- Clean / engineer -------------------------------------------------------
train <- clean(train_raw, fills, "train.csv")
cat("\nSurvival rate by engineered Title:\n"); print_rate(train, "Title")

# ---- Model ------------------------------------------------------------------
section("Step 3: Logistic regression (glm, family = binomial)")
set.seed(SEED)
val_idx <- unlist(lapply(split(seq_len(nrow(train)), train$Survived),
                         function(i) sample(i, round(0.2 * length(i)))))
holdout <- glm(FORMULA, data = train[-val_idx, ], family = binomial)
val_acc <- accuracy(train$Survived[val_idx], predict_class(holdout, train[val_idx, ]))
cat(sprintf("80/20 split check -> train acc: %.4f, held-out validation acc: %.4f\n",
            accuracy(train$Survived[-val_idx], predict_class(holdout, train[-val_idx, ])),
            val_acc))

model <- glm(FORMULA, data = train, family = binomial)
cat(sprintf("\nFinal model fit on all %d training rows:\n", nrow(train)))
print(round(summary(model)$coefficients, 3))

# ---- Training accuracy ------------------------------------------------------
section("Step 4: Accuracy on the training set")
train_acc <- accuracy(train$Survived, predict_class(model, train))
cat(sprintf("Training accuracy: %.4f (%.1f%%)\n", train_acc, 100 * train_acc))

# ---- Test set ---------------------------------------------------------------
section("Step 5: Load test.csv and predict")
test_raw <- load_csv("test.csv")
cat(sprintf("test.csv loaded: %d rows x %d columns\n", nrow(test_raw), ncol(test_raw)))
test <- clean(test_raw, fills, "test.csv")
test_pred <- predict_class(model, test)
cat(sprintf("\nPredicted survivors in test set: %d of %d (%.1f%%)\n",
            sum(test_pred), length(test_pred), 100 * mean(test_pred)))

out_path <- file.path(DATA_DIR, "predictions_r.csv")
write.csv(data.frame(PassengerId = test_raw$PassengerId, Survived = test_pred),
          out_path, row.names = FALSE)
cat("Predictions saved to", out_path, "(Kaggle submission format)\n")

# ---- Test accuracy ----------------------------------------------------------
section("Step 6: Accuracy on the test set")
if ("Survived" %in% names(test_raw)) {
  cat(sprintf("Test accuracy: %.4f\n", accuracy(test_raw$Survived, test_pred)))
} else {
  cat("Kaggle's test.csv does not include the true 'Survived' labels, so true test accuracy\n",
      "can only be computed by uploading predictions_r.csv to Kaggle.\n", sep = "")
  cat(sprintf("Best local estimate (held-out 20%% of train): %.4f\n", val_acc))
  gs_path <- file.path(DATA_DIR, "gender_submission.csv")
  if (file.exists(gs_path)) {
    gs <- read.csv(gs_path)
    agree <- mean(gs$Survived[match(test_raw$PassengerId, gs$PassengerId)] == test_pred)
    cat(sprintf("Agreement with Kaggle's gender_submission.csv baseline (women survive, men don't): %.4f\n",
                agree))
  }
}

section("Done")
