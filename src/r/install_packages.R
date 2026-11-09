# Installs the R packages used by titanic_model.R.
# The CRAN snapshot is frozen to a fixed date so every build gets the
# same package versions (dplyr 1.1.4).
options(repos = c(CRAN = "https://packagemanager.posit.co/cran/2024-10-30"))

packages <- c("dplyr")
install.packages(packages, Ncpus = parallel::detectCores())

# Fail the Docker build loudly if anything did not install
missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing) > 0) stop("Failed to install: ", paste(missing, collapse = ", "))
for (p in packages) cat(p, as.character(packageVersion(p)), "installed\n")
