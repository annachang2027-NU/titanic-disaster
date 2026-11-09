# R container: trains a logistic regression on the Titanic data.
FROM rocker/r-ver:4.4.1

WORKDIR /app

# Install packages first so this layer is cached between code edits
COPY install_packages.R .
RUN Rscript install_packages.R

COPY titanic_model.R .

# The data is NOT baked into the image. Mount it at run time:
#   docker run --rm -v "$(pwd)/src/data:/app/data" titanic-r
ENV DATA_DIR=/app/data

CMD ["Rscript", "titanic_model.R"]
