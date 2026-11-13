suppressPackageStartupMessages({
  library(salesperson)
  library(data.table)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("Usage: Rscript extract_salesperson_raw_features.R <coordinates_long.csv> <features.csv> <schema.csv>")
}

input_csv <- args[[1]]
output_csv <- args[[2]]
schema_csv <- args[[3]]
xy_long <- fread(input_csv)
required <- c("state_id", "node_idx", "x", "y")
missing <- setdiff(required, names(xy_long))
if (length(missing) > 0) {
  stop(paste("Missing columns:", paste(missing, collapse = ", ")))
}

state_ids <- unique(xy_long$state_id)
rows <- vector("list", length(state_ids))
schema <- NULL
for (ii in seq_along(state_ids)) {
  iid <- state_ids[[ii]]
  if (ii == 1 || ii %% 100 == 0 || ii == length(state_ids)) {
    cat(sprintf("raw salesperson progress: %d/%d\n", ii, length(state_ids)))
    flush.console()
  }
  rows[[ii]] <- tryCatch({
    instance <- xy_long[state_id == iid]
    setorder(instance, node_idx)
    network <- makeNetwork(coordinates = as.matrix(instance[, .(x, y)]))
    feature_set <- getFeatureSet(
      network,
      black.list = c("VRP"),
      include.costs = FALSE,
      normalize = FALSE,
      drop.duplicates = FALSE
    )
    if (is.null(schema)) {
      schema <- data.table(
        feature_name = paste0("spn_", names(feature_set)),
        r_typeof = vapply(feature_set, typeof, character(1L)),
        r_class = vapply(feature_set, function(x) paste(class(x), collapse = ";"), character(1L))
      )
    }
    values <- as.data.table(as.list(feature_set))
    setnames(values, names(values), paste0("spn_", names(values)))
    values[, state_id := iid]
    values[, feature_error := NA_character_]
    values
  }, error = function(e) {
    data.table(state_id = iid, feature_error = as.character(e$message))
  })
}

features <- rbindlist(rows, fill = TRUE)
setcolorder(features, c("state_id", setdiff(names(features), "state_id")))
fwrite(features, output_csv)
if (!is.null(schema)) {
  fwrite(schema, schema_csv)
}
cat(sprintf(
  "raw salesperson complete: %d states, %d feature columns, %d errors\n",
  nrow(features),
  sum(startsWith(names(features), "spn_")),
  sum(!is.na(features$feature_error))
))
