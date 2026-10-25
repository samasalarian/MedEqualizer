library(optparse)
library(plotly)
library(formattable)
library(dplyr)
library(tidyr)
library(ggplot2)
library(lubridate)
library(rlang)


args_all    <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args_all[grep("--file=", args_all)])
script_dir  <- if (length(script_path) > 0) {
  dirname(normalizePath(script_path))
} else {
  getwd()
}
root_dir    <- normalizePath(file.path(script_dir, "..", ".."))  # <root>
scripts_dir <- file.path(script_dir, "scripts")

source(file.path(scripts_dir, "Equity_metrics.R"))
source(file.path(scripts_dir, "sunburst_process.R"))
source(file.path(scripts_dir, "table_process.R"))

# ── CLI arguments ──────────────────────────────────────────────────────────────
option_list <- list(
  make_option("--data",        type = "character", help = "Dataset version: mimic3 or mimic4"),
  make_option("--model",       type = "character", help = "Synthetic model: ctgan, healthgan, medgan (or *_cat)"),
  make_option("--version",     type = "character", help = "Data split: before or after"),
  make_option("--tau",         type = "character", default = NULL,
              help = "Tau used for this run; when set (and --after_file not given), reads after_tau=<tau>.csv instead of after.csv"),
  make_option("--after_file",  type = "character", default = NULL,
              help = "Exact after-CSV basename to read (overrides --tau-based reconstruction)"),
  make_option("--categorized", action = "store_true", default = FALSE,
              help = "Use real_categorized.csv as reference and map RACE directly from ETHNICITY")
)
args <- parse_args(OptionParser(option_list = option_list))

valid_models <- c("ctgan", "healthgan", "medgan",
                  "ctgan_cat", "healthgan_cat", "medgan_cat")
if (!args$data    %in% c("mimic3", "mimic4"))  stop("--data must be mimic3 or mimic4")
if (!args$model   %in% valid_models)           stop(paste("--model must be one of:", paste(valid_models, collapse = ", ")))
if (!args$version %in% c("before", "after"))   stop("--version must be before or after")

data_version <- args$data    # "mimic3" | "mimic4"
model_name   <- args$model   # "ctgan"  | "healthgan" | "medgan"
split_ver    <- args$version # "before" | "after"


# ── Paths ──────────────────────────────────────────────────────────────────────
data_dir <- file.path(root_dir, "data")

data_folder <- paste0(toupper(substring(data_version, 1, 1)),
                      substring(data_version, 2))   # "Mimic3" or "Mimic4"

ref_file  <- if (args$categorized) "real_categorized.csv" else "real.csv"
synth_file <- if (split_ver == "after" && !is.null(args$after_file)) {
  args$after_file
} else if (args$categorized && split_ver == "before") {
  "before_categorized.csv"
} else if (split_ver == "after" && !is.null(args$tau)) {
  paste0("after_tau=", args$tau, ".csv")
} else {
  paste0(split_ver, ".csv")
}

reference_path <- file.path(data_dir, data_folder, "real", ref_file)
synthetic_path <- file.path(data_dir, data_folder, "synthetic", model_name, synth_file)

MIMICRacereference <- read.csv(reference_path)
MIMICRacesynthetic <- read.csv(synthetic_path)

# ── Preprocessing helpers ──────────────────────────────────────────────────────

add_race_mimic3 <- function(data) {
  col <- "ETHNICITY"
  # White: includes PORTUGUESE (European nationality → grouped with White)
  # Asian: excludes ASIAN - KOREAN (n=10) and ASIAN - THAI (n=3) → fall into Other
  # Other: catches all unmatched groups — Hispanic/Latino, Middle Eastern,
  #        Native Hawaiian, American Indian, Caribbean Island, Multi Race,
  #        South American, ASIAN - KOREAN, ASIAN - THAI
  data$RACE <- ifelse(data[[col]] %in% c("WHITE", "WHITE - OTHER EUROPEAN", "PORTUGUESE",
                                         "WHITE - BRAZILIAN", "WHITE - RUSSIAN",
                                         "WHITE - EASTERN EUROPEAN"), "White",
                      ifelse(data[[col]] %in% c("BLACK/AFRICAN AMERICAN", "BLACK/AFRICAN",
                                                "BLACK/HAITIAN", "BLACK/CAPE VERDEAN"), "Black",
                             ifelse(data[[col]] %in% c("ASIAN", "ASIAN - ASIAN INDIAN", "ASIAN - CHINESE",
                                                       "ASIAN - VIETNAMESE", "ASIAN - CAMBODIAN",
                                                       "ASIAN - OTHER", "ASIAN - JAPANESE",
                                                       "ASIAN - FILIPINO"), "Asian",
                                    ifelse(data[[col]] %in% c("UNKNOWN/NOT SPECIFIED", "UNABLE TO OBTAIN",
                                                              "PATIENT DECLINED TO ANSWER"), "Unknown",
                                           "Other"))))
  return(data)
}

add_race_mimic4 <- function(data) {
  col <- "ETHNICITY"
  # White: includes PORTUGUESE (European nationality → grouped with White)
  # Asian: includes ASIAN - KOREAN (n=54) — unlike MIMIC-3 where it falls into Other
  # Black: BLACK/CARIBBEAN ISLAND replaces BLACK/HAITIAN (not present in MIMIC-4)
  # Unknown: "UNKNOWN" (MIMIC-4 label) vs "UNKNOWN/NOT SPECIFIED" (MIMIC-3 label)
  # Other: Hispanic/Latino, Native Hawaiian, American Indian, South American,
  #        Multiple Race, and remaining unmatched groups
  white   <- c("WHITE", "WHITE - RUSSIAN", "PORTUGUESE", "WHITE - OTHER EUROPEAN",
               "WHITE - BRAZILIAN", "WHITE - EASTERN EUROPEAN")
  black   <- c("BLACK/AFRICAN AMERICAN", "BLACK/CAPE VERDEAN", "BLACK/AFRICAN",
               "BLACK/CARIBBEAN ISLAND")
  asian   <- c("ASIAN", "ASIAN - SOUTH EAST ASIAN", "ASIAN - CHINESE",
               "ASIAN - KOREAN", "ASIAN - ASIAN INDIAN")
  unknown <- c("UNKNOWN", "UNABLE TO OBTAIN", "PATIENT DECLINED TO ANSWER")
  data$RACE <- ifelse(data[[col]] %in% white,   "White",
                      ifelse(data[[col]] %in% black,   "Black",
                             ifelse(data[[col]] %in% asian,   "Asian",
                                    ifelse(data[[col]] %in% unknown, "Unknown",
                                           "Other"))))
  return(data)
}

add_age <- function(data) {
  data$AGE <- data$AGE_GROUP
  return(data)
}

add_mortality <- function(data) {
  data$mortality <- ifelse(as.integer(data$MORTALITY) == 0, "Alive",
                         ifelse(as.integer(data$MORTALITY) == 1, "Died",
                                NA_character_))
  return(data)
}

add_race_categorized <- function(data) {
  # ETHNICITY already holds 5-category values; copy directly to RACE
  data$RACE <- data$ETHNICITY
  return(data)
}

# ── Apply preprocessing ────────────────────────────────────────────────────────
if (args$categorized) {
  MIMICRacereference <- add_race_categorized(MIMICRacereference)
  MIMICRacesynthetic <- add_race_categorized(MIMICRacesynthetic)
} else if (data_version == "mimic3") {
  MIMICRacereference <- add_race_mimic3(MIMICRacereference)
  MIMICRacesynthetic <- add_race_mimic3(MIMICRacesynthetic)
} else {
  MIMICRacereference <- add_race_mimic4(MIMICRacereference)
  MIMICRacesynthetic <- add_race_mimic4(MIMICRacesynthetic)
}

MIMICRacereference <- add_age(MIMICRacereference)
MIMICRacesynthetic <- add_age(MIMICRacesynthetic)

MIMICRacereference <- add_mortality(MIMICRacereference)
MIMICRacesynthetic <- add_mortality(MIMICRacesynthetic)

# ── Core preprocessing / equity evaluation function ───────────────────────────
preprocess_mimic_data <- function(reference_data, synthetic_data, group_vars,
                                  output_file,
                                  sig_t   = 0.05,
                                  lower_t = -log(0.9),
                                  upper_t = -log(0.8)) {

  if ("GENDER" %in% group_vars) {
    reference_data <- reference_data %>%
      mutate(GENDER = recode(GENDER, 'M' = "Male", 'F' = "Female"))
    synthetic_data <- synthetic_data %>%
      mutate(GENDER = recode(GENDER, 'M' = "Male", 'F' = "Female"))
  }

  reference_processed <- reference_data %>%
    group_by(across(all_of(group_vars))) %>%
    summarise(background_n = n())
  colnames(reference_processed) <- c(group_vars, "background_n")

  synthetic_processed <- synthetic_data %>%
    group_by(across(all_of(group_vars))) %>%
    summarise(user_n = n())
  colnames(synthetic_processed) <- c(group_vars, "user_n")

  test_sunburst2(reference_processed,
                 group_vars,
                 synthetic_processed,
                 output_file,
                 sig_t   = sig_t,
                 lower_t = lower_t,
                 upper_t = upper_t)
}

# ── Run ────────────────────────────────────────────────────────────────────────
output_base <- "mortality.race.age.gender"

group_vars <- unlist(lapply(strsplit(output_base, "\\.")[[1]], function(x) {
  if (tolower(x) == "mortality") {
    return("mortality")
  } else {
    return(toupper(x))
  }
}))

# Incorporate run parameters into output filenames
run_tag     <- paste(data_version, model_name, split_ver, sep = "_")
output_csv  <- paste0("plot_", split_ver, ".csv")
output_html <- paste0("plot_", split_ver, ".html")

preprocess_mimic_data(
  reference_data = MIMICRacereference,
  synthetic_data = MIMICRacesynthetic,
  group_vars     = group_vars,
  output_file    = output_csv
)

df_processed <- read.csv(output_csv)
p <- generate_sunburst_plotly3(df_processed)

htmlwidgets::saveWidget(p, output_html, selfcontained = FALSE)
