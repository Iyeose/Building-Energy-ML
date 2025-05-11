# ===================================================================
# Project: Predicting Building Energy Consumption (HL & CL)
# Author: [Iyeose Uhumuavbi]
# ===================================================================

# -------------------------------------------------------------------
# 0. Setup
# -------------------------------------------------------------------

# Load libraries
library(readxl)
library(dplyr)
library(tidyr)
library(ggplot2)
library(corrplot)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(e1071)
library(Metrics) # For evaluation metrics like rmse, mae
library(ppcor)
library(neuralnet)


# Set seed for reproducibility
set.seed(123)

# -------------------------------------------------------------------
# 1. Data Loading and Initial Inspection
# -------------------------------------------------------------------
# Load data
# set working directory
setwd(dirname(file.choose()))
getwd()

# read file
energy_data <- read.csv("ENB2012_data.csv", stringsAsFactors = FALSE)


# Rename columns
colnames(energy_data) <- c("Relative_Compactness", "Surface_Area", "Wall_Area",
                           "Roof_Area", "Overall_Height", "Orientation",
                           "Glazing_Area", "Glazing_Area_Distribution",
                           "Heating_Load", "Cooling_Load")

# Initial inspection
print(head(energy_data))
print(str(energy_data))
print(summary(energy_data))

# -------------------------------------------------------------------
# 2. Exploratory Data Analysis (EDA)
# -------------------------------------------------------------------
# Check for missing values
print(paste("Total missing values:", sum(is.na(energy_data))))

# Check for duplicates
print(paste("Number of duplicate rows:", sum(duplicated(energy_data))))

# --- Visualizations ---

# Histograms for features (example, you can loop or do more)
ggplot(energy_data, aes(x = Relative_Compactness)) + geom_histogram(bins = 20, fill = "blue", alpha = 0.7) + theme_minimal() + ggtitle("Distribution of Relative Compactness")
ggplot(energy_data, aes(x = Overall_Height)) + geom_histogram(bins = 15, fill = "purple", alpha = 0.7) + theme_minimal() + ggtitle("Distribution of Overall Height")


# Histograms for target variables
ggplot(energy_data, aes(x = Heating_Load)) + geom_histogram(bins = 20, fill = "red", alpha = 0.7) + theme_minimal() + ggtitle("Distribution of Heating Load")
ggplot(energy_data, aes(x = Cooling_Load)) + geom_histogram(bins = 20, fill = "green", alpha = 0.7) + theme_minimal() + ggtitle("Distribution of Cooling Load")

# --- Box Plots ---
# Box plot for ALL variables (using pivot_longer and facet_wrap)

energy_data_long <- energy_data %>%
  pivot_longer(cols = everything(), names_to = "Variable", values_to = "Value")

ggplot(energy_data_long, aes(x = Variable, y = Value)) +
  geom_boxplot(fill = "lightblue") +
  facet_wrap(~ Variable, scales = "free", ncol = 4) + # Use 'free' scales for different ranges
  theme_minimal() +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank()) + # Hide x-axis text
  ggtitle("Box Plots for All Variables")

# Box plot for ONLY the two target variables

target_data_long <- energy_data %>%
  dplyr::select(Heating_Load, Cooling_Load) %>% # Use dplyr::select explicitly
  pivot_longer(cols = everything(), names_to = "Variable", values_to = "Value")

ggplot(target_data_long, aes(x = Variable, y = Value, fill = Variable)) +
  geom_boxplot() +
  scale_fill_manual(values = c("Heating_Load" = "red", "Cooling_Load" = "green")) +
  facet_wrap(~ Variable, scales = "free") + # Use 'free' scales
  theme_minimal() +
  theme(legend.position = "none", 
        axis.text.x = element_text(angle = 45, hjust = 1)) +
  ggtitle("Box Plots for Target Variables (Heating Load & Cooling Load)")


# --- Scatter plots (Feature vs Target) ---

ggplot(energy_data, aes(x = Overall_Height, y = Heating_Load)) + geom_point(alpha = 0.5) + theme_minimal() + ggtitle("Overall Height vs Heating Load")
ggplot(energy_data, aes(x = Surface_Area, y = Cooling_Load)) + geom_point(alpha = 0.5) + theme_minimal() + ggtitle("Surface Area vs Cooling Load")


# --- Correlation Analysis ---
# Full Correlation Matrix 

cor_matrix_spearman <- cor(energy_data, method = "spearman")
corrplot(cor_matrix_spearman, method = "circle", type = "upper", tl.col = "black", tl.srt = 45, title = "Spearman Correlation Matrix", mar=c(0,0,1,0))
print(cor_matrix_spearman,digits = 3)

# Heating Load vs Specific Features
cor.test(energy_data$Heating_Load, energy_data$Orientation, method = "spearman")
cor.test(energy_data$Heating_Load, energy_data$Glazing_Area, method = "spearman")
cor.test(energy_data$Heating_Load, energy_data$Glazing_Area_Distribution, method = "spearman")


# Cooling Load vs Specific Features
cor.test(energy_data$Cooling_Load, energy_data$Orientation, method = "spearman")
cor.test(energy_data$Cooling_Load, energy_data$Glazing_Area, method = "spearman")
cor.test(energy_data$Cooling_Load, energy_data$Glazing_Area_Distribution, method = "spearman")


# --- Partial Correlation Analysis ---

library(ppcor)
# between X1 and X2
pcor.test(energy_data$Heating_Load, energy_data$Relative_Compactness, energy_data$Surface_Area)
pcor.test(energy_data$Heating_Load, energy_data$Surface_Area, energy_data$Relative_Compactness)
# between X4 and X5
pcor.test(energy_data$Heating_Load, energy_data$Roof_Area, energy_data$Overall_Height)
pcor.test(energy_data$Heating_Load, energy_data$Overall_Height, energy_data$Roof_Area)

# --- Feature Selection (Dropping Columns) ---
cols_to_drop <- c("Orientation", "Glazing_Area_Distribution", "Relative_Compactness", "Roof_Area")

# Check if columns exist before dropping
cols_exist <- cols_to_drop %in% colnames(energy_data)
if (!all(cols_exist)) {
  warning(paste("Some columns specified for dropping do not exist:", paste(cols_to_drop[!cols_exist], collapse=", ")))
  cols_to_drop <- intersect(cols_to_drop, colnames(energy_data)) # Only drop existing ones
}

energy_data_selected <- energy_data %>%
  dplyr::select(-all_of(cols_to_drop)) # Explicitly use dplyr's select

summary(energy_data_selected)
head(energy_data_selected)
boxplot(energy_data_selected, main = "energy_Data_Selected", col = "bisque")

energy_data_selected_long <- energy_data_selected %>%
  pivot_longer(cols = everything(), names_to = "Variable", values_to = "Value")

ggplot(energy_data_selected_long, aes(x = Variable, y = Value)) +
  geom_boxplot(fill = "lightblue") +
  facet_wrap(~ Variable, scales = "free", ncol = 5) + # Use 'free' scales for different ranges
  theme_minimal() +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank()) + # Hide x-axis text
  ggtitle("Box Plots for All Selected Variables")

# -------------------------------------------------------------------
# 3. Data Preprocessing (Splitting and Z-score Scaling )
# -------------------------------------------------------------------
# Scaling will be applied AFTER splitting, fitting ONLY on training data.

# Separate features and targets from the selected data
target_vars <- c("Heating_Load", "Cooling_Load")
features_selected <- energy_data_selected %>% dplyr::select(-all_of(target_vars)) # Use dplyr::select explicitly
target_hl <- energy_data_selected$Heating_Load
target_cl <- energy_data_selected$Cooling_Load

# --- Data Splitting (80% train, 20% test) ---
# Split FIRST, using unscaled features
train_index <- createDataPartition(target_hl, p = 0.8, list = FALSE, times = 1)

# Create train/test sets for FEATURES (unscaled initially)
features_train_unscaled <- features_selected[train_index, ]
features_test_unscaled <- features_selected[-train_index, ]

# Create train/test sets for TARGETS
target_hl_train <- target_hl[train_index]
target_hl_test <- target_hl[-train_index]
target_cl_train <- target_cl[train_index]
target_cl_test <- target_cl[-train_index]

# --- Feature Scaling (Z-score using caret::preProcess) ---
# Fit feature scaler ONLY on training data
scaler_features <- preProcess(features_train_unscaled, method = c("center", "scale"))

# Apply the fitted scaler to both train and test features
features_train_scaled <- predict(scaler_features, features_train_unscaled)
features_test_scaled <- predict(scaler_features, features_test_unscaled)

# Check scaled data summary 
print(summary(head(features_train_scaled)))
boxplot(features_train_scaled, main = "features_train_scaled", col = "bisque")
print(summary(head(features_test_scaled)))
boxplot(features_test_scaled, main = "features_test_scaled", col = "bisque")


# Prepare data for plotting (combine train/test for comparison)
scaled_train_long <- features_train_scaled %>%
  mutate(Set = "Train") %>%
  pivot_longer(cols = -Set, names_to = "Variable", values_to = "Scaled_Value")

scaled_test_long <- features_test_scaled %>%
  mutate(Set = "Test") %>%
  pivot_longer(cols = -Set, names_to = "Variable", values_to = "Scaled_Value")

scaled_features_long <- rbind(scaled_train_long, scaled_test_long)

# Box plots of all scaled features (Train vs Test)
plot_box_scaled <- ggplot(scaled_features_long, aes(x = Variable, y = Scaled_Value, fill = Set)) +
  geom_boxplot(alpha = 0.7) +
  facet_wrap(~ Variable, scales = "free_x") + # Separate facets, free x scale if needed later
  scale_fill_manual(values = c("Train" = "skyblue", "Test" = "lightcoral")) +
  theme_minimal() +
  theme(axis.text.x = element_blank(), # Hide redundant x-axis labels within facets
        axis.ticks.x = element_blank(),
        legend.position = "bottom") +
  ggtitle("Box Plots of Standardized Features (Train vs Test Sets)") +
  ylab("Standardized Value (Z-score)") +
  xlab("Feature")

print(plot_box_scaled)



# --- Target Scaling (Z-score using caret::preProcess) --- ADDED for neuralnet
# Fit target scalers ONLY on training targets
scaler_hl <- preProcess(data.frame(Heating_Load = target_hl_train), method = c("center", "scale"))
scaler_cl <- preProcess(data.frame(Cooling_Load = target_cl_train), method = c("center", "scale"))
# Apply target scalers to training targets
target_hl_train_scaled <- predict(scaler_hl, data.frame(Heating_Load = target_hl_train))$Heating_Load
target_cl_train_scaled <- predict(scaler_cl, data.frame(Cooling_Load = target_cl_train))$Cooling_Load

# Store scaling parameters for later un-scaling predictions
scale_params_hl <- list(mean = scaler_hl$mean["Heating_Load"], sd = scaler_hl$std["Heating_Load"])
scale_params_cl <- list(mean = scaler_cl$mean["Cooling_Load"], sd = scaler_cl$std["Cooling_Load"])


# Check scaled data summary 

print(summary(head(features_train_scaled)))

print(summary(data.frame(HL_scaled=target_hl_train_scaled, CL_scaled=target_cl_train_scaled)))

# --- Prepare data frames for training ---
# Caret models (DT, RF, SVM) still use scaled features and ORIGINAL targets
train_data_hl_caret <- cbind(features_train_scaled, Heating_Load = target_hl_train)
train_data_cl_caret <- cbind(features_train_scaled, Cooling_Load = target_cl_train)

# NeuralNet needs SCALED features and SCALED targets
colnames(features_train_scaled) <- make.names(colnames(features_train_scaled)) # Sanitize names
colnames(features_test_scaled) <- make.names(colnames(features_test_scaled))   # Sanitize names
train_data_hl_nn <- cbind(features_train_scaled, Scaled_Heating_Load = target_hl_train_scaled)
train_data_cl_nn <- cbind(features_train_scaled, Scaled_Cooling_Load = target_cl_train_scaled)



# -------------------------------------------------------------------
# 4. Model Training
# -------------------------------------------------------------------
# --- Setup Caret Training Control ---
train_control <- trainControl(method = "cv",      # Cross-validation
                              number = 10,     # 10 folds
                              savePredictions = "final",
                              verboseIter = TRUE) # Show progress

# --- 4.1 Decision Tree (rpart) ---
# Train for Heating Load (HL)
# Explicitly use caret's train function
model_dt_hl <- caret::train(Heating_Load ~ ., data = train_data_hl_caret, method = "rpart", trControl = train_control, tuneLength = 10)

# Train for Cooling Load (CL)
model_dt_cl <- caret::train(Cooling_Load ~ .,
                            data = train_data_cl_caret,
                            method = "rpart",
                            trControl = train_control,
                            tuneLength = 10)

print(model_dt_hl)
rpart.plot(model_dt_hl$finalModel) # Nicer plot

print(model_dt_cl)
rpart.plot(model_dt_cl$finalModel) # Nicer plot


# --- 4.2 Random Forest (randomForest or ranger) ---

# Train for Heating Load (HL)
model_rf_hl <- caret::train(Heating_Load ~ .,
                            data = train_data_hl_caret,
                            method = "rf", # Or "ranger" which is faster
                            trControl = train_control,
                            tuneLength = 5, # Tunes mtry
                            importance = TRUE) # Calculate variable importance

# Train for Cooling Load (CL)
model_rf_cl <- caret::train(Cooling_Load ~ .,
                            data = train_data_cl_caret,
                            method = "rf",
                            trControl = train_control,
                            tuneLength = 5,
                            importance = TRUE)

print(model_rf_hl)
plot(varImp(model_rf_hl))
print(model_rf_cl)
plot(varImp(model_rf_cl))

# --- 4.3 Support Vector Machine (SVM with Radial Kernel) ---

# Train for Heating Load (HL)
model_svm_hl <- caret::train(Heating_Load ~ .,
                             data = train_data_hl_caret,
                             method = "svmRadial", # RBF Kernel
                             trControl = train_control,
                             tuneLength = 5) # Tunes C and sigma

# Train for Cooling Load (CL)
model_svm_cl <- caret::train(Cooling_Load ~ .,
                             data = train_data_cl_caret,
                             method = "svmRadial",
                             trControl = train_control,
                             tuneLength = 5)

print(model_svm_hl)
print(model_svm_cl)

# --- 4.4 Neural Network (neuralnet package) ---
# Define formula: Scaled_Target ~ Feature1 + Feature2 + ...
feature_names_nn <- paste(colnames(features_train_scaled), collapse = " + ")
formula_hl_nn <- as.formula(paste("Scaled_Heating_Load ~", feature_names_nn))
formula_cl_nn <- as.formula(paste("Scaled_Cooling_Load ~", feature_names_nn))

# Train neural network for HL
# hidden = c(5, 3) 
# stepmax: Increase significantly for potentially better convergence.
# threshold: Convergence threshold for error reduction.
# linear.output=TRUE for regression.
start_time_nn_hl <- Sys.time()
model_nn_hl <- neuralnet(
  formula_hl_nn,
  data = train_data_hl_nn,
  hidden = c(6, 4),      # Adjust architecture (layers, neurons)
  linear.output = TRUE, # For regression
  lifesign = "minimal", # Print progress ('full' for more detail)
  stepmax = 1e6,        # Increase maximum iterations
  threshold = 0.05      # Error reduction threshold for stopping
)
end_time_nn_hl <- Sys.time()
print(paste("NeuralNet HL training time:", round(difftime(end_time_nn_hl, start_time_nn_hl, units="secs")), "seconds"))

# Plot the HL network structure (might be complex for larger nets)
plot(model_nn_hl)

start_time_nn_cl <- Sys.time()
model_nn_cl <- neuralnet(
  formula_cl_nn,
  data = train_data_cl_nn,
  hidden = c(6, 4),      # Use same architecture or tune separately
  linear.output = TRUE,
  lifesign = "minimal",
  stepmax = 1e6,
  threshold = 0.05
)
end_time_nn_cl <- Sys.time()
print(paste("NeuralNet CL training time:", round(difftime(end_time_nn_cl, start_time_nn_cl, units="secs")), "seconds"))
plot(model_nn_cl)


# -------------------------------------------------------------------
# 5. Model Evaluation on Test Set
# -------------------------------------------------------------------

# --- Predictions ---

# DT (predicts on original scale as it was trained on it)
pred_dt_hl <- predict(model_dt_hl, newdata = features_test_scaled)
pred_dt_cl <- predict(model_dt_cl, newdata = features_test_scaled)
# RF (predicts on original scale)
pred_rf_hl <- predict(model_rf_hl, newdata = features_test_scaled)
pred_rf_cl <- predict(model_rf_cl, newdata = features_test_scaled)
# SVM (predicts on original scale)
pred_svm_hl <- predict(model_svm_hl, newdata = features_test_scaled)
pred_svm_cl <- predict(model_svm_cl, newdata = features_test_scaled)

# NeuralNet ---
# Predict using neuralnet::compute on SCALED test features
nn_pred_scaled_hl <- neuralnet::compute(model_nn_hl, as.data.frame(features_test_scaled))$net.result
nn_pred_scaled_cl <- neuralnet::compute(model_nn_cl, as.data.frame(features_test_scaled))$net.result

# Unscale the neuralnet predictions 
# Formula: original = (scaled * sd) + mean
pred_nn_hl <- (nn_pred_scaled_hl * scale_params_hl$sd) + scale_params_hl$mean
pred_nn_cl <- (nn_pred_scaled_cl * scale_params_cl$sd) + scale_params_cl$mean

# Convert predictions to vectors
pred_nn_hl <- as.vector(pred_nn_hl)
pred_nn_cl <- as.vector(pred_nn_cl)


# --- Calculate Metrics ---

calculate_metrics <- function(actual, predicted, model_name = "") {
  if(!is.numeric(actual)) actual <- as.numeric(as.character(actual))
  if(!is.numeric(predicted)) predicted <- as.numeric(as.character(predicted))
  
  valid_indices <- is.finite(actual) & is.finite(predicted)
  if(sum(!valid_indices) > 0) {
    warning(paste("Removed", sum(!valid_indices), "non-finite prediction/actual values for model:", model_name))
    actual <- actual[valid_indices]
    predicted <- predicted[valid_indices]
  }
  
  if(length(actual) < 2) {
    warning(paste("Not enough valid data points (< 2) to calculate metrics for model:", model_name))
    return(list(RMSE = NA, MAE = NA, R2 = NA))
  }
  
  rmse_val <- Metrics::rmse(actual, predicted)
  mae_val <- Metrics::mae(actual, predicted)
  sse <- sum((actual - predicted)^2)
  sst <- sum((actual - mean(actual))^2)
  r2_val <- ifelse(sst < .Machine$double.eps, NA, 1 - sse / sst) 
  if (!is.finite(r2_val)) r2_val <- NA
  return(list(RMSE = rmse_val, MAE = mae_val, R2 = r2_val))
}

# Calculate for all models using ORIGINAL scale test targets

metrics_dt_hl <- calculate_metrics(target_hl_test, pred_dt_hl, "DT HL")
metrics_dt_cl <- calculate_metrics(target_cl_test, pred_dt_cl, "DT CL")
metrics_rf_hl <- calculate_metrics(target_hl_test, pred_rf_hl, "RF HL")
metrics_rf_cl <- calculate_metrics(target_cl_test, pred_rf_cl, "RF CL")
metrics_svm_hl <- calculate_metrics(target_hl_test, pred_svm_hl, "SVM HL")
metrics_svm_cl <- calculate_metrics(target_cl_test, pred_svm_cl, "SVM CL")
# Use NeuralNet prediction variables and model name
metrics_nn_hl <- calculate_metrics(target_hl_test, pred_nn_hl, "NeuralNet HL")
metrics_nn_cl <- calculate_metrics(target_cl_test, pred_nn_cl, "NeuralNet CL")

# --- Compile Results Table ---
# Use NeuralNet metrics variables and label
results_hl <- data.frame(
  Model = c("Decision Tree", "Random Forest", "SVM", "NeuralNet"),
  RMSE = c(metrics_dt_hl$RMSE, metrics_rf_hl$RMSE, metrics_svm_hl$RMSE, metrics_nn_hl$RMSE),
  MAE = c(metrics_dt_hl$MAE, metrics_rf_hl$MAE, metrics_svm_hl$MAE, metrics_nn_hl$MAE),
  R2 = c(metrics_dt_hl$R2, metrics_rf_hl$R2, metrics_svm_hl$R2, metrics_nn_hl$R2)
)

results_cl <- data.frame(
  Model = c("Decision Tree", "Random Forest", "SVM", "NeuralNet"),
  RMSE = c(metrics_dt_cl$RMSE, metrics_rf_cl$RMSE, metrics_svm_cl$RMSE, metrics_nn_cl$RMSE),
  MAE = c(metrics_dt_cl$MAE, metrics_rf_cl$MAE, metrics_svm_cl$MAE, metrics_nn_cl$MAE),
  R2 = c(metrics_dt_cl$R2, metrics_rf_cl$R2, metrics_svm_cl$R2, metrics_nn_cl$R2)
)

# Print results rounded for clarity

print(results_hl %>% mutate(across(where(is.numeric), ~ round(., 3))))

print(results_cl %>% mutate(across(where(is.numeric), ~ round(., 3))))



# -------------------------------------------------------------------
# 6. Model Comparison and Visualization
# -------------------------------------------------------------------
# --- Visualizing Predictions vs Actuals (All Models) ---
# Create data frames for joining
actuals <- data.frame(
  ID = seq_along(target_hl_test),
  Actual_HL = target_hl_test,
  Actual_CL = target_cl_test
)
predictions <- data.frame(
  ID = seq_along(target_hl_test),
  DT_HL = pred_dt_hl, DT_CL = pred_dt_cl,
  RF_HL = pred_rf_hl, RF_CL = pred_rf_cl,
  SVM_HL = pred_svm_hl, SVM_CL = pred_svm_cl,
  NN_HL = pred_nn_hl, NN_CL = pred_nn_cl # Use NN predictions (already unscaled)
)

# Pivot longer for faceted plotting
plot_data <- predictions %>%
  pivot_longer(cols = -ID, names_to = c("Model", "Load"), names_sep = "_", values_to = "Predicted") %>%
  left_join(actuals %>% pivot_longer(cols = -ID, names_to = c("Type","Load"), names_sep = "_", values_to = "Actual"),
            by = c("ID", "Load")) %>%
  filter(Type == "Actual") %>%
  # Use dplyr::select explicitly here:
  dplyr::select(ID, Model, Load, Predicted, Actual)

# Check the resulting data frame
print(head(plot_data))
print(str(plot_data))


# Plot Predicted vs Actual for all models and both loads
plot_pred_actual <- ggplot(plot_data, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.5, size = 1) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  facet_grid(Load ~ Model, scales = "fixed") + # Use Model name "NN" here
  labs(title = "Predicted vs Actual Energy Loads by Model",
       x = "Actual Load", y = "Predicted Load") +
  theme_minimal(base_size = 9) +
  theme(strip.text = element_text(size = rel(0.9)),
        axis.text = element_text(size = rel(0.8)),
        plot.title = element_text(size = rel(1.1))) +
  coord_fixed(ratio = 1)
print(plot_pred_actual)


# --- Variable Importance (from RF model - NN doesn't provide easily) ---


# Get the importance object
imp_hl <- varImp(model_rf_hl, scale = FALSE)

# Check if importance extraction was successful and has the expected structure
if (!is.null(imp_hl) && !is.null(imp_hl$importance) && "Overall" %in% colnames(imp_hl$importance)) {
  
  # 1. Extract the importance data frame
  importance_data <- imp_hl$importance
  
  # 2. Create a new column for the feature names from the row names
  importance_data$Feature <- rownames(importance_data)
  
  # 3. Create the ggplot using this prepared data frame
  plot_imp_hl <- ggplot(importance_data, aes(x = reorder(Feature, Overall), y = Overall)) +
    # y = Overall maps the 'Overall' column from importance_data
    # x = reorder(Feature, Overall) maps the 'Feature' column, ordered by 'Overall'
    geom_col(fill = "steelblue") + # geom_col is simpler here
    coord_flip() + # Flips axes to make bars horizontal
    labs(title = "Variable Importance for HL (RF)",
         x = "Feature",            # Label for the y-axis AFTER flipping
         y = "Importance Score") + # Label for the x-axis AFTER flipping
    theme_minimal(base_size = 10)
  
  print(plot_imp_hl)
  
} else {
  warning("Could not extract variable importance for RF HL model or 'Overall' column not found.")
}

# Repeat the same process for the Cooling Load model (imp_cl)
imp_cl <- varImp(model_rf_cl, scale = FALSE)
if (!is.null(imp_cl) && !is.null(imp_cl$importance) && "Overall" %in% colnames(imp_cl$importance)) {
  
  importance_data_cl <- imp_cl$importance
  importance_data_cl$Feature <- rownames(importance_data_cl)
  
  plot_imp_cl <- ggplot(importance_data_cl, aes(x = reorder(Feature, Overall), y = Overall)) +
    geom_col(fill="darkorange") +
    coord_flip() +
    labs(title = "Variable Importance for CL (RF)", x = "Feature", y = "Importance Score") +
    theme_minimal(base_size = 10)
  print(plot_imp_cl)
  
} else {
  warning("Could not extract variable importance for RF CL model or 'Overall' column not found.")
}



# remove

rm(list=ls()) 

# ===================================================================
# End of Script
# ===================================================================