####################################################
#################0. PRE DATA STUFF##################
####################################################


#LOADING LIBRARIES
library(dplyr)
library(lmridge) #FOR lmridge()
library(broom) #FOR glance() AND tidy()
library(MASS) #FOR lm.ridge()
library(ggplot2) #FOR ggplot()
library(mgcv) #FOR gam()
library(rsample) #FOR initial_split() STRATIFIED RANDOM SAMPLING
library(e1071) #SVM LIBRARY
library(tidymodels) #INCLUDES parsnip PACKAGE FOR decision_tree()
library(caret) #FOR confusionMatrix()
library(rpart.plot)
library(tidymodels)
library(baguette) #FOR BAGGED TREES
library(xgboost) #FOR GRADIENT BOOSTING
library(caret) #FOR confusionMatrix()
library(vip) #FOR VARIABLE IMPORTANCE

#SET THE RANDOM SEED FOR REPRODUCIBILITY
set.seed(123)

#LOADING IN DATA
df<-read.csv('https://raw.githubusercontent.com/dikarusli/BUAN-381-Final-Project/refs/heads/main/BUAN-381-Dataset.csv')


#QUICK DATA ANALYSIS
View(df)
colnames(df)
anyNA(df)

M0 <- lm(MFEXP~., data = df)

cor(df)
#HIGHLY CORRELATED VARS: SQFT + NKWER
#CORRELATED VARS: WKHRS

summary(M0)
#SS VARS: (Intercept) + PUBCLIM + SQFT + WKHRS + NKHRS + DIM


#FRACTION OF DATA TO BE USED AS IN-SAMPLE TRAINING DATA
set.seed(123)
pin<-.7 #70% FOR TRAINING (IN-SAMPLE), 30% FOR TESTING (OUT-OF-SAMPLE)
pout<-0.15
obs_count<-dim(df)[1]

train_size <- ceiling(pin * obs_count)
valid_size <- floor(pout * obs_count)
test_size <- floor(pout * obs_count)

#RANDOMLY SHUFFLES THE ROW NUMBERS OF ORIGINAL DATASET
set.seed(123)
train_ind <- sample(obs_count, size = train_size)  # Training indices
remaining_ind <- setdiff(1:obs_count, train_ind)   # Remaining indices after training
test_ind <- sample(remaining_ind, size = test_size)  # Testing indices
valid_ind <- setdiff(remaining_ind, test_ind)  

#PULLING RANDOM ROWS FOR EACH SET
set.seed(123)
Train <- df[train_ind, ]
Test <- df[test_ind, ]
Valid <- df[valid_ind, ]

#CHECKING THE DIMENSIONS OF THE PARTITIONED DATA
dim(Train)
dim(Valid)
dim(Test)

####################################################
#################2. PRE DATA STUFF##################
####################################################

#RMSE
#Num. of SS values
#ADJR2

####################################################
###############3. CORRELATION MATRIX################
####################################################
cor(df)
correlations <- cor(df)
#Highest correlation: SQFT 0.78199830


####################################################
##########4. BIVARIATE REGRESSION MODELING##########
####################################################

### 4.a. BUILDING LINEAR MODEL FROM THE TRAINING DATA

#Error Metric: RMSE

#MODEL 1: Y=B0+B1X
cor(df$SQFT, df$MFEXP)
M1 <- lm(MFEXP ~ SQFT, Train)
summary(M1) #SUMMARY DIAGNOSTIC OUTPUT

#GENERATING PREDICTIONS ON THE TRAINING DATA
PRED_M1_IN <- predict(M1, Train) 
View(PRED_M1_IN) #VIEW IN-SAMPLE PREDICTIONS
View(M1$fitted.values) #FITTED VALUES ARE IN-SAMPLE PREDICTIONS

#GENERATING PREDICTIONS ON THE TEST DATA TO BENCHMARK OUT-OF-SAMPLE PERFORMANCE 
PRED_M1_OUT <- predict(M1, Valid) 
View(PRED_M1_OUT)

#COMPUTING / REPORTING IN-SAMPLE AND OUT-OF-SAMPLE ROOT MEAN SQUARED ERROR
(RMSE_M1_IN<-sqrt(sum((PRED_M1_IN-Train$MFEXP)^2)/length(PRED_M1_IN))) #computes in-sample error
(RMSE_M1_OUT<-sqrt(sum((PRED_M1_OUT-Valid$MFEXP)^2)/length(PRED_M1_OUT))) #computes out-of-sample 

#model does a better predicting in sample data

### 4.b. BUILDING A NON-LINEAR MODEL

#MODEL 2
M2 <- lm(MFEXP ~ SQFT + I(SQFT^2), Train)
summary(M2)

PRED_M2_IN <- predict(M2, Train) 
PRED_M2_OUT <- predict(M2, Valid) 
(RMSE_M2_IN<-sqrt(sum((PRED_M2_IN-Train$MFEXP)^2)/length(PRED_M2_IN)))
(RMSE_M2_OUT<-sqrt(sum((PRED_M2_OUT-Valid$MFEXP)^2)/length(PRED_M2_OUT)))

### 4.c. REGULARIZED MODEL
#BUILDING REG vs. UNREG MODELS FOR 2nd DEG POLY MODEL (M2)
M2_unreg<-lm(MFEXP~SQFT + I(SQFT^2),df) #BUILD UNREGULARIZE MODEL AS POINT OF COMPARISION

set.seed(123)
M2_reg<-lm.ridge(MFEXP~SQFT + I(SQFT^2), Train, lambda=seq(0,1,0.1)) #BUILD REGULARIZED MODEL
glance(M2_reg) #USING BROOM PACKAGE TO EXRACT OPTIMAL LAMBDA

#ADDITIONAL TUNING
set.seed(123)
M2_reg<-lm.ridge(MFEXP~SQFT + I(SQFT^2), Train, lambda=seq(0,10,0.01)) #BUILD REGULARIZED MODEL
M3<-M2_reg
glance(M3) #USING BROOM PACKAGE TO EXRACT OPTIMAL LAMBDA

M3.1<-lmridge(MFEXP~SQFT + I(SQFT^2), Train, lambda=seq(0,10,0.01))

#OPTIMAL LAMBDA = 8.07
optimal_lambda <- which.min(M3$GCV)
print(optimal_lambda)

coef_ridge <- coef(M3)[optimal_lambda,]
print(coef_ridge)

PRED_M3_IN <- as.vector(coef_ridge[1] + coef_ridge[2] * Train$SQFT + coef_ridge[3] * (Train$SQFT^2))
PRED_M3_OUT <- as.vector(coef_ridge[1] + coef_ridge[2] * Valid$SQFT + coef_ridge[3] * (Valid$SQFT^2))
(RMSE_M3_IN <- sqrt(mean((PRED_M3_IN - Train$MFEXP)^2)))
(RMSE_M3_OUT <- sqrt(mean((PRED_M3_OUT - Valid$MFEXP)^2)))


PRED_M3_IN <- predict(M3.1, Train) 
PRED_M3_OUT <- predict(M3.1, Valid) 
(RMSE_M3_IN <- sqrt(mean((PRED_M3_IN - Train$MFEXP)^2)))
(RMSE_M3_OUT <- sqrt(mean((PRED_M3_OUT - Valid$MFEXP)^2)))


#4.d. GENERALIZED ADDITIVE STRUCTURE - SPLINE
set.seed(123)
M4 <- gam(MFEXP ~ s(SQFT), data = Train, family = 'gaussian')
summary(M4)

PRED_M4_IN <- predict(M4, Train) 
PRED_M4_OUT <- predict(M4, Valid)
(RMSE_M4_IN <- sqrt(mean((PRED_M4_IN - Train$MFEXP)^2)))
(RMSE_M4_OUT <- sqrt(mean((PRED_M4_OUT - Valid$MFEXP)^2)))

### 4.e.BIVARIATE PLOT
x_min <-min(Train$SQFT)
x_max <-max(Train$SQFT)
x_grid <- seq(x_min,x_max,length.out = 100) #CREATES GRID OF X-AXIS VALUES

plot(Train$MFEXP ~ Train$SQFT, col='blue')



predictions_1 <- predict(M1, list(SQFT=x_grid))
predictions_2 <- predict(M2, list(SQFT=x_grid, SQFT2=(x_grid^2)))
predictions_3 <- (coef_ridge[1] + coef_ridge[2] * x_grid + coef_ridge[3] * (x_grid^2))
predictions_4 <- predict(M4, data.frame(SQFT = x_grid))

lines(x_grid, predictions_1, col='blue', lwd=3) #PLOTS M1
lines(x_grid, predictions_2, col='yellow', lwd=3) #PLOTS M2
lines(x_grid, predictions_3, col='green', lwd=3) #PLOTS M2_reg or M3
lines(x_grid, predictions_4, col='purple', lwd=3) #PLOTS M4
points(Valid$MFEXP ~ Valid$SQFT, col='red', pch=3, cex=.5)


TABLE_VAL_1 <- as.table(matrix(c(RMSE_M1_IN, RMSE_M2_IN, RMSE_M3_IN, RMSE_M4_IN, RMSE_M1_OUT, RMSE_M2_OUT, RMSE_M3_OUT, RMSE_M4_OUT), ncol=4, byrow=TRUE))
colnames(TABLE_VAL_1) <- c('LINEAR', 'REGULARIZED', 'SPLINE', 'GAMS')
rownames(TABLE_VAL_1) <- c('RMSE_IN', 'RMSE_OUT')
TABLE_VAL_1 #REPORT OUT-OF-SAMPLE ERRORS FOR BOTH HYPOTHESIS


####################################################
########5. MULTIVARIATE REGRESSION MODELING#########
####################################################

### 5.a. MULTIVARIATE LINEAR MODEL
correlations
summary(M0)

M5 <- lm(MFEXP ~ PUBCLIM + SQFT + WKHRS + NWKER + DIM, Train)
summary(M5)

PRED_M5_IN <- predict(M5, Train) 
PRED_M5_OUT <- predict(M5, Valid)
(RMSE_M5_IN <- sqrt(mean((PRED_M5_IN - Train$MFEXP)^2)))
(RMSE_M5_OUT <- sqrt(mean((PRED_M5_OUT - Valid$MFEXP)^2)))
### 5.b. REGULARIZATION
#BUILDING REG vs. UNREG MODELS FOR 2nd DEG POLY MODEL (M2)
M6 <- lm.ridge(MFEXP ~ PUBCLIM + SQFT + WKHRS + NWKER + DIM, Train, lambda=seq(0,1,0.1)) #BUILD REGULARIZED MODEL
glance(M6) #USING BROOM PACKAGE TO EXRACT OPTIMAL LAMBDA

#ADDITIONAL TUNING
set.seed(123)
M6 <- lm.ridge(MFEXP ~ PUBCLIM + SQFT + WKHRS + NWKER + DIM, Train, lambda=seq(0,10,0.01)) #BUILD REGULARIZED MODEL
glance(M6)
M6_Optimal_Lambda <- which.min(M6$GCV)
print(M6_Optimal_Lambda)

M6_coef_ridge <- coef(M6)[M6_Optimal_Lambda,]
print(M6_coef_ridge)

PRED_M6_IN <- as.vector(M6_coef_ridge[1] + M6_coef_ridge[2] * Train$PUBCLIM + M6_coef_ridge[3] * Train$SQFT +
                          M6_coef_ridge[4] * Train$WKHRS + M6_coef_ridge[5] * Train$NWKER + M6_coef_ridge[6] * Train$DIM)

PRED_M6_OUT <- as.vector(M6_coef_ridge[1] + M6_coef_ridge[2] * Valid$PUBCLIM + M6_coef_ridge[3] * Valid$SQFT +
                           M6_coef_ridge[4] * Valid$WKHRS + M6_coef_ridge[5] * Valid$NWKER + M6_coef_ridge[6] * Valid$DIM)

(RMSE_M6_IN <- sqrt(mean((PRED_M6_IN - Train$MFEXP)^2)))
(RMSE_M6_OUT <- sqrt(mean((PRED_M6_OUT - Valid$MFEXP)^2)))

### 5.c. MULTIVARIATE NONLINEAR MODEL
M7 <- lm(MFEXP ~ PUBCLIM + SQFT + WKHRS + NWKER + DIM, Train)

M7 <- lm(MFEXP ~ PUBCLIM + SQFT + WKHRS + NWKER + I(NWKER^2) + I(NWKER^3) + DIM, Train)

M7 <- lm(MFEXP ~ PUBCLIM + SQFT + I(SQFT^2) + log(WKHRS) + NWKER + I(NWKER^2) + I(NWKER^3) + DIM, Train)
summary(M7)

PRED_M7_IN <- predict(M7, Train)
PRED_M7_OUT <- predict(M7, Valid)
(RMSE_M7_IN <- sqrt(mean((PRED_M7_IN - Train$MFEXP)^2)))
(RMSE_M7_OUT <- sqrt(mean((PRED_M7_OUT - Valid$MFEXP)^2)))

### 5.d. SUPPORT VECTOR MACHINE

######ABOUT KERNELS FOR SVM CLASSIFIERS:######
#Let the linear signal be defined in the usual
#way as s=B1x1+...+Bkxk.  The kernels that can
#be used to capture nonlinear transformations
#are below:
#linear kernel (s): "linear"
#polynomial (gamma*s+coef0)^p : "polynomial"
#gaussian RBF (exp(-gamma*|u-v|^2)): "radial"
#sigmoid (tanh(gamma*s+coef0)): "sigmoid"
##############################################

kern_type<-"radial" #SPECIFY KERNEL TYPE
#PUBCLIM + SQFT + I(SQFT^2) + log(WKHRS) + NWKER + I(NWKER^2) + I(NWKER^3) + DIM
set.seed(123)
M8<- svm(MFEXP ~ PUBCLIM + SQFT + I(SQFT^2) + log(WKHRS) + NWKER + I(NWKER^2) + I(NWKER^3) + DIM,
                data = Train, 
                type = "eps-regression", #set to "eps-regression" for numeric prediction
                kernel = kern_type,
                cost=1,                   #REGULARIZATION PARAMETER
                gamma = 1/(ncol(Train)-1), #DEFAULT KERNEL PARAMETER
                coef0 = 0,                    #DEFAULT KERNEL PARAMETER
                degree= 3,                     #POLYNOMIAL KERNEL PARAMETER
                scale = FALSE)                #RESCALE DATA? (SET TO TRUE TO NORMALIZE)

print(M8) #DIAGNOSTIC SUMMARY

PRED_M8_IN <- predict(M8, Train)
PRED_M8_OUT <- predict(M8, Valid)
(RMSE_M8_IN <- sqrt(mean((PRED_M8_IN - Train$MFEXP)^2)))
(RMSE_M8_OUT <- sqrt(mean((PRED_M8_OUT - Valid$MFEXP)^2)))


#TUNING THE SVM BY CROSS-VALIDATION
set.seed(123)
tune_control<-tune.control(cross=10) #SET K-FOLD CV PARAMETERS
TUNE <- tune.svm(
          x = data.frame(
                  PUBCLIM = Train$PUBCLIM,
                  SQFT = Train$SQFT,
                  SQFT2 = Train$SQFT^2,
                  log_WKHRS = log(Train$WKHRS),
                  NWKER = Train$NWKER,
                  NWKER2 = Train$NWKER^2,
                  NWKER3 = Train$NWKER^3,
                  DIM = Train$DIM
                ),
                 y = Train[, "MFEXP"],
                 type = "eps-regression",
                 kernel = kern_type,
                 tunecontrol=tune_control,
                 cost=c(1, 10, 100, 1000), #REGULARIZATION PARAMETER
                 gamma = c(0.0001, 0.001, 0.01, 0.1), #KERNEL PARAMETER
                 coef0 = c(0,1),          #KERNEL PARAMETER
                 degree = 3)          #POLYNOMIAL KERNEL PARAMETER
print(TUNE) #OPTIMAL TUNING PARAMETERS FROM VALIDATION PROCEDURE

#RE-BUILD MODEL USING OPTIMAL TUNING PARAMETERS
set.seed(123)
M8_RETUNE <- svm(MFEXP ~ PUBCLIM + SQFT + I(SQFT^2) + log(WKHRS) + NWKER + I(NWKER^2) + I(NWKER^3) + DIM, 
                 data = Train, 
                 type = "eps-regression", 
                 kernel = kern_type,
                 degree = TUNE$best.parameters$degree,
                 gamma = TUNE$best.parameters$gamma,
                 coef0 = TUNE$best.parameters$coef0,
                 cost = TUNE$best.parameters$cost,
                 scale = FALSE)
print(M8) #DIAGNOSTIC SUMMARY
print(TUNE)
print(M8_RETUNE) #DIAGNOSTIC SUMMARY

PRED_M8_RETUNE_IN <- predict(M8_RETUNE, Train)
PRED_M8_RETUNE_OUT <- predict(M8_RETUNE, Valid)
(RMSE_M8_RETUNE_IN <- sqrt(mean((PRED_M8_RETUNE_IN - Train$MFEXP)^2)))
(RMSE_M8_RETUNE_OUT <- sqrt(mean((PRED_M8_RETUNE_OUT - Valid$MFEXP)^2)))


### 5.e. TREEEEEEE
#SPECIFYING THE REGRESSION TREE MODEL
reg_spec <- decision_tree(min_n = 20, #minimum number of observations for split
                          tree_depth = 30, #max tree depth
                          cost_complexity = 0.01)  %>% #regularization parameter
  set_engine("rpart") %>%
  set_mode("regression")
print(reg_spec)

#ESTIMATING THE MODEL (CAN BE DONE IN ONE STEP ABOVE WITH EXTRA %>%)
reg_fmla <- MFEXP ~ PUBCLIM + SQFT + I(SQFT^2) + log(WKHRS) + NWKER + I(NWKER^2) + I(NWKER^3) + DIM
reg_tree <- reg_spec %>%
  fit(formula = reg_fmla, data = Train)
print(reg_tree)

#VISUALIZING THE REGRESSION TREE
reg_tree$fit %>%
  rpart.plot(type = 4, extra = 1, roundint = FALSE)

#GENERATE PREDICTIONS AND COMBINE WITH VALID SET
PRED_M9_OUT <- predict(reg_tree, new_data = Valid) %>%
  bind_cols(Valid)

# Calculate RMSE for the validation set
RMSE_M9_OUT <- PRED_M9_OUT %>%
  metrics(truth = MFEXP, estimate = .pred) %>%
  filter(.metric == "rmse") %>%
  pull(.estimate)

# Generate predictions for the training set
PRED_M9_IN <- predict(reg_tree, new_data = Train) %>%
  bind_cols(Train)

# Calculate RMSE for the training set
RMSE_M9_IN <- PRED_M9_IN %>%
  metrics(truth = MFEXP, estimate = .pred) %>%
  filter(.metric == "rmse") %>%
  pull(.estimate)

RMSE_M9_IN
RMSE_M9_OUT


# #MANUAL TREE TUNING
# reg_spec <- decision_tree(min_n = 5, #minimum number of observations for split
#                           tree_depth = 5, #max tree depth
#                           cost_complexity = 0.005)  %>% #regularization parameter
#   set_engine("rpart") %>%
#   set_mode("regression")
# print(reg_spec)

# #ESTIMATING THE MODEL (CAN BE DONE IN ONE STEP ABOVE WITH EXTRA %>%)
# reg_fmla <- MFEXP ~ PUBCLIM + SQFT + I(SQFT^2) + log(WKHRS) + NWKER + I(NWKER^2) + I(NWKER^3) + DIM
# reg_tree <- reg_spec %>%
#   fit(formula = reg_fmla, data = Train)
# print(reg_tree)

# #VISUALIZING THE REGRESSION TREE
# reg_tree$fit %>%
#   rpart.plot(type = 4, extra = 1, roundint = FALSE)
# 
# #GENERATE PREDICTIONS AND COMBINE WITH VALID SET
# pred_reg <- predict(reg_tree, new_data = Valid) %>%
#   bind_cols(Valid)
# 
# #OUT-OF-SAMPLE ERROR ESTIMATES FROM yardstick OR ModelMetrics PACKAGE
# mae(pred_reg, estimate=.pred, truth=MFEXP) 
# rmse(pred_reg, estimate=.pred, truth=MFEXP)




#########################
##TUNING THE TREE MODEL##
#########################
set.seed(123)
nzv <- nearZeroVar(Train, saveMetrics = TRUE)
nzv

# reg_fmla <- MFEXP ~  SQFT + I(SQFT^2) + NWKER + I(NWKER^2) + I(NWKER^3)
# #BLANK TREE SPECIFICATION FOR TUNING
# tree_spec <- decision_tree(min_n = tune(),
#                            tree_depth = tune(),
#                            cost_complexity= tune()) %>%
#   set_engine("rpart") %>%
#   set_mode("regression")
# 
# #CREATING A TUNING PARAMETER GRID
# tree_grid <- grid_regular(parameters(tree_spec), levels = 3)
# #tree_grid <- grid_random(parameters(tree_spec), size = 3) #FOR RANDOM GRID
# 
# #TUNING THE MODEL ALONG THE GRID W/ CROSS-VALIDATION
# tune_results <- tune_grid(tree_spec,
#                           reg_fmla, #MODEL FORMULA
#                           resamples = vfold_cv(Train, v=3), #RESAMPLES / FOLDS
#                           grid = tree_grid, #GRID
#                           #metrics = metric_set(rmse)  #BENCHMARK METRIC
#                           )
# 
# #RETRIEVE OPTIMAL PARAMETERS FROM CROSS-VALIDATION
# best_params <- select_best(tune_results)




library(tidymodels)

# Formula for the model
reg_fmla <- MFEXP ~ SQFT + I(SQFT^2) + NWKER + I(NWKER^2) + I(NWKER^3)

# Tree specification with parameters to tune
tree_spec <- decision_tree(
  min_n = tune(),
  tree_depth = tune(),
  cost_complexity = tune()
) %>%
  set_engine("rpart") %>%
  set_mode("regression")

# Create tuning parameter grid
tree_grid <- grid_regular(
  parameters(tree_spec),
  levels = 3
)

# Create cross-validation folds
cv_folds <- vfold_cv(Train, v = 3)

# Perform tuning
set.seed(123)
tune_results <- tune_grid(
  tree_spec,
  reg_fmla, # Model formula
  resamples = cv_folds,
  grid = tree_grid,
  metrics = metric_set(rmse) # Benchmark metric
)

# Retrieve optimal parameters
best_params <- select_best(tune_results, metric = "rmse")

# Print results
print(best_params)






#FINALIZE THE MODEL SPECIFICATION
final_spec <- finalize_model(tree_spec, best_params)

#FIT THE FINALIZED MODEL
final_model <- final_spec %>% fit(reg_fmla, Train)

#VISUALIZING THE REGRESSION TREE
final_model$fit %>%
  rpart.plot(type = 4, extra = 1, roundint = FALSE)


#GENERATE PREDICTIONS AND COMBINE WITH VALID SET
PRED_M9_RETUNE_OUT <- predict(final_model, new_data = Valid) %>%
  bind_cols(Valid)

# Calculate RMSE for the validation set
RMSE_M9_RETUNE_OUT <- PRED_M9_RETUNE_OUT %>%
  metrics(truth = MFEXP, estimate = .pred) %>%
  filter(.metric == "rmse") %>%
  pull(.estimate)

# Generate predictions for the training set
PRED_M9_RETUNE_IN <- predict(final_model, new_data = Train) %>%
  bind_cols(Train)

# Calculate RMSE for the training set
RMSE_M9_RETUNE_IN <- PRED_M9_RETUNE_IN %>%
  metrics(truth = MFEXP, estimate = .pred) %>%
  filter(.metric == "rmse") %>%
  pull(.estimate)
RMSE_M9_RETUNE_IN
RMSE_M9_RETUNE_OUT


### 5.f. 
bf_fmla <- MFEXP ~ SQFT + I(SQFT^2) + NWKER + I(NWKER^2) + I(NWKER^3)

##############################
#SPECIFYING BAGGED TREE MODEL#
##############################
set.seed(123)
spec_bagged <- bag_tree(min_n = 20 , #minimum number of observations for split
                        tree_depth = 30, #max tree depth
                        cost_complexity = 0.01, #regularization parameter
                        class_cost = NULL)  %>% #for output class imbalance adjustment (binary data only)
  set_mode("regression") %>% #can set to regression for numeric prediction
  set_engine("rpart", times=100) #times = # OF ENSEMBLE MEMBERS IN FOREST
print(spec_bagged)

#FITTING THE MODEL
bagged_forest <- spec_bagged %>% fit(formula = bf_fmla, data = Train)
print(bagged_forest)

# Increase the limit
options(future.globals.maxSize = 1e9)

# Run the model tuning


#GENERATE PREDICTIONS AND COMBINE WITH VALID SET
PRED_M10_OUT <- predict(bagged_forest, new_data = Valid) %>%
  bind_cols(Valid)

# Calculate RMSE for the validation set
RMSE_M10_OUT <- PRED_M10_OUT %>%
  metrics(truth = MFEXP, estimate = .pred) %>%
  filter(.metric == "rmse") %>%
  pull(.estimate)

# Generate predictions for the training set
PRED_M10_IN <- predict(bagged_forest, new_data = Train) %>%
  bind_cols(Train)

# Calculate RMSE for the training set
RMSE_M10_IN <- PRED_M10_IN %>%
  metrics(truth = MFEXP, estimate = .pred) %>%
  filter(.metric == "rmse") %>%
  pull(.estimate)
RMSE_M10_IN
RMSE_M10_OUT



tree_spec <- decision_tree(
  min_n = tune(),
  tree_depth = tune(),
  cost_complexity = tune()
) %>%
  set_engine("rpart") %>%
  set_mode("regression")

# Create tuning parameter grid
tree_grid <- grid_regular(
  parameters(tree_spec),
  levels = 3
)

# Create cross-validation folds
cv_folds <- vfold_cv(Train, v = 3)

# Perform tuning
set.seed(123)
tune_results <- tune_grid(
  tree_spec,
  bf_fmla, # Model formula
  resamples = cv_folds,
  grid = tree_grid,
  metrics = metric_set(rmse) # Benchmark metric
)

# Retrieve optimal parameters
best_params <- select_best(tune_results, metric = "rmse")

# Print results
print(best_params)


#FINALIZE THE MODEL SPECIFICATION
final_spec <- finalize_model(tree_spec, best_params)

#FIT THE FINALIZED MODEL
final_model <- final_spec %>% fit(bf_fmla, Train)

#VISUALIZING THE REGRESSION TREE
final_model$fit %>%
  rpart.plot(type = 4, extra = 1, roundint = FALSE)

#GENERATE PREDICTIONS AND COMBINE WITH Train SET
PRED_M10_RETUNE_IN <- predict(final_model, new_data = Train) %>%
  bind_cols(Train)

# Calculate RMSE for the training set
RMSE_M10_RETUNE_IN <- PRED_M10_RETUNE_IN %>%
  metrics(truth = MFEXP, estimate = .pred) %>%
  filter(.metric == "rmse") %>%
  pull(.estimate)


#GENERATE PREDICTIONS AND COMBINE WITH VALID SET
PRED_M10_RETUNE_OUT <- predict(final_model, new_data = Valid) %>%
  bind_cols(Valid)

# Calculate RMSE for the training set
RMSE_M10_RETUNE_OUT <- PRED_M10_RETUNE_OUT %>%
  metrics(truth = MFEXP, estimate = .pred) %>%
  filter(.metric == "rmse") %>%
  pull(.estimate)

print(RMSE_M10_RETUNE_OUT)
print(RMSE_M10_RETUNE_IN)



### 5.g TABLE SUMMARY
TABLE_VAL_3 <- as.table(matrix(c(
  RMSE_M1_IN, RMSE_M2_IN, RMSE_M3_IN, RMSE_M4_IN, RMSE_M5_IN, RMSE_M6_IN, RMSE_M7_IN, RMSE_M8_IN, RMSE_M8_RETUNE_IN, RMSE_M9_IN, RMSE_M9_RETUNE_IN, RMSE_M10_IN, RMSE_M10_RETUNE_OUT,
  RMSE_M1_OUT, RMSE_M2_OUT, RMSE_M3_OUT, RMSE_M4_OUT, RMSE_M5_OUT, RMSE_M6_OUT, RMSE_M7_OUT, RMSE_M8_OUT, RMSE_M8_RETUNE_OUT, RMSE_M9_OUT, RMSE_M9_RETUNE_OUT, RMSE_M10_OUT, RMSE_M10_RETUNE_OUT
), ncol = 13, byrow = TRUE))

colnames(TABLE_VAL_3) <- c('M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M8_RETUNE', 'M9', 'M9_RETUNE', 'M10', 'M10_RETUNE')
rownames(TABLE_VAL_3) <- c('RMSE_IN', 'RMSE_OUT')
TABLE_VAL_3

TABLE_VAL_4 <- as.table(matrix(c(
  RMSE_M1_IN, RMSE_M2_IN, RMSE_M3_IN, RMSE_M4_IN, RMSE_M5_IN, RMSE_M6_IN, RMSE_M7_IN, RMSE_M8_IN, RMSE_M8_RETUNE_IN, RMSE_M9_IN, RMSE_M9_RETUNE_IN, RMSE_M10_IN, RMSE_M10_RETUNE_OUT,
  RMSE_M1_OUT, RMSE_M2_OUT, RMSE_M3_OUT, RMSE_M4_OUT, RMSE_M5_OUT, RMSE_M6_OUT, RMSE_M7_OUT, RMSE_M8_OUT, RMSE_M8_RETUNE_OUT, RMSE_M9_OUT, RMSE_M9_RETUNE_OUT, RMSE_M10_OUT, RMSE_M10_RETUNE_OUT
), ncol = 13, byrow = TRUE))
colnames(TABLE_VAL_4) <- c('BV Linear', 'BV Nonlinear REG', 'SPLINE', 'GAM', 'MV Linear', 'MV Linear REG', 'MV Nonlinear', 'SVM', 'SVM_RETUNE', 'Reg_Tree', 'Reg_Tree_RETUNE', 'Bag_Forest', 'Bag_Forest_RETUNE')
rownames(TABLE_VAL_4) <- c('RMSE_IN', 'RMSE_OUT')
TABLE_VAL_4




##BEST REGRESSION MODEL: MULTIVARIATE NONLINEAR MODEL
BEST_REG <- lm(MFEXP ~ PUBCLIM + SQFT + I(SQFT^2) + log(WKHRS) + NWKER + I(NWKER^2) + I(NWKER^3) + DIM, Test)
summary(BEST_REG)

PRED_BEST_REG_IN <- predict(BEST_REG, Train)
PRED_BEST_REG_OUT <- predict(BEST_REG, Test)
(RMSE_BEST_REG_IN <- sqrt(mean((PRED_BEST_REG_IN - Train$MFEXP)^2)))
(RMSE_BEST_REG_OUT <- sqrt(mean((PRED_BEST_REG_OUT - Test$MFEXP)^2)))
