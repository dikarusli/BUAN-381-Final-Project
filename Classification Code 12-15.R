library(dplyr)
library(caret)      # For confusionMatrix() and data partitioning
library(broom)      # For glance() and tidy()
library(MASS)       # For lm.ridge()
library(ggplot2)    # For ggplot()
library(mgcv)       # For gam()
library(rsample)    # For initial_split() stratified random sampling
library(e1071)      # SVM library
library(pROC)       # For ROC and AUC
library(tidymodels) # For ROC curve plotting with yardstick
library(rpart.plot)

df0 <- read.csv('https://raw.githubusercontent.com/dikarusli/BUAN-381-Final-Project/refs/heads/main/BUAN-381-Dataset.csv')
df <- read.csv('https://raw.githubusercontent.com/dikarusli/BUAN-381-Final-Project/refs/heads/main/BUAN-381-Dataset.csv')

# Correlation analysis
cor(df)

# Convert the response variable 'SOUSED' to a factor
df$SOUSED <- factor(df$SOUSED)
class(df$SOUSED)

df$REGION <- factor(df$REGION)
df$CENDIV <- factor(df$CENDIV)
df$PUBCLIM <- factor(df$PUBCLIM)
df$NGUSED <- factor(df$NGUSED)
df$FKUSED <- factor(df$FKUSED)
df$PRUSED <- factor(df$PRUSED)
df$SOUSED <- factor(df$SOUSED)
df$NGHT1 <- factor(df$NGHT1)
df$NGGENR <- factor(df$REGION)
df$WBOARDS <- factor(df$WBOARDS)
df$FLUOR <- factor(df$FLUOR)
df$OCSN <- factor(df$OCSN)
df$DIM <- factor(df$DIM)

View(df)
View(df0)

# Quick Data Analysis
print(colnames(df))
print(anyNA(df))
summary(df)

# Set the seed for reproducibility
set.seed(123)

split <- initial_split(df, prop = 0.7, strata = SOUSED)
Train <- training(split)
Temp <- testing(split)

# Temp into Validation and Test sets (each 15% of the total data)
validation_test_split <- initial_split(Temp, prop = 0.5, strata = SOUSED)
Valid <- training(validation_test_split)
Test <- testing(validation_test_split)


# Dimensions of partitioned data
print(dim(Train))
print(dim(Valid))
print(dim(Test))

#######################################
######## 8.a LOGIT REGRESSION #########
#######################################
M0 <- glm(SOUSED ~ ., data = Train, family = binomial(link = "logit"))
summary(M0)

M1_Logit <- glm(SOUSED ~ REGION + NWKER + I(NWKER^2), data = Train, family = binomial(link = "logit"))
M1_Logit <- glm(SOUSED ~ FLUOR, data = Train, family = binomial(link = "logit"))
summary(M1_Logit)

sum(df$SOUSED == 1)/nrow(df)
sum(Train$SOUSED == 1)/nrow(Train)
sum(Valid$SOUSED == 1)/nrow(Valid)

#TEST TO SEE IF VARS ARE A GOOD PREDICTOR
Test_Stat<-M1_Logit$null.deviance-M1_Logit$deviance #difference in deviance
Test_df<-M1_Logit$df.null-M1_Logit$df.residual #difference in degrees of freedom
1-pchisq(Test_Stat, Test_df) #p-value for null hypothesis H_0:
### Very small P value, reject H0 and accept HA: model is a good predictor

##construct confidence intervals using one of two methods:
confint(M1_Logit)  #using profiled log-likelihood
confint.default(M1_Logit) #using standard errors

##look at confidence interval next to point estimates AND
##converting values to odds-ratio interpretation using base e
#takes the coefficients to the base e for odds-ratio interpretation
point_conf_table<-cbind(M1_Logit$coefficients, confint(M1_Logit))
exp(point_conf_table)

#in one step
exp(cbind(M1_Logit$coefficients, confint(M1_Logit)))

###Look at the new coefficients -- a value greater than 1 shows an increase (x-1 = pos) and a value less than 1 will 
#show a decrease (x-1 = negative). A value of 2 is a 100% increase (2-1 = 1)

M1_Logit <- glm(SOUSED ~ REGION + NWKER + I(NWKER^2), data = Train, family = binomial(link = "logit"))
summary(M1_Logit)
# View(M1_Logit)
# sum(M1_Logit$fitted.values <= 0.5)

#generating predicted probabilities
M1_Logit_Pred_Train <-predict(M1_Logit, Train, type="response")
M1_Logit_Pred_valid <-predict(M1_Logit, Valid, type="response")

#converts predictions to boolean TRUE (1) or FALSE (0) based on 1/2 threshold on output probability
M1_Logit_BinPred_Train <- factor(M1_Logit_Pred_Train >= 0.1, levels = c(FALSE, TRUE))
M1_Logit_BinPred_Valid <- factor(M1_Logit_Pred_valid >= 0.1, levels = c(FALSE, TRUE))


M1_Logit_Actual_Train <- factor(Train$SOUSED == 1, levels = c(FALSE, TRUE))
M1_Logit_Actual_Valid <- factor(Valid$SOUSED == 1, levels = c(FALSE, TRUE))



View(M1_Logit_BinPred_Train)
View(M1_Logit_BinPred_Valid)

  
###columns are ACTUAL values (including type 1 and type 2 error): 184 students are ACTUALLY rejected and 96 are ACTUALLY accepted
###rows is how many we SHOULD admit/reject according to our prediction

#display summary analysis of confusion matrix in-sample
M1_Logit_CM_IN <- confusionMatrix(M1_Logit_BinPred_Train, M1_Logit_Actual_Train, positive= 'TRUE')
M1_Logit_CM_OUT <- confusionMatrix(M1_Logit_BinPred_Valid, M1_Logit_Actual_Valid, positive= 'TRUE')
M1_Logit_CM_IN
M1_Logit_CM_OUT


#######################################
####### 8.b PROBIT REGRESSION #########
#######################################
M1_Probit <- glm(SOUSED ~ REGION + NWKER + I(NWKER^2), data = Train, family = binomial(link = "probit"))
#M1_Probit <- glm(SOUSED ~ FLUOR, data = Train, family = binomial(link = "probit"))
summary(M1_Probit)

#TEST TO SEE IF VARS ARE A GOOD PREDICTOR
Test_Stat<-M1_Probit$null.deviance-M1_Probit$deviance #difference in deviance
Test_df<-M1_Probit$df.null-M1_Probit$df.residual #difference in degrees of freedom
1-pchisq(Test_Stat, Test_df) #p-value for null hypothesis H_0:
### Very small P value, reject H0 and accept HA: model is a good predictor

##construct confidence intervals using one of two methods:
confint(M1_Probit)  #using profiled log-likelihood
confint.default(M1_Probit) #using standard errors

##look at confidence interval next to point estimates AND
##converting values to odds-ratio interpretation using base e
#takes the coefficients to the base e for odds-ratio interpretation
point_conf_table<-cbind(M1_Probit$coefficients, confint(M1_Probit))
exp(point_conf_table)

#in one step
exp(cbind(M1_Probit$coefficients, confint(M1_Probit)))

###Look at the new coefficients -- a value greater than 1 shows an increase (x-1 = pos) and a value less than 1 will 
#show a decrease (x-1 = negative). A value of 2 is a 100% increase (2-1 = 1)

M1_Probit <- glm(SOUSED ~ REGION + NWKER + I(NWKER^2), data = Train, family = binomial(link = "probit"))
summary(M1_Probit)



#generating predicted probabilities
M1_Probit_Pred_Train <-predict(M1_Probit, Train, type="response")
M1_Probit_Pred_valid <-predict(M1_Probit, Valid, type="response")

# View(M1_Probit_Pred_Train)
# View(M1_Probit_Pred_valid)

#converts predictions to boolean TRUE (1) or FALSE (0) based on 1/2 threshold on output probability
M1_Probit_BinPred_Train <- factor(M1_Probit_Pred_Train >= 0.1, levels = c(FALSE, TRUE))
M1_Probit_BinPred_Valid <- factor(M1_Probit_Pred_valid >= 0.1, levels = c(FALSE, TRUE))


M1_Probit_Actual_Train <- factor(Train$SOUSED == 1, levels = c(FALSE, TRUE))
M1_Probit_Actual_Valid <- factor(Valid$SOUSED == 1, levels = c(FALSE, TRUE))

View(M1_Probit_BinPred_Train)
View(M1_Probit_BinPred_Valid)


###columns are ACTUAL values (including type 1 and type 2 error): 184 students are ACTUALLY rejected and 96 are ACTUALLY accepted
###rows is how many we SHOULD admit/reject according to our prediction

#display summary analysis of confusion matrix in-sample
M1_Probit_CM_IN <- confusionMatrix(M1_Probit_BinPred_Train, M1_Probit_Actual_Train, positive= 'TRUE')
M1_Probit_CM_OUT <- confusionMatrix(M1_Probit_BinPred_Valid, M1_Probit_Actual_Valid, positive= 'TRUE')
M1_Probit_CM_IN
M1_Probit_CM_OUT




#############################
####### 8.c ROC + AOC #######
#############################


#BUILD A DF WITH PREDICTIONS (PROBABILITIES) VS. ACTUAL VALUES (AS FACTORS)

###LOGIT
M1_Logit_pva_IN<-data.frame(preds=M1_Logit_Pred_Train, actual=factor(Train$SOUSED))
M1_Logit_pva_OUT<-data.frame(preds=M1_Logit_Pred_valid, actual=factor(Valid$SOUSED))

#ROC
roc_obj_IN <- roc(M1_Logit_pva_IN$actual, M1_Logit_pva_IN$preds)

#NOTE THIS PLOTS SENSITIVITY (TRUE POSITIVES) VS. SPECIFICITY (TRUE NEGATIVES)
plot(roc_obj_IN, col='blue', main="LOGIT IN ROC Curve")

#AUC
Logit_ROC_IN <- roc_obj_IN



roc_obj_OUT <- roc(M1_Logit_pva_OUT$actual, M1_Logit_pva_OUT$preds)
plot(roc_obj_OUT, col='blue', main="LOGIT OUT ROC Curve")
Logit_ROC_OUT <- roc_obj_OUT



###PROBIT
M1_Probit_pva_IN<-data.frame(preds=M1_Probit_Pred_Train, actual=factor(Train$SOUSED))
M1_Probit_pva_OUT<-data.frame(preds=M1_Probit_Pred_valid, actual=factor(Valid$SOUSED))

#ROC
roc_obj_IN <- roc(M1_Probit_pva_IN$actual, M1_Probit_pva_IN$preds)

#NOTE THIS PLOTS SENSITIVITY (TRUE POSITIVES) VS. SPECIFICITY (TRUE NEGATIVES)
plot(roc_obj_IN, col='blue', main="Probit IN ROC Curve")

#AUC
Probit_ROC_IN <- roc_obj_IN

roc_obj_OUT <- roc(M1_Probit_pva_OUT$actual, M1_Probit_pva_OUT$preds)
plot(roc_obj_OUT, col='blue', main="Probit OUT ROC Curve")
Probit_ROC_OUT <- roc_obj_OUT

print(Logit_ROC_IN)
print(Logit_ROC_OUT)
print(Probit_ROC_IN)
print(Probit_ROC_OUT)




#######################
####### 9.a SVM #######
#######################
M1_Logit <- glm(SOUSED ~ REGION + NWKER + I(NWKER^2), data = Train, family = binomial(link = "logit"))

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

#BUILD SVM CLASSIFIER
set.seed(123)
SVM_Model<- svm(SOUSED ~ REGION + NWKER + I(NWKER^2), 
                data = Train, 
                type = "C-classification", #set to "eps-regression" for numeric prediction
                kernel = kern_type,
                cost=1,                   #REGULARIZATION PARAMETER
                gamma = 1/(ncol(training)-1), #DEFAULT KERNEL PARAMETER
                coef0 = 0,                    #DEFAULT KERNEL PARAMETER
                degree=2,                     #POLYNOMIAL KERNEL PARAMETER
                scale = FALSE)                #RESCALE DATA? (SET TO TRUE TO NORMALIZE)

print(SVM_Model) #DIAGNOSTIC SUMMARY

#REPORT IN AND OUT-OF-SAMPLE ERRORS (1-ACCURACY)
(E_IN_PRETUNE<-1-mean(predict(SVM_Model, Train)==Train$SOUSED))
(E_OUT_PRETUNE<-1-mean(predict(SVM_Model, Test)==Test$SOUSED))

#TUNING THE SVM BY CROSS-VALIDATION
tune_control<-tune.control(cross=10) #SET K-FOLD CV PARAMETERS
set.seed(123)
TUNE <- tune.svm(x = data.frame(REGION_NUM=as.numeric(Train$REGION), NWKER=Train$NWKER, NWKER2=I((Train$NWKER)^2)),
                 y = Train$SOUSED,
                 type = "C-classification",
                 kernel = kern_type,
                 tunecontrol=tune_control,
                 cost=c(.01, .1, 1, 10, 100, 1000), #REGULARIZATION PARAMETER
                 gamma = 1/(ncol(Train)-1), #KERNEL PARAMETER
                 coef0 = c(0,1),           #KERNEL PARAMETER
                 degree = 2)          #POLYNOMIAL KERNEL PARAMETER


#RE-BUILD MODEL USING OPTIMAL TUNING PARAMETERS
SVM_Retune<- svm(SOUSED ~ REGION + NWKER + I(NWKER^2), 
                 data = Train, 
                 type = "C-classification", 
                 kernel = kern_type,
                 degree = TUNE$best.parameters$degree,
                 gamma = TUNE$best.parameters$gamma,
                 coef0 = TUNE$best.parameters$coef0,
                 cost = TUNE$best.parameters$cost,
                 scale = FALSE)

print(SVM_Retune) #DIAGNOSTIC SUMMARY

#REPORT IN AND OUT-OF-SAMPLE ERRORS (1-ACCURACY) ON RETUNED MODEL
(E_IN_RETUNE<-1-mean(predict(SVM_Retune, Train)==Train$SOUSED))
(E_OUT_RETUNE<-1-mean(predict(SVM_Retune, Valid)==Valid$SOUSED))


(E_IN_PRETUNE<-1-mean(predict(SVM_Model, Train)==Train$SOUSED))
(E_OUT_PRETUNE<-1-mean(predict(SVM_Model, Test)==Test$SOUSED))
(E_IN_RETUNE<-1-mean(predict(SVM_Retune, Train)==Train$SOUSED))
(E_OUT_RETUNE<-1-mean(predict(SVM_Retune, Valid)==Valid$SOUSED))


########################################
####### 9.b. CLASSIFICATION TREE #######
########################################

################################################################################
fmla <- SOUSED ~ REGION + NWKER + I(NWKER^2)

#SPECIFYING AND FITTING THE CLASSIFICATION TREE MODEL WITH DEFAULT PARAMETERS
default_tree <- decision_tree(min_n = 20 , #minimum number of observations for split
                              tree_depth = 30, #max tree depth
                              cost_complexity = 0.01)  %>% #regularization parameter
  set_engine("rpart") %>%
  set_mode("classification") %>%
  fit(fmla, Train)

#GENERATE IN-SAMPLE PREDICTIONS ON THE TRAIN SET AND COMBINE WITH TRAIN DATA
pred_class_in <- predict(default_tree, new_data = Train, type="class") %>%
  bind_cols(Train) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

#GENERATE IN-SAMPLE CONFUSION MATRIX AND DIAGNOSTICS
confusion <- table(pred_class_in$.pred_class, pred_class_in$SOUSED)
View(confusion)
confusionMatrix(confusion) #FROM CARET PACKAGE

#GENERATE OUT-OF-SAMPLE PREDICTIONS ON THE TEST SET AND COMBINE WITH TEST DATA
pred_class_out <- predict(default_tree, new_data = Valid, type="class") %>%
  bind_cols(Valid) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

#GENERATE OUT-OF-SAMPLE CONFUSION MATRIX AND DIAGNOSTICS
confusion <- table(pred_class_out$.pred_class, pred_class_out$SOUSED)
confusionMatrix(confusion) #FROM CARET PACKAGE

# #GENERATE ROC CURVE AND COMPUTE AUC OVER ALL TRUE / FALSE +'s
# autoplot(roc_curve(pred_class_in, estimate=.pred_1, truth=SOUSED))
# roc_auc(pred_class_in, estimate=.pred_1, truth=SOUSED)

#########################
##TUNING THE TREE MODEL##
#########################

#BLANK TREE SPECIFICATION FOR TUNING
tree_spec <- decision_tree(min_n = tune(),
                           tree_depth = tune(),
                           cost_complexity= tune()) %>%
  set_engine("rpart") %>%
  set_mode("classification")

#CREATING A TUNING PARAMETER GRID
tree_grid <- grid_regular(parameters(tree_spec), levels = 3)
#tree_grid <- grid_random(parameters(tree_spec), size = 3) FOR RANDOM GRID

#TUNING THE MODEL ALONG THE GRID W/ CROSS-VALIDATION
set.seed(123) #SET SEED FOR REPRODUCIBILITY WITH CROSS-VALIDATION
tune_results <- tune_grid(tree_spec,
                          fmla, #MODEL FORMULA
                          resamples = vfold_cv(Train, v=3), #RESAMPLES / FOLDS
                          grid = tree_grid, #GRID
                          metrics = metric_set(accuracy)) #BENCHMARK METRIC

#RETRIEVE OPTIMAL PARAMETERS FROM CROSS-VALIDATION
best_params <- select_best(tune_results)

#FINALIZE THE MODEL SPECIFICATION
final_spec <- finalize_model(tree_spec, best_params)

#FIT THE FINALIZED MODEL
final_model <- final_spec %>% fit(fmla, Train)

#GENERATE IN-SAMPLE PREDICTIONS ON THE TRAIN SET AND COMBINE WITH TRAIN DATA
pred_class_in <- predict(final_model, new_data = Train, type="class") %>%
  bind_cols(Train) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

#GENERATE IN-SAMPLE CONFUSION MATRIX AND DIAGNOSTICS
confusion <- table(pred_class_in$.pred_class, pred_class_in$SOUSED)
confusionMatrix(confusion) #FROM CARET PACKAGE

#GENERATE OUT-OF-SAMPLE PREDICTIONS ON THE TEST SET AND COMBINE WITH TEST DATA
pred_class_out <- predict(final_model, new_data = Valid, type="class") %>%
  bind_cols(Valid) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

#GENERATE OUT-OF-SAMPLE CONFUSION MATRIX AND DIAGNOSTICS
confusion <- table(pred_class_out$.pred_class, pred_class_out$SOUSED)
confusionMatrix(confusion) #FROM CARET PACKAGE

################################################################################
# #generating predicted probabilities
# M1_Logit_Pred_Train <-predict(default_tree, Train, type="class")
# M1_Logit_Pred_valid <-predict(M1_Logit, Valid, type="response")
# 
# #converts predictions to boolean TRUE (1) or FALSE (0) based on 1/2 threshold on output probability
# M1_Logit_BinPred_Train <- factor(M1_Logit_Pred_Train >= 0.1, levels = c(FALSE, TRUE))
# M1_Logit_BinPred_Valid <- factor(M1_Logit_Pred_valid >= 0.1, levels = c(FALSE, TRUE))
# 
# 
# M1_Logit_Actual_Train <- factor(Train$SOUSED == 1, levels = c(FALSE, TRUE))
# M1_Logit_Actual_Valid <- factor(Valid$SOUSED == 1, levels = c(FALSE, TRUE))

#View(M1_Logit_BinPred_Train)
#View(M1_Logit_BinPred_Valid)
################################################################################


####################################
##########GRAPHING THE TREE##########
####################################

class_spec <- decision_tree(min_n = 1 , #minimum number of observations for split
                            tree_depth = 30, #max tree depth
                            cost_complexity = 0.01)  %>% #regularization parameter
  set_engine("rpart") %>%
  set_mode("classification")
print(class_spec)

#ESTIMATING THE MODEL (CAN BE DONE IN ONE STEP ABOVE WITH EXTRA %>%)
class_fmla <- SOUSED ~ REGION + NWKER + I(NWKER^2)
class_tree <- class_spec %>%
  fit(formula = class_fmla, data = Train)
print(class_tree)

#VISUALIZING THE CLASSIFICATION TREE MODEL:
class_tree$fit %>%
  rpart.plot(type = 4, extra = 2, roundint = FALSE)

plotcp(class_tree$fit)




##############################
#SPECIFYING BAGGED TREE MODEL#
##############################
library(baguette) #FOR BAGGED TREES

fmla <- SOUSED ~ REGION + NWKER + I(NWKER^2)

spec_bagged <- bag_tree(min_n = 20 , #minimum number of observations for split
                        tree_depth = 30, #max tree depth
                        cost_complexity = 0.01, #regularization parameter
                        class_cost = NULL)  %>% #for output class imbalance adjustment (binary data only)
  set_mode("classification") %>% #can set to regression for numeric prediction
  set_engine("rpart", times=100) #times = # OF ENSEMBLE MEMBERS IN FOREST
spec_bagged

#FITTING THE MODEL
set.seed(123)
bagged_forest <- spec_bagged %>%
  fit(formula = fmla, data = Train)
print(bagged_forest)

# Increase the limit
options(future.globals.maxSize = 1e9)

#GENERATE IN-SAMPLE PREDICTIONS ON THE TRAIN SET AND COMBINE WITH TRAIN DATA
pred_class_bf_in <- predict(bagged_forest, new_data = Train, type="class") %>%
  bind_cols(Train) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

#GENERATE IN-SAMPLE CONFUSION MATRIX AND DIAGNOSTICS
confusion <- table(pred_class_bf_in$.pred_class, pred_class_bf_in$SOUSED)
confusionMatrix(confusion) #FROM CARET PACKAGE

#GENERATE OUT-OF-SAMPLE PREDICTIONS ON THE TEST SET AND COMBINE WITH TEST DATA
pred_class_bf_out <- predict(bagged_forest, new_data = Valid, type="class") %>%
  bind_cols(Valid) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

#GENERATE OUT-OF-SAMPLE CONFUSION MATRIX AND DIAGNOSTICS
confusion <- table(pred_class_bf_out$.pred_class, pred_class_bf_out$SOUSED)
confusionMatrix(confusion) #FROM CARET PACKAGE


###################################################################
fmla <- SOUSED ~ REGION + NWKER + I(NWKER^2)


#BLANK TREE SPECIFICATION FOR TUNING
tree_spec <- decision_tree(min_n = tune(),
                           tree_depth = tune(),
                           cost_complexity= tune()) %>%
  set_engine("rpart") %>%
  set_mode("classification")

#CREATING A TUNING PARAMETER GRID
tree_grid <- grid_regular(parameters(tree_spec), levels = 3)
#tree_grid <- grid_random(parameters(tree_spec), size = 3) FOR RANDOM GRID

#TUNING THE MODEL ALONG THE GRID W/ CROSS-VALIDATION
set.seed(123) #SET SEED FOR REPRODUCIBILITY WITH CROSS-VALIDATION
tune_results <- tune_grid(tree_spec,
                          fmla, #MODEL FORMULA
                          resamples = vfold_cv(Train, v=10), #RESAMPLES / FOLDS
                          grid = tree_grid, #GRID
                          metrics = metric_set(accuracy)) #BENCHMARK METRIC

#RETRIEVE OPTIMAL PARAMETERS FROM CROSS-VALIDATION
best_params <- select_best(tune_results)

#FINALIZE THE MODEL SPECIFICATION
final_spec <- finalize_model(tree_spec, best_params)

#FIT THE FINALIZED MODEL
final_model <- final_spec %>% fit(fmla, Train)

#GENERATE IN-SAMPLE PREDICTIONS ON THE TRAIN SET AND COMBINE WITH TRAIN DATA
pred_class_in <- predict(final_model, new_data = Train, type="class") %>%
  bind_cols(Train) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

#GENERATE IN-SAMPLE CONFUSION MATRIX AND DIAGNOSTICS
confusion <- table(pred_class_in$.pred_class, pred_class_in$SOUSED)
confusionMatrix(confusion) #FROM CARET PACKAGE

#GENERATE OUT-OF-SAMPLE PREDICTIONS ON THE TEST SET AND COMBINE WITH TEST DATA
pred_class_out <- predict(final_model, new_data = Valid, type="class") %>%
  bind_cols(Valid) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

#GENERATE OUT-OF-SAMPLE CONFUSION MATRIX AND DIAGNOSTICS
confusion <- table(pred_class_out$.pred_class, pred_class_out$SOUSED)
confusionMatrix(confusion) #FROM CARET PACKAGE












