#Install packages
library(dplyr)
library(caret)
#library(e1071)
library(pROC)
library(caTools)


#Utilise German Credit Data file from caret with pre-cleansed data 
#Set TARGET - BAD to 1 - i.e. we are going to predict defaults as the outcome
data(GermanCredit)
GCD <- tbl_df(GermanCredit) %>%
  mutate(ID = row_number(),
         Target = ifelse(Class == "Bad",1,0)) %>%
  select(ID,Class,Target,everything())

# Remove the Class variable, since they are replaced by the Target column
# ID probably doesn't help as it doesn't provide predictive information
# GCD <- GCD[, -which(grepl("ID|Class", names(GCD)))]

# Create a partition of 80% of the rows for training, 20% for validation
# Note there is no validation in the traditional SAS EM manner; instead CV training is used to optimise model
set.seed(42)
Partition_Index <- createDataPartition(GCD$Target,
                                       p=0.80, list=FALSE)
GCD_Training <- GCD[Partition_Index,]
GCD_Validation <- GCD[-Partition_Index,]


#(Check relative proportions of the target variables in the split data)
prop.table(table(GCD_Training$Target))
prop.table(table(GCD_Validation$Target))

#Build a train control object
GCD_TrCtrl <- trainControl(method="cv",
                           summaryFunction=defaultSummary, 
                           classProbs=T,
                           savePredictions = T)
#Add in pre-processing: imputation and normalisation

#Train Logistic regression model (family = Binomial, method=glm) glmStepAIC

# ensure proper variable class to Factor to perform classification instead of regression?
GCD_Training$Target = as.factor(GCD_Training$Target)
GCD_Validation$Target = as.factor(GCD_Validation$Target)

# GCD_Model_LR <- train(as.factor(Class) ~
#                      . -Target -ID,
#                    data=GCD_Training,
#                    method="glm",
#                 metric = "Accuracy",
#                    family="binomial",
#                    trControl=GCD_TrCtrl)
# warnings(GCD_Model_LR)

#Show model variable importance

# predictors(GCD_Model_LR)
varImportanceCheck = varImp(GCD_Model_LR)$importance
varImportanceCheck$predictor = row.names(varImportanceCheck)
varImportanceCheck = varImportanceCheck[order(varImportanceCheck$Overall, decreasing = F), ]

# plot variable importance to estimate thresholds by which to select
# active variable subset on second pass!
plot(varImp(GCD_Model_LR))
# plot(x=1:nrow(varImportanceCheck), y=varImportanceCheck$Overall)


# re-evaluate model based on new variables

# GCD_Training_REDUX = ""
# GCD_Training_REDUX = ""
#     
# GCD_Model_LR_REDUX <- train(Target ~
#         . -Class -ID,
#     data=GCD_Training[, varImportanceCheck],
#     method="glm",
#     family="binomial",
#     trControl=GCD_TrCtrl)

# stop()
# reset model

#Aply model to validation data set - Logistic Regression
GCD_Pred_LR.obj <-predict(GCD_Model_LR, GCD_Validation)
GCD_Predict_LR <- GCD_Validation %>%
  select(ID,Observed=Target) %>%
  mutate(Predicted = ifelse(GCD_Pred_LR.obj > 0.5, 1,0),
         Pred_Prob = GCD_Pred_LR.obj) %>%
  arrange(desc(Pred_Prob))

#ConfusionMatrix on Validation data
confusionMatrix(data = GCD_Predict_LR$Predicted,
                reference = GCD_Predict_LR$Observed)



# #Aply model to validation data set - Decision Tree
# GCD_Pred_DT.obj <-predict(GCD_Model_DT, GCD_Validation)
# GCD_Predict_DT <- GCD_Validation %>%
#   select(ID,Observed=Target) %>%
#   mutate(Predicted = ifelse(GCD_Pred_DT.obj > 0.5, 1,0),
#          Pred_Prob = GCD_Pred_DT.obj) %>%
#   arrange(desc(Pred_Prob))
# 
# #ConfusionMatrix on Validation data
# confusionMatrix(data = GCD_Predict_LR$Predicted,
#                 reference = GCD_Predict_LR$Observed)



#Output AUC values
ROC.Training_LR <- roc(GCD_Model_LR$pred$obs, GCD_Model_LR$pred$pred)
ROC.Validation_LR <- roc(GCD_Predict_LR$Observed, GCD_Predict_LR$Pred_Prob)
print("TRAINING"); auc(ROC.Training_LR)
print("VALIDATION"); auc(ROC.Validation_LR)

#Plot ROC curves
plot(ROC.Training_LR, col = "blue",
     print.auc = TRUE, print.auc.y = .4)
plot(ROC.Validation_LR, col = "magenta",
     print.auc = TRUE, print.auc.y = .2,
     add = TRUE)



