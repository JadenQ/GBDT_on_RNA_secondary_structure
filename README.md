# gbdt+LR_on_predicting_RNA_secondary_structure
##CTR.R:
Use xgboost + Linear regression in the model, 

1.using function xgb.create.features, the xgb model is used to recreate features from one-hot code

2.using xgbLinear to select the most important variables.(feature selection)

3.using linear regression model to predict based on the selected predictors
