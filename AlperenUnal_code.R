set.seed(123)
# Assigning manually imported datasets:
df_train = creditdefault_train
df_test = creditdefault_test

# Number of inputs
input_nums_train = nrow(df_train)
input_nums_test = nrow(df_test)

# Number of Variables
var_nums_train = ncol(df_train)
var_nums_test = ncol(df_test)

cat("Number of inputs for training set:", input_nums_train)
cat("Number of variables for training set:", var_nums_train, "\n")
cat("Number of inputs for test set:", input_nums_test)
cat("Number of variables for test set:", var_nums_test, "\n")

# Checking the missing values
colSums(is.na(df_train))
colSums(is.na(df_test))

# Types of variables
sapply(df_train, class)
sapply(df_test, class)


# Variables names need to be changed because of straightforward understandability.
new_names = c('Defaulter', 'AMOUNT_BAL', 'GENDER', 'EDUCATION', 'MARRIAGE', 'AGE', 
              'REPAY_SEPT', 'REPAY_AUG','REPAY_JULY', 'REPAY_JUNE', 'REPAY_MAY', 
              'REPAY_APR','BILL_SEPT', 'BILL_AUG', 'BILL_JULY', 'BILL_JUNE', 
              'BILL_MAY', 'BILL_APR','PAID_SEPT', 'PAID_AUG', 'PAID_JULY', 
              'PAID_JUNE', 'PAID_MAY', 'PAID_APR')
colnames(df_train) = new_names
colnames(df_test) = new_names

# The first step is detecting the probability of defaulters and non-defaulters.
# Creating a histogram with specific breaks for binary data.
hist_data = hist(df_train$Defaulter, breaks = c(-0.5, 0.5, 1.5), 
                  col = "navy", main = "Histogram of Defaulters", 
                  xlab = "Non-Defaulter and Defaulter", xlim = c(-0.5, 1.5), 
                  right = FALSE)

# Calculating the total number of defaulters and non-defaulters
total = sum(hist_data$counts)
proportions = hist_data$counts / total
text(x = c(0, 1), y = hist_data$counts, labels = round(proportions, 2), 
     pos = 3, cex = 0.8, col = "red")

# As be seen default payment proportion is %22. The Dataset is imbalanced this
# situation might effect future predictions as overfitting.

# The next step is checking the other variables' values.
summary(df_train)

# In terms of the summary's information, there are some undocumented and anomalous 
# values into the dataset. These columns need to be more deeply investigated.

unique_edu = unique(df_train$EDUCATION)
unique_edu
# As be seen there are undocumented values in the variable which are 0, 5 and 6.
# These values might be considered as 'other' which means '4'.
# Now 0, 5 and 6 will be embedded into 4.

df_train$EDUCATION[df_train$EDUCATION %in% c(0, 5, 6)] = 4
unique_edu = unique(df_train$EDUCATION)
unique_edu

# Doing same method to the test set:
df_test$EDUCATION[df_test$EDUCATION %in% c(0, 5, 6)] = 4
unique_edu = unique(df_test$EDUCATION)
unique_edu

# The same undocumented value issue is encountered by the MARRIAGE variable.
# The 0 value should be embedded into 3 value.
df_train$MARRIAGE[df_train$MARRIAGE == 0] = 3
unique_mar = unique(df_train$MARRIAGE)
unique_mar

df_test$MARRIAGE[df_test$MARRIAGE == 0] = 3
unique_mar = unique(df_test$MARRIAGE)
unique_mar


# The other anomalous situation occurs in the 'History of past payment' section.
# Those variables are 'REPAY_SEPT', 'REPAY_AUG',..., 'REPAY_APR'.
# Now, these variables should be more deeply investigated.


par(mfrow = c(2, 3))
hist(df_train$REPAY_SEPT, main = "Histogram for REPAY_SEPT", xlab = "REPAY_SEPT", col = "blue")
hist(df_train$REPAY_AUG, main = "Histogram for REPAY_AUG", xlab = "REPAY_AUG", col = "red")
hist(df_train$REPAY_JULY, main = "Histogram for REPAY_JULY", xlab = "REPAY_JULY", col = "green")
hist(df_train$REPAY_JUNE, main = "Histogram for REPAY_JUNE", xlab = "REPAY_JUNE", col = "orange")
hist(df_train$REPAY_MAY, main = "Histogram for REPAY_MAY", xlab = "REPAY_MAY", col = "purple")
hist(df_train$REPAY_APR, main = "Histogram for REPAY_APR", xlab = "REPAY_APR", col = "brown")
par(mfrow = c(1, 1))

# As can be seen, all the charts show that '0' is an outlier. In terms of 
# the proportion of defaulters, this 0 value could be considered as 'duly payment'.
# But this approaching can cause the overfitting due to this reason the values
# can remain as they are.

# Let's see other uniques' counts:
table(df_train$REPAY_APR)
table(df_train$REPAY_MAY)
table(df_train$REPAY_JUNE)
table(df_train$REPAY_JULY)
table(df_train$REPAY_AUG)
table(df_train$REPAY_SEPT)


# Function to calculate proportions.
calculate_proportion = function(col) {
  sum_neg = sum(col %in% c(-1, -2, 0))
  return(sum_neg / 15000)
}
# Apply the function to each column.
repay_columns = c('REPAY_SEPT', 'REPAY_AUG','REPAY_JULY', 'REPAY_JUNE', 'REPAY_MAY', 
            'REPAY_APR')
proportions = sapply(df_train[repay_columns], calculate_proportion)
print(proportions)

mean_of_proportions = sum(proportions) / 6
print(mean_of_proportions)

# Proportion of these values may increase the overfitting effect for the ML models.
# Due to this reason, I don't manipulate these values for now.
# I will consider them when I create models.


# For more deeply understanding, correlation between 'Defaulter' and the other 
# values should be investigated.

correlations = cor(df_train[ , -which(names(df_train) == "Defaulter")], df_train$Defaulter)
print(correlations)

# I will sort them with their absolute values for careful observation 
abs_correlations = abs(correlations)
names(abs_correlations) = names(df_train[ , -which(names(df_train) == "Defaulter")])
sorted_correlations = sort(abs_correlations, decreasing = TRUE)
print(sorted_correlations)

# According to these coefficients, history of past payment variables have 
# the highest correlation values with the target variable.
# After that the second one is amount of the given credit variable.

# Now needs to be created some visualisations 
# The dataset's all variables are numeric. But some labels are categorical.
# I don't want to change their data types but I need to separate them and
# I need to interpret them under the different approaching since I will separate them.

cat_cols = c('GENDER', 'EDUCATION', 'MARRIAGE', 'REPAY_SEPT', 'REPAY_AUG',
             'REPAY_JULY', 'REPAY_JUNE', 'REPAY_MAY', 'REPAY_APR')
target_col = 'Defaulter'
con_cols = setdiff(names(df_train), c(cat_cols, target_col))

cat("Categorical Columns:", cat_cols, "\n")
cat("Continuous Columns:", con_cols)

# I will use ggplot2 for creating graphs.
library(ggplot2)

# Dataset needs to reshape from wide to long format because of using all the columns.
library(tidyr)

long_df <- pivot_longer(df_train,
                        cols = all_of(cat_cols),
                        names_to = "CategoricalVariable",
                        values_to = "Value")

ggplot(long_df, aes(x = CategoricalVariable, fill = as.factor(Defaulter))) +
  geom_bar(position = "fill") +
  facet_wrap(~CategoricalVariable, scales = "free_x") +
  labs(title = "Stacked Bar Chart of Categorical Variables by Defaulter",
       x = "Categorical Variable", y = "Proportion")

# The Stacked Bar Chart shows all the zeros and ones are devided by same proportion.
# This visualisations can be interpretted in three ways:
# 1: One of the categories in 'Defaulter' overwhelmingly dominates.
# 2: There may be a problem with how the data is being plotted.
# 3: It is possible that the different categorical variables have very similar distributions.

# Let's see the total distribution af all categories first:
library(dplyr)

long_df <- df_train %>% 
  pivot_longer(cols = all_of(cat_cols), names_to = "Category", values_to = "Count")

ggplot(long_df, aes(x = as.factor(Count), fill = as.factor(Defaulter))) +
  geom_bar(position = 'stack') +
  facet_wrap(~Category, scales = 'free_x') +
  labs(title = "Total Distribution of Categorical Variables by Defaulter Status",
       x = "Category",
       y = "Total Count") +
  theme_minimal() +
  theme(legend.position = "bottom")
# The chart indicates some insights but it is not sufficient. 
# Let's see some count numbers and percentages:

for (col in cat_cols) {
  cat_table <- table(df_train[[col]], df_train$Defaulter)
  cat_percent <- prop.table(cat_table, margin = 1) # Calculate percentages row-wise
  
  print(paste("Variable:", col))
  print(cat_table)
  print(paste("Percentages:", col))
  print(cat_percent)
}

# According to graphs and percentages above, we can interpret these values more precisely:
# GENDER: Men have a slightly higher default rate than women, even though there are more women in the dataset.
# EDUCATION: Individuals with higher education levels generally have lower default rates, but the 'others' category defies this trend with an unusually low default rate.
# MARRIAGE: Married individuals have a marginally higher default rate than single individuals.
# REPAY_ Variables: Timely payments correlate with lower default rates. A two-month payment delay is a significant indicator of default risk, with default rates sharply increasing for such delays.
# Overall, repayment behavior is the strongest indicator of default risk among the variables, while demographic factors like gender, education, and marital status show more nuanced relationships with defaulting.

# In light of the analysis and graphs above, -2 can be considered as the full paid balance
# or inactive credit cards and 0 as the minimum amount of payment. 
# Let's explain the reason for this thought with various manual observations:

repay_vars = c('REPAY_SEPT', 'REPAY_AUG', 'REPAY_JULY', 'REPAY_JUNE', 'REPAY_MAY', 'REPAY_APR')
subset_df = subset(df_train, (REPAY_SEPT == 0) & (REPAY_AUG == 0) & 
                      (REPAY_JULY == 0) & (REPAY_JUNE == 0) & 
                      (REPAY_MAY == 0) & (REPAY_APR == 0))
# Now sample n rows from this subset
n = 10  
sampled_data = subset_df[sample(nrow(subset_df), n), ]
print(sampled_data)

# As can be seen from the example I chose, there is a difference of 10 to 20 times between 
# the bill statements and previous payments of individuals with a value of '0' in REPAY_variables.
# Despite such small payments, most of these individuals' credit card default payments
# are considered non-defaulters. For this reason, the value '0' in 
# the History of past payment data can be considered as the minimum amount of payment.

subset_df_2 = subset(df_train, (REPAY_SEPT == -2) & (REPAY_AUG == -2) & 
                     (REPAY_JULY == -2) & (REPAY_JUNE == -2) & 
                     (REPAY_MAY == -2) & (REPAY_APR == -2))
n = 10  
sampled_data_2 = subset_df_2[sample(nrow(subset_df_2), n), ]
print(sampled_data_2)

# When we apply the same process as the previous one with the '-2' variable, we see the following:
# It is seen that the bill statements and payment amounts are very close to each other,
# and the majority of the sample are seen as non-defaulters.
# Therefore, a value of '-2' can be considered as a fully paid balance or 
# inactive credit card.
# After analyzing 0 and '-2' values, my curiosity about the '-1' value increased. That's why I will apply the same steps to '-1'. 

subset_df_1 = subset(df_train, (REPAY_SEPT == -1) & (REPAY_AUG == -1) & 
                       (REPAY_JULY == -1) & (REPAY_JUNE == -1) & 
                       (REPAY_MAY == -1) & (REPAY_APR == -1))
n = 10  
sampled_data_1 = subset_df_1[sample(nrow(subset_df_1), n), ]
print(sampled_data_1)

# For the '-1' value, consistency is evident between bill statements and payment amounts. In other words, it shows similar features to -2. 
# As I mentioned, '-1', '-2', and '0' can be considered 'duly payment'. Because all the clients who have '-2', '-1', and '0' in their past payment history paid sufficient money for their cards. Due to this reason, I will combine all three values and name those as '0'. From now on, '0' = 'pay duly' in this project.
for (column in repay_columns) {
  df_train[[column]][df_train[[column]] %in% c(-1, -2)] = 0
}
lapply(df_train[repay_columns], unique)

# Doing same procedure for test set:
for (column in repay_columns) {
  df_test[[column]][df_test[[column]] %in% c(-1, -2)] = 0
}
lapply(df_test[repay_columns], unique)

# Now, let's start work on the continuous variables.

print(con_cols)
print(sorted_correlations)

# Deeply observation for "AMOUNT_BAL" (Amount of the given credit).
# Violin Plot:
ggplot(df_train, aes(x = as.factor(Defaulter), y = AMOUNT_BAL)) + 
  geom_violin(trim = FALSE) +
  labs(title = "Defaulter vs AMOUNT_BAL", x = "Defaulter", y = "AMOUNT_BAL")
#The plot indicates that lower balance amounts are more common for both groups, 
# but non-defaulters have a slightly more pronounced concentration at these lower amounts. 
# The long tails in both violins suggest that there are individuals in both groups with
# very high balances, although this is somewhat more common among non-defaulters.

# Density Plot:
ggplot(df_train, aes(x = AMOUNT_BAL, fill = as.factor(Defaulter))) + 
  geom_density(alpha = 0.7) +
  labs(title = "Density Plot of Defaulter vs AMOUNT_BAL", x = "AMOUNT_BAL", y = "Density")
# The plot reinforces these observations, showing high-density regions at lower balances 
# for defaulters and non-defaulters, with non-defaulters peaking more sharply,
# implying that non-defaulters are more likely to have lower balances overall. 
# The spread of the balance amounts is wider among defaulters, 
# suggesting more variability in the balance amounts within this group.



# All continuous variables' Box plots in the same visualisation:
long_data = df_train %>%
  pivot_longer(cols = con_cols, names_to = "variable", values_to = "value")
ggplot(long_data, aes(x = as.factor(Defaulter), y = value)) +
  geom_boxplot() +
  facet_wrap(~variable, scales = "free") +
  labs(title = "Continuous Variables vs Defaulter", x = "Defaulter", y = "Value") +
  theme_bw() +
  theme(strip.text = element_text(size = 8))
print(ggplot(long_data, aes(x = as.factor(Defaulter), y = value)) +
        geom_boxplot() +
        facet_wrap(~variable, scales = "free") +
        labs(title = "Continuous Variables vs Defaulter", x = "Defaulter", y = "Value") +
        theme_bw() +
        theme(strip.text = element_text(size = 8)))

colnames(df_train)


# The box plots indicate that defaulters tend to have higher median balances and bills, 
# suggesting a possible correlation between financial strain and defaulting.
# On the other hand, there doesn't seem to be any significant difference in age between defaulters and non-defaulters. 
# It's worth noting that the financial behavior of defaulters varies considerably, 
# with several outliers indicating individual differences in financial management.
# Overall, these findings suggest that while higher financial obligations may be associated with defaulting,
# it's important to take into account individual circumstances that can greatly impact financial behavior.

# I observed all the variables but I need to use more clear method for detecting
# relationship between target value and predictive variables.
# Pearson correlation is often but not the most informative measure for
# binary target and some categorical variables.

#Let's use logistic regression for selecting the most propar variables for our models

glm.fits = glm(Defaulter ~ AMOUNT_BAL + GENDER + EDUCATION + MARRIAGE + AGE + 
               REPAY_SEPT + REPAY_AUG + REPAY_JULY + REPAY_JUNE + REPAY_MAY + REPAY_APR + 
               BILL_SEPT + BILL_AUG + BILL_JULY + BILL_JUNE + BILL_MAY + BILL_APR + 
               PAID_SEPT + PAID_AUG + PAID_JULY + PAID_JUNE + PAID_MAY + PAID_APR,
             family = binomial, data = df_train)
summary(glm.fits)

reduced_model = step(glm.fits, direction = "backward")
summary(reduced_model)

# In terms of the backward selection, 
# AMOUNT_BAL, GENDER, MARRIAGE, REPAY_SEPT, REPAY_MAY, REPAY_APR, BILL_SEPT, 
# BILL_AUG, PAID_SEPT, PAID_AUG, and PAID_JULY have low p-values suggest a strong statistical significance.
# REPAY_SEPT shows the strongest positive association with defaulting, indicating its crucial role in predicting the outcome.
# AMOUNT_BAL has a negative coefficient, implying that higher balance amounts might be associated with a lower likelihood of defaulting.
# The model appears to fit the data well, as indicated by the substantial drop in deviance from the null model to the residual model. 
# The AIC of 13348 suggests the model's relative quality.

# Let's create a new model with this significant variables.
# Removing not significant variables which have a higher p-value from the model (in terms of backward selection)

glm.fits2 = glm(Defaulter ~ AMOUNT_BAL + GENDER + MARRIAGE + 
                 REPAY_SEPT + REPAY_JULY + REPAY_MAY + REPAY_APR + 
                 BILL_SEPT + BILL_AUG + PAID_SEPT + PAID_AUG + 
                 PAID_JULY,
               family = binomial, data = df_train)
summary(glm.fits2)

reduced_model2 = step(glm.fits2, direction = "backward")
summary(reduced_model2)
# 1- The initial model is more complex with more predictors. The reduced model, obtained through backward elimination, is simpler due to fewer predictors.
# 2- Both models exhibit a similar level of fit, as shown by close values of residual deviance and AIC. The reduced model, with a similar or slightly lower AIC, suggests efficient predictive power despite fewer variables.
# 3- The simpler reduced model is easier to interpret and use practically, while still maintaining high prediction accuracy.
# 4- The reduced model offers a more streamlined and potentially more effective version of the initial model, maintaining important predictive factors while reducing complexity.

#According to comparisons above, new datasets can be crated.
df_train_new = df_train[, !(names(df_train) %in% c("EDUCATION", "AGE", "REPAY_AUG",
                                                    "REPAY_JULY", "REPAY_JUNE", "BILL_JULY",
                                                    "BILL_JUNE", "BILL_MAY", "BILL_APR",
                                                    "PAID_JUNE", "PAID_MAY", "PAID_APR"))]

df_test_new = df_test[, !(names(df_test) %in% c("EDUCATION", "AGE", "REPAY_AUG",
                                                   "REPAY_JULY", "REPAY_JUNE", "BILL_JULY",
                                                   "BILL_JUNE", "BILL_MAY", "BILL_APR",
                                                   "PAID_JUNE", "PAID_MAY", "PAID_APR"))]

# FITTING CLASSIFICATION TREES
install.packages("tree")
library(tree)

df_train_new$Defaulter = factor(df_train_new$Defaulter, levels = c(0,1), labels = c("non-def", "def"))
df_train_new$GENDER = factor(df_train_new$GENDER, levels = c(1, 2), labels = c("male", "female"))
df_train_new$MARRIAGE = factor(df_train_new$MARRIAGE, levels = c(1, 2, 3), labels = c("married", "single", "others"))
df_train_new$REPAY_SEPT = factor(df_train_new$REPAY_SEPT)
df_train_new$REPAY_MAY = factor(df_train_new$REPAY_MAY)
df_train_new$REPAY_APR = factor(df_train_new$REPAY_APR)

df_test_new$Defaulter = factor(df_test_new$Defaulter, levels = c(0,1), labels = c("non-def", "def"))
df_test_new$GENDER = factor(df_test_new$GENDER, levels = c(1, 2), labels = c("male", "female"))
df_test_new$MARRIAGE = factor(df_test_new$MARRIAGE, levels = c(1, 2, 3), labels = c("married", "single", "others"))
df_test_new$REPAY_SEPT = factor(df_test_new$REPAY_SEPT)
df_test_new$REPAY_MAY = factor(df_test_new$REPAY_MAY)
df_test_new$REPAY_APR = factor(df_test_new$REPAY_APR)

# Training dataset

tree_model = tree(Defaulter ~ ., data = df_train_new)
# Plotting
plot(tree_model)
text(tree_model, pretty = 0)

summary(tree_model)
tree_model

# Evaluation with test set

tree_pred = predict(tree_model, newdata = df_test_new,
                 type = "class")
table(tree_pred, defaulting=df_test_new$Defaulter)

# Prunning

cv_tree = cv.tree(tree_model, FUN = prune.misclass)
names(cv_tree)
cv_tree

par(mfrow = c(1,2))
plot(cv_tree$size, cv_tree$dev, type="b")
plot(cv_tree$k, cv_tree$dev, type="b")


pruned_tree = prune.misclass(tree_model, best =3)
plot(pruned_tree)
text(pruned_tree, pretty=0)


# After this step I realize, the pruning process has resulted
# in a tree with only a single node and that is the root. 
# The cleaning step after the backward selection is not proper for the decision tree models.
# Because The decision tree algorithm and the backward selection for regression or 
# logistic regression models use different methodologies for selecting variables,
# which can lead to different variables being chosen as significant in each model.

# Now I will build decision tree model again without removing the columns from the initial dataset.
# The initial dataset is df_train.

df_train$GENDER = factor(df_train$GENDER)
df_train$MARRIAGE = factor(df_train$MARRIAGE)
df_train$REPAY_SEPT = factor(df_train$REPAY_SEPT)
df_train$REPAY_MAY = factor(df_train$REPAY_MAY)
df_train$REPAY_APR = factor(df_train$REPAY_APR)
df_train$REPAY_AUG = factor(df_train$REPAY_AUG)
df_train$REPAY_JUNE = factor(df_train$REPAY_JUNE)
df_train$REPAY_JULY = factor(df_train$REPAY_JULY)
df_train$EDUCATION = factor(df_train$EDUCATION)
df_train$Defaulter = factor(df_train$Defaulter)

df_test$GENDER = factor(df_test$GENDER)
df_test$MARRIAGE = factor(df_test$MARRIAGE)
df_test$REPAY_SEPT = factor(df_test$REPAY_SEPT)
df_test$REPAY_MAY = factor(df_test$REPAY_MAY)
df_test$REPAY_APR = factor(df_test$REPAY_APR)
df_test$REPAY_AUG = factor(df_test$REPAY_AUG)
df_test$REPAY_JUNE = factor(df_test$REPAY_JUNE)
df_test$REPAY_JULY = factor(df_test$REPAY_JULY)
df_test$EDUCATION = factor(df_test$EDUCATION)
df_test$Defaulter = factor(df_test$Defaulter)


tree_model2 = tree(Defaulter ~ ., data = df_train)
plot(tree_model2)

summary(tree_model2)
text(tree_model2, pretty =0)
tree_model2

# Evaluation with test set

tree_pred2 = predict(tree_model2, newdata = df_test,
                    type = "class")
confusion_matrix_1=table(tree_pred2, defaulting=df_test$Defaulter)

precision_1 = confusion_matrix_1[2, 2] / sum(confusion_matrix_1[2, ])
recall_1 = confusion_matrix_1[2, 2] / sum(confusion_matrix_1[, 2])
accuracy_1 = sum(diag(confusion_matrix_1)) / sum(confusion_matrix_1)
print(accuracy_1)
print(recall_1)
print(precision_1)
# In terms of the confusion matrix 81.73% of the test observations are correct

# Prunning

cv_tree2 = cv.tree(tree_model2, FUN = prune.misclass)
names(cv_tree2)
cv_tree2

par(mfrow = c(1,2))
plot(cv_tree2$size, cv_tree2$dev, type="b")
plot(cv_tree2$k, cv_tree2$dev, type="b")

pruned_tree2 = prune.misclass(tree_model2, best = 4)
plot(pruned_tree2)
text(pruned_tree2, pretty = 0)


pruned_pred = predict(pruned_tree2, df_test, type = "class")
confusion_matrix_dt=table( pruned_pred, defaulting = df_test$Defaulter)
confusion_matrix_dt
precision_dt = confusion_matrix_dt[2, 2] / sum(confusion_matrix_dt[2, ])
recall_dt = confusion_matrix_dt[2, 2] / sum(confusion_matrix_dt[, 2])
accuracy_dt = sum(diag(confusion_matrix_dt)) / sum(confusion_matrix_dt)
print(accuracy_dt)
print(recall_dt)
print(precision_dt)
# After pruning, the model reached 81.73% accuracy. the value of 'best' has assigned as its maximum value.
# Due to that reason, the model' has reached the highest level of accuracy after pruning as same as before pruning.


install.packages("randomForest")
library(randomForest)

# The factor variables in df_test set have categories that were not seen during the training of the model with df_train.
# This mismatch were not appeared in the decission tree model. We encounter it in bagging model.

# REPAY_AUG has 8 as a factor in df_train but df_test hasn't.
df_train$REPAY_AUG[df_train$REPAY_AUG == 8] = 7
df_train$REPAY_AUG = droplevels(df_train$REPAY_AUG)
table(df_train$REPAY_AUG)

# REPAY_MAY has 8 as a factor in df_test but df_train hasn't.
df_test$REPAY_MAY[df_test$REPAY_MAY == 8] = 7
df_test$REPAY_MAY = droplevels(df_test$REPAY_MAY)
table(df_test$REPAY_MAY)

# After this process I have to change 6 to 7 in df_train:
df_train$REPAY_MAY[df_train$REPAY_MAY == 6] = 7
df_train$REPAY_MAY = droplevels(df_train$REPAY_MAY)
table(df_train$REPAY_MAY)

# REPAY_APR has 8 as a factor in df_test but df_train hasn't.
df_test$REPAY_APR[df_test$REPAY_APR == 8] = 7
df_test$REPAY_APR = droplevels(df_test$REPAY_APR)
table(df_test$REPAY_APR)


bagging = randomForest(Defaulter ~ ., data = df_train, mtry =8, importance = TRUE)
bagging


pred_bag = predict(bagging, newdata = df_test)
plot(pred_bag, df_test$Defaulter)
abline(0,1)

# I will use confusion_matrix instead of MSE because target variable is binary.

confusion_matrix = table(pred_bag, Defaulting = df_test$Defaulter)
confusion_matrix


precision_bag = confusion_matrix[2, 2] / sum(confusion_matrix[2, ])
recall_bag = confusion_matrix[2, 2] / sum(confusion_matrix[, 2])
accuracy_bag = sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(precision_bag)
# Precision indicates that about 64.49% of the instances predicted as 'Defaulter' are correctly identified. 
# The model is moderately accurate in its positive predictions.
print(recall_bag)
# Recall shows the model correctly identifies only about 37.37% of actual 'Defaulter' cases. 
# Many true 'Defaulter' cases are being missed.
print(accuracy_bag)
# According to confusion matrix, bagging model's accuracy is 81.69%

#RANDOM FOREST

rf_def= randomForest(Defaulter ~ ., data=df_train, mtry = 5, importance = TRUE)
pred_rf= predict(rf_def, newdata = df_test)

varImpPlot(rf_def)
# MeanDecreaseAccuracy: It indicates how much accuracy the model loses on average when a variable is excluded. 
# Variables at the top of this chart have a greater impact on model accuracy.
# MeanDecreaseGini: It represents the decrease in node impurity that results from splits over a given variable, averaged over all trees.
# Variables at the top of this chart contribute more to the homogeneity of nodes and leaves.
# From the chart, REPAY_SEPT variable seems as a very strong predictor within the model.
# GENDER, EDUCATION, and MARRIAGE have less predictive power within the model.


confusion_matrix_rf = table(pred_rf, Defaulting = df_test$Defaulter)
confusion_matrix_rf

precision_rf = confusion_matrix_rf[2, 2] / sum(confusion_matrix_rf[2, ])
recall_rf = confusion_matrix_rf[2, 2] / sum(confusion_matrix_rf[, 2])
accuracy_rf = sum(diag(confusion_matrix_rf)) / sum(confusion_matrix_rf)
print(precision_rf)
# Precision (for 'Defaulter' predictions): 64.76% of the instances predicted as 'Defaulter' by the model are correct.
print(recall_rf)
# Recall (for actual 'Defaulter' instances): 36.77%. It means that out of all the true 'Defaulter' cases in the data,
# the model is able to detect a little over one-third of them.

# The model is fairly accurate when it predicts a case as a 'Defaulter', 
# but it tends to miss a substantial number of true 'Defaulter' cases.
# Improving its ability to detect more true 'Defaulter' cases
# without significantly raising the false positives would be beneficial.

# GRADIENT BOOSTING MODEL
# The old version of gbb is not working well so I need to upgrade it:

install.packages("devtools")
library(devtools)
install_github("gbm-developers/gbm3")
library(gbm3)


gbm_def = gbm(
  Defaulter ~ ., 
  data = df_train, 
  distribution = "bernoulli", 
  n.trees = 5000, 
  interaction.depth = 4,
)
summary(gbm_def)
# Variable importance from a gradient boosting model.
# The plot displays the relative influence or importance of each variable in the model.
# The graph illustrates that the repayment status in September (REPAY_SEPT) is the most significant predictor affecting the model's predictions,
# indicating a strong influence on the outcome.
# Other factors like REPAY_AUG and REPAY_MAY also show some importance but to a lesser extent. 
# Demographic factors such as GENDER, MARRIAGE, and EDUCATION have minimal influence.
pred_gbm = predict(gbm_def, newdata = df_test, n.trees=5000)

threshold = 0.5
binary_predictions = ifelse(pred_gbm > threshold, 1, 0)
actual_outcomes = df_test$Defaulter
conf_matrix = table(Predicted = binary_predictions, Defaulting = actual_outcomes)
print(conf_matrix)


precision_gbm = conf_matrix[2, 2] / sum(conf_matrix[2, ])
recall_gbm = conf_matrix[2, 2] / sum(conf_matrix[, 2])
accuracy_gbm = sum(diag(conf_matrix)) / sum(conf_matrix)


# Tuning leaarning rate:
gbm_def2 = gbm(
  Defaulter ~ ., 
  data = df_train, 
  distribution = "bernoulli", 
  n.trees = 5000, 
  interaction.depth = 4,
  shrinkage = 0.2,
)

pred_gbm2 = predict(gbm_def2, newdata = df_test, n.trees=5000)

threshold = 0.5
binary_predictions2 = ifelse(pred_gbm2 > threshold, 1, 0)
actual_outcomes = df_test$Defaulter
conf_matrix2 = table(Predicted = binary_predictions2, Defaulting = actual_outcomes)
print(conf_matrix2)


precision_gbm2 = conf_matrix2[2, 2] / sum(conf_matrix2[2, ])
recall_gbm2 = conf_matrix2[2, 2] / sum(conf_matrix2[, 2])
accuracy_gbm2 = sum(diag(conf_matrix2)) / sum(conf_matrix2)





conf_matrix2
cat("Accuracy of Gradient Boosting2 :", accuracy_gbm2, "\n")
cat("Precision of Gradient Boosting2:", precision_gbm2, "\n")
cat("Recall of Gradient Boosting2 :", recall_gbm2, "\n")

conf_matrix
cat("Accuracy of Gradient Goosting :", accuracy_gbm, "\n")
cat("Precision of Gradient Boosting:", precision_gbm, "\n")
cat("Recall of Gradient Boosting :", recall_gbm, "\n")

confusion_matrix_rf
cat("Accuracy of Random Forest :", accuracy_rf, "\n")
cat("Precision of Random Forest:", precision_rf, "\n")
cat("Recall of Random Forest :", recall_rf, "\n")

confusion_matrix
cat("Accuracy of Bagging :", accuracy_bag, "\n")
cat("Precision of Bagging:", precision_bag, "\n")
cat("Recall of Bagging :", recall_bag, "\n")

confusion_matrix_dt
cat("Accuracy of Decision Tree :", accuracy_dt, "\n")
cat("Precision of Decision Tree:", precision_dt, "\n")
cat("Recall of Decision Tree :", recall_dt, "\n")


accuracy_gbm2 = 0.798
precision_gbm2 = 0.5755509
recall_gbm2 = 0.3306209

accuracy_gbm = 0.8149
precision_gbm = 0.7123824
recall_gbm = 0.2739602

accuracy_rf = 0.8150667
precision_rf = 0.6421108
recall_rf = 0.3704039

accuracy_bag = 0.8144667
precision_bag = 0.6382429
recall_bag = 0.3722122

accuracy_dt = 0.8172667
precision_dt = 0.6827106
recall_dt = 0.3248945

f1_gbm2 = 2 * (precision_gbm2 * recall_gbm2) / (precision_gbm2 + recall_gbm2)
f1_gbm = 2 * (precision_gbm * recall_gbm) / (precision_gbm + recall_gbm)
f1_rf = 2 * (precision_rf * recall_rf) / (precision_rf + recall_rf)
f1_bag = 2 * (precision_bag * recall_bag) / (precision_bag + recall_bag)
f1_dt = 2 * (precision_dt * recall_dt) / (precision_dt + recall_dt)

f1_scores = c(
  GradientBoosting2 = f1_gbm2,
  GradientBoosting = f1_gbm,
  RandomForest = f1_rf,
  Bagging = f1_bag,
  DecisionTree = f1_dt
)

max_f1_model = names(which.max(f1_scores))
max_f1_value = max(f1_scores)


recall_values = c(
  GradientBoosting2 = 0.3306209,
  GradientBoosting = 0.2700422,
  RandomForest = 0.3682942,
  Bagging = 0.3716094,
  DecisionTree = 0.3248945
)


highest_recall_model = names(which.max(recall_values))
highest_recall_value = recall_values[highest_recall_model]


cat("Model with the highest recall is:", highest_recall_model, "with a recall of", highest_recall_value)
cat("Model with the highest F1 Score is:", max_f1_model, "with an F1 score of", round(max_f1_value, 4), "\n")

# To predict whether an individual is a defaulter or not, where '1' indicates a defaulter 
# and '0' indicates a non-defaulter, it's essential to consider not just the overall accuracy, 
# but also the precision and recall, especially if the cost of false negatives
# (failing to identify a defaulter) is high. Precision is a measure of how reliable a model is 
# when it predicts that someone will default. Recall, on the other hand, is a measure of 
# the model's ability to identify all actual defaulters. In cases like fraud detection or 
# predicting defaults, recall is often considered more important than precision 
# because the cost of missing an actual defaulter can be very high. 
# However, if a balance is sought to minimize the number of non-defaulters 
# who are incorrectly classified as defaulters, then the F1 score, a measure of 
# balance between precision and recall, should be aimed for.
# As a result, the Bagging model provided both the highest recall value and the highest F1 score.
# Due to this reason, this project's best model's is Bagging model.

