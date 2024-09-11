################################################################################
#                    EDA & models: Project overview                            #
################################################################################
#                          Valentina Zangirolami                               #
#                               Maggio 2024                                    #
################################################################################

#Final project guidelines

# 1. Create a R-project (optional)
# 2. Search and choose a dataset
# 3. Understand the problem and explore your data (EDA) 
# 4. Analyze potential statistical learning issues 
# 5. Choose proper models to deal with your data
# 6. Define metrics and compare results

# For the final project (exam): You should make a R-Markdown

################################################################################
#                         Create a R-project (optional)                        #
################################################################################

# See instructions in e-learning website: open the related pdf

################################################################################
#                               Search a dataset                               #
################################################################################

# You can find the dataset in several websites:
# 1. Kaggle
# 2. UCI Machine Learning Repository
# 3. Google Dataset search: https://datasetsearch.research.google.com/ 
#    (data from both papers and the above websites)

# Today, we are going to explore the problem of telco customer churn prediction
# You can download the dataset on e-learning 

# You can find complete dataset here https://t.ly/2rCPd

# see related dashboard on IBM Watson studio: https://t.ly/HMUbr

# (When we talk about telco, we mean telecommunication companies) 
# The data refers to home phone and internet services of customers in California.

# to run this script you MUST INSTALL ranger library (random forest - faster version)

### Load libraries

library(tidyverse)
library(tidymodels)
library(ggplot2)
library(geosphere)
library(rsample)
library(ROSE)
library(mgcv)
library(vip)

### Load dataset

churn_data <- read_csv("telco_churn_data.csv")
churn_data

dim(churn_data) # 7043 observations and 28 variables

# Each observation represents an unique customer. Data refers to Q3 fiscal quarter.

# Data description: https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113

################################################################################
#                           Data manipulation & EDA                            #
################################################################################

# To begin with, we can show the summary of our data
summary(churn_data)

# Missing values in TotalCharges

# Churn value and Churn Label represent the same information with a different codification.

churn_data <- churn_data %>% select(-c(ChurnValue, ChurnReason))

# Looking the data, we can observe something weird....

unique(churn_data$MultipleLines) 

# Yes, No, and No phone service. The class "No phone service" means 
# "No, since there is not a phone service".
# Obviously, multiple lines require that there exists a phone service
# Let's check

churn_data %>% filter(MultipleLines == "No phone service" & PhoneService == "Yes" ) # no obs

unique(churn_data$OnlineSecurity)

# Yes, No, and No internet service. As before, let's check with Internet Service variable

churn_data %>% filter(OnlineSecurity == "No internet service" & InternetService == "Yes" ) #no obs

# same issue for all features related to internet services
# For our purpose, we can reduce the three classes above to Yes/No

churn_data <- churn_data %>% mutate_if(is.character, funs(str_replace(., "^No (phone|internet) service$", "No")))

# At the beginning, we observed TotalCharges contains NAs. Let's check

churn_data %>% filter(is.na(TotalCharges))

# 11 NAs. Why are there missing values? It can happen for several
# reasons but let's check if, in this case, there is a potential justification.

# Total charges indicate the customer’s total charges, calculated to the end of the quarter.
# Maybe, it can be related to the activated services or the contract or the start/end date
# of the contract. Let's check

churn_data %>% filter(is.na(TotalCharges)) %>% select(Tenure, Contract, ChurnLabel, PhoneService, InternetService)

# Tenure = 0 for all observations... (Tenure indicates the total amount of 
# months that the customer has been with the company by the end of the quarter)

churn_data %>% filter(Tenure == 0) %>% select(Tenure, Contract, ChurnLabel, PhoneService, InternetService)

# same obs

unique(churn_data$Tenure)

# Tenure just contains integer numbers... thus, new clients (which join the company
# for less than a month) are basically excluded from the calculation of Total Charges.
# Effectively, there are not payments. We can say that the NAs mechanism related to Total Charges
# is MAR, and they should depend to Tenure. 

# The main techniques of imputation for MAR are the conditional mean or the conditional median.
# We can use a model which includes Tenure and other variables (such as MonthlyCharges, InternetService, ...)

# The problem arise on Tenure values which are 0 just in that observations...
# In my opinion, we can proceed in two differ ways: put 0 values (because effectively the company did not
# receive payments yet) or remove the observations (just to simplify the problem 
# and since we have few observations with that issue and we will not consider a real test set)

churn_data <- churn_data %>% filter(!is.na(TotalCharges))

# EDA

# Plot of the response variable to check the balance among classes
ggplot(churn_data, aes(x = ChurnLabel)) + 
  geom_bar(fill = "pink", 
           color="black") + labs(x = "Churn", title = "Class count for response variable") +
  theme_minimal()

# imbalanced classes: it can have a strong impact on predictions!!!

# We can analyze also the marginal effects

# Churn - Total Charges

ggplot(churn_data, aes(x = ChurnLabel, y = TotalCharges)) +
  geom_boxplot(fill = "lightblue", 
               color="black") + theme_minimal() +
  labs(title = "Total Charges distribution by Churn", x= "Churn", y = "Total Charges")

# maybe it's better to consider Monthly Charges..

ggplot(churn_data, aes(x = ChurnLabel, y = MonthlyCharges)) +
  geom_boxplot(fill = "lightblue", 
               color="black") + theme_minimal() +
  labs(title = "Monthly Charges distribution by Churn", x= "Churn", y = "Monthly Charges")

# Churn - Tenure

ggplot(churn_data, aes(x = ChurnLabel, y = Tenure)) +
  geom_boxplot(fill = "lightblue", 
               color="black") + theme_minimal() +
  labs(title = "Tenure distribution by Churn", x= "Churn", y = "Tenure in Months")

churn_data %>%
  ggplot(aes(Longitude, Latitude, color = ChurnLabel)) +
  geom_point(aes(size = Tenure), alpha = 1)+ scale_color_brewer(palette = "BuPu") + theme_light() +
  scale_size_continuous(range = c(0, 2.5))


################################################################################
#                             Feature Engineering                              #
################################################################################

# We can think to create new variables, for instance: we have 4 variables
# which describe additional features for internet service (Online Security, 
# Online Backup, Device Protection, Tech Support)
# Maybe, can be useful include a variable which just indicates the number of 
# active additional services...

churn_data <- churn_data %>% mutate(n_addIT_services = as.numeric(OnlineSecurity=="Yes") + as.numeric(OnlineBackup=="Yes") + as.numeric(DeviceProtection=="Yes") + as.numeric(TechSupport=="Yes"))

churn_data %>% select(n_addIT_services)

# Another new feature can be:  Haversine distance, Manhattan distance or Bearing degree for lat/long.
# We know data are from California, we can compute the distance from the capital (Sacramento)
# or the distance from the most populated city (Los Angeles)

# Sacramento center --> Latitude: 38.584505 Longitude: -121.491956

# haversine distance

churn_data <- churn_data %>% mutate(haversine_dist = mapply(function(lg, lt) distm(c(-121.491956, 38.584505), c(lg, lt), fun=distHaversine), Longitude, Latitude))

# manhattan distance

churn_data <- churn_data %>% mutate(manhattan_dist = mapply(function(lg) distm(c(-121.491956, 38.584505), c(lg, 38.584505), fun=distHaversine), Longitude) + mapply(function(lt) distm(c(-121.491956, 38.584505), c(-121.491956, lt), fun=distHaversine), Latitude))

# bearing degree

churn_data <- churn_data %>% mutate(bearing_degree = mapply(function(lg, lt) bearing(c(-121.491956, 38.584505), c(lg, lt)), Longitude, Latitude))

churn_data$ChurnLabel <- as.factor(churn_data$ChurnLabel)

################################################################################
#                                   Models                                     #
################################################################################

# In order to evaluate our models for prediction, we need to divide our dataset
# in train and validation sets, or use cross-validation.

# seed for reproducibility

set.seed(123)

# split training and validation sets

churn_split <- initial_split(churn_data)

train_data <- training(churn_split)

validation_data <- testing(churn_split)

# One way to overcome imbalance issues is to apply oversampling, undersampling or 
# a combination of both techniques

train_data_overs <- ovun.sample(ChurnLabel~., data=train_data, method = "both", seed=123)

table(train_data_overs$data$ChurnLabel) #balanced


################## RANDOM FOREST

# We can start with trying different hyperparameters 

# Random Forest with number of tree = 2000 and mtry=15 (see the help for further details)

# With tidymodels, we need to define the functional form of our model. Thus, in this case,
# we need to call rand_forest specifyng the main hyperparameters (you can look them up on the help page)
# and, if you want, you can also call the engine (i.e. what kind of random forest implementation you want)
# the most popular libraries on R are ranger and RandomForest.. In this case, we will use
# ranger 

tune_rf_spec <- rand_forest(
  mtry = 15,
  trees = 2000, mode = "classification") %>%
  set_engine("ranger", importance = "impurity")

# Another component of tidymodels is the recipe. In general, we can use it to preprocess
# the data and also create new predictors. In our case, we already add new features
# but the recipe can be helpful to perform most classical operation of data preprocessing
# or feature engineering
# For instance, in this case we will perform one-hot encoding for categorical predictors
# and we will remove columns from the data when the training set data have a single value
# (hence, it is better add it after step_dummy)

churn_rf_1 <- recipe(ChurnLabel~ ., train_data) %>% 
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors())

churn_rf_sam_1 <- recipe(ChurnLabel~ ., train_data_overs$data)  %>% 
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors())

# The third step requires to create a workflow: it is used to merge the model and the recipe

rf_work_1 <- 
  workflow() %>% 
  add_model(tune_rf_spec) %>% 
  add_recipe(churn_rf_1)

rf_work_sam_1 <- 
  workflow() %>% 
  add_model(tune_rf_spec) %>% 
  add_recipe(churn_rf_sam_1)

# Finally, we can fit our model

rf_fit_1 <- fit(rf_work_1, train_data)
rf_fit_over_1 <- fit(rf_work_sam_1, train_data_overs$data)

# variable importance 

rf_fit_1%>% 
  pull_workflow_fit() %>% 
  vip(num_features = 15)

rf_fit_over_1%>% 
  pull_workflow_fit() %>% 
  vip(num_features = 15)

# predictions on training set

fitt_rf1 <- rf_fit_1 %>% predict(train_data) %>% bind_cols(train_data)
fitt_over_rf1 <- rf_fit_over_1 %>% predict(train_data_overs$data) %>% bind_cols(train_data_overs$data)

# predictions on validation

pred_rf1 <- rf_fit_1 %>% predict(validation_data) %>% bind_cols(validation_data)
pred_over_rf1 <- rf_fit_over_1 %>% predict(validation_data) %>% bind_cols(validation_data)

#comparison

fitt_rf1 %>% metrics(truth = ChurnLabel, estimate = .pred_class) 
fitt_over_rf1 %>% metrics(truth = ChurnLabel, estimate = .pred_class)

pred_rf1 %>% metrics(truth = ChurnLabel, estimate = .pred_class) 
pred_over_rf1 %>% metrics(truth = ChurnLabel, estimate = .pred_class) 

# better accuracy in the first case, issue: overfitting

fitt_rf1 %>% f_meas(truth=ChurnLabel, estimate=.pred_class, event_level = "second")
fitt_over_rf1 %>% f_meas(truth=ChurnLabel, estimate=.pred_class, event_level = "second")

pred_rf1 %>% f_meas(truth=ChurnLabel, estimate=.pred_class, event_level = "second")
pred_over_rf1 %>% f_meas(truth=ChurnLabel, estimate=.pred_class, event_level = "second")

# Better f1score in the second case

# Our results are not really good... We can try to find other hyperparameters
# For instance, we can think to consider an higher value for mtry
# let's see if something will change

#Random Forest with number of tree = 2000 and mtry=20 (see the help for further details)

tune_rf_spec <- rand_forest(
  mtry = 20,
  trees = 2000 
) %>%
  set_mode("classification") %>%
  set_engine("ranger", importance = "impurity")

rf_work_1 <- 
  workflow() %>% 
  add_model(tune_rf_spec) %>% 
  add_recipe(churn_rf_1)

rf_work_sam_1 <- 
  workflow() %>% 
  add_model(tune_rf_spec) %>% 
  add_recipe(churn_rf_sam_1)

rf_fit_2 <- fit(rf_work_1, train_data)
rf_fit_over_2 <- fit(rf_work_sam_1, train_data_overs$data)

# variable importance 

rf_fit_2%>% 
  pull_workflow_fit() %>% 
  vip(num_features = 15)

rf_fit_over_2%>% 
  pull_workflow_fit() %>% 
  vip(num_features = 15)

# predictions on training set

fitt_rf2 <- rf_fit_2 %>% predict(train_data) %>% bind_cols(train_data)
fitt_over_rf2 <- rf_fit_over_2 %>% predict(train_data_overs$data) %>% bind_cols(train_data_overs$data)

# predictions on validation

pred_rf2 <- rf_fit_2 %>% predict(validation_data) %>% bind_cols(validation_data)
pred_over_rf2 <- rf_fit_over_2 %>% predict(validation_data) %>% bind_cols(validation_data)

#comparison

fitt_rf2 %>% metrics(truth = ChurnLabel, estimate = .pred_class) 
fitt_over_rf2 %>% metrics(truth = ChurnLabel, estimate = .pred_class)

pred_rf2 %>% metrics(truth = ChurnLabel, estimate = .pred_class) 
pred_over_rf2 %>% metrics(truth = ChurnLabel, estimate = .pred_class) 

# better accuracy in the first case

fitt_rf2 %>% f_meas(truth=ChurnLabel, estimate=.pred_class, event_level = "second")
fitt_over_rf2 %>% f_meas(truth=ChurnLabel, estimate=.pred_class, event_level = "second")

pred_rf2 %>% f_meas(truth=ChurnLabel, estimate=.pred_class, event_level = "second")
pred_over_rf2 %>% f_meas(truth=ChurnLabel, estimate=.pred_class, event_level = "second")

# Worst results... let's try to reduce the model complexity with the number of trees

# Random Forest with number of tree = 500 and mtry=15 (see the help for further details)

tune_rf_spec <- rand_forest(
  mtry = 15,
  trees = 500 
) %>%
  set_mode("classification") %>%
  set_engine("ranger", importance = "impurity")

rf_work_1 <- 
  workflow() %>% 
  add_model(tune_rf_spec) %>% 
  add_recipe(churn_rf_1)

rf_work_sam_1 <- 
  workflow() %>% 
  add_model(tune_rf_spec) %>% 
  add_recipe(churn_rf_sam_1)

rf_fit_3 <- fit(rf_work_1, train_data)
rf_fit_over_3 <- fit(rf_work_sam_1, train_data_overs$data)

# variable importance 

rf_fit_3%>% 
  pull_workflow_fit() %>% 
  vip(num_features = 15)

rf_fit_over_3%>% 
  pull_workflow_fit() %>% 
  vip(num_features = 15)

# predictions on training set

fitt_rf3 <- rf_fit_3 %>% predict(train_data) %>% bind_cols(train_data)
fitt_over_rf3 <- rf_fit_over_3 %>% predict(train_data_overs$data) %>% bind_cols(train_data_overs$data)

# predictions on validation

pred_rf3 <- rf_fit_3 %>% predict(validation_data) %>% bind_cols(validation_data)
pred_over_rf3 <- rf_fit_over_3 %>% predict(validation_data) %>% bind_cols(validation_data)

# comparison

fitt_rf3 %>% metrics(truth = ChurnLabel, estimate = .pred_class) 
fitt_over_rf3 %>% metrics(truth = ChurnLabel, estimate = .pred_class) 

pred_rf3 %>% metrics(truth = ChurnLabel, estimate = .pred_class) 
pred_over_rf3 %>% metrics(truth = ChurnLabel, estimate = .pred_class) 

# better accuracy in the first case

fitt_rf3 %>% f_meas(truth=ChurnLabel, estimate=.pred_class, event_level = "second")
fitt_over_rf3 %>% f_meas(truth=ChurnLabel, estimate=.pred_class, event_level = "second")

pred_rf3 %>% f_meas(truth=ChurnLabel, estimate=.pred_class, event_level = "second")
pred_over_rf3 %>% f_meas(truth=ChurnLabel, estimate=.pred_class, event_level = "second")

# No relevant changes compared with the first model

# Now let's try to tune hyperparameters using random search

# Suggestion: it is better to run the code below at home. If the code is too slow, 
# reduce the parameter "size" in rf_reg_tune! 

tune_rf_spec <- rand_forest(
  mtry = tune(),
  trees = tune(), min_n = tune(), mode = "classification") %>%
  set_engine("ranger", num.threads = 7)

churn_folds <- vfold_cv(churn_data)

rf_rec <-
  recipe(ChurnLabel~ ., train_data_overs$data) %>% 
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) 

rf_wflow <- 
  workflow() %>% 
  add_model(tune_rf_spec) %>% 
  add_recipe(rf_rec)


metr_res <- metric_set(kap, accuracy)

rf_reg_tune <-
  rf_wflow %>%
  tune_grid(
    churn_folds,
    grid = grid_random(
      trees(range = c(200, 1000)),
      mtry(range = c(10, 20)),
      min_n(range = c(6, 8)),
      size= 100
    ),
    metrics = metr_res
  )
rf_reg_tune

rf_reg_tune %>% 
  collect_metrics()

best_f1 <- rf_reg_tune %>% select_best(metric="kap")

final_rf <- finalize_model(
  tune_rf_spec,
  best_f1
)

final_rf

# best result
# mtry=11, trees=491, min_n=6

############### Extreme Gradient Boosting

# Suggestion: it is better to run the code below at home. If the code is too slow, 
# reduce the parameter "size"

tune_btree_spec <- boost_tree(learn_rate = tune(), tree_depth = tune(), trees = tune()) %>% 
  set_engine("xgboost", nthreads = parallel::detectCores()) %>% set_mode("classification")

btree_wflow <- 
  workflow() %>% 
  add_model(tune_btree_spec) %>% 
  add_recipe(rf_rec)

btree_reg_tune <-
  btree_wflow %>%
  tune_grid(
    churn_folds,
    grid = grid_random(
      learn_rate(),
      tree_depth(range = c(2, 8)),
      trees(range = c(5, 20)),
      size= 100
    ),
    metrics = metr_res
  )
btree_reg_tune

btree_reg_tune %>% 
  collect_metrics()

best_btree_f1 <- btree_reg_tune %>% select_best(metric="kap")

final_btree <- finalize_model(
  tune_btree_spec,
  best_btree_f1
)

final_btree

# best result
#trees = 11
#tree_depth = 6
#learn_rate = 0.0467735083

################################################################################
#                              Compare results                                 #
################################################################################

# Now we want to compare our best models on validation set. We consider 3 models:
# the first rf, the rf and the boosted tree with the best hyperparameters (rand. search)

tune_rf_spec <- rand_forest(
  mtry = 15,
  trees = 2000, mode = "classification") %>%
  set_engine("ranger", importance = "impurity")


churn_rf_sam_1 <- recipe(ChurnLabel~ ., train_data_overs$data) %>% 
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors())

rf_work_sam_1 <- 
  workflow() %>% 
  add_model(tune_rf_spec) %>% 
  add_recipe(churn_rf_sam_1)

rf_fit_over_1 <- fit(rf_work_sam_1, train_data_overs$data)


tune_rf_spec_2 <- rand_forest(
  mtry = 11,
  trees = 491, min_n = 6, mode = "classification") %>%
  set_engine("ranger", importance = "impurity")

rf_work_sam_2 <- 
  workflow() %>% 
  add_model(tune_rf_spec_2) %>% 
  add_recipe(churn_rf_sam_1)

rf_fit_over_2 <- fit(rf_work_sam_2, train_data_overs$data)


tune_btree_spec <- boost_tree(learn_rate = 0.0467735083, tree_depth = 6, trees = 11) %>% 
  set_engine("xgboost") %>% set_mode("classification")

btree_work_sam <- 
  workflow() %>% 
  add_model(tune_btree_spec) %>% 
  add_recipe(churn_rf_sam_1)

rf_btree_over <- fit(btree_work_sam, train_data_overs$data)

# predictions on training set

fitt_over_rf1 <- rf_fit_over_1 %>% predict(train_data_overs$data) %>% bind_cols(train_data_overs$data)
fitt_over_rf2 <- rf_fit_over_2 %>% predict(train_data_overs$data) %>% bind_cols(train_data_overs$data)
fitt_over_btree <- rf_btree_over %>% predict(train_data_overs$data) %>% bind_cols(train_data_overs$data)

# predictions on validation

pred_over_rf1 <- rf_fit_over_1 %>% predict(validation_data) %>% bind_cols(validation_data)
pred_over_rf2 <- rf_fit_over_2 %>% predict(validation_data) %>% bind_cols(validation_data)
pred_over_btree <- rf_btree_over %>% predict(validation_data) %>% bind_cols(validation_data)


#comparison

fitt_over_rf1 %>% metrics(truth = ChurnLabel, estimate = .pred_class)
fitt_over_rf2 %>% metrics(truth = ChurnLabel, estimate = .pred_class)
fitt_over_btree %>% metrics(truth = ChurnLabel, estimate = .pred_class)

pred_over_rf1 %>% metrics(truth = ChurnLabel, estimate = .pred_class) 
pred_over_rf2 %>% metrics(truth = ChurnLabel, estimate = .pred_class) 
pred_over_btree %>% metrics(truth = ChurnLabel, estimate = .pred_class)


fitt_over_rf1 %>% f_meas(truth=ChurnLabel, estimate=.pred_class, event_level = "second")
fitt_over_rf2 %>% f_meas(truth=ChurnLabel, estimate=.pred_class, event_level = "second")
fitt_over_btree %>% f_meas(truth=ChurnLabel, estimate=.pred_class, event_level = "second")

pred_over_rf1 %>% f_meas(truth=ChurnLabel, estimate=.pred_class, event_level = "second")
pred_over_rf2 %>% f_meas(truth=ChurnLabel, estimate=.pred_class, event_level = "second")
pred_over_btree %>% f_meas(truth=ChurnLabel, estimate=.pred_class, event_level = "second")

# On the validation set we have similar results, rf suffers a lot from overfitting
