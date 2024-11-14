

# f_ML with 4 and with 21 ----

# Recall that in the xgboost package manual they mention there are 'Linear Booster'
# parameters, called lambda, lambda_bias, and alpha. These are the hyperparameters for
# L2 regularization on weights, L2 regularization on bias, and L1 regularization on 
# weights, respectively (L1 regularization on bias is not important). Dr. Xu calls 
# them penalties. 
# There is not currently support for tuning these in tidymodels. Apparently Julia Silge 
# is waiting for more interest in adding that. Darn.
# The defaults for all of these is 0. These three hyperparameters are not listed in 
# the options for xgboost in tidymodels:
# show_model_info("boost_tree")


f_ML <- function(dat,modeling_method,id_m){
  
  # . ----
  # FOR EQUATIONS 1-3 ----
  if(id_m %in% c(1,2,3)){
    
    # M   <- cbind(My,  # will become dat[,1]
    #              Mx1, # will become dat[,2]
    #              Mx2, # will become dat[,3]
    #              
    #              # ADD Mx3 ----
    #              Mx3, # will become dat[,4]
    #              
    #              # ... ADD Mx4  ----
    #              Mx4, # will become dat[,5]
    #              
    #              MsIA,# will become dat[,6]
    #              MsIB,# will become dat[,7]
    #              Mw)  # will become dat[,8] 
    y      <- dat[,1]
    x1     <- dat[,2]
    x2     <- dat[,3]
    
    # ADD x3
    x3     <- dat[,4]
    
    # ... ADD Mx4  ----
    x4     <- dat[,5]
    
    # Sample Indicators for A and B
    sIA    <- dat[,6]
    sIB    <- dat[,7]
    
    # 
    sx1A   <- x1[sIA == 1]
    sx1B   <- x1[sIB == 1]
    
    # 
    sx2A   <- x2[sIA == 1]
    sx2B   <- x2[sIB == 1]
    
    # ADD sx3A and B 
    sx3A   <- x3[sIA == 1]
    sx3B   <- x3[sIB == 1]
    
    # ... ADD sx4A and B ----
    sx4A   <- x4[sIA == 1]
    sx4B   <- x4[sIB == 1]
    
    # 
    syB    <- y[sIB == 1]
    
    # ... ADD sx4B to datB ----
    datB   <- cbind(syB,sx1B,sx2B,sx3B,sx4B)
    datB2  <- as.data.frame(datB)
    
    # ... ADD sx4A to datXA ----
    datXA  <- cbind(sx1A,sx2A,sx3A,sx4A)
    datXA2 <- as.data.frame(datXA)
    
    # The colnames for A are labelled with B's because we're gonna predict using A as new data 
    # (has to have same column names)
    colnames(datXA2) <- c('sx1B','sx2B','sx3B','sx4B')
    
    # ADD tibbles for datB2 & datXA2 ----
    datB2_tbl <- datB2 %>% as_tibble()
    datXA2_tbl <- datXA2 %>% as_tibble()
    
    # * 0) recipe (COMMON) ----
    common_recipe <- recipe(
      syB ~ ., data = datB2_tbl
    )
    
    # Fit model to sample B ----
    if (modeling_method        == "GAM") {
      fit <- gam(syB ~ s(sx1B) + s(sx2B) + s(sx3B) + s(sx4B), # ... GAM's x4 ----
                 data = datB2)
    }
    
    if (modeling_method == "LM") {
      fit <- linear_reg() %>%
        set_engine("lm") %>%
        fit(syB ~ sx1B + sx2B + sx3B + sx4B, # ... LM's x4 ----
            data = datB2)
    }
    
    if (modeling_method        == "STEP") {
    	full.model = lm(syB ~ ., data = datB2)
	fit <- stepAIC(full.model, direction = "both", trace = FALSE)
    }

    if (modeling_method == "RANDOM_FOREST") {
      
      # * 1) model spec r forest ----
      randforest_spec <- rand_forest(
        mtry = tune(),
        trees = tune(),
        min_n = tune()
      ) %>% 
        set_mode("regression") %>% 
        set_engine("randomForest")
      
      # * 2) workflow ----
      randforest_workflow <- workflow() %>% 
        add_recipe(common_recipe) %>% 
        add_model(randforest_spec)
      
      # * 3) resamples ----
      set.seed(123)
      dat_folds <- vfold_cv(datB2_tbl,v = 10)
      rf_cube <- grid_latin_hypercube(
        finalize(mtry(), datB2_tbl[,2:5]),
	#trees(c(10,1500), trans=NULL),
	trees(),
        #min_n(c(2,20), trans= NULL),
        min_n(),
        #loss_reduction(),
        #sample_size = sample_prop(), 
        size = 30
      )

      # * 4) tune_grid() ----
      set.seed(123)
      randforest_grid <- tune_grid(
        randforest_workflow,
        resamples = dat_folds,
        grid = rf_cube
      )
      #autoplot(randforest_grid, metric = "rmse")
 
      # * 5) finalize_workflow() ----
      final_randforest_workflow <- randforest_workflow %>% 
        finalize_workflow(select_best(randforest_grid,metric = "rmse"))
      
      # * 6) last_fit() ----
      fit <- pull_workflow_spec(final_randforest_workflow) %>% 
        fit(syB ~ ., data = datB2_tbl)
      
    }
    
    if (modeling_method == "XGBOOST") {
      
      # * 1) model spec xgboost ----
      boost_spec <- boost_tree(
        tree_depth = tune(),
        trees = tune(),              # add tune() here as well - 20, 50, 100, 200, 300, etc.
        learn_rate = tune(),
        mtry = tune(),
        min_n = tune(),
        #loss_reduction = tune(),
        sample_size = 0.8
      ) %>% 
        set_mode("regression") %>% 
        set_engine("xgboost")
      
      # * 2) workflow ----
      boost_workflow <- workflow() %>% 
        add_recipe(common_recipe) %>% 
        add_model(boost_spec)
      
      # * 3) resamples ----
      set.seed(123)
      dat_folds <- vfold_cv(datB2_tbl,v = 10)
      
      # * 4) tune_grid() ----
      boost_cube <- grid_latin_hypercube(
        tree_depth(),
	trees(c(100,1800), trans=NULL),
        learn_rate(c(0.005, 0.5), trans= NULL),
        finalize(mtry(), datB2_tbl),
        min_n(c(1, 40), trans= NULL),
        #loss_reduction(),
        #sample_size = sample_prop(), 
        
        size = 30
      )
      
      set.seed(123)
      boost_grid <- tune_grid(
        boost_workflow,
        resamples = dat_folds,
        grid = boost_cube,
        control = control_grid(save_pred = T)
      )

	#autoplot(boost_grid, metric = "rmse") 
      
      # * 5) finalize_workflow() ----
      final_boost_workflow <- boost_workflow %>% 
        finalize_workflow(select_best(boost_grid,metric = "rmse"))
      
      # * 6) last_fit() ----
      fit <- pull_workflow_spec(final_boost_workflow) %>% 
        fit(syB ~ ., data = datB2_tbl)
      
    }
    
    if (modeling_method == "TREE") {
      # * 1) model spec rpart ----
      rpart_spec <- decision_tree(
        tree_depth = tune(),
        min_n = tune(),
        cost_complexity = tune()
      ) %>% 
        set_mode("regression") %>% 
        set_engine("rpart")
      
      # * 2) workflow ----
      rpart_workflow <- workflow() %>% 
        add_recipe(common_recipe) %>% 
        add_model(rpart_spec)
      
      # * 3) resamples ----
      set.seed(123)
      dat_folds <- vfold_cv(datB2_tbl,v = 10)
      
      # * 4) tune_grid() ----
      set.seed(123)
      rpart_cube <- grid_latin_hypercube(
        cost_complexity(),
        min_n(c(5,35), trans=NULL),
        tree_depth(c(4, 16), trans=NULL), 
        size = 30
      )
      rpart_grid <- tune_grid(
        rpart_workflow,
        resamples = dat_folds,
        grid = rpart_cube
      )
      #autoplot(rpart_grid, metric = "rmse") 
      
      # * 5) finalize_workflow() ----
      final_rpart_workflow <- rpart_workflow %>% 
        finalize_workflow(select_best(rpart_grid,metric = "rmse"))
      
      # * 6) last_fit() ----
      fit <- pull_workflow_spec(final_rpart_workflow) %>% 
        fit(syB ~ ., data = datB2_tbl)
      
    }
    
    if (modeling_method == "SVM") {
      
      # ... ----
      # * 1) model spec SVM ----
      svm_spec <- svm_rbf(
        cost = tune(),
        rbf_sigma = tune(),
        margin = tune()
      ) %>% 
        set_mode("regression") %>% 
        set_engine("kernlab")
      
      # * 2) workflow ----
      svm_workflow <- workflow() %>% 
        add_recipe(common_recipe) %>% 
        add_model(svm_spec)
      
      # * 3) resamples ----
      set.seed(123)
      dat_folds <- vfold_cv(datB2_tbl) # default v=10
      
      # * 4) tune_grid() ----
      set.seed(123)
      svm_cube <- grid_latin_hypercube(
        cost(c(2^0,2^5), trans=NULL),
        rbf_sigma(c(10^-5,0.5), trans= NULL),
        svm_margin(), 
        size = 30
      )
      svm_grid <- tune_grid(
        svm_workflow,
        resamples = dat_folds,
        grid = svm_cube # SVM doesn't take long at grid = 20 (just 4 min)
      )
      #autoplot(svm_grid, metric = "rmse") 

      # * 5) finalize_workflow() ----
      final_svm_workflow <- svm_workflow %>% 
        finalize_workflow(select_best(svm_grid,metric = "rmse"))
      
      # * 6) last_fit() ----
      fit <- pull_workflow_spec(final_svm_workflow) %>% 
        fit(syB ~ ., data = datB2_tbl)
      
    } 
    
  } 
  
  # . ----
  # FOR EQUATION 4 ----
  if(id_m == 4){
    
    # M   <- cbind(My,   # will become dat[,1]
    #              Mx1,  # will become dat[,2]
    #              Mx2,  # will become dat[,3]
    #              
    #              # ADD Mx3 ----
    #              Mx3,  # will become dat[,4]
    #              
    #              # ... ADD 4.thru.21 ----
    #              Mx4,  # will become dat[,5]
    #              Mx5,  # will become dat[,6]
    #              Mx6,  # will become dat[,7]
    #              Mx7,  # will become dat[,8]
    #              Mx8,  # will become dat[,9]
    #              Mx9,  # will become dat[,10]
    #              Mx10, # will become dat[,11]
    #              Mx11, # will become dat[,12]
    #              Mx12, # will become dat[,13]
    #              Mx13, # will become dat[,14]
    #              Mx14, # will become dat[,15]
    #              Mx15, # will become dat[,16]
    #              Mx16, # will become dat[,17]
    #              Mx17, # will become dat[,18]
    #              Mx18, # will become dat[,19]
    #              Mx19, # will become dat[,20]
    #              Mx20, # will become dat[,21]
    #              
    #              MsIA, # will become dat[,22]
    #              MsIB, # will become dat[,23]
    #              Mw)   # will become dat[,24] 
    y      <- dat[,1]
    x1     <- dat[,2]
    x2     <- dat[,3]
    
    # ADD x3 
    x3     <- dat[,4]
    
    # ... ADD x(4.thru.20) ----
    x4     <- dat[,5]
    x5     <- dat[,6]
    x6     <- dat[,7]
    x7     <- dat[,8]
    x8     <- dat[,9]
    x9     <- dat[,10]
    x10     <- dat[,11]
    x11     <- dat[,12]
    x12     <- dat[,13]
    x13     <- dat[,14]
    x14     <- dat[,15]
    x15     <- dat[,16]
    x16     <- dat[,17]
    x17     <- dat[,18]
    x18     <- dat[,19]
    x19     <- dat[,20]
    x20     <- dat[,21]
    
    # Sample Indicators for A and B
    sIA    <- dat[,22]
    sIB    <- dat[,23]
    
    
    # 
    sx1A   <- x1[sIA == 1]
    sx1B   <- x1[sIB == 1]
    
    # 
    sx2A   <- x2[sIA == 1]
    sx2B   <- x2[sIB == 1]
    
    # ADD sx3A and B 
    sx3A   <- x3[sIA == 1]
    sx3B   <- x3[sIB == 1]
    
    # ... ADD x(4.thru.20) ----
    sx4A   <- x4[sIA == 1]
    sx4B   <- x4[sIB == 1]
    
    sx5A   <- x5[sIA == 1]
    sx5B   <- x5[sIB == 1]
    
    sx6A   <- x6[sIA == 1]
    sx6B   <- x6[sIB == 1]
    
    sx7A   <- x7[sIA == 1]
    sx7B   <- x7[sIB == 1]
    
    sx8A   <- x8[sIA == 1]
    sx8B   <- x8[sIB == 1]
    
    sx9A   <- x9[sIA == 1]
    sx9B   <- x9[sIB == 1]
    
    sx10A   <- x10[sIA == 1]
    sx10B   <- x10[sIB == 1]
    
    sx11A   <- x11[sIA == 1]
    sx11B   <- x11[sIB == 1]
    
    sx12A   <- x12[sIA == 1]
    sx12B   <- x12[sIB == 1]
    
    sx13A   <- x13[sIA == 1]
    sx13B   <- x13[sIB == 1]
    
    sx14A   <- x14[sIA == 1]
    sx14B   <- x14[sIB == 1]
    
    sx15A   <- x15[sIA == 1]
    sx15B   <- x15[sIB == 1]
    
    sx16A   <- x16[sIA == 1]
    sx16B   <- x16[sIB == 1]
    
    sx17A   <- x17[sIA == 1]
    sx17B   <- x17[sIB == 1]
    
    sx18A   <- x18[sIA == 1]
    sx18B   <- x18[sIB == 1]
    
    sx19A   <- x19[sIA == 1]
    sx19B   <- x19[sIB == 1]
    
    sx20A   <- x20[sIA == 1]
    sx20B   <- x20[sIB == 1]
    
    # 
    syB    <- y[sIB == 1]
    
    # ... ADD sx(4.thru.20)B to datB ----
    datB   <- cbind(
      syB,
      sx1B,sx2B,sx3B,
      sx4B,sx5B,sx6B,sx7B,sx8B,sx9B,sx10B,sx11B,sx12B,sx13B,sx14B,sx15B,sx16B,sx17B,sx18B,sx19B,sx20B
    )
    datB2  <- as.data.frame(datB)
    
    # ADD sx3A to datXA ----
    datXA  <- cbind(
      # No Y.
      sx1A,sx2A,sx3A,
      sx4A,sx5A,sx6A,sx7A,sx8A,sx9A,sx10A,sx11A,sx12A,sx13A,sx14A,sx15A,sx16A,sx17A,sx18A,sx19A,sx20A
    )
    datXA2 <- as.data.frame(datXA)
    
    # The colnames for A are labelled with B's because we're gonna predict using A as new data 
    # (has to have same column names)
    colnames(datXA2) <- c(
      'sx1B','sx2B','sx3B',
      'sx4B','sx5B','sx6B','sx7B','sx8B','sx9B','sx10B','sx11B','sx12B','sx13B','sx14B','sx15B','sx16B','sx17B','sx18B','sx19B','sx20B'
    )
    
    # ADD tibbles for datB2 & datXA2 ----
    datB2_tbl <- datB2 %>% as_tibble()
    datXA2_tbl <- datXA2 %>% as_tibble()
    
    # * 0) recipe (COMMON) ----
    common_recipe <- recipe(
      syB ~ ., data = datB2_tbl
    )
    
    # Fit model to sample B ----
    if (modeling_method        == "GAM") {
      fit <- gam(
        syB ~ s(sx1B) + s(sx2B) + s(sx3B) + 
          s(sx4B) + s(sx5B) + s(sx6B) + s(sx7B) + s(sx8B) + s(sx9B) + s(sx10B) + s(sx11B) + s(sx12B) + s(sx13B) + s(sx14B) + s(sx15B) + s(sx16B) + s(sx17B) + s(sx18B) + s(sx19B) + s(sx20B),
        data = datB2
      )
    }
    
    if (modeling_method == "LM") {
      fit <- linear_reg() %>%
        set_engine("lm") %>%
        fit(syB ~ .,
            data = datB2_tbl)
    }
	
    if (modeling_method        == "STEP") {
    	full.model = lm(syB ~ ., data = datB2)
	fit <- stepAIC(full.model, direction = "both", trace = FALSE)
    }
    
    if (modeling_method == "RANDOM_FOREST") {
      
      # * 1) model spec r forest ----
      randforest_spec <- rand_forest(
        mtry = tune(),
        trees = tune(),
        min_n = tune()
      ) %>% 
        set_mode("regression") %>% 
        set_engine("randomForest")
      
      # * 2) workflow ----
      randforest_workflow <- workflow() %>% 
        add_recipe(common_recipe) %>% 
        add_model(randforest_spec)
      
      # * 3) resamples ----
      set.seed(123)
      dat_folds <- vfold_cv(datB2_tbl,v = 10)
      
      # * 4) tune_grid() ----
      set.seed(123)
      randforest_grid <- tune_grid(
        randforest_workflow,
        resamples = dat_folds,
        grid = 15
      )
      
      # * 5) finalize_workflow() ----
      final_randforest_workflow <- randforest_workflow %>% 
        finalize_workflow(select_best(randforest_grid,metric = "rmse"))
      
      # * 6) last_fit() ----
      fit <- pull_workflow_spec(final_randforest_workflow) %>% 
        fit(syB ~ ., data = datB2_tbl)
      
    }
    
    if (modeling_method == "XGBOOST") {
      
      # * 1) model spec xgboost ----
      boost_spec <- boost_tree(
        tree_depth = tune(),
        trees = tune(),              # add tune() here as well - 20, 50, 100, 200, 300, etc.
        learn_rate = tune(),
        mtry = tune(),
        min_n = tune(),
        loss_reduction = tune(),
        sample_size = tune()
      ) %>% 
        set_mode("regression") %>% 
        set_engine("xgboost")
      
      # * 2) workflow ----
      boost_workflow <- workflow() %>% 
        add_recipe(common_recipe) %>% 
        add_model(boost_spec)
      
      # * 3) resamples ----
      set.seed(123)
      dat_folds <- vfold_cv(datB2_tbl,v = 10)
      
      # * 4) tune_grid() ----
      boost_cube <- grid_latin_hypercube(
        tree_depth(),
        learn_rate(),
        finalize(mtry(), datB2_tbl),
        min_n(),
        loss_reduction(),
        sample_size = sample_prop(), 
        
        size = 30
      )
      
      set.seed(123)
      boost_grid <- tune_grid(
        boost_workflow,
        resamples = dat_folds,
        grid = boost_cube,
        control = control_grid(save_pred = T)
      )
      
      # * 5) finalize_workflow() ----
      final_boost_workflow <- boost_workflow %>% 
        finalize_workflow(select_best(boost_grid,metric = "rmse"))
      
      # * 6) last_fit() ----
      fit <- pull_workflow_spec(final_boost_workflow) %>% 
        fit(syB ~ ., data = datB2_tbl)
      
    }
    
    if (modeling_method == "TREE") {
      # * 1) model spec rpart ----
      rpart_spec <- decision_tree(
        tree_depth = tune(),
        min_n = tune(),
        cost_complexity = tune()
      ) %>% 
        set_mode("regression") %>% 
        set_engine("rpart")
      
      # * 2) workflow ----
      rpart_workflow <- workflow() %>% 
        add_recipe(common_recipe) %>% 
        add_model(rpart_spec)
      
      # * 3) resamples ----
      set.seed(123)
      dat_folds <- vfold_cv(datB2_tbl,v = 10)
      
      # * 4) tune_grid() ----
      set.seed(123)
      rpart_grid <- tune_grid(
        rpart_workflow,
        resamples = dat_folds,
        grid = 15
      )
      
      # * 5) finalize_workflow() ----
      final_rpart_workflow <- rpart_workflow %>% 
        finalize_workflow(select_best(rpart_grid,metric = "rmse"))
      
      # * 6) last_fit() ----
      fit <- pull_workflow_spec(final_rpart_workflow) %>% 
        fit(syB ~ ., data = datB2_tbl)
      
    }
    
    if (modeling_method == "SVM") {
      
      # ... ----
      # * 1) model spec SVM ----
      svm_spec <- svm_rbf(
        cost = tune(),
        rbf_sigma = tune(),
        margin = tune()
      ) %>% 
        set_mode("regression") %>% 
        set_engine("kernlab")
      
      # * 2) workflow ----
      svm_workflow <- workflow() %>% 
        add_recipe(common_recipe) %>% 
        add_model(svm_spec)
      
      # * 3) resamples ----
      set.seed(123)
      dat_folds <- vfold_cv(datB2_tbl) # default v=10
      
      # * 4) tune_grid() ----
      set.seed(123)
      svm_grid <- tune_grid(
        svm_workflow,
        resamples = dat_folds,
        grid = 40 # SVM doesn't take long at grid = 20 (just 4 min)
      )
      
      # * 5) finalize_workflow() ----
      final_svm_workflow <- svm_workflow %>% 
        finalize_workflow(select_best(svm_grid,metric = "rmse"))
      
      # * 6) last_fit() ----
      fit <- pull_workflow_spec(final_svm_workflow) %>% 
        fit(syB ~ ., data = datB2_tbl)
      
    } 
    
  }
  
  
  # . ----
  # FOR EQUATIONS 1-4 ----
  # (PREDICT ON A) ----
  # GAM predict produces an array, the others a tibble.
  # So, regardless of what is produced, it's converted to a data.frame, then
  # to a tibble, with .pred as column name.
  iyA     <- predict(fit,datXA2)
  iyA     <- as.data.frame(iyA)
  iyA     <- as_tibble(iyA) %>% set_names(".pred")
  # Then, the median is taken of the column, not the iyA directly. That's where the
  # error was!
  
  # etheta1 and 2 are global mean and median (no domain)
  etheta1 <- mean(iyA$.pred)  #sum(iyA$.pred*weight)/sum(weight _LLCPWT)
  etheta2 <- median(iyA$.pred)
  etheta  <- c(etheta1,etheta2)
  
  return(etheta)
}


