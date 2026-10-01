library(keras3)
require(terra)


#############################
## deep model using keras
#############################


#####
# prepare input data
####

# load data (stored in the attribute table of a vector file)
lai_data <- vect("C:/02_Lehre/02_WiSe/OEKB301579_ Advanced_methods_in_remote_sensing_Machine_learning_and_cloud_computing/02_DL_in_r_code/Day_2_practical/LAI_data_Vienna.gpkg")

# create short predictor names
predictor_names <- paste0("S2_band", 1:10)

# extract predictors and response from the vector dataset
y <- as.numeric(lai_data$LAI)
x <- as.data.frame(lai_data[,2:11]) 

# asign short names to predictors
names(x) <- predictor_names
# make sure that response is stored in correct data type format
storage.mode(y) <- "double"

# now split the data into training and test data
# get 80% training data
train <- sample(seq(1,nrow(x),1), 0.8*nrow(x), replace = F)

# build subset
x_train <- as.matrix(x[train,])
y_train <- as.numeric(y[train])

# take unselected samples as test data
x_test <- as.matrix(x[-train,])
y_test <- as.matrix(y[-train])


#####
# design deep sequential model with two hidden layers in keras
####

# define the model architecture (number of hidden layers, number of
# inputs and outputs)
model <- keras_model_sequential() |>
  # create first (hidden) layer
  layer_dense(
    # define number of hidden units in the layer
    units = 20,
    # define activation function
    activation = "relu",
    # define how man inputs predictor variables go into the layer
    # here it would be the 10 bands of Sentinel-2
    input_shape = 10
  ) |>
  # create second hidden layer
  layer_dense(
    # define number of hidden units in the layer
    units = 20,
    # define activation function
    activation = "relu",
    # define how man inputs predictor variables go into the layer
    # here it would be the 10 bands of Sentinel-2
    input_shape = 10
  ) |>
  # create last layer which is the output layer
  layer_dense(
    # we estimate only one value, namely LAI, so we use units=1
    units = 1
  )

# define the model compilation
model |> compile(
  # we select "adam" as model optimizer (we will learn
  # more about this soon)
  optimizer = optimizer_adam(learning_rate = 0.001),
  # we define a loss function, in this case the mean standard error
  loss = "mse",
  # we define an error metric which will be reported during model training
  metrics = c("mae")
)

# we define an early stopping criterion 
early_stop <- callback_early_stopping(
  # the early stopping is based on the validation loss (not the training loss)
  monitor = "val_loss",
  # the patience parameters defines how man epochs without improvements are
  # needed that the model stops the training
  patience = 30,
  # this defines that the algorithm remembers the parameters of the best
  # model until the stopping criterion kicks in
  restore_best_weights = TRUE
)

# we now can fit the model by inserting the training and validation,
# defining the number of epochs (we will learn what this is later),
# batch size, validation split and we call the just defined early
# stopping criterion. Verbose=1 means that we want to be informed
# about the model training continuously
history <- model |> fit(
  x_train,
  y_train,
  epochs = 100,
  batch_size = 32,
  validation_split = 0.2,
  callbacks = list(early_stop),
  verbose = 1
)




#####
# Evaluation on completely unseen test data
####


# run the trained model on the test data
pred_test <- model |> predict(x_test)

# convert the predictions into a number vector format
pred_test <- as.numeric(pred_test)

# calculate common model performance metrics by comparing
# the predictions to the reference values
RMSE <- sqrt(mean((y_test - pred_test)^2))
MAE <- mean(abs(y_test - pred_test))
R2 <- 1 -
  sum((y_test - pred_test)^2) /
  sum((y_test - mean(y_test))^2)

# report the model performance metrics
c(
  RMSE = RMSE,
  MAE = MAE,
  R2 = R2
)

# plot predicted vs. observed LAI values
plot(
  y_test,
  pred_test,
  xlab = "Observed LAI",
  ylab = "Predicted LAI",
  main = "Neural Network: Test Set",
  ylim=c(-1,8), xlim=c(-1,8)
)

abline(a = 0,b = 1,lty = 2, col="darkred")


#####
# Apply model to raster dataset
####

# Load Sentinel-2 image
s2 <- rast("C:/02_Lehre/02_WiSe/OEKB301579_ Advanced_methods_in_remote_sensing_Machine_learning_and_cloud_computing/03_example_datasets/01_LAI_Austria/Sentinel_2_10bands_subset.tif")

# Select the 10 predictor bands
s2 <- s2[[1:10]]

# Give them the same names as the training data
names(s2) <- paste0("S2_band", 1:10)

# Function that applies the trained shallow network
# to the raster data and that can be called
# by terra's "predict()"-function

keras_predict <- function(model, data) {

  # convert raster data to a matrix  
  data <- as.matrix(data)
  
  # make sure that the matrix is saved in the correct data formar
  storage.mode(data) <- "double"
  
  # apply the actual model prediction
  prediction <- model |> predict(
    data,
    batch_size = 1024,
    verbose = 0
  )
  
  # return the predictions as vector
  as.numeric(prediction)
}


# Apply neural network to every pixel
lai_prediction <- terra::predict(
  s2,
  model,
  fun = keras_predict,
  filename = "LAI_prediction.tif",
  overwrite = TRUE
)


# Display result
plot(lai_prediction, range=c(0,8))



###########################################################################
###########################################################################
###########################################################################


library(torch)


#############################
## shallow model using torch
#############################


#install.packages("torch")
library(torch)
library(coro)

# -----------------------------
# 1. Prepare the data
# -----------------------------

#####
# prepare input data
####

# load data (stored in the attribute table of a vector file)
lai_data <- vect("C:/02_Lehre/02_WiSe/OEKB301579_ Advanced_methods_in_remote_sensing_Machine_learning_and_cloud_computing/02_DL_in_r_code/Day_2_practical/LAI_data_Vienna.gpkg")

# create short predictor names
predictor_names <- paste0("S2_band", 1:10)

# extract predictors and response from the vector dataset
y <- as.numeric(lai_data$LAI)
x <- as.data.frame(lai_data[,2:11]) 

# asign short names to predictors
names(x) <- predictor_names
# make sure that response is stored in correct data type format
storage.mode(y) <- "double"

# now split the data into training and test data
# get 80% training data
train <- sample(seq(1,nrow(x),1), 0.8*nrow(x), replace = F)

# build subset
x_train <- as.matrix(x[train,])
storage.mode(x_train) <- "double"

y_train <- as.numeric(y[train])

# take unselected samples as test data
x_test <- as.matrix(x[-train,])
storage.mode(x_test) <- "double"
y_test <- as.matrix(y[-train])

# -----------------------------
# 3. Standardize predictors
# -----------------------------

x_mean <- apply(x_train, 2, mean)
x_sd   <- apply(x_train, 2, sd)

x_train <- scale(x_train, center = x_mean, scale = x_sd)
x_test  <- scale(x_test, center = x_mean, scale = x_sd)

# Convert to torch tensors
x_train <- torch_tensor(x_train, dtype = torch_float())
x_test  <- torch_tensor(x_test, dtype = torch_float())

y_train <- torch_tensor(
  matrix(y_train, ncol = 1),
  dtype = torch_float()
)

y_test <- torch_tensor(
  matrix(y_test, ncol = 1),
  dtype = torch_float()
)

# -----------------------------
# 4. Define the neural network
# -----------------------------

net <- nn_module(
  "ShallowNN",
  
  initialize = function() {
    
    self$hidden1 <- nn_linear(
      in_features = 10,
      out_features = 20
    )
    
    self$hidden2 <- nn_linear(
      in_features = 20,
      out_features = 20
    )
    
    self$output <- nn_linear(
      in_features = 20,
      out_features = 1
    )
  },
  
  forward = function(x) {
    
    x <- self$hidden1(x)
    x <- torch_relu(x)
    
    x <- self$hidden2(x)
    x <- torch_relu(x)
    
    x <- self$output(x)
    
    x
  }
)

model <- net()

# -----------------------------
# 5. Loss and optimizer
# -----------------------------

loss_fn <- nn_mse_loss()

optimizer <- optim_adam(
  model$parameters,
  lr = 0.001
)


# -----------------------------
# 6. Mini-batch training
# -----------------------------

batch_size <- 32
epochs <- 100

# Create dataset
train_dataset <- tensor_dataset(
  x_train,
  y_train
)

# Create dataloader
train_loader <- dataloader(
  train_dataset,
  batch_size = batch_size,
  shuffle = TRUE
)

for (epoch in 1:epochs) {
  
  model$train()
  
  epoch_loss <- 0
  
  coro::loop(for (batch in train_loader) {
    
    optimizer$zero_grad()
    
    prediction <- model(batch[[1]])
    
    loss <- loss_fn(
      prediction,
      batch[[2]]
    )
    
    loss$backward()
    
    optimizer$step()
    
    epoch_loss <- epoch_loss + loss$item()
  })
  
  if (epoch %% 10 == 0) {
    
    cat(
      "Epoch:", epoch,
      "Loss:", epoch_loss / length(train_loader),
      "\n"
    )
  }
}

# -----------------------------
# 7. Test predictions
# -----------------------------

model$eval()

pred <- model(x_test)



# -----------------------------
# 8. Performance
# -----------------------------

rmse <- sqrt(
  mean((y_test - pred)^2)
)

mae <- mean(
  abs(y_test - pred)
)

r2 <- 1 -
  sum((y_test - pred)^2) /
  sum((y_test - mean(y_test))^2)

c(
  RMSE = rmse,
  MAE = mae,
  R2 = r2
)



# plot predicted vs. observed LAI values
plot(
  y_test,
  pred,
  xlab = "Observed LAI",
  ylab = "Predicted LAI",
  main = "Neural Network: Test Set",
  ylim=c(-1,8), xlim=c(-1,8)
)

abline(a = 0,b = 1,lty = 2, col="darkred")



#####
# Apply model to raster dataset
####

# Load Sentinel-2 image
s2 <- rast("C:/02_Lehre/02_WiSe/OEKB301579_ Advanced_methods_in_remote_sensing_Machine_learning_and_cloud_computing/03_example_datasets/01_LAI_Austria/Sentinel_2_10bands_subset.tif")

# Select the 10 predictor bands
s2 <- s2[[1:10]]

# Give them the same names as the training data
names(s2) <- paste0("S2_band", 1:10)

# Function that applies the trained shallow network
# to the raster data and that can be called
# by terra's "predict()"-function

torch_predict <- function(model, data) {
  
  # Convert raster values to matrix
  data <- as.matrix(data)
  
  # Make sure numeric
  storage.mode(data) <- "double"
  
  # IMPORTANT:
  # Standardize using the statistics from the TRAINING data
  data <- scale(
    data,
    center = x_mean,
    scale = x_sd
  )
  
  # Convert to torch tensor
  data <- torch_tensor(
    data,
    dtype = torch_float()
  )
  
  # Prediction
  prediction <- model(data)
  
  # Convert torch tensor back to R
  as.numeric(
    prediction$detach()$cpu()
  )
}


# Apply neural network to every pixel
lai_prediction <- terra::predict(
  s2,
  model,
  fun = torch_predict,
  filename = "LAI_prediction_torch.tif",
  overwrite = TRUE
)


# Display result
plot(lai_prediction, range=c(0,8))


