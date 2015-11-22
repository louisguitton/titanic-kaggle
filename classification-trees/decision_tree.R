setwd("C:\\Users\\Louis\\Documents\\CodingProjects\\titanic-kaggle")

# Import the training set: train
train_url <- "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train <- read.csv(train_url)
  
# Import the testing set: test
test_url <- "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test <- read.csv(test_url)

# Your train and test set are still loaded
str(train)
str(test)

# Load in the R package  
library(rpart)

# Build the decision tree
my_tree <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data=train, method="class" )

# Visualize the decision tree using plot() and text()
plot(my_tree)
text(my_tree)

# Load in the packages to create a fancified version of your tree
library(rattle)
library(rpart.plot)
library(RColorBrewer)

# Time to plot your fancy tree
fancyRpartPlot(my_tree)

# Make your prediction using the test set
my_prediction <- predict(my_tree,test,"class")

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
my_solution <- data.frame(PassengerId = test$PassengerId, Survived = my_prediction)

# Check that your data frame has 418 entries
nrow(my_solution)==418

# Write your solution to a csv file with the name my_solution.csv
write.csv(my_solution, row.names=FALSE, file="my_solution.csv" )