# go to directory containing data
setwd("C:\\Users\\Louis\\Documents\\CodingProjects\\titanic-kaggle")

# get as much as possible out of read.csv (munging)
readData <- function(path.name, file.name, column.types, missing.types) {
  read.csv( url( paste(path.name, file.name, sep="") ), 
            colClasses=column.types,
            na.strings=missing.types )
}

# access data from my github repo
Titanic.path <- "https://raw.github.com/louisguitton/titanic-kaggle/master/"
train.data.file <- "train.csv"
test.data.file <- "test.csv"

# Data Munging : specifying types
missing.types <- c("NA", "")
train.column.types <- c('integer',   # PassengerId
                        'factor',    # Survived 
                        'factor',    # Pclass
                        'character', # Name
                        'factor',    # Sex
                        'numeric',   # Age
                        'integer',   # SibSp
                        'integer',   # Parch
                        'character', # Ticket
                        'numeric',   # Fare
                        'character', # Cabin
                        'factor'     # Embarked
)
test.column.types <- train.column.types[-2]     # # no Survived column in test.csv

train.raw <- readData(Titanic.path, train.data.file, train.column.types, missing.types)
test.raw <- readData(Titanic.path, test.data.file, test.column.types, missing.types)
df.train <- train.raw
df.test <- test.raw

# Data Visualization: Description of each feature
barplot(table(df.train$Survived),
        names.arg = c("Perished", "Survived"),
        main="Survived (passenger fate)", col="black")
barplot(table(df.train$Pclass), 
        names.arg = c("first", "second", "third"),
        main="Pclass (passenger traveling class)", col="firebrick")
barplot(table(df.train$Sex), main="Sex (gender)", col="darkviolet")
hist(df.train$Age, main="Age", xlab = NULL, col="brown")
barplot(table(df.train$SibSp), main="SibSp (siblings + spouse aboard)", 
        col="darkblue")
barplot(table(df.train$Parch), main="Parch (parents + kids aboard)", 
        col="gray50")
hist(df.train$Fare, main="Fare (fee paid for ticket[s])", xlab = NULL, 
     col="darkgreen")
barplot(table(df.train$Embarked), 
        names.arg = c("Cherbourg", "Queenstown", "Southampton"),
        main="Embarked (port of embarkation)", col="sienna")

# Data Visualization: Mosaic plot
# Rich first
mosaicplot(df.train$Pclass ~ df.train$Survived, 
           main="Passenger Fate by Traveling Class", shade=FALSE, 
           color=TRUE, xlab="Pclass", ylab="Survived")
# Women first
mosaicplot(df.train$Sex ~ df.train$Survived, 
           main="Passenger Fate by Gender", shade=FALSE, color=TRUE, 
           xlab="Sex", ylab="Survived")
# Survival of the fittest?
boxplot(df.train$Age ~ df.train$Survived, 
        main="Passenger Fate by Age",
        xlab="Survived", ylab="Age")
# Harbour ?
mosaicplot(df.train$Embarked ~ df.train$Survived, 
           main="Passenger Fate by Port of Embarkation",
           shade=FALSE, color=TRUE, xlab="Embarked", ylab="Survived")

## map missing data by provided feature
require(Amelia)
missmap(df.train, main="Titanic Training Data - Missings Map", 
        col=c("yellow", "black"), legend=FALSE)
# 20% 0f age is missing => wew will predict it with a decision tree

## function for extracting honorific (i.e. title) from the Name feature
getTitle <- function(data) {
  right <- strsplit(data$Name, ", ")
  for(i in 1:dim(data)[1]){
    data$Title[i] <- strsplit(right[[i]][2],". ")[[1]][1]
  }
  return (data$Title)
}
df.train$Title <- getTitle(df.train)
df.train$Title <- factor(df.train$Title, c("Capt","Col","Major","Sir","Lady","Rev",
                                           "Dr","Don","Jonkheer","the Countess","Mrs",
                                           "Ms","Mr","Mme","Mlle","Miss","Master","Noble"))
summary(df.train$Title[is.na(df.train$Age)])
## list of titles with missing Age value(s) requiring imputation
titles.na.train <- c("Dr", "Master", "Mrs", "Miss", "Mr")

require(Hmisc) #for the impute function
imputeMedian <- function(impute.var, filter.var, var.levels) {
  for (v in var.levels) {
    impute.var[ which( filter.var == v)] <- impute(impute.var[ 
      which( filter.var == v)])
  }
  return (impute.var)
}

df.train$Age <- imputeMedian(df.train$Age, df.train$Title,titles.na.train)

df.train$Embarked[which(is.na(df.train$Embarked))] <- 'S'

## impute missings on Fare feature with median fare by Pclass
df.train$Fare[ which( df.train$Fare == 0 )] <- NA
df.train$Fare <- imputeMedian(df.train$Fare, df.train$Pclass, as.numeric(levels(df.train$Pclass)))

boxplot(df.train$Age ~ df.train$Title, 
        main="Passenger Age by Title", xlab="Title", ylab="Age")

## function for assigning a new title value to old title(s) 
changeTitles <- function(data, old.titles, new.title) {
  for (honorific in old.titles) {
    data$Title[data$Title == honorific] <- new.title
  }
  return (data$Title)
}
## Title consolidation
df.train$Title <- changeTitles(df.train, c("Capt", "Col", "Don", "Dr", "Jonkheer", "Lady", "Major", "Rev", "Sir"),"Noble")
df.train$Title <- changeTitles(df.train, c("the Countess", "Ms"), "Mrs")
df.train$Title[is.na(df.train$Title)] <- "Mrs"
df.train$Title <- changeTitles(df.train, c("Mlle", "Mme"), "Miss")
df.train$Title <- factor(df.train$Title)
summary(df.train$Title)


#Other Feature Engineering
require(plyr)     # for the revalue function 
require(stringr)  # for the str_sub function

## test a character as an EVEN single digit
isEven <- function(x) x %in% c("0","2","4","6","8") 
## test a character as an ODD single digit
isOdd <- function(x) x %in% c("1","3","5","7","9") 

## function to add features to training or test data frames
featureEngrg <- function(data) {
  ## Using Fate ILO Survived because term is shorter and just sounds good
  data$Fate <- data$Survived
  ## Revaluing Fate factor to ease assessment of confusion matrices later
  data$Fate <- revalue(data$Fate, c("1" = "Survived", "0" = "Perished"))
  ## Boat.dibs attempts to capture the "women and children first"
  ## policy in one feature.  Assuming all females plus males under 15
  ## got "dibs' on access to a lifeboat
  data$Boat.dibs <- "No"
  data$Boat.dibs[which(data$Sex == "female" | data$Age < 15)] <- "Yes"
  data$Boat.dibs <- as.factor(data$Boat.dibs)
  ## Family consolidates siblings and spouses (SibSp) plus
  ## parents and children (Parch) into one feature
  data$Family <- data$SibSp + data$Parch
  ## Fare.pp attempts to adjust group purchases by size of family
  data$Fare.pp <- data$Fare/(data$Family + 1)
  ## Giving the traveling class feature a new look
  data$Class <- data$Pclass
  data$Class <- revalue(data$Class, 
                        c("1"="First", "2"="Second", "3"="Third"))
  ## First character in Cabin number represents the Deck 
  data$Deck <- substring(data$Cabin, 1, 1)
  data$Deck[ which( is.na(data$Deck ))] <- "UNK"
  data$Deck <- as.factor(data$Deck)
  ## Odd-numbered cabins were reportedly on the port side of the ship
  ## Even-numbered cabins assigned Side="starboard"
  data$cabin.last.digit <- str_sub(data$Cabin, -1)
  data$Side <- "UNK"
  data$Side[which(isEven(data$cabin.last.digit))] <- "port"
  data$Side[which(isOdd(data$cabin.last.digit))] <- "starboard"
  data$Side <- as.factor(data$Side)
  data$cabin.last.digit <- NULL
  return (data)
}

## add remaining features to training data frame
df.train <- featureEngrg(df.train)

train.keeps <- c("Fate", "Sex", "Boat.dibs", "Age", "Title", 
                 "Class", "Deck", "Side", "Fare", "Fare.pp", 
                 "Embarked", "Family")
df.train.munged <- df.train[train.keeps]

# resume at https://github.com/wehrley/wehrley.github.io/blob/master/SOUPTONUTS.md#fitting-a-model
## split training data into train batch and test batch
require(caret)
set.seed(23)
training.rows <- createDataPartition(df.train.munged$Fate, 
                                     p = 0.8, list = FALSE)
train.batch <- df.train.munged[training.rows, ]
test.batch <- df.train.munged[-training.rows, ]

#Logistic regression
Titanic.logit.1 <- glm(Fate ~ Sex + Class + Age + Family + Embarked + Fare, 
                       data = train.batch, family=binomial("logit"))
anova(Titanic.logit.1, test="Chisq") #conclusion = fare is not so relevant

Titanic.logit.2 <- glm(Fate ~ Sex + Class + Age + Family + Embarked + Fare.pp,                        
                       data = train.batch, family=binomial("logit"))
anova(Titanic.logit.2, test="Chisq") #conclusion = we drop Fare

## Define control function to handle optional arguments for train function
## Models to be assessed based on largest absolute area under ROC curve
cv.ctrl <- trainControl(method = "repeatedcv", repeats = 3,
                        summaryFunction = twoClassSummary,
                        classProbs = TRUE)
set.seed(45)
glm.tune.1 <- train(Fate ~ Sex + Class + Age + Family + Embarked,
                    data = train.batch,
                    method = "glm",
                    metric = "ROC",
                    trControl = cv.ctrl)
summary(glm.tune.1) #conclusion= Embarked D and S are redondant

#let summarize the two in a variable true=S false=otherwise
set.seed(35)
glm.tune.2 <- train(Fate ~ Sex + Class + Age + Family + I(Embarked=="S"),
                      data = train.batch, method = "glm",
                      metric = "ROC", trControl = cv.ctrl)
summary(glm.tune.2)

#let put in place Title
set.seed(35)
glm.tune.3 <- train(Fate ~ Sex + Class + Title + Age 
                      + Family + I(Embarked=="S"), 
                      data = train.batch, method = "glm",
                      metric = "ROC", trControl = cv.ctrl)
summary(glm.tune.3)

#let's drop Age and collapse the title classes
set.seed(35)
glm.tune.4 <- train(Fate ~ Class + I(Title=="Mr") + I(Title=="Noble") 
                      + Age + Family + I(Embarked=="S"), 
                      data = train.batch, method = "glm",
                      metric = "ROC", trControl = cv.ctrl)
summary(glm.tune.4)

# the grown men from third class are dead basically
set.seed(35)
glm.tune.5 <- train(Fate ~ Class + I(Title=="Mr") + I(Title=="Noble") 
                      + Age + Family + I(Embarked=="S") 
                      + I(Title=="Mr"&Class=="Third"), 
                      data = train.batch, 
                      method = "glm", metric = "ROC", 
                      trControl = cv.ctrl)
summary(glm.tune.5)

#logistic regressoin seems fine but
#https://github.com/wehrley/wehrley.github.io/blob/master/SOUPTONUTS.md#other-models

#ADABOOST
## note the dot preceding each variable
ada.grid <- expand.grid(.iter = c(50, 100),
                        .maxdepth = c(4, 8),
                        .nu = c(0.1, 1))
set.seed(35)
ada.tune <- train(Fate ~ Sex + Class + Age + Family + Embarked, 
                  data = train.batch,
                  method = "ada",
                  metric = "ROC",
                  tuneGrid = ada.grid,
                  trControl = cv.ctrl)
/
