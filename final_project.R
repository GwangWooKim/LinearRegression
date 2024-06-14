library(DataExplorer)
library(car)
library(coefplot)
library(MASS)
library(leaps)
library(glmnet)
library(boot)
library(caret)

library(readr)
SeoulBikeData <- read_csv("SeoulBikeData.csv")

# EDA
df = SeoulBikeData
introduce(df)
plot_intro(df)
plot_missing(df)
plot_bar(df)
# plot_bar(df, with = 'Seasons')
plot_bar(df, by = "Seasons")
plot_bar(df, by = "Holiday")
plot_bar(df, by = "Functioning_Day")
plot_histogram(df)
plot_qq(df[,c('Rented_Bike_Count')])

# boxcox
y = df[['Rented_Bike_Count']]
lambda <- boxcox(y + 1 ~ 1)
lambda <- lambda$x[which.max(lambda$y)]
y_bc <- (y^lambda - 1) / lambda

plot_correlation(na.omit(df))
# plot_prcomp(df)
plot_boxplot(df, by = "Rented_Bike_Count")
plot_boxplot(df, by = "Seasons")

# first try
fit = lm(Rented_Bike_Count ~ Hour + Temperature + Humidity + Wind_speed 
         + Visibility + Dew_point_temperature + Solar_Radiation + Rainfall
         + Snowfall + Seasons + Holiday + Functioning_Day, data = df)
summary(fit)
coefplot(fit)
plot(fit)
vif(fit)

# refit
df[, "Rented_Bike_Count_bc"] = y_bc
fit_2 = lm(Rented_Bike_Count_bc ~ Hour + Temperature + Humidity + Wind_speed 
           + Visibility + Solar_Radiation + Rainfall
           + Snowfall + Seasons + Holiday + Functioning_Day, data = df)
summary(fit_2)
coefplot(fit_2)
plot(fit_2)
vif(fit_2)

# hist
hist(y, breaks = 30, col = rgb(0, 0, 1, alpha = 0.4))
hist(fit$fitted.values, breaks = 50, col = rgb(1, 0, 0, alpha = 0.4), add = TRUE)
legend("topright", legend = c("y(data)", "y(pred)"), fill = c("blue", "red"))

hist(y_bc, breaks = 30, col = rgb(0, 0, 1, alpha = 0.4))
hist(fit_2$fitted.values, breaks = 50, col = rgb(1, 0, 0, alpha = 0.4), add = TRUE)
legend("topright", legend = c("y(data)", "y(pred)"), fill = c("blue", "red"))

# test seasonality
fit_reduced = lm(Rented_Bike_Count_bc ~ Hour + Temperature + Humidity + Wind_speed 
                 + Visibility + Solar_Radiation + Rainfall
                 + Snowfall + Holiday + Functioning_Day, data = df)
y_pred <- predict(fit_2)
y_mean <- mean(y_bc)
SSR_full = sum((y_mean - y_pred)^2)
y_pred <- predict(fit_reduced)
SSR_reduced = sum((y_mean - y_pred)^2)
SSR = SSR_full - SSR_reduced
MSR = SSR / (14-12)
SSE = sum((resid(fit_2))^2)
MSE = SSE / (8760 - 14)
MSR/MSE # 436.2541 and F(2, 8746, 0.95) = 2.99 -> effective

# model selection
set.seed(123)
step(fit_2,direction="both") # it coincides the previous test results

fit_3 = lm(Rented_Bike_Count_bc ~ Hour + Temperature + Humidity
          + Solar_Radiation + Rainfall
          + Seasons + Holiday + Functioning_Day, data = df)

# y outlier
out = as.data.frame(cbind('e' =resid(fit_3),
      'h' =hatvalues(fit_3),
      't' =rstudent(fit_3)
))

# t(1-0.95/(2n), n-p-1) = 3.873
n = 8760
p = 11
t = 1-0.95/(2*n)

th = 3.873
outlier_y_indices = which(out$t >= th)

# for x use th = 2p/n
th = 2*p/n
outlier_x_indices = which(out$h >= th)

out_indices = union(outlier_x_indices, outlier_y_indices)
df_drop = df[-out_indices,]
rownames(df_drop) = 1: nrow(df_drop)
#write.csv(df_drop, "df_drop_20240613.csv", row.names = FALSE)

plot_bar(df_drop) # all holidays and functioning days are removed!

fit_4 = lm(Rented_Bike_Count_bc ~ Hour + Temperature + Humidity
           + Solar_Radiation + Rainfall
           + Seasons, data = df_drop)
plot(fit_4) # 4701 6746 7471

# influential
inf = as.data.frame(cbind(
  "DFFITS" =dffits(fit_4),
  "D" =cooks.distance(fit_4)
))

n = 7975
th = 2*sqrt(p/n)
influential_indices_DFFFIT = which(inf$DFFITS >= th)
th = 4/n
influential_indices_Cooks = which(inf$D >= th)
influential_indices = union(influential_indices_DFFFIT, influential_indices_Cooks)

4701 %in% influential_indices
6746 %in% influential_indices
7471 %in% influential_indices

# ridge regression
X=model.matrix(fit_4)[,-1]
Y=df_drop$Rented_Bike_Count_bc
c=seq(0,1,length=1000)
ridge.mod=glmnet(X,Y,alpha=0,lambda=c)
plot(ridge.mod)

set.seed(123)
cv.out=cv.glmnet(X,Y,alpha=0,lambda=c, nfolds=5)
plot(cv.out)
bestlam=cv.out$lambda.min
bestlam

out=glmnet(X,Y,alpha=0,lambda=bestlam)
predict(out,type="coefficients",s=bestlam)
coefficients(fit_4)

# final
df_drop <- read_csv("df_drop_20240613.csv")
lst_l = c()
lst_r = c()
set.seed(123)
n=nrow(df_drop)
k=5 # used for k-fold
numbers <- 1:n
shuffled_numbers <- sample(numbers)

for (i in 1:5) {
  test_indices = shuffled_numbers[((i-1)*(n/k)+1):(i*(n/k))]
  train_indices = setdiff(1:n, test_indices)
  df_train = df_drop[c(train_indices),]
  df_test = df_drop[c(test_indices),]
  fit_5 = lm(Rented_Bike_Count_bc ~ Hour + Temperature + Humidity
             + Solar_Radiation + Rainfall
             + Seasons, data = df_train)
  data = df_test['Rented_Bike_Count_bc']
  pred <- predict(fit_5, newdata = df_test)
  temp = (data-pred)^2
  lst_l = c(lst_l, mean(temp$Rented_Bike_Count_bc))
  
  X=model.matrix(fit_5)[,-1]
  Y=df_train$Rented_Bike_Count_bc
  c=seq(0,1,length=100)
  cv.out=cv.glmnet(X,Y,alpha=0,lambda=c, nfolds=5)
  bestlam=cv.out$lambda.min
  print(bestlam)
  out=glmnet(X,Y,alpha=0,lambda=bestlam)
  
  fit_temp = lm(Rented_Bike_Count_bc ~ Hour + Temperature + Humidity
                + Solar_Radiation + Rainfall
                + Seasons, data = df_test)
  X=model.matrix(fit_temp)[,-1]
  
  data = df_test['Rented_Bike_Count_bc']
  pred <- predict(out, newx = X)
  temp = (data-pred)^2
  lst_r = c(lst_r, mean(temp$Rented_Bike_Count_bc))
}

mean(lst_l)
mean(lst_r)
