# Anomaly Detection With R
## Getting started with R.
install.packages("AnomalyDetection")
library(AnomalyDetection)
library(tidyverse)

## USArrests demo veri setini kullanacağız.
## Veri kümesindeki 50 satır arasında 15 rasgele satır alarak verilerin yalnızca bir alt kümesini kullanacağız. 
## sample() fonksiyonunu kullanarak gerçekleştirebiliyoruz. 
## Daha sonra scale() fonksiyonu ile veri setini standardize ediyoruz:

head(USArrests)

set.seed(123)
ss <- sample(1:50, 15)   # Rastgele 15 satır alınması için komut yazıldı.
df <- USArrests[ss, ]    # Verisetinden 15 gözlem seçildi.
df.scaled <- scale(df)   

df.scaled

## stats, factoextra, cluster package will be installed.

library(stats)
install.packages("factoextra")
library(factoextra)
library(cluster)

## To compute Euclidean distance, you can use the R base dist() function, as follow:
dist.eucl <- dist(df.scaled, method = "euclidean")

# Reformat as a matrix
# Subset the first 3 columns and rows and Round the values
round(as.matrix(dist.eucl)[1:3, 1:3], 1)

## Correlation-based uzaklık yaygın olarak kullanılır.

dist.cor <- get_dist(df.scaled, method = "pearson")

# Display a subset
round(as.matrix(dist.cor)[1:3, 1:3], 1)

# Flower datasına bakıldığında
data(flower)
head(flower, 3)

# Verinin yapısı
str(flower)

# Uzaklık matrisi oluşturulur.
dd <- daisy(flower)
round(as.matrix(dd)[1:3, 1:3], 2)

# Uzaklık matrisinin görselleştirilmesi
# burada "factoextra" paketinde bulunan 'fviz_dist' fonksiyonunu  kullanacağız.

fviz_dist(dist.eucl) # Euclidean ölçümüne göre uzaklık gösterimi
fviz_dist(dist.cor) # Pearson ölçümüne göre uzaklık gösterimi

## Red: high similarity (ie: low dissimilarity) | Blue: low similarity

## DataCamp çalışması  

install.packages("devtools")
devtools::install_github("twitter/AnomalyDetection")
library(AnomalyDetection)

install.packages("FNN")
library(FNN)
datasets::cars

plot(speed ~ dist, data = cars)
cars_knn <- get.knn(data=cars , k=5)

get.knn()
names(cars_knn)
# buradaki ilk 3 satırın k=5 olduğu durumda noktaların birbirlerine olan uzaklıkları görmekteyiz.
head(cars_knn$nn.dist, 3)

#satır bazlı skorların ortalamasını bulmaktayız.
cars_scores <- rowMeans(cars_knn$nn.dist)

## veri setinin ölçeklendirilmesi(scaled edilmesi)
cars_Scaled <- scale(cars)
cars_knn <- get.knn(data=cars , k=5)
cars$score <- rowMeans(cars_knn$nn.dist)

plot(speed ~ dist, cex = sqrt(score), data = cars, pch = 20)


## LOF - dbscan
install.packages("dbscan")
library(dbscan)

cars_lof <- lof(scale(cars), k=7)

cars_lof[1:10]

## LOF > 1 anomali olma ihtimali yüksek olanlar
## LOF < 1 anomali olma ihtimali düşük olanlar

cars$score_lof <- cars_lof
plot(speed ~ dist, cex = sqrt(score_lof), pch = 20)

## LOF CALCULATION

# Calculate the LOF for wine data
wine_lof <- lof(scale(wine), k = 5)

# Append the LOF score as a new column
wine$score <- wine_lof

## LOF VISUALIZATION

# Scatterplot showing pH, alcohol and LOF score
plot(pH ~ alcohol, data = wine, cex = score, pch = 20)

## LOF VS KKNN

# Scaled wine data
wine_scaled <- scale(wine)

# Calculate and append kNN distance as a new column
wine_nn <- get.knn(wine_scaled, k = 10)
wine$score_knn <- rowMeans(wine_nn$nn.dist)     

# Calculate and append LOF as a new column
wine$score_lof <- lof(wine_scaled, k = 10)

# Find the row location of highest kNN
which.max(wine$score_knn)

# Find the row location of highest LOF
which.max(wine$score_lof)

## ISOLATION FOREST ##

