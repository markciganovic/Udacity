fileURL <- "https://s3.amazonaws.com/udacity-hosted-downloads/ud651/wineQualityReds.csv"
if (!file.exists("wineQualityReds.csv")){
  download.file(fileURL, "wineQualityReds.csv")
}
library(tidyverse)
library(corrplot)
library(gridExtra)
library(moments)

winedata <- read_csv("wineQualityReds.csv", col_types = "_dddddddddddi")
options(digits=2, tibble.width = Inf, tibble.print_max = Inf)

ggplot(aes(x = quality), data = winedata) + geom_bar()
winedata %>% group_by(quality) %>% summarise(n = n())

h <- Map(function(variable, names){
  return(ggplot(aes(x = variable), data = winedata) + geom_histogram(bins = 30) +
           xlab(names))}, select(winedata, -quality), names(winedata)[1:11])
grid.arrange(grobs = h, top = "Histograms of Wine Attributes")

winedata %>% gather(attribute, value, 1:11) %>% group_by(attribute) %>% 
  summarise(min = min(value), 
            first_quantile = quantile(value, .25),
            median = median(value),
            mean = mean(value),
            third_quantile = quantile(value, .75),
            max = max(value), skew = skewness(value))

d <- Map(function(variable, names){
  return(ggplot(aes(x = variable), data = winedata) + 
           geom_density(aes(group = quality, color = factor(quality), 
                            fill = factor(quality)), alpha = 0.3) +
           labs(y  = "density of values", x = names, color = "quality", 
                fill = "quality") +
           scale_colour_brewer(palette = "Spectral") +  
           scale_fill_brewer(palette = "Spectral"))}, select(winedata, -quality), 
  names(winedata)[1:11])
grid.arrange(d[[1]] + theme(legend.position = "none"), 
             d[[2]] + theme(legend.position = "none") + ylab(NULL), 
             d[[3]] + ylab(NULL), d[[4]] + theme(legend.position = "none"), 
             d[[5]] + theme(legend.position = "none") + ylab(NULL), 
             d[[6]] + ylab(NULL), d[[7]] + theme(legend.position = "none"), 
             d[[8]] + ylab(NULL) + theme(legend.position = "none"), 
             d[[9]] + ylab(NULL), d[[10]] + theme(legend.position = "none"), 
             d[[11]] + ylab(NULL), 
             top = "Density Plots (Independent Variables based upon Quality)")

b <- Map(function(variable, names){
  return(ggplot(aes(x = factor(quality), y = variable), data = winedata) + 
           geom_boxplot() + labs(x = "quality", y = names))}, 
  select(winedata, -quality), names(winedata)[1:11])
grid.arrange(grobs = b, top = "Box Plots (Independent Variables based upon Quality)")

winedata %>% gather(attribute, value, 1:11) %>% group_by(attribute, quality) %>% 
  summarise(min = min(value), 
            first_quantile = quantile(value, .25), 
            median = median(value), 
            mean = mean(value),  
            third_quantile = quantile(value, .75), 
            max = max(value))

corrplot(cor(winedata), method = "color", addCoef.col="black", number.cex = .75, 
         title = "Correlation Coefficient Table", mar=c(0,0,1,0))

s <- Map(function(x_variable, y_variable, x_names, y_names){
  return(ggplot(aes(x = x_variable, y = y_variable), data = winedata) + 
           geom_jitter(aes(color = factor(quality)), alpha = .5) + 
           labs(color = "quality") +
           scale_colour_brewer(palette = "Spectral") +  
           scale_fill_brewer(palette = "Spectral") + 
           labs(x = x_names, y = y_names))}, 
  winedata[,c(1,3,3,11,11)], winedata[, c(2,2,1,5,4)], 
  names(winedata)[c(1,3,3,11,11)], names(winedata)[c(2,2,1,5,4)])

w <-  Map(function(x_variable, y_variable, x_names, y_names){
  return(ggplot(aes(x = x_variable, y = y_variable), data = winedata) + 
           geom_jitter(alpha = .3) + facet_wrap(~quality) + 
           labs(x = x_names, y = y_names))}, 
  winedata[,c(1,3,3,11,11)], winedata[, c(2,2,1,5,4)], 
  names(winedata)[c(1,3,3,11,11)], names(winedata)[c(2,2,1,5,4)])

Map(function(w, s){
  return(grid.arrange(w, s, ncol = 2))}, w, s)

s[[1]] + labs(y  = "volatile acidity (acetic acid - g / dm^3)", 
              x = "fixed acidity (acetic acid - g / dm^3)",
              title = "volatile acidity vs fixed acidity")

s[[2]] + labs(y  = "volatile acidity (acetic acid - g / dm^3)", 
              x = "citric acid - g / dm^3",
              title = "volatile acidity vs citric acid")

s[[3]] + labs(y  = "fixed acidity (tartaric acid - g / dm^3)", 
              x = "citric acid - g / dm^3",
              title = "fixed acidity vs citric acid")

model <- lm((quality ~ .), data = winedata)
formula <- formula(model)
as.formula(
  paste0("quality ~ ", round(coefficients(model)[1],2), "", 
         paste(sprintf(" %+.2f*%s ", 
                       coefficients(model)[-1],  
                       names(coefficients(model)[-1])), 
               collapse="")))
