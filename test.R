library(tidyverse)
library(lubridate)

data
data <- data %>%
  mutate(time = as.Date(time)) %>%
  group_by(time) %>%
  summarise_all(mean)
data


plot(data$temperature)
