if (!"pacman" %in% installed.packages()[,"Package"]) install.packages("pacman")

pacman::p_load(tidyverse,
               # curl,
               magrittr,
               lubridate, 
               zoo, 
               mFilter, 
               # devtools, 
               # knitr, 
               rio, 
               eurostat)

# theme for ggplot
theme <-  theme_bw()+ theme(strip.background=element_blank(),
                            strip.text=element_text(size=15),
                            title=element_text(size=16),
                            panel.border=element_blank(),
                            panel.grid.major=element_line(size=1),
                            legend.key=element_rect(colour="white"),
                            legend.position="bottom",
                            legend.text=element_text(size=10),
                            axis.text=element_text(size=10),
                            plot.title=element_text(hjust=0.5))
blueObsMacro <- "#0D5BA4"

awm <- import("awm19up15.csv")

awm %<>%
  transmute(gdp            = YER, # GDP (Real)
            defgdp         = YED, # GDP Deflator
            conso          = PCR, # Private Consumption (Real)
            defconso       = PCD, # Consumption Deflator
            inves          = ITR, # Gross Investment (Real)
            definves       = ITD, # Gross Investment Deflator
            wage           = WIN, # Compensation to Employees (Nominal)
            shortrate      = STN, # Short-Term Interest Rate (Nominal)
            employ         = LNN, # Total Employment (Persons)
            period         = as.Date(as.yearqtr(V1))) %>%
  gather(var, value, -period, convert = TRUE)

TED <- "TED---Output-Labor-and-Labor-Productivity-1950-2015.xlsx" 

# 19 countries in the Euro area (same as the AWM database)
EAtot_names <- c("Austria", "Belgium", "Cyprus", "Estonia", "Finland", "France", 
                 "Germany", "Greece", "Ireland", "Italy", "Latvia", "Lithuania","Luxembourg", 
                 "Malta", "Netherlands", "Portugal", "Slovak Republic","Slovenia", "Spain")

hours_confboard <- 
  import(TED, sheet = "Total Hours Worked", skip=2) %>%
  rename(country=Country)  %>%
  filter(country %in% EAtot_names) %>%
  select(-1) %>%
  gather(period, value, -country, na.rm=TRUE) %>%
  mutate(period = as.Date(paste0(period,"-07-01")),
         value = as.numeric(value)) %>%
  filter(period >= "1970-07-01" & period <= "2012-07-01")

chain <- function(to_rebase, basis, date_chain) {
  
  date_chain <- as.Date(date_chain, "%Y-%m-%d")
  
  valref <- basis %>%
    filter(period == date_chain) %>%
    transmute(var, value_ref = value) 
  
  res <- to_rebase %>%
    filter(period <= date_chain) %>%
    arrange(desc(period)) %>%
    group_by(var) %>%
    mutate(growth_rate = c(1, value[-1]/lag(value)[-1])) %>%
    full_join(valref, by = "var") %>%
    group_by(var) %>%
    transmute(period, value = cumprod(growth_rate)*value_ref)%>%
    ungroup() %>% 
    bind_rows(filter(basis, period > date_chain)) %>% 
    arrange(period)
  
  return(res)

}


# sum over the 14 countries
EA14_names <- c(filter(hours_confboard,period=="1970-07-01")$country)
hours_confboard_14 <- 
  hours_confboard %>%
  filter(country %in% EA14_names) %>% 
  group_by(period) %>% 
  summarize(value = sum(value),
            var = "hours")

# sum over the whole countries
hours_confboard_tot <- 
  hours_confboard %>%
  group_by(period) %>% 
  summarize(value = sum(value),
            var = "hours")

hours_confboard_chained <- 
  chain(to_rebase = hours_confboard_14, 
        basis = hours_confboard_tot, 
        date_chain = "1990-07-01")

hours_confboard_chained_q <- 
  tibble(period=seq(as.Date("1970-07-01"),
                    as.Date("2012-07-01"),
                    by = "quarter"),
         value=NA) %>% 
  left_join(hours_confboard_chained,by="period") %>% 
  select(-value.x) %>% 
  rename(value=value.y)

hours <- hours_confboard_chained_q

hours_approx <- 
  hours %>% 
  mutate(value=na.approx(value),
         var="hours_approx")

hours_spline <- 
  hours %>% 
  mutate(value=na.spline(value),
         var="hours_spline")

hoursts <- ts(hours$value,start=c(1970,4),f=4)
smoothed_hoursts <- tsSmooth(StructTS(hoursts,type="trend"))[,1]

hours_StructTS <- 
  hours %>% 
  mutate(value=smoothed_hoursts,
         var="hours_kalman")

hours_filtered <- bind_rows(hours_approx,hours_spline,hours_StructTS)

hours_filtered_levgr <- 
  hours_filtered %>% 
  mutate(value=log(value)-log(lag(value))) %>% 
  data.frame(ind2="2- Growth rates") %>% 
  bind_rows(data.frame(hours_filtered,ind2="1- Levels")) %>% 
  filter(period>="1971-01-01")

hours <- hours_StructTS %>% 
  mutate(var="hours")

df <- get_eurostat("namq_10_a10_e", 
                     filters = list(
                         geo = "EA19", 
                        freq = "Q",
                        unit = "THS_HW",
                        nace_r2 = "TOTAL",
                         s_adj = "SCA",
                        na_item = "EMP_DC"#,
                        # lastTimePeriod = 1
                                   ), 
                     time_format = "num")
# "Eurostat/namq_10_a10_e/Q.THS_HW.TOTAL.SCA.EMP_DC.EA19")

# convert Conference board annual hours worked series in 2000 basis index
valref <- filter(hours_confboard_chained,period=="2000-07-01")$value
hoursconfboard_ind <- 
  hours_confboard_chained %>% 
  transmute(period=period,
            var="Annual hours (original, Conference board)",
            value=value/valref)

# Quarterly hours worked series from Eurostat
# df <- rdb(ids = "Eurostat/namq_10_a10_e/Q.THS_HW.TOTAL.SCA.EMP_DC.EA19")
eurostat_data <-
  df %>% 
  select(period = time,
         value = values) %>% 
  mutate(var = "Quarterly hours (original, Eurostat)")

valref <- 
  eurostat_data %>% 
  filter(year(as.yearqtr(period))==2000) %>%
  summarize(value=mean(value))

eurostat_data_ind <- 
  eurostat_data %>% 
   mutate(value=value/valref$value,
          period = as.Date(as.yearqtr(period)))

# convert interpolated hours worked series in 2000 basis index
valref <- 
  hours %>% 
  filter(year(period)==2000) %>% 
  summarize(value=mean(value))

hours_ind <- 
  hours %>% 
  transmute(period,
            var="Quarterly hours (interpolated)",
            value=value/valref$value)


check <- rbind(hoursconfboard_ind,
                   hours_ind,
                   eurostat_data_ind)

# We build the URL for the DBnomics API to get annual population series for the 19 countries of the Euro Area
EAtot_code <- c("AT", "BE", "CY", "DE_TOT", "EE", "IE", 
                "EL", "ES","FX", "IT", "LT","LV", "LU", 
                "NL", "PT", "SK", "FI", "MT", "SI")

df <- get_eurostat("demo_pjanbroad", 
                     filters = list(
                         geo = EAtot_code, 
                        freq = "A",
                        unit = "NR",
                        sex = "T",
                         age = "Y15-64"# ,
                        # na_item = "EMP_DC"#,
                        # lastTimePeriod = 1
                                   ), 
                     time_format = "num")

# url_country <- paste0("A.NR.Y15-64.T.",paste0(EAtot_code, collapse = "+"))
# df <- rdb("Eurostat","demo_pjanbroad",mask = url_country)

pop_eurostat_bycountry <-
  df %>% 
  select(country = geo, period = time, value = values) %>% 
  mutate(period = ymd(paste0(period, "-01-01"))) %>% 
   filter(period >= "1970-01-01", 
         period <= "2013-01-01",
         !is.na(value))

plot_pop_eurostat_bycountry <-
  pop_eurostat_bycountry %>% 
  mutate(value = value/1000000)

# We sum the annual population for 16 countries in the Euro area
EA16_code <- filter(pop_eurostat_bycountry,period=="1970-01-01")$country
pop_a_16 <- 
  pop_eurostat_bycountry %>% 
  filter(country %in% EA16_code) %>% 
  group_by(period) %>% 
  summarize(value = sum(value),
            var = "pop")

# We sum the annual population for all the available countries
pop_a_tot <- 
  pop_eurostat_bycountry %>%
  group_by(period) %>% 
  summarize(value = sum(value),
            var="pop")

# We use the chain function detailed in the appendix
pop_chained <- 
  chain(to_rebase = pop_a_16, 
        basis = pop_a_tot, 
        date_chain = "1982-01-01")

pop_chained_q <- 
  tibble(period=seq(as.Date("1970-01-01"),
                    as.Date("2013-01-01"),
                    by = "quarter"),
         value=NA) %>% 
  left_join(pop_chained, by="period") %>% 
  select(-value.x) %>% 
  rename(value=value.y)

pop <- pop_chained_q

pop_approx <- 
  pop %>% 
  mutate(value=na.approx(value),
         var="pop_approx")

pop_spline <- 
  pop %>% 
  mutate(value=na.spline(value),
         var="pop_spline")

popts <- ts(pop$value,start=c(1970,1),f=4)
smoothed_popts <- tsSmooth(StructTS(popts,type="trend"))[,1]

pop_StructTS <- 
  pop %>% 
  mutate(value=smoothed_popts,
         var="pop_kalman")

pop_filtered <- bind_rows(pop_approx,pop_spline,pop_StructTS)

pop_filtered_levgr <- pop_filtered %>% 
  mutate(value=log(value)-log(lag(value))) %>% 
  data.frame(ind2="2- Growth rates") %>% 
  bind_rows(data.frame(pop_filtered,ind2="1- Levels")) %>% 
  filter(period>="1970-04-01")

pop <- pop_StructTS %>% 
  mutate(var="pop")

df <- get_eurostat("lfsq_pganws", 
                     filters = list(
                              freq = "Q",
                              unit = "THS_PER",
                              sex = "T",
                              citizen = "TOTAL",
                              age = "Y15-64",
                              wstatus = "POP",
                              geo = "EA20"
                    ))

# convert Conference board annual hours worked series in 2000 basis index
valref <- filter(pop_chained,period=="2005-01-01")$value
pop_a_ind <- 
  pop_chained %>% 
  transmute(period=period,
            var="Annual population (original, Eurostat)",
            value=value/valref)

# URL for quarterly population series
# df <- rdb(ids="Eurostat/lfsq_pganws/Q.THS.T.TOTAL.Y15-64.POP.EA19")

# freq	unit	sex	citizen	age	wstatus	geo	time	values

eurostat_data <- 
  df %>%
  select(period = time, 
         geo, 
         value = values) %>%
  rename(var= geo) %>%
  mutate(var= "Quarterly population (orginal, Eurostat)") %>%
  filter(period >= "2005-01-01")

valref <- 
  eurostat_data%>% 
  filter(year(period)==2005) %>% 
  summarize(value=mean(value))

eurostat_data_ind <- 
  eurostat_data %>% 
  mutate(value=value/valref$value)

# convert interpolated population series in 2000 basis index
valref <- 
  pop %>% 
  filter(year(period)==2005) %>% 
  summarize(value=mean(value))

pop_ind <- 
  pop %>% 
  transmute(period,
            var="Quarterly population (interpolated)",
            value=value/valref$value)

check <- bind_rows(pop_a_ind,
                   eurostat_data_ind,
                   pop_ind)

df <- get_eurostat("lfsq_pganws", 
                     filters = list(
                              freq = "Q",
                              unit = "THS_PER",
                              sex = "T",
                              citizen = "TOTAL",
                              age = "Y15-64",
                              wstatus = "POP",
                              geo = "EA20"
                    ))

# df <- rdb(ids="Eurostat/lfsq_pganws/Q.THS.T.TOTAL.Y15-64.POP.EA19")

old_data <- bind_rows(awm,
                      hours,
                      pop)

# URL for GDP/Consumption/Investment volumes and prices data
# variable.list <- c("B1GQ","P31_S14_S15","P51G")
# measure.list <- c("CLV10_MEUR","PD10_EUR")
# url_var <- paste0(variable.list,collapse = "+")
# url_meas <- paste0(measure.list,collapse = "+")
# filter <- paste0("Q.",url_meas,".SCA.", url_var, ".EA19")
# df <- rdb("Eurostat","namq_10_gdp",mask = filter)
df <- get_eurostat("namq_10_gdp", 
                     filters = list(
                         freq = "Q",
                         unit = c("CLV10_MEUR","PD10_EUR"),
                         s_adj = "SCA",
                         na_item = c("B1GQ","P31_S14_S15","P51G"),
                         geo = "EA19")
                    )
d1 <-
  df %>%
  select(period = time, 
         value = values,
         unit, 
         na_item,
         # series_name
        ) %>% 
  rename(var = na_item) %>% 
  mutate( var = ifelse(var=="B1GQ"&unit=="CLV10_MEUR","gdp",
                       ifelse(var=="B1GQ","defgdp",
                              ifelse(var=="P31_S14_S15"&unit=="CLV10_MEUR","conso",
                                     ifelse(var=="P31_S14_S15","defconso",
                                            ifelse(var=="P51G"&unit=="CLV10_MEUR","inves","definves")))))) %>%
  transmute(period,var,value)

# URL for wage series
# df <- rdb(ids="Eurostat/namq_10_a10/Q.CP_MEUR.SCA.TOTAL.D1.EA19")
df <- get_eurostat("namq_10_a10", 
                     filters = list(
                         freq = "Q",
                         unit = "CP_MEUR",
                         s_adj = "SCA",
                         nace_r2 = "TOTAL",
                         na_item = "D1",
                         geo = "EA19")
                    )
d2 <- 
  df %>%
  select(period = time, 
         value = values,
         unit) %>%
  rename(var=unit) %>%
  mutate(var="wage")

# URL for hours and employement
# url_meas <- "THS_HW+THS_PER"
# filter <- paste0("Q.",url_meas,".TOTAL.SCA.EMP_DC.EA19")
# df <- rdb("Eurostat","namq_10_a10_e",mask = filter)
df <- get_eurostat("namq_10_a10_e", 
                     filters = list(
                         freq = "Q",
                         unit = c("THS_HW","THS_PER"),
                         s_adj = "SCA",
                         nace_r2 = "TOTAL",
                         na_item = "EMP_DC",
                         geo = "EA19")
                    )
d3 <- 
  df %>%
  select(period = time, 
         value = values,
         unit) %>%
  rename(var= unit) %>%
  mutate(var=ifelse(var=="THS_HW","hours","employ")) %>% 
  transmute(period,var,value)

# URL for quarterly 3-month rates
# df <- rdb(ids="Eurostat/irt_st_q/Q.IRT_M3.EA")
df <- get_eurostat("irt_st_q", 
                     filters = list(
                         freq = "Q",
                         int_rt = "IRT_M3",
                         geo = "EA")
                    )
d4 <- 
  df %>%
  select(period = time, 
         value = values,
         geo) %>%
  rename(var= geo) %>%
  mutate(var= "shortrate")

# URL for quarterly population series
# df <- rdb(ids="Eurostat/lfsq_pganws/Q.THS.T.TOTAL.Y15-64.POP.EA19")
df <- get_eurostat("lfsq_pganws", 
                     filters = list(
                              freq = "Q",
                              unit = "THS_PER",
                              sex = "T",
                              citizen = "TOTAL",
                              age = "Y15-64",
                              wstatus = "POP",
                              geo = "EA20"
                    ))
d5 <- 
  df %>%
  select(period = time, 
         value = values,
         geo) %>%
  rename(var= geo) %>%
  mutate(var= "pop") %>%
  filter(period >= "2005-01-01")

recent_data <- bind_rows(d1,d2,d3,d4,d5)

maxDate <- 
  recent_data %>% 
  group_by(var) %>% 
  summarize(maxdate=max(period)) %>% 
  arrange(maxdate)

minmaxDate <- min(maxDate$maxdate)
recent_data %<>% filter(period <= minmaxDate)

vars <- c("gdp","conso","inves","defgdp","defconso","definves","shortrate", "hours", "wage", "employ")
new_df <- 
  recent_data %>%
  filter(var %in% vars)
old_df <- 
  awm %>%
  filter(var %in% vars) %>%
  bind_rows(hours)
df1 <- chain(basis = new_df,
             to_rebase = old_df,
             date_chain = "1999-01-01")

recent_pop_q <- filter(recent_data, var == "pop")

minDatePopQ <- min(recent_pop_q$period)

pop <- chain(basis = recent_pop_q,
             to_rebase= pop,
             date_chain=minDatePopQ)

plot_pop <- pop %>% 
  mutate(value=log(value)-log(lag(value))) %>% 
  data.frame(ind2="Growth rates") %>% 
  filter(period>="1970-04-01")

popts <- ts(pop$value,start=c(1970,1),f=4)
smoothed_popts <- hpfilter(popts, freq=1600)$trend
  
pop_StructTS <- 
  pop %>% 
  mutate(value=as.numeric(smoothed_popts),
         var="Smoothed population")
plot_pop <-
  pop %>%
  mutate(var="Original population")

pop_filtered <- bind_rows(plot_pop, pop_StructTS)

pop_filtered_levgr <- pop_filtered %>% 
  mutate(value=log(value)-log(lag(value))) %>% 
  data.frame(ind2="2- Growth rates") %>% 
  bind_rows(data.frame(pop_filtered,ind2="1- Levels")) %>% 
  filter(period>="1970-04-01")

pop <- pop_StructTS %>%
  mutate(var = "pop")

final_df <- bind_rows(df1, pop)

plot_df <- final_df
listVar <- list("Real GDP [1]" = "gdp",
                "Real consumption [2]" = "conso",
                "Real investment [3]" = "inves",
                "GDP deflator [4]" = "defgdp",
                "Consumption deflator [5]" = "defconso",
                "Investment deflator [6]" = "definves",
                "Real wage [7]" = "wage",
                "Hours worked [8]"= "hours",
                "Employment [9]" = "employ",
                "Interest rate [10]" = "shortrate",
                "Population [11]" = "pop")

plot_df$var <- factor(plot_df$var)
levels(plot_df$var)<-listVar
                     
ggplot(plot_df,aes(period,value))+
  geom_line(colour=blueObsMacro)+
  facet_wrap(~var,scales = "free_y",ncol = 3)+
  scale_x_date(expand = c(0.01,0.01)) +
  theme + xlab(NULL) + ylab(NULL) +
  theme(strip.text=element_text(size=12),
        axis.text=element_text(size=7))

EA_SW_rawdata <-
  final_df %>%
  spread(key = var, value= value)

EA_SW_rawdata %>%
  write.csv("EA_SW_rawdata.csv", row.names=FALSE)

EA_SW_data <-
  final_df %>% 
  mutate(period=gsub(" ","",as.yearqtr(period))) %>%
  spread(key = var, value = value) %>%
  transmute(period = period,
            gdp_rpc=1e+6*gdp/(pop*1000),
            conso_rpc=1e+6*conso/(pop*1000),
            inves_rpc=1e+6*inves/(pop*1000),
            defgdp=defgdp,
            wage_rph=1e+6*wage/defgdp/(hours*1000),
            hours_pc=1000*hours/(pop*1000),
            pinves_defl=definves/defgdp,
            pconso_defl=defconso/defgdp,
            shortrate=shortrate/100,
            employ=1000*employ/(pop*1000))

EA_SW_data %>% 
  na.omit() %>%
  write.csv("EA_SW_data.csv", row.names=FALSE)

final_df %>% 
  # mutate(period=gsub(" ","",as.yearqtr(period))) %>%
  spread(key = var, value = value) %>%
  transmute(period = period,
            gdp_rpc=1e+6*gdp/(pop*1000),
            conso_rpc=1e+6*conso/(pop*1000),
            inves_rpc=1e+6*inves/(pop*1000),
            defgdp=defgdp,
            wage_rph=1e+6*wage/defgdp/(hours*1000),
            hours_pc=1000*hours/(pop*1000),
            pinves_defl=definves/defgdp,
            pconso_defl=defconso/defgdp,
            shortrate=shortrate/100,
            employ=1000*employ/(pop*1000)) %>% 
    pivot_longer(!period) -> adjusted_EA_SW_data

ggplot(adjusted_EA_SW_data,aes(period,value))+
  geom_line(colour=blueObsMacro)+
  facet_wrap(~name,scales = "free_y",ncol = 3)+
  scale_x_date(expand = c(0.01,0.01)) +
  theme + xlab(NULL) + ylab(NULL) +
  theme(strip.text=element_text(size=12),
        axis.text=element_text(size=7))