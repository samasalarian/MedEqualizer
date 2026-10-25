colorpicker <- function(z,cut1 = log(0.8),cut2=log(0.9),cut3=-log(0.9),cut4=-log(0.8)){
  if(is.na(z)){return("white")}
  else if(z>cut4){return("white")}
  else if(z == -Inf){return("white")}
  else {return("black")}
}

bgpicker <- function(z,cut1 = log(0.8),cut2=log(0.9),cut3=-log(0.9),cut4=-log(0.8)){
  if(is.na(z)){return("black")}
  else if(z == -Inf){return("#ab2328")}
  else if(z <= cut1){return("#d58570")}
  else if( z > cut1 & z <= cut2){return("#eabcad")}
  else if( z > cut2 & z <= cut3){return("#d4e6e8")}
  else if( z > cut3 & z <= cut4){return("#a5b0cb")}
  else if (z>cut4){return("#00205b")}
}




make_table_combine<-function(background_info,RCT_info,x, sig_t, cutoff1,cutoff2, selected_metric) {
  user_colnames<-colnames(RCT_info)

  nonlab_factors<- user_colnames[!user_colnames%in% c("user_n")]
  num_nonlab<- length(nonlab_factors)

  background_info_df<-background_info %>%
    group_by(.dots = nonlab_factors[1]) %>% 
    summarise(background_n = sum(background_n))
  names(background_info_df)[1] <- "Group_Name"
  background_info_df<-na.omit(background_info_df)
  total_population_background<-sum(as.integer(background_info_df$background_n))
  background_info_df$total_background<-rep(total_population_background, nrow(background_info_df))
  
  user_info_df<-RCT_info%>%
    group_by(.dots = nonlab_factors[1]) %>% 
    summarise(user_n = sum(user_n))
  names(user_info_df)[1] <- "Group_Name"
  user_info_df<-na.omit(user_info_df)
  total_population_user<-sum(as.integer(user_info_df$user_n))
  user_info_df$total_user<-rep(total_population_user, nrow(user_info_df))
  
  for (i in 2:num_nonlab){
    background_info_df_var<-background_info %>%
      group_by(.dots = nonlab_factors[i]) %>% 
      summarise(background_n = sum(background_n))
    names(background_info_df_var)[1] <- "Group_Name"
    background_info_df_var<-na.omit(background_info_df_var)
    total_population_background_var<-sum(as.integer(background_info_df_var$background_n))
    background_info_df_var$total_background<-rep(total_population_background_var, nrow(background_info_df_var))
    
    user_info_df_var<-RCT_info%>%
      group_by(.dots = nonlab_factors[i]) %>% 
      summarise(user_n = sum(user_n))
    names(user_info_df_var)[1] <- "Group_Name"
    user_info_df_var<-na.omit(user_info_df_var)
    total_population_user_var<-sum(as.integer(user_info_df_var$user_n))
    user_info_df_var$total_user<-rep(total_population_user, nrow(user_info_df_var))
    background_info_df<-plyr::rbind.fill(background_info_df, background_info_df_var) 
    user_info_df<-plyr::rbind.fill(user_info_df, user_info_df_var) 
  }

  
  merged_df<-merge(background_info_df, user_info_df, by="Group_Name", all=TRUE)
  
  
  merged_df$user_n<-merged_df$user_n %>% replace_na(0)
  merged_df$background_n<-merged_df$background_n %>% replace_na(0)
  
  merged_df$total_user<-merged_df$total_user %>% replace_na(0)
  merged_df$total_background<-merged_df$total_background %>% replace_na(0)
  
  
  merged_df$participant_rate<- mapply(Rate_Calculation,as.integer(merged_df$total_user),as.integer(merged_df$total_background))
  
  #calculate the observed & background rates 
  merged_df$Observed_Rate<- mapply(Rate_Calculation,as.integer(merged_df$user_n),as.integer(merged_df$total_user))
  merged_df$Background_Rate<- mapply(Rate_Calculation,as.integer(merged_df$background_n),as.integer(merged_df$total_background))
  
  merged_df$pValue<- mapply(compare_population_proportion_2,as.integer(merged_df$background_n),as.integer(merged_df$total_background),as.integer(merged_df$user_n),as.integer(merged_df$total_user),rep(1-sig_t,nrow(merged_df)))
  merged_df$BH_p<- mapply(p.adjust,merged_df$pValue,method = "BH")
  merged_df$whether_significant<- mapply(whether_significant,as.numeric(merged_df$BH_p), sig_t)

  if (selected_metric == "LDI"){
    merged_df$EquityValue<- mapply(Log_Disparate_Impact,as.numeric(merged_df$Background_Rate),as.numeric(merged_df$Observed_Rate), for_plot = FALSE)
    merged_df$EquityLable<-mapply(whether_biased_label, as.numeric(merged_df$EquityValue),as.numeric(merged_df$BH_p),sig_t,cutoff1,cutoff2,cutoff2)
  }
  
  merged_df<-merged_df %>%mutate(Group_Name =  factor(Group_Name, levels = x)) %>%arrange(Group_Name)
  
  
  new_df_univariable<- data.frame(Group_Name = merged_df$Group_Name,
                                  BH_p =  merged_df$BH_p,
                                  Significant_Level =  merged_df$whether_significant,
                                  Equity_Value= merged_df$EquityValue,
                                  Equity_Level= merged_df$EquityLable,
                                  stringsAsFactors=FALSE)
  
  new_df_univariable<-new_df_univariable%>% mutate_if(is.numeric, round, digits=5)
  
  return(new_df_univariable)
  
}

preprocess_comparison_df<-function(sig_t, cut1,cut2, selected_metrics, file_type){
  ATUS_attributes<-c("Female", "Male","15-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75+")
  MIMIC_attributes<-c("Female", "Male","<=45", "46-65", "66-80","81+","White","Black","Unknown","Asian","Other","Alive","Died")
  MIMIC_attributes2<-c("Female", "Male","\u2264 45", "46-65", "66-80","81+","White","Black","Unknown","Asian","Other","Alive","Died")
  
  t1<-cut1
  t2<-cut2

  
  if (file_type == "ATUS"){
    df_comparison_studies<-make_table_combine(ATUSreference_processed, ATUSsynthetic_processed, ATUS_attributes,sig_t,t1,t2,selected_metrics)
    colnames(df_comparison_studies)[2:5] <- paste(colnames(df_comparison_studies)[2:5],"(1)",  sep = "")
  }
  else if (file_type == "MIMIC"){
    df_comparison_studies<-make_table_combine(MIMICRacereference_processed, MIMICRacesynthetic_processed,MIMIC_attributes,sig_t,t1,t2,selected_metrics)
    colnames(df_comparison_studies)[2:5] <- paste(colnames(df_comparison_studies)[2:5],"(1)",  sep = "")
  }
  
  if (file_type == "ATUS"){
    df_comparison_studies_new<-df_comparison_studies %>%mutate(Group_Name =  factor(Group_Name, levels = ATUS_attributes)) %>%arrange(Group_Name)
    df_comparison_studies_new$Group_Name<-ATUS_attributes
  }
  else if (file_type == "MIMIC"){
    df_comparison_studies_new<-df_comparison_studies %>%mutate(Group_Name =  factor(Group_Name, levels = MIMIC_attributes)) %>%arrange(Group_Name)
    df_comparison_studies_new$Group_Name<-MIMIC_attributes2
  }
  df_comparison_studies_new<-df_comparison_studies_new[c(1,seq(4,ncol(df_comparison_studies_new),by=4))]
  return(df_comparison_studies_new)
}

  
generate_table <- function(sig_t, lower_t, upper_t, metric_name,file_name){
  df_demo_comparison<-preprocess_comparison_df(sig_t,lower_t, upper_t, metric_name,file_name)
  colnames(df_demo_comparison)<-c("Characteristics","Equity")
  
  rownames(df_demo_comparison) <- NULL
  df_demo_comparison[,2:ncol(df_demo_comparison)] <- sapply(df_demo_comparison[,2:ncol(df_demo_comparison)],as.numeric)
  
  if (file_name == "ATUS"){
    demo_result<-format_table (df_demo_comparison, 
                               align =c("l","c","c","c"), 
                               lapply(2:ncol(df_demo_comparison), function(col) {
                                 area(row=1:nrow(df_demo_comparison), col) ~ formatter("span",
                                                                                       style = x ~ style(display = "block",
                                                                                                         "border-radius" = "4px",
                                                                                                         "padding-right" = "4px",
                                                                                                         color = sapply(x,colorpicker),
                                                                                                         "background-color" = sapply(x,bgpicker)),
                                                                                       x ~ sprintf("%.3f", x))
                               }
                               
                               ))%>%
      kable_styling("striped", full_width = TRUE,fixed_thead = TRUE) %>%
      pack_rows("Gender", 1,2) %>%
      pack_rows("Age group (years)", 3,9)
  }
  else if (file_name == "MIMIC"){
    demo_result<-format_table (df_demo_comparison, 
                               align =c("l","c","c","c"), 
                               lapply(2:ncol(df_demo_comparison), function(col) {
                                 area(row=1:nrow(df_demo_comparison), col) ~ formatter("span",
                                                                                       style = x ~ style(display = "block",
                                                                                                         "border-radius" = "4px",
                                                                                                         "padding-right" = "4px",
                                                                                                         color = sapply(x,colorpicker),
                                                                                                         "background-color" = sapply(x,bgpicker)),
                                                                                       x ~ sprintf("%.3f", x))
                               }
                               
                               ))%>%
      kable_styling("striped", full_width = TRUE,fixed_thead = TRUE) %>%
      pack_rows("Gender", 1,2) %>%
      pack_rows("Age group (years)",3,6)%>%
      pack_rows("Race/Ethnicity",7,11) %>%
      pack_rows("Mortality", 12,13)
  }
  
  gt::html(demo_result)
  
}
