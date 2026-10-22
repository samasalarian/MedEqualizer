generate_sunburst_df<-function(background_df,variables_list,user_data_analysis,sig_t,tol_1, tol_2,equity_metric_selected, whether_NMI){
  
  num_variables_df<-length(variables_list)
  background_info_df<- background_df %>%
    group_by(.dots = variables_list[1:num_variables_df]) %>% 
    summarise(background_n = sum(background_n))
  background_info_df<-na.omit(background_info_df)
  total_population_background<- sum(background_info_df$background_n)
  background_info_df$total_background<-rep(total_population_background, nrow(background_info_df))
  
  user_info_df<- user_data_analysis %>%
    group_by(.dots = variables_list[1:num_variables_df]) %>% 
    summarise(user_n = sum(user_n))
  user_info_df<-na.omit(user_info_df)
  total_population_user<- sum(user_info_df$user_n)
  user_info_df$total_user<-rep(total_population_user, nrow(user_info_df))
  
  participant_rate<-total_population_user/total_population_background
  
  index_max<-num_variables_df-1
  
  if (index_max>0){
    
    for (var_index in index_max:1){
      background_var<-background_df %>%
        group_by(.dots = variables_list[1:var_index]) %>% 
        summarise(background_n = sum(background_n))
      background_var<-na.omit(background_var)
      
      user_info_df_var<-user_data_analysis%>%
        group_by(.dots = variables_list[1:var_index])%>%
        summarise(user_n = sum(user_n))
      user_info_df_var<-na.omit(user_info_df_var)
      
      total_population_user_var<-sum(as.integer(user_info_df_var$user_n))
      total_population_background_var<-sum(as.integer(background_var$background_n))
      
      background_var$total_background<-rep(total_population_background_var, nrow(background_var))
      user_info_df_var$total_user<-rep(total_population_user_var, nrow(user_info_df_var))
      
      background_info_df<-plyr::rbind.fill(background_info_df, background_var) 
      user_info_df<-plyr::rbind.fill(user_info_df, user_info_df_var) 
    }
    
  }
  
  
  #combine user input data with the background info
  merged_df<-merge(background_info_df, user_info_df, by=1:(ncol(background_info_df)-2), all=TRUE)
  merged_df$user_n<-merged_df$user_n %>% tidyr::replace_na(0)
  merged_df$background_n<-merged_df$background_n %>% tidyr::replace_na(0)
  
  merged_df$total_user<-merged_df$total_user %>% replace_na(0)
  merged_df$total_background<-merged_df$total_background %>% tidyr::replace_na(0)
  
  #calculate the observed & background rates 
  merged_df$Observed_Rate<- mapply(Rate_Calculation,as.integer(merged_df$user_n),as.integer(merged_df$total_user))
  merged_df$Background_Rate<- mapply(Rate_Calculation,as.integer(merged_df$background_n),as.integer(merged_df$total_background))
  
  merged_df$pValue<- mapply(compare_population_proportion_2,as.integer(merged_df$background_n),as.integer(merged_df$total_background),as.integer(merged_df$user_n),as.integer(merged_df$total_user),rep(1-sig_t,nrow(merged_df)))
  merged_df$BH_p<- p.adjust(merged_df$pValue,method = "BH")
  
  if (whether_NMI){
    merged_df$EquityValue<- mapply(equity_metric_selected,as.numeric(merged_df$Background_Rate),as.numeric(merged_df$Observed_Rate),as.numeric(participant_rate), for_plot = FALSE)
    merged_df$EquityLable<-mapply(whether_biased_label_NMI, as.numeric(merged_df$EquityValue),as.numeric(merged_df$BH_p),sig_t,tol_1,tol_2)
    merged_df$EquityColors<-mapply(add_colors_NMI, as.numeric(merged_df$EquityValue),as.numeric(merged_df$BH_p),sig_t,tol_1,tol_2)
  }
  else{
    merged_df$EquityValue<- mapply(equity_metric_selected,as.numeric(merged_df$Background_Rate),as.numeric(merged_df$Observed_Rate), for_plot = FALSE)
    merged_df$EquityLable<-mapply(whether_biased_label, as.numeric(merged_df$EquityValue),as.numeric(merged_df$BH_p),sig_t,tol_1,tol_2,tol_2)
    merged_df$EquityColors<-mapply(add_colors, as.numeric(merged_df$EquityValue),as.numeric(merged_df$BH_p),sig_t,tol_1,tol_2,tol_2)
  }
  merged_df$na_count <- apply(merged_df[,1:num_variables_df,drop=F], 1, function(x) sum(is.na(x)))
  
  num_groups<-nrow(merged_df)
  
  new_df<- data.frame(ids = character(num_groups),
                      labels = character(num_groups),
                      parents = character(num_groups),
                      Observed_Rate = merged_df$Observed_Rate,
                      Ideal_Rate = merged_df$Background_Rate,
                      EquityColors=merged_df$EquityColors,
                      EquityLable=merged_df$EquityLable,
                      EquityValue=merged_df$EquityValue,
                      Observed_Number=merged_df$user_n,
                      Trial_Number = merged_df$total_user,
                      stringsAsFactors=FALSE)
  
  for (group_index in 1:num_groups){
    level<-num_variables_df-merged_df[group_index,'na_count']
    if (level>1){
      row_list <- unlist(merged_df[group_index,1:level])
      id_name<- paste(row_list,collapse=" - ")
      label_name<- as.character(row_list[level])
      parent_name<-paste(row_list[1:(level-1)],collapse=" - ")
    }
    else{
      id_name<- as.character(merged_df[group_index,1])
      label_name<- id_name
      parent_name<-""
    }
    
    new_df[group_index,'ids']<-id_name
    new_df[group_index,'labels']<-label_name
    new_df[group_index,'parents']<-parent_name
    
  }
  
  new_df$EquityValue<- mapply(signif,  new_df$EquityValue,digits = 3)
  new_df$Observed_Rate<- mapply(signif,  new_df$Observed_Rate,digits = 3)
  new_df$Ideal_Rate<- mapply(signif,  new_df$Ideal_Rate,digits = 3)
  
  return(new_df)
}

generate_sunburst_plotly3<-function(new_df){
  plot_ly(data= new_df)%>% add_trace(
    type='sunburst',
    ids=new_df$ids,
    labels=new_df$labels,
    parents=new_df$parents,
    leaf = list(opacity = 1),
    marker = list(colors = new_df$EquityColors),
    #textinfo = "current path+label",
    maxdepth =-1,
    insidetextorientation='radial',
    #hoverinfo = "current path+label+text",
    hovertext = ~paste(new_df$EquityLable,'</br>','Ideal Rate:', formattable(new_df$Ideal_Rate, digits = 3, format = "f"),'</br>','Observed Rate:', formattable(new_df$Observed_Rate, digits = 3, format = "f"),'</br>','No. of Participants:', new_df$Observed_Number),
    insidetextfont  = 2)

}

test_sunburst2<-function(given_background,varList,given_rct,write_filename, sig_t, lower_t, upper_t){
  df_processed<-generate_sunburst_df(given_background,as.list(varList),given_rct,sig_t,lower_t,upper_t,Log_Disparate_Impact, FALSE)
  write.csv(x= df_processed, file= write_filename)
  generate_sunburst_plotly3(df_processed)
}