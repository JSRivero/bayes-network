library(bnlearn)
library(readr)
library(Rgraphviz)
library(tictoc)

read_test <- function(){
  ep_draw = as.data.frame(t(read_csv('artificial_eps.csv')))
  colnames(ep_draw) = as.character(unlist(ep_draw[1,]))
  ep_draw = ep_draw[-1,]
  ep_draw <- as.data.frame(lapply(ep_draw, function(x) if(is.numeric(x)) factor(x) else x))
}


get_folder_date <- function(date,percent, wind, mov){
  unpack[, files] <- lista_files(percent, wind, mov)
  File = files[which(startsWith(files,date))[1]]
  return(File)
}

lista_files <- function(percent, wind, mov){
  general_path = 'C:/Users/Javier/Documents/MEGA/Universitattt/Master/Thesis/CDS_data/Sliced_ep_draw_new'
  dir_percent = paste('Slice', as.character(percent), sep = '_')
  dir_slice = paste('Slice',as.character(wind),as.character(mov), sep = '_')
  path = file.path(general_path, dir_percent, dir_slice)
  lista = list.files(path = path)
  return(list(path, lista))
}

read_dates <- function(date1, date2, percent, wind, mov, time_para, delay){
  unpack[path, files] <- lista_files(percent, wind, mov)
  firstFile = files[which(startsWith(files,date1))[1]]
  secondFile = files[which(startsWith(files,date2))[1]]
  data1 = reading_v2(time_para, delay, path, firstFile)
  data2 = reading_v2(time_para, delay, path, secondFile)
  return(list(data1, data2))
}

read_one_date <- function(date, percent, wind, mov, time_para, delay){
  unpack[path, files] <- lista_files(percent, wind, mov)
  firstFile = files[which(startsWith(files,date))[1]]
  data = reading_v2(time_para, delay, path, firstFile)
  return(data)
}


reading_v2 <- function(time_para, delay, path, directory){
  if (delay == 0){
    filename = paste(directory,'_raw_ep_drawups',as.character(time_para),'.csv',sep='')
  } else{
    filename_without_extension = paste(directory,'ep','drawups',as.character(time_para),
                                       'delay',as.character(delay), sep = '_')
    filename = paste(filename_without_extension,'csv', sep = '.')
  }
  path_to_file = paste(path,directory,filename,sep = '/')
  ep_draw = as.data.frame(t(read_csv(path_to_file, cols(.dfault = col_integer()), col_names = FALSE)))
  
  colnames(ep_draw) = as.character(unlist(ep_draw[1,]))
  ep_draw = ep_draw[-1,]
  rownames(ep_draw) = as.character(unlist(ep_draw[,1]))
  ep_draw = ep_draw[,-1]
  # aux = ncol(ep_draw)
  # ep_draw[,(aux-1):aux] = NULL
  
  # To eliminate the name of the institutions from the levels
  # Apparently, even though the row is deleted, the values still
  # remain as levels, and it completely screws the network
  
  ep_draw <- as.data.frame(lapply(ep_draw, function(x) if(is.factor(x)) factor(x) else x))
  
  # Finally, we need to remove the counterparties that only have one level:
  # aux1 = cbind(ep_draw)
  for (c in colnames(ep_draw)){
    if (length(levels(ep_draw[,c])) == 1){
      colm = which(colnames(ep_draw) == c)
      ep_draw = ep_draw[,-colm]
    }
  }
  return(ep_draw)
}

reading_original <- function(time_para, delay, path, directory){
  if (delay == 0){
    filename = paste('raw_ep_drawups',as.character(time_para),'.csv',sep='')
  } else{
    filename_without_extension = paste('ep','drawups',as.character(time_para),
                                       'delay',as.character(delay), sep = '_')
    filename = paste(filename_without_extension,'csv', sep = '.')
  }
  path_to_file = paste(path,directory,filename,sep = '/')
  ep_draw = as.data.frame(t(read_csv(path_to_file, cols(.dfault = col_integer()), col_names = FALSE)))
  
  colnames(ep_draw) = as.character(unlist(ep_draw[1,]))
  ep_draw = ep_draw[-1,]
  rownames(ep_draw) = as.character(unlist(ep_draw[,1]))
  ep_draw = ep_draw[,-1]
  # aux = ncol(ep_draw)
  # ep_draw[,(aux-1):aux] = NULL
  
  # To eliminate the name of the institutions from the levels
  # Apparently, even though the row is deleted, the values still
  # remain as levels, and it completely screws the network
  
  ep_draw <- as.data.frame(lapply(ep_draw, function(x) if(is.factor(x)) factor(x) else x))
  
  # Finally, we need to remove the counterparties that only have one level:
  # aux1 = cbind(ep_draw)
  for (c in colnames(ep_draw)){
    if (length(levels(ep_draw[,c])) == 1){
      colm = which(colnames(ep_draw) == c)
      ep_draw = ep_draw[,-colm]
    }
  }
  return(ep_draw)
}

delete_counterparty <- function(df, counterparty){
  colm = which(colnames(df) == counterparty)
  return(df[,-colm])
}


reading_split <- function(time_para, delay, wind, move, name_directory, all){
  unpack[path_ep, lista_dirs] = lista_files(wind, move)
  for (file in lista_dirs){
    
  }
}


reading <- function(time_para, delay){
  
  path = 'C:/Users/Javier/Documents/MEGA/Universitattt/Master/Thesis/CDS_data/epsilon_drawups'
  
  directory = paste(path, paste('stddev_',as.character(time_para),sep = ''), sep = '/')
  
  if (delay == 0){
    file_name = paste('raw_ep_drawups_',as.character(time_para),'.csv',sep='')
  } else{
    file_name = paste('ep_drawups_', as.character(time_para), '_delay_', as.character(delay),'.csv',sep='')
  }
  
  path = paste(directory,file_name, sep = '/')
  ep_draw = as.data.frame(t(read_csv(path, cols(.dfault = col_integer()), col_names = FALSE)))
  colnames(ep_draw) = as.character(unlist(ep_draw[1,]))
  ep_draw = ep_draw[-1,]
  rownames(ep_draw) = as.character(unlist(ep_draw[,1]))
  ep_draw = ep_draw[,-1]
  aux = ncol(ep_draw)
  ep_draw[,(aux-1):aux] = NULL
  
  # To eliminate the name of the institutions from the levels
  # Apparently, even though the row is deleted, the values still
  # remain as levels, and it completely screws the network
  ep_draw <- as.data.frame(lapply(ep_draw, function(x) if(is.factor(x)) factor(x) else x))
  
  return(ep_draw)
}

name_var <- function(v){
  return(deparse(substitute(v)))
}

name_var_csv <- function(v){
  return(paste(deparse(substitute(v)),'csv', sep = '.'))
}

direct_net <- function(bn, used_score, Data){
  aux = bn
  undir_arcs = undirected.arcs(aux)
  while (nrow(undir_arcs) != 0){
    aux2 = choose.direction(aux, data = Data, arc = undir_arcs[1,], debug = FALSE, criterion = used_score)
    undir_arcs_2 = undirected.arcs(aux2)
    if (nrow(undir_arcs)==nrow(undir_arcs_2)){
      arc_to_change = undir_arcs[sample(1:2,1),]
      aux2 = set.arc(aux, from = arc_to_change[1], to = arc_to_change[2])
    }
    undir_arcs = undir_arcs_2
    aux = aux2
  }


  # if (nrow(undir_arcs) != 0){
  #   print('funciona')
  #   for (i in nrow(undir_arcs)){
  #     aux2 = choose.direction(aux, data = ep_draw, arc = undir_arcs[i,], debug = TRUE, criterion = used_score)
  #     aux = aux2
  #   }
  # }

  return(aux)
}

compute_prob_v2 <- function(network, inst_def, data){
  node_names = nodes(network)
  parameters = bn.fit(network, data)
  
  CPGSD <- as.data.frame(matrix(0.0,length(node_names),1), row.names = node_names)
  colnames(CPGSD) = c('SD')
  param = parameters
  
  for (name_node in node_names){
    st_SD = paste("(", as.character(inst_def), "=='","1.0')", sep = "")
    
    st = paste("(", name_node, "=='1.0')", sep = "")
    cmd_SD = paste("cpquery(param, event = ", st, ", evidence = ", st_SD, ", n = 4 * 10**5)", sep = "")
    CPGSD[name_node,'SD'] <- eval(parse(text = cmd_SD))
    
    st = paste("(", name_node, "=='0.5')", sep = "")
    cmd_SD = paste("cpquery(param, event = ", st, ", evidence = ", st_SD, ", n = 4 * 10**5)", sep = "")
    CPGSD[name_node,'SD'] <-  CPGSD[name_node,'SD'] +  eval(parse(text = cmd_SD))
    
    
    # st_NSD = paste("(", as.character(inst_def), "=='", "0.0')", sep = "")
    # cmd_NSD = paste("cpquery(param, ", st, ", ", st_NSD, ", n = 10**5)", sep = "")
    # CPGSD[name_node,'NSD'] <- eval(parse(text = cmd_NSD))
  }
  
  CPGSD <- CPGSD[!(row.names(CPGSD) %in% c(inst_def)),,drop = FALSE]
  CPGSD <- CPGSD[order(-CPGSD$SD), , drop = FALSE]
  # arrange(cbind(row.names(CPGSD),CPGSD),desc(SD))
  return(CPGSD)
  
}

compute_prob_inv_v2 <- function(network, inst_def, data){
  node_names = nodes(network)
  parameters = bn.fit(network, data)
  
  CPGSD <- as.data.frame(matrix(0.0,length(node_names),1), row.names = node_names)
  colnames(CPGSD) = c('SD')
  param = parameters
  
  for (name_node in node_names){
    st_SD = paste("(", name_node, "=='","1.0')", sep = "")
    
    st = paste("(",  as.character(inst_def), "=='1.0')", sep = "")
    cmd_SD = paste("cpquery(param, event = ", st, ", evidence = ", st_SD, ", n = 4 * 10**5)", sep = "")
    CPGSD[name_node,'SD'] <- eval(parse(text = cmd_SD))
    
    st = paste("(",  as.character(inst_def), "=='0.5')", sep = "")
    cmd_SD = paste("cpquery(param, event = ", st, ", evidence = ", st_SD, ", n = 4 * 10**5)", sep = "")
    CPGSD[name_node,'SD'] <-  CPGSD[name_node,'SD'] +  eval(parse(text = cmd_SD))
    
    
    # st_NSD = paste("(", as.character(inst_def), "=='", "0.0')", sep = "")
    # cmd_NSD = paste("cpquery(param, ", st, ", ", st_NSD, ", n = 10**5)", sep = "")
    # CPGSD[name_node,'NSD'] <- eval(parse(text = cmd_NSD))
  }
  
  CPGSD <- CPGSD[!(row.names(CPGSD) %in% c(inst_def)),,drop = FALSE]
  CPGSD <- CPGSD[order(-CPGSD$SD), , drop = FALSE]
  # arrange(cbind(row.names(CPGSD),CPGSD),desc(SD))
  return(CPGSD)
  
}

compute_prob_group <- function(network, data, group){
  node_names = nodes(network)
  parameters = bn.fit(network, data)
  
  nodes_event = node_names[-match(group,node_names)]
  
  CPGSD <- as.data.frame(matrix(0.0,length(node_names),1), row.names = node_names)
  colnames(CPGSD) = c('SD')
  param = parameters
  st_SD = paste('((',group[1]," == '1.0')",sep = '')
  for (g in group[-1]){
    st_SD = paste(st_SD," & (",g," == '1.0')",sep = '')
  }
  st_SD = paste(st_SD,')',sep = '')
  print(st_SD)
  
  for (name_node in nodes_event){
    st = paste("(", name_node, "=='1.0')", sep = "")
    cmd_SD = paste("cpquery(param, event = ", st, ", evidence = ", st_SD, ", method='ls', n = 10**6)", sep = "")
    CPGSD[name_node,'SD'] <- eval(parse(text = cmd_SD))
    
    st = paste("(", name_node, "=='0.5')", sep = "")
    cmd_SD = paste("cpquery(param, event = ", st, ", evidence = ", st_SD, ", method='ls', n = 10**6)", sep = "")
    CPGSD[name_node,'SD'] <-  CPGSD[name_node,'SD'] +  eval(parse(text = cmd_SD))
    
    
    # st_NSD = paste("(", as.character(inst_def), "=='", "0.0')", sep = "")
    # cmd_NSD = paste("cpquery(param, ", st, ", ", st_NSD, ", n = 10**5)", sep = "")
    # CPGSD[name_node,'NSD'] <- eval(parse(text = cmd_NSD))
  }
  for (name_node in group){
    CPGSD[name_node,'SD'] <-1.0
  }
  
  # CPGSD <- CPGSD[!(row.names(CPGSD) %in% c(inst_def)),,drop = FALSE]
  CPGSD <- CPGSD[order(-CPGSD$SD), , drop = FALSE]
  # arrange(cbind(row.names(CPGSD),CPGSD),desc(SD))
  return(CPGSD)
  
}

uncer_group <- function(network, data, group, M){
  node_names = nodes(network)
  parameters = bn.fit(network, data)
  
  nodes_event = node_names[-match(group,node_names)]
  
  CPGSD <- as.data.frame(matrix(0.0,length(node_names),M), row.names = node_names)
  CPGSD <- compute_prob_group(network, data, group)
  for (i in 2:M){
    tic(i)
    CPGSD_aux <- compute_prob_group(network, data, group)
    CPGSD <- merge(CPGSD,CPGSD_aux, by = 0)
    rownames(CPGSD) = as.character(unlist(CPGSD[,1]))
    CPGSD = CPGSD[,-1]
    toc()
  }
  m = as.data.frame(apply(CPGSD,1,mean),row.names = rownames(CPGSD))
  s = as.data.frame(apply(CPGSD,1,sd),row.names = rownames(CPGSD))
  aux = merge(m,s,by = 0)
  rownames(aux) = as.character(unlist(aux[,1]))
  aux = aux[,-1]
  colnames(aux) = c('mean','std')
  aux <- aux[order(-aux$mean),,drop = FALSE]
  return(aux)
}

compare_prob <- function(net1, net2, data, inst){
  aux1 = compute_prob_v2(net1,inst,data)
  aux2 = compute_prob_v2(net2,inst,data)
  tot = merge(aux1,aux2,by=0)
  rownames(tot) = as.character(unlist(tot[,1]))
  tot = tot[,-1]
  colnames(tot) = c('net1','net2')
  tot <- tot[order(-tot$net1),,drop=FALSE]
  return(tot)
}

uncer_prob <- function(net,data, inst_def, M){
  para = bn.fit(net,data)
  node_names = nodes(net)
  node_names = node_names[-match(inst_def,node_names)]
  # M = 100
  prob = data.frame(matrix(0.0,length(node_names),M), row.names = node_names)
  for (m in 1:M){
    for (node in node_names){
      prob[node,m] = cond_prob(para, inst_def, node)
    }
  }
  tot_prob = data.frame()
  m = apply(prob,1,mean)
  s = apply(prob,1,sd)
  for (n in rownames(prob)){
     tot_prob[n,'mean']=m[n]
     tot_prob[n,'std'] = s[n]
  }
  tot_prob <- tot_prob[order(-tot_prob$mean),,drop=FALSE]
  return(tot_prob)
}

compute_prob <- function(net_information, inst_def, parameters){
  # List containing probabilities of defacult given sovereign default
  
  CPGSD <- as.data.frame(matrix(0.0,length(node_names),1), row.names = node_names)
  colnames(CPGSD) = c('SD')
  param = parameters
  
  for (name_node in node_names){
    st_SD = paste("(", as.character(inst_def), "=='","1.0')", sep = "")
    
    st = paste("(", name_node, "=='1.0')", sep = "")
    cmd_SD = paste("cpquery(param, event = ", st, ", evidence = ", st_SD, ", n = 4 * 10**5)", sep = "")
    CPGSD[name_node,'SD'] <- eval(parse(text = cmd_SD))
    
    st = paste("(", name_node, "=='0.5')", sep = "")
    cmd_SD = paste("cpquery(param, event = ", st, ", evidence = ", st_SD, ", n = 4 * 10**5)", sep = "")
    CPGSD[name_node,'SD'] <-  CPGSD[name_node,'SD'] +  eval(parse(text = cmd_SD))
    
    
    # st_NSD = paste("(", as.character(inst_def), "=='", "0.0')", sep = "")
    # cmd_NSD = paste("cpquery(param, ", st, ", ", st_NSD, ", n = 10**5)", sep = "")
    # CPGSD[name_node,'NSD'] <- eval(parse(text = cmd_NSD))
  }
  
  CPGSD <- CPGSD[!(row.names(CPGSD) %in% c(inst_def)),,drop = FALSE]
  CPGSD <- CPGSD[order(-CPGSD$SD), , drop = FALSE]
  # arrange(cbind(row.names(CPGSD),CPGSD),desc(SD))
  return(CPGSD)
}

compute_inv_prob <- function(node_names, inst_def, parameters){
  # List containing probabilities of defacult given sovereign default
  CPGSD <- as.data.frame(matrix(0.0,length(node_names),1), row.names = node_names)
  colnames(CPGSD) = c('SD')
  param = parameters
  
  for (name_node in node_names){
    st = paste("(", name_node, "=='1.0')", sep = "")
    
    st_SD = paste("(", as.character(inst_def), "=='","1.0')", sep = "")
    cmd_SD = paste("cpquery(param, event = ", st_SD, ", evidence = ", st, ", n = 4 * 10**5)", sep = "")
    CPGSD[name_node,'SD'] <- eval(parse(text = cmd_SD))
    
    st_SD = paste("(", as.character(inst_def), "=='0.5')", sep = "")
    cmd_SD = paste("cpquery(param, event = ", st_SD, ", evidence = ", st, ", n = 4 * 10**5)", sep = "")
    CPGSD[name_node,'SD'] <- CPGSD[name_node,'SD'] +  eval(parse(text = cmd_SD))
    
    
    # st_NSD = paste("(", as.character(inst_def), "=='", "0.0')", sep = "")
    # cmd_NSD = paste("cpquery(param, ", st, ", ", st_NSD, ", n = 10**5)", sep = "")
    # CPGSD[name_node,'NSD'] <- eval(parse(text = cmd_NSD))
  }
  
  CPGSD <- CPGSD[!(row.names(CPGSD) %in% c(inst_def)),,drop = FALSE]
  CPGSD <- CPGSD[order(-CPGSD$SD), , drop = FALSE]
  # arrange(cbind(row.names(CPGSD),CPGSD),desc(SD))
  return(CPGSD)
}

matrix_prob <- function(network, data){
  param = bn.fit(network, data)
  node_list = nodes(network)
  num = length(node_list)
  mat = as.data.frame(matrix(0,num,num))
  colnames(mat)= as.character(unlist(node_list))
  rownames(mat)= as.character(unlist(node_list))
  
  for (r in rownames(mat)){
    for (c in colnames(mat)){
      if (r!=c){
        mat[r,c] <-  cond_prob(param = param, evidence = r, event = c)
      }
    }
  }
  return(mat)
}

cond_prob <- function(param, evidence, event){
  
  st_SD = paste("(", as.character(evidence), "=='","1.0')", sep = "")
  
  st = paste("(", as.character(event), "=='1.0')", sep = "")
  cmd_SD = paste("cpquery(param, event = ", st, ", evidence = ", st_SD, ", n = 4 * 10**5, method = 'lw')", sep = "")
  result <- eval(parse(text = cmd_SD))
  
  st = paste("(", event, "=='0.5')", sep = "")
  cmd_SD = paste("cpquery(param, event = ", st, ", evidence = ", st_SD, ", n = 4 * 10**5, method = 'lw')", sep = "")
  result <-  result +  eval(parse(text = cmd_SD))
  
  return(result)
}

plot_net <- function(bayesian_net){
  g1 = as.graphAM(bayesian_net)
  # colors = c()
  # for (n in nodes(net_bic_not)){
  #   colors = append(colors, rgb(255, 255, 0, maxColorValue=255))
  # }
  plot(g1, attrs=list(node=list(fillcolor="lightgreen", fontsize = 30),edge=list(color="black")))
}

plot_net_colors <- function(bayesian_net, prob, darkness){
  col = colnames(prob)[1]
  g1 = as.graphAM(bayesian_net)
  for (n in nod){
    if (!n%in%p){
      l = append(l,n)
    }
  }
  colors = c()
  nAttrs <- list()
  nodes = nodes(bayesian_net)
  names(nodes) = nodes(bayesian_net)
  nAttrs <- list()
  nAttrs$label = nodes
  for (n in nodes(net_bic_not)){
    if (darkness == 0){
      colors = append(colors, rgb(255*(1-prob[n,col]), 255*(1-prob[n,col]), 255*(1-prob[n,col]), maxColorValue=255))
    }else{
      colors = append(colors, rgb(255*(darkness-prob[n,col])/darkness, 255*(1-prob[n,col]), 255*(1-prob[n,col]), maxColorValue=255))
    }
  }
  names(colors) = nodes(bayesian_net)
  nAttrs$fillcolor = colors
  plot(g1, nodeAttrs = nAttrs, attrs=list(node=list(fillcolor=colors, fontsize = 30),edge=list(color="black")))
}

compute_scores <- function(bnet, dataset){
  scores <- as.data.frame(matrix(0.0,1,5), row.names = c('net'))
  colnames(scores) <- c('loglik','bic','k2','bds','bde')
  for (s in colnames(scores)){
    scores['net',s] <- bnlearn::score(bnet, dataset, type = s)
  }
  return(scores)
}

score_table <- function(net, data){
  scores = compute_scores(net, data)
  for (c in colnames(scores)){
    cat(paste(' &', format(scores['net',c], digits = 6)))
  }
}

table_prob_v2 <- function(prob){
  i = 0
  for (r in row.names(prob)){
    i = i +1 
    cat(paste(i, ' & ',r, sep = ''))
    for (c in colnames(prob)){
      cat(paste(' & ', as.numeric(format(prob[r,c], digits = 4))))
    }
    cat('\\\\\n\\hline\n')
  }
}

del_cp <- function(df,cp){
  i = which(rownames(df) == cp)
  return(df[-i,,drop = F])
}

new_tableprob <- function(lista_df){
  df = lista_df[[1]]
  m = array(0,0)
  m = append(m, mean(df$SD))
  b = 1
  for (c in lista_df[-1]){
    df = cbind(df,names = rownames(c), SD = c)
    m = append(m,mean(c$SD))
  }
  
  cat('\\begin{center}\n\\begin{tabular}{')
  for (k in 1:(ncol(df)+2)){
    cat('|c')
  }
  cat(paste('|}\n\\cline{2-', as.character(ncol(df)+2) , '}\n'),'')
  cat('\\multicolumn{1}{c|}{ }')
  for (k in 1:length(lista_df)){
    cat(' & \\multicolumn{2}{|c|}{ SCORE }')
  }
  
  cat('\\\\\n\\hline\nnr.')
  for (k in 1:length(lista_df)){
    cat(' & Company & Prob')
  }
  cat('\\\\\n')
  cat('\\hline\n')
  co = colnames(df)
  i = 0
  for (r in rownames(df)){
    # print(r)
    i = i+1
    cat(paste(i, ' & ', r, sep = ''))
    for (c in 1:ncol(df)){
      # print(c)
      if (startsWith(co[c], 'names')){
        cat(paste(' & ', df[r,c]))
      }else{
        cat(paste(' & ', as.numeric(format(df[r,c]*100, digits = 4)), '\\%', sep = ''))
      }
    }
    cat('\\\\\n\\hline\n')
  }
  cat('Mean')
  for (a in m){
    cat(' & - & ')
    cat(as.numeric(format(a*100, digits = 4)))
    cat('\\%')
  }
  cat('\\\\\n\\hline\n')
  cat('\\end{tabular}\n\\end{center}\n')
}

table_times <- function(df){
  cat('\\begin{center}\n\\begin{tabular}{|c|c|c|c|c|}\n\\hline\n')
  cat('Period & Score & HC & Tabu 50 & Tabu 100 \\\\\n')
  cat('\\hline\n\\hline\n')
  for (r in rownames(df)){
    if (startsWith(r,'k2')){
      cat('\\multirow{2}{*}{1} & k2 ')
    }else{
      cat('& bds')
    }
    for(c in colnames(df)){
      cat(' & ')
      cat(df[r,c])
    }
    cat('\\\\\n\\hline\n')
  }
  cat('\\end{tabular}\n\\end{center}')
}

print_prob_as_dic <- function(b){
  for (a in rownames(b)){
    cat("\'")
    cat(a)
    cat("\'")
    cat(': ')
    cat(b[a,'SD'])
    cat(', ')
  }
}

print_edges_list <- function(b){
  for (i in 1:(length(arcs(b))/2)){
    a = arcs(b)[i,]
    cat("['")
    cat(a[1])
    cat("','")
    cat(a[2])
    cat("'],")
  }
}
  

# table_prob <- function(prob){
#   i = 0
#   for (r in row.names(prob)){
#     i = i +1 
#     cat(paste(i, ' & ', r,' & ', as.numeric(format(prob[r,'SD'], digits = 6)), '\\\\', sep = ''))
#     cat('\n\\hline\n')
#   }
# }

BN <- function(Data, used_score, RR, algthm, parameter){
  if (algthm == 'tabu'){
    boot = boot.strength(data = Data, R = RR, algorithm = algthm, algorithm.args=list(score=used_score,
                                        tabu = parameter[1], max.tabu = parameter[2], max.iter = parameter[3]))
  } else{
    boot = boot.strength(data = Data, R = RR, algorithm = algthm, algorithm.args=list(score=used_score))
  }
  
  avg_net = averaged.network(boot)
  final_net = direct_net(avg_net, used_score, Data)
  
  # print(nodes(final_net))
  
  param.bayes = bn.fit(final_net, data = Data, method = 'bayes')
  CPGSD_bayes = compute_prob(nodes(final_net), 'RUSSIA', param.bayes)
  # CPGSD_inv_bayes = compute_inv_prob(nodes(final_net), 'Russian.Fedn', param.bayes)
  
  param.mle = bn.fit(final_net, data = Data, method = 'mle')
  CPGSD_mle = compute_prob(nodes(final_net), 'RUSSIA', param.mle)
  # CPGSD_inv_mle = compute_inv_prob(nodes(final_net), 'Russian.Fedn', param.mle)
  
  return(list(CPGSD_bayes, CPGSD_mle, final_net))
  
}

unpack <- structure(NA,class="result")
"[<-.result" <- function(x,...,value) {
  args <- as.list(match.call())
  args <- args[-c(1:2,length(args))]
  length(value) <- length(args)
  for(i in seq(along=args)) {
    a <- args[[i]]
    if(!missing(a)) eval.parent(substitute(a <- v,list(a=a,v=value[[i]])))
  }
  x
}

mont.carl.BN <- function(Data, score, bootstrap, algthm, param, iter_mont){
  
  # Firstly we get the counterparties from the columns of the data
  counterparties = colnames(Data)
  num = length(counterparties)
  
  # Then we get get the array without RUSSIA
  index_RUSSIA = which(counterparties == 'RUSSIA')
  count_wo_rus = counterparties[-index_RUSSIA]
  
  tot_bayes = as.data.frame(matrix(0.0,num-1,iter_mont))
  rownames(tot_bayes) = as.character(unlist(count_wo_rus))
  colnames(tot_bayes) = 1:iter_mont
  
  tot_mle = as.data.frame(matrix(0.0,num-1,iter_mont))
  rownames(tot_mle) = as.character(unlist(count_wo_rus))
  colnames(tot_mle) = 1:iter_mont
  
  adjmat = as.data.frame(matrix(0, num, num))
  colnames(adjmat)= as.character(unlist(counterparties))
  rownames(adjmat)= as.character(unlist(counterparties))
  
  for (i in 1:iter_mont){
    print(i)
    tic('BN')
    # source('~/MEGA/Universitattt/Master/Thesis/Code/rbayes/epsilon_csv.R')
    unpack[CPGSD_bayes, CPGSD_mle, final_net] = BN(Data = Data, used_score = used_score, RR = R, algthm = algorithm, tabu_param)
    for (c in count_wo_rus){
      tot_mle[c,i] = CPGSD_mle[c,]
      tot_bayes[c,i] = CPGSD_bayes[c,]
    }
    arcs_net = arcs(final_net)
    
    for (i in 1:nrow(arcs_net)){
      # print(i)
      # print(arcs_net[i,])
      # print(adjmat[arcs_net[i,1],arcs_net[i,2]])
      adjmat[arcs_net[i,1],arcs_net[i,2]] = adjmat[arcs_net[i,1],arcs_net[i,2]] + 1
    }
    toc()
  }
  # Mean and variance of probabilities:
  # Bayes method
  prob_bayes = as.data.frame(matrix(0.0,num-1,2))
  rownames(prob_bayes) = as.character(unlist(count_wo_rus))
  colnames(prob_bayes) = c('mean','std')
  prob_bayes[,'mean'] = apply(tot_bayes,1,mean)
  prob_bayes[,'std'] = apply(tot_bayes,1,sd)
  prob_bayes = prob_bayes[order(-prob_bayes$mean), , drop = FALSE]
  
  # MLE method
  prob_mle = as.data.frame(matrix(0.0,num-1,2))
  rownames(prob_mle) = as.character(unlist(count_wo_rus))
  colnames(prob_mle) = c('mean','std')
  prob_mle[,'mean'] = apply(tot_mle,1,mean)
  prob_mle[,'std'] = apply(tot_mle,1,sd)
  prob_mle = prob_mle[order(-prob_mle$mean), , drop = FALSE]
  
  adjmat = adjmat/iter_mont
  
  return(list(prob_bayes, prob_mle, adjmat, final_net))
}