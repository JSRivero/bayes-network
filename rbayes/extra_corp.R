source('C:/Users/Javier/Documents/MEGA/Thesis/Code/rbayes/functions.R')

reading_extra <- function(time_para, delay, path, directory){
  if (delay == 0){
    filename = paste('raw_ep_drawups',as.character(time_para),'_extra.csv',sep='')
  } else{
    filename_without_extension = paste('ep','drawups',as.character(time_para),
                                       'delay',as.character(delay), sep = '_')
    filename = paste(filename_without_extension,'_extra.csv', sep = '')
  }
  path_to_file = paste(path,directory,filename,sep = '/')
  ep_draw = as.data.frame(t(read_csv(path_to_file, cols(.dfault = col_integer()), col_names = FALSE)))
  
  colnames(ep_draw) = as.character(unlist(ep_draw[1,]))
  ep_draw = ep_draw[-1,]
  rownames(ep_draw) = as.character(unlist(ep_draw[,1]))
  ep_draw = ep_draw[,-1]
  
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

path_filtered = 'C:/Users/Javier/Documents/MEGA/Thesis/filtered/epsilon'


path_to_save = 'C:/Users/Javier/Documents/MEGA/Thesis/filtered/extra_corporation'

delay = 3
time = 10

#Define score, algorithm and number of bootstraps

RR = 200
algorithm = 'hc'

abs = 0
filter_market = TRUE

if (filter_market){
  directory1 = paste('std',as.character(time),'abs',as.character(abs), sep = '_')
}else{
  directory1 = paste('std',as.character(time),'abs',as.character(abs),'notfiltered', sep = '_')
}

data1 = reading_extra(time, delay, path_filtered, directory1)

for (used_score in c('bds','k2')){
  tic(used_score)
  net1 = averaged.network(boot.strength(data = data1, R = RR, algorithm = algorithm,
                                        algorithm.args=list(score=used_score)))
  
  net1 = direct_net(net1, used_score = used_score, Data = data1)
  
  prob = compute_prob_v2(net1, 'RUSSIA', data1)
  
  write.csv(prob, file.path(path_to_save, paste(used_score,'prob.csv',sep = '_')))
  
  if (filter_market){
    cmd = paste('net_',as.character(used_score),'_filter_extra = net1', sep = '')
    cmd_prob = paste('prob_',as.character(used_score),'_filter_extra = prob', sep = '')
  }else{
    cmd = paste('net_',as.character(used_score),'_not_extra = net1', sep = '')
    cmd_prob = paste('prob_',as.character(used_score),'_not_extra = prob', sep = '')
  }
  eval(parse(text = cmd))
  eval(parse(text = cmd_prob))
  
  toc()
  
}

if (filter_market){
  name_file = paste('networks_filter_extra.RData',sep = '')
  save(net_bds_filter_extra,net_k2_filter_extra,prob_bds_filter_extra, prob_k2_filter_extra,
       file = file.path(path_to_save, name_file))
}else{
  name_file = paste('networks_notfilter_extra.RData',sep = '')
  save(net_bds_not_extra,net_k2_not_extra, prob_bds_not_extra, prob_k2_not_extra,
       file = file.path(path_to_save_net, name_file))
}