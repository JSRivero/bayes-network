source('C:/Users/Javier/Documents/MEGA/Thesis/Code/rbayes/functions.R')

save_nets_probs <- function(filter_market,delete_market,scores){
  if (filter_market){
    if (delete_market){
      name_file = 'networks_deleted.RData'
      cmd = paste('save(net_',scores[1],'_deleted, prob_', scores[1],'_deleted',sep = '')
      for (i in 2:length(scores)){
        cmd = paste(cmd,', net_',scores[i],'_deleted, prob_', scores[i], '_deleted', sep = '')
      }
      cmd = paste(cmd,', file = file.path(path_to_save, name_file))',sep='')
      
    }else{
      name_file = 'networks_filter.RData'
      cmd = paste('save(net_',scores[1],'_filter, prob_', scores[1],'_filter',sep = '')
      for (i in 2:length(scores)){
        cmd = paste(cmd,', net_',scores[i],'_filter, prob_', scores[i], '_filter', sep = '')
      }
      cmd = paste(cmd,', file = file.path(path_to_save, name_file))',sep='')
    }
  }else{
    name_file = 'networks_filter.RData'
    cmd = paste('save(net_',scores[1],'_not, prob_', scores[1],'_not',sep = '')
    for (i in 2:length(scores)){
      cmd = paste(cmd,', net_',scores[i],'_not, prob_', scores[i], '_not', sep = '')
    }
    cmd = paste(cmd,', file = file.path(path_to_save, name_file))',sep='')
  }
  eval(parse(text = cmd))
}

period = '5Y'

delay = 3
time = 10

path = paste('C:/Users/Javier/Documents/MEGA/Thesis/results',period,sep='_')

# if (total){
#   if (delay == 0){
#     path = 'C:/Users/Javier/Documents/MEGA/Thesis/no_delay_results'
#   }else{
#     path = 'C:/Users/Javier/Documents/MEGA/Thesis/results_5Y'
#   }
# }else{
#   path = 'C:/Users/Javier/Documents/MEGA/Thesis/results_1Y'
# }

path_epsilon = file.path(path,'epsilon')

#Define score, algorithm and number of bootstraps

RR = 1000
algorithm = 'hc'
scores = c('bic','bds')

abs = 0
filter_market = FALSE
delete_market = FALSE

if (filter_market){
  if (delete_market){
    path_to_save = file.path(path,'networks_deleted')
    directory1 = paste('std',as.character(time),'abs',as.character(abs),'deleted_market', sep = '_')
  }else{
    path_to_save = file.path(path,'networks_filter')
    directory1 = paste('std',as.character(time),'abs',as.character(abs), sep = '_')
  }
}else{
  path_to_save = file.path(path,'networks_not_filtered')
  directory1 = paste('std',as.character(time),'abs',as.character(abs),'notfiltered', sep = '_')
}

data1 = reading_original(time, delay, path_epsilon, directory1)
data1 = delete_counterparty(data1,'EVRGSA')

oil_companies = c('GAZPRU','AKT','ROSNEF','LUKOIL')
banks = c('BOM','SBERBANK','BKECON')

for (used_score in scores){
  tic(used_score)
  net1 = averaged.network(boot.strength(data = data1, R = RR, algorithm = algorithm,
                                        algorithm.args=list(score=used_score)))
  
  net1 = direct_net(net1, used_score = used_score, Data = data1)
  
  prob = compute_prob_v2(net1, 'RUSSIA', data1)
  prob_oil = compute_prob_group(net1, data1, oil_companies)
  prob_banks = compute_prob_group(net1, data1, banks)
  write.csv(prob_oil, file.path(path_to_save, paste(used_score, period,'oil_pro.csv',sep = '_')))
  write.csv(prob_banks, file.path(path_to_save, paste(used_score, period,'bank_pro.csv',sep = '_')))
  
  mat_prob = matrix_prob(net1,data1)
  
  
  if (filter_market){
    if(delete_market){
      cmd = paste('net_',as.character(used_score),'_deleted = net1', sep = '')
      cmd_prob = paste('prob_',as.character(used_score),'_deleted = prob', sep = '')
      write.csv(prob, file.path(path_to_save, paste(used_score,'prob.csv',sep = '_')))
      write.csv(mat_prob, file.path(path_to_save, paste(used_score,'matrix.csv',sep = '_')))
    }else{
      cmd = paste('net_',as.character(used_score),'_filter = net1', sep = '')
      cmd_prob = paste('prob_',as.character(used_score),'_filter = prob', sep = '')
      write.csv(prob, file.path(path_to_save, paste(used_score,'prob.csv',sep = '_')))
      write.csv(mat_prob, file.path(path_to_save, paste(used_score,'matrix.csv',sep = '_')))
    }
  }else{
    cmd = paste('net_',as.character(used_score),'_not = net1', sep = '')
    cmd_prob = paste('prob_',as.character(used_score),'_not = prob', sep = '')
    write.csv(prob, file.path(path_to_save, paste(used_score,'prob_not_filtered.csv',sep = '_')))
    write.csv(mat_prob, file.path(path_to_save, paste(used_score,'matrix_not_filtered.csv',sep = '_')))
  }
  eval(parse(text = cmd))
  eval(parse(text = cmd_prob))
  
  plot_net(net1)
  
  dev.copy(png, file.path(path_to_save, paste(used_score,'png',sep='.')), width = 1200, height = 900)
  dev.off()
  
  toc()
  
}

save_nets_probs(filter_market, delete_market, scores)

# if (filter_market){
#   name_file = paste('networks_filter.RData',sep = '')
#   save(net_bds_filter,net_bic_filter, prob_bds_filter, prob_bic_filter,
#        file = file.path(path_to_save, name_file))
# }else{
#   name_file = paste('networks_notfilter.RData',sep = '')
#   save(net_bds_not,net_bic_not, prob_bds_not, prob_bic_not,
#        file = file.path(path_to_save, name_file))
# }