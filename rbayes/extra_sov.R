source('C:/Users/Javier/Documents/MEGA/Thesis/Code/rbayes/functions.R')


path_filtered = 'C:/Users/Javier/Documents/MEGA/Thesis/filtered/epsilon'


path_to_save = 'C:/Users/Javier/Documents/MEGA/Thesis/filtered/extra_sovereign'

delay = 3
time = 10

#Define score, algorithm and number of bootstraps

RR = 1000
algorithm = 'hc'

abs = 0
filter_market = TRUE

if (filter_market){
  directory1 = paste('std',as.character(time),'abs',as.character(abs), sep = '_')
}else{
  directory1 = paste('std',as.character(time),'abs',as.character(abs),'notfiltered', sep = '_')
}

data1 = reading_double_sov(time, delay, path_filtered, directory1)

for (used_score in c('bds','k2')){
  tic(used_score)
  net1 = averaged.network(boot.strength(data = data1, R = RR, algorithm = algorithm,
                                        algorithm.args=list(score=used_score)))
  
  net1 = direct_net(net1, used_score = used_score, Data = data1)
  
  prob = compute_prob_v2(net1, 'RUSSIA', data1)
  
  write.csv(prob, file.path(path_to_save, paste(used_score,'_extra_sov.csv',sep = '_')))
  
  if (filter_market){
    cmd = paste('net_',as.character(used_score),'_filter_two_sov = net1', sep = '')
    cmd_prob = paste('prob_',as.character(used_score),'_filter_two_sov = prob', sep = '')
    print(paste('net_',as.character(used_score),'_filter_two_sov = net1', sep = ''))
  }else{
    cmd = paste('net_',as.character(used_score),'_not_two_sov = net1', sep = '')
    cmd_prob = paste('prob_',as.character(used_score),'_not_two_sov = prob', sep = '')
  }
  eval(parse(text = cmd))
  eval(parse(text = cmd_prob))
  toc()
  
}

if (filter_market){
  name_file = paste('networks_filter_two_sov.RData',sep = '')
  save(net_bds_filter_two_sov,net_k2_filter_two_sov,prob_bds_filter_two_sov,prob_k2_filter_two_sov,
       file = file.path(path_to_save, name_file))
}else{
  name_file = paste('networks_notfilter_two_sov.RData',sep = '')
  save(net_bds_not_two_sov,net_k2_not_two_sov,prob_bds_not_two_sov,prob_k2_not_two_sov,
       file = file.path(path_to_save_net, name_file))
}