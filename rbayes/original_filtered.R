source('C:/Users/Javier/Documents/MEGA/Thesis/Code/rbayes/functions.R')

path_filtered = 'C:/Users/Javier/Documents/MEGA/Thesis/filtered/epsilon'
path_prob = 'C:/Users/Javier/Documents/MEGA/Thesis/filtered/probabilities'
path_net = 'C:/Users/Javier/Documents/MEGA/Thesis/filtered/networks'

delay = 1

#Define score, algorithm and number of bootstraps

used_score = 'loglik'
RR = 10
algorithm = 'hc'

abs = 0
filter_market = FALSE

path_to_save_prob= file.path(path_prob,used_score)
path_to_save_net= file.path(path_net,used_score)

dir.create(path_to_save_net, showWarnings = FALSE)
dir.create(path_to_save_prob, showWarnings = FALSE)

for (time in list(10,15,20)){
  if (filter_market){
    directory1 = paste('std',as.character(time),'abs',as.character(abs), sep = '_')
  }else{
    directory1 = paste('std',as.character(time),'abs',as.character(abs),'notfiltered', sep = '_')
  }
  dir.create(file.path(path_to_save_prob,directory1))
  
  for (delay in list(1,2,3)){
    tic()
    
    data1 = reading_original(time, delay, path_filtered, directory1)
    net1 = averaged.network(boot.strength(data = data1, R = RR, algorithm = algorithm,
                                            algorithm.args=list(score=used_score)))
      
    net1 = direct_net(net1, used_score = used_score, Data = data1)
    
    plot_net(net1)
      
    if (filter_market){
      cmd = paste('net_',as.character(time),'_',as.character(delay),' = net1', sep = '')
    }else{
      cmd = paste('net_',as.character(time),'_',as.character(delay),'_not = net1', sep = '')
    }
      
      
    eval(parse(text = cmd))
    
    bayes1 = compute_prob_v2(net1,'RUSSIA',data1)
    
    write.csv(bayes1, file = file.path(path_to_save_prob, directory1, paste('prob_',as.character(time),'_delay_',
                                                                            as.character(delay),'.csv',sep ='')))
    
    cat('Done ')
    cat(as.character(time))
    cat(' with Delay: ')
    cat(as.character(delay))
    cat('\n')
    toc()
  }
}
if (filter_market){
  name_file = paste('networks_',as.character(abs),'_filter.RData',sep = '')
  save(net_10_1,net_10_2,net_10_3,net_15_1,net_15_2,net_15_3,net_20_1,net_20_2,net_20_3,
       file = file.path(path_to_save_net, name_file))
}else{
  name_file = paste('networks_',as.character(abs),'_notfilter.RData',sep = '')
  save(net_10_1_not,net_10_2_not,net_10_3_not,net_15_1_not,net_15_2_not,net_15_3_not,net_20_1_not,net_20_2_not,net_20_3_not,
       file = file.path(path_to_save_net, name_file))
}