source('C:/Users/Javier/Documents/MEGA/Thesis/Code/rbayes/functions.R')

reading_delete <- function(time_para, delay, path, directory, deletions){
  ep_draw = reading_original(time_para, delay, path, directory)
  
  to_delete = intersect(colnames(ep_draw),deletions)
  
  for (c in to_delete){
    colnum = which(colnames(ep_draw) == c)
    ep_draw = ep_draw[,-colnum]
  }
  return(ep_draw)
}

path_filtered = 'C:/Users/Javier/Documents/MEGA/Thesis/filtered/epsilon'
path_to_save = 'C:/Users/Javier/Documents/MEGA/Thesis/filtered/delete_corporation'

delay = 3
time = 10

#Define score, algorithm and number of bootstraps

RR = 1000
algorithm = 'hc'

abs = 0
filter_market = TRUE

deletions = c('RUSAGB')
dirname = paste(deletions, collapse = '_')

# Directory to read epsilon draw ups from
if (filter_market){
  directory1 = paste('std',as.character(time),'abs',as.character(abs), sep = '_')
  dirname = paste('filter', dirname, sep = '_')
}else{
  directory1 = paste('std',as.character(time),'abs',as.character(abs),'notfiltered', sep = '_')
  dirname = paste('not', dirname, sep = '_')
}

path_to_save = file.path(path_to_save, dirname)

dir.create(path_to_save, showWarnings = FALSE)


data1 = reading_delete(time, delay, path_filtered, directory1, deletions)

for (used_score in c('bds','k2')){
  tic(used_score)
  net1 = averaged.network(boot.strength(data = data1, R = RR, algorithm = algorithm,
                                        algorithm.args=list(score=used_score)))
  
  net1 = direct_net(net1, used_score = used_score, Data = data1)
  
  prob = compute_prob_v2(net1, 'RUSSIA', data1)
  
  write.csv(prob, file.path(path_to_save, paste(used_score,'prob.csv',sep = '_')))
  
  if (filter_market){
    cmd = paste('net_',as.character(used_score),'_filter_deleted = net1', sep = '')
    cmd_prob = paste('prob_',as.character(used_score),'_filter_deleted = prob', sep = '')
  }else{
    cmd = paste('net_',as.character(used_score),'_not_deleted = net1', sep = '')
    cmd_prob = paste('prob_',as.character(used_score),'_not_deleted = prob', sep = '')
  }
  eval(parse(text = cmd))
  eval(parse(text = cmd_prob))
  
  toc()
  
}

if (filter_market){
  name_file = paste('networks_filter_deleted.RData',sep = '')
  save(net_bds_filter_deleted,net_k2_filter_deleted,prob_bds_filter_deleted, prob_k2_filter_deleted,
       file = file.path(path_to_save, name_file))
}else{
  name_file = paste('networks_notfilter_deleted.RData',sep = '')
  save(net_bds_not_deleted,net_k2_not_deleted, prob_bds_not_deleted, prob_k2_not_deleted,
       file = file.path(path_to_save_net, name_file))
}