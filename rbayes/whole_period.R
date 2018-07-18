source('C:/Users/Javier/Documents/MEGA/Thesis/Code/rbayes/functions.R')

reading_total <- function(time_para, delay){
  if (delay == 0){
    filename = paste('raw_ep_drawups',as.character(time_para),'.csv',sep='')
  } else{
    filename_without_extension = paste('ep','drawups',as.character(time_para),
                                       'delay',as.character(delay), sep = '_')
    filename = paste(filename_without_extension,'csv', sep = '.')
  }
  path = 'C:/Users/Javier/Documents/MEGA/Thesis/CDS_data/epsilon_drawups_new'
  directory = paste('std',time_para,'long',sep = '_')
  path_to_file = file.path(path,directory,filename)
  
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

used_score = 'k2'
RR = 1000
algorithm = 'hc'

time = 10
delay = 3

tic()

data = reading_total(time,delay)

net_tot_k2 = averaged.network(boot.strength(data, R = RR, algorithm = algorithm, algorithm.args = list(score = used_score)))

plot_net(net_tot_k2)

net_tot_k2 = direct_net(net_tot_k2, used_score, data)

prob_whole_k2 = compute_prob_v2(net_tot_k2,'RUSSIA',data)

toc()