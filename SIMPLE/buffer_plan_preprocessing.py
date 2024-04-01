import numpy as np
import torch
import time
import numba as nb
import random
import multiprocessing as mp
import argparse
import os

parser=argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name')
parser.add_argument('--config', type=str, help='path to config file')
parser.add_argument('--gpu', type=str, default='0', help='which GPU to use')
parser.add_argument('--model_name', type=str, default='', help='name of stored model')
parser.add_argument('--dim_edge_feat', type=int, default=128, help='dim of edge feat')
parser.add_argument('--dim_node_feat', type=int, default=128, help='dim of node feat')
parser.add_argument('--mem_dim', type=int, default=100, help='dim of state vector')
parser.add_argument('--threshold',type=float, default=0.1, help='placement budget')
parser.add_argument('--mode',type=str, default='seq', help='placement strategy for multiple data types (sequential or parallel)')
parser.add_argument('--strategy',type=str, default='interval', help='data placement strategy. Candidate: interval, static.')
args=parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import time
import random
import dgl
import numpy as np
from trainer import *
from sampler import *
from data_processing import *
import time

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def static_data_placement(weight_node, weight_edge, threshold, data_name):
    node_idx, edge_idx = np.arange(len(weight_node)), np.arange(len(weight_edge))
    limit_node, limit_edge = int(threshold*len(weight_node)), int(threshold*len(weight_edge))
    plan_node = np.argsort(weight_node)[::-1][:limit_node]
    plan_edge = np.argsort(weight_edge)[::-1][:limit_edge]
    np.save('./intervals/'+data_name+'_static_node.npy', plan_node)
    np.save('./intervals/'+data_name+'_static_edge.npy', plan_edge)
    
def gen_intervals(uni_list, count_list, num_data):
    #initialize
    n = len(uni_list)
    last_flag = np.zeros(num_data,dtype=np.bool_)
    last_used = np.ones(num_data,dtype=int) * n
    count_all = np.zeros(num_data, dtype=int)
    start, end, IDs, interval_weight = [], [], [], []
    interval_weight_ = []
    #start gen intervals
    for i in range(n):
        #analyze the current batch
        uni, counts = uni_list[i], count_list[i]
        count_all[uni] = counts
        #generate intervals
        interval_flag = last_flag[uni] == True
        target_ID = uni[interval_flag]
        target_last_used = last_used[target_ID] 
        len_interval = i - target_last_used 
        net_weight = count_all[target_ID] / len_interval 
        #append
        start.extend(target_last_used) #start
        end.extend([i]*len(target_last_used)) #end
        IDs.extend(target_ID)
        interval_weight.extend(net_weight) 
        #update
        last_used[uni] = i
        last_flag[uni] = 1
    return start, end, IDs, interval_weight

def transform_intervals(start, end, IDs, interval_weight):
    start = np.array(start,dtype=np.int32)
    end = np.array(end,dtype=np.int32)
    IDs = np.array(IDs, dtype=np.int32)
    interval_weight = np.array(interval_weight)
    weight_order = np.argsort(interval_weight)[::-1]
    start = start[weight_order]
    end = end[weight_order]
    IDs = IDs[weight_order]
    return start, end, IDs

@nb.njit()
def select_interval_ID(start, end, num_batch, threshold):
    sel_interval_ID = []
    budget = np.ones(num_batch-1,dtype=np.int32)*threshold
    for i in range(len(start)):
        start_, end_ = start[i], end[i]
        flag = (budget[start_:end_]<1).any()
        if not flag:
            budget[start_:end_] -= 1
            sel_interval_ID.append(i)
        else:
            continue
    return sel_interval_ID

def test_saved_ratio(start, end, interval_weight):
    saved = np.sum(interval_weight)
    print('total saved:', saved)

def select_interval(sel_ID, start, end, IDs):
    sel_ID = np.array(sel_ID)
    start = start[sel_ID]
    end = end[sel_ID]
    IDs = IDs[sel_ID]
    
    return start, end, IDs


def sort_interval_by_time(start, end, IDs):
    order = np.argsort(start)
    start = start[order]
    end = end[order]
    IDs = IDs[order]
    return start, end, IDs

@nb.njit()
def gen_batch_plan(start, end, IDs, last_batch, left_index, batch_id, stop): #sorted by start
    #requiring start sorted
    if last_batch == 0:
        i = last_batch
        while start[i] <= batch_id:
            i += 1 # when the while loop breaks, then start[i] would be batch_id +1 now. 
        left_index = np.nonzero(end[last_batch:i] > batch_id)[0] + last_batch 
        target_ID = IDs[:i]
    else:
        i = last_batch
        while i<stop and start[i] <= batch_id:
            i += 1
        backward_ID = left_index[end[left_index] > batch_id]
        ID_cat = np.concatenate((backward_ID, np.arange(last_batch, i)))
        target_ID = IDs[ID_cat]
        left_index = ID_cat[end[ID_cat] > batch_id]
    #target_ID = np.sort(target_ID)
    return target_ID, i, left_index

def gen_total_plan(start, end, IDs, num_batch):
    cache_plan = []
    last_batch = 0
    left_index = np.array([],dtype=np.int32)
    stop = len(start)
    for i in range(num_batch-1):
        plan, last_batch, left_index = \
        gen_batch_plan(start, end, IDs, last_batch, left_index, i, stop)
        cache_plan.append(plan)
    return cache_plan

def save_cache_plan(cache_plan, name):
    np.save(name+'_cache_plan.npy', np.array(cache_plan,dtype=object), allow_pickle=True)
    
def check_mem(dims, nums):
    return dims * nums
    
def alloc_budget(vols, mems, dims, limit, data_name, layer, mailbox_size, threshold):
    budget = (limit*vols)
    total_budget = np.sum(budget)
    #this is the total memory budget, while for the data_placement_single, the budget is number of data. 
    #This needs convertion.
    store_flag = np.ones(len(vols),dtype=bool)
    if mailbox_size == 1:
        if budget[0] > mems[0]:
            store_flag[0] = 0
            budget[0] = mems[0]
            budget[1] = total_budget - mems[0] #allocate extra space.
    else:
        budget[0] = total_budget
        budget[1] = 0
        store_flag[1] = 0
        if budget[0] > mems[0]:
            budget[0] = mems[0]
    budget = (budget/dims).astype(int)
    for i in np.nonzero(store_flag==True)[0]:
        print('Type-{} data needs to be dynamically placed.'.format(i)) #TYPE-0: node; TYPE-1: edge.
    if layer > 1 and threshold == 0.1:
        np.save('./intervals/'+data_name+'_'+str(layer)+'_budget.npy', budget)
    if layer >1 and threshold != 0.1:
        np.save('./intervals/'+data_name+'_'+str(layer)+'_'+str(threshold)+'_budget.npy', budget)
    if layer == 1 and mailbox_size == 1:
        np.save('./intervals/'+data_name+'_budget.npy', budget)
    if layer == 1 and mailbox_size > 1:
        np.save('./intervals/'+data_name+'_budget_apan.npy', budget)
    return budget, store_flag

def data_placement_single(uni_list, count_list, num_data, num_batch, limit, name, data_name, layer, mailbox_size, threshold):
    t0 = time.time()
    start, end, IDs, interval_weight = gen_intervals(uni_list, count_list, num_data)
    del uni_list, count_list
    t1 = time.time()
    print('Interval generation takes:{:.2f}s'.format(t1-t0))

    start, end, IDs = transform_intervals(start, end, IDs, interval_weight)
    del interval_weight
    t2 = time.time()
    print('Interval transformation takes:{:.2f}s'.format(t2-t1))
    
    sel_ID = select_interval_ID(start, end, num_batch, limit)
    t3 = time.time()
    print('Interval-ID selection takes:{:.2f}s'.format(t3-t2))
    
    if len(sel_ID) != 0:
        start, end, IDs = select_interval(sel_ID, start, end, IDs)
        del sel_ID
    t4 = time.time()
    print('Interval selection takes:{:.2f}s'.format(t4-t3))
    
    if len(start) != 0:
        start, end, IDs = sort_interval_by_time(start, end, IDs)
    t5 = time.time()
    print('Interval sorting takes:{:.2f}s'.format(t5-t4))
    
    #save the selected intervals: start_batch_ids, end_batch_ids, mapped_data_ids
    #example:
    #np.save('./intervals/'+data_name+'_start_'+name+'_'+str(layer)+'.npy', start)
    #np.save('./intervals/'+data_name+'_end_'+name+'_'+str(layer)+'.npy', end)
    #np.save('./intervals/'+data_name+'_ids_'+name+'_'+str(layer)+'.npy', IDs)
    #t6 = time.time()
    #print('Saving intervals takes:{:.2f}s'.format(t6-t5))    
    
# set_seed(0)

t0 = time.time()
g, df = load_graph(args.data)
num_edge = len(df)
num_node = g['indptr'].shape[0]-1
print('num_node',num_node)
print('num_edge', num_edge)
t1 = time.time()
print('loading graph takes:{:.2f}s'.format(t1-t0))
print('start intializing...')
t0 = time.time()
sample_param, memory_param, gnn_param, train_param = parse_config(args.config)
train_edge_end = df[df['ext_roll'].gt(0)].index[0]
val_edge_end = df[df['ext_roll'].gt(1)].index[0]

sampler = None
if not ('no_sample' in sample_param and sample_param['no_sample']):
    sampler = ParallelSampler(g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32),
                              sample_param['num_thread'], 1, sample_param['layer'], sample_param['neighbor'],
                              sample_param['strategy']=='recent', sample_param['prop_time'],
                              sample_param['history'], float(sample_param['duration']))

if memory_param['mailbox_size'] > 1:
    sampler = None
    
group_indexes = list()
group_indexes.append(np.array(df[:train_edge_end].index // train_param['batch_size']))

#dims = np.array([2*args.mem_dim+args.dim_edge_feat, args.mem_dim, args.dim_edge_feat])
if memory_param['mailbox_size'] == 1:
    dims = np.array([3*args.mem_dim+args.dim_edge_feat+args.dim_node_feat, args.dim_edge_feat])
else:
    dims = np.array([args.mem_dim+(2*args.mem_dim+args.dim_edge_feat)*memory_param['mailbox_size'],args.dim_edge_feat])
    no_edge = True
#num_data = np.array([num_node,num_node,num_edge])
print('dims:',dims)
num_data = np.array([num_node, num_edge])
vols = np.zeros(len(dims),dtype=int)
mems = check_mem(dims, num_data)
total_mem = np.sum(mems)
names = ['node','edge']

#intialization for 'interval one'.
if args.strategy == 'interval':
    uni_list_node = []
    count_list_node = []

    uni_list_edge = []
    count_list_edge = []
    count = 0

if args.strategy == 'static':
    weight_node = np.zeros(num_node, dtype=np.int32)
    weight_edge = np.zeros(num_edge, dtype=np.int32)

    
t1 = time.time()
print('initialization takes:{:.2f}s'.format(t1-t0))

print('pre sampling...')
t0 = time.time()

for _, rows in df[:train_edge_end].groupby(group_indexes[random.randint(0, len(group_indexes) - 1)]):
    root_nodes = np.concatenate([rows.src.values, rows.dst.values]).astype(np.int32)
    ts = np.concatenate([rows.time.values, rows.time.values]).astype(np.float32)
    if sampler is None:
        nodes = root_nodes
    else:
        sampler.sample(root_nodes, ts)
        ret = sampler.get_ret()
        if sample_param['layer'] > 1:
            r0 = ret[0]
            r1 = ret[1]
            nodes = r1.nodes()
            edges = np.concatenate((r0.eid(),r1.eid()))
        else:
            r = ret[0]
            nodes = r.nodes()
            edges = r.eid()
    uni_node, count_node = np.unique(nodes, return_counts=True)
    uni_edge, count_edge = np.unique(edges, return_counts=True)
    if args.strategy == 'interval':
        uni_list_node.append(uni_node)
        count_list_node.append(count_node)
        vols[0] += np.sum(count_node)*dims[0]
        uni_list_edge.append(uni_edge)
        count_list_edge.append(count_edge)
        vols[1] += np.sum(count_edge)*dims[1]
    if args.strategy == 'static':
        weight_node[uni_node] += count_node
        weight_edge[uni_edge] += count_edge
    
######################################################################   
        
if args.strategy == 'interval':
    vols = vols/np.sum(vols)
    if len(uni_list_edge) != 0:
        uni_list_all = [uni_list_node, uni_list_edge]
        count_list_all = [count_list_node, count_list_edge]
        del uni_list_node, uni_list_edge, count_list_node, count_list_edge
    else:
        uni_list_all = [uni_list_node]
        count_list_all = [count_list_node]
        del uni_list_node, count_list_node
    del g, df
    num_batch = len(uni_list_all[0])
    print('vols', vols)
    
t1 = time.time()
print('pre sampling and analyzing takes:{:.2f}s'.format(t1-t0))

if args.strategy == 'static':
    t0 = time.time()
    static_data_placement(weight_node, weight_edge, args.threshold, args.data)
    t1 = time.time()
    print('Static data placement takes:{:.2f}s'.format(t1-t0))
    
        

if args.strategy == 'interval':
    t0 = time.time()
    limit = int(args.threshold*total_mem)
    budget, store_flag = alloc_budget(vols, mems, dims, limit, args.data, gnn_param['layer'], memory_param['mailbox_size'], args.threshold)
    t1 = time.time()
    print('Budget allocation takes:{:.2f}s'.format(t1-t0))
    if args.mode == 'seq':
        t_start = time.time()
        n = len(budget)
        for i in range(n):
            if store_flag[i] == True:
                print('Start preparation for type-{} data.'.format(i))
                t0 = time.time()
                data_placement_single(uni_list_all[i>0],count_list_all[i>0], num_data[i], num_batch, budget[i], names[i], args.data, gnn_param['layer'], memory_param['mailbox_size'],args.threshold)
                t1 = time.time()
                print('Generating plan for type-{} data takes:{:.2f}s'.format(names[i], t1-t0))
        #sequential
        t_end = time.time()
        print('Sequential plan generation takes:{:.2f}s'.format(t_end-t_start))

    if args.mode == 'par':
    #parallel
        t0 = time.time()
        processes = []
        n = len(budget)
        for i in range(n):
            if store_flag[i] == True:
                print('start preparation for type-{} data.'.format(i))
                p = mp.Process(target=data_placement_single, \
                args=(uni_list_all[i>0],count_list_all[i>0], num_data[i], num_batch, budget[i], names[i], args.data, gnn_param['layer'], memory_param['mailbox_size'], args.threshold))
                p.start()
                processes.append(p)
        for p in processes:
            p.join()
        t1 = time.time()
        print('Parallel plan generation takes:{:.2f}s'.format(t1-t0))