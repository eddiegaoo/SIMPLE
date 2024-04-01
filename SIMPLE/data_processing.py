import torch
import os
import yaml
import dgl
import time
import pandas as pd
import numpy as np
import time
import numba as nb

def load_feat(d, rand_de=0, rand_dn=0):
    node_feats = None
    if os.path.exists('../DATA/{}/node_features.pt'.format(d)):
        node_feats = torch.load('DATA/{}/node_features.pt'.format(d))
        if node_feats.dtype == torch.bool:
            node_feats = node_feats.type(torch.float32)
    edge_feats = None
    if os.path.exists('../DATA/{}/edge_features.pt'.format(d)):
        edge_feats = torch.load('DATA/{}/edge_features.pt'.format(d))
        if edge_feats.dtype == torch.bool:
            edge_feats = edge_feats.type(torch.float32)
    if rand_de > 0:
        if d == 'STACKOVERFLOW':
            edge_feats = torch.randn(63497049, 172)
        if d == 'LASTFM':
            edge_feats = torch.randn(1293103, rand_de)
        elif d == 'MOOC':
            edge_feats = torch.randn(411749, rand_de)
    if rand_dn > 0:
        if d == 'LASTFM':
            node_feats = torch.randn(1980, rand_dn)
        elif d == 'MOOC':
            edge_feats = torch.randn(7144, rand_dn)
    return node_feats, edge_feats

def load_graph(d):
    df = pd.read_csv('DATA/{}/edges.csv'.format(d))
    g = np.load('DATA/{}/ext_full.npz'.format(d))
    return g, df

def parse_config(f):
    conf = yaml.safe_load(open(f, 'r'))
    sample_param = conf['sampling'][0]
    memory_param = conf['memory'][0]
    gnn_param = conf['gnn'][0]
    train_param = conf['train'][0]
    return sample_param, memory_param, gnn_param, train_param

def to_dgl_blocks_orca(ret):
    mfgs = list()
    for i in range(2):
        for r in ret:
            b = dgl.create_block((r.col(), r.row()), num_src_nodes=r.dim_in(), num_dst_nodes=r.dim_out())
            b.srcdata['ID'] = torch.from_numpy(r.nodes())
            b.edata['dt'] = torch.from_numpy(r.dts())[b.num_dst_nodes():]
            b.srcdata['ts'] = torch.from_numpy(r.ts())
            b.edata['ID'] = torch.from_numpy(r.eid())
            mfgs.append(b.to('cuda:0'))
    mfgs = list(map(list, zip(*[iter(mfgs)] * 1)))
    mfgs.reverse()
    return mfgs
            

def to_dgl_blocks(ret, hist, reverse=False, cuda=True):
    mfgs = list()
    trans_time = 0
    for r in ret:
        if not reverse:
            b = dgl.create_block((r.col(), r.row()), num_src_nodes=r.dim_in(), num_dst_nodes=r.dim_out())
            #node_idx = r.nodes()
            b.srcdata['ID'] = torch.from_numpy(r.nodes())
            b.edata['dt'] = torch.from_numpy(r.dts())[b.num_dst_nodes():]
            b.srcdata['ts'] = torch.from_numpy(r.ts())
        else:
            b = dgl.create_block((r.row(), r.col()), num_src_nodes=r.dim_out(), num_dst_nodes=r.dim_in())
            b.dstdata['ID'] = torch.from_numpy(r.nodes())
            b.edata['dt'] = torch.from_numpy(r.dts())[b.num_src_nodes():]
            b.dstdata['ts'] = torch.from_numpy(r.ts())
        #edge_idx = r.eid()
        b.edata['ID'] = torch.from_numpy(r.eid())
        if cuda:
            mfgs.append(b.to('cuda:0'))
        else:
            mfgs.append(b)
    mfgs = list(map(list, zip(*[iter(mfgs)] * hist)))
    mfgs.reverse()
    return mfgs

def load_intervals(data_name, type_name, mailbox_size, threshold, multi_layer=False, mailbox=False, emb_reuse=False):
    if multi_layer and not mailbox:
        start = np.load('./intervals/'+data_name+'_start_'+type_name + '_tgat.npy')
        end = np.load('./intervals/'+data_name+'_end_'+type_name + '_tgat.npy')
        IDs = np.load('./intervals/'+data_name+'_ids_'+type_name + '_tgat.npy')
    if multi_layer and mailbox and threshold==0.1:
        start = np.load('./intervals/'+data_name+'_start_'+type_name + '_2.npy')
        end = np.load('./intervals/'+data_name+'_end_'+type_name + '_2.npy')
        IDs = np.load('./intervals/'+data_name+'_ids_'+type_name + '_2.npy')
    if (not multi_layer and mailbox_size == 1) or emb_reuse:
        start = np.load('./intervals/'+data_name+'_start_'+type_name + '.npy')
        end = np.load('./intervals/'+data_name+'_end_'+type_name + '.npy')
        IDs = np.load('./intervals/'+data_name+'_ids_'+type_name + '.npy')
    return start, end, IDs
                  
def intervals_to_gpu(start, end, IDs, device):
    start_tensor = torch.from_numpy(start).to(device)
    end_tensor = torch.from_numpy(end).to(device)
    IDs_tensor = torch.from_numpy(IDs).to(device)
    return start_tensor, end_tensor, IDs_tensor #remember to del original CPU ones.
                   
def load_total_intervals(d, budget, num_node, num_edge, mailbox_size, threshold, multi_layer=False, mailbox=False, emb_reuse=False):
    node_start, node_end, node_IDs = None, None, None
    edge_start, edge_end, edge_IDs = None, None, None
    if budget[0] != num_node:
        node_start, node_end, node_IDs = load_intervals(d, 'node', mailbox_size, threshold, multi_layer, mailbox, emb_reuse)
    if budget[1] != num_edge and budget[1] > 0:
        edge_start, edge_end, edge_IDs = load_intervals(d, 'edge', mailbox_size, threshold, multi_layer, mailbox, emb_reuse)
    return node_start, node_end, node_IDs, edge_start, edge_end, edge_IDs

def load_budget(d, mailbox_size, threshold, multi_layer=False, mailbox=False, emb_reuse=False):
    if multi_layer and not mailbox:
        budget = np.load('./intervals/'+d+'_budget_tgat.npy')
    if multi_layer and mailbox and threshold == 0.1:
        budget = np.load('./intervals/'+d+'_2_budget.npy')
    if multi_layer and mailbox and threshold != 0.1:
        budget = np.load('./intervals/'+d+'_2_'+str(threshold)+'_budget.npy')
    if (not multi_layer and mailbox_size == 1) or emb_reuse:
        print('yes here.')
        budget = np.load('./intervals/'+d+'_budget.npy')
    if not multi_layer and mailbox_size > 1:
        budget = np.load('./intervals/'+d+'_budget_apan.npy')
    return budget

def gen_batch_plan_tensor(start, end, IDs, batch_id):
    flag = (start <= batch_id) & (end > batch_id)
    return IDs[flag].long()


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
    return target_ID, i, left_index

def plan_to_gpu(plan, device):
    return torch.from_numpy(plan).long().to(device)

def init_batch_plan_fetch(start, device):
    last_batch = 0
    left_index = np.array([],dtype=np.int32)
    stop = len(start)
    plan = torch.tensor([], dtype=torch.long, device=device)
    return last_batch, left_index, stop, plan

def preprocessing_partial_plan(start, end, IDs, limit): #load intervals to GPU.
    cache_plan = 0
    for batch_id in range(limit):
        flag = (start<= batch_id) & (end > batch_id)
        target_ID = IDs[flag]
    left_index = (start <batch_id) & (end > batch_id)

def gen_total_plan(start, end, IDs, num_batch):
    last_batch = 0
    left_index = np.array([],dtype=np.int32)
    stop = len(start)
    for i in range(num_batch-1):
        plan, last_batch, left_index = \
        gen_batch_plan(start, end, IDs, last_batch, left_index, i, stop)


def node_to_dgl_blocks(root_nodes, ts, cuda=True):
    mfgs = list()
    b = dgl.create_block(([],[]), num_src_nodes=root_nodes.shape[0], num_dst_nodes=root_nodes.shape[0])
    b.srcdata['ID'] = torch.from_numpy(root_nodes)
    b.srcdata['ts'] = torch.from_numpy(ts)
    if cuda:
        mfgs.insert(0, [b.to('cuda:0')])
    else:
        mfgs.insert(0, [b])
    return mfgs

def update_indicators(idx, plan_last, plan, map_curr, gpu_flag, gpu_map, num_data, device):
    #initialization part.
    total_ids = torch.cat((plan_last, idx)) 
    cut = len(plan_last)
    map_curr.fill_(-1)
    dum = torch.arange(len(plan), dtype=torch.long, device=device)
    #indicator transformation
    gpu_flag.fill_(0) #re-initialize gpu_flag. gpu_flag is on GPU.
    gpu_flag[plan] = 1
    cache_mask = gpu_flag[total_ids]
    cache_map = total_ids[cache_mask] 
    #bypass 'np.unique/torch.unique' operation.
    local_order = torch.arange(len(cache_map), dtype=torch.long, device=device) 
    map_curr[cache_map] = local_order
    map_curr = map_curr[map_curr>=0] #unique order
    plan = cache_map[map_curr] #re-order the plan.
    #generate seperate ids
    ID_I, ID_II = map_curr<cut, map_curr>=cut
    local_ID_I, local_ID_II = map_curr[ID_I], map_curr[ID_II]
    local_ID_II -= cut
    #update gpu_map global_id --> local_id, local_ID_I is to access the id of cached data from last batch.
    #local_ID_II is to access the occurred data in the current batch.
    gpu_map.fill_(-1)
    gpu_map[plan[ID_I]] = dum[:len(local_ID_I)]
    gpu_map[plan[ID_II]] = dum[len(local_ID_I):len(local_ID_I)+ len(local_ID_II)]
    
    return gpu_flag, gpu_map, local_ID_I, local_ID_II

def update_indicators_2layer(idx, plan_last, plan, map_curr, gpu_flag, gpu_map, num_data, device):
    #initialization part.
    total_ids = torch.cat((plan_last, idx[0], idx[1])) #if cpu numba plan.
    cut0 = len(plan_last)
    cut1 = len(idx[0]) + cut0
    map_curr.fill_(-1)
    dum = torch.arange(len(plan), dtype=torch.long, device=device)
    #indicator transformation
    gpu_flag.fill_(0) #re-initialize gpu_flag. gpu_flag is on GPU.
    gpu_flag[plan] = 1
    cache_mask = gpu_flag[total_ids]
    cache_map = total_ids[cache_mask] 
    #bypass 'np.unique/torch.unique' operation.
    local_order = torch.arange(len(cache_map), dtype=torch.long, device=device) 
    map_curr[cache_map] = local_order
    map_curr = map_curr[map_curr>=0] #unique order
    plan = cache_map[map_curr] #re-order the plan.
    #generate seperate ids
    ID_I, ID_II, ID_III = map_curr<cut0, (map_curr>=cut0)&(map_curr<cut1), map_curr>=cut1
    local_ID_I, local_ID_II, local_ID_III = map_curr[ID_I], map_curr[ID_II], map_curr[ID_III]
    local_ID_II -= cut0
    local_ID_III -= cut1
    gpu_map.fill_(-1)
    gpu_map[plan[ID_I]] = dum[:len(local_ID_I)]
    gpu_map[plan[ID_II]] = dum[len(local_ID_I):len(local_ID_I)+ len(local_ID_II)]
    gpu_map[plan[ID_III]] = \
    dum[len(local_ID_I)+ len(local_ID_II):len(local_ID_I)+ len(local_ID_II)+len(local_ID_III)]
    
    return gpu_flag, gpu_map, local_ID_I, local_ID_II, local_ID_III

def update_buffs(buffs, data_II, local_ID_I, local_ID_II):
    #data_I is the cached data from last batch (buffs itself).
    #data_II is the occurred data in the current batch.
    buffs[:len(local_ID_I)] = buffs[local_ID_I] #data_I is the cached data from last batch.
    buffs[len(local_ID_I):len(local_ID_I)+ len(local_ID_II)] = data_II[local_ID_II]
    return buffs



def update_buffs_2layer(buffs, data_II, data_III, local_ID_I, local_ID_II, local_ID_III):
    buffs[:len(local_ID_I)] = buffs[local_ID_I] #data_I is the cached data from last batch.
    buffs[len(local_ID_I):len(local_ID_I)+len(local_ID_II)] = data_II[local_ID_II]
    buffs[len(local_ID_I)+len(local_ID_II):len(local_ID_I)+len(local_ID_II)+len(local_ID_III)] = \
    data_III[local_ID_III]
    return buffs

def gen_flag_and_mask_2layer(idx, gpu_flag, gpu_map, plan): #idx is a tuple: (#0 idx, #1 idx)
    gpu_mask_0 = gpu_flag[idx[0]]
    gpu_mask_1 = gpu_flag[idx[1]]
    gpu_mask = (gpu_mask_0, gpu_mask_1)
    
    gpu_ids_0 = idx[0][gpu_mask_0]
    gpu_ids_1 = idx[1][gpu_mask_1]
    
    cpu_ids_0 = idx[0][~gpu_mask_0]
    cpu_ids_1 = idx[1][~gpu_mask_1]
    cpu_ids = (cpu_ids_0, cpu_ids_1)
    
    gpu_local_ids_0 = gpu_map[gpu_ids_0]
    gpu_local_ids_1 = gpu_map[gpu_ids_1]
    gpu_local_ids = (gpu_local_ids_0,gpu_local_ids_1)
    return gpu_mask, gpu_local_ids, cpu_ids

def gen_flag_and_mask(idx, gpu_flag, gpu_map, plan): 
    #flag/map can be node or edge.
    gpu_mask = gpu_flag[idx]
    gpu_ids = idx[gpu_mask]
    gpu_local_ids = gpu_map[gpu_ids]
    cpu_ids = idx[~gpu_mask]
    
    return gpu_mask, gpu_local_ids, cpu_ids

def pre_load_all(start, end, IDs, num_batch, to_gpu):
    if to_gpu:
        batch_plan = []
        for i in range(num_batch):
            batch_plan.append(gen_batch_plan_tensor(start, end, IDs, i).cpu())
    else:
        batch_plan = []
        last_batch = 0
        left_index = np.array([],dtype=np.int32)
        stop = len(start)
        total_time = 0
        for i in range(num_batch):
            #print('batch:',i)
            t0 = time.time()
            plan, last_batch, left_index = \
            gen_batch_plan(start, end, IDs, last_batch, left_index, i, stop)
            batch_plan.append(torch.from_numpy(plan).long())
            t1 = time.time()
            total_time += t1-t0
        print('total_time:{:.2f}s'.format(total_time))
    return batch_plan
        
    
def load_batch_plan(data_name, type_name):
    batch_plan = np.load('./intervals/'+data_name+'_'+type_name+'_'+ 'cache_plan.npy',allow_pickle=True)
    return batch_plan

def load_batched_data_2layer(mfgs, node_idx, edge_idx, node_feats, edge_feats, node_gpu_mask, node_gpu_local_ids, node_cpu_ids, edge_gpu_mask, edge_gpu_local_ids, edge_cpu_ids, nfeat_buffs=None, efeat_buffs=None):
    #remember to initialize the buffs.
    if node_feats is not None:
        for b in mfgs[0]:
            if nfeat_buffs is not None:
                len_buff = nfeat_buffs.shape[0]
                if len_buff < len(node_idx):
                    amount, ext = len(node_idx)//len_buff, len(node_idx)%len_buff
                    srch = nfeat_buffs
                    for i in range(amount-1):
                        srch = torch.cat((srch, nfeat_buffs))
                    srch = torch.cat((srch, nfeat_buffs[:ext]))
                else:
                    srch = nfeat_buffs[:len(node_idx)]
                srch[node_gpu_mask] = nfeat_buffs[node_gpu_local_ids]
                srch[~node_gpu_mask] = node_feats[node_cpu_ids].cuda()

            else:
                srch = node_feats[node_idx] #all_on_gpu
            b.srcdata['h'] = srch
            
    if edge_feats is not None:
        layer = 0
        for mfg in mfgs:
            for b in mfg:
                if b.num_src_nodes() > b.num_dst_nodes():
                    if efeat_buffs is not None:
                        len_buff = efeat_buffs.shape[0]
                        if len_buff < len(edge_idx[layer]):
                            amount, ext = len(edge_idx[layer])//len_buff, len(edge_idx[layer])%len_buff
                            srch = efeat_buffs
                            for i in range(amount-1):
                                srch = torch.cat((srch, efeat_buffs))
                            srch = torch.cat((srch, efeat_buffs[:ext]))
                        else:
                            srch = efeat_buffs[:len(edge_idx[layer])]
                        srch[edge_gpu_mask[layer]] = efeat_buffs[edge_gpu_local_ids[layer]]
                        srch[~edge_gpu_mask[layer]] = edge_feats[edge_cpu_ids[layer]].cuda()
                    else:
                        srch = edge_feats[edge_idx[layer]]
                    b.edata['f'] = srch
            layer += 1
    return mfgs

def load_mem_edge_feats(edge_idx, edge_feats, edge_gpu_mask, edge_gpu_local_ids, edge_cpu_ids, efeat_buffs=None):
    if edge_feats is not None:
        if efeat_buffs is not None:
            len_buff = efeat_buffs.shape[0]
            if len_buff < len(edge_idx):
                amount, ext = len(edge_idx)//len_buff, len(edge_idx)%len_buff
                mem_edge_feats = efeat_buffs
                for i in range(amount-1):
                    mem_edge_feats = torch.cat((mem_edge_feats, efeat_buffs))
                mem_edge_feats = torch.cat((mem_edge_feats, efeat_buffs[:ext]))
            else:
                mem_edge_feats = efeat_buffs[:len(edge_idx)]
            mem_edge_feats[edge_gpu_mask] = efeat_buffs[edge_gpu_local_ids]
            mem_edge_feats[~edge_gpu_mask] = edge_feats[edge_cpu_ids].cuda()
        else:
            mem_edge_feats = edge_feats[edge_idx]
    return mem_edge_feats

def load_batched_data_orca(mfgs, node_idx, edge_idx, node_feats, edge_feats, node_gpu_mask, node_gpu_local_ids, node_cpu_ids, edge_gpu_mask, edge_gpu_local_ids, edge_cpu_ids, nfeat_buffs=None, efeat_buffs=None):
    #remember to initialize the buffs.
    if node_feats is not None:
        for b in mfgs[0]:
            if nfeat_buffs is not None:
                len_buff = nfeat_buffs.shape[0]
                if len_buff < len(node_idx):
                    amount, ext = len(node_idx)//len_buff, len(node_idx)%len_buff
                    srch = nfeat_buffs
                    for i in range(amount-1):
                        srch = torch.cat((srch, nfeat_buffs))
                    srch = torch.cat((srch, nfeat_buffs[:ext]))
                else:
                    srch = nfeat_buffs[:len(node_idx)]
                srch[node_gpu_mask] = nfeat_buffs[node_gpu_local_ids]
                srch[~node_gpu_mask] = node_feats[node_cpu_ids].cuda()

            else:
                srch = node_feats[node_idx] #all_on_gpu
            b.srcdata['h'] = srch
            
    if edge_feats is not None:
        if efeat_buffs is not None:
            len_buff = efeat_buffs.shape[0]
            if len_buff < len(edge_idx):
                amount, ext = len(edge_idx)//len_buff, len(edge_idx)%len_buff
                srch = efeat_buffs
                for i in range(amount-1):
                    srch = torch.cat((srch, efeat_buffs))
                srch = torch.cat((srch, efeat_buffs[:ext]))
            else:
                srch = efeat_buffs[:len(edge_idx)]
            srch[edge_gpu_mask] = efeat_buffs[edge_gpu_local_ids]
            srch[~edge_gpu_mask] = edge_feats[edge_cpu_ids].cuda()
        else:
            srch = edge_feats[edge_idx]
            
        for mfg in mfgs:
            for b in mfg:
                b.edata['f'] = srch
                
    return mfgs

def load_batched_data(mfgs, node_idx, edge_idx, node_feats, edge_feats, node_gpu_mask, node_gpu_local_ids, node_cpu_ids, edge_gpu_mask, edge_gpu_local_ids, edge_cpu_ids, nfeat_buffs=None, efeat_buffs=None):
    #remember to initialize the buffs.
    if node_feats is not None:
        for b in mfgs[0]:
            #cache lookup is decompsed into the gen_flag_and_mask func().
            if nfeat_buffs is not None:
                len_buff = nfeat_buffs.shape[0]
                if len_buff < len(node_idx):
                    amount, ext = len(node_idx)//len_buff, len(node_idx)%len_buff
                    srch = nfeat_buffs
                    for i in range(amount-1):
                        srch = torch.cat((srch, nfeat_buffs))
                    srch = torch.cat((srch, nfeat_buffs[:ext]))
                else:
                    srch = nfeat_buffs[:len(node_idx)]
                srch[node_gpu_mask] = nfeat_buffs[node_gpu_local_ids]
                srch[~node_gpu_mask] = node_feats[node_cpu_ids].cuda()

            else:
                srch = node_feats[node_idx] #all_on_gpu
            b.srcdata['h'] = srch
            
    if edge_feats is not None:
        for mfg in mfgs:
            for b in mfg:
                if b.num_src_nodes() > b.num_dst_nodes():
                    #cache lookup is decompsed into the gen_flag_and_mask func().
                    #load data
                    #idx = b.edata['ID'].long()
                    if efeat_buffs is not None:
                        len_buff = efeat_buffs.shape[0]
                        if len_buff < len(edge_idx):
                            amount, ext = len(edge_idx)//len_buff, len(edge_idx)%len_buff
                            srch = efeat_buffs
                            for i in range(amount-1):
                                srch = torch.cat((srch, efeat_buffs))
                            srch = torch.cat((srch, efeat_buffs[:ext]))
                        else:
                            srch = efeat_buffs[:len(edge_idx)]
                        srch[edge_gpu_mask] = efeat_buffs[edge_gpu_local_ids]
                        srch[~edge_gpu_mask] = edge_feats[edge_cpu_ids].cuda()
                    else:
                        srch = edge_feats[edge_idx]
                    b.edata['f'] = srch
    return mfgs
    
def prepare_input(mfgs, node_feats, edge_feats, plan_edge, plan_node, num_node, num_edge, combine_first=False, pinned=False, nfeat_buffs=None, efeat_buffs=None, nids=None, eids=None):
    if node_feats is not None:
        for b in mfgs[0]:
            srch = node_feats[b.srcdata['ID'].long()].float()
            b.srcdata['h'] = srch.cuda()
   
    if edge_feats is not None:
        for mfg in mfgs:
            for b in mfg:
                if b.num_src_nodes() > b.num_dst_nodes():
                    gpu_flag = torch.zeros
                    srch = edge_feats[b.edata['ID'].long()].float()
                    b.edata['f'] = srch.cuda()
                    
    return mfgs

def init_flags_and_maps(num_node, num_edge, budget, device):
    gpu_flag_e, gpu_flag_n = None, None
    gpu_map_e, gpu_map_n = None, None
    map_curr_e, map_curr_n = None, None
    dum_e, dum_n = None, None
    if budget[0] < num_node:
        gpu_flag_n = torch.zeros(num_node, dtype=torch.bool, device=device)
        gpu_map_n = torch.zeros(num_node, dtype=torch.long, device=device).fill_(-1)
        map_curr_n = gpu_map_n.clone()
    if budget[1] < num_edge:
        gpu_flag_e = torch.zeros(num_edge, dtype=torch.bool, device=device)
        gpu_map_e = torch.zeros(num_edge, dtype=torch.long, device=device).fill_(-1)
        map_curr_e = gpu_map_e.clone()
    return gpu_flag_e, gpu_flag_n, gpu_map_e, gpu_map_n, map_curr_e, map_curr_n
                            
def reset_indicators(gpu_flag, gpu_map):
    gpu_flag.fill_(0)
    gpu_map.fill_(-1)
    
def reset_buffs(buffs):
    buffs.fill_(0)


def allocate_buffs(budget, node_feats, edge_feats, num_node, num_edge, device, nfeat_flag):
    nfeat_buffs, efeat_buffs = None, None
    if node_feats is not None and budget[0] < num_node and nfeat_flag:
        nfeat_buffs = torch.zeros((budget[0],node_feats.shape[1]), dtype=node_feats.dtype, device=device)
    if edge_feats is not None and budget[1] < num_edge and budget[1] != 0:
        efeat_buffs = torch.zeros((budget[1],edge_feats.shape[1]), dtype=edge_feats.dtype, device=device)
    return nfeat_buffs, efeat_buffs