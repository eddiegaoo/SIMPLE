import torch
import dgl
from layers import TimeEncode
import time

class MailBox():

    def __init__(self, memory_param, num_nodes, dim_edge_feat, _node_memory=None, _node_memory_ts=None,_mailbox=None, _mailbox_ts=None, _next_mail_pos=None, _update_mail_pos=None):
        self.memory_param = memory_param
        self.dim_edge_feat = dim_edge_feat
        if memory_param['type'] != 'node':
            raise NotImplementedError
        self.node_memory = torch.zeros((num_nodes, memory_param['dim_out']), dtype=torch.float32) if _node_memory is None else _node_memory
        self.node_memory_ts = torch.zeros(num_nodes, dtype=torch.float32) if _node_memory_ts is None else _node_memory_ts
        self.mailbox = torch.zeros((num_nodes, memory_param['mailbox_size'], 2 * memory_param['dim_out'] + dim_edge_feat), dtype=torch.float32) if _mailbox is None else _mailbox
        self.mailbox_ts = torch.zeros((num_nodes, memory_param['mailbox_size']), dtype=torch.float32) if _mailbox_ts is None else _mailbox_ts
        self.next_mail_pos = torch.zeros((num_nodes), dtype=torch.long) if _next_mail_pos is None else _next_mail_pos
        self.update_mail_pos = _update_mail_pos
        self.mailbox_buffs = None
        self.node_memory_buffs = None
        self.device = torch.device('cpu')
        
    def reset(self):
        self.node_memory.fill_(0)
        self.node_memory_ts.fill_(0)
        self.mailbox.fill_(0)
        self.mailbox_ts.fill_(0)
        self.next_mail_pos.fill_(0)
        if self.mailbox_buffs is not None:
            self.mailbox_buffs.fill_(0)
        if self.node_memory_buffs is not None:
            self.node_memory_buffs.fill_(0)
    
    def ts_to_gpu(self):
        self.mailbox_ts = self.mailbox_ts.cuda()
        self.node_memory_ts = self.node_memory_ts.cuda()
        self.next_mail_pos = self.next_mail_pos.cuda()
        #self.device = torch.device('cuda:0')
    
    def move_to_gpu(self):
        self.node_memory = self.node_memory.cuda()
        self.node_memory_ts = self.node_memory_ts.cuda()
        self.mailbox = self.mailbox.cuda()
        self.mailbox_ts = self.mailbox_ts.cuda()
        self.next_mail_pos = self.next_mail_pos.cuda()
        self.device = torch.device('cuda:0')

    
    def allocate_mailbox_buffs(self, budget, memory_param):
        self.mailbox_buffs = torch.zeros((budget[0], memory_param['mailbox_size'], 2 * memory_param['dim_out'] + self.dim_edge_feat), dtype=torch.float32).cuda()
        self.node_memory_buffs = torch.zeros((budget[0], memory_param['dim_out']), dtype=torch.float32).cuda()
    
    def update_mailbox_buffs(self, mail_data, mem_data, local_ID_I, local_ID_II, mailbox_size):
        #data_I is the cached data from last batch (buffs itself).
        #data_II is the occurred data in the current batch.
        self.mailbox_buffs[:len(local_ID_I)] = self.mailbox_buffs[local_ID_I] 
        if mailbox_size > 1 and len(local_ID_II) != 0: 
            self.mailbox_buffs[len(local_ID_I):len(local_ID_I)+ len(local_ID_II)] = \
            mail_data[local_ID_II].reshape(mail_data[local_ID_II].shape[0], mailbox_size, -1)
        if mailbox_size == 1:
            self.mailbox_buffs[len(local_ID_I):len(local_ID_I)+ len(local_ID_II)] = \
            mail_data[local_ID_II].unsqueeze(1)
        
        self.node_memory_buffs[:len(local_ID_I)] = self.node_memory_buffs[local_ID_I]
        self.node_memory_buffs[len(local_ID_I):len(local_ID_I)+ len(local_ID_II)] = \
        mem_data[local_ID_II]
        
    
    def allocate_pinned_memory_buffers(self, sample_param, batch_size):
        limit = int(batch_size * 3.3)
        if 'neighbor' in sample_param:
            for i in sample_param['neighbor']:
                limit *= i + 1
        self.pinned_node_memory_buffs = list()
        self.pinned_node_memory_ts_buffs = list()
        self.pinned_mailbox_buffs = list()
        self.pinned_mailbox_ts_buffs = list()
        for _ in range(sample_param['history']):
            self.pinned_node_memory_buffs.append(torch.zeros((limit, self.node_memory.shape[1]), pin_memory=True))
            self.pinned_node_memory_ts_buffs.append(torch.zeros((limit,), pin_memory=True))
            self.pinned_mailbox_buffs.append(torch.zeros((limit, self.mailbox.shape[1], self.mailbox.shape[2]), pin_memory=True))
            self.pinned_mailbox_ts_buffs.append(torch.zeros((limit, self.mailbox_ts.shape[1]), pin_memory=True))
    
    def prep_input_mails_eval(self, mfg):
        for i, b in enumerate(mfg):
            b.srcdata['mem'] = self.node_memory[b.srcdata['ID'].long()].cuda()
            b.srcdata['mem_ts'] = self.node_memory_ts[b.srcdata['ID'].long()]
            b.srcdata['mem_input'] = self.mailbox[b.srcdata['ID'].long()].cuda().reshape(b.srcdata['ID'].shape[0], -1)
            b.srcdata['mail_ts'] = self.mailbox_ts[b.srcdata['ID'].long()]
    
    def prep_input_mails(self, mfg, idx, gpu_mask, gpu_local_ids, cpu_ids):
        for i, b in enumerate(mfg):
            #idx = b.srcdata['ID'].long()            
            #for timestamps
            mem_ts = self.node_memory_ts[idx]
            mail_ts = self.mailbox_ts[idx]
            b.srcdata['mem_ts'] = mem_ts #already on GPU
            b.srcdata['mail_ts'] = mail_ts #already on GPU

            #for the majority of data.
            if self.node_memory_buffs is not None:
                len_buff = self.node_memory_buffs.shape[0]
                if len_buff < len(idx):
                    amount, ext = len(idx)//len_buff, len(idx)%len_buff
                    mem = self.node_memory_buffs
                    for i in range(amount-1):
                        mem = torch.cat((mem, self.node_memory_buffs))
                    mem = torch.cat((mem, self.node_memory_buffs[:ext]))
                else:
                    mem = self.node_memory_buffs[:len(idx)]
                mem[gpu_mask] = self.node_memory_buffs[gpu_local_ids]
                mem[~gpu_mask] = self.node_memory[cpu_ids].cuda()
            else:
                mem = self.node_memory[idx]
            b.srcdata['mem'] = mem
            
            if self.mailbox_buffs is not None:
                if len_buff < len(idx):
                    amount, ext = len(idx)//len_buff, len(idx)%len_buff
                    mem_input = self.mailbox_buffs
                    for i in range(amount-1):
                        mem_input = torch.cat((mem_input, self.mailbox_buffs))
                    mem_input = torch.cat((mem_input, self.mailbox_buffs[:ext]))
                else:
                    mem_input = self.mailbox_buffs[:len(idx)]
                mem_input = mem_input.reshape(len(idx),-1)
                if len(gpu_local_ids) != 0:
                    mem_input[gpu_mask] = self.mailbox_buffs[gpu_local_ids].reshape(len(gpu_local_ids), -1)
                mem_input[~gpu_mask] = self.mailbox[cpu_ids].cuda().reshape(len(cpu_ids),-1)
            else:
                mem_input = self.mailbox[idx].reshape(len(idx),-1)
            b.srcdata['mem_input'] = mem_input
            
    def update_memory_eval(self, nid, memory, root_nodes, ts, neg_sample=1):
        if nid is None:
            return
        num_true_src_dst = root_nodes.shape[0] // (neg_samples + 2) * 2
        with torch.no_grad():
            nid = nid[:num_true_src_dst]
            memory = memory[:num_true_src_dst]
            ts = ts[:num_true_src_dst]
            self.node_memory[nid.long()] = memory.to(self.device)
            self.node_memory_ts[nid.long()] = ts

    def update_memory(self, nid, memory, root_nodes, ts, gpu_flag, gpu_map, neg_samples=1):
        if nid is None:
            return
        num_true_src_dst = root_nodes.shape[0] // (neg_samples + 2) * 2
        with torch.no_grad():
            nid = nid[:num_true_src_dst]
            memory = memory[:num_true_src_dst]
            ts = ts[:num_true_src_dst]
            uni, inv = torch.unique(nid, sorted=False, return_inverse=True)
            perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
            perm = inv.new_empty(uni.size(0)).scatter_(0, inv, perm)
            nid = nid[perm]
            memory = memory[perm]
            ts = ts[perm]
            nid = nid.long()
            self.node_memory_ts[nid] = ts
            
            if gpu_flag is not None:
                #identify where to store.
                mask_ = gpu_flag[nid]
                gpu_ids_ = nid[mask_]
                gpu_local_ids_ = gpu_map[gpu_ids_]
                cpu_ids_ = nid[~mask_]
                #store.
                self.node_memory_buffs[gpu_local_ids_] = memory[mask_]
                self.node_memory[cpu_ids_] = memory[~mask_].to(self.device)
            else:
                self.node_memory[nid.long()] = memory
    
    def update_mailbox_eval(self, nid, memory, root_nodes, ts, edge_feats, block, neg_samples=1):
        with torch.no_grad():
            num_true_edges = root_nodes.shape[0] // 2
            # TGN
            if self.memory_param['deliver_to'] == 'self':
                src = root_nodes[:num_true_edges]
                dst = root_nodes[num_true_edges:num_true_edges * 2]
                mem_src = memory[:num_true_edges]
                mem_dst = memory[num_true_edges:num_true_edges * 2]
                if self.dim_edge_feat > 0:
                    src_mail = torch.cat([mem_src, mem_dst, edge_feats], dim=1)
                    dst_mail = torch.cat([mem_dst, mem_src, edge_feats], dim=1)
                else:
                    src_mail = torch.cat([mem_src, mem_dst], dim=1)
                    dst_mail = torch.cat([mem_dst, mem_src], dim=1)
                mail = torch.cat([src_mail, dst_mail], dim=1).reshape(-1, src_mail.shape[1])
                nid = torch.cat([src.unsqueeze(1), dst.unsqueeze(1)], dim=1).reshape(-1)
                mail_ts = ts[:num_true_edges * 2]
                # find unique nid to update mailbox
                uni, inv = torch.unique(nid, return_inverse=True)
                perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
                perm = inv.new_empty(uni.size(0)).scatter_(0, inv, perm)
                nid = nid[perm]
                mail = mail[perm]
                mail_ts = mail_ts[perm]
                if self.memory_param['mail_combine'] == 'last':
                    self.mailbox[nid.long(), self.next_mail_pos[nid.long()]] = mail.to(self.device)
                    self.mailbox_ts[nid.long(), self.next_mail_pos[nid.long()]] = mail_ts
    
    def update_mailbox(self, nid, memory, root_nodes, ts, edge_feats, block, gpu_flag, gpu_map, neg_samples=1):
        with torch.no_grad():
            #num_true_edges = root_nodes.shape[0] // (neg_samples + 2)
            num_true_edges = root_nodes.shape[0] // 2
            # TGN
            if self.memory_param['deliver_to'] == 'self':
                src = root_nodes[:num_true_edges]
                dst = root_nodes[num_true_edges:num_true_edges * 2]
                mem_src = memory[:num_true_edges]
                mem_dst = memory[num_true_edges:num_true_edges * 2]
                if self.dim_edge_feat > 0:
                    src_mail = torch.cat([mem_src, mem_dst, edge_feats], dim=1)
                    dst_mail = torch.cat([mem_dst, mem_src, edge_feats], dim=1)
                else:
                    src_mail = torch.cat([mem_src, mem_dst], dim=1)
                    dst_mail = torch.cat([mem_dst, mem_src], dim=1)
                mail = torch.cat([src_mail, dst_mail], dim=1).reshape(-1, src_mail.shape[1])
                nid = torch.cat([src.unsqueeze(1), dst.unsqueeze(1)], dim=1).reshape(-1)
                mail_ts = ts[:num_true_edges * 2]
                uni, inv = torch.unique(nid, sorted=False, return_inverse=True)
                perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
                perm = inv.new_empty(uni.size(0)).scatter_(0, inv, perm)
                nid = nid[perm]
                mail = mail[perm]
                mail_ts = mail_ts[perm]
                nid = nid.long()
                self.mailbox_ts[nid, self.next_mail_pos[nid]] = mail_ts
                #now the thing is that when we store the updated mails, we need to store some of them
                #on GPU and CPU respectively. nid is the unique data ID.
                if self.memory_param['mail_combine'] == 'last':
                    if gpu_flag is not None:
                        mask_ = gpu_flag[nid]
                        gpu_ids_ = nid[mask_]
                        gpu_local_ids_ = gpu_map[gpu_ids_]
                        cpu_ids_ = nid[~mask_]
                        #store.
                        self.mailbox_buffs[gpu_local_ids_, self.next_mail_pos[gpu_local_ids_]] = mail[mask_]
                        self.mailbox[cpu_ids_, self.next_mail_pos[cpu_ids_]] = mail[~mask_].to(self.device)
                    else:
                        self.mailbox[nid.long(), self.next_mail_pos[nid.long()]] = mail
            # APAN
            elif self.memory_param['deliver_to'] == 'neighbors':
                mem_src = memory[:num_true_edges]
                mem_dst = memory[num_true_edges:num_true_edges * 2]
                if self.dim_edge_feat > 0:
                    src_mail = torch.cat([mem_src, mem_dst, edge_feats], dim=1)
                    dst_mail = torch.cat([mem_dst, mem_src, edge_feats], dim=1)
                else:
                    src_mail = torch.cat([mem_src, mem_dst], dim=1)
                    dst_mail = torch.cat([mem_dst, mem_src], dim=1)
                mail = torch.cat([src_mail, dst_mail], dim=0)
                mail = torch.cat([mail, mail[block.edges()[0].long()]], dim=0)
                mail_ts = ts[:num_true_edges * 2]
                mail_ts = torch.cat([mail_ts, mail_ts[block.edges()[0].long()]], dim=0)
                if self.memory_param['mail_combine'] == 'mean':
                    (nid, idx) = torch.unique(block.dstdata['ID'], return_inverse=True)
                    mail = torch.scatter(mail, idx, reduce='mean', dim=0)
                    mail_ts = torch.scatter(mail_ts, idx, reduce='mean')
                    self.mailbox[nid.long(), self.next_mail_pos[nid.long()]] = mail
                    self.mailbox_ts[nid.long(), self.next_mail_pos[nid.long()]] = mail_ts
                elif self.memory_param['mail_combine'] == 'last':
                    nid = block.dstdata['ID']
                    # find unique nid to update mailbox
                    uni, inv = torch.unique(nid, return_inverse=True)
                    perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
                    perm = inv.new_empty(uni.size(0)).scatter_(0, inv, perm)
                    nid = nid[perm]
                    mail = mail[perm]
                    mail_ts = mail_ts[perm]
                    nid = nid.long()
                    self.mailbox_ts[nid, self.next_mail_pos[nid]] = mail_ts
                    if gpu_flag is not None:
                        mask_ = gpu_flag[nid]
                        gpu_ids_ = nid[mask_]
                        gpu_local_ids_ = gpu_map[gpu_ids_]
                        cpu_ids_ = nid[~mask_]
                        #store.
                        self.mailbox_buffs[gpu_local_ids_, self.next_mail_pos[gpu_local_ids_]] = mail[mask_]
                        self.mailbox[cpu_ids_, self.next_mail_pos[cpu_ids_]] = mail[~mask_].to(self.device)
                    else:
                        self.mailbox[nid, self.next_mail_pos[nid]] = mail
                else:
                    raise NotImplementedError
                if self.memory_param['mailbox_size'] > 1:
                    if self.update_mail_pos is None:
                        self.next_mail_pos[nid] = torch.remainder(self.next_mail_pos[nid] + 1, self.memory_param['mailbox_size'])
                    else:
                        self.update_mail_pos[nid] = 1
            else:
                raise NotImplementedError

    def update_next_mail_pos(self):
        if self.update_mail_pos is not None:
            nid = torch.where(self.update_mail_pos == 1)[0]
            self.next_mail_pos[nid] = torch.remainder(self.next_mail_pos[nid] + 1, self.memory_param['mailbox_size'])
            self.update_mail_pos.fill_(0)

    def offload_for_eval(self, gpu_flag, gpu_map):
        cpu_ids = torch.nonzero(gpu_flag).squeeze().long()
        gpu_ids = gpu_map[cpu_ids]
        self.mailbox[cpu_ids, self.next_mail_pos[cpu_ids]] = self.mailbox_buffs[gpu_ids].cpu()
        self.node_memory[cpu_ids] = self.node_memory_buffs[gpu_ids].cpu()
        
    
class GRUMemeoryUpdater(torch.nn.Module):

    def __init__(self, memory_param, dim_in, dim_hid, dim_time, dim_node_feat):
        super(GRUMemeoryUpdater, self).__init__()
        self.dim_hid = dim_hid
        self.dim_node_feat = dim_node_feat
        self.memory_param = memory_param
        self.dim_time = dim_time
        self.updater = torch.nn.GRUCell(dim_in + dim_time, dim_hid)
        self.last_updated_memory = None
        self.last_updated_ts = None
        self.last_updated_nid = None
        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)
        if memory_param['combine_node_feature']:
            if dim_node_feat > 0 and dim_node_feat != dim_hid:
                self.node_feat_map = torch.nn.Linear(dim_node_feat, dim_hid)

    def forward(self, mfg):
        for b in mfg:
            if self.dim_time > 0:
                time_feat = self.time_enc(b.srcdata['ts'] - b.srcdata['mem_ts'])
                b.srcdata['mem_input'] = torch.cat([b.srcdata['mem_input'], time_feat], dim=1)
            updated_memory = self.updater(b.srcdata['mem_input'], b.srcdata['mem'])
            self.last_updated_ts = b.srcdata['ts'].detach().clone()
            self.last_updated_memory = updated_memory.detach().clone()
            self.last_updated_nid = b.srcdata['ID'].detach().clone()
            if self.memory_param['combine_node_feature']:
                if self.dim_node_feat > 0:
                    if self.dim_node_feat == self.dim_hid:
                        b.srcdata['h'] += updated_memory
                    else:
                        b.srcdata['h'] = updated_memory + self.node_feat_map(b.srcdata['h'])
                else:
                    b.srcdata['h'] = updated_memory

class RNNMemeoryUpdater(torch.nn.Module):

    def __init__(self, memory_param, dim_in, dim_hid, dim_time, dim_node_feat):
        super(RNNMemeoryUpdater, self).__init__()
        self.dim_hid = dim_hid
        self.dim_node_feat = dim_node_feat
        self.memory_param = memory_param
        self.dim_time = dim_time
        self.updater = torch.nn.RNNCell(dim_in + dim_time, dim_hid)
        self.last_updated_memory = None
        self.last_updated_ts = None
        self.last_updated_nid = None
        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)
        if memory_param['combine_node_feature']:
            if dim_node_feat > 0 and dim_node_feat != dim_hid:
                self.node_feat_map = torch.nn.Linear(dim_node_feat, dim_hid)

    def forward(self, mfg):
        for b in mfg:
            if self.dim_time > 0:
                time_feat = self.time_enc(b.srcdata['ts'] - b.srcdata['mem_ts'])
                b.srcdata['mem_input'] = torch.cat([b.srcdata['mem_input'], time_feat], dim=1)
            updated_memory = self.updater(b.srcdata['mem_input'], b.srcdata['mem'])
            self.last_updated_ts = b.srcdata['ts'].detach().clone()
            self.last_updated_memory = updated_memory.detach().clone()
            self.last_updated_nid = b.srcdata['ID'].detach().clone()
            if self.memory_param['combine_node_feature']:
                if self.dim_node_feat > 0:
                    if self.dim_node_feat == self.dim_hid:
                        b.srcdata['h'] += updated_memory
                    else:
                        b.srcdata['h'] = updated_memory + self.node_feat_map(b.srcdata['h'])
                else:
                    b.srcdata['h'] = updated_memory

class TransformerMemoryUpdater(torch.nn.Module):

    def __init__(self, memory_param, dim_in, dim_out, dim_time, train_param):
        super(TransformerMemoryUpdater, self).__init__()
        self.memory_param = memory_param
        self.dim_time = dim_time
        self.att_h = memory_param['attention_head']
        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)
        self.w_q = torch.nn.Linear(dim_out, dim_out)
        self.w_k = torch.nn.Linear(dim_in + dim_time, dim_out)
        self.w_v = torch.nn.Linear(dim_in + dim_time, dim_out)
        self.att_act = torch.nn.LeakyReLU(0.2)
        self.layer_norm = torch.nn.LayerNorm(dim_out)
        self.mlp = torch.nn.Linear(dim_out, dim_out)
        self.dropout = torch.nn.Dropout(train_param['dropout'])
        self.att_dropout = torch.nn.Dropout(train_param['att_dropout'])
        self.last_updated_memory = None
        self.last_updated_ts = None
        self.last_updated_nid = None

    def forward(self, mfg):
        for b in mfg:
            Q = self.w_q(b.srcdata['mem']).reshape((b.num_src_nodes(), self.att_h, -1))
            mails = b.srcdata['mem_input'].reshape((b.num_src_nodes(), self.memory_param['mailbox_size'], -1))
            if self.dim_time > 0:
                time_feat = self.time_enc(b.srcdata['ts'][:, None] - b.srcdata['mail_ts']).reshape((b.num_src_nodes(), self.memory_param['mailbox_size'], -1))
                mails = torch.cat([mails, time_feat], dim=2)
            K = self.w_k(mails).reshape((b.num_src_nodes(), self.memory_param['mailbox_size'], self.att_h, -1))
            V = self.w_v(mails).reshape((b.num_src_nodes(), self.memory_param['mailbox_size'], self.att_h, -1))
            att = self.att_act((Q[:,None,:,:]*K).sum(dim=3))
            att = torch.nn.functional.softmax(att, dim=1)
            att = self.att_dropout(att)
            rst = (att[:,:,:,None]*V).sum(dim=1)
            rst = rst.reshape((rst.shape[0], -1))
            rst += b.srcdata['mem']
            rst = self.layer_norm(rst)
            rst = self.mlp(rst)
            rst = self.dropout(rst)
            rst = torch.nn.functional.relu(rst)
            b.srcdata['h'] = rst
            self.last_updated_memory = rst.detach().clone()
            self.last_updated_nid = b.srcdata['ID'].detach().clone()
            self.last_updated_ts = b.srcdata['ts'].detach().clone()

