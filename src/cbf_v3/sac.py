"""Compact, auditable PyTorch SAC with twin Q, target Q, entropy tuning."""
from __future__ import annotations
import copy, math
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

LOG_STD_MIN=-5.; LOG_STD_MAX=2.

def mlp(in_dim,out_dim,hidden=256):
    return nn.Sequential(nn.Linear(in_dim,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,out_dim))

class Actor(nn.Module):
    def __init__(self,obs_dim,act_dim,hidden=256):
        super().__init__();self.net=mlp(obs_dim,2*act_dim,hidden);self.act_dim=act_dim
    def forward(self,obs):
        z=self.net(obs);mu,raw_std=z.chunk(2,dim=-1)
        logstd=torch.tanh(raw_std)
        logstd=LOG_STD_MIN+.5*(LOG_STD_MAX-LOG_STD_MIN)*(logstd+1.)
        return mu,logstd
    def sample(self,obs,deterministic=False):
        mu,logstd=self.forward(obs)
        if deterministic:return torch.tanh(mu),None
        dist=Normal(mu,logstd.exp());z=dist.rsample();act=torch.tanh(z)
        logprob=dist.log_prob(z)-torch.log(torch.clamp(1.-act.square(),min=1e-6))
        return act,logprob.sum(-1,keepdim=True)

class Replay:
    def __init__(self,capacity,obs_dim,act_dim):
        self.capacity=int(capacity);self.n=0;self.ptr=0
        self.obs=np.zeros((capacity,obs_dim),np.float32)
        self.next=np.zeros((capacity,obs_dim),np.float32)
        self.act=np.zeros((capacity,act_dim),np.float32)
        self.rew=np.zeros((capacity,1),np.float32)
        self.terminal=np.zeros((capacity,1),np.float32)
    def add_batch(self,obs,act,rew,nxt,terminated):
        count=len(obs);inds=np.mod(np.arange(self.ptr,self.ptr+count),self.capacity)
        self.obs[inds]=obs;self.act[inds]=act;self.rew[inds,0]=rew
        self.next[inds]=nxt;self.terminal[inds,0]=terminated
        self.ptr=int((self.ptr+count)%self.capacity);self.n=min(self.capacity,self.n+count)
    def sample(self,n,device,rng):
        ix=rng.integers(0,self.n,size=n)
        return tuple(torch.as_tensor(arr[ix],device=device) for arr in
                     (self.obs,self.act,self.rew,self.next,self.terminal))

class SAC:
    def __init__(self,obs_dim,act_dim,device='cpu',hidden=256,lr=3e-4,gamma=.99,tau=.005):
        if act_dim<1:raise ValueError('SAC requires actions; use fixed baseline directly')
        self.device=torch.device(device);self.gamma=gamma;self.tau=tau
        self.actor=Actor(obs_dim,act_dim,hidden).to(device)
        self.q1=mlp(obs_dim+act_dim,1,hidden).to(device)
        self.q2=mlp(obs_dim+act_dim,1,hidden).to(device)
        self.tq1=copy.deepcopy(self.q1);self.tq2=copy.deepcopy(self.q2)
        self.log_ent=torch.nn.Parameter(torch.tensor(0.,device=device))
        self.opt_actor=torch.optim.Adam(self.actor.parameters(),lr=lr)
        self.opt_q=torch.optim.Adam(list(self.q1.parameters())+list(self.q2.parameters()),lr=lr)
        self.opt_ent=torch.optim.Adam([self.log_ent],lr=lr)
        self.target_entropy=-float(act_dim)
        self.updates=0
    @torch.no_grad()
    def act(self,obs,deterministic=False):
        ob=torch.as_tensor(np.asarray(obs,dtype=np.float32),device=self.device)
        if ob.ndim==1:ob=ob[None,:]
        out,_=self.actor.sample(ob,deterministic)
        return out.cpu().numpy()
    def update(self,batch):
        ob,act,rew,nxt,term=batch
        with torch.no_grad():
            an,ln=self.actor.sample(nxt)
            qn=torch.minimum(self.tq1(torch.cat((nxt,an),-1)),self.tq2(torch.cat((nxt,an),-1)))
            target=rew+self.gamma*(1.-term)*(qn-self.log_ent.exp().detach()*ln)
        qa=self.q1(torch.cat((ob,act),-1));qb=self.q2(torch.cat((ob,act),-1))
        loss_q=((qa-target)**2+(qb-target)**2).mean()
        self.opt_q.zero_grad(set_to_none=True);loss_q.backward()
        self.opt_q.step()
        pi,logpi=self.actor.sample(ob)
        qi=torch.minimum(self.q1(torch.cat((ob,pi),-1)),self.q2(torch.cat((ob,pi),-1)))
        loss_pi=(self.log_ent.exp().detach()*logpi-qi).mean()
        self.opt_actor.zero_grad(set_to_none=True);loss_pi.backward()
        self.opt_actor.step()
        loss_ent=-(self.log_ent*(logpi.detach()+self.target_entropy)).mean()
        self.opt_ent.zero_grad(set_to_none=True);loss_ent.backward();self.opt_ent.step()
        with torch.no_grad():
            for target_net,online in ((self.tq1,self.q1),(self.tq2,self.q2)):
                for tp,p in zip(target_net.parameters(),online.parameters()):tp.lerp_(p,self.tau)
        self.updates+=1
        return {'q':float(loss_q.detach()),'pi':float(loss_pi.detach()),
                'ent':float(self.log_ent.exp().detach())}
    def checkpoint(self,meta=None):
        return {'actor':self.actor.state_dict(),'q1':self.q1.state_dict(),'q2':self.q2.state_dict(),
            'tq1':self.tq1.state_dict(),'tq2':self.tq2.state_dict(),'log_ent':self.log_ent.detach().cpu(),
            'opt_actor':self.opt_actor.state_dict(),'opt_q':self.opt_q.state_dict(),
            'opt_ent':self.opt_ent.state_dict(),'updates':self.updates,'meta':meta or {}}
    def restore(self,path,weights_only=False):
        # Explicit weights_only=False only for files created and trusted by THIS project.
        obj=torch.load(path,map_location=self.device,weights_only=False)
        for name in ('actor','q1','q2','tq1','tq2'):
            getattr(self,name).load_state_dict(obj[name])
        if not weights_only:
            self.log_ent.data.copy_(obj['log_ent'].to(self.device))
            self.opt_actor.load_state_dict(obj['opt_actor']);self.opt_q.load_state_dict(obj['opt_q'])
            self.opt_ent.load_state_dict(obj['opt_ent']);self.updates=int(obj.get('updates',0))
        return obj.get('meta',{})
