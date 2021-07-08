
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
import time
from collections import OrderedDict


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0,include_input = True):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : include_input,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

# Positional encoding
class Trigonometric_kernel:
    def __init__(self, L = 10, include_input=True):

        self.L = L
 
        self.embed_fn, self.out_ch= get_embedder(L,include_input = include_input)

    '''
    INPUT
     x: input vectors (N,C) 

     OUTPUT

     pos_kernel: (N, calc_dim(C) )
    '''
    def __call__(self, x):
        return self.embed_fn(x)

    def calc_dim(self, dims=0):
        return self.out_ch


 



class SpaceNet(nn.Module):


    def __init__(self,  use_viewdirs=False, include_input = True):
        super(SpaceNet, self).__init__()

        print('SpaceNet enabled.')

        self.tri_kernel_pos = Trigonometric_kernel(L=10,include_input = include_input)
        self.tri_kernel_dir = Trigonometric_kernel(L=4, include_input = include_input)

        self.c_pos = 3
        self.use_viewdirs = use_viewdirs

        self.pos_dim = self.tri_kernel_pos.calc_dim(3)
        self.dir_dim = self.tri_kernel_dir.calc_dim(3)

        backbone_dim = 256
        head_dim = 128

        self.placeholder_quant = nn.Identity()
        self.placeholder_quant_post = nn.Identity(name='post')


        l = OrderedDict()
        for i in range(4):
            if i ==0:
                l['%d_stage1'%i]=nn.Linear(self.pos_dim, backbone_dim)
            else:
                l['%d_stage1'%i]=(nn.Linear(backbone_dim, backbone_dim))
            l['%d_stage1_relu'%i]=(nn.ReLU(inplace=True))
    
        self.stage1 = nn.Sequential(l)

        l = OrderedDict()
        
        for i in range(3):
            if i ==0:
                l['%d_stage2'%i]=(nn.Linear(backbone_dim+self.pos_dim, backbone_dim))
            else:
                l['%d_stage2_relu'%i]=(nn.ReLU(inplace=True))
                l['%d_stage2'%i]=(nn.Linear(backbone_dim, backbone_dim))
           
        self.stage2 = nn.Sequential(l)


        l = OrderedDict()
        pres = backbone_dim
        for i in range(1):
            l['%d_density_relu'%i]=(nn.ReLU(inplace=True))
            if i ==1-1:
                l['%d_density'%i]=(nn.Linear(pres, 1))
            else:
                l['%d_density'%i]=(nn.Linear(pres, head_dim))
                pres = head_dim
           
        self.density_net = nn.Sequential(l)


        l = OrderedDict()
        pres = backbone_dim+self.dir_dim
        for i in range(2):
            l['%d_rgb_relu'%i]=(nn.ReLU(inplace=True))
            if i ==2-1:
                l['%d_rgb'%i]=(nn.Linear(pres, 3))
            else:
                l['%d_rgb'%i]=(nn.Linear(pres, head_dim))
                pres = head_dim





        self.rgb_net = nn.Sequential(l)


    '''
    INPUT
    pos: 3D positions (N,L,c_pos) or (N,c_pos)
    rays: corresponding rays  (N,6)

    OUTPUT

    rgb: color (N,L,3) or (N,3)
    density: (N,L,1) or (N,1)

    '''
    def forward(self, inputs):

        pos, dirs = torch.split(inputs, [3,3], dim=-1)


   
        pos = self.placeholder_quant(pos)

        pos = self.tri_kernel_pos(pos)
        
        pos = self.placeholder_quant_post(pos)

        dirs = self.placeholder_quant(dirs)
        dirs = self.tri_kernel_dir(dirs)
        dirs = self.placeholder_quant_post(dirs)

        #torch.cuda.synchronize()
        #print('transform :',time.time()-beg)

        #beg = time.time()
        #print(pos.size())
        x = self.stage1(pos)
        x = self.stage2(torch.cat([x,pos],dim =1))


        density = self.density_net(x)


        if self.use_viewdirs:
            rgbs = self.rgb_net(torch.cat([x,dirs],dim =1))
            outputs = torch.cat([rgbs, density], -1)
        else:
            outputs = density

        #torch.cuda.synchronize()
        #print('fc:',time.time()-beg)

        return outputs



