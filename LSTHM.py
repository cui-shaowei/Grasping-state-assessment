import torch
import time
from torch import nn
import torch.nn.functional as F


class LSTHM(nn.Module):

	def __init__(self,cell_size,in_size,hybrid_in_size):
		super(LSTHM, self).__init__()
		self.cell_size=cell_size
		self.in_size=in_size
		self.W=nn.Linear(in_size,4*self.cell_size)
		self.U=nn.Linear(cell_size,4*self.cell_size)
		self.V=nn.Linear(hybrid_in_size,4*self.cell_size)

	def forward(self,x,ctm1,htm1,ztm1):
		input_affine=self.W(x)
		output_affine=self.U(htm1)
		hybrid_affine=self.V(ztm1)
		
		sums=input_affine+output_affine+hybrid_affine

		#biases are already part of W and U and V
		f_t=F.sigmoid(sums[:,:self.cell_size])
		i_t=F.sigmoid(sums[:,self.cell_size:2*self.cell_size])
		o_t=F.sigmoid(sums[:,2*self.cell_size:3*self.cell_size])
		ch_t=F.tanh(sums[:,3*self.cell_size:])
		c_t=f_t*ctm1+i_t*ch_t
		h_t=F.tanh(c_t)*o_t
		return c_t,h_t


class MAB(nn.Module):

	def __init__(self, dim_visual,dim_tactile,dim_reduce_visual,dim_reduce_tactile,hybird_dim, num_atts):
		super(MAB, self).__init__()
		self.dim_visual=dim_visual
		self.dim_tactile=dim_tactile
		self.dim_reduce_visual=dim_reduce_visual
		self.dim_reduce_tactile=dim_reduce_tactile
		self.num_atts=num_atts
		self.hybird_dim=hybird_dim
		self.dim_sum=self.dim_tactile+self.dim_visual
		self.attention_model = nn.Sequential(nn.Linear(self.dim_sum,self.dim_sum*self.num_atts))
		self.dim_reduce_nets_visual=nn.Sequential(nn.Linear(self.dim_visual*self.num_atts,self.dim_reduce_visual))
		self.dim_reduce_nets_tactile=nn.Sequential(nn.Linear(self.dim_tactile*self.num_atts,self.dim_reduce_tactile))
		self.dim_reduce_nets=[self.dim_reduce_nets_visual,self.dim_reduce_nets_tactile]
		self.g_net=nn.Linear(self.dim_reduce_tactile+self.dim_reduce_visual,self.hybird_dim)
	# def __call__(self, in_modalities):
	# 	return self.fusion(in_modalities)

	def forward(self, in_modalities):
		# getting some simple integers out
		num_modalities = len(in_modalities)
		# simply the tensor that goes into attention_model
		in_tensor = torch.cat(in_modalities, dim=1)
		# calculating attentions
		atts = F.softmax(self.attention_model(in_tensor), dim=1)
		# calculating the tensor that will be multiplied with the attention
		out_tensor = torch.cat([in_modalities[i].repeat(1, self.num_atts) for i in range(num_modalities)], dim=1)
		# calculating the attention
		att_out = atts * out_tensor

		# now to apply the dim_reduce networks
		# first back to however modalities were in the problem
		start = 0
		out_modalities = []
		for i in range(num_modalities):
			modality_length = in_modalities[i].shape[1] * self.num_atts
			out_modalities.append(att_out[:, start:start + modality_length])
			start = start + modality_length

		# apply the dim_reduce
		dim_reduced = [self.dim_reduce_nets[i](out_modalities[i]) for i in range(num_modalities)]
		# multiple attention done :)
		output_z=self.g_net(torch.cat((dim_reduced[0],dim_reduced[1]),dim=1))
		return output_z

	# def forward(self, x):
	# 	print("Not yet implemented for nn.Sequential")
	# 	exit(-1)