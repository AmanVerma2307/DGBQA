import torch
from summary import print_model_summary

class MHSA(torch.nn.Module):
    
    """
    Multi head self-attention
    """

    def __init__(self,d_in,d_model,num_heads):

        super().__init__()
        self.d_in = d_in # input dimensions
        self.d_model = d_model # model dimensions
        self.num_heads = num_heads # number of heads

        assert d_model % num_heads == 0

        self.depth = self.d_model // num_heads

        self.query_dense = torch.nn.Linear(self.d_in,
                                           self.d_model)
        self.key_dense = torch.nn.Linear(self.d_in,
                                         self.d_model)
        self.value_dense = torch.nn.Linear(self.d_in,
                                           self.d_model)
        
        self.dense = torch.nn.Linear(self.d_model,
                                     self.d_model)
        
    def split_heads(self, inputs):
        batch_size = inputs.size(0) # Batch size
        inputs = inputs.view(batch_size,-1,self.num_heads,self.depth)
        return inputs.permute((0,2,1,3)) # shape -> [B,H,T,D]
    
    def scaled_dot_product_attention(self, q, k, v):
        
        B = q.size(0) # Batch size: B
        N = q.size(2) # N: max_seq_len

        attn = torch.bmm(q,k.permute((0,1,3,2)))/torch.sqrt(torch.Tensor(self.depth)) # attn -> [B,H,N,N]
        attn = torch.nn.functional.softmax(attn,dim=-1) # attn -> [B,H,N,N]
        output = torch.matmul(attn,v) # [B,H,N,depth]

        #output = output.permute((0,2,1,3)) # Reshape: [B,H,N,depth] -> [B,N,H,depth]
        #output = output.view(B,N,-1) # Reshape -> [B,N,d_model]
        
        return output
    
    def forward(self, x):

        """
        Multi head self-attention

        INPUTS:-
        1) x: Input tokens of shape [B,N,d_model]

        OUTPUTS:-
        1) x: Output tokens of shape [B,B,d_model]
        """

        B = x.size(0) # Batch size
        N = x.size(1) # max_seq_len

        q = self.split_heads(self.query_dense(x)) # Query -> [B,N,H,depth]
        k = self.split_heads(self.key_dense(x)) # Key -> [B,N,H,depth]
        v = self.split_heads(self.value_dense(x)) # Value -> [B,N,H,depth]

        x = self.scaled_dot_product_attention(q,k,v) # x -> [B,N,H,depth]

        x = x.permute((0,2,1,3)) # Reshape: [B,H,N,depth] -> [B,N,H,depth]
        x = x.view(B,N,-1) # Reshape -> [B,N,d_model]

        return x
    
class encoder(torch.nn.Module):

    """
    Encoder module
    """

    def __init__(self,
                 d_model,
                 num_heads,
                 dff,
                 rate,
                 max_seq_len):

        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate
        self.max_seq_len = max_seq_len

        self.att = torch.nn.MultiheadAttention(self.d_model,
                                               self.num_heads,
                                               self.rate,
                                               kdim=self.d_model,
                                               vdim=self.d_model,
                                               batch_first=True)
        
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(self.d_model,self.dff),
            torch.nn.ReLU(),
            torch.nn.Linear(self.dff,self.d_model),
            torch.nn.ReLU(),
        )

        self.layernorm1 = torch.nn.LayerNorm([self.max_seq_len,self.d_model],eps=1e-6)
        self.layernorm2 = torch.nn.LayerNorm([self.max_seq_len,self.d_model],eps=1e-6)

        self.droput1 = torch.nn.Dropout(self.rate)
        self.droput2 = torch.nn.Dropout(self.rate)

    def forward(self,x):

        attn_output, _ = self.att(x,x,x) # MHSA
        attn_output = self.droput1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.droput2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

if __name__ == "__main__":

    ip = torch.normal(0,1,(10,288,32))
    ip_head = torch.normal(0,1,(10,288,32))

    mhsa = MHSA(32,
                32,
                16)

    #encoder_layer = encoder(32,
    #                        16,
    #                        128,
    #                        0.3,
    #                        288)
    
    print(mhsa(ip_head).size())
    print_model_summary(mhsa,
                        (288,16,2))

    total_params = sum(p.numel() for p in mhsa.parameters())
    print('Total parameters: '+str(total_params))

    




