import torch
from utils.encoder import *
from utils.summary import *
class tubeletEmbedding(torch.nn.Module):

    def __init__(self,
                 input_dim,
                 embed_dim,
                 patch_size,
                 T,
                 H,
                 W):
        
        super().__init__()
        self.input_dim = input_dim # Input channel dimensions
        self.embed_dim = embed_dim # Processing dimensions
        self.patch_size = patch_size # Shape of the patches (p_t,p_h,p_w)
        self.T = T
        self.H = H
        self.W = W

        n_t = (((T - patch_size[0])//patch_size[0])+1)
        n_h = (((H - patch_size[1])//patch_size[1])+1)
        n_w = (((W - patch_size[2])//patch_size[2])+1)
        max_seq_len = n_t*n_h*n_w        

        self.emebdding_layer = torch.nn.Conv3d(in_channels=self.input_dim,
                                               out_channels=self.embed_dim,
                                               kernel_size=self.patch_size,
                                               stride=self.patch_size,
                                               )
        #torch.nn.init.xavier_uniform(self.emebdding_layer.weight)
        
        self.pos_embeddings = torch.nn.Parameter(torch.normal(mean=0.0,
                                                              std=1.0,
                                                              size=(max_seq_len,self.embed_dim)))
        
    def forward(self, x):

        """
        Tubelet embedding layer

        INPUTS:-
        1) x: Input video of shape [N,C,T,H,W]

        OUTPUTS:-
        1) x: Tokenized input of shape [N,T',embed_dim] with learnable positional encoding added
        """

        batch_size = x.size(0) # Current batch size
        x = self.emebdding_layer(x)
        x = x.view(batch_size,self.embed_dim,-1) 
        x = torch.add(x.permute((0,2,1)),self.pos_embeddings)
        return x  # Shape -> [B,T,d_model]

class conv3d(torch.nn.Module):

    def __init__(self,C_in,C_out,kernel_size):

        super().__init__() 
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size

        self.conv = torch.nn.Conv3d(self.C_in,
                                    self.C_out,
                                    self.kernel_size,
                                    padding='same')
        #torch.nn.init.xavier_uniform(self.conv.weight)

    def forward(self,x):
        x = self.conv(x)
        return torch.nn.functional.relu(x)
    
class posEncoding(torch.nn.Module):

    def __init__(self,patch_size,T,H,W,d_model):
        super().__init__()
        self.patch_size = patch_size
        self.T = T
        self.H = H
        self.W = W
        self.d_model = d_model

        n_t = (((T - patch_size[0])//patch_size[0])+1)
        n_h = (((H - patch_size[1])//patch_size[1])+1)
        n_w = (((W - patch_size[2])//patch_size[2])+1)
        self.max_seq_len = n_t*n_h*n_w

        self.embedding = torch.nn.Embedding(self.max_seq_len,
                                            self.d_model)
        
    def forward(self,x):

        """
        Positional encoding layer

        INPUTS:-
        1) x: Input embedding tokens of shape [B,T,d_model]

        OUTPUTS:-
        1) x: Output embedding tokens of shape [B,T,d_model]
        """
        positions = torch.range(0,self.max_seq_len-1,1)
        return x + self.embedding(positions)
class res3dViViT(torch.nn.Module):

    """
    Res3D-ViViT class
    """

    def __init__(self,
                 input_dim,
                 embed_dim,
                 patch_size,
                 T,
                 H,
                 W,
                 C,
                 d_model,
                 num_heads,
                 dff,
                 rate,
                 G,
                 I):
        
        super().__init__()
        self.input_dim = input_dim # Input channel dimensions
        self.embed_dim = embed_dim # Processing dimensions
        self.patch_size = patch_size # Shape of the patches (p_t,p_h,p_w)
        self.T = T # Input temporal dims
        self.H = H # Input height dims
        self.W = W # Input width dims
        self.C = C # Number of channels in the input
        self.d_model = d_model # Embedding dimensions
        self.num_heads = num_heads # Total number of heads
        self.dff = dff # Feed-forward dimension
        self.rate = rate # Dropout rate
        self.G = G # Number of gesture classes
        self.I = I # Number of identity classes

        n_t = (((T - patch_size[0])//patch_size[0])+1)
        n_h = (((H - patch_size[1])//patch_size[1])+1)
        n_w = (((W - patch_size[2])//patch_size[2])+1)
        self.max_seq_len = n_t*n_h*n_w

        ##### Convolutional layers
        self.conv11 = conv3d(self.C,16,(3,3,3))
        self.conv12 = conv3d(16,16,(3,3,3))
        self.conv13 = conv3d(16,16,(3,3,3))

        self.conv21 = conv3d(16,32,(3,3,3))
        self.conv22 = conv3d(32,32,(3,3,3))
        self.conv23 = conv3d(32,32,(3,3,3))

        ##### Tubelet embedding layer
        self.embedLayer = tubeletEmbedding(self.input_dim,
                                           self.embed_dim,
                                           self.patch_size,
                                           self.T,
                                           self.H,
                                           self.W)        
        
        self.encoder1 = encoder(self.embed_dim,
                                self.num_heads,
                                self.dff,
                                self.rate,
                                self.max_seq_len)
        
        self.encoder2 = encoder(self.embed_dim,
                                self.num_heads,
                                self.dff,
                                self.rate,
                                self.max_seq_len)
        
        ##### Output layers
        self.dense_op = torch.nn.Linear(self.embed_dim,32)

        self.dense_hgr = torch.nn.Linear(self.embed_dim,self.G)
        self.softmax_hgr = torch.nn.Softmax(dim=-1)

        self.dense_id = torch.nn.Linear(self.embed_dim,self.I)
        self.softmax_id = torch.nn.Softmax(dim=-1)
        
    def forward(self,x):

        """
        Res3D-ViViT Model

        INPUTS:-
        1) x: Input of shape [N,C,T,H,W]

        OUTPUTS:-
        1) f_hgr: Softmax outputs of HGR network of shape (G,)
        2) f_id: Softmax outputs of ID network of shape (I,)
        3) f_theta: Final embeddings of shape (d_model,)
        """

        ###### Res3D Backbone
        ##### Block-1
        x_res1 = self.conv11(x)
        x = self.conv12(x_res1)
        x = self.conv13(x)
        x = torch.add(x,x_res1)

        ##### Block-2
        x_res2 = self.conv21(x)
        x = self.conv22(x_res2)
        x = self.conv23(x)
        x = torch.add(x,x_res2)

        ###### Embedding layer
        x = self.embedLayer(x)

        ###### Transformer layer
        x = self.encoder1(x)
        x = self.encoder2(x)

        ###### Output
        f_theta = torch.nn.functional.relu(self.dense_op(torch.mean(x,dim=1,keepdim=False))) # embeddings
        dense_hgr = self.dense_hgr(f_theta)
        dense_id = self.dense_id(f_theta)

        return dense_hgr, dense_id, f_theta

###### Model summaries

if __name__ == "__main__":

    device = torch.device("cuda:0")
    ip_tensor = torch.normal(mean=0.0,
                             std=1.0,
                             size = (64,4,40,32,32)).to(device)
    
    def init_weights(m):
        if (isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv3d)):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
    model = res3dViViT(32,
                       32,
                       (5,5,5),
                       40,
                       32,
                       32,
                       4,
                       32,
                       16,
                       128,
                       0.3,
                       11,
                       10)

    total_params = sum(p.numel() for p in model.parameters())
    print('Total parameters: '+str(total_params))    

    print_model_summary(model,(4, 40, 32, 32))    
    #model.apply(init_weights)
    #summary(model, input_size=(4, 40, 32, 32))
    model = model.to(device)
    
    dense_hgr, dense_id, f_theta = model(ip_tensor)
    print(dense_hgr.size())
    print(dense_id.size())
    print(f_theta.size())
    
    #embedding_layer = tubeletEmbedding(3,32,(15,5,5),60,64,64).to(device)
    
    #conv_layer = conv3d(3,16,(3,3,3)).to(device)
    #op_conv = conv_layer(ip_tensor)
    #print(op_conv.size())
    
    #encoder_1 = torch.nn.TransformerEncoderLayer(32,
    #                                             8,
    #                                             128,
    #                                             batch_first=True).to(device)
    #encoder_2 = torch.nn.TransformerEncoderLayer(32,
    #                                             8,
    #                                             128,
    #                                             batch_first=True).to(device)
    
    #op_tensor = embedding_layer(ip_tensor)
    #op_tensor = encoder_1(op_tensor)
    #op_tensor = encoder_2(op_tensor)

    #print(op_tensor.size())
