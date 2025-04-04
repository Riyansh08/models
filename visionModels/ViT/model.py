import torch 
import math 
import concentration  # type: ignore
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProduct(nn.Module): 
    def __init__(self):
        super(ScaledDotProduct).__init__()
        self.softmax = nn.Softmax(-1)
        
    def forward(self ,  q , k , v , mask : bool , dropout : nn.Dropout):
        
        d_k = q.shape[-1]
        scaled = torch.matmul(q , k.transpose[-1 , -2] / math.sqrt(d_k))
        
        if mask is not None:
            scaled.masked_fill_(mask==0 , -1e9)
            
        attention = self.softmax(scaled)
        
        if dropout is not None:
            attention = dropout(attention)
            
        values = torch.matmul(attention , v )
        
        return values , attention  
    
## Now multi-head attention 

class MultiHeadAttention(nn.Module):
    def __init__(self , config):
        super().__init__()
        
        self.d_model = config["hidden_size"]
        self.heads = config["num_attention_heads"]
        
        assert self.d_Model % self.heads == 0, "Dimensions of model must be divisible by number of heads " 
        
        self.d_k = self.d_model // self.heads
        bias = config["bias"] 
        
        # Now we will initialize linear projections and weight matrices
        
        self.k_w = nn.Linear(self.d_model , self.d_Model , bias = True)
        self.v_w = nn.Linear(self.d_model , self.d_model , bias = True)
        self.q_w = nn.Linear(self.d_model , self.d_model , bias = True)
        self.o_w = nn.Linear(self.d_model , self.d_model , bias = True)
        
        # Now we will initialise the crux - Attention mechanism 
        
        self.attention = ScaledDotProduct()
        
        self.attention_dropout = nn.Dropout(config["attention_dropout"])
        self.output_dropout = nn.Dropout(config["output_dropout"])
        
        
        
    def forward(self , x  , output_attentions : True):
        
        q = k= v = x 
        mask = None 
        
        query = self.q_w(q)
        key = self.k_w(k)
        value = self.v_w(v)
        
        
        # Now reshape and transpose for multi-head attention 
        
        query.view(query.shape[0] , query.shape[1] , self.heads , self.d_k).transpose(1 , 2)
        
        key.view(query.shape[0] , query.shape[1] , self.heads , self.d_k).transpose(1 , 2)
        
        value.view(query.shape[0] , query.shape[1] , self.heads , self.d_k).transpose(1 , 2)
        
        
        x , attention_scores = self.attention(query , key , value , mask , self.attention_dropout)
        
        #now shape is - 32, 12, 197, 64) 
        #we need to reshape it to 32 , 197 , 768
        
        x = x.transpose(1 , 2).contiguous().view(x.shape[0] , -1 , self.heads * self.d_k)
        
        output = self.o_w(x)
        
        output = self.output_dropout(output)
        
        return output , attention_scores if output_attentions else None


#we have implemented the crux of the model - MUlti head attention 

#Now we will implement Layer Norm 

#mean - 0 and variance - 1 over the last dimension 

class LayerNormalization(nn.Module):
    def __init__(self , parameters_shape , eps = 1e-5 ):
        super().__init__()
        self.parameters_shape = parameters_shape
        self.eps = eps 
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        
        self.beta = nn.Parameter(torch.zeros(parameters_shape))
        
        
    def forward(self, inputs):
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim = dims , keepdim = True) 
        variance = ((inputs - mean) ** 2).mean(dim = dims , keepdim = True)
        std = torch.sqrt(variance + self.eps)
        normalised = (input - mean) /std 
        output = self.gamma * normalised + self.beta
        return output 
    
    
#Now we will implement the the activation function - GELU
# see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
        
   
class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        # Using the approximation for GELU
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    

#Now MLP 

class MLP(nn.Module):
    def __init__(self , config):
        super(MLP , self ).__init__()
        self.fc1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.activation = GELU() # Using GELU activation function
        self.fc2 = nn.Linear(config["intermediate_size"], config["hidden_size"])
        self.dropout = nn.Dropout(config["hidden_dropout"])
        
        
    def forward(self , x):
        x = self.fc1(x)
        x = self.activation(x)
       
        x = self.fc2(x)
        x = self.dropout(x)
        return x 
        
        
# BLOCK       

class TransformerBlock(nn.Module):
    
   def __init__(self , config):
      super(TransformerBlock, self).__init__()
      self.attention = MultiHeadAttention(config)
      self.layer_norm1 = LayerNormalization(config["hidden_size"])
      self.mlp = MLP(config)
      self.layer_norm2 = LayerNormalization(config["hidden_size"])
    
    
   def forward(self , x, output_attention : False):
        
      output_attention , outputprobs = self.attention(self.layer_norm1(x), output_attention = output_attention)
      
      x = output_attention + x # residual connection
      
      mlp_out = self.mlp(self.layer_norm2(x))
      
      x = mlp_out + x
      
      if not output_attention:
          return x , None
      else: 
          return x , outputprobs
      
#Encoder Blocks 

class Encoder:
    
    def __init__(self , config):
        super(Encoder , self).__init__()
        self.blocks = nn.ModuleList([])
        
        for _ in range(config["num_hidden_layers"]):
            block = TransformerBlock(config)
            
            self.blocks.append(block)
            
    def forward(self , x , output_attention = False):
        
        attentions = []
        for block in self.blocks:
           
            x, attention_probs = block(x, output_attention=True)
            if output_attention:
                attentions.append(attention_probs)
            
            if not output_attention:
                return (x , None)
            else:
                return (x , attentions)

# Patch Embeddings 

class PatchEmbeddings(nn.Module):
    
    def __init__(self  , config):
        
        super(PatchEmbeddings, self)._init__()
        self.image_size = config["image_size"] #224
        self.patch_size = config["patch_size"] # 16
        self.num_channels = config["num_channels"] # 3
        self.hidden_size = config["hidden_size"] # 768 
        
        assert self.image_size % self.patch_size == 0, "Image size must be divisible by patch size."
        
        self.num_patches = (self.image_size // self.patch_size) **2
        
        self.projection = nn.Conv2d(self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)


    def forward(self , x ):
        x = self.projection(x) 
        x.flatten(2).transpose(1 , 2)
        return x 
    
    
class Embeddings(nn.Module):
    
    
    def __init__(self , config):
        super(Embeddings , self).__init__()
        #224 x 224 x 3 is converted to batch , num_patches, hidden_size
        self.patch_embedding = PatchEmbeddings(config)
        
        self.cls_token = nn.Parameter(torch.randn(1 , 1 , config["hidden_size"]))
        self.position_embeddings = nn.Parameter(torch.randn(1, config["num_patches"] + 1, config["hidden_size"]))  # +1 for CLS token
        
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])
        
    def forward(self, x):
        x = self.patch_embeddings(x)
        batch_size, _, _ = x.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x

class ViT(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config["image_size"]
        self.hidden_size = config["hidden_size"]
        self.num_classes = config["num_classes"]
       
        self.embedding = Embeddings(config)
       
        self.encoder = Encoder(config)
        
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        
        self.apply(self._init_weights)

    def forward(self, x, output_attentions=False):
       
        embedding_output = self.embedding(x)
      
        encoder_output, all_attentions = self.encoder(embedding_output, output_attentions=output_attentions)
        logits = self.classifier(encoder_output[:, 0, :])
       
        if not output_attentions:
            return (logits, None)
        else:
            return (logits, all_attentions)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config["initializer_range"])
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Embeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.cls_token.dtype)
