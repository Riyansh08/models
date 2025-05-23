import torch 
import torch.nn as nn
import math 
import torch.nn.functional as F
import numpy as np 
import pandas as pd

# to be implemented: CLIP model 

class SingleAttentionHead(nn.Module):
     
    def __init__(self , dimension , attention_head_size , dropout , bias = False):
        super(SingleAttentionHead, self).__init__()
        
        self.dimension = dimension 
        self.attention_head_size = attention_head_size
        self.dropout = dropout 
        
        self.query = nn.Linear(dimension , attention_head_size , bias = bias)
        self.key = nn.Linear(dimension, attention_head_size, bias=bias)
        self.value = nn.Linear(dimension, attention_head_size, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self , x ):
        # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, attention_head_size)
      query = self.query(x)
      key = self.key(x)
      value = self.value(x)
        # Calculate the attention scores
        # softmax(Q*K.T/sqrt(head_size))*V
      attention_scores = torch.matmul(query, key.transpose(-1, -2))
      attention_scores_norm = attention_scores / math.sqrt(self.attention_head_size)
      attention_weights = F.softmax(attention_scores_norm , dim = -1)
      
      attention_weights = self.dropout(attention_weights)
      attention_output = torch.matmul(attention_weights , value)
      
      return ( attention_output , attention_weights)
  
class ScaledDotProduct(nn.Module):
    
    def __init__(self):
        super(ScaledDotProduct , self).__init__()
        
        self.softmax = nn.Softmax(dim = -1) # softmax 
        
    def forward(self , k , q , v , mask ,  dropout :nn.Dropout):
        
        d_k = q.size(-1) # Get the dimension of the query 
        scores = torch.matmul(q , k.transpose(-1 , -2) ) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = self.softmax(scores)
        if dropout:
         attention_weights = dropout(attention_weights)
        attention_output  =  torch.matmul(attention_weights , v)
        
        return ( attention_output , attention_weights ) 
    
    
class MultiHeadAttention(nn.Module):
    
    def __init__(self , dimension , num_heads , dropout : 0.0 , bias = False):
        super(MultiHeadAttention , self ).__init__()
        
        self.dimension= dimension 
        self.num_heads = num_heads
        assert dimension % num_heads == 0, "dimension must be divisible by num_heads"
        
        self.head_dim = dimension // num_heads
        self.dropout = nn.Dropout(dropout)
        
        # Initialize projection layers
        
        self.w_q = nn.Linear(dimension, dimension, bias=bias)
        self.w_k = nn.Linear(dimension, dimension, bias=bias)
        self.w_v = nn.Linear(dimension, dimension, bias=bias)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.attention = ScaledDotProduct()
        
    # Forward pass
    def forward(self , x , mask : None , output_attentions : False):
        q = v = k = x
        
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.q_v(v)
         
        # Reshape the input for multi-head attention
        key = key.view(key.size(0), key.size(1), self.num_heads, self.head_dim).transpose(1 , 2)
        query = query.view(query.size(0), query.size(1), self.num_heads, self.head_dim).transpose(1 , 2)
        value = value.view(value.shape[0], value.shape[1], self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        x , attention_weights = self.attention(key , query , value , mask , self.dropout)
        # Reshape the output back to the original shape
        x = x.transpose(1 , 2).contigious().view(x.size(0), -1, self.head_dim * self.num_heads)
        
        x = self.w_o(x)
        x = self.dropout(x)
        if output_attentions:
            return x , attention_weights
        else:
         return x 
     
#positional encoding - CLIP uses sinusoidal positional encoding 

class PostionalEncoding(nn.Module):
  def __init__(self, d_model, max_seq_length):
    super().__init__()
    self.d_model = d_model
    self.max_seq_length = max_seq_length

    pe = torch.zeros(max_seq_length, d_model)
    position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0)/d_model))

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    self.register_buffer('pe', pe.unsqueeze(0))

  def forward(self, x):
    seq_len = x.size(1)
    return x + self.pe[:, :seq_len]
        
#Note :Original Paper uses Sinusoidal positional encoding      
class ROPE(nn.Module):
    def __init__(self, dim):
        super(ROPE, self).__init__()
        self.dim = dim
        assert dim % 2 == 0, "RoPE dimension must be even."

        # Precompute inverse frequency terms
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        # x shape: (batch, num_heads, seq_len, head_dim)
        seq_len = x.size(-2)
        t = torch.arange(seq_len, device=x.device).float()  
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  

        emb = torch.cat((freqs.sin(), freqs.cos()), dim=-1)  
        emb = emb[None, None, :, :] 
        x1, x2 = x[..., ::2], x[..., 1::2]  
        x_rotated = torch.cat([
            x1 * emb[..., :self.dim//2] - x2 * emb[..., self.dim//2:],  
            x1 * emb[..., self.dim//2:] + x2 * emb[..., :self.dim//2]  
        ], dim=-1)

        return x_rotated

        
class NewGELUActivation(nn.Module):
    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

class MLP(nn.Module):
    def __init__(self, dimension , dim_ratio = 4 , activation = NewGELUActivation):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(dimension , dimension * dim_ratio)
        self.activation = NewGELUActivation()
        self.fc2 = nn.Linear(dimension * dim_ratio , dimension)
        
    def forward(self , x ):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

class PatchEmbedding(nn.Module):
    
    def __init__(self , img_size , d_model , patch_size , num_channels):
        super(PatchEmbedding , self).__init__()
        self.image_size = img_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.hidden_size = d_model
        
        self.num_patches = (img_size[0] * img_size(1) ) // patch_size**2
  
        self.projection = nn.Conv2d(num_channels , d_model, kernel_size = self.patch_size , stride = self.patch_size)
        
    def forward(self  , x):
        x = self.projection(x)
        x = x.flatten(2).transpose(1 , 2)
        return x  
    
class Embeddings(nn.Module):
       
    # Combiing the patch embeddings with the class token and position embeddings.
    # NOTE - paper also had dropout, but not implemented here.
    
    def __init__(self , d_model , num_patches , num_channels, patch_size , img_size):
        super(Embeddings , self).__init__()
        
        self.patch_embeddings = PatchEmbedding(img_size , d_model , patch_size , num_channels )
        self.class_token = torch.nn.Parameter(torch.randn(1 , 1, d_model)) #normal distribution 
        self.positional_embedding = nn.Parameter(torch.randn(1 , num_patches + 1 , d_model)) #normal distribution
        
    def forward(self , x):
        x = self.patch_embeddings(x)
        batch_size , _ , _ = x.size()
        
        class_token = self.class_token.expand(batch_size  , -1 , -1)
        x = torch.cat((class_token , x) , dim = 1)
        x = x + self.positional_embedding 
        return x

class EncoderBlock(nn.Module):
     def __init__(self , d_model , num_heads):
         super(EncoderBlock , self).__init__()
         self.d_model = d_model 
         self.num_heads = num_heads
         self.attention = MultiHeadAttention(d_model , num_heads)
         self.layernorm_1 = nn.LayerNorm(d_model)
         self.mlp = MLP(d_model)
         self.layernorm_2 = nn.LayerNorm(d_model)
         
     def forward(self , x  , output_attentions = False ):
         attention_output , _ = self.attention(self.layernorm_1(x) , output_attentions = output_attentions)
         attention_output = attention_output + x
         output = self.mlp(self.layernorm_2(x))
        # Skip connection 
         x = x + output
         return x 
class Encoder(nn.Module):
    def __init__(self , d_model , num_heads , n_layers):
        super(Encoder , self).__init__()
        self.d_model = d_model 
        self.num_heads = num_heads
        self.n_blocks = nn.ModuleList([])
        for _ in range(n_layers):
            block = EncoderBlock(d_model , num_heads)
            
            self.n_blocks.append(block)
            
    def forward(self , x , output_attentions = False):
        for block in self.n_blocks:
            x = block(x , output_attentions = output_attentions)
        return x 
class ViT(nn.Module):
    
    def __init__(self , d_model , num_heads , img_size , patch_size , num_channels , n_heads , n_layers , out_dim):
        super(ViT , self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.embedding = Embeddings(d_model, img_size, patch_size, num_channels)

        self.encoder = Encoder(d_model, n_heads, n_layers)
        self.ln_final = nn.LayerNorm(d_model)
        self.output = nn.Parameter(torch.randn(self.hidden_size, self.emb_dim))
     
        
    def forward(self , x):
          x = self.embedding(x)
          x = self.encoder(x)
          x = self.ln_final(x)
          cls_token = x[:, 0]
          cls_token = cls_token @ self.output
          cls_token = cls_token/ torch.norm(cls_token, dim=-1, keepdim=True)
          
          return cls_token
# Tokenizer- to convert text to tokens.
#NOTE - This is charecter level tokenizer. NOT BPE OR SENTENCE PIECE USED IN GPTs 
   
def tokenizer(text , encode = True , mask = False , max_len = 512):
    
    # Implement the tokenizer for the text input
    if encode:
        out = chr(2) + text + chr(3)
        if len(out) > max_len:
            out = out[:max_len]
            out+=chr(3)
        out = out + "".join([chr(0) for _ in range(max_len - len(out))])
        out = torch.IntTensor(list(out.encode('utf-8')))
        mask = torch.ones(len(out.nonzero()))
        if len(mask) < max_len:
            mask = torch.cat((mask, torch.zeros(max_len - len(mask)))).type(torch.IntTensor)
        else:
            mask = mask.type(torch.IntTensor)
    else:
        # Decode the text
        out = [chr(x) for x in text[1:len(mask.nonzero()) - 1]]
        out = "".join(out)
        mask = None
        
    return out, mask
        
class TextEncoder(nn.Module):
    def __init__(self , vocab_size , num_heads , num_layers , d_model , max_len , emb_dim):
        super(TextEncoder , self).__init__()
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_model = d_model
        self.max_len = max_len
        self.emb_dim = emb_dim
        
        self.embedding = nn.Embedding(vocab_size , d_model)
        self.positional_embedding = PostionalEncoding(d_model , max_len)
        self.transformer_encoder = nn.ModuleList([EncoderBlock(d_model, num_heads) for _ in range(num_layers)])
    
    def forward(self, text, mask = None):

      x = self.embed(text)

      x = self.positional_embedding(x)

      for encoder_layer in self.transformer_encoder:
          x = encoder_layer(x)

      #The output of the encoder layers is the text features. We are going to be using the features from the EOT embedding.

      x = x[torch.arange(text.shape[0]), torch.sub(torch.sum(mask[:,0],dim=1),1)]

      if self.projection is not None:
         x = x @ self.projection

      x = x / torch.norm(x, dim=-1, keepdim = True)

      return x

class TextEncoder_Retrieval(nn.Module):
  def __init__(self, vocab_size, d_model, max_seq_length, n_layers,n_heads, emb_dim):
      super().__init__()

      self.max_seq_length = max_seq_length

      self.embed = nn.Embedding(vocab_size, d_model)

      self.positional_embedding = PostionalEncoding(d_model, max_seq_length)

      self.transformer_encoder = nn.ModuleList([EncoderBlock(d_model, n_heads) for _ in range(n_layers)])

      self.projection = nn.Parameter(torch.randn(d_model, emb_dim))


 # # For image retrieval
  def forward(self, text, mask=None):
        x = self.embed(text)
        x = self.positional_embedding(x)

        for encoder_layer in self.transformer_encoder:
            x = encoder_layer(x)

        if mask is not None:
            # Get the lengths of each sequence (i.e., find the last non-padded token)
            seq_lengths = mask.sum(dim=1) - 1  # Subtract 1 to get the index
            x = x[torch.arange(text.shape[0]), seq_lengths]
        else:
            x = x[:, -1]  # If no mask is provided, take the last token in the sequence.

        if self.projection is not None:
            x = x @ self.projection

        x = x / torch.norm(x, dim=-1, keepdim=True)

        return x

class CLIP(nn.Module):

    def __init__(self, emb_dim, vit_layers, vit_d_model, img_size, patch_size, n_channels, vit_heads, vocab_size, max_seq_length, text_heads, text_layers, text_d_model, retrieval = False):
        super().__init__()

        self.vision_encoder = ViT(vit_d_model, img_size, patch_size, n_channels, vit_heads, vit_layers, emb_dim) 

        self.text_encoder = TextEncoder(vocab_size, text_d_model, max_seq_length, text_layers, text_heads, emb_dim)

        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def CLIPLoss(self, logits, device = "cuda"):
       
   
        labels = torch.arange(logits.shape[0]).to(device)  

        loss_v = nn.functional.cross_entropy(logits.transpose(-2,-1), labels)

        loss_t = nn.functional.cross_entropy(logits, labels)
        loss = (loss_v + loss_t) / 2

        return loss

    def forward(self, image, text, mask=None):
      V_e = self.vision_encoder(image)  
      T_e = self.text_encoder(text, mask)

      logits = (V_e @ T_e.transpose(-2, -1)) * torch.exp(self.temperature)

      loss = self.CLIPLoss(logits, self.device)

      return loss

 