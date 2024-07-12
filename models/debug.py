if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    x_attention_mask = torch.full((3, 3), 0).bool().to("cuda")
    p_attention_mask = torch.full((4, 4), 0).bool().to("cuda")
    t_attention_mask = torch.full((1, 1), 0).bool().to("cuda")
    y_attention_mask = torch.triu(torch.ones(5, 5), diagonal=1).bool().to("cuda")
    x_lens = torch.LongTensor([3]).to("cuda")
    p_lens = torch.LongTensor([4]).to("cuda")
    t_lens = torch.LongTensor([1]).to("cuda")
    new_y_lens = torch.LongTensor([5]).to("cuda")
    print(x_attention_mask.shape, p_attention_mask.shape, t_attention_mask.shape, y_attention_mask.shape)
    
    x_attn_mask = F.pad(x_attention_mask, (0, p_lens.max() + t_lens.max()+new_y_lens.max()), value=True) 
    p_attn_mask = F.pad(p_attention_mask, (x_lens.max(), t_lens.max()+new_y_lens.max()), value=True)
    t_attn_mask = F.pad(t_attention_mask, (x_lens.max()+p_lens.max(), new_y_lens.max()), value=True)
    y_attn_mask = F.pad(y_attention_mask, (x_lens.max()+p_lens.max()+t_lens.max(), 0), value=False)

    
    print(x_attn_mask.shape, p_attn_mask.shape, t_attn_mask.shape, y_attn_mask.shape)
    # print(x_attn_mask)
    # print(p_attn_mask) 
    # print(t_attn_mask) 
    # print(y_attn_mask)
    xpty_attn_mask = torch.concat([x_attn_mask, p_attn_mask, t_attn_mask, y_attn_mask], dim=0)
    print(xpty_attn_mask)