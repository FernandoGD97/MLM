def encode_texts(model,tokenizer,texts,device,batch_size=32,pooling="mean",normalize=True,max_length=128):
    import torch
    output=[]; backbone=getattr(model,"base_model",model); was_training=model.training; model.eval()
    with torch.inference_mode():
        for i in range(0,len(texts),batch_size):
            batch=tokenizer(texts[i:i+batch_size],padding=True,truncation=True,max_length=max_length,return_tensors="pt")
            batch={k:v.to(device) for k,v in batch.items()}; hidden=backbone(**batch).last_hidden_state
            if pooling.lower()=="cls": pooled=hidden[:,0]
            elif pooling=="mean":
                mask=batch["attention_mask"].unsqueeze(-1); pooled=(hidden*mask).sum(1)/mask.sum(1).clamp_min(1)
            else: raise ValueError(f"Unknown pooling: {pooling}")
            if normalize: pooled=torch.nn.functional.normalize(pooled,p=2,dim=-1)
            output.append(pooled.float().cpu())
    model.train(was_training)
    return torch.cat(output) if output else torch.empty((0,getattr(model.config,"hidden_size",0)))
