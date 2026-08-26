import pytest
torch=pytest.importorskip("torch")
from encoder_pretraining.collators import CLMCollator

class Tok: pad_token_id=0
def test_clm_shift_and_padding():
    b=CLMCollator(Tok())([{"input_ids":[5,6,7]},{"input_ids":[8,9]}])
    assert b["labels"].tolist()==[[6,7,-100],[9,-100,-100]]
    assert b["attention_mask"].tolist()==[[1,1,1],[1,1,0]]

def test_verified_bert_has_no_future_leakage():
    tr=pytest.importorskip("transformers")
    from encoder_pretraining.objectives import set_attention_mode
    config=tr.BertConfig(vocab_size=30,hidden_size=16,num_hidden_layers=1,num_attention_heads=2,intermediate_size=32,hidden_dropout_prob=0,attention_probs_dropout_prob=0)
    model=tr.BertForMaskedLM(config).eval(); set_attention_mode(model,"clm")
    a=torch.tensor([[1,2,3,4]]); b=torch.tensor([[1,2,8,9]])
    with torch.no_grad(): la=model(a).logits; lb=model(b).logits
    assert torch.allclose(la[:,:2],lb[:,:2],atol=1e-6)
    set_attention_mode(model,"mlm")
    with torch.no_grad(): ba=model(a).logits; bb=model(b).logits
    assert not torch.allclose(ba[:,:2],bb[:,:2])
