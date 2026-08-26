from ..base import DownstreamEvaluator
class NerLabEvaluator(DownstreamEvaluator):
    """Interface only: CPT checkpoints need a fixed supervised NER probe before inference."""
    def evaluate(self,model,tokenizer,output_dir,epoch,final=False):
        raise NotImplementedError("NER requires the future fixed fine-tuning/probe protocol; CPT weights alone have no NER head")
