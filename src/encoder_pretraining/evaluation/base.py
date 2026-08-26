from abc import ABC,abstractmethod
class DownstreamEvaluator(ABC):
    @abstractmethod
    def evaluate(self,model,tokenizer,output_dir,epoch,final=False): ...
