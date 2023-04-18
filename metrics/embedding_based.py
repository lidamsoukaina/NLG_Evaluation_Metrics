import moverscore

# BERTScore
# MoverScore
class MoverScore(Metric):
    def __init__(self, model_name_or_path, tokenizer=None, device=None):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.tokenizer = tokenizer
        self.device = device

    def compute(self, predictions, references):
        return moverscore.score(
            predictions,
            references,
            model_name_or_path=self.model_name_or_path,
            tokenizer=self.tokenizer,
            device=self.device,
        )


# DepthScore
# BaryScore
# BARTScore
# InfoLM
