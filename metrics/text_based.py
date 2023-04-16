from metric import Metric

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.translate.chrf_score import sentence_chrf
import evaluate


class BLEUScore(Metric):
    def __init__(self):
        super(BLEUScore, self).__init__("BLEU", sentence_bleu)

    def __call__(self, reference: str, story: str, **kwargs):
        # TODO: consider using nltk.translate.bleu_score.SmoothingFunction()
        reference_tokens = [reference.split()]
        story_tokens = story.split()
        return self.score_function(reference_tokens, story_tokens, **kwargs)


class SacreBLEUScore(Metric):
    def __init__(self):
        sacrebleu = evaluate.load("sacrebleu")
        super(SacreBLEUScore, self).__init__("SacreBLEU", sacrebleu.compute)

    def __call__(self, reference: str, story: str, **kwargs):
        # TODO: Maybe consider precision output of compute function
        return self.score_function(
            references=[reference], predictions=[story], **kwargs
        )["score"]


class METEOScore(Metric):
    def __init__(self):
        super(METEOScore, self).__init__("METEOR", meteor_score)

    def __call__(self, reference: str, story: str, **kwargs):
        reference_tokens = [reference.split()]
        story_tokens = story.split()
        return self.score_function(reference_tokens, story_tokens, **kwargs)


class ChrFScore(Metric):
    def __init__(self):
        super(ChrFScore, self).__init__("ChrF", sentence_chrf)

    def __call__(self, reference: str, story: str, **kwargs):
        story_tokens = story.split()
        human_tokens = reference.split()
        return self.score_function(human_tokens, story_tokens, **kwargs)


## Tests
if __name__ == "__main__":
    import pandas as pd

    # input data
    df = pd.read_csv("data/hanna_stories_annotations.csv")
    df_gpt = df[df.Model == "GPT"]
    reference_story = df_gpt["Human"].iloc[1]
    story = df_gpt["Story"].iloc[1]
    # test blue score
    BLEUScore = BLEUScore()
    print("Bleu score for generated story ", BLEUScore(reference_story, story))
    # test meteor score
    METEOScore = METEOScore()
    print("Meteor score for generated story ", METEOScore(reference_story, story))
    # test chrF score
    ChrFScore = ChrFScore()
    print("ChrF score for generated story ", ChrFScore(reference_story, story))
    # test sacrebleu score
    SacreBLEUScore = SacreBLEUScore()
    print(
        "SacreBLEU score for generated story ", SacreBLEUScore(reference_story, story)
    )
