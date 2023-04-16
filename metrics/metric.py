class Metric:
    """Base class for metrics"""

    def __init__(self, name: str, score_function: callable):
        self.name = name
        self.score_function = score_function

    def __call__(self, human: str, story: str, **kwargs):
        return self.score_function(human, story, **kwargs)

    def __repr__(self):
        return self.name
