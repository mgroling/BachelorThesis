from stable_baselines.deepq.policies import FeedForwardPolicy

class CustomDQNPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy, self).__init__(*args, **kwargs,
                                           layers=[1024, 512, 256],
                                           layer_norm=False,
                                           feature_extraction="mlp")