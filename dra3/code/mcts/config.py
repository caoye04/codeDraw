class MCTSConfig:
    def __init__(
        self, 
        C: float = 1.0, 
        n_search: int = 50,
        temperature: float = 1.0,
        with_noise: bool = False,
        dir_epsilon: float = 0.25,
        dir_alpha: float = 0.03,
        *args, **kwargs
    ):
        self.C = C
        self.n_search = n_search
        self.temperature = temperature
        self.with_noise = with_noise
        self.dir_epsilon = dir_epsilon
        self.dir_alpha = dir_alpha