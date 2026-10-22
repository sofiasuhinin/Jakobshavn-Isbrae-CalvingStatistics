import numpy as np
from dataclasses import dataclass


def generate_path(generator, final_time):
    xs = [0.0]
    ts = [0.0]

    while ts[-1] < final_time:
        dt, dx = generator()
        ts.append(ts[-1] + dt)
        xs.append(xs[-1] - dx)

    return np.array(ts), np.array(xs)


@dataclass
class CompoundGammaGenerator:
    rng: np.random.Generator
    time_scale: float
    size_scale: float
    time_shape: float = 1.0
    size_shape: float = 1.0

    def __call__(self):
        dt = self.rng.gamma(scale=self.time_scale, shape=self.time_shape)
        dx = self.rng.gamma(scale=self.size_scale, shape=self.size_shape)
        return dt, dx


@dataclass
class SumGenerator:
    rng: np.random.Generator
    generators: object
    dts: np.ndarray = None
    dxs: np.ndarray = None

    def __post_init__(self):
        events = np.array([generator() for generator in self.generators])
        self.dts = events[:, 0]
        self.dxs = events[:, 1]

    def __call__(self):
        index = np.argmin(self.dts)
        dt, dx = self.dts[index], self.dxs[index]
        self.dts -= dt
        new_dt, new_dx = self.generators[index]()
        self.dts[index] = new_dt
        self.dxs[index] = new_dx

        return dt, dx
