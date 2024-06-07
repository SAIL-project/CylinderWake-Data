import psutil
import rootutils
import time
from tqdm import tqdm
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import hydrogym.firedrake as hgym
from firedrake import Interpolate, assemble, inner, sqrt

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)
from cylinderdata.utils import H5DatasetCallback


def compute_fields(flow: hgym.RotaryCylinder):
    velocity = flow.u
    velocity_x = velocity.sub(0)
    velocity_y = velocity.sub(1)
    pressure = flow.p
    vorticity = flow.vorticity()
    magnitude = assemble(Interpolate(sqrt(inner(velocity, velocity)), flow.pressure_space))
    return [velocity_x, velocity_y, pressure, vorticity, magnitude]


def log(flow: hgym.RotaryCylinder):
    mem_usage = psutil.virtual_memory().percent
    CL, CD = flow.get_observations()
    return CL, CD, mem_usage


def generate_cylinder(sim: DictConfig, interval: int, seed: int, num: int = 0):
    # Define system
    flow = hgym.RotaryCylinder(
        Re=sim.re,
        mesh=sim.mesh,
        velocity_order=sim.velocity_order,
    )

    # Callbacks
    print_fmt = "t: {0:0.2f},\t\t CL: {1:0.3f},\t\t CD: {2:0.03f}\t\t Mem: {3:0.1f}"
    steps = round(sim.episode_length / (interval * sim.dt))
    callbacks = [
        hgym.utils.io.LogCallback(postprocess=log, nvals=3, print_fmt=print_fmt, interval=20),
        H5DatasetCallback(
            filename=f"../Cylinder-Dataset/cylinder{seed}.h5",
            t_start=sim.cook_length,
            flow=flow,
            fields=compute_fields,
            steps=steps,
            grid_N=(128, 512),
            grid_domain=((-2, 2), (-2, 14)),
            interval=interval,
        ),
    ]

    # Run simulation
    hgym.integrate(
        flow,
        t_span=(0, sim.episode_length + sim.cook_length),
        dt=sim.dt,
        callbacks=callbacks,
        stabilization=sim.stabilization,
    )


@hydra.main(version_base=None, config_path="config", config_name="generate")
def main(cfg: DictConfig) -> None:
    num = HydraConfig.get().job.num
    time.sleep(num / 10)
    pbar = tqdm(total=cfg.count, desc=f"Generating Dataset {num}", position=2 * num, leave=False)
    for i in range(cfg.count):
        generate_cylinder(cfg.sim, cfg.interval, cfg.base_seed + i, num=num)
        pbar.update(1)


if __name__ == "__main__":
    main()
