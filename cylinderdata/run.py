import hydra
import rootutils
from omegaconf import DictConfig
import hydrogym.firedrake as hgym
from hydrogym.firedrake.utils.io import LogCallback

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)
from cylinderdata.utils.callbacks import CylinderVisCallback


def log(flow: hgym.RotaryCylinder):
    CL, CD = flow.get_observations()
    return CL, CD


def run_cylinder(sim: DictConfig, interval: int):
    # Define system
    flow = hgym.RotaryCylinder(
        Re=sim.re,
        mesh=sim.mesh,
        velocity_order=sim.velocity_order,
    )

    # Callbacks
    print_fmt = "t: {0:0.2f},\t\t CL: {1:0.3f},\t\t CD: {2:0.03f}"
    callbacks = [
        LogCallback(postprocess=log, nvals=2, print_fmt=print_fmt, interval=interval),
        CylinderVisCallback(interval=interval),
    ]

    # Run simulation
    hgym.integrate(
        flow,
        t_span=(0, sim.episode_length),
        dt=sim.dt,
        callbacks=callbacks,
        stabilization=sim.stabilization,
    )


@hydra.main(version_base=None, config_path="config", config_name="run")
def main(cfg: DictConfig) -> None:
    run_cylinder(cfg.sim, cfg.interval)


if __name__ == "__main__":
    main()
