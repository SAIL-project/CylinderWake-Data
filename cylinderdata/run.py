import hydra
import rootutils
from omegaconf import DictConfig
import hydrogym.firedrake as hgym
from hydrogym.firedrake.utils.io import CheckpointCallback

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)
from cylinderdata.utils.callbacks import (
    CylinderVisCallback,
    LogControlCallback,
    LogObservationCallback,
)


def run_cylinder(cfg: DictConfig):

    # Define system
    sim = cfg.sim
    if sim.checkpoint != "":
        flow = hgym.RotaryCylinder(
            Re=sim.re,
            mesh=sim.mesh,
            velocity_order=sim.velocity_order,
            restart=sim.checkpoint,
        )
    else:
        flow = hgym.RotaryCylinder(
            Re=sim.re,
            mesh=sim.mesh,
            velocity_order=sim.velocity_order,
        )

    # Callbacks
    callbacks = [
        LogObservationCallback(interval=cfg.interval, tf=sim.episode_length),
        LogControlCallback(interval=1),
        CheckpointCallback(interval=100, filename="checkpoint.h5"),
    ]
    if cfg.show:
        callbacks.append(CylinderVisCallback(interval=cfg.interval))

    # Controller
    controller = hydra.utils.instantiate(
        cfg.controller,
        max_control=flow.MAX_CONTROL,
        control_duration=cfg.control_duration,
        start_time=cfg.control_start,
    )

    # Run simulation
    hgym.integrate(
        flow,
        t_span=(0, sim.episode_length),
        dt=sim.dt,
        callbacks=callbacks,
        stabilization=sim.stabilization,
        controller=controller,
    )


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    run_cylinder(cfg)


if __name__ == "__main__":
    main()
