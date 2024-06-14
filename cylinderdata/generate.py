import rootutils
import hydra
from omegaconf import DictConfig
import hydrogym.firedrake as hgym
from firedrake import Interpolate, assemble, inner, sqrt

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)
from cylinderdata.utils import H5DatasetCallback, LogControlCallback, LogObservationCallback


def compute_fields(flow: hgym.RotaryCylinder):
    velocity = flow.u
    velocity_x = velocity.sub(0)
    velocity_y = velocity.sub(1)
    pressure = flow.p
    vorticity = flow.vorticity()
    magnitude = assemble(Interpolate(sqrt(inner(velocity, velocity)), flow.pressure_space))
    return [velocity_x, velocity_y, pressure, vorticity, magnitude]


def generate_cylinder(cfg: DictConfig):
    # Define system
    sim = cfg.sim
    flow = hgym.RotaryCylinder(
        Re=sim.re,
        mesh=sim.mesh,
        velocity_order=sim.velocity_order,
    )

    # Controller
    controller = hydra.utils.instantiate(
        cfg.control, max_control=flow.MAX_CONTROL, control_duration=cfg.control_duration
    )

    # Callbacks
    steps = round(sim.episode_length / (cfg.interval * sim.dt))
    callbacks = [
        LogObservationCallback(interval=cfg.interval, tf=sim.episode_length),
        LogControlCallback(interval=cfg.interval),
        H5DatasetCallback(
            filename="../Cylinder-Dataset/cylinder.h5",
            t_start=sim.cook_length,
            flow=flow,
            fields=compute_fields,
            steps=steps,
            grid_N=(128, 512),
            grid_domain=((-2, 2), (-2, 14)),
            interval=cfg.interval,
        ),
    ]

    # Run simulation
    hgym.integrate(
        flow,
        t_span=(0, sim.episode_length + sim.cook_length),
        dt=sim.dt,
        callbacks=callbacks,
        controller=controller,
        stabilization=sim.stabilization,
    )


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    generate_cylinder(cfg.sim)


if __name__ == "__main__":
    main()
