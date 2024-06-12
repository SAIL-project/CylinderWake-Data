import hydra
import rootutils
from omegaconf import DictConfig
import hydrogym.firedrake as hgym
from hydrogym.firedrake.utils.io import LogCallback
import matplotlib.pyplot as plt

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)
from cylinderdata.utils.callbacks import CylinderVisCallback


def log(flow: hgym.RotaryCylinder):
    CL, CD = flow.get_observations()
    return CL, CD


def run_cylinder(sim: DictConfig, control: DictConfig, interval: int):
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

    # Controller
    controller = hydra.utils.instantiate(control)

    # Run simulation
    print("Running simulation..")
    hgym.integrate(
        flow,
        t_span=(0, sim.episode_length),
        dt=sim.dt,
        callbacks=callbacks,
        stabilization=sim.stabilization,
        controller=controller.control,
    )

    # Plot
    print("Plotting..")
    fig, ax = plt.subplots()
    ax.plot(controller.time, controller.omega)
    ax.set(xlabel='time', ylabel='omega')
    ax.grid()
    fig.savefig("test.png")


@hydra.main(version_base=None, config_path="config", config_name="run")
def main(cfg: DictConfig) -> None:
    run_cylinder(cfg.sim, cfg.controller, cfg.interval)


if __name__ == "__main__":
    main()
