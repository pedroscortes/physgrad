"""
Visualization components for physics simulations.

This module provides real-time visualization capabilities with interactive controls
and mathematical overlays for analyzing simulation behavior.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import dash
    from dash import dcc, html, Input, Output, State
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False

from .core import Simulation, Particle, RigidBody


class Visualizer(ABC):
    """Abstract base class for physics simulation visualizers."""

    def __init__(self, simulation: Simulation):
        self.simulation = simulation
        self.is_running = False

    @abstractmethod
    def update(self) -> None:
        """Update visualization with current simulation state."""
        pass

    @abstractmethod
    def render(self) -> None:
        """Render the current frame."""
        pass

    @abstractmethod
    def start(self) -> None:
        """Start the visualization."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop the visualization."""
        pass


class MatplotlibVisualizer(Visualizer):
    """Matplotlib-based 3D visualization."""

    def __init__(self, simulation: Simulation, figsize: Tuple[int, int] = (10, 8)):
        super().__init__(simulation)
        self.figsize = figsize
        self.fig = None
        self.ax = None
        self.particle_plots = {}
        self.rigid_body_plots = {}
        self.force_vectors = {}
        self.trajectory_lines = {}

        # Visualization settings
        self.show_forces = True
        self.show_trajectories = True
        self.show_grid = True
        self.show_axes = True
        self.particle_size = 50
        self.force_scale = 0.1

        # Trajectory storage
        self.trajectory_data = {}

    def _setup_plot(self):
        """Initialize the 3D plot."""
        self.fig = plt.figure(figsize=self.figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Set up the plot appearance
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('Physics Simulation')

        # Set equal aspect ratio
        domain = self.simulation.config.domain_size
        max_range = max(domain) / 2

        self.ax.set_xlim([-max_range, max_range])
        self.ax.set_ylim([-max_range, max_range])
        self.ax.set_zlim([-max_range, max_range])

        if self.show_grid:
            self.ax.grid(True)

    def update(self):
        """Update visualization with current simulation state."""
        if self.ax is None:
            return

        # Clear previous plots
        self.ax.clear()
        self._setup_plot()

        # Plot particles
        self._plot_particles()

        # Plot rigid bodies
        self._plot_rigid_bodies()

        # Plot forces
        if self.show_forces:
            self._plot_forces()

        # Plot trajectories
        if self.show_trajectories:
            self._plot_trajectories()

    def _plot_particles(self):
        """Plot all particles in the simulation."""
        if not self.simulation.particles:
            return

        positions = np.array([p.position for p in self.simulation.particles])
        velocities = np.array([p.velocity for p in self.simulation.particles])
        masses = np.array([p.mass for p in self.simulation.particles])

        # Color by velocity magnitude
        vel_magnitudes = np.linalg.norm(velocities, axis=1)
        colors = vel_magnitudes

        # Size by mass
        sizes = self.particle_size * masses / np.mean(masses)

        scatter = self.ax.scatter(
            positions[:, 0], positions[:, 1], positions[:, 2],
            c=colors, s=sizes, alpha=0.7, cmap='viridis'
        )

        # Add colorbar
        if hasattr(self, '_colorbar'):
            self._colorbar.remove()
        self._colorbar = plt.colorbar(scatter, ax=self.ax, label='Velocity (m/s)')

    def _plot_rigid_bodies(self):
        """Plot rigid bodies as wireframes or solid shapes."""
        for rb in self.simulation.rigid_bodies:
            if rb.shape == "sphere":
                self._plot_sphere(rb)
            elif rb.shape == "box":
                self._plot_box(rb)
            elif rb.shape == "cylinder":
                self._plot_cylinder(rb)

    def _plot_sphere(self, rigid_body: RigidBody):
        """Plot a spherical rigid body."""
        center = rigid_body.center_of_mass
        radius = rigid_body.dimensions[0]

        # Create sphere surface
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
        y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]

        self.ax.plot_surface(x, y, z, alpha=0.3, color='blue')

    def _plot_box(self, rigid_body: RigidBody):
        """Plot a box-shaped rigid body."""
        center = rigid_body.center_of_mass
        dimensions = rigid_body.dimensions

        # Define box vertices
        dx, dy, dz = dimensions / 2
        vertices = np.array([
            [-dx, -dy, -dz], [dx, -dy, -dz], [dx, dy, -dz], [-dx, dy, -dz],
            [-dx, -dy, dz], [dx, -dy, dz], [dx, dy, dz], [-dx, dy, dz]
        ]) + center

        # Define box edges
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
        ]

        for edge in edges:
            points = vertices[edge]
            self.ax.plot3D(*points.T, 'b-', alpha=0.6)

    def _plot_cylinder(self, rigid_body: RigidBody):
        """Plot a cylindrical rigid body."""
        # Simplified cylinder representation
        center = rigid_body.center_of_mass
        radius = rigid_body.dimensions[0]
        height = rigid_body.dimensions[1]

        # Plot as two circles connected by lines
        theta = np.linspace(0, 2*np.pi, 20)

        # Bottom circle
        x_bottom = radius * np.cos(theta) + center[0]
        y_bottom = radius * np.sin(theta) + center[1]
        z_bottom = np.full_like(x_bottom, center[2] - height/2)

        # Top circle
        x_top = radius * np.cos(theta) + center[0]
        y_top = radius * np.sin(theta) + center[1]
        z_top = np.full_like(x_top, center[2] + height/2)

        self.ax.plot(x_bottom, y_bottom, z_bottom, 'b-', alpha=0.6)
        self.ax.plot(x_top, y_top, z_top, 'b-', alpha=0.6)

        # Connect with vertical lines
        for i in range(0, len(theta), 4):
            self.ax.plot([x_bottom[i], x_top[i]],
                        [y_bottom[i], y_top[i]],
                        [z_bottom[i], z_top[i]], 'b-', alpha=0.6)

    def _plot_forces(self):
        """Plot force vectors on particles."""
        for particle in self.simulation.particles:
            if np.linalg.norm(particle.acceleration) > 1e-6:
                force = particle.mass * particle.acceleration
                force_scaled = force * self.force_scale

                self.ax.quiver(
                    particle.position[0], particle.position[1], particle.position[2],
                    force_scaled[0], force_scaled[1], force_scaled[2],
                    color='red', alpha=0.7, arrow_length_ratio=0.1
                )

    def _plot_trajectories(self):
        """Plot particle trajectories."""
        for particle in self.simulation.particles:
            if particle.id not in self.trajectory_data:
                self.trajectory_data[particle.id] = []

            self.trajectory_data[particle.id].append(particle.position.copy())

            # Keep only recent trajectory points
            max_points = 100
            if len(self.trajectory_data[particle.id]) > max_points:
                self.trajectory_data[particle.id] = self.trajectory_data[particle.id][-max_points:]

            # Plot trajectory
            if len(self.trajectory_data[particle.id]) > 1:
                trajectory = np.array(self.trajectory_data[particle.id])
                self.ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                           alpha=0.5, linewidth=1)

    def render(self):
        """Render the current frame."""
        if self.fig:
            self.fig.canvas.draw()
            plt.pause(0.01)

    def start(self):
        """Start the visualization."""
        self.is_running = True
        self._setup_plot()
        plt.show(block=False)

    def stop(self):
        """Stop the visualization."""
        self.is_running = False
        if self.fig:
            plt.close(self.fig)


class PlotlyVisualizer(Visualizer):
    """Plotly-based interactive 3D visualization."""

    def __init__(self, simulation: Simulation):
        super().__init__(simulation)
        if not _PLOTLY_AVAILABLE:
            raise ImportError("Plotly not available. Install with: pip install plotly")

        self.fig = None
        self.trajectory_data = {}

    def _create_figure(self):
        """Create the Plotly figure."""
        self.fig = go.Figure()

        # Set up 3D scene
        domain = self.simulation.config.domain_size
        max_range = max(domain) / 2

        self.fig.update_layout(
            scene=dict(
                xaxis=dict(range=[-max_range, max_range], title='X (m)'),
                yaxis=dict(range=[-max_range, max_range], title='Y (m)'),
                zaxis=dict(range=[-max_range, max_range], title='Z (m)'),
                aspectmode='cube'
            ),
            title='Physics Simulation',
            showlegend=True
        )

    def update(self):
        """Update visualization with current simulation state."""
        if self.fig is None:
            self._create_figure()

        # Clear existing traces
        self.fig.data = []

        # Add particles
        self._add_particles()

        # Add rigid bodies
        self._add_rigid_bodies()

        # Add trajectories
        self._add_trajectories()

    def _add_particles(self):
        """Add particle scatter plot."""
        if not self.simulation.particles:
            return

        positions = np.array([p.position for p in self.simulation.particles])
        velocities = np.array([p.velocity for p in self.simulation.particles])
        masses = np.array([p.mass for p in self.simulation.particles])

        vel_magnitudes = np.linalg.norm(velocities, axis=1)

        self.fig.add_trace(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='markers',
            marker=dict(
                size=masses * 5,
                color=vel_magnitudes,
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title="Velocity (m/s)")
            ),
            name='Particles',
            text=[f'Particle {p.id}<br>Mass: {p.mass:.2f}<br>Velocity: {np.linalg.norm(p.velocity):.2f}'
                  for p in self.simulation.particles],
            hovertemplate='%{text}<extra></extra>'
        ))

    def _add_rigid_bodies(self):
        """Add rigid body visualizations."""
        for rb in self.simulation.rigid_bodies:
            if rb.shape == "sphere":
                self._add_sphere(rb)
            elif rb.shape == "box":
                self._add_box(rb)

    def _add_sphere(self, rigid_body: RigidBody):
        """Add spherical rigid body."""
        center = rigid_body.center_of_mass
        radius = rigid_body.dimensions[0]

        # Create sphere mesh
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
        y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]

        self.fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            opacity=0.3,
            colorscale='Blues',
            showscale=False,
            name=f'Rigid Body {rigid_body.id}'
        ))

    def _add_box(self, rigid_body: RigidBody):
        """Add box-shaped rigid body."""
        center = rigid_body.center_of_mass
        dimensions = rigid_body.dimensions

        # Create box mesh (simplified)
        dx, dy, dz = dimensions / 2
        vertices = np.array([
            [-dx, -dy, -dz], [dx, -dy, -dz], [dx, dy, -dz], [-dx, dy, -dz],
            [-dx, -dy, dz], [dx, -dy, dz], [dx, dy, dz], [-dx, dy, dz]
        ]) + center

        # Add box as scatter points (simplified representation)
        self.fig.add_trace(go.Scatter3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            mode='markers',
            marker=dict(size=8, color='blue', opacity=0.6),
            name=f'Rigid Body {rigid_body.id}'
        ))

    def _add_trajectories(self):
        """Add particle trajectories."""
        for particle in self.simulation.particles:
            if particle.id not in self.trajectory_data:
                self.trajectory_data[particle.id] = []

            self.trajectory_data[particle.id].append(particle.position.copy())

            # Keep only recent points
            max_points = 50
            if len(self.trajectory_data[particle.id]) > max_points:
                self.trajectory_data[particle.id] = self.trajectory_data[particle.id][-max_points:]

            if len(self.trajectory_data[particle.id]) > 1:
                trajectory = np.array(self.trajectory_data[particle.id])
                self.fig.add_trace(go.Scatter3d(
                    x=trajectory[:, 0],
                    y=trajectory[:, 1],
                    z=trajectory[:, 2],
                    mode='lines',
                    line=dict(width=2, color=f'rgba(255,0,0,0.5)'),
                    name=f'Trajectory {particle.id}',
                    showlegend=False
                ))

    def render(self):
        """Render the current frame."""
        if self.fig:
            self.fig.show()

    def start(self):
        """Start the visualization."""
        self.is_running = True
        self._create_figure()

    def stop(self):
        """Stop the visualization."""
        self.is_running = False


class RealTimeVisualizer:
    """Real-time visualization with automatic updates."""

    def __init__(self, simulation: Simulation, update_interval: float = 0.1,
                 visualizer_type: str = "matplotlib"):
        self.simulation = simulation
        self.update_interval = update_interval

        if visualizer_type == "matplotlib":
            self.visualizer = MatplotlibVisualizer(simulation)
        elif visualizer_type == "plotly" and _PLOTLY_AVAILABLE:
            self.visualizer = PlotlyVisualizer(simulation)
        else:
            raise ValueError(f"Unsupported visualizer type: {visualizer_type}")

        self.animation = None

    def _animate(self, frame):
        """Animation function for matplotlib."""
        self.simulation.step()
        self.visualizer.update()
        return []

    def run(self, max_steps: Optional[int] = None):
        """Run real-time visualization."""
        self.visualizer.start()

        if isinstance(self.visualizer, MatplotlibVisualizer):
            # Use matplotlib animation
            self.animation = FuncAnimation(
                self.visualizer.fig, self._animate,
                frames=max_steps, interval=int(self.update_interval * 1000),
                repeat=False, blit=False
            )
            plt.show()
        else:
            # Manual update loop for other visualizers
            step = 0
            while max_steps is None or step < max_steps:
                self.simulation.step()
                self.visualizer.update()
                self.visualizer.render()
                step += 1

    def stop(self):
        """Stop the visualization."""
        if self.animation:
            self.animation.event_source.stop()
        self.visualizer.stop()


class InteractiveControls:
    """Interactive controls for real-time parameter adjustment."""

    def __init__(self):
        self.controls = {}
        self.callbacks = {}

    def add_slider(self, name: str, min_val: float, max_val: float,
                  initial_val: float, callback: Optional[Callable] = None):
        """Add a slider control."""
        self.controls[name] = {
            "type": "slider",
            "min": min_val,
            "max": max_val,
            "value": initial_val
        }
        if callback:
            self.callbacks[name] = callback

    def add_button(self, name: str, callback: Optional[Callable] = None):
        """Add a button control."""
        self.controls[name] = {
            "type": "button"
        }
        if callback:
            self.callbacks[name] = callback

    def add_checkbox(self, name: str, initial_val: bool = False,
                    callback: Optional[Callable] = None):
        """Add a checkbox control."""
        self.controls[name] = {
            "type": "checkbox",
            "value": initial_val
        }
        if callback:
            self.callbacks[name] = callback

    def create_dash_app(self, simulation: Simulation) -> dash.Dash:
        """Create a Dash app with interactive controls."""
        if not _PLOTLY_AVAILABLE:
            raise ImportError("Dash/Plotly not available")

        app = dash.Dash(__name__)

        # Create layout with controls
        controls_layout = []
        for name, control in self.controls.items():
            if control["type"] == "slider":
                controls_layout.append(
                    html.Div([
                        html.Label(name.replace("_", " ").title()),
                        dcc.Slider(
                            id=f"slider-{name}",
                            min=control["min"],
                            max=control["max"],
                            value=control["value"],
                            step=(control["max"] - control["min"]) / 100,
                            marks={
                                control["min"]: str(control["min"]),
                                control["max"]: str(control["max"])
                            }
                        )
                    ])
                )
            elif control["type"] == "button":
                controls_layout.append(
                    html.Button(name.replace("_", " ").title(), id=f"button-{name}")
                )
            elif control["type"] == "checkbox":
                controls_layout.append(
                    dcc.Checklist(
                        id=f"checkbox-{name}",
                        options=[{"label": name.replace("_", " ").title(), "value": name}],
                        value=[name] if control["value"] else []
                    )
                )

        app.layout = html.Div([
            html.H1("Physics Simulation Controls"),
            html.Div(controls_layout, style={"width": "30%", "float": "left"}),
            dcc.Graph(id="simulation-plot", style={"width": "70%", "float": "right"}),
            dcc.Interval(id="interval-component", interval=100, n_intervals=0)
        ])

        return app


class MathematicalOverlay:
    """Mathematical overlay for displaying equations and energy plots."""

    def __init__(self):
        self.energy_history = []
        self.time_history = []
        self.equations = []

    def add_energy_plot(self):
        """Add energy conservation plot."""
        self.show_energy = True

    def add_equation_display(self):
        """Add equation display."""
        self.show_equations = True
        # Common physics equations
        self.equations = [
            r"$F = ma$",
            r"$E_k = \frac{1}{2}mv^2$",
            r"$E_p = mgh$",
            r"$p = mv$"
        ]

    def update(self, simulation: Simulation):
        """Update overlay with current simulation data."""
        # Record energy data
        if hasattr(self, 'show_energy'):
            kinetic = simulation.get_kinetic_energy()
            potential = simulation.get_potential_energy()
            total = kinetic + potential

            self.energy_history.append({
                "kinetic": kinetic,
                "potential": potential,
                "total": total
            })
            self.time_history.append(simulation.time)

            # Keep only recent history
            max_points = 1000
            if len(self.energy_history) > max_points:
                self.energy_history = self.energy_history[-max_points:]
                self.time_history = self.time_history[-max_points:]

    def render_energy_plot(self, ax):
        """Render energy conservation plot."""
        if not self.energy_history:
            return

        times = np.array(self.time_history)
        kinetic = [e["kinetic"] for e in self.energy_history]
        potential = [e["potential"] for e in self.energy_history]
        total = [e["total"] for e in self.energy_history]

        ax.clear()
        ax.plot(times, kinetic, label="Kinetic Energy", color="red")
        ax.plot(times, potential, label="Potential Energy", color="blue")
        ax.plot(times, total, label="Total Energy", color="green", linestyle="--")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Energy (J)")
        ax.set_title("Energy Conservation")
        ax.legend()
        ax.grid(True)

    def render_equations(self, ax):
        """Render physics equations."""
        if hasattr(self, 'show_equations'):
            ax.text(0.05, 0.95, "\n".join(self.equations),
                   transform=ax.transAxes, fontsize=12,
                   verticalalignment='top', bbox=dict(boxstyle="round", facecolor="wheat"))


# Convenience function for quick visualization
def visualize_simulation(simulation: Simulation, visualizer_type: str = "matplotlib",
                        real_time: bool = True, max_steps: Optional[int] = None):
    """Quick visualization setup for a simulation."""
    if real_time:
        viz = RealTimeVisualizer(simulation, visualizer_type=visualizer_type)
        viz.run(max_steps=max_steps)
    else:
        if visualizer_type == "matplotlib":
            viz = MatplotlibVisualizer(simulation)
        elif visualizer_type == "plotly":
            viz = PlotlyVisualizer(simulation)
        else:
            raise ValueError(f"Unsupported visualizer type: {visualizer_type}")

        viz.start()
        for _ in range(max_steps or 100):
            simulation.step()
            viz.update()
            viz.render()
        viz.stop()