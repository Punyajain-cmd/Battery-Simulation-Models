"""
Battery Stress Visualization and Analysis Tools
================================================
Tools for visualizing and analyzing stress calculation results

Author: Research Implementation
Date: February 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Dict, Optional, Tuple
import os


class StressVisualizer:
    """Visualization tools for battery stress analysis"""
    
    def __init__(self, results: Dict, figsize: Tuple[int, int] = (12, 8)):
        """
        Parameters:
        -----------
        results : dict
            Results dictionary from IntegratedStressAnalysis
        figsize : tuple
            Default figure size
        """
        self.results = results
        self.figsize = figsize
        
        # Set publication-quality plotting defaults
        plt.rcParams.update({
            'font.size': 11,
            'font.family': 'serif',
            'axes.labelsize': 12,
            'axes.titlesize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 14,
            'lines.linewidth': 1.5,
            'axes.grid': True,
            'grid.alpha': 0.3
        })
    
    def plot_particle_stress_evolution(self, 
                                       time_indices: Optional[list] = None,
                                       save_path: Optional[str] = None):
        """
        Plot radial distribution of stress at different time snapshots
        """
        
        particle = self.results['particle']
        t = particle['t']
        r = particle['r']
        sigma_theta = particle['sigma_theta']
        
        if time_indices is None:
            # Select 5 evenly spaced time points
            time_indices = np.linspace(0, len(t)-1, 5, dtype=int)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Hoop stress
        for idx in time_indices:
            label = f't = {t[idx]:.1f} s'
            ax1.plot(r*1e6, sigma_theta[idx, :]/1e6, label=label)
        
        ax1.set_xlabel('Radial Position (μm)')
        ax1.set_ylabel('Hoop Stress (MPa)')
        ax1.set_title('Hoop Stress Distribution in Particle')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Radial stress
        sigma_r = particle['sigma_r']
        for idx in time_indices:
            label = f't = {t[idx]:.1f} s'
            ax2.plot(r*1e6, sigma_r[idx, :]/1e6, label=label)
        
        ax2.set_xlabel('Radial Position (μm)')
        ax2.set_ylabel('Radial Stress (MPa)')
        ax2.set_title('Radial Stress Distribution in Particle')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()
    
    def plot_particle_stress_contour(self, 
                                     time_idx: int = -1,
                                     save_path: Optional[str] = None):
        """
        Plot 2D contour of hoop stress in particle cross-section
        """
        
        particle = self.results['particle']
        r = particle['r']
        sigma_theta = particle['sigma_theta'][time_idx, :]
        t = particle['t'][time_idx]
        
        # Create 2D mesh for visualization
        theta = np.linspace(0, 2*np.pi, 100)
        R, Theta = np.meshgrid(r, theta)
        
        # Convert to Cartesian
        X = R * np.cos(Theta)
        Y = R * np.sin(Theta)
        
        # Replicate stress for all angles (spherically symmetric)
        Z = np.tile(sigma_theta, (len(theta), 1))
        
        fig, ax = plt.subplots(figsize=(8, 7))
        
        # Contour plot
        levels = np.linspace(np.min(sigma_theta), np.max(sigma_theta), 20)
        contour = ax.contourf(X*1e6, Y*1e6, Z/1e6, levels=levels, cmap='RdYlBu_r')
        
        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(contour, cax=cax)
        cbar.set_label('Hoop Stress (MPa)')
        
        ax.set_xlabel('x (μm)')
        ax.set_ylabel('y (μm)')
        ax.set_title(f'Hoop Stress Distribution (t = {t:.1f} s)')
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()
    
    def plot_electrode_stress_map(self, 
                                  stress_component: str = 'von_mises',
                                  save_path: Optional[str] = None):
        """
        Plot 2D heatmap of stress through electrode thickness over time
        
        Parameters:
        -----------
        stress_component : str
            'xx', 'yy', 'zz', 'von_mises', or 'hydrostatic'
        """
        
        electrode = self.results['electrode']
        t = electrode['t']
        z = electrode['z']
        
        # Get stress component
        if stress_component == 'xx':
            stress = electrode['sigma_xx']
            label = r'$\sigma_{xx}$ (MPa)'
        elif stress_component == 'yy':
            stress = electrode['sigma_yy']
            label = r'$\sigma_{yy}$ (MPa)'
        elif stress_component == 'zz':
            stress = electrode['sigma_zz']
            label = r'$\sigma_{zz}$ (MPa)'
        elif stress_component == 'von_mises':
            sigma_xx = electrode['sigma_xx']
            sigma_yy = electrode['sigma_yy']
            sigma_zz = electrode['sigma_zz']
            stress = np.sqrt(0.5 * (
                (sigma_xx - sigma_yy)**2 + 
                (sigma_yy - sigma_zz)**2 + 
                (sigma_zz - sigma_xx)**2
            ))
            label = r'$\sigma_{VM}$ (MPa)'
        elif stress_component == 'hydrostatic':
            stress = (electrode['sigma_xx'] + electrode['sigma_yy'] + 
                     electrode['sigma_zz']) / 3.0
            label = r'$\sigma_h$ (MPa)'
        else:
            raise ValueError(f"Unknown stress component: {stress_component}")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create mesh
        T, Z = np.meshgrid(t, z*1e6)
        
        # Contour plot
        levels = np.linspace(np.min(stress), np.max(stress), 30)
        contour = ax.contourf(T, Z.T, stress/1e6, levels=levels, cmap='viridis')
        
        # Colorbar
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label(label)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Electrode Depth (μm)')
        ax.set_title(f'Electrode Stress Evolution - {stress_component.upper()}')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()
    
    def plot_layer_stress_profiles(self,
                                   time_indices: Optional[list] = None,
                                   save_path: Optional[str] = None):
        """
        Plot stress profiles through thickness at different times
        """
        
        electrode = self.results['electrode']
        t = electrode['t']
        z = electrode['z']
        sigma_xx = electrode['sigma_xx']
        sigma_zz = electrode['sigma_zz']
        
        if time_indices is None:
            time_indices = np.linspace(0, len(t)-1, 5, dtype=int)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # In-plane stress
        for idx in time_indices:
            label = f't = {t[idx]:.1f} s'
            ax1.plot(sigma_xx[idx, :]/1e6, z*1e6, label=label, marker='o', markersize=4)
        
        ax1.set_xlabel(r'In-Plane Stress $\sigma_{xx}$ (MPa)')
        ax1.set_ylabel('Depth from Current Collector (μm)')
        ax1.set_title('In-Plane Stress Profile')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Through-thickness stress
        for idx in time_indices:
            label = f't = {t[idx]:.1f} s'
            ax2.plot(sigma_zz[idx, :]/1e6, z*1e6, label=label, marker='s', markersize=4)
        
        ax2.set_xlabel(r'Through-Thickness Stress $\sigma_{zz}$ (MPa)')
        ax2.set_ylabel('Depth from Current Collector (μm)')
        ax2.set_title('Through-Thickness Stress Profile')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()
    
    def plot_curvature_evolution(self, save_path: Optional[str] = None):
        """Plot electrode curvature over time"""
        
        electrode = self.results['electrode']
        t = electrode['t']
        kappa = electrode['kappa']
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        ax.plot(t, kappa, 'b-', linewidth=2)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Curvature (1/m)')
        ax.set_title('Electrode Curvature Evolution')
        ax.grid(True, alpha=0.3)
        
        # Add max curvature annotation
        max_kappa_idx = np.argmax(np.abs(kappa))
        ax.plot(t[max_kappa_idx], kappa[max_kappa_idx], 'ro', markersize=8)
        ax.annotate(f'Max: {kappa[max_kappa_idx]:.2e} m⁻¹',
                   xy=(t[max_kappa_idx], kappa[max_kappa_idx]),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()
    
    def plot_comprehensive_summary(self, save_path: Optional[str] = None):
        """Create comprehensive summary figure with all key results"""
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Particle hoop stress (final time)
        ax1 = fig.add_subplot(gs[0, 0])
        particle = self.results['particle']
        r = particle['r']
        sigma_theta = particle['sigma_theta']
        ax1.plot(r*1e6, sigma_theta[-1, :]/1e6, 'b-', linewidth=2)
        ax1.set_xlabel('Radial Position (μm)')
        ax1.set_ylabel('Hoop Stress (MPa)')
        ax1.set_title('Particle Stress (Final)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Electrode stress heatmap
        ax2 = fig.add_subplot(gs[0, 1:])
        electrode = self.results['electrode']
        t = electrode['t']
        z = electrode['z']
        sigma_xx = electrode['sigma_xx']
        T, Z = np.meshgrid(t, z*1e6)
        contour = ax2.contourf(T, Z.T, sigma_xx/1e6, levels=30, cmap='viridis')
        cbar = plt.colorbar(contour, ax=ax2)
        cbar.set_label(r'$\sigma_{xx}$ (MPa)')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Depth (μm)')
        ax2.set_title('Electrode In-Plane Stress Evolution')
        
        # 3. Layer stress profiles
        ax3 = fig.add_subplot(gs[1, 0])
        time_indices = np.linspace(0, len(t)-1, 5, dtype=int)
        for idx in time_indices:
            ax3.plot(sigma_xx[idx, :]/1e6, z*1e6, label=f'{t[idx]:.0f}s')
        ax3.set_xlabel(r'$\sigma_{xx}$ (MPa)')
        ax3.set_ylabel('Depth (μm)')
        ax3.set_title('Layer Stress Profiles')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # 4. Curvature evolution
        ax4 = fig.add_subplot(gs[1, 1])
        kappa = electrode['kappa']
        ax4.plot(t, kappa*1e3, 'r-', linewidth=2)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Curvature (×10⁻³ m⁻¹)')
        ax4.set_title('Electrode Curvature')
        ax4.grid(True, alpha=0.3)
        
        # 5. Von Mises stress evolution
        ax5 = fig.add_subplot(gs[1, 2])
        sigma_vm = np.sqrt(0.5 * (
            (electrode['sigma_xx'] - electrode['sigma_yy'])**2 + 
            (electrode['sigma_yy'] - electrode['sigma_zz'])**2 + 
            (electrode['sigma_zz'] - electrode['sigma_xx'])**2
        ))
        max_vm = np.max(sigma_vm, axis=1)
        avg_vm = np.mean(sigma_vm, axis=1)
        ax5.plot(t, max_vm/1e6, 'b-', linewidth=2, label='Maximum')
        ax5.plot(t, avg_vm/1e6, 'g--', linewidth=2, label='Average')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Von Mises Stress (MPa)')
        ax5.set_title('Von Mises Stress Metrics')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Concentration evolution
        ax6 = fig.add_subplot(gs[2, :])
        c = electrode['c']
        T, Z = np.meshgrid(t, z*1e6)
        contour = ax6.contourf(T, Z.T, c/electrode['c'].max(), levels=30, cmap='YlOrRd')
        cbar = plt.colorbar(contour, ax=ax6)
        cbar.set_label('Normalized Concentration')
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Depth (μm)')
        ax6.set_title('Li Concentration Evolution')
        
        plt.suptitle('Comprehensive Battery Stress Analysis Summary', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()


class StressAnalyzer:
    """Advanced analysis tools for stress data"""
    
    @staticmethod
    def load_stress_data(filename: str) -> Dict:
        """Load stress data from npz file"""
        data = np.load(filename)
        return {key: data[key] for key in data.files}
    
    @staticmethod
    def compute_failure_probability(sigma_max: np.ndarray,
                                    sigma_f: float,
                                    weibull_modulus: float = 5.0) -> np.ndarray:
        """
        Compute failure probability using Weibull statistics
        
        Parameters:
        -----------
        sigma_max : np.ndarray
            Maximum stress history
        sigma_f : float
            Characteristic strength
        weibull_modulus : float
            Weibull modulus (typical: 5-10 for ceramics)
            
        Returns:
        --------
        P_f : np.ndarray
            Failure probability at each time
        """
        
        # Weibull distribution
        P_f = 1.0 - np.exp(-(sigma_max / sigma_f)**weibull_modulus)
        
        return P_f
    
    @staticmethod
    def identify_critical_layers(sigma: np.ndarray, 
                                 threshold_percentile: float = 95.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Identify layers with critical stress levels
        
        Parameters:
        -----------
        sigma : np.ndarray (n_t, n_layers)
            Stress field
        threshold_percentile : float
            Percentile for critical stress
            
        Returns:
        --------
        critical_layers : np.ndarray
            Indices of critical layers
        time_exceed : np.ndarray
            Time points when each layer exceeds threshold
        """
        
        threshold = np.percentile(sigma, threshold_percentile)
        
        critical_layers = []
        time_exceed = []
        
        for layer_idx in range(sigma.shape[1]):
            exceed_times = np.where(sigma[:, layer_idx] > threshold)[0]
            if len(exceed_times) > 0:
                critical_layers.append(layer_idx)
                time_exceed.append(exceed_times[0])
        
        return np.array(critical_layers), np.array(time_exceed)
    
    @staticmethod
    def compute_stress_gradient(sigma: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Compute stress gradient through thickness
        
        Returns:
        --------
        grad_sigma : np.ndarray (n_t, n_layers-1)
            Stress gradient (Pa/m)
        """
        
        grad_sigma = np.diff(sigma, axis=1) / np.diff(z)
        
        return grad_sigma
    
    @staticmethod
    def export_for_abaqus(stress_data: Dict, 
                         output_file: str = 'abaqus_stress_input.txt'):
        """
        Export stress data in format for ABAQUS user subroutine
        
        Format: time, layer_index, sigma_xx, sigma_yy, sigma_zz
        """
        
        t = stress_data['time']
        sigma_xx = stress_data['sigma_xx']
        sigma_yy = stress_data['sigma_yy']
        sigma_zz = stress_data['sigma_zz']
        
        with open(output_file, 'w') as f:
            f.write("# ABAQUS Stress Input Data\n")
            f.write("# time(s), layer, sigma_xx(Pa), sigma_yy(Pa), sigma_zz(Pa)\n")
            
            n_t, n_layers = sigma_xx.shape
            for i in range(n_t):
                for j in range(n_layers):
                    f.write(f"{t[i]:.6e}, {j}, "
                           f"{sigma_xx[i,j]:.6e}, {sigma_yy[i,j]:.6e}, {sigma_zz[i,j]:.6e}\n")
        
        print(f"ABAQUS stress data exported to {output_file}")
        print(f"  Time steps: {n_t}")
        print(f"  Layers: {n_layers}")
    
    @staticmethod
    def export_for_comsol(stress_data: Dict,
                         output_file: str = 'comsol_stress_input.txt'):
        """
        Export stress data in format for COMSOL interpolation function
        
        Format: CSV with headers
        """
        
        t = stress_data['time']
        z = stress_data['z_coordinates']
        sigma_xx = stress_data['sigma_xx']
        sigma_yy = stress_data['sigma_yy']
        sigma_zz = stress_data['sigma_zz']
        
        with open(output_file, 'w') as f:
            f.write("% COMSOL Interpolation Function Data\n")
            f.write("% time(s), z(m), sigma_xx(Pa), sigma_yy(Pa), sigma_zz(Pa)\n")
            f.write("time,z,sigma_xx,sigma_yy,sigma_zz\n")
            
            n_t, n_layers = sigma_xx.shape
            for i in range(n_t):
                for j in range(n_layers):
                    f.write(f"{t[i]:.6e},{z[j]:.6e},"
                           f"{sigma_xx[i,j]:.6e},{sigma_yy[i,j]:.6e},{sigma_zz[i,j]:.6e}\n")
        
        print(f"COMSOL stress data exported to {output_file}")
    
    @staticmethod
    def generate_analysis_report(stress_data: Dict, 
                                 material_name: str,
                                 output_file: str = 'stress_analysis_report.txt'):
        """Generate comprehensive text report of stress analysis"""
        
        sigma_xx = stress_data['sigma_xx']
        sigma_yy = stress_data['sigma_yy']
        sigma_zz = stress_data['sigma_zz']
        sigma_vm = stress_data['sigma_von_mises']
        sigma_h = stress_data['sigma_hydrostatic']
        
        with open(output_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("BATTERY ELECTRODE STRESS ANALYSIS REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Material: {material_name}\n")
            f.write(f"Electrode Thickness: {stress_data['electrode_thickness']*1e6:.2f} μm\n")
            f.write(f"Porosity: {stress_data['porosity']*100:.1f}%\n")
            f.write(f"Number of Layers: {len(stress_data['z_coordinates'])}\n")
            f.write(f"Time Steps: {len(stress_data['time'])}\n\n")
            
            f.write("-"*70 + "\n")
            f.write("STRESS STATISTICS\n")
            f.write("-"*70 + "\n\n")
            
            f.write(f"In-Plane Stress (σ_xx):\n")
            f.write(f"  Maximum: {np.max(sigma_xx)/1e6:.2f} MPa\n")
            f.write(f"  Minimum: {np.min(sigma_xx)/1e6:.2f} MPa\n")
            f.write(f"  Average: {np.mean(sigma_xx)/1e6:.2f} MPa\n\n")
            
            f.write(f"Through-Thickness Stress (σ_zz):\n")
            f.write(f"  Maximum: {np.max(sigma_zz)/1e6:.2f} MPa\n")
            f.write(f"  Minimum: {np.min(sigma_zz)/1e6:.2f} MPa\n")
            f.write(f"  Average: {np.mean(sigma_zz)/1e6:.2f} MPa\n\n")
            
            f.write(f"Von Mises Stress:\n")
            f.write(f"  Maximum: {np.max(sigma_vm)/1e6:.2f} MPa\n")
            f.write(f"  Average: {np.mean(sigma_vm)/1e6:.2f} MPa\n\n")
            
            f.write(f"Hydrostatic Stress:\n")
            f.write(f"  Maximum: {np.max(sigma_h)/1e6:.2f} MPa\n")
            f.write(f"  Minimum: {np.min(sigma_h)/1e6:.2f} MPa\n\n")
            
            f.write("-"*70 + "\n")
            f.write("CRITICAL LOCATIONS\n")
            f.write("-"*70 + "\n\n")
            
            max_vm_idx = np.unravel_index(np.argmax(sigma_vm), sigma_vm.shape)
            f.write(f"Maximum Von Mises Stress:\n")
            f.write(f"  Time: {stress_data['time'][max_vm_idx[0]]:.2f} s\n")
            f.write(f"  Layer: {max_vm_idx[1]}\n")
            f.write(f"  Depth: {stress_data['z_coordinates'][max_vm_idx[1]]*1e6:.2f} μm\n")
            f.write(f"  Value: {sigma_vm[max_vm_idx]/1e6:.2f} MPa\n\n")
            
            f.write("="*70 + "\n")
        
        print(f"Analysis report saved to {output_file}")


if __name__ == "__main__":
    print("Battery Stress Visualization Tools")
    print("Import this module and use StressVisualizer class with your results")
