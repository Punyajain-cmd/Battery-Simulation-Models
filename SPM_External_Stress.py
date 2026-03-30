import pybamm
import numpy as np
import matplotlib.pyplot as plt

from pybamm.models.submodels.active_material.loss_active_material import LossActiveMaterial
from pybamm.models.submodels.particle_mechanics.swelling_only import SwellingOnly
from pybamm.models.submodels.particle_mechanics.crack_propagation import CrackPropagation

# Base model
model = pybamm.lithium_ion.SPM({
})
# Define custom external stress input parameter
stress_times = np.array([0, 1000, 2000, 3600])
stress_values = np.array([5e3, 20e3, 50e3, 50e3])  # Hold last value flat
ext_stress = pybamm.Interpolant(stress_times, stress_values, pybamm.t)#, interpolator="zero")
model.variables["External mechanical stress [Pa]"] = ext_stress


# Custom LAM submodel with external stress influence
class CustomStressDrivenLAM(LossActiveMaterial):
    def __init__(self, param, domain, options, x_average, phase):
        super().__init__(param, domain, options=options, phase=phase, x_average=x_average)
        self.domain_param = param.domain_parameters[domain]
        self.phase_param = getattr(self.domain_param, phase)  # "primary" or "secondary"
        self.x_average = x_average

    def get_fundamental_variables(self):
        domain, Domain = self.domain_Domain
        phase = self.phase_name

        if self.x_average is True:
            eps_solid_xav = pybamm.Variable(
                f"X-averaged {domain} electrode {phase}active material volume fraction",
                domain="current collector",
            )
            eps_solid = pybamm.PrimaryBroadcast(eps_solid_xav, f"{domain} electrode")
        else:
            eps_solid = pybamm.Variable(
                f"{Domain} electrode {phase}active material volume fraction",
                domain=f"{domain} electrode",
                auxiliary_domains={"secondary": "current collector"},
            )
        variables = self._get_standard_active_material_variables(eps_solid)
        lli_due_to_lam = pybamm.Variable(
            f"Loss of lithium due to loss of {phase}active material "
            f"in {domain} electrode [mol]"
        )

        variables.update(
            {
                f"Loss of lithium due to loss of {phase}active material "
                f"in {domain} electrode [mol]": lli_due_to_lam
            }
        )
        return variables

    def get_coupled_variables(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        deps_solid_dt = 0
        lam_option = getattr(getattr(self.options, domain), self.phase)[
            "loss of active material"
        ]
        if "stress" in lam_option:
            # obtain the rate of loss of active materials (LAM) by stress
            # This is loss of active material model by mechanical effects
            if self.x_average is True:
                stress_t_surf = variables[
                    f"X-averaged {domain} {phase_name}particle surface tangential stress [Pa]"
                ]
                stress_r_surf = variables[
                    f"X-averaged {domain} {phase_name}particle surface radial stress [Pa]"
                ]
            else:
                stress_t_surf = variables[
                    f"{Domain} {phase_name}particle surface tangential stress [Pa]"
                ]
                stress_r_surf = variables[
                    f"{Domain} {phase_name}particle surface radial stress [Pa]"
                ]

            beta_LAM = self.phase_param.beta_LAM
            stress_critical = self.phase_param.stress_critical
            m_LAM = self.phase_param.m_LAM

            external_stress = variables["External mechanical stress [Pa]"]
            stress_h_surf = (stress_r_surf + 2 * stress_t_surf) / 3 + external_stress
            # compressive stress make no contribution
            stress_h_surf *= stress_h_surf > 0
            # assuming the minimum hydrostatic stress is zero for full cycles
            stress_h_surf_min = stress_h_surf * 0
            j_stress_LAM = (
                -beta_LAM
                * ((stress_h_surf - stress_h_surf_min) / stress_critical) ** m_LAM
            )
            deps_solid_dt += j_stress_LAM

        if "reaction" in lam_option:
            beta_LAM_sei = self.phase_param.beta_LAM_sei
            if self.x_average is True:
                a_j_sei = variables[
                    f"X-averaged {domain} electrode {phase_name}SEI "
                    "volumetric interfacial current density [A.m-3]"
                ]
            else:
                a_j_sei = variables[
                    f"{Domain} electrode {phase_name}SEI volumetric "
                    "interfacial current density [A.m-3]"
                ]

            j_stress_reaction = beta_LAM_sei * a_j_sei / self.param.F
            deps_solid_dt += j_stress_reaction

        if "current" in lam_option:
            # obtain the rate of loss of active materials (LAM) driven by current
            if self.x_average is True:
                T = variables[f"X-averaged {domain} electrode temperature [K]"]
            else:
                T = variables[f"{Domain} electrode temperature [K]"]

            j_current_LAM = self.domain_param.LAM_rate_current(
                self.param.current_density_with_time, T
            )
            deps_solid_dt += j_current_LAM

        variables.update(
            self._get_standard_active_material_change_variables(deps_solid_dt)
        )
        return variables

    def set_rhs(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        if self.x_average is True:
            eps_solid = variables[
                f"X-averaged {domain} electrode {phase_name}active material volume fraction"
            ]
            deps_solid_dt = variables[
                f"X-averaged {domain} electrode {phase_name}active material "
                "volume fraction change [s-1]"
            ]
        else:
            eps_solid = variables[
                f"{Domain} electrode {phase_name}active material volume fraction"
            ]
            deps_solid_dt = variables[
                f"{Domain} electrode {phase_name}active material volume fraction change [s-1]"
            ]


        lli_due_to_lam = variables[
            f"Loss of lithium due to loss of {phase_name}active material "
            f"in {domain} electrode [mol]"
        ]
        # Multiply by mol.m-3 * m3 to get mol
        c_s_rav = variables[
            f"R-averaged {domain} {phase_name}particle concentration [mol.m-3]"
        ]
        V = self.domain_param.L * self.param.A_cc

        self.rhs = {
            # minus sign because eps_solid is decreasing and LLI measures positive
            lli_due_to_lam: -V * pybamm.x_average(c_s_rav * deps_solid_dt),
            eps_solid: deps_solid_dt,
        }

    def set_initial_conditions(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        eps_solid_init = self.phase_param.epsilon_s

        if self.x_average is True:
            eps_solid_xav = variables[
                f"X-averaged {domain} electrode {phase_name}active material volume fraction"
            ]
            self.initial_conditions = {eps_solid_xav: pybamm.x_average(eps_solid_init)}
        else:
            eps_solid = variables[
                f"{Domain} electrode {phase_name}active material volume fraction"
            ]
            self.initial_conditions = {eps_solid: eps_solid_init}

        lli_due_to_lam = variables[
            f"Loss of lithium due to loss of {phase_name}active material "
            f"in {domain} electrode [mol]"
        ]
        self.initial_conditions[lli_due_to_lam] = pybamm.Scalar(0)

        #



class CustomSwelling(SwellingOnly):
 def __init__(self, param, domain, options, phase="primary"):
        super().__init__(param, domain, options, phase)

def get_fundamental_variables(self):
        domain, Domain = self.domain_Domain

        zero = pybamm.FullBroadcast(
            pybamm.Scalar(0), f"{domain} electrode", "current collector"
        )
        zero_av = pybamm.x_average(zero)
        variables = self._get_standard_variables(zero)
        variables.update(
            {
                f"{Domain} particle cracking rate [m.s-1]": zero,
                f"X-averaged {domain} particle cracking rate [m.s-1]": zero_av,
            }
        )
        return variables

def get_coupled_variables(self, variables):
        variables.update(self._get_standard_surface_variables(variables))
        variables.update(self._get_mechanical_results(variables))
        if self.size_distribution:
            variables.update(self._get_mechanical_size_distribution_results(variables))
        return variables




class CustomCrackPropagation(CrackPropagation):
    def __init__(self, param, domain, x_average, options, phase="primary"):
        super().__init__(param, domain, options, phase)
        self.x_average = x_average



    def get_fundamental_variables(self):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_param.phase_name
        if self.x_average:
            if self.size_distribution:
                l_cr_av_dist = pybamm.Variable(
                    f"X-averaged {domain} {phase_name}particle crack length distribution [m]",
                    domains={
                        "primary": f"{domain} particle size",
                        "secondary": "current collector",
                    },
                    scale=self.phase_param.l_cr_0,
                )
                l_cr_dist = pybamm.SecondaryBroadcast(
                    l_cr_av_dist, f"{domain} electrode"
                )
                l_cr_av = pybamm.size_average(l_cr_av_dist)
            else:
                l_cr_av = pybamm.Variable(
                    f"X-averaged {domain} {phase_name}particle crack length [m]",
                    domain="current collector",
                    scale=self.phase_param.l_cr_0,
                )
            l_cr = pybamm.PrimaryBroadcast(l_cr_av, f"{domain} electrode")
        else:
            if self.size_distribution:
                l_cr_dist = pybamm.Variable(
                    f"{Domain} {phase_name}particle crack length distribution [m]",
                    domains={
                        "primary": f"{domain} particle size",
                        "secondary": f"{domain} electrode",
                        "tertiary": "current collector",
                    },
                    scale=self.phase_param.l_cr_0,
                )
                l_cr = pybamm.size_average(l_cr_dist)
            else:
                l_cr = pybamm.Variable(
                    f"{Domain} {phase_name}particle crack length [m]",
                    domain=f"{domain} electrode",
                    auxiliary_domains={"secondary": "current collector"},
                    scale=self.phase_param.l_cr_0,
                )

        variables = self._get_standard_variables(l_cr)
        if self.size_distribution:
            variables.update(self._get_standard_size_distribution_variables(l_cr_dist))

        return variables

    def get_coupled_variables(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_param.phase_name
        variables.update(self._get_standard_surface_variables(variables))
        variables.update(self._get_mechanical_results(variables))
        if self.size_distribution:
            variables.update(self._get_mechanical_size_distribution_results(variables))
        T = variables[f"{Domain} electrode temperature [K]"]
        k_cr = self.phase_param.k_cr(T)
        m_cr = self.phase_param.m_cr
        b_cr = self.phase_param.b_cr
        if self.size_distribution:
            stress_t_surf = variables[
                f"{Domain} {phase_name}particle surface tangential stress distribution [Pa]"
            ]
        else:
            stress_t_surf = variables[
                f"{Domain} {phase_name}particle surface tangential stress [Pa]"
            ]
        if self.size_distribution:
            l_cr = variables[
                f"{Domain} {phase_name}particle crack length distribution [m]"
            ]
        else:
            l_cr = variables[f"{Domain} {phase_name}particle crack length [m]"]
        # # compressive stress will not lead to crack propagation
        dK_SIF = (stress_t_surf + ext_stress) * b_cr * pybamm.sqrt(np.pi * l_cr) * ((stress_t_surf + ext_stress) >= 0)
        dl_cr = k_cr * (dK_SIF**m_cr) / 3600  # divide by 3600 to replace t0_cr
        variables.update(
            {
                f"{Domain} {phase_name}particle cracking rate [m.s-1]": dl_cr,
                f"X-averaged {domain} {phase_name}particle cracking rate [m.s-1]": pybamm.x_average(
                    dl_cr
                ),
            }
        )
        return variables

    def set_rhs(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_param.phase_name
        if self.x_average is True:
            if self.size_distribution:
                l_cr = variables[
                    f"X-averaged {domain} {phase_name}particle crack length distribution [m]"
                ]
            else:
                l_cr = variables[
                    f"X-averaged {domain} {phase_name}particle crack length [m]"
                ]
            dl_cr = variables[
                f"X-averaged {domain} {phase_name}particle cracking rate [m.s-1]"
            ]
        else:
            if self.size_distribution:
                l_cr = variables[
                    f"{Domain} {phase_name}particle crack length distribution [m]"
                ]
            else:
                l_cr = variables[f"{Domain} {phase_name}particle crack length [m]"]
            dl_cr = variables[f"{Domain} {phase_name}particle cracking rate [m.s-1]"]
        self.rhs = {l_cr: dl_cr}

    def set_initial_conditions(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_param.phase_name
        l_cr_0 = self.phase_param.l_cr_0
        if self.x_average is True:
            if self.size_distribution:
                l_cr = variables[
                    f"X-averaged {domain} {phase_name}particle crack length distribution [m]"
                ]
                l_cr_0 = pybamm.PrimaryBroadcast(l_cr_0, f"{domain} particle size")
            else:
                l_cr = variables[
                    f"X-averaged {domain} {phase_name}particle crack length [m]"
                ]
        else:
            if self.size_distribution:
                l_cr = variables[
                    f"{Domain} {phase_name}particle crack length distribution [m]"
                ]
                l_cr_0 = pybamm.PrimaryBroadcast(l_cr_0, f"{domain} electrode")
                l_cr_0 = pybamm.PrimaryBroadcast(l_cr_0, f"{domain} particle size")
            else:
                l_cr = variables[f"{Domain} {phase_name}particle crack length [m]"]
                l_cr_0 = pybamm.PrimaryBroadcast(l_cr_0, f"{domain} electrode")
        self.initial_conditions = {l_cr: l_cr_0}

    def add_events_from(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_param.phase_name
        if self.x_average is True:
            l_cr = variables[
                f"X-averaged {domain} {phase_name}particle crack length [m]"
            ]
        else:
            l_cr = variables[f"{Domain} {phase_name}particle crack length [m]"]
        self.events.append(
            pybamm.Event(
                f"{domain} {phase_name} particle crack length larger than particle radius",
                1 - pybamm.max(l_cr) / self.phase_param.R_typ,
            )
        )















# Parameters
parameter_values = pybamm.ParameterValues("OKane2022")


# Define experiment
exp = pybamm.Experiment([
    "Charge at 1.5C until 4.2 V",
    "Hold at 4.2 V until C/50",
    "Discharge at 2.5C until 2.5 V"
] * 3)

# Create and solve simulation
sim = pybamm.Simulation(model, parameter_values=parameter_values, experiment=exp)

# Remove the Experiment block and use custom t_eval
t_eval = np.linspace(0, 3600, 300)
solution = sim.solve(t_eval=t_eval)


sim.plot([
    "Terminal voltage [V]",
    "Discharge capacity [A.h]",
    "Negative electrode active material volume fraction",
    "External mechanical stress [Pa]",
])

