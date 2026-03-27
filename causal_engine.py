"""
Causal Intelligence Engine
===========================
Implements:
1. Structural Causal Models (SCM) for brain aging.
2. Do-calculus for intervention analysis.
3. Generative simulation of future aging trajectories.
"""

import numpy as np

class StructuralCausalModel:
    """
    Simulates a causal DAG for brain aging.
    Variables:
    - Sleep (S)
    - Stress (T)
    - Nutrition (N)
    - Alpha_Power (A)
    - Theta_Power (Th)
    - Brain_Age (BA)
    
    DAG: 
    S -> A
    S -> BA
    T -> Th
    T -> BA
    N -> A
    A -> BA
    Th -> BA
    """
    def __init__(self):
        # Coefficients for dependencies
        self.coeffs = {
            'S_to_A': 2.5,
            'N_to_A': 1.5,
            'S_to_BA': -0.5,
            'T_to_Th': 3.0,
            'T_to_BA': 0.8,
            'A_to_BA': -0.4,
            'Th_to_BA': 0.6,
        }

    def simulate(self, interventions=None):
        """
        Run a structural simulation with optional do-interventions.
        interventions: {'Sleep': 8.0} -> forces Sleep to 8 hours (do-operation).
        """
        rng = np.random.default_rng()
        
        # Exogenous / Baseline values
        sleep = interventions.get('Sleep', rng.uniform(5, 9))
        stress = interventions.get('Stress', rng.uniform(1, 10))
        nutrition = interventions.get('Nutrition', rng.uniform(1, 10))
        
        # Endogenous pathways
        alpha = (self.coeffs['S_to_A'] * sleep) + (self.coeffs['N_to_A'] * nutrition) + rng.normal(0, 2)
        theta = (self.coeffs['T_to_Th'] * stress) + rng.normal(0, 1.5)
        
        # Brain Age Gap (effect on gap)
        # Higher Alpha = Lower Gap (Protective)
        # Higher Theta = Higher Gap (Aging)
        # Higher Sleep = Lower Gap
        # Higher Stress = Higher Gap
        ba_gap = (self.coeffs['S_to_BA'] * sleep) + \
                 (self.coeffs['T_to_BA'] * stress) + \
                 (self.coeffs['A_to_BA'] * alpha) + \
                 (self.coeffs['Th_to_BA'] * theta) + \
                 rng.normal(0, 1)
                 
        return {
            'inputs': {'Sleep': round(sleep, 1), 'Stress': round(stress, 1), 'Nutrition': round(nutrition,1)},
            'mediators': {'Alpha_Power': round(alpha, 1), 'Theta_Power': round(theta, 1)},
            'outcome': {'Brain_Age_Gap': round(ba_gap, 1)}
        }

class InterventionEngine:
    """Uses the SCM to perform do-calculus and health recommendations."""
    def __init__(self):
        self.scm = StructuralCausalModel()

    def perform_do_calculus(self, target_variable, target_value):
        """
        Simulate the effect of a specific intervention.
        Example: target_variable='Sleep', target_value=9.0
        """
        # Run 100 simulations with the intervention
        results = [self.scm.simulate(interventions={target_variable: target_value}) for _ in range(100)]
        avg_gap = np.mean([r['outcome']['Brain_Age_Gap'] for r in results])
        
        # Run 100 simulations without intervention (baseline)
        baseline = [self.scm.simulate(interventions={}) for _ in range(100)]
        baseline_gap = np.mean([r['outcome']['Brain_Age_Gap'] for r in baseline])
        
        causal_effect = avg_gap - baseline_gap
        
        return {
            'intervention': f"do({target_variable} = {target_value})",
            'average_outcome_gap': round(float(avg_gap), 2),
            'baseline_gap': round(float(baseline_gap), 2),
            'causal_effect': round(float(causal_effect), 2),
            'interpretation': "Reduction in accelerated aging" if causal_effect < 0 else "Increase in accelerated aging"
        }

    def predict_future_trajectory(self, current_gap, years=5, lifestyle_adjustment='none'):
        """Predict future brain age gap over time under different conditions."""
        trajectories = []
        gap = current_gap
        
        adj_factor = 0
        if lifestyle_adjustment == 'improved': adj_factor = -0.5
        elif lifestyle_adjustment == 'worsened': adj_factor = 0.5
        
        for year in range(1, years + 1):
            # Natural aging trend + adjustment
            gap += (0.2 + adj_factor) 
            trajectories.append({'year': year, 'projected_gap': round(gap, 2)})
            
        return trajectories
