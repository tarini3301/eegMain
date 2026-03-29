"""
NeuroAge Recommendation Agent (RL)
==================================
A Reinforcement Learning (RL) agent that suggests lifestyle optimizations.
Uses a Q-learning style simulation to find the best health 'actions'.
"""

import numpy as np

class RecommendationAgent:
    def __init__(self):
        # Actions: [Sleep+, Stress-, Nutrition+, Meditation+, Exercise+]
        self.actions = ['Improve Sleep', 'Reduce Stress', 'Optimize Nutrition', 'Daily Meditation', 'Cardio Exercise']
        self.n_actions = len(self.actions)
        
        # Simulated Q-table [Simple State: Brain Age Gap Level]
        # Levels: 0 (Younger), 1 (Normal), 2 (Accelerated)
        self.q_table = np.zeros((3, self.n_actions))
        
        # Pre-populate with 'expert' knowledge (Simplified Training)
        self.q_table[2] = [0.8, 0.9, 0.5, 0.7, 0.6] # For accelerated: Stress and Sleep are high reward
        self.q_table[1] = [0.4, 0.3, 0.6, 0.2, 0.8] # For normal: Exercise and Nutrition are better for maintenance
        self.q_table[0] = [0.2, 0.1, 0.4, 0.1, 0.5] # For younger: Maintenance

    def get_recommendations(self, brain_age_gap, stress_index):
        """
        Policy Selection based on current patient state.
        Determines the top 3 actions with highest expected 'Reward' (Brain Age Reduction).
        """
        # Determine State
        if brain_age_gap > 2:
            state = 2 # Accelerated
        elif brain_age_gap < -2:
            state = 0 # Healthy
        else:
            state = 1 # Normal
            
        # Get Q-values for current state
        q_values = self.q_table[state].copy()
        
        # Adjust for Stress Index (Heuristic bonus)
        if stress_index > 7:
            q_values[1] += 0.5 # Bonus for 'Reduce Stress'
            q_values[3] += 0.3 # Bonus for 'Meditation'
            
        # Select Top 3 Actions
        top_indices = np.argsort(q_values)[::-1][:3]
        
        recommendations = []
        for idx in top_indices:
            action_name = self.actions[idx]
            reward_score = q_values[idx]
            
            # Map rewards to clinical descriptions
            desc = self._get_action_description(action_name)
            
            recommendations.append({
                'action': action_name,
                'reward': round(float(reward_score), 2),
                'reasoning': desc
            })
            
        return recommendations

    def _get_action_description(self, action):
        descriptions = {
            'Improve Sleep': '🧠 **Glymphatic Protocol:** Target 7.5-8.5 hours in a cool (65°F / 18°C) dark environment to optimize amyloid-beta clearance through the sleep-driven brain-washing cycle.',
            'Reduce Stress': '🌊 **Vagal Regulation:** Implement HRV-based biofeedback or deep diaphragmatic breathing to lower high-beta "overdrive" and improve the theta/alpha metabolic ratio.',
            'Optimize Nutrition': '🥬 **Nrf2-Activation:** High-omega-3 intake combined with sulforaphane-rich foods to trigger internal antioxidant production and reduce neural membrane oxidation.',
            'Daily Meditation': '⚡ **Gamma Synchronization:** Specifically 40Hz-focused mindfulness sessions to reinforce parvalbumin interneuron health and improve long-range connectivity.',
            'Cardio Exercise': '🏃 **BDNF Priming:** 150 minutes of Zone-2 (60-70% max HR) aerobic work weekly to maximize Brain-Derived Neurotrophic Factor and maintain hippocampal volume.'
        }
        return descriptions.get(action, "Clinical-grade neuro-intervention recommended.")

# Global instance
agent = RecommendationAgent()
