"""
Health Recommendations Engine
==============================
Generates personalized health recommendations based on:
1. Brain age gap (predicted - chronological age)
2. SHAP feature contributions (which EEG frequency bands show accelerated aging)

Recommendations are evidence-based and sourced from neurophysiology literature.
"""

# ─────────────────────────────────────────────
# Feature-specific recommendations (by Frequency Band)
# ─────────────────────────────────────────────

BAND_RECOMMENDATIONS = {
    "Delta": {
        "icon": "🌊",
        "aging_recommendations": [
            "🧠 **Glymphatic Insight:** Delta waves are the 'engine' of the brain's waste-clearance (glymphatic) system. Lack of slow-wave sleep prevents the removal of amyloid-beta and tau proteins, accelerating neural aging.",
            "Excessive Delta while awake often points to 'metabolic slowing.' Intrusive slow waves can cloud focus and signal early-stage neurological decline.",
            "🛌 **Sleep Protocol:** Prioritize 7.5-9 hours of consistent, deep sleep in a 65°F (18°C) environment. Use Magnesium Glycinate (200-400mg) to reinforce GABA-ergic slow-wave integrity.",
            "Avoid blue light (screens) 90 minutes before bed; artificial light at night suppresses the naturally occurring Delta-reinforcing melatonin surge."
        ],
        "youthful_note": "Your Delta activity is optimally suppressed during wakefulness, indicating a sharp, metabolically efficient 'alert' state.",
    },
    "Theta": {
        "icon": "🧘",
        "aging_recommendations": [
            "📊 **Theta-Beta Marker:** A high Theta/Beta ratio is a gold-standard indicator for cognitive slowing and ADHD-like inattentiveness. In aging, increased Theta suggests 'cortical thinning' and reduced information processing speed.",
            "Increased Theta power is strongly linked to neuro-inflammation. It indicates the brain is 'idling' at a suboptimal frequency during complex tasks.",
            "🍉 **Metabolic Support:** Practice Intermittent Fasting (e.g., 16:8) which promotes Autophagy—the cellular recycling process that cleans 'clogged' Theta-producing circuits.",
            "🚀 **Cognitive Training:** Engage in Dual N-Back tasks for 15-20 minutes daily. This forces the brain to shift from slow Theta to more efficient high-frequency learning states."
        ],
        "youthful_note": "Your Theta activity shows healthy cognitive control and robust resistance to age-related 'drowsy-state' slowing.",
    },
    "Alpha": {
        "icon": "⚡",
        "aging_recommendations": [
            "🕰️ **The Brain's Clock:** Your 'Alpha-Peak Frequency' (APF) is your brain's internal processing speed. Slowing of this rhythm below 9Hz is a primary sign of functional brain aging and memory risk.",
            "Decreased Alpha power often signals a 'hyper-aroused' nervous system that can't enter the parasympathetic (rest-and-digest) state required for repair.",
            "🌳 **Forest Bathing:** Spend 40+ minutes in nature ('Shinrin-yoku'). The fractal patterns in nature naturally entrain the brain's posterior Alpha rhythm, reducing sympathetic stress.",
            "🧘 **Alpha-Breathing:** Practice Box Breathing (In 4s, Hold 4s, Out 4s, Hold 4s). This increases 'coherence'—the synchronization of Alpha waves across your frontal and parietal lobes."
        ],
        "youthful_note": "Your Alpha power is strong and robust; your Alpha-peak is healthy and fast, typical of a high-performance youthful brain.",
    },
    "Beta": {
        "icon": "🎯",
        "aging_recommendations": [
            "⚖️ **Beta Balance:** Differentiate between Beta-1 (13-20Hz: Focus) and High-Beta (>25Hz: Cortisol & Anxiety). Excessive High-Beta for long periods 'burns out' neural energy and drives oxidative stress.",
            "Chronic High-Beta indicates an overactive stress response ('fight or flight'), which prevents the brain from transitioning into deep recovery states.",
            "📉 **Biofeedback:** Implement HRV (Heart Rate Variability) training using a wearable device. Improving your HRV is the most effective way to 'cool down' hyper-active Beta circuits.",
            "🍵 **L-Theanine:** Consider green tea or L-Theanine supplements to modulate Beta activity, allowing for 'calm focus' (Beta-1) without the damaging High-Beta anxiety."
        ],
        "youthful_note": "Your Beta wave distribution is perfectly balanced, suggesting an alert, focused mind with minimal stress-induced noise.",
    },
    "Gamma": {
        "icon": "🧠",
        "aging_recommendations": [
            "⚡ **Neural Binding:** Gamma waves (30-80Hz) represent the 'binding' of information across different brain regions. High Gamma is essential for complex memory, creativity, and deep insight.",
            "Low Gamma is linked to the aging of parvalbumin (PV) interneurons—specialized cells that act as the neuro-pacemakers for high-performance cognitive states.",
            "🍫 **Flavonoid Support:** Consume high-polyphenol 'Brain Foods' like blueberries, dark chocolate (>70%), and organic walnuts to protect the PV-interneurons from age-related degradation.",
            "🎻 **Novel Learning:** Learn a complex new motor skill, like playing a musical instrument or dancing. These activities force diverse neural ensembles to synchronize at Gamma frequencies."
        ],
        "youthful_note": "Your Gamma activity indicates masterpiece-level neural synchronization and profound readiness for complex cognitive processing.",
    }
}


def get_overall_assessment(brain_age_gap, cognitive_score=None, risk_score=None):
    """
    Return an overall assessment based on the brain age gap and multi-target scores.
    """
    if brain_age_gap is None:
        return {
            "level": "info",
            "title": "Brain Age Prediction Complete",
            "summary": "Provide your chronological age for a personalized brain age gap assessment.",
            "color": "#60a5fa",
        }
    
    gap = brain_age_gap
    assessment = {}
    
    if gap <= -5:
        assessment = {
            "level": "excellent",
            "title": "Exceptional Brain Health",
            "summary": f"Your brain's EEG profile appears {abs(gap):.1f} years YOUNGER than your chronological age. Indicate excellent neurophysiological health.",
            "color": "#22c55e",
        }
    elif gap <= -2:
        assessment = {
            "level": "good",
            "title": "Good Brain Health",
            "summary": f"Your brain's EEG profile appears {abs(gap):.1f} years younger than expected. Your brain is aging well.",
            "color": "#4ade80",
        }
    elif gap <= 2:
        assessment = {
            "level": "normal",
            "title": "Normal Brain Aging",
            "summary": f"Your functional brain age is within {abs(gap):.1f} years of your chronological age. Typical for your age group.",
            "color": "#facc15",
        }
    elif gap <= 5:
        assessment = {
            "level": "attention",
            "title": "Mild Accelerated Aging",
            "summary": f"Your brain's EEG profile appears {gap:.1f} years OLDER than expected. Consider lifestyle support.",
            "color": "#fb923c",
        }
    else:
        assessment = {
            "level": "concern",
            "title": "Significant Accelerated Aging",
            "summary": f"Your brain's EEG profile appears {gap:.1f} years OLDER than expected. Clinical review suggested.",
            "color": "#ef4444",
        }

    # Integrate Multi-Target Insights
    if cognitive_score is not None:
        if cognitive_score < 50:
            assessment["summary"] += f" ⚠️ Low cognitive score ({cognitive_score}/100) detected."
        elif cognitive_score < 70:
            assessment["summary"] += f" ● Moderate cognitive score ({cognitive_score}/100)."
            
    if risk_score is not None:
        if risk_score > 7:
            assessment["summary"] += f" 🛑 High risk score ({risk_score}/10) requires attention."
        elif risk_score > 4:
            assessment["summary"] += f" 📎 Elevated risk score ({risk_score}/10)."

    return assessment


def generate_recommendations(brain_age_gap, feature_contributions, cognitive_score=None, risk_score=None):
    """
    Generate personalized health recommendations.
    """
    overall = get_overall_assessment(brain_age_gap, cognitive_score, risk_score)
    
    recommendations = []
    
    for contrib in feature_contributions:
        fname = contrib["feature"]
        raw_shap = contrib["shap_value"]
        adj_shap = contrib.get("adjusted_shap_value", raw_shap)
        raw_lime = contrib.get("lime_value", 0.0)
        adj_lime = contrib.get("adjusted_lime_value", raw_lime)
        
        # e.g., "Frontal_Alpha_Power" -> region="Frontal", band="Alpha"
        parts = fname.split('_')
        if len(parts) >= 2:
            region = parts[0]
            band = parts[1]
        else:
            continue
            
        rec_info = BAND_RECOMMENDATIONS.get(band, None)
        if rec_info is None:
            continue
        
        # Determine if this feature is pushing toward aging or youthful
        # We use the adjusted SHAP value to identify abnormal deviation from chronological age.
        is_aging = adj_shap > 0.4  # threshold for meaningful abnormal contribution
        is_youthful = adj_shap < -0.4
        
        region_display = f"{region} Lobe ({band} Band)"
        
        if is_aging:
            rec = {
                "feature": fname,
                "display_name": contrib["display_name"],
                "region": region_display,
                "icon": rec_info["icon"],
                "status": "attention",
                "status_label": "Accelerated Aging Indicator",
                "shap_value": raw_shap,
                "adjusted_shap_value": adj_shap,
                "lime_value": raw_lime,
                "adjusted_lime_value": adj_lime,
                "value": contrib["value"],
                "unit": contrib["unit"],
                "tips": rec_info["aging_recommendations"],
            }
        elif is_youthful:
            rec = {
                "feature": fname,
                "display_name": contrib["display_name"],
                "region": region_display,
                "icon": rec_info["icon"],
                "status": "positive",
                "status_label": "Youthful Indicator",
                "shap_value": raw_shap,
                "adjusted_shap_value": adj_shap,
                "lime_value": raw_lime,
                "adjusted_lime_value": adj_lime,
                "value": contrib["value"],
                "unit": contrib["unit"],
                "tips": [rec_info["youthful_note"]],
            }
        else:
            rec = {
                "feature": fname,
                "display_name": contrib["display_name"],
                "region": region_display,
                "icon": rec_info["icon"],
                "status": "neutral",
                "status_label": "Normal for your age",
                "shap_value": raw_shap,
                "adjusted_shap_value": adj_shap,
                "lime_value": raw_lime,
                "adjusted_lime_value": adj_lime,
                "value": contrib["value"],
                "unit": contrib["unit"],
                "tips": ["This frequency band is consistent with normal aging for your specific chronological age."],
            }
        
        recommendations.append(rec)
    
    # General lifestyle recommendations for neural health (always shown)
    general_tips = [
        {
            "icon": "🏃",
            "title": "Metabolic Neuro-Priming",
            "detail": "150+ minutes of Zone 2 aerobic exercise weekly. This triggers the release of **BDNF** (Brain-Derived Neurotrophic Factor), a 'fertilizer' for new neurons that reverses age-related hippocampal shrinkage.",
        },
        {
            "icon": "🥑",
            "title": "Neuro-Nutrient Strategy",
            "detail": "Prioritize high DHA/EPA Omega-3s and Sulforaphane (from broccoli sprouts). These compounds activate the **Nrf2 pathway**, the brain's primary internal defense against oxidative stress and aging.",
        },
        {
            "icon": "🌙",
            "title": "Autophagic Sleep Cycle",
            "detail": "Maintain a consistent 7-9 hour sleep window. Deep NREM sleep is the only time the brain performs **Autophagy**—a 'self-eating' process that recycles damaged proteins before they form toxic plaques.",
        },
        {
            "icon": "🧪",
            "title": "Hormetic Stress (Sauna/Cold)",
            "detail": "Occasional exposure to extreme heat (sauna) or cold (showers) triggers **Heat Shock Proteins (HSPs)**, which repair misfolded proteins and strengthen the brain's resilience to chronic aging.",
        },
        {
            "icon": "🧩",
            "title": "Cognitive Reserve Building",
            "detail": "Consistently learn skills that occupy 'novel' neural territory (e.g., 3D modeling, new languages). This builds **Cognitive Reserve**, allowing the brain to maintain high function even as physical aging occurs.",
        },
        {
            "icon": "🤝",
            "title": "Social Synaptic Pruning",
            "detail": "High-quality social interaction reduces 'loneliness-induced neuro-inflammation.' Engaging in complex conversation is one of the most demanding and protective tasks for the human frontal lobe.",
        },
    ]
    
    return {
        "overall_assessment": overall,
        "feature_recommendations": recommendations,
        "general_tips": general_tips,
    }
