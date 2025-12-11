class StructuralBrain:
    def __init__(self):
        # Base Thresholds
        self.LIMITS = {
            'vib': 8.0,      # DIN 4150-3
            'gas': 0.6,      # NFPA
            'crack': 2.0,    # IS 456
            'tilt': 1.0,     # INSARAG
            'temp': 60.0,    # Fire
            'support': 0.5   # Stability
        }
 
    def get_alpha(self, wall_type, roof_type, collapse, age_years, col_condition, storeys):
        """Calculates Fragility with Wall/Roof Interaction."""
        alpha = 1.0
        context_tags = []
 
        # 1. WALL MATERIAL (The Primary Support)
        if wall_type == "Brick": 
            alpha += 0.5; context_tags.append("Brittle_Walls")
        elif wall_type == "Mud": 
            alpha += 1.5; context_tags.append("Weak_Walls")
        elif wall_type == "Stone": 
            alpha += 0.8; context_tags.append("Heavy_Masonry")
        elif wall_type == "RCC": 
            context_tags.append("Strong_Walls") # No penalty
        
        # 2. ROOF MATERIAL (The Load)
        # Interaction Logic: Heavy Roof on Weak Walls is fatal.
        if roof_type == "Concrete":
            context_tags.append("Heavy_Roof")
            if wall_type in ["Mud", "Stone", "Brick"]:
                alpha += 0.5 # PENALTY: Top-Heavy Crushing Risk
                context_tags.append("CRITICAL_TOP_HEAVY")
        
        elif roof_type == "Sheet":
            alpha -= 0.2 # BONUS: Lightweight Roof (Less pendulum effect)
            context_tags.append("Light_Roof")
        
        elif roof_type == "Timber":
            context_tags.append("Flexible_Roof") # Neutral
 
        # 3. COLLAPSE PATTERN
        if collapse == "Lean-To": alpha += 0.6; context_tags.append("Friction_Dependent")
        elif collapse == "Cantilever": alpha += 1.0; context_tags.append("Hanging")
        elif collapse == "V-Shape": alpha += 0.3; context_tags.append("Void_Space")
 
        # 4. AGE & 5. COLUMNS (Same as before)
        try:
            age = int(age_years)
            if age < 10: alpha -= 0.1
            elif age > 50: alpha += 0.4; context_tags.append("Degraded")
        except: pass
 
        if col_condition == "Few": alpha += 0.4
        elif col_condition == "Buckled": alpha += 0.5; context_tags.append("Compromised_Support")
 
        # 6. SLENDERNESS
        if int(storeys) >= 4: 
            alpha += 0.4; self.LIMITS['tilt'] = 0.5; context_tags.append("High_Rise")
 
        return round(alpha, 2), context_tags
 
    def get_dynamic_weights(self, context_tags):
        weights = {'kinetic': 0.4, 'env': 0.3, 'geom': 0.3}
 
        # Priority Shift for Top-Heavy Structures
        if "CRITICAL_TOP_HEAVY" in context_tags:
            # Vibration is extremely dangerous for heavy roofs on weak walls
            weights = {'kinetic': 0.5, 'env': 0.2, 'geom': 0.3}
        elif "Hanging" in context_tags:
            weights = {'kinetic': 0.3, 'env': 0.2, 'geom': 0.5}
        
        return weights
 
    def predict(self, data, alpha, context_tags):
        # Standard Prediction Logic (Same as before)
        eff_vib = data['vib_mm_s'] * alpha
        weights = self.get_dynamic_weights(context_tags)
        
        # Severity Calculations
        sev_vib = min(eff_vib / self.LIMITS['vib'], 1.0)
        sev_tilt = min(data['tilt_deg'] / self.LIMITS['tilt'], 1.0)
        risk_kin = max(sev_vib, sev_tilt)
 
        sev_gas = min(data['gas_drop'] / self.LIMITS['gas'], 1.0)
        sev_temp = 1.0 if data['temp_c'] > self.LIMITS['temp'] else 0.0
        risk_env = max(sev_gas, sev_temp)
 
        sev_crack = min(data['crack_mm'] / self.LIMITS['crack'], 1.0)
        if data['support_ratio'] <= 0.5: sev_sup = 1.0
        elif data['support_ratio'] >= 1.0: sev_sup = 0.0
        else: sev_sup = (1.0 - data['support_ratio']) * 2
        risk_geom = max(sev_crack, sev_sup)
 
        # Score
        penalty = (risk_kin * weights['kinetic'] * 100) + \
                  (risk_env * weights['env'] * 100) + \
                  (risk_geom * weights['geom'] * 100)
        score = max(0, 100 - int(penalty))
 
        # Status
        if score >= 80: status = "STABLE"
        elif score >= 50: status = "SLIGHTLY STABLE"
        elif score >= 25: status = "UNSTABLE"
        else: status = "CRITICAL"
 
        # Veto & Advice
        veto = ""
        if eff_vib > self.LIMITS['vib']: veto = "VIBRATION LIMIT"
        if data['gas_drop'] > self.LIMITS['gas']: veto = "TOXIC GAS"
        if data['temp_c'] > self.LIMITS['temp']: veto = "FIRE DETECTED"
        if data['support_ratio'] < self.LIMITS['support']: veto = "COLLAPSE RISK"
 
        if veto: status = f"EVACUATE ({veto})"; score = 0
 
        advice = self.generate_advice(risk_kin, risk_env, risk_geom, data, eff_vib, context_tags)
 
        # --- LBC & S_final Integration ---
        # Determining Sector Color and Band based on Score
        if score >= 80:
            lbc_band = "Zone I (High Capacity)"
            sector_color = "GREEN"
        elif score >= 50:
            lbc_band = "Zone II (Reduced Capacity)"
            sector_color = "YELLOW"
        elif score >= 25:
            lbc_band = "Zone III (Low Capacity)"
            sector_color = "ORANGE"
        else:
            lbc_band = "Zone IV (Structural Failure)"
            sector_color = "RED"
 
        lbc_info = {
            'lbc_band': lbc_band,
            'S_final': score,
            'residual_capacity': score / 100.0,
            'intermediate_collapse_hazard': (100.0 - score) / 10.0,
            'V_dynamic': round(eff_vib, 2),
            'penalty_score': int(penalty),
            'sector_color': sector_color
        }
 
        return status, eff_vib, score, advice, lbc_info
 
    def generate_advice(self, r_kin, r_env, r_geom, data, eff_vib, tags):
        # Advice tailored to the Wall/Roof combo
        if "CRITICAL_TOP_HEAVY" in tags and r_kin > 0.4:
            return "HEAVY ROOF RISK: Weak walls cannot support the slab. Any vibration is fatal. EXIT NOW."
        
        # Standard Advice Logic
        threats = {'Kinetic': r_kin, 'Environmental': r_env, 'Geometric': r_geom}
        dom = max(threats, key=threats.get)
        if threats[dom] < 0.2: return "Structure stable. Proceed."
 
        if dom == 'Kinetic': return f"Resonance ({eff_vib:.1f} mm/s). Halt heavy tools."
        if dom == 'Environmental': return "Atmosphere toxic. Use BA Sets."
        if dom == 'Geometric': return "Structural shifting detected. Check shoring."
        return "Exercise Caution."