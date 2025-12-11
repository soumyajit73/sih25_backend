import os
import sys
import json
import math
from flask import Flask, request, jsonify
 
# --- CONFIGURATION & PATH SETUP ---
# Ensure these match your actual folder structure
sys.path.append(os.path.abspath("Mock_Aerodrop/modules"))
sys.path.append(os.path.abspath("3D data extraction"))
 
try:
    from analyzer import StructuralBrain
    from unified_inspector import fallback_pipeline 
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import modules.\n{e}")
    sys.exit(1)
 
app = Flask(__name__)
 
# Folder where you put your .ply files
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
 
class PipelineService:
    def __init__(self):
        self.brain = StructuralBrain()
 
    def parse_pod_packet(self, raw_string, gas_baseline=100.0):
        """
        Parses the 7-Field Arduino String:
        Input: "TX: POD1,1,13.36,0.58,0.037,63.4,32.1"
        Mapping:
          [2] Roll   -> Tilt
          [3] Pitch  -> Tilt
          [4] Vib(G) -> Vib(mm/s)
          [5] Gas    -> Risk %
          [6] Temp   -> Value (New!)
        """
        try:
            clean = raw_string.strip().replace("TX: ", "")
            parts = clean.split(',')
            
            # Need at least 6 fields (Old format)
            if len(parts) < 6: return None
 
            # --- EXTRACT RAW VALUES ---
            roll = float(parts[2])
            pitch = float(parts[3])
            raw_vib_g = float(parts[4])
            gas_kohm = float(parts[5])
            
            # CHECK FOR 7th FIELD (TEMPERATURE)
            if len(parts) >= 7:
                temp_c = float(parts[6]) # Use real value from Pod
            else:
                temp_c = 25.0            # Fallback if missing
 
            # --- CONVERSIONS ---
            # 1. TILT (Max deviation)
            tilt_deg = max(abs(roll), abs(pitch))
            
            # 2. VIBRATION (G -> mm/s @ 15Hz)
            accel_mm_s2 = raw_vib_g * 9806.65
            vib_mm_s = accel_mm_s2 / (2 * 3.14159 * 15)
 
            # 3. GAS (Drop %)
            gas_drop = 0.0
            if gas_baseline > 0:
                gas_drop = max(0.0, (gas_baseline - gas_kohm) / gas_baseline)
 
            return {
                "vib_mm_s": round(vib_mm_s, 2),
                "tilt_deg": round(tilt_deg, 2),
                "gas_drop": round(gas_drop, 2),
                "temp_c": round(temp_c, 1)
            }
        except Exception as e:
            print(f"Packet Parse Error: {e}")
            return None
 
    def process_request(self, ply_path, manual_inputs, raw_packet):
        # --- STEP 1: STATIC 3D ANALYSIS (.ply) ---
        # Opens the file and calculates geometry metrics
        print(f"[API] Analyzing 3D File: {ply_path}")
        scan_report = fallback_pipeline(ply_path, voxel_size=0.02)
        
        # A. Get Metrics
        static_tilt = scan_report.get('max_tilt_degrees', 0.0)
        buckling_cm = scan_report.get('deflection', {}).get('measured_deflection_mm_98pct', 0.0) / 10.0
        
        # B. Calculate Support Ratio (Points on Wall / Total Points)
        #
        try:
            raw_pts = scan_report.get('cleaned_points', 1)
            wall_pts = scan_report.get('plane_inliers', 1)
            static_support_ratio = min(1.0, max(0.0, float(wall_pts)/float(raw_pts)))
        except:
            static_support_ratio = 1.0
 
        # C. Calculate Crack Proxy (Surface Variance)
        #
        frac_critical = scan_report.get('normal_variation_stats', {}).get('frac_var_gt_0.18', 0.0)
        static_crack_mm = min(10.0, round(frac_critical * 6.0, 3))
 
        # --- STEP 2: LIVE POD PARSING ---
        live_data = self.parse_pod_packet(raw_packet)
        if not live_data:
            raise ValueError("Invalid Packet Format")
 
        # --- STEP 3: CONTEXT & FRAGILITY ---
        #
        alpha, tags = self.brain.get_alpha(
            wall_type=manual_inputs.get('wall_type', 'RCC'),
            roof_type=manual_inputs.get('roof_type', 'Concrete'),
            collapse=manual_inputs.get('collapse_pattern', 'Pancake'),
            age_years=manual_inputs.get('age', 20),
            col_condition=manual_inputs.get('columns', 'Many'),
            storeys=manual_inputs.get('floors', 2)
        )
 
        # Apply Static Penalties
        scan_penalties = []
        if static_tilt > 5.0:
            alpha += 0.2
            tags.append("Severe_Tilt")
            scan_penalties.append(f"Static Tilt {static_tilt:.1f} (>5.0)")
        if buckling_cm > 10.0:
            alpha += 0.3
            tags.append("Buckled_Walls")
            scan_penalties.append(f"Buckling {buckling_cm:.1f}cm (>10.0)")
 
        # --- STEP 4: PREDICTION ---
        # MERGE: Live Data + Static 3D Data
        combined_data = {
            'vib_mm_s': live_data['vib_mm_s'],       # From Pod
            'tilt_deg': live_data['tilt_deg'],       # From Pod (Live)
            'gas_drop': live_data['gas_drop'],       # From Pod
            'temp_c': live_data['temp_c'],           # From Pod (Now 32.1)
            'crack_mm': static_crack_mm,             # From .ply
            'support_ratio': static_support_ratio    # From .ply
        }
 
        status, eff_vib, score, advice, lbc_info = self.brain.predict(combined_data, alpha, tags)
 
        return {
            "status": status,
            "final_score": score,
            "advice": advice,
            "fragility_factor_alpha": round(alpha, 2),
            "breakdown": {
                "static_metrics": {
                    "scan_tilt_deg": round(static_tilt, 2),
                    "scan_buckling_cm": round(buckling_cm, 2),
                    "calc_crack_mm": static_crack_mm,
                    "calc_support_ratio": round(static_support_ratio, 2),
                    "penalties": scan_penalties
                },
                "live_metrics": live_data,
                "lbc_zone": lbc_info['lbc_band'],
                "residual_capacity": lbc_info['residual_capacity']
            }
        }
 
pipeline = PipelineService()
 
@app.route('/api/assess', methods=['POST'])
def assess_structure():
    """
    Endpoint: 
    1. Looks for 'filename' in 'uploads/'
    2. Takes 'packet' string
    3. Takes 'manual_config' JSON
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    
    # Extract Inputs
    packet = data.get('packet')       # "TX: POD1,..."
    filename = data.get('filename')   # "cloud.ply"
    manual_config = data.get('manual_config') # { "wall_type":... }
 
    if not all([packet, filename, manual_config]):
        return jsonify({"error": "Missing packet, filename, or manual_config"}), 400
 
    # Locate File
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.isfile(filepath):
        return jsonify({"error": f"File '{filename}' not found in {UPLOAD_FOLDER}"}), 404
 
    try:
        # Run Pipeline
        result = pipeline.process_request(filepath, manual_config, packet)
        
        # Optional: Clean up side-effect files from unified_inspector
        for junk in ["cleaned_cloud.ply", "wall_inliers.ply", "unified_report.json"]:
            if os.path.exists(junk):
                os.remove(junk)
 
        return jsonify(result)
 
    except Exception as e:
        return jsonify({"error": str(e)}), 500
 
if __name__ == '__main__':
    app.run(debug=True, port=5000)