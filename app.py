import os
import sys
import json
import math
from flask import Flask, request, jsonify
 
# --- CONFIGURATION & PATH SETUP ---
sys.path.append(os.path.abspath("Mock_Aerodrop/modules"))
sys.path.append(os.path.abspath("3D data extraction"))
 
try:
    from analyzer import StructuralBrain
    # We use the fallback_pipeline (SciPy) as the standard processor 
    # because it is lightweight and robust for servers.
    from unified_inspector import fallback_pipeline 
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import modules.\n{e}")
    sys.exit(1)
 
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
 
class PipelineService:
    def __init__(self):
        self.brain = StructuralBrain()
 
    def parse_pod_packet(self, raw_string, gas_baseline=100.0):
        """
        STRICT PARSER: Expects exactly 7 fields.
        Format: "TX: POD_ID, State, Roll, Pitch, Vib(G), Gas, Temp"
        Example: "TX: POD2,1,13.36,0.58,0.037,63.4,32.1"
        """
        try:
            clean = raw_string.strip().replace("TX: ", "")
            parts = clean.split(',')
            
            # STRICT CHECK: Must have at least 7 fields (ID...Temp)
            if len(parts) < 7: 
                print(f"[ERROR] Packet too short. Received {len(parts)} fields, expected 7.")
                return None
 
            # --- 1. IDENTIFICATION ---
            pod_id = parts[0]  # Capture dynamic ID (e.g., POD2)
 
            # --- 2. RAW VALUES (Guaranteed to exist) ---
            roll = float(parts[2])
            pitch = float(parts[3])
            raw_vib_g = float(parts[4])
            gas_kohm = float(parts[5])
            temp_c = float(parts[6])  # No fallback: Always read index 6
 
            # --- 3. CONVERSIONS ---
            # Tilt (Max deviation)
            tilt_deg = max(abs(roll), abs(pitch))
            
            # Vibration (G -> mm/s @ 15Hz)
            accel_mm_s2 = raw_vib_g * 9806.65
            vib_mm_s = accel_mm_s2 / (2 * 3.14159 * 15)
 
            # Gas (Drop %)
            gas_drop = 0.0
            if gas_baseline > 0:
                gas_drop = max(0.0, (gas_baseline - gas_kohm) / gas_baseline)
 
            return {
                "pod_id": pod_id,
                "vib_mm_s": round(vib_mm_s, 2),
                "tilt_deg": round(tilt_deg, 2),
                "gas_drop": round(gas_drop, 2),
                "temp_c": round(temp_c, 1)
            }
        except Exception as e:
            print(f"Packet Parse Error: {e}")
            return None
 
    def process_request(self, ply_path, manual_inputs, raw_packet):
        # --- STEP 1: STATIC 3D ANALYSIS ---
        print(f"[API] Analyzing 3D File: {ply_path}")
        scan_report = fallback_pipeline(ply_path, voxel_size=0.02)
        
        # Extract Static Metrics
        static_tilt = scan_report.get('max_tilt_degrees', 0.0)
        buckling_cm = scan_report.get('deflection', {}).get('measured_deflection_mm_98pct', 0.0) / 10.0
        
        # Calculate Support Ratio
        #
        try:
            raw_pts = scan_report.get('cleaned_points', 1)
            wall_pts = scan_report.get('plane_inliers', 1)
            static_support_ratio = min(1.0, max(0.0, float(wall_pts)/float(raw_pts)))
        except:
            static_support_ratio = 1.0
 
        # Calculate Crack Proxy
        #
        frac_critical = scan_report.get('normal_variation_stats', {}).get('frac_var_gt_0.18', 0.0)
        static_crack_mm = min(10.0, round(frac_critical * 6.0, 3))
 
        # --- STEP 2: LIVE POD PARSING ---
        live_data = self.parse_pod_packet(raw_packet)
        if not live_data:
            raise ValueError("Invalid Packet Format: Must contain 7 fields (incl. Temp)")
 
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
        combined_data = {
            'vib_mm_s': live_data['vib_mm_s'],
            'tilt_deg': live_data['tilt_deg'],
            'gas_drop': live_data['gas_drop'],
            'temp_c': live_data['temp_c'],           # Guaranteed Real Value
            'crack_mm': static_crack_mm,
            'support_ratio': static_support_ratio
        }
 
        status, eff_vib, score, advice, lbc_info = self.brain.predict(combined_data, alpha, tags)
 
        return {
            "source_pod_id": live_data['pod_id'], # Returns the ID (e.g., POD2)
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
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    packet = data.get('packet')
    filename = data.get('filename')
    manual_config = data.get('manual_config')
 
    # Strict Input Check
    if not all([packet, filename, manual_config]):
        return jsonify({"error": "Missing packet, filename, or manual_config"}), 400
 
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.isfile(filepath):
        return jsonify({"error": f"File '{filename}' not found in {UPLOAD_FOLDER}"}), 404
 
    try:
        result = pipeline.process_request(filepath, manual_config, packet)
        
        # Clean up side-effect files
        for junk in ["cleaned_cloud.ply", "wall_inliers.ply", "unified_report.json"]:
            if os.path.exists(junk):
                os.remove(junk)
 
        return jsonify(result)
 
    except Exception as e:
        return jsonify({"error": str(e)}), 500
 
if __name__ == '__main__':
    app.run(debug=True, port=5000)