import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
import math
from scipy.signal import savgol_filter
import warnings
import io

warnings.filterwarnings('ignore')

# Set font for displaying charts
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

title = "Corrugated Steel Tube Confined Rubber Concrete Pier Axial Compression Capacity Prediction System"

st.set_page_config(    
    page_title=f"{title}",
    page_icon="⭕",
    layout="wide"
)

def load_model_from_upload(uploaded_file):
    """Load model from uploaded file"""
    try:
        # Read uploaded file content into memory
        model = joblib.load(io.BytesIO(uploaded_file.read()))
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

def apply_physical_constraint(strain_values, predictions, tolerance=1e-6):
    """
    Apply physical constraint: when strain ≈ 0, load = 0
    
    Parameters:
    strain_values: array of strain values
    predictions: array of predicted load values
    tolerance: tolerance for determining if strain is close to 0
    
    Returns:
    adjusted_predictions: predictions with physical constraint applied
    constraint_applied: boolean array indicating where constraint was applied
    """
    adjusted_predictions = predictions.copy()
    constraint_applied = np.isclose(strain_values, 0, atol=tolerance)
    adjusted_predictions[constraint_applied] = 0
    
    return adjusted_predictions, constraint_applied

st.markdown(f'''
    <h1 style="font-size: 28px; text-align: center; color: #2E86AB; border-bottom: 3px solid #2E86AB; margin-bottom: 2rem; padding-bottom: 10px;">
    🏗️ {title}
    </h1>''', unsafe_allow_html=True)
    
st.markdown('''
    <style>
    .stMainBlockContainer {
        padding-top: 32px;
        padding-bottom: 32px;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        border-left: 4px solid #2E86AB;
    }
    .upload-container {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border: 2px dashed #2E86AB;
        margin-bottom: 20px;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .constraint-box {
        background-color: #e8f5e8;
        border: 1px solid #4caf50;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    </style>''', unsafe_allow_html=True)

# Model file upload area
st.markdown('<div class="upload-container">', unsafe_allow_html=True)
st.markdown("### 📁 Model File Upload")
st.markdown("Please upload the trained model files:")

col_upload1, col_upload2 = st.columns(2)

with col_upload1:
    voting_file = st.file_uploader(
        "Upload Voting Ensemble Model File", 
        type=['pkl'], 
        key="voting_model",
        help="Please upload voting_regressor.pkl file"
    )

with col_upload2:
    scaler_file = st.file_uploader(
        "Upload Y-value Scaler File", 
        type=['pkl'], 
        key="scaler_y",
        help="Please upload scaler_y.pkl file"
    )

st.markdown('</div>', unsafe_allow_html=True)

# Check if files are uploaded
if voting_file is None or scaler_file is None:
    st.warning("⚠️ Please upload the required model files before making predictions")
    st.markdown("""
    ### 📋 Required Files:
    1. **voting_regressor.pkl** - Voting ensemble regression model
    2. **scaler_y.pkl** - Y-value normalizer
    
    ### 🔧 How to obtain these files:
    - After running your provided training code, these files will be automatically saved in the training script directory
    - Make sure to find these two .pkl files after training completion and upload them
    
    ### ⚠️ Feature Names Description:
    Model expects space-separated feature names:
    - ['D', 't', 'L/D', 'Helix Angle', 'fcuAc', 'fyAs', 'h/l', 'Concrete Types', 'Confinement Factor', 'Strain']
    """)
    st.stop()

# Load models
with st.spinner("Loading models..."):
    voting_model = load_model_from_upload(voting_file)
    scaler_y = load_model_from_upload(scaler_file)

if voting_model is None or scaler_y is None:
    st.error("❌ Model loading failed, please check file format")
    st.stop()

st.success("✅ Models loaded successfully!")

# Display expected feature names
try:
    expected_features = voting_model.feature_names_in_
    st.info(f"🔍 Model expected feature names: {list(expected_features)}")
except:
    st.warning("⚠️ Cannot retrieve model expected feature names, will use space-separated feature names")

# Model information display
st.markdown("""
### 🤖 Model Information
- **Model Type**: Voting Ensemble Regressor (XGBoost + LightGBM + CatBoost)
- **Weight Configuration**: XGBoost(0.4) + LightGBM(0.3) + CatBoost(0.3)
- **Optimization Method**: Bayesian Hyperparameter Optimization
- **Input Features**: 10 parameters (space-separated English feature names)
- **Output**: Nu (Axial Compression Capacity, kN)
- **Physical Constraint**: Strain = 0 → Load = 0 (automatically applied)
""")

# Physical constraint settings
st.markdown("### ⚖️ Physical Constraint Settings")
constraint_col1, constraint_col2 = st.columns(2)

with constraint_col1:
    apply_constraint = st.checkbox(
        "Apply Physical Constraint (Strain=0 → Load=0)", 
        value=True,
        help="When enabled, predictions at strain≈0 will be forced to 0 to satisfy physical laws"
    )
    
with constraint_col2:
    if apply_constraint:
        constraint_tolerance = st.number_input(
            "Strain Tolerance for Zero Constraint", 
            value=1e-6, 
            format="%.2e",
            help="Strain values within this tolerance of 0 will be set to 0 load"
        )
    else:
        constraint_tolerance = 1e-6

if apply_constraint:
    st.markdown(f'''
    <div class="constraint-box">
    ⚖️ <strong>Physical Constraint Active</strong>: When strain ≤ {constraint_tolerance:.2e}, the predicted load will be set to 0.<br>
    This ensures the prediction results comply with the physical law that no load exists without deformation.
    </div>
    ''', unsafe_allow_html=True)

# Parameter input interface
st.markdown("### 📊 Model Parameter Input")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🔧 Geometric Parameters**")
    D = st.number_input("D - Diameter (mm)", value=315.0, format="%.2f", help="Steel tube outer diameter")
    t = st.number_input("t - Wall Thickness (mm)", value=2.0, format="%.2f", help="Steel tube wall thickness")
    L = st.number_input("L - Length (mm)", value=750.0, format="%.2f", help="Total component length")
    h = st.number_input("h - Corrugation Height (mm)", value=25.0, format="%.2f", help="Corrugation ridge height")
    l = st.number_input("l - Corrugation Spacing (mm)", value=75.0, format="%.2f", help="Distance between adjacent corrugations")

with col2:
    st.markdown("**🧱 Material Parameters**")
    fcu = st.number_input("fcu - Concrete Cube Compressive Strength (MPa)", value=40.0, format="%.2f")
    fy = st.number_input("fy - Steel Yield Strength (MPa)", value=235.0, format="%.2f")
    # Changed to rubber replacement ratio
    rubber_replacement_ratio = st.number_input("Rubber Replacement Ratio", 
                                              value=0.5, 
                                              min_value=0.0, 
                                              max_value=1.0, 
                                              format="%.3f",
                                              help="Volume ratio of rubber particles replacing concrete aggregate, range 0-1")
    K = st.number_input("K - Wave Number", value=10.0, format="%.2f", help="Total number of spiral corrugations")

with col3:
    st.markdown("**📈 Strain Range Settings**")
    start_strain = st.number_input("Start Strain", value=0.000, format="%.6f", help="Initial strain value")
    end_strain = st.number_input("End Strain", value=0.030, format="%.6f", help="Final strain value")
    N_points = st.number_input("Number of Points", value=100, min_value=10, max_value=1000, help="Number of strain sequence points")

# Data validation
if start_strain >= end_strain:
    st.error("❌ Start strain must be less than end strain!")
    st.stop()

if any(val <= 0 for val in [D, t, L, fcu, fy]):
    st.error("❌ Geometric parameters and material strengths must be greater than 0!")
    st.stop()

# Check wall thickness reasonableness
if t >= D/2:
    st.error("❌ Wall thickness cannot be greater than or equal to half the diameter!")
    st.stop()

# Calculate derived parameters
st.markdown("### 🔧 Calculated Derived Parameters")

# Basic geometric calculations
L_D_ratio = L / D if D != 0 else 0

# Helix angle calculation
if K != 0 and l != 0:
    B = l * K  # Spiral circumference
    ratio = B / (math.pi * D) if D != 0 else 0
    helix_angle = math.degrees(math.atan(ratio))
else:
    helix_angle = 0.0

# h/l ratio
h_l_ratio = h / l if l != 0 else 0

# Amplification factor (determined by h/l ratio)
h_l_tolerance = 0.01  # Tolerance
if abs(h_l_ratio - 13/68) < h_l_tolerance:
    amplification_factor = 1.084
elif abs(h_l_ratio - 25/75) < h_l_tolerance:
    amplification_factor = 1.247
elif abs(h_l_ratio - 25/125) < h_l_tolerance:
    amplification_factor = 1.07
elif abs(h_l_ratio - 50/150) < h_l_tolerance:
    amplification_factor = 1.179
elif abs(h_l_ratio - 55/200) < h_l_tolerance:
    amplification_factor = 1.142
elif h == 0:
    amplification_factor = 1.0
else:
    amplification_factor = 1.0
    st.markdown(f'''
    <div class="warning-box">
    ⚠️ <strong>Warning</strong>: h/l ratio {h_l_ratio:.3f} is not within preset range, using default amplification factor 1.0<br>
    Preset ranges: 13/68≈0.191, 25/75≈0.333, 25/125=0.200, 50/150≈0.333, 55/200=0.275
    </div>
    ''', unsafe_allow_html=True)

# Cross-sectional area calculations (corrected version)
Ac = math.pi * (D - t) **2 / 4  # Concrete cross-sectional area (mm²) - using inner diameter
As = math.pi * D * t * amplification_factor    # Steel cross-sectional area (mm²) - including amplification factor

# Composite parameters - corrected unit to N·mm²
fcuAc = fcu * Ac  # N·mm²
fyAs = fy * As    # N·mm²

# Steel ratio
steel_ratio = As / Ac if Ac != 0 else 0

# Confinement factor
confinement_factor = steel_ratio * fy / fcu if fcu != 0 else 0

# Display calculation results - corrected to 4-column layout to show more parameters
param_col1, param_col2, param_col3, param_col4 = st.columns(4)

with param_col1:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric("L/D Length-Diameter Ratio", f"{L_D_ratio:.3f}")
    st.metric("Helix Angle (°)", f"{helix_angle:.2f}")
    st.markdown('</div>', unsafe_allow_html=True)

with param_col2:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric("Concrete Area Ac (mm²)", f"{Ac:.0f}")
    st.metric("Steel Area As (mm²)", f"{As:.0f}")
    st.markdown('</div>', unsafe_allow_html=True)

with param_col3:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric("fcuAc (N·mm²)", f"{fcuAc:.0f}")
    st.metric("fyAs (N·mm²)", f"{fyAs:.0f}")
    st.markdown('</div>', unsafe_allow_html=True)

with param_col4:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric("h/l Corrugation Ratio", f"{h_l_ratio:.3f}")
    st.metric("Amplification Factor", f"{amplification_factor:.3f}")
    st.metric("Steel Ratio", f"{steel_ratio:.4f}")
    st.metric("Confinement Factor", f"{confinement_factor:.3f}")
    st.markdown('</div>', unsafe_allow_html=True)

# Generate strain sequence
strain_values = np.linspace(start_strain, end_strain, N_points)

# Prepare model input data (using space-separated feature names)
X_input = []
for strain in strain_values:
    # According to model expected feature name order (space-separated)
    row = [
        D,                          # D
        t,                          # t  
        L_D_ratio,                 # L/D
        helix_angle,               # Helix Angle
        fcuAc,                     # fcuAc
        fyAs,                      # fyAs
        h_l_ratio,                 # h/l
        rubber_replacement_ratio,   # Concrete Types (rubber replacement ratio)
        confinement_factor,        # Confinement Factor
        strain                     # Strain
    ]
    X_input.append(row)

# Feature names (using space-separated, completely consistent with training model)
feature_names = [
    "D", "t", "L/D", "Helix Angle", "fcuAc", "fyAs", 
    "h/l", "Concrete Types", "Confinement Factor", "Strain"
]

# Prediction and visualization
st.markdown("### 📈 Prediction Results")

# Create DataFrame
df_input = pd.DataFrame(X_input, columns=feature_names)

# Display first few rows of input data for debugging
with st.expander("🔍 View Input Data (for debugging)", expanded=False):
    st.write("**First 5 rows of input data:**")
    st.dataframe(df_input.head())
    st.write("**Feature names we provided:**")
    st.write(list(df_input.columns))
    try:
        expected_features = voting_model.feature_names_in_
        st.write("**Model expected feature names:**")
        st.write(list(expected_features))
        
        # Check if feature names match
        if list(df_input.columns) == list(expected_features):
            st.success("✅ Feature names match perfectly!")
        else:
            st.error("❌ Feature names do not match!")
            st.write("**Difference comparison:**")
            for i, (provided, expected) in enumerate(zip(df_input.columns, expected_features)):
                if provided != expected:
                    st.write(f"Position {i}: provided '{provided}', expected '{expected}'")
    except:
        st.write("Cannot retrieve model expected feature names")

# Model prediction
try:
    with st.spinner("Making predictions..."):
        Nu_predicted_norm = voting_model.predict(df_input)
        # Inverse transform to get actual Nu values (kN)
        Nu_predicted_raw = scaler_y.inverse_transform(Nu_predicted_norm.reshape(-1, 1)).flatten()
        
        # Apply physical constraint if enabled
        if apply_constraint:
            Nu_predicted, constraint_applied = apply_physical_constraint(
                strain_values, Nu_predicted_raw, constraint_tolerance
            )
            n_constrained = np.sum(constraint_applied)
            
            if n_constrained > 0:
                st.markdown(f'''
                <div class="constraint-box">
                ⚖️ <strong>Physical Constraint Applied</strong>: {n_constrained} points with strain ≤ {constraint_tolerance:.2e} have been set to 0 load.
                </div>
                ''', unsafe_allow_html=True)
        else:
            Nu_predicted = Nu_predicted_raw
            constraint_applied = np.zeros_like(strain_values, dtype=bool)
        
        st.success(f"✅ Prediction completed! Generated {len(Nu_predicted)} prediction points")
            
except Exception as e:
    st.error(f"❌ Prediction process error: {e}")
    st.error("Please check if input parameters are within reasonable range, or if model files are correct")
    
    # Display more debugging information
    st.write("**Debugging information:**")
    st.write(f"Input data shape: {df_input.shape}")
    st.write(f"Input feature names: {list(df_input.columns)}")
    try:
        expected_features = voting_model.feature_names_in_
        st.write(f"Model expected feature names: {list(expected_features)}")
        st.write(f"Feature count match: {len(df_input.columns) == len(expected_features)}")
        st.write(f"Feature names match: {list(df_input.columns) == list(expected_features)}")
    except Exception as debug_e:
        st.write(f"Cannot retrieve model expected feature names: {debug_e}")
    st.stop()

# Create two-column layout
result_col1, result_col2 = st.columns([1, 2])

with result_col1:
    st.markdown("**📋 Data Table**")
    result_df = pd.DataFrame({
        "Strain": strain_values,
        "Raw Prediction (kN)": Nu_predicted_raw if apply_constraint else Nu_predicted,
        "Final Prediction (kN)": Nu_predicted,
        "Constraint Applied": constraint_applied if apply_constraint else [False] * len(strain_values)
    })
    
    # Only show constraint columns if constraint is applied
    if not apply_constraint:
        result_df = result_df[["Strain", "Final Prediction (kN)"]]
    
    st.dataframe(result_df, use_container_width=True, height=400)
    
    # Statistical information
    st.markdown("**📊 Statistical Information**")
    st.write(f"- Maximum capacity: {np.max(Nu_predicted):.2f} kN")
    st.write(f"- Minimum capacity: {np.min(Nu_predicted):.2f} kN")
    st.write(f"- Average capacity: {np.mean(Nu_predicted):.2f} kN")
    st.write(f"- Capacity range: {np.max(Nu_predicted) - np.min(Nu_predicted):.2f} kN")
    
    if apply_constraint:
        n_constrained = np.sum(constraint_applied)
        st.write(f"- Points with constraint applied: {n_constrained}")
        if n_constrained > 0:
            max_diff = np.max(np.abs(Nu_predicted_raw - Nu_predicted))
            st.write(f"- Maximum constraint adjustment: {max_diff:.2f} kN")
    
    # Download data
    csv = result_df.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        label="📥 Download Prediction Data (CSV)",
        data=csv,
        file_name=f"axial_capacity_prediction_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

with result_col2:
    st.markdown("**📊 Load-Strain Curve**")
    
    # Smoothing parameter settings
    smooth_col1, smooth_col2 = st.columns(2)
    with smooth_col1:
        apply_smoothing = st.checkbox("Apply Curve Smoothing", value=True)
        window_size = st.number_input("Window Size (odd number)", value=9, min_value=3, max_value=51, step=2, 
                                     help="Savitzky-Golay filter window size, must be odd",
                                     disabled=not apply_smoothing)
    with smooth_col2:
        poly_order = st.number_input("Polynomial Order", value=2, min_value=1, max_value=5,
                                    disabled=not apply_smoothing)
        show_original = st.checkbox("Show Original Curve", value=True)
        if apply_constraint:
            show_raw = st.checkbox("Show Raw Prediction (before constraint)", value=False)
        else:
            show_raw = False
    
    # Apply Savitzky-Golay filter for curve smoothing
    if apply_smoothing and len(Nu_predicted) >= window_size:
        try:
            smoothed_Nu = savgol_filter(Nu_predicted, window_size, poly_order)
        except Exception as e:
            st.warning(f"Smoothing failed: {e}")
            smoothed_Nu = Nu_predicted
    else:
        smoothed_Nu = Nu_predicted
        if apply_smoothing:
            st.warning("⚠️ Insufficient data points, cannot apply smoothing filter")
    
    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot raw prediction if requested
    if show_raw and apply_constraint:
        ax.plot(strain_values, Nu_predicted_raw, ':', alpha=0.5, color='gray', 
               label='Raw Prediction (before constraint)', linewidth=1.5)
    
    # Plot curves
    if show_original and apply_smoothing:
        ax.plot(strain_values, Nu_predicted, '--', alpha=0.6, color='lightblue', 
               label='Original Prediction (with constraint)', linewidth=1.5)
    
    line_color = 'red' if apply_smoothing else 'blue'
    line_label = 'Smoothed Prediction' if apply_smoothing else 'Final Prediction'
    ax.plot(strain_values, smoothed_Nu, '-', color=line_color, 
           label=line_label, linewidth=2.5)
    
    # Mark constraint points if any
    if apply_constraint and np.any(constraint_applied):
        constraint_strains = strain_values[constraint_applied]
        constraint_loads = Nu_predicted[constraint_applied]
        ax.scatter(constraint_strains, constraint_loads, color='green', s=60, 
                  marker='o', alpha=0.8, label='Constraint Applied', zorder=4)
    
    # Find peak point
    peak_idx = np.argmax(smoothed_Nu)
    peak_strain = strain_values[peak_idx]
    peak_Nu = smoothed_Nu[peak_idx]
    
    # Mark peak point
    ax.scatter(peak_strain, peak_Nu, color='black', s=120, zorder=5, 
              label='Peak Point', edgecolors='white', linewidth=2)
    
    # Peak point annotation
    offset_x = (end_strain - start_strain) * 0.15
    offset_y = (np.max(smoothed_Nu) - np.min(smoothed_Nu)) * 0.1
    ax.annotate(f'Peak: {peak_Nu:.2f} kN\nStrain: {peak_strain:.6f}',
                xy=(peak_strain, peak_Nu),
                xytext=(peak_strain + offset_x, peak_Nu + offset_y),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=11,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="yellow", alpha=0.8, edgecolor='black'))
    
    # Chart beautification
    ax.set_xlabel('Strain', fontsize=14, fontweight='bold')
    ax.set_ylabel('Axial Compression Capacity Nu (kN)', fontsize=14, fontweight='bold')
    title_text = 'Axial Compression Capacity-Strain Curve Prediction'
    if apply_constraint:
        title_text += ' (with Physical Constraint)'
    ax.set_title(title_text, fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set chart style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    
    # Set axis range
    y_margin = (np.max(smoothed_Nu) - np.min(smoothed_Nu)) * 0.1
    ax.set_ylim(np.min(smoothed_Nu) - y_margin, np.max(smoothed_Nu) + y_margin)
    
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

# Results summary
st.markdown("### 🎯 Prediction Summary")
summary_col1, summary_col2, summary_col3 = st.columns(3)

with summary_col1:
    st.success(f"🏆 **Peak Axial Compression Capacity**\n{peak_Nu:.2f} kN")

with summary_col2:
    st.info(f"📍 **Peak Corresponding Strain**\n{peak_strain:.6f}")

with summary_col3:
    ultimate_strain_idx = int(len(strain_values) * 0.9)  # 90% position strain
    ultimate_Nu = smoothed_Nu[ultimate_strain_idx]
    st.warning(f"📈 **Ultimate State Capacity**\n{ultimate_Nu:.2f} kN")

# Physical constraint summary
if apply_constraint:
    st.markdown("### ⚖️ Physical Constraint Summary")
    constraint_summary_col1, constraint_summary_col2, constraint_summary_col3 = st.columns(3)
    
    with constraint_summary_col1:
        n_constrained = np.sum(constraint_applied)
        st.metric("Points with Constraint Applied", n_constrained)
    
    with constraint_summary_col2:
        if n_constrained > 0:
            max_adjustment = np.max(np.abs(Nu_predicted_raw - Nu_predicted))
            st.metric("Maximum Constraint Adjustment (kN)", f"{max_adjustment:.2f}")
        else:
            st.metric("Maximum Constraint Adjustment (kN)", "0.00")
    
    with constraint_summary_col3:
        st.metric("Constraint Tolerance", f"{constraint_tolerance:.2e}")

# Display detailed input parameter summary
with st.expander("📋 Detailed Input Parameter Summary", expanded=False):
    summary_data = {
        "Parameter Name": feature_names,
        "Value": [D, t, L_D_ratio, helix_angle, fcuAc, fyAs, h_l_ratio, rubber_replacement_ratio, confinement_factor, "Variable Sequence"],
        "Unit": ["mm", "mm", "-", "°", "N·mm²", "N·mm²", "-", "-", "-", "-"],
        "Description": [
            "Steel tube outer diameter", "Steel tube wall thickness", "Length-diameter ratio", "Helix angle", 
            "Concrete strength × area", "Steel strength × area", "Corrugation height ratio", 
            "Rubber replacement ratio (0-1)", "Confinement factor", "Strain variable"
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)

# Display cross-sectional area calculation details
with st.expander("📐 Cross-sectional Area Calculation Details", expanded=False):
    st.markdown("### Cross-sectional Area Calculation Formulas and Results")
    
    calc_col1, calc_col2 = st.columns(2)
    
    with calc_col1:
        st.markdown("**Concrete Cross-sectional Area Ac:**")
        st.latex(r"A_c = \frac{\pi \cdot (D-t)^2}{4}")
        st.write(f"= π × ({D:.1f} - {t:.1f})² / 4")
        st.write(f"= π × {(D-t):.1f}² / 4")
        st.write(f"= {Ac:.0f} mm²")
        
    with calc_col2:
        st.markdown("**Steel Cross-sectional Area As:**")
        st.latex(r"A_s = \pi \cdot D \cdot t \cdot \alpha")
        st.write(f"= π × {D:.1f} × {t:.1f} × {amplification_factor:.3f}")
        st.write(f"= {As:.0f} mm²")
        st.write(f"where α = {amplification_factor:.3f} (corrugation amplification factor)")
        
    st.markdown("**Composite Parameters:**")
    st.write(f"- fcuAc = {fcu:.1f} × {Ac:.0f} = {fcuAc:.0f} N·mm²")
    st.write(f"- fyAs = {fy:.1f} × {As:.0f} = {fyAs:.0f} N·mm²")
    st.write(f"- Steel ratio = As/Ac = {As:.0f}/{Ac:.0f} = {steel_ratio:.4f}")
    st.write(f"- Confinement factor = (As/Ac) × fy/fcu = {steel_ratio:.4f} × {fy:.1f}/{fcu:.1f} = {confinement_factor:.3f}")

# Add usage instructions
with st.expander("📖 Usage Instructions", expanded=False):
    st.markdown("""
    ### 🔧 Parameter Description
    
    **Geometric Parameters:**
    - **D**: Steel tube outer diameter, unit: mm
    - **t**: Steel tube wall thickness, unit: mm  
    - **L**: Total component length, unit: mm
    - **h**: Corrugation ridge height, unit: mm
    - **l**: Distance between adjacent corrugations, unit: mm
    - **K**: Total number of spiral corrugations
    
    **Material Parameters:**
    - **fcu**: Concrete cube compressive strength, unit: MPa
    - **fy**: Steel yield strength, unit: MPa
    - **Rubber Replacement Ratio**: Volume ratio of rubber particles replacing concrete aggregate, range 0-1
    
    **Strain Settings:**
    - **Start Strain**: Analysis initial strain value
    - **End Strain**: Analysis final strain value
    - **Number of Points**: Number of strain sequence points for analysis
    
    ### 📊 Model Features
    
    **Derived Parameters (automatically calculated):**
    - **L/D**: Length-diameter ratio
    - **Helix Angle**: Spiral corrugation angle (degrees)
    - **fcuAc**: Concrete strength × cross-sectional area (N·mm²)
    - **fyAs**: Steel strength × cross-sectional area (N·mm²)
    - **h/l**: Corrugation height ratio
    - **Confinement Factor**: Steel confinement coefficient
    
    **Amplification Factor Rules:**
    - h/l ≈ 0.191 (13/68): factor = 1.084
    - h/l ≈ 0.333 (25/75): factor = 1.247
    - h/l = 0.200 (25/125): factor = 1.07
    - h/l ≈ 0.333 (50/150): factor = 1.179
    - h/l = 0.275 (55/200): factor = 1.142
    - h = 0: factor = 1.0
    - Other cases: factor = 1.0
    
    ### 🎯 Output Results
    
    **Prediction Results:**
    - **Load-Strain Curve**: Complete axial compression capacity vs strain relationship
    - **Peak Capacity**: Maximum axial compression capacity and corresponding strain
    - **Ultimate State Capacity**: Capacity at 90% strain range
    - **Statistical Information**: Maximum, minimum, average, and range of capacity values
    
    **Data Export:**
    - CSV format download with strain and capacity data
    - Timestamped filename for easy management
    
    ### ⚠️ Important Notes
    
    **Parameter Constraints:**
    - All geometric parameters and material strengths must be > 0
    - Wall thickness must be < diameter/2
    - Start strain must be < end strain
    - Rubber replacement ratio must be between 0-1
    
    **Model Accuracy:**
    - Model trained on specific parameter ranges
    - Best accuracy within training data range
    - Extrapolation beyond training range may reduce accuracy
    
    **Curve Smoothing:**
    - Savitzky-Golay filter available for noise reduction
    - Adjustable window size and polynomial order
    - Option to display both original and smoothed curves
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 14px; margin-top: 2rem;">
🏗️ Corrugated Steel Tube Confined Rubber Concrete Pier Axial Compression Capacity Prediction System<br>
Powered by Machine Learning Ensemble Models (XGBoost + LightGBM + CatBoost)<br>
For research and engineering applications
</div>
""", unsafe_allow_html=True)