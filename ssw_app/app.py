import streamlit as st
import pandas as pd
import numpy as np
from netCDF4 import Dataset, num2date
from ssw_tools import *
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="SD4SP: SSW Diagnostic Tool",
    page_icon="🌪️",
    layout="wide"
)

# --- DATA LOADING ---
@st.cache_data
def load_numpy_data():
    with Dataset('nl_zm_ua10_day_ERA5_1950-2021.nc', mode='r') as nc:
        # Use [:] to load data into memory as numpy arrays
        u_wind = nc.variables['ua'][:] 
        levels = nc.variables['level'][:]
        lats   = nc.variables['lat'][:]
        lons   = nc.variables['lon'][:]
        time   = nc.variables['time'][:] 
        
    return {
        "u": u_wind,
        "lev": levels,
        "lat": lats,
        "lon": lons,
        "time": time
    }

# --- INTERFACE ---
st.title("🌪️ Stratospheric Sudden Warming (SSW) Tool")
st.markdown("Developed for the **SD4SP Project** | Analyzing Stratosphere-Troposphere Coupling")
#    ------------------------
st.write("### 🔍 Diagnóstico de archivos:")
st.write(f"Directorio actual: {os.getcwd()}")
st.write(f"Contenido de la carpeta actual: {os.listdir('.')}")

if os.path.exists('ssw_app'):
    st.write(f"Contenido de ssw_app/: {os.listdir('ssw_app')}")
else:
    st.write("❌ No encuentro la carpeta 'ssw_app'")
#    ------------------



# Load data once
data = load_numpy_data()

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("Configuration")
    
    definition = st.selectbox(
        "SSW Definition Criterion (U5570_11M is recommended*):",
        options=["U5570_11M", "U5570", "U60", "U65"],
        index=0
    )

    var_name = st.selectbox(
        "Surface Impact Variable:",
        options=["Temperature (2mt)", "Precipitation (tp)", "Pressure (psl)"]
    )
    
    st.divider()
    run_detection = st.button("Detect SSW Events", type="primary")

# --- DETECTION LOGIC ---
if run_detection:
    with st.spinner("Detecting SSW events..."):
        # Ensure 'root' is defined or passed correctly inside detect_ese
        year0 = 1950
        root = Dataset("nl_zm_ua_day_ERA5_1950-2021.nc") # Or your specific path
        
        # Calling your custom function
        ssw212, ssw365, sfw212, sfw365, nyr, perc = detect_ese(root, "_"+definition, year0)
        
        # Convert numeric dates to datetime objects
        raw_dates = num2date(ssw365, units=f"days since {year0}-01-01", calendar='noleap')
        
        # Store as standard python datetimes in session state
        st.session_state['ssw_dates'] = [pd.to_datetime(str(d)) for d in raw_dates]

# --- RESULTS & PLOTTING ---
if 'ssw_dates' in st.session_state:
    dates = st.session_state['ssw_dates']
    
    if len(dates) > 0:
        col1, col2 = st.columns([1, 2.5])
        
        with col1:
            st.subheader("📅 Event Selection")
            
            map_options = ["Full Composite"] + [d.strftime('%Y-%m-%d') for d in dates]
            
            selection = st.selectbox(
                "Choose event to plot:",
                options=map_options,
                index=0
            )
            
            st.divider()
            st.write(f"**Total events:** {len(dates)}")
            st.dataframe(pd.DataFrame(dates, columns=["Date"]), height=300)

        with col2:
            tab1, tab2 = st.tabs(["Downward Propagation", "Surface Impacts"])
            
            # Setup inputs for functions
            if selection == "Full Composite":
                event_input = dates
                is_composite = True
            else:
                event_input = [pd.to_datetime(selection)]
                is_composite = False

            with tab1:
                st.subheader(f"Vertical Propagation: {selection}")
                with st.spinner("Calculating vertical cross-section..."):
                    # Pass 'data' dictionary so function has access to numpy arrays
                    print("bef",event_input)
                    fig1 = plot_propagation(event_input, composite=is_composite)
                    st.pyplot(fig1)
                    
            with tab2:
                st.subheader(f"Surface Anomaly Map: {selection}")
                with st.spinner("Calculating surface impacts..."):
                    # Pass 'data' dictionary and chosen variable
                    fig2 = plot_impacts(event_input, var_name, composite=is_composite)
                    st.pyplot(fig2)
    else:
        st.warning("No events detected for the selected criteria.")
