import streamlit as st
import pandas as pd
import numpy as np
from netCDF4 import Dataset, num2date
from ssw_tools import *
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="SD4SP: SSW Diagnostic Tool",
    layout="wide"
)

# --- DATA LOADING ---
@st.cache_data
def load_numpy_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'nl_zm_ua10_day_ERA5_1950-2021.nc')
    
    if not os.path.exists(file_path):
        st.error(f"File not found in: {file_path}")
        return None

    with Dataset(file_path, mode='r') as nc:
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
st.title("Stratospheric Sudden Warming (SSW) Tool")
st.markdown("Developed for the **SD4SP Project** | Analyzing Stratosphere-Troposphere Coupling")

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
        year0 = 1950
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, 'nl_zm_ua10_day_ERA5_1950-2021.nc')
        
        if not os.path.exists(file_path):
            st.error(f"File not found in: {file_path}")
        else:
            with Dataset(file_path, mode='r') as root:
                ssw212, ssw365, sfw212, sfw365, nyr, perc = detect_ese(root, "_"+definition, year0)
                raw_dates = num2date(ssw365, units=f"days since {year0}-01-01", calendar='noleap')
                # Save formatted strings
                st.session_state['ssw_dates'] = [pd.to_datetime(str(d)).strftime('%-d %b %Y') for d in raw_dates]

# --- RESULTS & PLOTTING ---
if 'ssw_dates' in st.session_state:
    dates = st.session_state['ssw_dates']
    
    if len(dates) > 0:
        col1, col2 = st.columns([1, 2.5])
        
        with col1:
            st.subheader("Event Selection")
            map_options = ["Full Composite"] + dates
            selection = st.selectbox("Choose event to plot:", options=map_options, index=0)
            
            st.divider()
            st.write(f"**Total events:** {len(dates)}")
            st.dataframe(pd.DataFrame(dates, columns=["Date"]), height=300)

        with col2:
            tab1, tab2 = st.tabs(["Downward Propagation", "Surface Impacts"])

            # Logic to determine if plotting a single date or a composite
            if selection == "Full Composite":
                input_data = dates
                is_composite = True
            else:
                input_data = selection # String format: "1 Jan 1958"
                is_composite = False

            with tab1:
                st.subheader(f"Vertical Propagation: {selection}")
                with st.spinner("Calculating vertical cross-section..."):
                    fig1 = plot_propagation(input_data, composite=is_composite)
                    if fig1 is not None:
                        st.pyplot(fig1)
                    else:
                        st.info("No data available to display for this selection.")
                                            
            with tab2:
                st.subheader(f"Surface Anomaly Map: {selection}")
                with st.spinner("Calculating surface impacts..."):
                    fig2 = plot_impacts(input_data, var_name, composite=is_composite)
                    if fig2 is not None:
                        st.pyplot(fig2)
    else:
        st.warning("No events detected for the selected criteria.")
