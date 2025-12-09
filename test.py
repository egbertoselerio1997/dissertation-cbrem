import qsdsan as qs
import matplotlib.pyplot as plt
import numpy as np

def run_dhs_simulation_refined_plotting(t_max=120, influent_sf=10.0):
    # --- 1. System Setup ---
    # Define components and model
    cmps = qs.processes.create_asm2d_cmps()
    qs.set_thermo(cmps)
    asm2d_model = qs.processes.ASM2d()
    
    # Operational Parameters
    total_height = 2.0   # meters
    total_volume = 10.0  # m3
    flow_rate = 50.0     # m3/d
    num_segments = 3
    recycle_ratio = 0 
    
    seg_height = total_height / num_segments
    seg_vol = total_volume / num_segments
    
    # High KLa for trickling filter (passive aeration)
    kla_value = 150.0 

    # --- 2. Influent Definition ---
    inf_concs = {
        'S_I': 30.0,    
        'S_F': influent_sf, 
        'S_A': 5.0,    
        'X_I': 51.2,    
        'X_S': 202.32,  
        'X_H': 28.17,   
        'X_PAO': 0.0,   
        'X_PP': 0.0,    
        'X_PHA': 0.0,   
        'X_AUT': 0.0,   
        'S_NH4': 31.56, 
        'S_N2': 0.0,    
        'S_NO3': 0.0,   
        'S_PO4': 5.0,   
        'S_ALK': 7.0,   
        'S_O2': 0.0     
    }

    # --- 3. Create Streams ---
    influent = qs.WasteStream('raw_influent', T=293.15)
    influent.set_flow_by_concentration(flow_rate, concentrations=inf_concs, units=('m3/d', 'mg/L'))
    
    recycle_stream = qs.WasteStream('recycle_stream', T=293.15)
    recycle_stream.set_flow_by_concentration(flow_rate * recycle_ratio, concentrations=inf_concs, units=('m3/d', 'mg/L'))
    
    mixed_influent = qs.WasteStream('mixed_influent')
    final_effluent = qs.WasteStream('final_effluent')

    # --- 4. Create Unit Operations ---
    M1 = qs.sanunits.Mixer('M1', ins=[influent, recycle_stream], outs=mixed_influent)
    
    segments = []
    current_input = M1-0 
    
    for i in range(num_segments):
        aer = qs.processes.DiffusedAeration(
            f'aer_seg_{i}', DO_ID='S_O2', KLa=kla_value, DOsat=8.0, V=seg_vol
        )
        
        seg = qs.sanunits.CSTR(
            f'Seg_{i}', ins=current_input, V_max=seg_vol, 
            aeration=aer, DO_ID='S_O2', suspended_growth_model=asm2d_model
        )
        
        seg.set_init_conc(**inf_concs)
        segments.append(seg)
        current_input = seg.outs[0]

    split_fraction = recycle_ratio / (1 + recycle_ratio)
    S1 = qs.sanunits.Splitter('S1', ins=segments[-1]-0, 
                              outs=[recycle_stream, final_effluent], 
                              split=split_fraction)

    # --- 5. System Definition and Simulation ---
    sys = qs.System('dhs_recycle_sys', path=[M1, *segments, S1], recycle=recycle_stream)
    
    print(f"Simulating for {t_max} days with Influent S_F = {influent_sf} mg/L...")
    sys.set_tolerance(rmol=1e-6)
    sys.simulate(t_span=(0, t_max), method='BDF', atol=1e-6, rtol=1e-6)
    print("Simulation complete.")

    # --- 6. Data Extraction ---
    target_solubles = ['S_F', 'S_NH4', 'S_NO3', 'S_O2']
    target_biomass = ['X_H', 'X_AUT', 'X_PAO']
    
    data = {cmp: [] for cmp in target_solubles + target_biomass}
    data['COD'] = []
    heights = []

    for i, seg in enumerate(segments):
        stream = seg.outs[0]
        current_height = (i + 1) * seg_height
        heights.append(current_height)
        
        if stream.F_vol > 0:
            for cmp in target_solubles + target_biomass:
                conc = (stream.imass[cmp] / stream.F_vol) * 1000 
                data[cmp].append(conc)
            data['COD'].append(stream.COD)
        else:
            for key in data:
                data[key].append(0)

    # --- 7. Refined Plotting ---
    # Setup styles for smaller text
    title_fs = 10
    label_fs = 9
    tick_fs = 8
    legend_fs = 8

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    
    # Plot 1: COD and Organic Substrate
    ax1.plot(heights, data['COD'], 'o-', label='Total COD', color='black', markersize=4)
    ax1.plot(heights, data['S_F'], 's--', label='Readily Biodegradable (S_F)', color='blue', markersize=4)
    ax1.set_xlabel('DHS Filter Depth (m)', fontsize=label_fs)
    ax1.set_ylabel('Concentration (mg/L)', fontsize=label_fs)
    ax1.set_title('Organic Carbon Profile', fontsize=title_fs)
    ax1.tick_params(axis='both', labelsize=tick_fs)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(fontsize=legend_fs)

    # Plot 2: Nitrogen and Oxygen
    ax2.plot(heights, data['S_NH4'], 'd-', label='Ammonium (S_NH4)', color='red', markersize=4)
    ax2.plot(heights, data['S_NO3'], '^--', label='Nitrate (S_NO3)', color='green', markersize=4)
    ax2.plot(heights, data['S_O2'], 'x:', label='Dissolved Oxygen', color='cyan', markersize=4)
    ax2.set_xlabel('DHS Filter Depth (m)', fontsize=label_fs)
    ax2.set_ylabel('Concentration (mg/L)', fontsize=label_fs)
    ax2.set_title('Nutrients & Oxygen', fontsize=title_fs)
    ax2.tick_params(axis='both', labelsize=tick_fs)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(fontsize=legend_fs)

    # Plot 3: Live Biomass Distributions (Dual Axis)
    # Primary Y-axis (Left): Heterotrophs
    line1 = ax3.plot(heights, data['X_H'], 'o-', label='Heterotrophs (X_H)', color='brown', markersize=4)
    ax3.set_xlabel('DHS Filter Depth (m)', fontsize=label_fs)
    ax3.set_ylabel('Heterotroph Conc. (mg/L)', fontsize=label_fs, color='brown')
    ax3.tick_params(axis='y', labelcolor='brown', labelsize=tick_fs)
    ax3.tick_params(axis='x', labelsize=tick_fs)
    ax3.set_title('Live Biomass Distribution', fontsize=title_fs)
    ax3.grid(True, linestyle=':', alpha=0.6)

    # Secondary Y-axis (Right): Autotrophs and PAOs
    ax3_twin = ax3.twinx()
    line2 = ax3_twin.plot(heights, data['X_AUT'], 's--', label='Autotrophs (X_AUT)', color='orange', markersize=4)
    line3 = ax3_twin.plot(heights, data['X_PAO'], '^:', label='PAOs (X_PAO)', color='purple', markersize=4)
    ax3_twin.set_ylabel('Autotroph/PAO Conc. (mg/L)', fontsize=label_fs, color='#555555')
    ax3_twin.tick_params(axis='y', labelcolor='#555555', labelsize=tick_fs)

    # Combine legends from both axes
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, fontsize=legend_fs, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1)

    plt.suptitle(f'Steady State Profile (Influent S_F={influent_sf})', fontsize=12)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust layout to make room for bottom legend if needed
    plt.show()

if __name__ == '__main__':
    run_dhs_simulation_refined_plotting(t_max=60, influent_sf=10.0)