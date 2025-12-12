import numpy as np
import math
import random
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as ticker

import geopandas as gpd
import contextily as ctx
import os
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from shapely.geometry import Point, LineString, box
import pandas as pd

from models import *


def drone_inspection(M, system_matrices, predictor_coefficients):

    paths = {
        'lines': os.path.join('./maps/power_line.shp'),
        'towers': os.path.join('./maps/power_tower.shp'),
        'plants': os.path.join('./maps/power_plant.shp'),
        'sub_points': os.path.join('./maps/power_substation_point.shp'),
        'sub_polygons': os.path.join('maps/power_substation_polygon.shp'),
        'compensators': os.path.join('maps/power_compensator.shp'),
        'switches': os.path.join('maps/power_switch.shp'),
        'transformers': os.path.join('maps/power_transformer.shp'),
        'generators': os.path.join('maps/power_generator_point.shp'),
    }

    gdfs = {}
    CRS_METERS = "epsg:3857" # Use a projected CRS in meters

    for name, path in paths.items():
        if not os.path.exists(path):
            print(f"Warning: File not found, skipping: {path}")
            continue
        try:
            gdf = gpd.read_file(path)
            gdfs[name] = gdf.to_crs(CRS_METERS)
            print(f"Successfully loaded: '{name}'")
        except Exception as e:
            print(f"Error loading '{path}': {e}")

    if 'lines' not in gdfs:
        raise FileNotFoundError("Error: 'power_line.shp' not found, cannot proceed.")
    gdf_lines = gdfs['lines'][gdfs['lines'].geom_type == 'LineString'].copy()
    gdf_lines['vertex_count'] = gdf_lines.geometry.apply(lambda line: len(line.coords))

    gdf_lines_sorted = gdf_lines.sort_values(by='vertex_count', ascending=False)

    # Select the top (most complex) line
    most_complex_line_series = gdf_lines_sorted.iloc[0]
    target_line_gdf = gpd.GeoDataFrame([most_complex_line_series], crs=CRS_METERS)
    target_line_geom = target_line_gdf.geometry.iloc[0]
    total_vertices = most_complex_line_series.vertex_count

    minx, miny, maxx, maxy = target_line_geom.bounds

    width = maxx - minx
    height = maxy - miny
    padding_x = width * 0.1
    padding_y = height * 0.1

    plot_xlim = (minx - padding_x, maxx + padding_x)
    plot_ylim = (miny - padding_y, maxy + padding_y)

    zoom_box = box(plot_xlim[0], plot_ylim[0], plot_xlim[1], plot_ylim[1])

    def create_trajectory_function(line_geometry, T_total):
        _total_length = line_geometry.length
        _line_geom = LineString(list(reversed(line_geometry.coords)))

        def tracking_coordinates(t):
            if t < 0:
                t = 0
            if t > T_total:
                t = T_total
                
            fraction_along = t / T_total
            distance_along = fraction_along * _total_length
            point_on_line = _line_geom.interpolate(distance_along)
            
            y_1 = point_on_line.x
            y_2 = point_on_line.y
            
            return y_1, y_2

        return tracking_coordinates
    TOTAL_TIME_T = 500 
    tracking_function = create_trajectory_function(target_line_geom, TOTAL_TIME_T)

    trajectories_special = []
    trajectories_mse = []
    trajectories_mae = []
    NUM_RUNS = M
    costs_special = []
    costs_mse = []
    costs_mae = []
    parameter_a_1_special = []
    parameter_a_2_special = []
    parameter_a_1_mse = []
    parameter_a_2_mse = []
    parameter_a_1_mae = []
    parameter_a_2_mae = []

    for run in range(NUM_RUNS):
        # Run 1: Special Loss
        impc = IMPC_tracking(tracking_function, system_matrices, coefficient = predictor_coefficients)
        k = impc.run_mpc()
        
        # Run 2: Usual Loss
        impc2 = IMPC_tracking(tracking_function, system_matrices, coefficient = predictor_coefficients)
        k2 = impc2.run_mpc(loss_type='mse')

        impc3 = IMPC_tracking(tracking_function, system_matrices, coefficient = predictor_coefficients)
        k3 = impc3.run_mpc(loss_type='mae')

        # Store results (assuming k and k2 are numpy arrays of shape (T, 2))
        trajectories_special.append(k)
        trajectories_mse.append(k2)
        trajectories_mae.append(k3)
        costs_special.append(impc.cost[:-1])
        costs_mse.append(impc2.cost[:-1])
        costs_mae.append(impc3.cost[:-1])
        parameter_a_1_special.append(impc.predictor.a_1_list)
        parameter_a_2_special.append(impc.predictor.a_2_list)
        parameter_a_1_mse.append(impc2.predictor.a_1_list)
        parameter_a_2_mse.append(impc2.predictor.a_2_list)
        parameter_a_1_mae.append(impc3.predictor.a_1_list)
        parameter_a_2_mae.append(impc3.predictor.a_2_list)

    trajectories_special = np.array(trajectories_special) 
    trajectories_mse = np.array(trajectories_mse)
    trajectories_mae = np.array(trajectories_mae)
    costs_special = np.array(costs_special) 
    costs_mse = np.array(costs_mse)
    costs_mae = np.array(costs_mae)
    parameter_a_1_special = np.array(parameter_a_1_special)
    parameter_a_2_special = np.array(parameter_a_2_special)
    parameter_a_1_mse = np.array(parameter_a_1_mse)
    parameter_a_2_mse = np.array(parameter_a_2_mse)
    parameter_a_1_mae = np.array(parameter_a_1_mae)
    parameter_a_2_mae = np.array(parameter_a_2_mae)
    # Trajectories (Shape: T x 2)
    mean_traj_special = trajectories_special.mean(axis=0)
    std_traj_special = trajectories_special.std(axis=0)
    mean_traj_mse = trajectories_mse.mean(axis=0)
    std_traj_mse = trajectories_mse.std(axis=0)
    mean_traj_mae = trajectories_mae.mean(axis=0)
    std_traj_mae = trajectories_mae.std(axis=0)

    mean_parameter_a_1_special = parameter_a_1_special.mean(axis=0)
    std_parameter_a_1_special = parameter_a_1_special.std(axis=0)
    mean_parameter_a_2_special = parameter_a_2_special.mean(axis=0)
    std_parameter_a_2_special = parameter_a_2_special.std(axis=0)
    mean_parameter_a_1_mse = parameter_a_1_mse.mean(axis=0)
    std_parameter_a_1_mse = parameter_a_1_mse.std(axis=0)
    mean_parameter_a_2_mse = parameter_a_2_mse.mean(axis=0)
    std_parameter_a_2_mse = parameter_a_2_mse.std(axis=0)
    mean_parameter_a_1_mae = parameter_a_1_mae.mean(axis=0)
    std_parameter_a_1_mae = parameter_a_1_mae.std(axis=0)
    mean_parameter_a_2_mae = parameter_a_2_mae.mean(axis=0)
    std_parameter_a_2_mae = parameter_a_2_mae.std(axis=0)
    # Costs (Shape: T-1)
    mean_cost_special = costs_special.mean(axis=0)
    std_cost_special = costs_special.std(axis=0)
    mean_cost_mse = costs_mse.mean(axis=0)
    std_cost_mse = costs_mse.std(axis=0)
    mean_cost_mae = costs_mae.mean(axis=0)
    std_cost_mae = costs_mae.std(axis=0)

    impc_baseline = IMPC_tracking(tracking_function, system_matrices, coefficient = predictor_coefficients)
    impc_baseline.run_mpc(run_baseline = True)
    baseline_cost = impc_baseline.cost[:-1]

    SHADOW_WIDTH_M = 200 # 50 meters (100m corridor width)
    SHADOW_COLOR = 'white' # Use a light, noticeable color
    SHADOW_ALPHA = 0.7 # Use low opacity (15%) for a shadow effect

    shadow_polygon = target_line_geom.buffer(SHADOW_WIDTH_M)
    shadow_gdf = gpd.GeoSeries([shadow_polygon], crs=CRS_METERS)
    shadow_gdf_in_view = shadow_gdf[shadow_gdf.geometry.intersects(zoom_box)]
    fig, ax = plt.subplots(figsize=(15, 15))

    # --- SET THE ZOOM FIRST ---
    ax.set_xlim(plot_xlim)
    ax.set_ylim(plot_ylim)

    # 1. Create the shadow polygon (buffer)
    shadow_polygon = target_line_geom.buffer(SHADOW_WIDTH_M)
    shadow_gdf = gpd.GeoSeries([shadow_polygon], crs=CRS_METERS)
    shadow_gdf_in_view = shadow_gdf[shadow_gdf.geometry.intersects(zoom_box)]

    # 2. Plot the shadow area (SET ZORDER=2)
    if not shadow_gdf_in_view.empty:
        shadow_gdf_in_view.plot(
            ax=ax, 
            color=SHADOW_COLOR, 
            alpha=SHADOW_ALPHA, 
            edgecolor='none', 
            label='Tracking Corridor (50m Buffer)', 
            zorder=2 # <-- FIXED: Now above the basemap (zorder=0)
        )

    fig1 = plt.figure(figsize=(15, 15))
    plt.rc('font',family='Times New Roman')
    ax_main = fig1.add_axes([0.35, 0, 0.55, 1])  
    ax_main.set_xlim(plot_xlim)
    ax_main.set_ylim(plot_ylim)
    if not shadow_gdf_in_view.empty:
        shadow_gdf_in_view.plot(
            ax=ax_main, 
            color=SHADOW_COLOR, 
            alpha=SHADOW_ALPHA, 
            edgecolor='none', 
            label='Tracking Corridor (50m Buffer)', 
            zorder=2 # <-- FIXED: Now above the basemap (zorder=0)
        )

    other_lines_in_view = gdfs['lines'][gdfs['lines'].geometry.intersects(zoom_box)]
    if not other_lines_in_view.empty:
        other_lines_in_view.plot(ax=ax_main, color='grey', linewidth=2, alpha=0.5, label='Other Transmission Lines', zorder=3) # ADDED ZORDER=3


    if 'sub_polygons' in gdfs:
        items_in_view = gdfs['sub_polygons'][gdfs['sub_polygons'].geometry.intersects(zoom_box)]
        if not items_in_view.empty:
            items_in_view.plot(ax=ax_main, color='orange', alpha=0.3, edgecolor='black', label='Substation (Polygon)', zorder=4) # ADDED ZORDER=4


    if 'towers' in gdfs:
        items_in_view = gdfs['towers'][gdfs['towers'].geometry.intersects(zoom_box)]
        if not items_in_view.empty:
            items_in_view.plot(ax=ax_main, marker='o', color='deeppink', markersize=10, label='Towers', zorder=5) # ADDED ZORDER=5

    if 'sub_points' in gdfs:
        items_in_view = gdfs['sub_points'][gdfs['sub_points'].geometry.intersects(zoom_box)]
        if not items_in_view.empty:
            items_in_view.plot(ax=ax_main, marker='s', color='orange', markersize=50, edgecolor='black', label='Substation (Point)', zorder=5)

    if 'transformers' in gdfs:
        items_in_view = gdfs['transformers'][gdfs['transformers'].geometry.intersects(zoom_box)]
        if not items_in_view.empty:
            items_in_view.plot(ax=ax_main, marker='^', color='green', markersize=40, edgecolor='black', label='Transformers', zorder=5)

    if 'generators' in gdfs:
        items_in_view = gdfs['generators'][gdfs['generators'].geometry.intersects(zoom_box)]
        if not items_in_view.empty:
            items_in_view.plot(ax=ax_main, marker='*', color='magenta', markersize=150, edgecolor='black', label='Generators', zorder=5)

    if 'plants' in gdfs:
        items_in_view = gdfs['plants'][gdfs['plants'].geometry.intersects(zoom_box)]
        if not items_in_view.empty:
            items_in_view.plot(ax=ax_main, marker='h', color='black', markersize=100, label='Plants', zorder=5)

    target_line_gdf.plot(ax=ax_main, color='black', linewidth=2, label='Target Path', zorder=10) 

    ax_main.plot(mean_traj_special[:, 0], mean_traj_special[:, 1], 
                color='lime', linewidth=3, label=r'Special Loss $L$', zorder=12)

    # 2. Plot the Variance Shadow (Special Loss)
    ax_main.fill_between(
        mean_traj_special[:, 0], 
        mean_traj_special[:, 1] - std_traj_special[:, 1],  # Lower Y bound
        mean_traj_special[:, 1] + std_traj_special[:, 1],  # Upper Y bound
        color='lime', alpha=0.15, zorder=10
    )

    # 3. Plot the Mean Trajectory (Usual Loss)
    ax_main.plot(mean_traj_mse[:, 0], mean_traj_mse[:, 1], 
                color='red', linewidth=3, label='MSE Loss', zorder=11)

    # 4. Plot the Variance Shadow (Usual Loss)
    ax_main.fill_between(
        mean_traj_mse[:, 0], 
        mean_traj_mse[:, 1] - std_traj_mse[:, 1], 
        mean_traj_mse[:, 1] + std_traj_mse[:, 1], 
        color='red', alpha=0.15, zorder=9
    )

    ax_main.plot(mean_traj_mae[:, 0], mean_traj_mae[:, 1], 
                color='darkblue', linewidth=3, label='MAE Loss', zorder=11)

    # 4. Plot the Variance Shadow (Usual Loss)
    ax_main.fill_between(
        mean_traj_mae[:, 0], 
        mean_traj_mae[:, 1] - std_traj_mae[:, 1], 
        mean_traj_mae[:, 1] + std_traj_mae[:, 1], 
        color='darkblue', alpha=0.15, zorder=9
    )

    ctx.add_basemap(ax_main, source=ctx.providers.OpenStreetMap.Mapnik, # source=ctx.providers.CartoDB.Positron, zoom='auto'
            )
    ax_main.set_axis_off()
    handles, labels = ax_main.get_legend_handles_labels()
    ax_main.legend(handles=handles, labels=labels, loc='upper left') 
    ax_main.set_title('Drone Tracking Trajectory')

    ax_cost = fig1.add_axes([0.05, 0.54, 0.27, 0.08])
    time_steps = np.arange(TOTAL_TIME_T - 1)
    ax_cost.plot(time_steps, mean_cost_special - baseline_cost, color='lime', linewidth=2, label=r'Special Loss $L$', zorder = 2)
    ax_cost.fill_between(
        time_steps, 
        mean_cost_special - std_cost_special, 
        mean_cost_special + std_cost_special,
        color='lime', alpha=0.3
    )
    ax_cost.plot(time_steps, mean_cost_mse - baseline_cost, color='red', linewidth=2, label='MSE Loss')
    ax_cost.fill_between(
        time_steps, 
        mean_cost_mse - std_cost_mse, 
        mean_cost_mse + std_cost_mse,
        color='red', alpha=0.3, zorder=0
    )

    ax_cost.plot(time_steps, mean_cost_mae - baseline_cost, color='darkblue', linewidth=2, label='MAE Loss', zorder = 1)
    ax_cost.fill_between(
        time_steps, 
        mean_cost_mae - std_cost_mae, 
        mean_cost_mae + std_cost_mae,
        color='darkblue', alpha=0.3
    )

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True) 
    formatter.set_powerlimits((-1, 1)) # Set the range where it switches to scientific notation

    ax_cost.set_title('Regret Comparison')
    # ax_cost.set_xlabel('Time Step $t$')
    ax_cost.set_xticks([])
    ax_cost.set_ylabel('Regret')
    ax_cost.yaxis.set_major_formatter(formatter)
    # ax_cost.legend(loc='best')

    ax_para_1 = fig1.add_axes([0.05, 0.44, 0.27, 0.08])
    ax_para_1.plot(time_steps, mean_parameter_a_1_special[:-1], color='lime', linewidth=2, zorder=3)
    ax_para_1.fill_between(
        time_steps, 
        mean_parameter_a_1_special[:-1] - std_parameter_a_1_special[:-1], 
        mean_parameter_a_1_special[:-1] + std_parameter_a_1_special[:-1],
        color='lime', alpha=0.3
    )
    ax_para_1.plot(time_steps, mean_parameter_a_1_mse[:-1], color='red', linewidth=2)
    ax_para_1.plot(time_steps, [-0.2]*len(time_steps), color='gainsboro',linewidth=2, label=r'True Value of $\theta_1$', zorder=1)
    ax_para_1.fill_between(
        time_steps, 
        mean_parameter_a_1_mse[:-1] - std_parameter_a_1_mse[:-1], 
        mean_parameter_a_1_mse[:-1] + std_parameter_a_1_mse[:-1],
        color='red', alpha=0.3
    )
    ax_para_1.plot(time_steps, mean_parameter_a_1_mae[:-1], color='darkblue', linewidth=2, zorder=2)
    ax_para_1.fill_between(
        time_steps, 
        mean_parameter_a_1_mae[:-1] - std_parameter_a_1_mae[:-1], 
        mean_parameter_a_1_mae[:-1] + std_parameter_a_1_mae[:-1],
        color='darkblue', alpha=0.3
    )
    ax_para_1.set_title(r'Convergence Result of $\theta_1$')
    ax_para_1.set_ylabel(r'Value of $\theta_1$')
    ax_para_1.set_xticks([])
    # ax_para_1.yaxis.set_major_formatter(formatter)
    ax_para_1.legend(loc='best')

    ax_para_2 = fig1.add_axes([0.05, 0.34, 0.27, 0.08])
    ax_para_2.plot(time_steps, mean_parameter_a_2_special[:-1], color='lime', linewidth=2, zorder=3)
    ax_para_2.fill_between(
        time_steps, 
        mean_parameter_a_2_special[:-1] - std_parameter_a_2_special[:-1], 
        mean_parameter_a_2_special[:-1] + std_parameter_a_2_special[:-1],
        color='lime', alpha=0.3
    )
    ax_para_2.plot(time_steps, mean_parameter_a_2_mse[:-1], color='red', linewidth=2)
    ax_para_2.plot(time_steps, [-0.2]*len(time_steps), color='gainsboro',linewidth=2, label=r'True Value of $\theta_2$', zorder=1)
    ax_para_2.fill_between(
        time_steps, 
        mean_parameter_a_2_mse[:-1] - std_parameter_a_2_mse[:-1], 
        mean_parameter_a_2_mse[:-1] + std_parameter_a_2_mse[:-1],
        color='red', alpha=0.3
    )
    ax_para_2.plot(time_steps, mean_parameter_a_2_mae[:-1], color='darkblue', linewidth=2, zorder=2)
    ax_para_2.fill_between(
        time_steps, 
        mean_parameter_a_2_mae[:-1] - std_parameter_a_2_mae[:-1], 
        mean_parameter_a_2_mae[:-1] + std_parameter_a_2_mae[:-1],
        color='darkblue', alpha=0.3
    )
    ax_para_2.set_xlabel(r'Time Step $t$')
    ax_para_2.set_title(r'Convergence Result of $\theta_2$')
    ax_para_2.set_ylabel(r'Value of $\theta_2$')
    # ax_para_2.yaxis.set_major_formatter(formatter)
    ax_para_2.legend(loc='best')

    ax_zoom = fig1.add_axes([0.2, 0.3, 0.55, 0.07])

    if not other_lines_in_view.empty:
        other_lines_in_view.plot(ax=ax_zoom, color='grey', linewidth=2, alpha=0.5, label='Other Transmission Lines', zorder=3) # ADDED ZORDER=3


    if 'sub_polygons' in gdfs:
        items_in_view = gdfs['sub_polygons'][gdfs['sub_polygons'].geometry.intersects(zoom_box)]
        if not items_in_view.empty:
            items_in_view.plot(ax=ax_zoom, color='orange', alpha=0.3, edgecolor='black', label='Substation (Polygon)', zorder=4) # ADDED ZORDER=4


    if 'towers' in gdfs:
        items_in_view = gdfs['towers'][gdfs['towers'].geometry.intersects(zoom_box)]
        if not items_in_view.empty:
            items_in_view.plot(ax=ax_zoom, marker='o', color='blue', markersize=10, label='Towers', zorder=5) # ADDED ZORDER=5

    if 'sub_points' in gdfs:
        items_in_view = gdfs['sub_points'][gdfs['sub_points'].geometry.intersects(zoom_box)]
        if not items_in_view.empty:
            items_in_view.plot(ax=ax_zoom, marker='s', color='orange', markersize=50, edgecolor='black', label='Substation (Point)', zorder=5)

    if 'transformers' in gdfs:
        items_in_view = gdfs['transformers'][gdfs['transformers'].geometry.intersects(zoom_box)]
        if not items_in_view.empty:
            items_in_view.plot(ax=ax_zoom, marker='^', color='green', markersize=40, edgecolor='black', label='Transformers', zorder=5)

    if 'generators' in gdfs:
        items_in_view = gdfs['generators'][gdfs['generators'].geometry.intersects(zoom_box)]
        if not items_in_view.empty:
            items_in_view.plot(ax=ax_zoom, marker='*', color='magenta', markersize=150, edgecolor='black', label='Generators', zorder=5)

    if 'plants' in gdfs:
        items_in_view = gdfs['plants'][gdfs['plants'].geometry.intersects(zoom_box)]
        if not items_in_view.empty:
            items_in_view.plot(ax=ax_zoom, marker='h', color='black', markersize=100, label='Plants', zorder=5)

    target_line_gdf.plot(ax=ax_zoom, color='black', linewidth=2, label='Target Path', zorder=10) 

    ax_zoom.plot(mean_traj_special[:, 0], mean_traj_special[:, 1], 
                color='lime', linewidth=3, label='Special Loss', zorder=12)

    # 2. Plot the Variance Shadow (Special Loss)
    ax_zoom.fill_between(
        mean_traj_special[:, 0], 
        mean_traj_special[:, 1] - std_traj_special[:, 1],  # Lower Y bound
        mean_traj_special[:, 1] + std_traj_special[:, 1],  # Upper Y bound
        color='lime', alpha=0.3, zorder=12
    )

    # 3. Plot the Mean Trajectory (Usual Loss)
    ax_zoom.plot(mean_traj_mse[:, 0], mean_traj_mse[:, 1], 
                color='red', linewidth=3, label='Usual Loss', zorder=11)

    # 4. Plot the Variance Shadow (Usual Loss)
    ax_zoom.fill_between(
        mean_traj_mse[:, 0], 
        mean_traj_mse[:, 1] - std_traj_mse[:, 1], 
        mean_traj_mse[:, 1] + std_traj_mse[:, 1], 
        color='red', alpha=0.3, zorder=9
    )

    ax_zoom.plot(mean_traj_mae[:, 0], mean_traj_mae[:, 1], 
                color='darkblue', linewidth=3, label='MAE', zorder=11)

    # 4. Plot the Variance Shadow (Usual Loss)
    ax_zoom.fill_between(
        mean_traj_mae[:, 0], 
        mean_traj_mae[:, 1] - std_traj_mae[:, 1], 
        mean_traj_mae[:, 1] + std_traj_mae[:, 1], 
        color='darkblue', alpha=0.15, zorder=9
    )

    ctx.add_basemap(ax_zoom, source=ctx.providers.OpenStreetMap.Mapnik, attribution=False, cmap = 'grey', alpha = 0.4# source=ctx.providers.CartoDB.Positron, zoom='auto'
            )

    ax_zoom.set_xlim(651500, 655850)
    ax_zoom.set_ylim(6.3572 * 1e6, 6.35825* 1e6)
    ax_zoom.set_axis_off()
    x0, x1 = 651500, 655850
    y0, y1 = 6.3572 * 1e6, 6.35825* 1e6

    rect = patches.Rectangle(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        linewidth=1.8,
        facecolor='azure',
        alpha = 0.5,
        edgecolor='blueviolet',
        linestyle='--'
    )
    ax_main.add_patch(rect)
    con_ll = ConnectionPatch(
        xyA=(x0, y0),
        coordsA=ax_main.transData,
        xyB=(x0, y1),
        coordsB=ax_zoom.transData,
        linestyle = '--',
        linewidth=1.5
    )
    con_ul = ConnectionPatch(
        xyA=(x1, y0),
        coordsA=ax_main.transData,
        xyB=(x1, y1),
        coordsB=ax_zoom.transData,
        linestyle = '--',
        linewidth=1.5
    )
    fig1.add_artist(con_ll)
    fig1.add_artist(con_ul)

    ax_zoom2 = fig1.add_axes([0.5, 0.3, 0.55, 0.07])
    if not other_lines_in_view.empty:
        other_lines_in_view.plot(ax=ax_zoom2, color='grey', linewidth=2, alpha=0.5, label='Other Transmission Lines', zorder=3) # ADDED ZORDER=3


    if 'sub_polygons' in gdfs:
        items_in_view = gdfs['sub_polygons'][gdfs['sub_polygons'].geometry.intersects(zoom_box)]
        if not items_in_view.empty:
            items_in_view.plot(ax=ax_zoom2, color='orange', alpha=0.3, edgecolor='black', label='Substation (Polygon)', zorder=4) # ADDED ZORDER=4


    if 'towers' in gdfs:
        items_in_view = gdfs['towers'][gdfs['towers'].geometry.intersects(zoom_box)]
        if not items_in_view.empty:
            items_in_view.plot(ax=ax_zoom2, marker='o', color='blue', markersize=10, label='Towers', zorder=5) # ADDED ZORDER=5

    if 'sub_points' in gdfs:
        items_in_view = gdfs['sub_points'][gdfs['sub_points'].geometry.intersects(zoom_box)]
        if not items_in_view.empty:
            items_in_view.plot(ax=ax_zoom2, marker='s', color='orange', markersize=50, edgecolor='black', label='Substation (Point)', zorder=5)

    if 'transformers' in gdfs:
        items_in_view = gdfs['transformers'][gdfs['transformers'].geometry.intersects(zoom_box)]
        if not items_in_view.empty:
            items_in_view.plot(ax=ax_zoom2, marker='^', color='green', markersize=40, edgecolor='black', label='Transformers', zorder=5)

    if 'generators' in gdfs:
        items_in_view = gdfs['generators'][gdfs['generators'].geometry.intersects(zoom_box)]
        if not items_in_view.empty:
            items_in_view.plot(ax=ax_zoom2, marker='*', color='magenta', markersize=150, edgecolor='black', label='Generators', zorder=5)

    if 'plants' in gdfs:
        items_in_view = gdfs['plants'][gdfs['plants'].geometry.intersects(zoom_box)]
        if not items_in_view.empty:
            items_in_view.plot(ax=ax_zoom2, marker='h', color='black', markersize=100, label='Plants', zorder=5)

    target_line_gdf.plot(ax=ax_zoom2, color='black', linewidth=2, label='Target Path', zorder=10) 

    ax_zoom2.plot(mean_traj_special[:, 0], mean_traj_special[:, 1], 
                color='lime', linewidth=3, label='Special Loss', zorder=12)

    # 2. Plot the Variance Shadow (Special Loss)
    ax_zoom2.fill_between(
        mean_traj_special[:, 0], 
        mean_traj_special[:, 1] - std_traj_special[:, 1],  # Lower Y bound
        mean_traj_special[:, 1] + std_traj_special[:, 1],  # Upper Y bound
        color='lime', alpha=0.3, zorder=12
    )

    # 3. Plot the Mean Trajectory (Usual Loss)
    ax_zoom2.plot(mean_traj_mse[:, 0], mean_traj_mse[:, 1], 
                color='red', linewidth=3, label='Usual Loss', zorder=11)

    # 4. Plot the Variance Shadow (Usual Loss)
    ax_zoom2.fill_between(
        mean_traj_mse[:, 0], 
        mean_traj_mse[:, 1] - std_traj_mse[:, 1], 
        mean_traj_mse[:, 1] + std_traj_mse[:, 1], 
        color='red', alpha=0.3, zorder=9
    )

    ax_zoom2.plot(mean_traj_mae[:, 0], mean_traj_mae[:, 1], 
                color='darkblue', linewidth=3, label='MAE', zorder=11)

    # 4. Plot the Variance Shadow (Usual Loss)
    ax_zoom2.fill_between(
        mean_traj_mae[:, 0], 
        mean_traj_mae[:, 1] - std_traj_mae[:, 1], 
        mean_traj_mae[:, 1] + std_traj_mae[:, 1], 
        color='darkblue', alpha=0.15, zorder=9
    )

    ctx.add_basemap(ax_zoom2, source=ctx.providers.OpenStreetMap.Mapnik, attribution=False, cmap = 'grey', alpha = 0.4# source=ctx.providers.CartoDB.Positron, zoom='auto'
            )

    ax_zoom2.set_xlim(660100, 660600)
    ax_zoom2.set_ylim(6.36022 * 1e6, 6.36035* 1e6)
    ax_zoom2.set_axis_off()
    x0, x1 = 660100, 660600
    y0, y1 = 6.36015 * 1e6, 6.36040* 1e6
    rect = patches.Rectangle(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        linewidth=1.8,
        facecolor='azure',
        alpha = 0.5,
        edgecolor='blueviolet',
        linestyle='--', zorder = 15
    )
    ax_main.add_patch(rect)
    x0, x1 = 660100, 660600
    y0, y1 = 6.36022 * 1e6, 6.36035* 1e6
    con_ll = ConnectionPatch(
        xyA=(x0, y0),
        coordsA=ax_main.transData,
        xyB=(x0, y1),
        coordsB=ax_zoom2.transData,
        linestyle = '--',
        linewidth=1.5
    )
    con_ul = ConnectionPatch(
        xyA=(x1, y0),
        coordsA=ax_main.transData,
        xyB=(x1, y1),
        coordsB=ax_zoom2.transData,
        linestyle = '--',
        linewidth=1.5
    )
    fig1.add_artist(con_ll)
    fig1.add_artist(con_ul)

    plt.show()
