import os
import rasterio
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from configs import (load_dem,create_mask,apply_mask,dem_path,shp_path,utrue_path)

np.random.seed(20)
tf.random.set_seed(20)

def load_data(data_path, nt):
    data = []
    start_date = datetime(2021, 6, 15)
    for day in range(nt):
        current_date = start_date + timedelta(days=day)
        filename = os.path.join(data_path, current_date.strftime('%Y%m%d') + '.tif')
        if not os.path.exists(filename):
            raise FileNotFoundError(f"file not found: {filename}")
        with rasterio.open(filename) as src:
            img = src.read(1).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                img[img == nodata] = np.nan
            data.append(img)
    data = np.array(data)  #  (num_days, height, width)
    return data

# ----------------------------
# 2. LOAD DATA
# ----------------------------
start_date = datetime(2021, 6, 15)
end_date = datetime(2021, 12, 31)
nt = (end_date - start_date).days + 1

# load data PATH
uf_path = 'data/Long_ConvLSTM_SE/'

fig_dir = 'fig/'
output_dir = os.path.join(fig_dir, 'Long_ConvLSTM_SE_Nudging_150')
os.makedirs(output_dir, exist_ok=True)

dem = load_dem(dem_path)
utrue_raw = load_data(utrue_path, nt)
uf_raw = load_data(uf_path, nt)

mask = create_mask(dem, shp_path)
utrue_masked = apply_mask(utrue_raw, mask, fill_value=0)  # (200, H, W)
uf_masked = apply_mask(uf_raw, mask, fill_value=0)  # (200, H, W)

h, w = utrue_masked.shape[1], utrue_masked.shape[2]  #  (H, W)
utrue = utrue_masked.reshape(nt, h, w)  # (200, H, W)
uf = uf_masked.reshape(nt, h, w)  # (200, H, W)

# ----------------------------
# 3. Constructing observation data
# ----------------------------
mean = 0.0
sd2 = 1e-2
sd1 = np.sqrt(sd2)

# ----------------------------
# 4. Nudging
# ----------------------------
dt = 1.0
tau =0.3 * dt
obs_fraction = 0.5
ua = np.copy(uf)

# u^a = u^f + tau * (u^{obs} - u^f)
for day in range(nt):
    obs_mask = np.random.rand(h, w) < obs_fraction
    obs_values = utrue[day, :, :] + np.random.normal(mean, sd1, (h, w))
    ua[day, obs_mask] = uf[day, obs_mask] + tau * (obs_values[obs_mask] - uf[day, obs_mask])

ua = apply_mask(ua, mask, fill_value=0)

# ----------------------------
# 5. Save the assimilation results as tif files
# ----------------------------
save_dir = 'data/Long_ConvLSTM_SE_Nudging_150/'
os.makedirs(save_dir, exist_ok=True)

with rasterio.open(dem_path) as src:
    meta = src.meta.copy()
meta.update(count=1, dtype='float32')

for i in range(nt):
    current_date = start_date + timedelta(days=i)
    out_filename = os.path.join(save_dir, current_date.strftime('%Y%m%d') + '.tif')
    with rasterio.open(out_filename, 'w', **meta) as dst:
        dst.write(ua[i, :, :].astype(np.float32), 1)

# Visualize every 20 days' predictions
print("Visualizing every 20 days' predictions...")

# ----------------------------
# 6. metrics
# ----------------------------
metrics = []

mse_val = mean_squared_error(utrue.flatten(), uf.flatten())
mae_val = mean_absolute_error(utrue.flatten(), uf.flatten())
r2_val = r2_score(utrue.flatten(), uf.flatten())
rmse_val = np.sqrt(mse_val)
pcc_val = np.corrcoef(utrue.flatten(), uf.flatten())[0, 1]
bias_val = np.mean(uf.flatten() - utrue.flatten()) if len(utrue.flatten()) > 0 else np.nan

mse_val = mean_squared_error(utrue.flatten(), ua.flatten())
mae_val = mean_absolute_error(utrue.flatten(), ua.flatten())
r2_val = r2_score(utrue.flatten(), ua.flatten())
rmse_val = np.sqrt(mse_val)
pcc_val = np.corrcoef(utrue.flatten(), ua.flatten())[0, 1]
bias_val = np.mean(ua.flatten() - utrue.flatten()) if len(utrue.flatten()) > 0 else np.nan

# ----------------------------
# 7. Draw comparison charts and daily indicators
# ----------------------------
def plot_comparison_every_20_days(utrue, uf, ua, dates, fig_dir):
    nt = len(dates)
    for i in range(0, nt, 200):
        fig, axes = plt.subplots(5, 4, figsize=(20, 16))

        for j in range(4):
            day = i + j * 50
            if day >= nt:
                break

            date_str = dates[day].strftime('%Y%m%d')

            true_data = np.ma.array(utrue[day], mask=~mask)
            pred_data = np.ma.array(uf[day], mask=~mask)
            corrected_data = np.ma.array(ua[day], mask=~mask)

            ax1 = axes[0, j]
            ax1.set_title(f"True {date_str}")
            im1 = ax1.imshow(true_data, cmap='RdYlGn', origin='upper')
            ax1.axis('on')

            ax2 = axes[1, j]
            ax2.set_title(f"ConvLSTM-Att")
            im2 = ax2.imshow(pred_data, cmap='RdYlGn', origin='upper')
            ax2.axis('on')

            error_pred = np.abs(true_data - pred_data)
            ax3 = axes[2, j]
            ax3.set_title(f"Error(ConvLSTM-Att)")
            im3 = ax3.imshow(error_pred, cmap='RdYlGn', origin='upper')
            ax3.axis('on')

            ax4 = axes[3, j]
            ax4.set_title(f"ConvLSTM_SE_Nudging")
            im4 = ax4.imshow(corrected_data, cmap='RdYlGn', origin='upper')
            ax4.axis('on')

            error_corrected = np.abs(true_data - corrected_data)
            ax5 = axes[4, j]
            ax5.set_title(f"Error(ConvLSTM_SE_Nudging)")
            im5 = ax5.imshow(error_corrected, cmap='RdYlGn', origin='upper')
            ax5.axis('on')

        plt.tight_layout(rect=[0, 0.05, 1, 1])  # [left, bottom, right, top]

        fig.tight_layout(h_pad=1, rect=[1, 0.3, 1, 1])

        cbar_ax = fig.add_axes([0.2, 0.02, 0.6, 0.01])  # [left, bottom, width, height]
        cbar = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal')
        cbar.ax.tick_params(labelsize=16)

        fig_path = os.path.join(fig_dir, f'{dates[i].strftime("%Y%m%d")}.png')
        plt.savefig(fig_path, dpi=1000, bbox_inches='tight')
        plt.close()
        print(f"Comparison plot saved to {fig_path}")


def plot_daily_metrics(utrue, uf, ua, dates, fig_dir):
    nt = len(dates)
    mse_uf_list = []
    mae_uf_list = []
    r2_uf_list = []
    rmse_uf_list = []

    mse_ua_list = []
    mae_ua_list = []
    r2_ua_list = []
    rmse_ua_list = []

    for day in range(nt):
        true_val = utrue[day].flatten()
        pred_val = uf[day].flatten()
        corrected_val = ua[day].flatten()

        mse_uf_list.append(mean_squared_error(true_val, pred_val))
        mae_uf_list.append(mean_absolute_error(true_val, pred_val))
        r2_uf_list.append(r2_score(true_val, pred_val))
        rmse_uf_list.append(np.sqrt(mean_squared_error(true_val, pred_val)))

        mse_ua_list.append(mean_squared_error(true_val, corrected_val))
        mae_ua_list.append(mean_absolute_error(true_val, corrected_val))
        r2_ua_list.append(r2_score(true_val, corrected_val))
        rmse_ua_list.append(np.sqrt(mean_squared_error(true_val, corrected_val)))

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # MSE
    axes[0, 0].plot(dates, mse_ua_list, label='MSE (ConvLSTM_SE_Nudging)', color='blue')
    axes[0, 0].plot(dates, mse_uf_list, label='MSE (ConvLSTM_SE)', color='orange', linestyle='--')
    axes[0, 0].set_title("MSE")
    axes[0, 0].set_xlabel("Date")
    axes[0, 0].set_ylabel("MSE")
    axes[0, 0].legend()

    # MAE
    axes[0, 1].plot(dates, mae_ua_list, label='MAE (ConvLSTM_SE_Nudging)', color='green')
    axes[0, 1].plot(dates, mae_uf_list, label='MAE (ConvLSTM_SE)', color='red', linestyle='--')
    axes[0, 1].set_title("MAE")
    axes[0, 1].set_xlabel("Date")
    axes[0, 1].set_ylabel("MAE")
    axes[0, 1].legend()

    # R²
    axes[1, 0].plot(dates, r2_ua_list, label='R² (ConvLSTM_SE_Nudging)', color='red')
    axes[1, 0].plot(dates, r2_uf_list, label='R² (ConvLSTM_SE)', color='purple', linestyle='--')
    axes[1, 0].set_title("R²")
    axes[1, 0].set_xlabel("Date")
    axes[1, 0].set_ylabel("R²")
    axes[1, 0].legend()

    # RMSE
    axes[1, 1].plot(dates, rmse_ua_list, label='RMSE (ConvLSTM_SE_Nudging)', color='purple')
    axes[1, 1].plot(dates, rmse_uf_list, label='RMSE (ConvLSTM_SE)', color='brown', linestyle='--')
    axes[1, 1].set_title("RMSE")
    axes[1, 1].set_xlabel("Date")
    axes[1, 1].set_ylabel("RMSE")
    axes[1, 1].legend()

    plt.tight_layout()
    fig_path = os.path.join(fig_dir, "ConvLSTM_SE_Nudging.png")
    plt.savefig(fig_path, dpi=800, bbox_inches='tight')
    plt.close()
    print(f"Daily metrics plot saved to {fig_path}")

    data = {
        "Date": dates,
        "MSE_true_vs_corrected": mse_ua_list,
        "MSE_true_vs_predicted": mse_uf_list,
        "MAE_true_vs_corrected": mae_ua_list,
        "MAE_true_vs_predicted": mae_uf_list,
        "R2_true_vs_corrected": r2_ua_list,
        "R2_true_vs_predicted": r2_uf_list,
        "RMSE_true_vs_corrected": rmse_ua_list,
        "RMSE_true_vs_predicted": rmse_uf_list,
    }
    df_metrics = pd.DataFrame(data)
    save_csv_path = os.path.join(output_dir, "ConvLSTM_SE_Nudging.csv")
    df_metrics.to_csv(save_csv_path, index=False)
    print(f"Daily metrics CSV saved to {save_csv_path}")

dates = [start_date + timedelta(days=i) for i in range(nt)]

plot_comparison_every_20_days(utrue, uf, ua, dates, output_dir)

plot_daily_metrics(utrue, uf, ua, dates, output_dir)