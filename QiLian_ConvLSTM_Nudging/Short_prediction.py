import os
import csv
import pickle
import datetime
import rasterio
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.dates as mdates
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from configs import (input_dir, model_dir, plot_dir, prediction_dir, dem_path, shp_path,
                     create_mask, generator_config, scaler_path)
from mpl_toolkits.axes_grid1 import make_axes_locatable


def save_tiff(prediction, dem_meta, tiff_path):
    """
    Save prediction results as a GeoTIFF file.
    """
    dem_meta.update({
        "count": 1,
        "dtype": 'float32'
    })

    with rasterio.open(tiff_path, 'w', **dem_meta) as dst:
        dst.write(prediction.astype(rasterio.float32), 1)


def plot_metrics(metrics, save_path):
    """
    Plot evaluation metrics and save as an image.
    """
    names = list(metrics.keys())
    values = list(metrics.values())

    plt.figure(figsize=(8, 6))
    bars = plt.bar(names, values, color=['skyblue', 'salmon', 'lightgreen', 'violet'])
    plt.xlabel('Evaluation Metrics')
    plt.ylabel('Values')
    plt.title('Prediction Model Evaluation Metrics')
    plt.ylim(0, max(values) * 1.2)

    # Display values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom')

    plt.savefig(save_path)
    plt.close()
    print(f"Evaluation metrics plot saved to {save_path}")


def plot_prediction_sample(true_val, pred_val, date_str, save_path, extent):
    """
    Plot a comparison between true and predicted values for a single sample.
    """
    error_val = abs(true_val - pred_val)  # Error value = |True - Predicted|

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 1 row, 3 columns

    # True Values
    ax1 = axes[0]
    ax1.set_title(f"True Values ({date_str})")
    im1 = ax1.imshow(true_val, cmap='viridis', extent=extent, origin='upper')
    ax1.axis('on')
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="2%", pad=0.05)
    cb1 = plt.colorbar(im1, cax=cax1)
    cb1.ax.tick_params(labelsize=10)

    # Predicted Values
    ax2 = axes[1]
    ax2.set_title(f"Predicted Values ({date_str})")
    im2 = ax2.imshow(pred_val, cmap='viridis', extent=extent, origin='upper')
    ax2.axis('on')
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="2%", pad=0.05)
    cb2 = plt.colorbar(im2, cax=cax2)
    cb2.ax.tick_params(labelsize=10)

    # Error Values
    ax3 = axes[2]
    ax3.set_title(f"Error Values ({date_str})")
    im3 = ax3.imshow(error_val, cmap='YlOrRd', extent=extent, origin='upper')
    ax3.axis('on')
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="2%", pad=0.05)
    cb3 = plt.colorbar(im3, cax=cax3)
    cb3.ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=800, bbox_inches='tight')
    plt.close()
    print(f"Prediction sample plot saved to {save_path}")


def predict_model(model_path, input_dir, generator_config, plot_dir, output_dir, scaler_path, dem_path, shp_path):
    """
    Use the trained model to perform recursive predictions on test data and save results.
    """
    # Load the normalization scalers
    print("Loading normalization scalers...")
    with open(scaler_path, 'rb') as f:
        scalers = pickle.load(f)
    print("Normalization scalers loaded.")

    # Load the trained model
    print("Loading trained model...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded.")

    # Load test data
    print("Loading test data...")
    X_test = np.load(os.path.join(input_dir, 'X_test.npy'))  # Shape: (200, 14, 83, 207, 1)
    y_test = np.load(os.path.join(input_dir, 'y_test.npy'))  # Shape: (200, 83, 207)
    print(f"Test data loaded. X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # Inverse normalize y_test if possible
    y_test_inverse_path = os.path.join(input_dir, 'y_test_inverse.npy')
    if not os.path.exists(y_test_inverse_path):
        print(f"Warning: Inverse normalized y_test file {y_test_inverse_path} not found. Proceeding without it.")
        y_test_inverse = None
    else:
        y_test_inverse = np.load(y_test_inverse_path)
        print("Inverse normalized y_test loaded.")

    # Initialize variables for recursive prediction
    print("Preparing for recursive prediction...")
    time_steps = generator_config['time_steps']  # Assuming this is 14
    num_samples, _, height, width, _ = X_test.shape
    predictions_inverse = np.zeros((num_samples, height, width), dtype=np.float32)

    # Load the scaler for y (target variable)
    scaler_y = scalers['scaler_y_test']  # Adjust key as per your scaler file

    # For inverse normalization
    def inverse_transform(scaler, data):
        return scaler.inverse_transform(data.reshape(-1, 1)).reshape(data.shape)

    print("Starting recursive predictions...")

    for sample_idx in range(num_samples):
        # Initialize the input window with the first 'time_steps' days
        input_window = X_test[sample_idx].copy()  # Shape: (14, 83, 207, 1)

        # Predict one day ahead
        # Prepare input for the model
        input_for_model = input_window[np.newaxis, ...]  # Shape: (1, 14, 83, 207, 1)

        # Predict the next day
        pred_scaled = model.predict(input_for_model)  # Shape: (1, 83, 207)
        pred_scaled = pred_scaled[0]  # Shape: (83, 207)

        # Inverse normalize the prediction
        pred_inverse = inverse_transform(scaler_y, pred_scaled)  # Shape: (83, 207)
        predictions_inverse[sample_idx] = pred_inverse

        # Update the input window for the next prediction
        # Expand dimensions to match (1, 83, 207, 1)
        pred_scaled_expanded = pred_scaled[np.newaxis, ..., np.newaxis]  # Shape: (1, 83, 207, 1)

        # Concatenate along the time step axis
        input_window = np.concatenate((input_window[1:], pred_scaled_expanded), axis=0)  # Shape: (14, 83, 207, 1)

        if (sample_idx + 1) % 20 == 0 or sample_idx == num_samples - 1:
            print(f"Sample {sample_idx + 1}/{num_samples} predicted.")

    print("Recursive predictions completed.")

    # Save predictions
    predictions_scaled = predictions_inverse  # Assuming you have inverse normalized
    predictions_scaled_path = os.path.join(output_dir, 'ConvLSTM_SE_scaled.npy')
    np.save(predictions_scaled_path, predictions_scaled)
    print(f"Predictions (inverse normalized) saved to {predictions_scaled_path}")

    # Create mask
    print("Creating mask...")
    mask = create_mask(dem_path, shp_path)
    print("Mask created.")

    # Apply mask to predictions
    print("Applying mask to predictions...")
    predictions_inverse_masked = np.where(mask, predictions_inverse, np.nan)
    y_test_inverse_masked = np.where(mask, y_test_inverse, np.nan) if y_test_inverse is not None else None
    print("Mask applied.")

    # Calculate prediction dates
    print("Calculating prediction dates...")
    # Assuming test starts on 2021-06-01 and each sample corresponds to one day
    test_start_date = datetime.date(2021, 6, 1) + datetime.timedelta(days=time_steps)
    num_predictions = predictions_inverse_masked.shape[0]
    test_dates = [test_start_date + datetime.timedelta(days=i) for i in range(num_predictions)]
    print(f"Test prediction date range: {test_dates[0]} to {test_dates[-1]}")

    # Save predictions as TIFF files
    print("Saving predictions as TIFF files...")
    tiff_output_dir = os.path.join(output_dir, 'Short_ConvLSTM_SE')
    os.makedirs(tiff_output_dir, exist_ok=True)

    with rasterio.open(dem_path) as dem_src:
        dem_meta = dem_src.meta.copy()
        bounds = dem_src.bounds
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    for i in range(num_predictions):
        prediction = predictions_inverse_masked[i]
        date_str = test_dates[i].strftime('%Y%m%d')
        tiff_filename = f'{date_str}.tif'
        tiff_path = os.path.join(tiff_output_dir, tiff_filename)
        save_tiff(prediction, dem_meta, tiff_path)
        if (i + 1) % 20 == 0 or i == num_predictions - 1:
            print(f"Saved TIFF file: {tiff_path}")

    # Calculate evaluation metrics
    print("Calculating evaluation metrics...")
    metrics = {}
    if y_test_inverse_masked is not None:
        valid_mask = ~np.isnan(y_test_inverse_masked)
        y_true = y_test_inverse_masked[valid_mask]
        y_pred = predictions_inverse_masked[valid_mask]

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
    else:
        print("Cannot compute inverse normalized metrics because y_test_inverse is unavailable.")
        # Compute metrics on scaled data if inverse normalized data is not available
        y_test_scaled = y_test.copy()
        valid_mask = ~np.isnan(y_test_scaled)
        y_true = y_test_scaled[valid_mask]
        y_pred = predictions_scaled[valid_mask]

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }

    print("Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    # Save metrics
    metrics_path = os.path.join(output_dir, 'ConvLSTM_SE_metrics.pkl')
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)
    print(f"Metrics saved to {metrics_path}")

    csv_path = os.path.join(output_dir, 'ConvLSTM_SE_metrics.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['MSE', 'RMSE', 'MAE', 'R2'])
        writer.writerow([metrics['MSE'], metrics['RMSE'], metrics['MAE'], metrics['R2']])
    print(f"Metrics saved to {csv_path}")

    # Plot metrics
    smc_plot_dir = os.path.join(plot_dir, 'Short_ConvLSTM_SE')
    os.makedirs(smc_plot_dir, exist_ok=True)
    metrics_plot_path = os.path.join(smc_plot_dir, 'ConvLSTM_SE_metrics.png')
    plot_metrics(metrics, metrics_plot_path)

    # Calculate and save daily metrics
    print("Calculating daily evaluation metrics...")
    daily_metrics = []

    for i in range(num_predictions):
        true_val = y_test_inverse_masked[i] if y_test_inverse_masked is not None else y_test[i]
        pred_val = predictions_inverse_masked[i] if y_test_inverse_masked is not None else predictions_scaled[i]
        date_str = test_dates[i].strftime('%Y%m%d')

        y_true_flat = true_val.flatten()
        y_pred_flat = pred_val.flatten()

        valid_mask = ~np.isnan(y_true_flat) & ~np.isnan(y_pred_flat)

        y_true_clean = y_true_flat[valid_mask]
        y_pred_clean = y_pred_flat[valid_mask]

        if y_true_clean.size == 0:
            print(f"Warning: No valid data points for date {date_str}.")
            mse = np.nan
            rmse = np.nan
            mae = np.nan
            r2 = np.nan
        else:
            mse = mean_squared_error(y_true_clean, y_pred_clean)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true_clean, y_pred_clean)
            try:
                r2 = r2_score(y_true_clean, y_pred_clean)
            except:
                r2 = np.nan

        daily_metrics.append({
            'Date': date_str,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        })

    daily_metrics_df = pd.DataFrame(daily_metrics)
    daily_metrics_csv_path = os.path.join(output_dir, 'ConvLSTM_SE_daily_metrics.csv')
    daily_metrics_df.to_csv(daily_metrics_csv_path, index=False)
    print(f"Daily evaluation metrics saved to {daily_metrics_csv_path}")

    # Plot daily metrics
    print("Plotting daily evaluation metrics every 20 days...")
    for i in range(0, num_predictions, 20):
        date_str = test_dates[i].strftime('%Y%m%d')

        # Get metrics for the current day
        metrics_for_day = daily_metrics[i]

        # Extract values
        mse_i = metrics_for_day['MSE']
        rmse_i = metrics_for_day['RMSE']
        mae_i = metrics_for_day['MAE']
        r2_i = metrics_for_day['R2']

        # Create a list of metrics and their corresponding labels
        metrics = [mse_i, rmse_i, mae_i, r2_i]
        labels = ['MSE', 'RMSE', 'MAE', 'R²']

        # Plot bar chart for the daily metrics
        plt.figure(figsize=(8, 6))
        bars = plt.bar(labels, metrics, color=['skyblue', 'salmon', 'lightgreen', 'violet'])
        plt.title(f"Daily Metrics for {date_str}")
        plt.ylabel('Metric Value')
        plt.ylim(0, max(metrics) * 1.2)

        # Display values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.4f}', ha='center', va='bottom')
        # Save the plot with the filename format: ConvLSTM_SE_YYMMDD_metrics.png
        plot_filename = f"{test_dates[i].strftime('%y%m%d')}_ConvLSTM_SE_metrics.png"
        plot_path = os.path.join(smc_plot_dir, plot_filename)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()

        print(f"Daily metrics plot saved to {plot_path}")

    print("Plotting daily evaluation metrics...")
    if not daily_metrics_df.empty:
        daily_metrics_df['Date'] = pd.to_datetime(daily_metrics_df['Date'])

        plt.figure(figsize=(14, 8))

        plt.plot(daily_metrics_df['Date'], daily_metrics_df['MSE'], label='MSE', color='blue', marker='o')
        plt.plot(daily_metrics_df['Date'], daily_metrics_df['RMSE'], label='RMSE', color='green', marker='o')
        plt.plot(daily_metrics_df['Date'], daily_metrics_df['MAE'], label='MAE', color='orange', marker='o')
        plt.plot(daily_metrics_df['Date'], daily_metrics_df['R2'], label='R²', color='red', marker='o')

        locator = mdates.DayLocator(interval=30)
        formatter = mdates.DateFormatter('%Y%m%d')
        plt.gca().xaxis.set_major_locator(locator)
        plt.gca().xaxis.set_major_formatter(formatter)

        plt.title('Daily Evaluation Metrics', fontsize=18)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Metric Values', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.legend(fontsize=12)
        plt.tight_layout()

        daily_metrics_plot_path = os.path.join(smc_plot_dir, 'ConvLSTM_SE_daily_metrics_line_plot.png')
        plt.savefig(daily_metrics_plot_path, dpi=300)
        plt.close()
        print(f"Daily evaluation metrics plot saved to {daily_metrics_plot_path}")
    else:
        print("Warning: No daily evaluation metrics available for plotting.")

    # Visualize every 20 days' predictions
    print("Visualizing every 20 days' predictions...")

    interval = 20
    for i in range(0, num_predictions, interval):
        if y_test_inverse_masked is not None:
            true_val = y_test_inverse_masked[i]
            pred_val = predictions_inverse_masked[i]
        else:
            true_val = y_test[i]
            pred_val = predictions_inverse[i]

        date_str = test_dates[i].strftime('%Y%m%d')
        plot_filename = f'{test_dates[i].strftime("%Y%m%d")}.png'
        plot_path = os.path.join(smc_plot_dir, plot_filename)
        plot_prediction_sample(true_val, pred_val, date_str, plot_path, extent)

    print(f"Prediction visualizations saved to {smc_plot_dir}")

def main():
    model_path = os.path.join(model_dir, 'ConvLSTM_SE_model.h5')
    output_dir = os.path.join(prediction_dir, 'Short_ConvLSTM_SE')
    os.makedirs(output_dir, exist_ok=True)


    # Set Chinese font if needed
    rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    rcParams['axes.unicode_minus'] = False

    print("Starting prediction and evaluation...")
    predict_model(
        model_path=model_path,
        input_dir=input_dir,
        generator_config=generator_config,
        plot_dir=plot_dir,
        output_dir=output_dir,
        scaler_path=scaler_path,
        dem_path=dem_path,
        shp_path=shp_path,
    )
    print("Prediction and evaluation completed. Results saved.")


if __name__ == "__main__":
    main()
