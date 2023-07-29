# UnsupervisedHI
Unsupervised approach to Health Index estimation​

Plots and trained models: https://drive.google.com/drive/folders/1_cjUOqNM5LV7HuL8t4pkXyK8FtYCJWf1?usp=drive_link

## Files description:
<ul>
  <li>experiments - some tries ro improve results</li>
  <li>himodule:</li>
    <ul>
      <li>ae_metrics - metric class Mean Absolute Percentage Error</li>
      <li>custom_classes - models architectures, datasets and dataloaders classes</li>
      <li>linear_regression - LinearRegression architecture</li>
      <li>normalisation - data normalisation methods</li>
      <li>rul_metrics - metrics for true and predicted values if HI curves and RUL</li>
      <li>secondary_funcs - data spleeting, object saving and seeds fixation functions</li>
    </ul>
  <li>RUL_adding_ipynb - create true Remaining Useful Life column</li>
  <li>DEA_Visualization - clean up dataset</li>
  <li>ae_training - simple MLP model training</li>
  <li>ae_training_normal - autoencoder training on normal data and comparison of reconstruction errors for normal and abnormal data</li>
  <li>ae_training_with_window - MLP model training with a sliding window</li>
  <li>сae_error - HI curves creating by Convolutional model</li>
  <li>сae_training - Convolutional model training</li>
  <li>compare_simple_and_window - simple MLP and MLP trained with sliding window comparison</li>
  <li>construct_dataset - dataset creating (combines the functionality of RUL_adding_ipynb and DEA_Visualization)</li>
  <li>linear_regression - LinearRegression model training by sensors values as input and HI as output</li>
  <li>normal_anomaly_visualisation - comparison of normal and anomaly points reconstruction errors</li>
  <li>predict_rul - predict RUL of test data by matching train and test HI curves</li>
  <li>smoothing - HI curves interpolation</li>
  <li>window_error_visualisation - HI curves creating by windowed MLP</li>
</ul>
