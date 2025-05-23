import optuna
import os
import plotly  # Optuna's visualization typically uses Plotly

# Ensure the directory for saving plots exists
output_dir = "optuna_visualizations"
os.makedirs(output_dir, exist_ok=True)

study_name = "pretrain-packet-transformer-study"  # From your tune_pretrain.py
storage_name = f"sqlite:///{study_name}.db"  # From your tune_pretrain.py

try:
    study = optuna.load_study(study_name=study_name, storage=storage_name)
    print(f"Study '{study_name}' loaded successfully with {len(study.trials)} trials.")

    if not study.trials:
        print("The study has no trials. Cannot generate visualizations.")
    else:
        # Define the parameters that were tuned, matching names in your objective function
        tuned_params = [
            "lr_pretrain",
            "batch_size_pretrain",
            "d_model",
            "num_heads",
            "num_layers",
            "dropout_transformer_body"
        ]
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        # 1. Optimization History Plot
        # Shows the best objective value found as trials progress.
        try:
            fig_history = optuna.visualization.plot_optimization_history(study)
            history_path = os.path.join(output_dir, "optimization_history.html")
            fig_history.write_html(history_path)
            print(f"Optimization history plot saved to: {history_path}")
        except Exception as e:
            print(f"Could not generate optimization history plot: {e}")

        # 2. Parameter Importances Plot
        # Shows the relative importance of each hyperparameter.
        try:
            if len(completed_trials) > 1:
                fig_importance = optuna.visualization.plot_param_importances(study)
                importance_path = os.path.join(output_dir, "param_importances.html")
                fig_importance.write_html(importance_path)
                print(f"Parameter importances plot saved to: {importance_path}")
            else:
                print("Not enough completed trials to generate parameter importances plot.")
        except Exception as e:
            print(f"Could not generate parameter importances plot: {e}")

        # 3. Slice Plot
        # Shows how the objective value changes with individual hyperparameters.
        try:
            fig_slice = optuna.visualization.plot_slice(study, params=tuned_params)
            slice_path = os.path.join(output_dir, "slice_plot.html")
            fig_slice.write_html(slice_path)
            print(f"Slice plot saved to: {slice_path}")
        except Exception as e:
            print(f"Could not generate slice plot: {e}")

        # 4. Parallel Coordinate Plot
        # Helps visualize high-dimensional relationships between parameters and the objective.
        try:
            fig_parallel = optuna.visualization.plot_parallel_coordinate(study, params=tuned_params)
            parallel_path = os.path.join(output_dir, "parallel_coordinate_plot.html")
            fig_parallel.write_html(parallel_path)
            print(f"Parallel coordinate plot saved to: {parallel_path}")
        except Exception as e:
            print(f"Could not generate parallel coordinate plot: {e}")

        # 5. Contour Plot (for selected pairs of parameters)
        # Shows how pairs of hyperparameters interact to affect the objective value.
        if len(tuned_params) >= 2:
            contour_pairs = [
                ("lr_pretrain", "dropout_transformer_body"),
                ("d_model", "num_layers"),
                ("batch_size_pretrain", "lr_pretrain")
            ]
            for i, pair in enumerate(contour_pairs):
                param1_present = any(pair[0] in trial.params for trial in completed_trials if trial.params)
                param2_present = any(pair[1] in trial.params for trial in completed_trials if trial.params)

                if param1_present and param2_present and len(completed_trials) > 1:
                    try:
                        fig_contour = optuna.visualization.plot_contour(study, params=list(pair))
                        contour_path = os.path.join(output_dir, f"contour_plot_{pair[0]}_vs_{pair[1]}.html")
                        fig_contour.write_html(contour_path)
                        print(f"Contour plot for {pair[0]} vs {pair[1]} saved to: {contour_path}")
                    except Exception as e:
                        print(f"Could not generate contour plot for {pair[0]} vs {pair[1]}: {e}")
                else:
                    print(
                        f"Skipping contour plot for {pair[0]} vs {pair[1]} due to missing params or insufficient completed trials.")
        else:
            print("Not enough parameters to generate contour plots.")

        # 6. Intermediate Values Plot
        # Shows learning curves for trials, useful for understanding pruning behavior.
        try:
            fig_intermediate = optuna.visualization.plot_intermediate_values(study)
            intermediate_path = os.path.join(output_dir, "intermediate_values.html")
            fig_intermediate.write_html(intermediate_path)
            print(f"Intermediate values plot saved to: {intermediate_path}")
        except Exception as e:
            print(
                f"Could not generate intermediate values plot (often requires trials to have reported intermediate values): {e}")

        print(f"\nAll visualization HTML files will be saved in the directory: '{output_dir}'")

except FileNotFoundError:
    print(
        f"Error: The study database file '{study_name}.db' was not found. Please ensure the file exists in the current working directory or provide the correct path.")
except ModuleNotFoundError:
    print(
        "Error: Optuna or Plotly might not be installed in your Python environment. Please install them (e.g., 'pip install optuna plotly') and try again.")
except Exception as e:
    print(f"An error occurred: {e}")