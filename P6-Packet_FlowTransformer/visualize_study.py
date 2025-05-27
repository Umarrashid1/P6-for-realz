import optuna
import os
import plotly  # Optuna's visualization typically uses Plotly
import sklearn

# Ensure the directory for saving plots exists
output_dir = "optuna_visualizations_baseline"
os.makedirs(output_dir, exist_ok=True)

study_name = "baseline-flow-transformer-study"  # From your tune_pretrain.py
storage_name = f"sqlite:///{study_name}.db"  # From your tune_pretrain.py

try:
    study = optuna.load_study(study_name=study_name, storage=storage_name)
    print(f"Study '{study_name}' loaded successfully with {len(study.trials)} trials.")

    if not study.trials:
        print("The study has no trials. Cannot generate visualizations or print data.")
    else:
        # === ADDED DATA EXTRACTION LINES START ===
        print("\n--- All Trial Data (from study.trials_dataframe()) ---")
        try:
            # Using to_string() to get a string representation of the DataFrame
            # This can be very verbose if there are many trials or many columns.
            # You might adjust by selecting specific columns or using .head() if needed.
            print(study.trials_dataframe().to_string())
        except Exception as e:
            print(f"Could not print trials_dataframe: {e}")

        print("\n--- Parameter Importances ---")
        # Parameter importances can only be calculated if there are completed trials.
        completed_trials_for_importance = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(completed_trials_for_importance) > 1: # Optuna typically needs at least 2 completed trials
            try:
                importances = optuna.importance.get_param_importances(study)
                if importances:
                    for param_name, importance_value in importances.items():
                        print(f"Parameter: {param_name}, Importance: {importance_value:.4f}")
                else:
                    print("Parameter importances dictionary is empty (e.g., study might only have one parameter or issues with importance calculation).")
            except RuntimeError as e:
                 print(f"Could not calculate parameter importances (RuntimeError): {e}. This might happen if objective values are constant or too few diverse trials completed.")
            except Exception as e:
                print(f"Could not calculate or print parameter importances: {e}")
        else:
            print("Not enough completed trials (need >1) to calculate parameter importances.")
        # === ADDED DATA EXTRACTION LINES END ===

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
        print("\n--- Generating Plots ---")
        try:
            fig_history = optuna.visualization.plot_optimization_history(study)
            history_path = os.path.join(output_dir, "optimization_history.html")
            fig_history.write_html(history_path)
            print(f"Optimization history plot saved to: {history_path}")
        except Exception as e:
            print(f"Could not generate optimization history plot: {e}")

        # 2. Parameter Importances Plot (Visual)
        # Shows the relative importance of each hyperparameter.
        try:
            if len(completed_trials) > 1: # Check if there are enough completed trials
                fig_importance_visual = optuna.visualization.plot_param_importances(study)
                importance_path_visual = os.path.join(output_dir, "param_importances_visual.html")
                fig_importance_visual.write_html(importance_path_visual)
                print(f"Parameter importances (visual) plot saved to: {importance_path_visual}")
            # If not enough trials for visual, textual importance already handled above
        except RuntimeError as e:
            print(f"Could not generate parameter importances plot (visual) (RuntimeError): {e}")
        except Exception as e:
            print(f"Could not generate parameter importances (visual) plot: {e}")


        # 3. Slice Plot
        # Shows how the objective value changes with individual hyperparameters.
        try:
            fig_slice = optuna.visualization.plot_slice(study, params=tuned_params)
            slice_path = os.path.join(output_dir, "slice_plot.html")
            fig_slice.write_html(slice_path)
            print(f"Slice plot saved to: {slice_path}")
        except ValueError as e:
            print(f"Could not generate slice plot (ValueError, possibly due to no completed trials with all params): {e}")
        except Exception as e:
            print(f"Could not generate slice plot: {e}")

        # 4. Parallel Coordinate Plot
        # Helps visualize high-dimensional relationships between parameters and the objective.
        try:
            fig_parallel = optuna.visualization.plot_parallel_coordinate(study, params=tuned_params)
            parallel_path = os.path.join(output_dir, "parallel_coordinate_plot.html")
            fig_parallel.write_html(parallel_path)
            print(f"Parallel coordinate plot saved to: {parallel_path}")
        except ValueError as e:
            print(f"Could not generate parallel coordinate plot (ValueError, possibly due to no completed trials with all params): {e}")
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
                # Check if both parameters in the pair exist in at least one completed trial's params
                # This is a simplified check; plot_contour itself has more robust checks.
                param1_in_study = any(trial.state == optuna.trial.TrialState.COMPLETE and pair[0] in trial.params for trial in study.trials)
                param2_in_study = any(trial.state == optuna.trial.TrialState.COMPLETE and pair[1] in trial.params for trial in study.trials)

                if param1_in_study and param2_in_study and len(completed_trials) > 1:
                    try:
                        fig_contour = optuna.visualization.plot_contour(study, params=list(pair))
                        contour_path = os.path.join(output_dir, f"contour_plot_{pair[0]}_vs_{pair[1]}.html")
                        fig_contour.write_html(contour_path)
                        print(f"Contour plot for {pair[0]} vs {pair[1]} saved to: {contour_path}")
                    except ValueError as e:
                        print(f"Could not generate contour plot for {pair[0]} vs {pair[1]} (ValueError): {e}")
                    except Exception as e:
                        print(f"Could not generate contour plot for {pair[0]} vs {pair[1]}: {e}")
                else:
                    print(
                        f"Skipping contour plot for {pair[0]} vs {pair[1]} due to missing params in completed trials or insufficient trials.")
        else:
            print("Not enough parameters tuned to generate contour plots.")

        # 6. Intermediate Values Plot
        # Shows learning curves for trials, useful for understanding pruning behavior.
        try:
            fig_intermediate = optuna.visualization.plot_intermediate_values(study)
            intermediate_path = os.path.join(output_dir, "intermediate_values.html")
            fig_intermediate.write_html(intermediate_path)
            print(f"Intermediate values plot saved to: {intermediate_path}")
        except Exception as e: # Optuna raises a generic Exception if no trials have intermediate values
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