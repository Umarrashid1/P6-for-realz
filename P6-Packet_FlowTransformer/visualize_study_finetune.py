import optuna
import os
import plotly  # Optuna's visualization typically uses Plotly

# import sklearn # sklearn was not directly used

# Ensure the directory for saving plots exists
output_dir = "optuna_visualizations_finetune"  # CHANGED for fine-tune study
os.makedirs(output_dir, exist_ok=True)

study_name = "finetune-flow-study"  # CHANGED for the fine-tune study
storage_name = f"sqlite:///{study_name}.db"

try:
    study = optuna.load_study(study_name=study_name, storage=storage_name)
    print(f"Study '{study_name}' loaded successfully with {len(study.trials)} trials.")

    if not study.trials:
        print("The study has no trials. Cannot generate visualizations or print data.")
    else:
        # --- Data Extraction (remains the same) ---
        print("\n--- All Trial Data (from study.trials_dataframe()) ---")
        try:
            print(study.trials_dataframe().to_string())
        except Exception as e:
            print(f"Could not print trials_dataframe: {e}")

        print("\n--- Parameter Importances ---")
        completed_trials_for_importance = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(completed_trials_for_importance) > 1:
            try:
                importances = optuna.importance.get_param_importances(study)
                if importances:
                    for param_name, importance_value in importances.items():
                        print(f"Parameter: {param_name}, Importance: {importance_value:.4f}")
                else:
                    print("Parameter importances dictionary is empty.")
            except RuntimeError as e:
                print(f"Could not calculate parameter importances (RuntimeError): {e}.")
            except Exception as e:
                print(f"Could not calculate or print parameter importances: {e}")
        else:
            print("Not enough completed trials (need >1) to calculate parameter importances.")
        # --- End of Data Extraction ---

        # Define the parameters that were tuned for THIS study (fine-tune)
        tuned_params = [  # CHANGED for fine-tune study
            "lr_finetune",
            "batch_size_flow",
            "classifier_dropout_flow"
        ]
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        print("\n--- Generating Plots ---")
        # 1. Optimization History Plot
        try:
            fig_history = optuna.visualization.plot_optimization_history(study)
            history_path = os.path.join(output_dir, "optimization_history.html")
            fig_history.write_html(history_path)
            print(f"Optimization history plot saved to: {history_path}")
        except Exception as e:
            print(f"Could not generate optimization history plot: {e}")

        # 2. Parameter Importances Plot (Visual)
        try:
            if len(completed_trials) > 1:
                fig_importance_visual = optuna.visualization.plot_param_importances(study)  # Uses all params in study
                importance_path_visual = os.path.join(output_dir, "param_importances_visual.html")
                fig_importance_visual.write_html(importance_path_visual)
                print(f"Parameter importances (visual) plot saved to: {importance_path_visual}")
        except RuntimeError as e:
            print(f"Could not generate parameter importances plot (visual) (RuntimeError): {e}")
        except Exception as e:
            print(f"Could not generate parameter importances (visual) plot: {e}")

        # 3. Slice Plot
        try:
            valid_slice_params = [p for p in tuned_params if any(p in t.params for t in completed_trials if t.params)]
            if valid_slice_params:
                fig_slice = optuna.visualization.plot_slice(study, params=valid_slice_params)
                slice_path = os.path.join(output_dir, "slice_plot.html")
                fig_slice.write_html(slice_path)
                print(f"Slice plot saved to: {slice_path}")
            else:
                print("No valid parameters found in completed trials for slice plot.")
        except ValueError as e:
            print(f"Could not generate slice plot (ValueError): {e}")
        except Exception as e:
            print(f"Could not generate slice plot: {e}")

        # 4. Parallel Coordinate Plot
        try:
            valid_parallel_params = [p for p in tuned_params if
                                     any(p in t.params for t in completed_trials if t.params)]
            if valid_parallel_params:
                fig_parallel = optuna.visualization.plot_parallel_coordinate(study, params=valid_parallel_params)
                parallel_path = os.path.join(output_dir, "parallel_coordinate_plot.html")
                fig_parallel.write_html(parallel_path)
                print(f"Parallel coordinate plot saved to: {parallel_path}")
            else:
                print("No valid parameters found in completed trials for parallel coordinate plot.")
        except ValueError as e:
            print(f"Could not generate parallel coordinate plot (ValueError): {e}")
        except Exception as e:
            print(f"Could not generate parallel coordinate plot: {e}")

        # 5. Contour Plot (for selected pairs of parameters) - CHANGED contour_pairs
        if len(tuned_params) >= 2:
            # Create all possible unique pairs from the tuned_params for fine-tuning
            from itertools import combinations

            contour_pairs = list(combinations(tuned_params, 2))

            for i, pair in enumerate(contour_pairs):
                param1_in_study = any(
                    trial.state == optuna.trial.TrialState.COMPLETE and pair[0] in trial.params for trial in
                    study.trials if trial.params)
                param2_in_study = any(
                    trial.state == optuna.trial.TrialState.COMPLETE and pair[1] in trial.params for trial in
                    study.trials if trial.params)

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
                        f"Skipping contour plot for {pair[0]} vs {pair[1]} due to missing params or insufficient completed trials.")
        else:
            print("Not enough parameters tuned (need at least 2) to generate contour plots.")

        # 6. Intermediate Values Plot
        try:
            fig_intermediate = optuna.visualization.plot_intermediate_values(study)
            intermediate_path = os.path.join(output_dir, "intermediate_values.html")
            fig_intermediate.write_html(intermediate_path)
            print(f"Intermediate values plot saved to: {intermediate_path}")
        except Exception as e:
            print(f"Could not generate intermediate values plot: {e}")

        print(f"\nAll visualization HTML files will be saved in the directory: '{output_dir}'")

except FileNotFoundError:
    print(
        f"Error: The study database file '{study_name}.db' was not found. Please ensure it exists in the current working directory.")

except Exception as e:
    print(f"An error occurred: {e}")