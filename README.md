# Quantum Dice CPU Gibbs Routing App

This Streamlit app is a deploy-safe UI for the Team 6 Quantum Dice routing workflow.

It imports the existing project libraries from the parent folder and runs the existing p-bit Gibbs solver with `device="cpu"`.

## Features

- Enter any number of paired routes: `DPi -> DTi`.
- Input coordinates as longitude and latitude.
- Select 5 satellites from the existing synthetic constellation catalog:
  - 2 satellites for Layer 1.
  - 3 satellites for Layer 2.
- Build the same two-layer QUBO routing model used in the notebook.
- Optimize with CPU Gibbs.
- Repair raw Gibbs output using `repair_global_solution_two_layer` from `LibOrbitSolver.py`.
- Visualize DP, DT, selected satellites, and repaired routes on a map.

## Run

From the project root:

```powershell
streamlit run .\quantum-dice-gibbs-app\app.py
```

## Notes

- The app uses PyTorch on CPU.
- Brute Force and ORBIT are intentionally not used in this UI.
- Default penalties are higher than some notebook examples to encourage feasible raw Gibbs outputs, but the final route is repaired regardless.
