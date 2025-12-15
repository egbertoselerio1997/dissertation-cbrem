### Context
- This is a codebase of a research project.
- The research project is on the development of a new surrogate model of activated sludge processes called Coupled Bilinear Regression Equations (CBRE).
- CBRE serves as an alternative to common machine learning tools for executing numerical optimization on activated sludge processes.
- It is readilly linearizable when used in an optimization framework, hence the optimization model implementations that are also in this codebase.
- Everything in this codebase is intended to evaluate and analyze the capability of CBRE.

### My general preference for this codebase:
- All results must be stored in `~/data`
- All code must be in `~/src`
- `~/src/analysis` contain all executions that are not related to optimization, training, or simulation
- `~/src/optimization`, `~/src/training`, and `~/src/simulation` contain all executions that are not related to optimization, training, and simulation, respectively

### Important Codebase Characteristics
- All of the code are fetching data elsewhere
- Thus, changing codebase directory must account for these paths

### Tools Used
- For analysis, we extract coefficients, compare simulation vs optimization results, generate predictions, and generate reports for a comprehensive analysis using four separate .py files
- For simulation, we use QSDsan
- For optimization, we use Pyomo as the framework
- For training, we developed a custom training for the mathematical model we use here. The mathematical model must not be revised. It is the most important idea contained in this repository. All the rest exist to support it.

### Current Issues with the Codebase
- The arrangement of the codebase is suboptimal
- The arrangement/directory architechture is confusing