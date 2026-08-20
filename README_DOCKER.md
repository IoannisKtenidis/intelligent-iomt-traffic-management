# Dockerized LoRa Simulator Environment

This project is fully dockerized to ensure a reproducible environment with all Python dependencies (`simpy`, `numpy`, `xgboost`, `pandas`, `scikit-learn`, `matplotlib`) pre-installed.

Generated results (CSVs and figures) are automatically written back to your local machine using Docker volume mounts.

---

## 1. Prerequisites
Make sure you have Docker and Docker Compose installed on your system:
* [Docker Desktop](https://www.docker.com/products/docker-desktop/)

---

## 2. Build the Docker Image
To build the Docker image, run the following command in your terminal at the project root directory:

```bash
docker-compose build
```

---

## 3. Running Simulations

### Run the Default Script
By default, running docker-compose will execute `scripts/run_all_scenarios.py`:

```bash
docker-compose up
```

### Run a Specific Script
You can run any script by overriding the default command:

#### Run Monte Carlo Simulation Sweep
```bash
docker-compose run --rm simulator python scripts/run_parallel_monte_carlo.py
```

#### Run Adaptive SF Sweep (up to 12000 Nodes)
```bash
docker-compose run --rm simulator python scripts/run_adaptive_new_simulation_sweep.py
```

#### Run SF12 Sweep (Ideal vs ML Classifier + LBT)
```bash
docker-compose run --rm simulator python scripts/run_sf12_new_simulation_sweep.py
```

#### Train the XGBoost Models
```bash
docker-compose run --rm simulator python src/core/train_xgboost.py
```

---

## 4. Where to Find Output Results
All output results will be stored in your local directories:
* **CSV and dat files**: In the `Results_Data/` directory.
* **Plots and diagrams**: In the `Figures/` directory.
