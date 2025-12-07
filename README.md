# PAEC: Production-Aware Exposure Compensation

**Stabilizing Retrieval-Augmented Translation via Lyapunov-Guided Control**

PAEC is a control-theoretic framework for $`k`$-Nearest Neighbor Machine Translation ($`k`$NN-MT). It addresses the stochastic instability and high latency of traditional retrieval-augmented models by modeling the decoding process as a dynamical system stabilization problem.

By optimizing for **Control Lyapunov Function (CLF)** conditions, the PAEC system learns a policy that provably constrains error propagation while adhering to strict production resource constraints (Latency, Memory, Throughput).

## **Key Features**

* **Production-Aware Simulation:** A rigorous simulator that mimics real-world traffic patterns (Gamma distribution loads, diurnal cycles) to train robust control policies.

* **Lyapunov-Guided Control:** Uses a trained Dynamics Model ($`T_{\theta}`$) to predict state evolution and minimize translation error energy ($`V\left(\mathcal{E}\right)`$).

* **Teacher-Student Distillation:** Distills the expensive Online Planner ($`\sim 27`$s/sentence) into a lightweight Offline Policy Network ($`\sim 1.9`$s/sentence), achieving a **$`\mathbf{93\%}`$ latency reduction**.

* **S8 Validation Suite:** A comprehensive theoretical validation suite ensuring Lipschitz continuity and Lyapunov stability.

## **Installation**

### **Prerequisites**

* Linux environment (Tested on Ubuntu)

* Python 3.12 (Main environment) & Python 3.8 (Fairseq environment)

* NVIDIA GPU (CUDA 11.x/12.x)

### **Setup**

We provide an automated setup script that handles Fairseq, FAISS, and dependency installation.

```bash
bash scripts/00_env_setup/01_env_setup.sh
```

This script will:

1. Set up the project structure.

2. Install core dependencies (PyTorch, SentencePiece, etc.).

3. Set up a dedicated `fairseq_env` for the underlying NMT models.

4. Clone and patch the `knn-box` library.

## **Workflow & Usage**

The following steps reproduce the experiments defined in `03_PAEC_MT_Validation.ipynb`.

### **1. Calibration**

Calibrate the system baselines to measure the current hardware's optimal throughput ($R_{\text{opt}}$) and latency ($L_{\text{SLA}}$).

    python scripts/00_calibrate_baselines.py

### **2. Data Preparation**

Prepare the corpus, train the NMT model, and build the Datastore.

> Check the settings in `./src/config.py` for parameter details before generating corpus and running build pipeline.

```bash
# 1. Clean, deduplicate, and split the raw corpus
python scripts/01_data_preparations/01_prepare_corpus.py

# 2. Train BPE, Preprocess Data, Train NMT Model, and Build Datastore
# (This may take several hours depending on your hardware)
bash scripts/01_data_preparations/02_build_pipeline.sh
```

### **3. Generate Training Data**

Run the heuristic policies in the simulator to generate trajectory data for the Dynamics Model.

```bash
# Add flag `--debug` with command-line pipeline if you want to print the process log
python scripts/01_generate_training_data.py #  --debug
```

### **4. Train Dynamics Model ($`\boldsymbol{T_{\theta}}`$)**

Train the Transformer-based Dynamics Model using the "Champion" configuration. This includes Curriculum Learning, N-step CLF loss, and S8 validation.

```bash
# Example
python scripts/t_train_Transformer.py \
    --epochs 30 \
    --batch_size 256 \
    --num_workers 4 \
    --use_curriculum \
    --curriculum_phase1_epochs 10 \
    --use_nstep_clf \
    --nstep_H 3 \
    --use_cvar_loss \
    --cvar_alpha 0.8 \
    --lambda_cbf 0.0 \
    --phi_crit 0.0 \
    --use_spectral_norm \
    --lambda_adt 0.0 \
    --s8_enable \
    --s8_jacobian_robust \
    --s8_lyapunov_full \
    --s8_error_bounds \
    --s8_multistep_decay \
    --s8_cbf_invariance \
    --s8_jacobian_samples 128 \
    --s8_multistep_horizon 10 \
    --save_dir ./models/dynamics_model/Champion
```

### **5. Train Policy Network ($`\boldsymbol{\pi_{\phi}}`$)**

Distill the optimal control signals from the Dynamics Model (Teacher) into the Student Policy Network.

```bash
# Example
python scripts/05_train_policy_network.py \
    --teacher-model-dir ./models/dynamics_model/Champion \
    --policy-data-path ./data/processed/policy_network_training_data.npz \
    --output-model-dir ./models/policy_model \
    --epochs-student 50 \
    --batch-size 256 \
    --learning-rate 1e-4 \
    --debug
```

### **6. Validation & Evaluation**

Evaluate the system using the trained Offline Policy against baselines (Vanilla/Adaptive $`k`$NN-MT).

```bash
# Example
python scripts/paec_mt_validation.py \
    --paec-test-mode 0 \
    --test-size 2000 \
    --beam-size 5 \
    --output-dir ./results/paec_validation \
    --debug
```

## **Project Structure**

* `src/core`: Core data structures (State vectors: Error, Pressure, Context).
* `src/simulation`: Production constraint simulator and heuristic policies.
* `src/system`: $`k`$NN-MT system logic and beam search decoding.
* `scripts`: Training, data generation, and analysis scripts.

## **Citation**

If you use PAEC in your research, please cite our paper:

```latex
@article{paec2026,
    title={PAEC $`k`$NN-MT: Stabilizing Retrieval-Augmented Translation via Lyapunov-Guided Control},
    author={Zixiang Xu},
    journal={ACL},
    year={2026}
}
```

## **Acknowledgements**

This project builds upon the following open-source libraries:

* [Fairseq](https://github.com/facebookresearch/fairseq)

* [knn-box](https://github.com/NJUNLP/knn-box)

* [FAISS](https://github.com/facebookresearch/faiss)

## **License**

This project is licensed under the MIT License. See the LICENSE file for details.


