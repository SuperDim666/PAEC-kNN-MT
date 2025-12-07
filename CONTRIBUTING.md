# **Contributing to PAEC**

We want to make contributing to this project as easy and transparent as possible. Whether you are fixing a bug, proposing a new feature, or improving documentation, we welcome your help!

## **Pull Requests**

We actively welcome your pull requests.

1. **Fork the repo** and create your branch from `master`.

2. **Code Style**: Please ensure your code adheres to the project's coding standards.

    * We use Python type hints (`typing`) extensively.

    * Ensure `src/config.py` is respected for all hyperparameters.

3. **Tests**: If you've added code that should be tested, please add tests.

    * For dynamics modeling changes, ensure the `S8 Validation Suite` (`scripts/t_train_Transformer.py`) still passes.

4. **Documentation**: If you've changed APIs or script arguments, update the documentation in the code and `README.md`.

5. **License**: Ensure your contributions are compatible with the project's license (MIT).

## **Issues**

We use GitHub issues to track public bugs. Please ensure your description is clear and has sufficient instructions to be able to reproduce the issue.

### **Reporting Bugs**

* **Environment**: Output of `python --version`, `torch.__version__`, and OS details.

* **Steps to Reproduce**: A minimal code snippet or command line arguments.

* **Logs**: Relevant error messages or stack traces.

## **Development Setup**

Please refer to `scripts/00_env_setup/01_env_setup.sh` to set up your development environment. We recommend using a dedicated virtual environment (or the provided `fairseq_env` structure) to avoid dependency conflicts.

## **Attribution**

PAEC incorporates components from **knn-box** and **fairseq**. If your contribution involves modifying core logic related to these libraries, please ensure you respect their original licenses and attribution requirements.

## **License**

By contributing to PAEC, you agree that your contributions will be licensed under the LICENSE file in the root directory of this source tree.
