# Control Panel Module Overview
Content:  
A) Precise Conrol-- *RepE*   
B) Evaluation Metrics



## A) Precise Control 
## Challenge & Solution

Achieving **precise control** over cognitive bias in language models faces several **challenges**:  
- Different biases manifest with varying intensity and separability in the representation space.  
- Models may carry inherent initial biases even before intervention.  
- Sensitivity to the control coefficient can differ widely across tasks and bias types, making one-size-fits-all solutions ineffective.

Our design addresses these challenges through four dedicated modules:  
- **RepECoef** serves as the foundational parameter for bias adjustment.  
- **RangeAutoTuner** automatically identifies the most effective control range for each bias and task, eliminating the need for manual tuning and increasing robustness.  
- **BiasNeutralizer** aligns all control operations to a unified, bias-free baseline, ensuring consistent and comparable results across different models and bias directions.  
- **ControlCoef** applies corrections from RangeAutoTuner and BiasNeutralizer to achieve precise and calibrated control over bias intensity.

## Module Overview

| Module Name        | Functionality                                                   | Description                                                                                       |
|--------------------|-----------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| **RepECoef**       | Foundational parameter for bias adjustment                      | Serves as the base parameter for controlling the intensity of bias projection.                   |
| **RangeAutoTuner** | Automatically tunes and recommends the effective control range  | Dynamically analyzes model sensitivity to RepeCoef for different biases and tasks, automatically determining the optimal range for robust control. |
| **BiasNeutralizer**| Standardizes and aligns the bias baseline                      | Calibrates the bias-free baseline across different models and tasks, ensuring comparability and consistency of control operations. |
| **ControlCoef**    | Achieves precise and calibrated control                        | Integrates corrections from RangeAutoTuner and BiasNeutralizer to deliver robust and precise bias control. |

---


## B) Evaluation Metrics


1. 单调性（Violations of Monotonicity 单调性违例数）
2. 平滑性 (一阶差分；二阶差分)
3. 峰值及区间 (困惑度/Prompt另外的) 

### 1. Control Intensity Response (CIR)
- **What it measures**: How effectively the model responds to changes in the ControlCoef. This measures the linearity or sensitivity of the control mechanism.
- **How to compute**:  
  1. Vary the ControlCoef across a predefined range (e.g., -1.0 to 1.0).  
  2. Measure the model’s output distribution (e.g., probabilities, logits, or embedding shifts) for each control coefficient.  
  3. Use metrics like R² (coefficient of determination) and Pearson Correlation Coefficient to measure the relationship between the control coefficient and the change in output.  
- **Formula (R²)**:  
  R² = 1 - (Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²)  
  Where:
  - yᵢ: Observed output values
  - ŷᵢ: Predicted output values based on ControlCoef
  - ȳ: Mean of observed output values
- **Formula (Pearson Correlation Coefficient)**:  
  ρ = Σ(cᵢ - c̄)(aᵢ - ā) / √[Σ(cᵢ - c̄)² Σ(aᵢ - ā)²]  
  Where:
  - cᵢ: Control coefficient value for the i-th sample
  - aᵢ: Corresponding attribute score
  - c̄, ā: Mean values of control coefficients and attribute scores respectively
- **Expected result**: A higher R² or correlation coefficient (close to +1 or -1) indicates more precise, predictable control.
- **Recommended Experiment Design**:  
  1. Select a set of prefixes and control targets.  
  2. Set a series of ControlCoef values (e.g., -1, -0.5, 0, 0.5, 1).  
  3. For each ControlCoef, sample multiple outputs and compute attribute scores (e.g., sentiment probability, topic relevance).  
  4. Plot a scatter diagram of ControlCoef vs. attribute scores and compute both R² and Pearson correlation coefficient.  
- **Reference**:  
  - Plug and Play Language Models: A Simple Approach to Controlled Text Generation (ICLR 2020).
  - A Plug-and-Play Method for Controlled Text Generation (ACL 2021).

 ### 2. 
