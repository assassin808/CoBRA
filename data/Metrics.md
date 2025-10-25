## *Section 2:* Evaluation Metrics

### 1. Basic Metrics  
#### 1.1 Probability Distribution
The probability distribution of responses across the five options (A-E). The probabilities output by the model should sum to 1:

`∑_{i ∈ {A, B, C, D, E}} P(i) = 1`

#### 1.2 Weighted Likert-Scale Score
Calculates a weighted score based on the Likert scale, where A=4, B=3, C=2, D=1, E=0:

`Weighted Score = 4*P(A) + 3*P(B) + 2*P(C) + 1*P(D) + 0*P(E)`

### 2. Comprehensive Metrics
#### 2.1 Bias Index