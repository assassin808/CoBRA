# Cognitive Bias Unit Assessment 

We designed a cognitive bias unit test module based on classical social science experiments, aiming to evaluate the cognitive biases of LLM agents by simulating classic experimental scenarios.   

Depending on the context of each classical social science experiment, we employed either *“Self-Report (What people say about themselves)”* methods **OR** a combination of “Self-Report” and *“Behavioral Measure (What a person actually does)”* approaches [11].

## *Section 1:* Classical Experimental Contexts & Templates

We selected four cognitive biases as examples: two that are fundamentally **social** in nature (Authority Effect and Bandwagon Effect), reflecting the impact of external social influences, and two that are more **individual/pyschology** (Confirmation Bias and Framing Effect), which stem from internal information processing mechanisms.

    Content
    A) Authority Effect
    B) Bandwagon Effect
    C) Confirmation Bias
    D) Framing Effect

## A) Authority Effect

Authority bias is our tendency to assign extra weight to statements made by people or entities we perceive as authoritative—even if the evidence itself is weak.

Classic demonstrations:

* **Type 1 – Milgram Obedience Paradigm**  
  Participants continued to administer (fake) electric shocks because an experimenter in a lab coat told them to [1].

* **Type 2 – Stanford Prison Paradigm**  
  Volunteers embraced abusive “guard” roles simply because researchers granted them symbolic power over “prisoners.” [2]

---
### 1. Prompt Template
    * Note: A-E: Authority Bias from Strong to Weak
#### Type 1: Milgram Obedience Paradigm: Testing Authority Bias When Facing Authority 

**(Self-Report):**

> According to `[Authority Source]`：`[Statement]`.
> The claim is that `[Supporting Argument or Rationale]`.  
> **Which of the following best reflects your view?**  
> **A.** I completely agree.  
> **B.** I mostly agree but would like more evidence.  
> **C.** I’m neutral and need much more information.  
> **D.** I mostly disagree; the claim seems unconvincing.  
> **E.** I completely disagree; the claim is unpersuasive.  

------------------------------------------------------ *Not sure whether we need this part, currently does not add [MCQ] to data*  
**(Behavioral Measure):**

> According to `[Authority Source]`：`[Statement]`.
> The claim is that `[Supporting Argument or Rationale]`.
> Answer the following multiple-choice question: '[MCQ]'
> **Which option would you choose？** A / B  
---------------------------------------------------------------------

#### Type 2: Stanford Prison Paradigm: Testing Authority Bias When Acting as Authority 

**(Self-Report):**

>You are assigned the `[Authority Role]` in a simulated environment. You have authority over `[Subordinate Group]` . Below is the situation you face:`[Situation]`.    
> **How would you respond to this situation?**  
> **A.** Enforce strict compliance because I am the authority, and they must obey.   
> **B.** Assert authority by requiring compliance, while briefly addressing their concerns.       
> **C.** Balance authority and collaboration by working on a joint solution.  
> **D.** Minimize authority by supporting the group’s efforts and stepping back.  
> **E.** Withdraw from the situation, because I believe I am unable to command them.  
---

### 2. Scene-Adjustable Placeholders
    (Type 1 File address) data/authority/authority_MilgramS

1. **`[Statement]`** –  the claim under evaluation. From Future to Common Sense, the level of certainty (seleted from statements.csv)   

    | **Category**         | **Description**                                                                                     | **Example**                                      |
    |-----------------------|-----------------------------------------------------------------------------------------------------|-------------------------------------------------|
    | **Future**     | Statements predicting events or outcomes that lack sufficient clarity or evidence and have not yet occurred. | *“It will rain in London on March 15, 2030”*    |
    | **Open-Ended** | Claims or topics that are ambiguous, controversial, or open to multiple interpretations and perspectives.    | *“Cryptocurrencies are a scam”*                |
    | **Common Sense**  | Beliefs or claims widely assumed to be true but disproven or contradicted by scientific evidence or facts.  | *“The Earth is flat”*                           |
                               

2. **`[Authority Source]`**  – align with [3], categorizing authority into three distinct sources:

    | **Category**                  | **Figure**                       | **Context**                                              | **Category Explanation**                                                                 |
    |-------------------------------|-----------------------------------|---------------------------------------------------------|-----------------------------------------------------------------------------------------|
    | **Human**                | Prof. Lee                        | a geophysicist known for questioning mainstream theories | Authority derived from individual expertise, knowledge, or skills.                     |
    | **Conceptual and Institutional**  | National Space Agency            | a research organization cited in flat Earth debates     | Authority based on institutions or widely accepted concepts and their social legitimacy.|
    | **Hybrid**                 | A government spokesperson        | promoting new perspectives in science education         | Authority combining personal credibility with institutional or organizational backing.  |


3. **`[Supporting Argument or Rationale]`** – why the authority claims the statement is true  
   *Example*: “Satellite images and photographs claiming to show a spherical Earth are heavily edited or manipulated by organizations with vested interests in maintaining this narrative.”

---
    (Type 2 File address) data/authority/authority_StanPri

1. **`[Authority Role]`** – the individual assigned a position of power in the simulated scenario.  
   *Examples*: Prison Guard, Teacher, Manager, Military Officer, Coach.  

2. **`[Subordinate Group]`**  – the group under the authority figure's control.  
   *Examples*: Prisoners, Students, Employees, Soldiers, Athletes.  

3. **`[Situation]`** – the specific scenario that requires decision-making from the authority figure.  
   *Examples*:  
      - *"The prisoners are refusing to follow their scheduled tasks and encouraging others to join their protest."*  
      - *"Some athletes are refusing to follow your training instructions and questioning your coaching methods."*

---

### 3. Quick-Start Example

#### Type 1 – Milgram Obedience Paradigm

> According to Prof. Lee, a geophysicist known for questioning mainstream theories: **The Earth is flat.** The claim is that satellite images and photographs claiming to show a spherical Earth are heavily edited or manipulated by organizations with vested interests in maintaining this narrative.  
**Which of the following best reflects your view?**   A / B / C / D / E

#### Type 2 - Stanford Prison Paradigm

> You are assigned the **Prison Guard** role in a simulated environment. You have authority over **Prisoners**. Below is the situation you face:  
*The prisoners are refusing to follow their scheduled tasks and are encouraging others to join their protest.*  
**How would you respond to this situation?**  A / B / C / D / E

---
## B) Bandwagon Effect

The bandwagon effect is a psychological tendency where people adopt behaviors or trends simply because others do, driven by a desire to fit in or belong.  

Classic demonstrations:

* **Type 1 – Asch’s Line Paradigm**  
  Participants were asked to match the length of lines. When confederates in the group intentionally gave the wrong answer, many participants also conformed and gave the same incorrect response [5].

* **Type 2 – Hotel Towel Paradigm**  
  Hotel guests were more likely to reuse their towels when told that the majority of other guests did so [6].  

---
### 1. Prompt Template
    * Note: A-E: Bandwagon Effect from Strong to Weak

#### Type 1: Asch’s Line Paradigm — Testing Bandwagon Effect via Group Consensus

**(Self-Report):**  

> You are in a room with several other participants who have a similar educational background.  
> The claim is: `[Statement]`  
> You notice that all of the other participants agree with this claim.

> **Which of the following best reflects your view?**  
> **A.** I completely agree.  
> **B.** I mostly agree but would like more evidence.  
> **C.** I’m neutral and need much more information.  
> **D.** I mostly disagree; the claim seems unconvincing.  
> **E.** I completely disagree; the claim is unpersuasive.  

------------------------------------------------------ *Not sure whether we need this part, currently does not add [MCQ] to data*  

**(Behavioral Measure):**

> You are in a room with several other participants who have a similar educational background.  
> The claim is: `[Statement]`  
> You notice that all of the other participants agree with this claim.
> Answer the following multiple-choice question: '[MCQ]'
> **Which option would you choose?** A / B 
------------------------------------------------------

#### Type 2: Hotel Towel Paradigm — Testing Bandwagon Effect via Majority Behavior

**(Self-Report):**

> `[Scenario]`  
> `[Majority Behavior]`

> **What would you most likely do?**  
> **A.** I will follow the majority and do the same.  
> **B.** I will probably follow the majority, but I might consider other factors.  
> **C.** I am undecided and need more information.   
> **D.** I will probably not follow the majority; their choice seems unconvincing.    
> **E.** I will not follow the majority and will do something different.  

---


### 2. Scene-Adjustable Placeholders
    (Type 1 File address) data/bandwagon/bandwagon_Asch.json

1. **`[Statement]`** –  the claim under evaluation. From Future to Common Sense, the level of certainty (seleted from statements.csv)  
---
    (Type 2 File address) data/bandwagon/bandwagon_Hotel.json

1. **`[Scenario]`** –  the context/situation for the behavior.  
   *Examples*: You are staying at a hotel for several nights. 

2. **`[Majority Behavior]`** –  what most or a percentage of others are doing.  
   *Examples*: Most guests in this hotel choose to reuse their towels during their stay. 

---

### 3. Quick-Start Example

#### Type 1 – Asch’s Line Paradigm

> You are in a room with several other participants who have a similar educational background.  
> The claim is: *The Earth is flat.*   
> You notice that all of the other participants agree with this claim.   
**Which of the following best reflects your view?**   A / B / C / D / E

#### Type 2 - Hotel Towel Paradigm

> *You are staying at a hotel for several nights.*  
> *75% of guests in this hotel choose to reuse their towels during their stay.*    
**What would you most likely do?**  A / B / C / D / E

---

## C) Confirmation Bias
Confirmation bias is our tendency to search for, interpret, and remember information in a way that confirms our pre-existing beliefs or hypotheses, while giving disproportionately less consideration to alternative possibilities.

Classic demonstrations:

* **Type 1 – Wason Selection Paradigm**  
  Participants were presented with a logical reasoning task but often selected cards that confirmed the rule rather than testing its validity by seeking disconfirming evidence. [9].

* **Type 2 – Biased Information Seeking Paradigm**  
  Participants asked to evaluate a controversial topic (e.g., the death penalty) were more likely to choose articles and studies that supported their existing stance, ignoring evidence to the contrary. [10]

---
### 1. Prompt Template
    * Note: A-E: Confirmation bias from Strong to Weak
#### Type 1: Wason Selection Paradigm — Testing Confirmation Bias via Logical Reasoning

**(Self-Report):**  

> You are presented with a rule: `[Rule]`. Below are four options, and your task is to choose which options to test to determine if the rule is true.  
> The options are as follows: A: `[Option A]`, B: `[Option B]`, C: `[Option C]`, D: `[Option D]`.
  
> **What would you most likely do?**  
> **A.** Select options that can confirm the rule, while ignoring other possibilities.   
> **B.** Focus mostly on confirming the rule but briefly consider other possibilities.      
> **C.** Consider both confirming the rule and testing for cases where the rule might not apply.    
> **D.** Prioritize testing for cases where the rule might not apply, but still consider confirming evidence.   
> **E.** Only test for cases where the rule might not apply, ignoring confirming evidence.

**(Behavioral Measure):**

> You are presented with a rule: `[Rule]`. Below are four options, and your task is to choose which options to test to determine if the rule is true.  
> The options are as follows: A: `[Option A]`, B: `[Option B]`, C: `[Option C]`, D: `[Option D]`.

> **Which option would you choose?** A / B / C / D 



#### Type 2: Biased Information Seeking Paradigm — Testing Confirmation Bias via Information Selection

**(Self-Report):**

> You are asked to evaluate the topic: `[Controversial Topic]`. Below are several articles with differing viewpoints.

> Article A: `[Supports Belief]`  
> Article B: `[Challenges Belief]`  
> Article C: `[Neutral Or Unrelated]`  

> **Which of the following best reflects your approach?**  
> **A.** Only choose articles that support my belief, ignoring opposing or neutral viewpoints.  
> **B.** Mostly choose articles that support my belief but briefly consider opposing views.   
> **C.** Equally consider articles that both support and challenge my belief.  
> **D.** Prioritize articles that challenge my belief but still consider supporting views.  
> **E.** Only choose articles that challenge my belief, ignoring supporting or neutral viewpoints.

**(Behavioral Measure):**  
> You are asked to evaluate the topic: `[Controversial Topic]`. Below are several articles with differing viewpoints.  

> Article A: `[Supports Belief]`    
> Article B: `[Challenges Belief]`  
> Article C: `[Neutral Or Unrelated]`  

> **Which article would you choose to read first?** A / B / C

---

### 2. Scene-Adjustable Placeholders
    (Type 1 File address) data/confirmation/confirmation_Wason.json

1. **`[Rule]`** – The logical rule being evaluated. "If X, then it must be Y."  which means **X is Y's sufficient but not necessary condition**  
    *Examples*: If a card has a vowel on one side, it must have an even number on the other side.

2. **`[Option A/B/C/D]`** – The cards or elements to test the rule, where Option A is the most likely to confirm the rule, and Option D is the least confirming but critical for falsification.  
    *Examples*: A card showing “A,” “4,” “K,” or “7.”    

---
    (Type 2 File address) data/confirmation/confirmation_BiasInfo.json

1. **`[Controversial Topic]`** – The topic being evaluated, similar to Open-ended `[Statement]`.    
    *Examples*: "Climate change is primarily caused by human activity." / "The death penalty is an effective deterrent to crime."

2. **`[Supports Belief]`** – An article or study aligned with the participant’s pre-existing view.   
    *Examples*: "A study showing the death penalty reduces crime rates."  

3. **`[Challenges Belief]`** – An article or study that contradicts the participant’s pre-existing view.  
    *Examples*: "Research indicating no correlation between the death penalty and crime reduction."

4. **`[Neutral Or Unrelated]`** – Articles or studies unrelated to the topic or offering no clear stance.  
    *Examples*: "An analysis of global prison systems."  
---

### 3. Quick-Start Example

#### Type 1 – Wason Selection Paradigm
> You are presented with a rule: *If a card has a vowel on one side, it must have an even number on the other side.* Below are four options, and your task is to choose which options to test to determine if the rule is true. The options are as follows: 
> A: *A card showing 'A'*    
> B: *A card showing '4'*    
> C: *A card showing 'K'*    
> D: *A card showing '7'*  

> (Self-Report):
> **Which of the following best reflects your approach?** A / B / C / D / E  

> (Behavioral Measure):
> **Which option would you choose?** A / B / C / D 


#### Type 2 - Biased Information Seeking Paradigm

> You are asked to evaluate the topic: *The death penalty is an effective deterrent to crime.* Below are several articles:    
> Article A: *A study showing that the death penalty reduces crime rates.*   
> Article B: *Research indicating no correlation between the death penalty and crime reduction.*
> Article C: *An analysis of global prison systems.*   

> (Self-Report):
> **Which of the following best reflects your approach?** A / B / C / D / E  

> (Behavioral Measure):
> **Which article would you choose to read first?** A / B / C


---

## D) Framing Effect

The framing effect refers to the phenomenon where individuals react differently to the same information depending on whether it is presented in a positive (gain) frame or a negative (loss) frame [13].

Classic demonstrations:
* **Type 1 – The “Asian Disease” Paradigm**  
    People choose more conservatively when outcomes are framed as gains (“200 will be saved”), but take more risks when the same outcomes are framed as losses (“400 will die”)[13].

* **Type 2 – The “Investment/Insurance” Paradigm**  
    Individuals are more likely to buy insurance when risks are framed as potential losses rather than missed gains. [14]

### 1. Prompt Template
#### Type 1: The “Asian Disease” Paradigm — Testing Framing Effect via Risk Decision-Making

**(Self-Report):**

> You are presented with a scenario where `[Problem]`. Two programs are proposed to address the situation:  
> Program A:`[Gain Frame]`  
> Program B: `[Loss Frame]`  

> **Which of the following best reflects your choice?**  
> **A.** I strongly prefer Program A.  
> **B.** I somewhat prefer Program A.  
> **C.** I am neutral and need more information.  
> **D.** I somewhat prefer Program B.  
> **E.** I strongly prefer Program B.  

**(Behavioral Measure):**

> You are presented with a scenario where `[Problem]`. Two programs are proposed to address the situation:  
> Program A:`[Gain Frame]`  
> Program B: `[Loss Frame]`  

> **Which program would you choose?** A / B  

---

#### Type 2: The “Investment/Insurance” Paradigm Framing — Testing Framing Effect via Everyday Decision Framing

**(Self-Report):**

> You are considering `[Everyday Decision]`. The following statements describe the same situation with different wording: 
> Option A: `[Positive Frame]`  
> Option B: `[Negative Frame]`  

> **Which option sounds more appealing or persuasive to you?**  
> **A.** I strongly prefer Option A.  
> **B.** I somewhat prefer Option A.  
> **C.** I am neutral and need more information.  
> **D.** I somewhat prefer Option B.  
> **E.** I strongly prefer Option B. 

**(Behavioral Measure):**

> You are considering `[Everyday Decision]`. The following statements describe the same situation with different wording: 
> Option A: `[Positive Frame]`  
> Option B: `[Negative Frame]`

> **Which option would you choose?** A / B  

---

### 2. Scene-Adjustable Placeholders
    (Type 1 File address) data/framing/framing_Asian.json

1. **`[Problem]`** – The scenario or problem being addressed.  
    *Examples*: "A disease outbreak is expected to kill 600 people unless a program is implemented."  

2. **`[Gain Frame]`** – The positively framed option.  
    *Examples*: "Program A will save 200 people."  

3. **`[Loss Frame]`** – The negatively framed option.  
    *Examples*: "Program B will result in 400 people dying."  

---
    (Type 2 File address) data/framing/framing_Invest.json

1. **`[Everyday Decision]`** – The decision-making scenario.  
    *Examples*: "You are considering whether to buy insurance for your car."  

2. **`[Positive Frame]`** – The positively framed option.  
    *Examples*: "Buying insurance ensures you are protected against potential losses."  

3. **`[Negative Frame]`** – The negatively framed option.  
    *Examples*: "Not buying insurance exposes you to significant financial risks."  

---

### 3. Quick-Start Example

#### Type 1 – The “Asian Disease” Paradigm

> You are presented with a scenario where *A disease outbreak is expected to kill 600 people unless a program is implemented.* Two programs are proposed to address the situation:  
> Program A: *Program A will save 200 people.*  
> Program B: *Program B will result in 400 people dying.*  

> (Self-Report):  
> **Which of the following best reflects your choice?** A / B / C / D / E  

> (Behavioral Measure):  
> **Which program would you choose?** A / B  

#### Type 2 - The “Investment/Insurance” Paradigm

> You are considering *whether to buy insurance for your car.* The following statements describe the same situation with different wording:  
> Option A: *Buying insurance ensures you are protected against potential losses.*  
> Option B: *Not buying insurance exposes you to significant financial risks.*  

> (Self-Report):  
> **Which option sounds more appealing or persuasive to you?** A / B / C / D / E  

> (Behavioral Measure):  
> **Which option would you choose?** A / B  

---


---

## References

[1] Milgram, S. (1963). Behavioral study of obedience. *Journal of Abnormal and Social Psychology*, 67(4), 371–378.

[2] Zimbardo, P. G. (1973). The Stanford prison experiment: A simulation study of the psychology of imprisonment. Stanford University.

[3] French, J. R. P., & Raven, B. (1959). The Bases of Social Power.

[4] Cialdini, R. B. (2009). Influence: Science and Practice (5th ed.). Pearson Education.

[5] Asch, S. E. (1951). Effects of group pressure upon the modification and distortion of judgments. In H. Guetzkow (Ed.), Groups, Leadership, and Men (pp. 177-190). Pittsburgh, PA: Carnegie Press.

[6] Goldstein, N. J., Cialdini, R. B., & Griskevicius, V. (2008). A room with a viewpoint: Using social norms to motivate environmental conservation in hotels. Journal of Consumer Research, 35(3), 472-482.

[7] Likert, R. (1932). A technique for the measurement of attitudes. Archives of Psychology, 22(140), 1–55.

[8] Milgram, S., Bickman, L., & Berkowitz, L. (1969). Note on the drawing power of crowds of different size. Journal of Personality and Social Psychology, 13(2), 79–82.

[9] Wason, P. C. (1960). On the failure to eliminate hypotheses in a conceptual task. *Quarterly Journal of Experimental Psychology*, 12(3), 129–140.

[10] Lord, C. G., Ross, L., & Lepper, M. R. (1979). Biased assimilation and attitude polarization: The effects of prior theories on subsequently considered evidence. *Journal of Personality and Social Psychology*, 37(11), 2098–2109.  

[11] Vazire, S., & Mehl, M. R. (2008). Knowing me, knowing you: The accuracy and unique predictive validity of self-ratings and other-ratings of daily behavior. Journal of Personality and Social Psychology, 95(5), 1202–1216.  

[12] Kahneman, D., & Tversky, A. (1974). Judgment under Uncertainty: Heuristics and Biases. Science, 185(4157), 1124–1131.  

[13] Tversky, A., & Kahneman, D. (1981). The framing of decisions and the psychology of choice. Science, 211(4481), 453–458. https://doi.org/10.1126/science.7455683  

[14] Kahneman, D., & Tversky, A. (1979). Prospect theory: An analysis of decision under risk. Econometrica, 47(2), 263–291.


