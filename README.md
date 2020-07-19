# Cut to the trace
This is the algorithm going along our paper entited "Cut to the Trace! Process-Aware Partitioning of Long-Running Cases in Customer Journey Logs".

## Abstract
Customer journeys are recordings of multiple events that customers perform over time when interacting with organizations. These interactions are often recorded into so-called journey logs, which can be analyzed using process mining. Customer journeys correspond to long-running and complex traces that may temper the use of process mining for customer journey analysis. A common way to make long-running traces suitable for process mining algorithms is to partition them at the largest time differences between events. However, this technique ignores any process context that journeys may present. In this work, we propose a probabilistic framework that generalizes previous techniques and introduce two novel partitioning approaches that can take into account process context. The first one is inspired by the directly-follows relation, a predominant abstraction in process discovery. The second approach leverages LSTMs, a type of Neural Networks that learn long-term dependencies in sequences, which enables us to consider an even richer contextual representation. We show that both approaches outperform time partitioning techniques on both synthetic and real event logs.

## How to use?
In [example/main.py](example/main.py), you will find a code that explains how to cut using the three techniques : (1) TAP, (2) LCPAP, and (3) GCPAP.

## Experiment
The folder [experiment/](experiment/) contains the code to redo the experiment from the paper and also the raw results.

## Any Question?
Please feel free to contact us at gael.bernard@unil.ch.