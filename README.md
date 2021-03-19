# MSR-Assignment-3
# Soft Alignment Model for Bug Duplication

This is a further research project on paper - "A Soft Alignment Model for Bug Deduplication". The paper aims at the state-of-art development of a novel soft alignment method which helps to identify the distinct bug reports for the same bug registered by multiple users. A novel deep learning model named SABD is implemented to achieve this. Further, the results are evaluated based on the rank-based metric.

# Metadata

A research project as part of the MSR course at MSR couse 2020/21 at UniKo, CS department, SoftLang Team.

**Paper Title**: "A Soft Alignment Model for Bug Deduplication" by the authors Irving Muller Rodrigues, Daniel Aloise, Eraldo Rezende Fernandes and Michel Dagenais.

**DBLP Link:** https://dblp.org/rec/conf/msr/RodriguesAFD20.html?view=bibtex (https://2020.msrconf.org/details/msr-2020-papers/31/A-Soft-Alignment-Model-for-Bug-Deduplication)

The [Code](https://github.com/irving-muller/soft_alignment_model_bug_deduplication) & [Dataset](https://zenodo.org/record/3922012#.YBUloehKhnI) of the paper can be found here.

And the reproduction [Code & Data](https://github.com/AbinayaThulsi/MSR-Soft-Alignment-Model-for-Bug-Duplication) can be found here.

# Threat

The model is slower than the methods based on Siamese neural networks. Hence the runtime itself is high with GPU. In our case, as mentioned by in assignment 2 we ran using CPU so the time taken will be even more. Hence, we decided to work on reducing the runtime using other algorithms.

# Theory

If this has to be used in real-time and GPU is not available then it will be difficult to find the bugs as the time taken will be more. We are planning to address this issue by using different algorithm in Siamese network which is YOCO(You Only Compare Once). YOCO compares many records in one forward propogation. Using this we can reduce the run time. We have got some idea about few alternate algorithms that can be used. We will try to overcome this issue by trying the different algorithms.

# Research

As mentioned during the agreement the only threat we found was runtime issue. So we started looking for other algorithms which can be used to to reduce the runtime.

1.We started looking into YOCO(You Only Compare Once) and YOLO(You Only Look Once). These are the techniques used to improve the performance in siamese network. YOLO is the algorithm which detects or finds duplicates in one forward propogation instead of going in loop and YOCO compares in one forward propogation. So we thought of using this concept in siamese algorithm and see if it performs better than our technique(sabd) 

2.We started with Siamese algorithm .We started working on siamese_pairs code and we made some code changes([assignment 2 can be referred to understand data extraction part](https://github.com/AbinayaThulsi/MSR-Soft-Alignment-Model-for-Bug-Duplication) Command to Execute : python3 experiments/siamese_pairs.py  -F HOME_DIR/experiments with HOME_DIR/duplicate-bug-report/json_parameters/siamese_pair_test_eclipse.json "recall_rate.window=365").We thought of implementing YOCO and YOLO concept in siamese pair code but we couldnt find more resource for YOCO and YOLO and how it can be used in textual module , The resource we found was for image module and we could have used the concept and implemented in siamese code we have updated but as we started searching deeply about both the algorithms we understood that sabd is already better than siamese and we dint wanted to waste much time in fixing the code. Instead of wasting our time on this we started looking into other techniques.
