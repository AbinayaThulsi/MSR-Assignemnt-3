# MSR-Assignemnt-3
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
