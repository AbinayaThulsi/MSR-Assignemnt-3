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

As mentioned during the agreement the only threat that we came accross was runtime issue. Since the fixed-length representations of the reports is being generated jointly, it seemed to be hard to save computation time. Hence, the SABD model is slower based on Siamese neural networks. In-Order to reduce the runtime, we looked for many alternative sources. 
## First Approach :
One such alternative which we came across was implementing an algorithm called [YOLO](https://medium.com/@kuzuryu71/improving-siamese-network-performance-f7c2371bdc1e). The idea behind the choice of YOLO was, it compares many bugs in one forward propagation steps. This helps in reducing the execution time faster. 
We started looking into YOCO (You Only Compare Once) and YOLO(You Only Look Once). These are the techniques used to improve the performance in siamese network. YOLO is the algorithm which detects or finds duplicates in one forward propogation instead of going in loop and YOCO compares in one forward propogation. We went through many papers and GitHub repositories to address our threat in the similar way.  Since there were not many resources available, we ourself tried to implement the idea of algorithm. But, we were not able to fix the issue.

## Second Approach:
Secondly, we decided to proceed with of changing Siamese algorithm, using YOCO and YOLO in this algorithm. We started working on siamese_pairs code and we made some code changes([assignment 2 can be referred to understand data extraction part](https://github.com/AbinayaThulsi/MSR-Soft-Alignment-Model-for-Bug-Duplication).

##### Command to Execute :
```bash
python3 experiments/siamese_pairs.py  -F HOME_DIR/experiments with HOME_DIR/duplicate-bug-report/json_parameters/siamese_pair_test_eclipse.json "recall_rate.window=365").
```
But it was able to run only until the 1st epoch, afterwards we started getting segmentation error. Later, We thought of implementing YOCO and YOLO concept in siamese pair code but we couldnt find more resource for YOCO and YOLO and how it can be used in textual module , The resource we found was for image module and we could have used the concept and implemented in siamese code we have updated but as we started searching deeply about both the algorithms we understood that sabd is already better than [siamese](https://ieeexplore.ieee.org/document/1467314)(sabd might be slower than siamese but when real time is considered sabd outperforms siamese pair and siamese triplet) So if we succeed in saving runtime then performance in terms of recall rate would still be less and we dint wanted to waste much time in fixing the code. Instead of wasting our time on this we started looking into other techniques.

## Third Approach:
As the [attention mechanism](https://arxiv.org/abs/1409.0473) computes interdependent representations for each report jointly for pair of possibly duplicate reports, it is the main reason for the model being slow and not being efficiently faster. So, we even tried to split and generate the representation in batches. But the bug duplicate report detection accuracy was not remarkable. As already stated in the paper, even though fixed representation consumes more time the model performs the best only by that. 

# Conclusion
We did not stop with only three approaches. We kept on looking into many other reference papers and codes. But these three are the main approaches we started implementing. After looking into many papers we understood that sabd is using best approaches compared to other papers, which was mentioned in our paper also but we just wanted to try few things and come to conclusion. We still cant say the time cant be reduced by any method. As mentioned in second approach siamese will take less time but we can start researching on how to increase recall rate using siamese ,But we have not come accross anything with acess to few resources and with the time constraint we had. We have tried with many code changes either we couldnt overcome the errors faced or the code was even more time consuming than sabd. 
