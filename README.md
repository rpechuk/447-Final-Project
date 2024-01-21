# Semantically Informed Slang Interpretation - Code and Data Repository

#### By: [Zhewei Sun](http://www.cs.toronto.edu/~zheweisun/)

This is the github repository for the NAACL paper "[Semantically Informed Slang Interpretation](https://aclanthology.org/2022.naacl-main.383/)".


```
Sun, Z., Zemel, R., and Xu, Y. (2022) Semantically informed slang interpretation. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics - Human Language Technologies.
```

### /Code

The attached IPython notebook contains the supplementary code for 'Semantically Informed Slang Interpretation'. Since we cannot publically release all revelant datasets used in the study due to copyright terms, the purpose of this code package is to provide an illustration of how the main results from the paper can be reproduced. Specifically, the code package includes all required non-standard code dependencies and code in the IPython notebook show how results can be reproduced using these libraries.

### /Data

This data package contains slang definition entries with example usage sentences used to perform experiments in the paper. It is a subset of the Urban Dictionary dataset released by the paper "Learning to Explain Non-Standard English Words and Phrases" (Ke ni and William Yang Wang, 2017). Each row in the attached csv files include a slang's word form, its slang definition, and a usage context sentence with the slang expression replaced by a '[\*SLANGAAAP\*]' token. The data is splited into three partitions for training, development, and testing as described in the paper.