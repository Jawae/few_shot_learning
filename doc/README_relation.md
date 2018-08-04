Forked from [here](https://github.com/dragen1860/LearningToCompare-Pytorch).

There are some more repos on this paper in the wild:

- Few-shot part, check [here](https://github.com/floodsung/LearningToCompare_FSL)

- Zero-shot part, check [here](https://github.com/lzrobots/LearningToCompare_ZSL)


To run it in the terminal:

    sh scripts/cvpr18.sh
    
For arguments available, check the ``get_parser`` function in  `cvpr18_relation/main.py` file.
The universal arguments across different methods in located in `basic_opt.py`.

### Re-implementation

TODO.

### Performance
From the forked repo [here](https://github.com/dragen1860/LearningToCompare-Pytorch#mini-imagenet).

| Model                               | Fine Tune | 5-way Acc. |        | 20-way Acc |        |
|-------------------------------------|-----------|------------|--------|------------|--------|
|                                     |           | 1-shot     | 5-shot | 1-shot     | 5-shot |
| Matching Nets                       | N         | 43.56%     | 55.31% | 17.31%     | 22.69% |
| Meta-LSTM                           |           | 43.44%     | 60.60% | 16.70%     | 26.06% |
| MAML                                | Y         | 48.7%      | 63.11% | 16.49%     | 19.29% |
| Meta-SGD                            |           | 50.49%     | 64.03% | 17.56%     | 28.92% |
| TCML                                |           | 55.71%     | 68.88% | -          | -      |
| Learning to Compare            | N         | 57.02%     | 71.07% | -          | -      |
| **Ours, similarity ensemble**				      | N         |  55.2%     |    68.8%      |          |        |
| **Ours, feature ensemble**				      | N         |  55.2%     |    70.1%      |          |        |




