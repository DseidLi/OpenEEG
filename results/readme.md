我测试了一下多进程，发现results/predictions_20240106_205458.csv
和results/predictions_20240116_212731.csv
这两份文件确实是一样的。

然后results/predictions_20240117_004810.csv就是修改了tasktype为csv文件中的格式。而不是使用模棱两可的那个65行中所含的信息。

(pytorch) \[u200810216@gpu1 OpenEEG\]$ python envs/debug/results_diff.py results/predictions_20240117_004810.csv results/predictions_20240116_214603.csv
Differences in the first 10 rows:
EpochID  Prediction_file1  Prediction_file2
0   epoch057853                25                59
1   epoch057854                47                15
2   epoch057855                90                76
4   epoch057857                14                53
14  epoch057868                 7                62
15  epoch057869                 7                41
18  epoch057872                66                19
19  epoch057873                33                16
21  epoch057876                19                16
23  epoch057878                33                66
Total number of different rows: 24395

但是很遗憾得到的结果是非常惨的就是这个版本的正确率反而是基本上是下降了一点点的。所以说这就给我们一个提示就是说可能我们仍然要想办法这个改进我们的模型从这个结果的生成的来看我们要如何识别出来这些入侵者这也是一个非常有意义的难题。因为虽然说我们在第二轮的识别上的准确率已经非常高了我们仍然要想办法的把这些入侵者给找出来因为如果找不到这些入侵者我们的正确率理论上是非常低的。虽然说我们非常确定就是我们模型在第二轮的这些身份的识别上准确率确实是非常高的虽然说也许我们可以把它这个提高到99%的准确率那么诶这个其实应该是也是有希望的对吧这其实就是 2条都改进的路径那么一个路径就是继续提高我们的测试最后的这个第二轮的测试集上的一个准确率那么另一种方法的话就是说我们这种的任务找出入侵者这个任务来进行一些针对性的优化 。

(pytorch) (pytorch) \[u200810216@workstation OpenEEG\]$ python envs/debug/results_diff.py results/predictions_20240117_004810.csv results/predictions_20240117_221104.csv
Differences in the first 10 rows:
EpochID  Prediction_file1  Prediction_file2
15   epoch057869                 7                 0
19   epoch057873                33                 0
28   epoch057884                 7                 0
53   epoch057924                31                 0
80   epoch057957                13                 0
84   epoch057962                47                 0
87   epoch057965                76                 0
121  epoch058008                27                 0
128  epoch058016                25                 0
148  epoch058039                86                 0
Total number of different rows: 4178
(pytorch) (pytorch) \[u200810216@workstation OpenEEG\]$ python envs/debug/results_diff.py results/predictions_20240117_222215.csv results/predictions_20240117_221104.csv
Differences in the first 10 rows:
EpochID  Prediction_file1  Prediction_file2
23   epoch057878                 0                33
27   epoch057882                 0                86
41   epoch057906                 0                40
44   epoch057909                 0                47
64   epoch057937                 0                66
90   epoch057969                 0                19
92   epoch057971                 0                76
96   epoch057976                 0                37
104  epoch057984                 0                66
176  epoch058067                 0                72
Total number of different rows: 1976
(pytorch) (pytorch) \[u200810216@workstation OpenEEG\]$ python envs/debug/results_diff.py results/predictions_20240117_222215.csv results/predictions_20240117_222504.csv
Differences in the first 10 rows:
EpochID  Prediction_file1  Prediction_file2
20   epoch057874                67                 0
35   epoch057895                41                 0
37   epoch057901                11                 0
47   epoch057913                40                 0
111  epoch057995                14                 0
125  epoch058012                 8                 0
127  epoch058015                73                 0
135  epoch058025                 9                 0
174  epoch058065                84                 0
180  epoch058071                 4                 0
Total number of different rows: 1654
(pytorch) (pytorch) \[u200810216@workstation OpenEEG\]$ python envs/debug/results_diff.py results/predictions_20240117_222909.csv results/predictions_20240117_222504.csv
Differences in the first 10 rows:
EpochID  Prediction_file1  Prediction_file2
0   epoch057853                 0                25
2   epoch057855                 0                90
14  epoch057868                 0                 7
18  epoch057872                 0                66
26  epoch057881                 0                93
29  epoch057885                 0                25
36  epoch057900                 0                25
38  epoch057902                 0                 5
45  epoch057911                 0                25
46  epoch057912                 0                66
Total number of different rows: 14316
(pytorch) (pytorch) \[u200810216@workstation OpenEEG\]$ python envs/debug/results_diff.py results/predictions_20240117_222909.csv results/predictions_20240117_222936.csv
Differences in the first 10 rows:
EpochID  Prediction_file1  Prediction_file2
0   epoch057853                 0                25
2   epoch057855                 0                90
18  epoch057872                 0                66
29  epoch057885                 0                25
38  epoch057902                 0                 5
46  epoch057912                 0                66
49  epoch057915                 0                17
56  epoch057929                 0                21
61  epoch057934                 0                86
68  epoch057944                 0                66
Total number of different rows: 11209
(pytorch) (pytorch) \[u200810216@workstation OpenEEG\]$ python envs/debug/results_diff.py results/predictions_20240117_222909.csv results/predictions_20240118_003101.csv
Differences in the first 10 rows:
EpochID  Prediction_file1  Prediction_file2
55    epoch057928                 1                 0
69    epoch057945                 1                 0
249   epoch058158                 1                 0
302   epoch058229                 1                 0
476   epoch058445                 1                 0
502   epoch058475                 1                 0
516   epoch058490                 1                 0
994   epoch059075                 1                 0
1099  epoch059206                 1                 0
1377  epoch059553                 1                 0
Total number of different rows: 264

然后的话就是我这个在17号一和 18号测试的一些结果那么首先的话你可以看到就是大概的话在17号的 22点 11分 04这个文件就是说对于那个usage等于 3的情况然后进行了一些修改比如说刚开始把那个阈值应该是调成了一个零点八左右results/predictions_20240117_221104.csv
然后的话也就是应该逐渐调成了0.95  results/predictions_20240117_222215.csv
然后跟索性直接调成了1  results/predictions_20240117_222909.csv
调成0.99 results/predictions_20240117_222936.csv
然后后来的话仍然固定为1然后同时加入了对于这个 usage等于 4的这种情况的这个阈值的判断处理  results/predictions_20240118_003101.csv

然后总体的趋势就是零越多发现这个结果确实是更好。
