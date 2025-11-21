## **[LoRADS](https://github.com/COPT-Public/LoRADS)**

#### A Low Rank ADMM Splitting Approach for Semidefinite Programming

LoRADS is an enhanced first-order method solver for low rank Semi-definite programming problems (SDPs). LoRADS is written in ANSI C and is maintained by Cardinal Operations COPT development team. More features are still under active development. For more details, please see [the paper](https://arxiv.org/abs/2403.09133).

#### Optimization Problem:

LoRADS focus on the following problem:

$$
\min_{\mathcal{A} \mathbf{X} = \mathbf{b}, \mathbf{X}\in \mathbb{S}_+^n} \left\langle \mathbf{C}, \mathbf{X} \right\rangle
$$

##### Features of the problem:

- linear objective
- affine constraints
- positive-semidefinite variables

#### Current release:

Old version: LoRADS is now under active development and release a pre-built binary (v1.0.0) which reads SDPA (.dat-s) format files, solves the SDP problems. Users testing solvers in Linux can download the binary from **[the release site](https://github.com/COPT-Public/LoRADS/releases/tag/v1.0.0)**

New release 2025.02.13: we now release the complete Lorads code as open source.

#### Getting started:

After downloading the binary form the release site, you could execute

```sh
unzip LoRADS_v_1_0_0-alpha.zip
chmod +x LoRADS_v_1_0_0-alpha
```

Now, by running

```sh
LoRADS_v_1_0_0-alpha SDPAFILE.dat-s
```

we can solve SDPs represented in standard SDPA format.

If everything goes well, we would see logs like below:

```
-----------------------------------------------------------
  L         OOO      RRRR       A      DDDD       SSS
  L        O   O     R   R     A A     D   D     S
  L        O   O     RRRR     AAAAA    D   D      SSS
  L        O   O     R  R     A   A    D   D         S
  LLLLL     OOO      R   R    A   A    DDDD       SSS
-----------------------------------------------------------
Input file name: SDPAFILE.dat-s
timesLogRank   : 2.00
phase1Tol      : 1.00e-03
initRho        : 1/n
rhoMax         : 5000.00
rhoFreq        : 5
rhoFactor      : 1.20
heursitcFactor : 1.00
maxIter        : 10000
timeSecLimit   : 10000.00
-----------------------------------------------------------
Reading SDPA file in 0.093176 seconds
nConstrs = 2964, sdp nBlks = 22, lp Cols = 0
Pre-solver starts
  Processing the cones
  End preprocess
**First using BM method as warm start
Iter:0 objVal:1.13715e+03 dualObj:0.00000e+00 ConstrVio(1):3.01346e+00 ConstrVio(Inf):2.71372e+01 PDGap:9.99121e-01 rho:0.02 minIter:0 trace:6329.42 Time:0.02
Iter:1 objVal:2.56699e+02 dualObj:8.34520e+01 ConstrVio(1):1.15143e+00 ConstrVio(Inf):1.03690e+01 PDGap:5.07830e-01 rho:0.04 minIter:3 trace:5421.49 Time:0.05
Iter:2 objVal:1.26633e+03 dualObj:8.46344e+01 ConstrVio(1):2.60669e-01 ConstrVio(Inf):2.34741e+00 PDGap:8.74058e-01 rho:0.08 minIter:8 trace:5072.61 Time:0.09
...
Iter:8 objVal:6.74243e+01 dualObj:6.73879e+01 ConstrVio(1):1.03085e-03 ConstrVio(Inf):9.28310e-03 PDGap:2.67973e-04 rho:5.18 minIter:275 trace:396.97 Time:1.76
Iter:9 objVal:6.76276e+01 dualObj:6.83311e+01 ConstrVio(1):5.09176e-04 ConstrVio(Inf):4.58529e-03 PDGap:5.13659e-03 rho:10.36 minIter:408 trace:397.59 Time:2.57
**Complete ALM+BM warm start
objVal:6.79500e+01 dualObj:6.81592e+01 ConstrVio:1.10875e-04 Assym:0.00000e+00 DualInfe:1.00000e+00 PDGap:1.52578e-03 rho:10.36 minIter:830 Time:5.12
BM 2 ADMM time consuming :0.000022
**Change method into ADMM Split method
Iter:0 objVal:6.79624e+01 dualObj:6.80730e+01 ConstrVio(1):7.21993e-05 ConstrVio(Inf):6.50177e-04 PDGap:8.07182e-04 rho:12.44 cgIter:1235 trace:398.15 Time:0.21
Iter:1 objVal:6.79678e+01 dualObj:6.80502e+01 ConstrVio(1):3.91495e-05 ConstrVio(Inf):3.52554e-04 PDGap:6.01839e-04 rho:12.44 cgIter:2549 trace:398.16 Time:0.42
Iter:2 objVal:6.79700e+01 dualObj:6.79961e+01 ConstrVio(1):2.43977e-05 ConstrVio(Inf):2.19709e-04 PDGap:1.90828e-04 rho:12.44 cgIter:3877 trace:398.16 Time:0.65
...
Iter:19 objVal:6.79539e+01 dualObj:6.79445e+01 ConstrVio(1):1.60800e-06 ConstrVio(Inf):1.44805e-05 PDGap:6.87056e-05 rho:21.49 cgIter:26245 trace:398.16 Time:4.48
Iter:20 objVal:6.79541e+01 dualObj:6.79417e+01 ConstrVio(1):8.59376e-07 ConstrVio(Inf):7.73895e-06 PDGap:9.01021e-05 rho:25.79 cgIter:27525 trace:398.16 Time:4.69
-----------------------------------------------------------------------
End Program due to reaching `final terminate criteria`:
-----------------------------------------------------------------------
Objective function Value are:
	 1.Primal Objective:            : 6.80e+01
	 2.Dual Objective:              : 6.79e+01
Dimacs Error are:
	 1.Constraint Violation(1)      : 8.59e-07
	 2.Dual Infeasibility(1)        : 1.60e-04
	 3.Primal Dual Gap              : 9.01e-05
	 4.Primal Variable Semidefinite : 0.00e+00
	 5.Constraint Violation(Inf)    : 7.74e-06
	 6.Dual Infeasibility(Inf)      : 3.13e-03
-----------------------------------------------------------------------
Solving SDPAFILE.dat-s in 9.837610 seconds
Solving + calculate full dual infeasibility in 9.880412 seconds
```

#### Parameters

LoRADS provides users with customizable parameters to fine-tune the solving process according to specific problem requirements (if needed). Below is a detailed description of each parameter:

| **Parameter**   | **Description**                                                                          | **Type** | **Default Value** |
| --------------- | ---------------------------------------------------------------------------------------- | -------- | ----------------- |
| timesLogRank    | Multiplier for the O(log(m)) rank calculation (rank = **timesLogRank** $\times$ log(m)). | float    | 2.0               |
| phase1Tol       | Tolerance for ending Phase I.                                                            | float    | 1e-3              |
| initRho         | Initial value for the penalty parameter $\rho$.                                          | float    | 1/n               |
| rhoMax          | Maximum value for the penalty parameter $\rho$.                                          | float    | 5000.0            |
| rhoFreq         | Frequency of increasing $\rho$ (increased every **rhoFreq** iterations).                 | int      | 5                 |
| rhoFactor       | Multiplier for increasing $\rho$ ($\rho =$ **rhoFactor** $\times$ $\rho$).               | float    | 1.2               |
| heuristicFactor | Heuristic factor applied when switching to Phase II.                                     | float    | 1.0               |
| maxIter         | Maximum iteration number for the ADMM algorithm.                                         | int      | 10000             |
| timeSecLimit    | Solving time limitation in seconds.                                                      | float    | 10000.0           |

For example, to set **timesLogRank** to 1.0 and solve a problem, we can execute

```
LoRADS_v_1_0_0-alpha SDPAFILE.dat-s --timesLogRank 1.0
```

## Full Results

Since the IJOC page limit restriction, we give the full results as follows:

#### **Results on MaxCut Problems From Gset**

| Problem | n, m  | LoRADS time (s) | LoRADS errorₘₐₓ | SDPLR time (s) | SDPLR errorₘₐₓ | SDPNAL+ time (s) | SDPNAL+ errorₘₐₓ | COPT time (s) | COPT errorₘₐₓ | ManiSDP time (s) | ManiSDP errorₘₐₓ |
| ------- | ----- | --------------- | --------------- | -------------- | -------------- | ---------------- | ---------------- | ------------- | ------------- | ---------------- | ---------------- |
| G1      | 800   | 0.5             | 4.9E-07         | 0.8            | 1.6E-06        | 36.3             | 2.6E-07          | 0.6           | 2.4E-09       | 1.0              | 9.7E-17          |
| G5      | 800   | 0.5             | 1.1E-07         | 0.7            | 2.6E-07        | 32.1             | 6.4E-06          | 0.9           | 3.4E-09       | 0.8              | 7.5E-17          |
| G9      | 800   | 0.3             | 3.6E-06         | 0.6            | 1.5E-06        | 33.4             | 1.4E-05          | 0.8           | 1.1E-08       | 0.6              | 9.0E-17          |
| G13     | 800   | 0.1             | 2.6E-07         | 0.2            | 3.3E-06        | 52.8             | 1.3E-05          | 0.4           | 4.1E-08       | 3.2              | 1.7E-14          |
| G17     | 800   | 0.2             | 6.8E-07         | 0.4            | 3.5E-06        | 38.6             | 8.8E-07          | 0.7           | 7.5E-09       | 2.3              | 4.3E-15          |
| G21     | 800   | 0.1             | 1.9E-07         | 0.4            | 5.9E-07        | 34.7             | 8.9E-06          | 0.6           | 4.1E-08       | 2.1              | 9.0E-15          |
| G25     | 2000  | 0.3             | 8.7E-06         | 1.4            | 9.1E-07        | 288              | 1.0E-05          | 3.2           | 5.9E-09       | 4.2              | 2.2E-16          |
| G29     | 2000  | 0.8             | 6.0E-07         | 2.6            | 3.3E-06        | 293              | 9.3E-06          | 4.0           | 4.8E-09       | 11.8             | 2.2E-15          |
| G33     | 2000  | 0.3             | 8.1E-07         | 1.4            | 2.0E-07        | 394              | 1.9E-05          | 1.7           | 4.5E-08       | 7.3              | 5.7E-10          |
| G37     | 2000  | 0.6             | 4.9E-07         | 2.1            | 6.4E-06        | 295              | 4.7E-06          | 3.2           | 4.8E-09       | 9.2              | 1.7E-15          |
| G41     | 2000  | 0.8             | 1.5E-06         | 2.2            | 1.6E-06        | 464              | 4.9E-06          | 3.4           | 2.9E-08       | 5.9              | 4.3E-16          |
| G45     | 1000  | 0.3             | 1.7E-07         | 0.4            | 1.4E-06        | 59.8             | 3.7E-07          | 1.2           | 6.1E-09       | 1.3              | 4.7E-17          |
| G49     | 3000  | 0.1             | 1.2E-08         | 0.8            | 5.3E-07        | 484              | 5.2E-04          | 2.8           | 5.3E-09       | 6.3              | 4.1E-18          |
| G53     | 1000  | 0.2             | 7.1E-07         | 0.5            | 1.0E-06        | 23.1             | 1.6E-03          | 1.1           | 7.6E-09       | 2.7              | 4.2E-15          |
| G54     | 1000  | 0.2             | 1.1E-06         | 0.5            | 4.9E-06        | 59.2             | 1.9E-06          | 0.8           | 4.7E-09       | 1.7              | 8.4E-15          |
| G55     | 5000  | 0.8             | 1.9E-07         | 3.4            | 4.3E-07        | 5993             | 1.4E-06          | 20.8          | 7.3E-09       | 43.0             | 5.7E-16          |
| G56     | 5000  | 0.8             | 2.1E-07         | 4.0            | 8.5E-07        | 4770             | 7.1E-06          | 20.2          | 5.0E-09       | 27.7             | 4.4E-15          |
| G57     | 5000  | 0.8             | 2.3E-07         | 5.6            | 3.2E-07        | 7391             | 2.8E-05          | 15.2          | 1.1E-08       | 31.1             | 8.5E-10          |
| G58     | 5000  | 1.8             | 4.8E-07         | 14.3           | 3.7E-06        | 8768             | 4.0E-05          | 29.3          | 6.5E-09       | 71.1             | 2.1E-13          |
| G59     | 5000  | 3.6             | 3.8E-06         | 26.2           | 1.0E-06        | 7712             | 1.3E-05          | 29.2          | 8.5E-09       | 73.0             | 1.1E-15          |
| G60     | 7000  | 0.7             | 8.3E-06         | 7.5            | 6.4E-07        | t                | t                | 43.2          | 3.8E-09       | 62.3             | 2.9E-16          |
| G61     | 7000  | 1.3             | 2.0E-07         | 6.6            | 2.2E-06        | t                | t                | 48.5          | 8.5E-09       | 127              | 1.9E-13          |
| G62     | 7000  | 1.2             | 2.4E-07         | 12.7           | 1.4E-07        | t                | t                | 28.8          | 6.8E-09       | 61.6             | 3.3E-08          |
| G63     | 7000  | 2.8             | 2.4E-07         | 23.2           | 3.7E-07        | t                | t                | 70.4          | 6.7E-09       | 133              | 9.2E-12          |
| G64     | 7000  | 7.6             | 5.4E-06         | 61.1           | 6.0E-07        | t                | t                | 72.6          | 3.0E-09       | 135              | 1.7E-11          |
| G65     | 8000  | 1.5             | 1.5E-08         | 21.5           | 5.4E-07        | t                | t                | 41.1          | 5.6E-09       | 102              | 5.0E-08          |
| G66     | 9000  | 1.6             | 1.3E-08         | 23.3           | 5.3E-08        | t                | t                | 49.8          | 9.6E-09       | 198              | 2.7E-08          |
| G67     | 10000 | 1.8             | 1.5E-08         | 35.4           | 9.3E-07        | t                | t                | 66.7          | 8.3E-09       | 147              | 1.9E-08          |

**Notes:**

- [1] For our solver, we apply a phase-switching tolerance of 1e-2 and a heuristic factor of 10 for all problems tested in this table.
- [2] For ManiSDP, we use the specific parameter settings and function provided in their MaxCut example on GitHub.

#### **Results on Large-scale MaxCut Problems**

| Problem                         | n       | LoRADS time (s) | LoRADS errorₘₐₓ | SDPLR time (s) | SDPLR errorₘₐₓ |
| ------------------------------- | ------- | --------------- | --------------- | -------------- | -------------- |
| p2p-Gnutella04                  | 10879   | 0.8             | 8.2E-06         | 25.6           | 7.1E-07        |
| vsp_befref_fxm_2_4_air02        | 14109   | 5.0             | 1.8E-06         | 156            | 5.8E-08        |
| delaunay_n14                    | 16384   | 2.5             | 3.2E-06         | 595            | 2.1E-08        |
| cs4                             | 22499   | 2.8             | 2.2E-06         | 359            | 1.2E-07        |
| cit-HepTh                       | 27770   | 21.8            | 3.2E-06         | 1582           | 2.5E-08        |
| delaunay_n15                    | 32768   | 7.5             | 2.2E-06         | 3668           | 3.9E-07        |
| cit-HepPh                       | 34546   | 4.1             | 4.6E-06         | 3564           | 1.1E-07        |
| p2p-Gnutella30                  | 36682   | 3.0             | 1.8E-06         | 303            | 1.7E-07        |
| vsp_sctap1-2b_and_seymourl      | 40174   | 51.5            | 2.3E-06         | -              | -              |
| vsp_bump2_e18_aa01_model1_crew1 | 56438   | 57.9            | 1.4E-06         | -              | -              |
| delaunay_n16                    | 65536   | 16.4            | 1.3E-06         | -              | -              |
| fe_tooth                        | 78136   | 35.2            | 4.7E-07         | -              | -              |
| 598a                            | 110971  | 58.1            | 4.1E-07         | -              | -              |
| fe_ocean                        | 143437  | 23.7            | 6.1E-09         | -              | -              |
| amazon0302                      | 262111  | 47.7            | 2.3E-07         | -              | -              |
| amazon0505                      | 410236  | 307             | 9.9E-07         | -              | -              |
| delaunay_n19                    | 524288  | 149             | 6.5E-08         | -              | -              |
| rgg_n_2_19_s0                   | 524288  | 251             | 1.9E-06         | -              | -              |
| delaunay_n20                    | 1048576 | 414             | 1.5E-06         | -              | -              |
| rgg_n_2_20_s0                   | 1048576 | 541             | 1.2E-06         | -              | -              |

**Notes:**

- [1] For instances with \( n \geq 40000 \), we only evaluate **LoRADS** and use "-" for other solvers due to their significantly long solving time.
- [2] For **LoRADS**, we apply a phase-switching tolerance of 1e+1 and a heuristic factor of 100 for all problems tested in this table.

#### **Results on Matrix Completion Problems**

| Problem   | n      | m        | LoRADS time | LoRADS errorₘₐₓ | SDPLR time | SDPLR errorₘₐₓ | SDPNAL+ time | SDPNAL+ errorₘₐₓ | COPT time | COPT errorₘₐₓ | ManiSDP time | ManiSDP errorₘₐₓ |
| --------- | ------ | -------- | ----------- | --------------- | ---------- | -------------- | ------------ | ---------------- | --------- | ------------- | ------------ | ---------------- |
| MC_1000   | 1000   | 199424   | 1.9         | 1.8E-06         | 36.2       | 2.3E-05        | 5.6          | 1.5E-06          | 16.7      | 1.3E-06       | 1.4          | 4.7E-10          |
| MC_2000   | 2000   | 550536   | 8.3         | 2.7E-07         | 245        | 2.4E-06        | 43.4         | 2.0E-06          | 124       | 2.0E-06       | 5.7          | 1.8E-08          |
| MC_3000   | 3000   | 930328   | 9.1         | 3.0E-06         | 853        | 1.6E-05        | 179          | 1.4E-06          | 551       | 4.8E-06       | 16.4         | 1.0E-09          |
| MC_4000   | 4000   | 1318563  | 33.9        | 1.3E-07         | 2376       | 4.1E-06        | 232          | 3.0E-06          | 1626      | 6.8E-06       | 28.6         | 1.3E-08          |
| MC_5000   | 5000   | 1711980  | 55.2        | 3.2E-07         | 4501       | 1.9E-06        | 741          | 2.1E-07          | 4142      | 6.7E-06       | 71.5         | 1.5E-08          |
| MC_6000   | 6000   | 2107303  | 61.2        | 4.2E-07         | 6405       | 1.1E-05        | 901          | 1.8E-06          | 8485      | 5.1E-06       | 81.6         | 2.9E-09          |
| MC_8000   | 8000   | 2900179  | 76.6        | 3.2E-07         | t          | t              | 2408         | 6.4E-06          | t         | t             | 251          | 1.4E-08          |
| MC_10000  | 10000  | 3695929  | 39.2        | 1.2E-06         | t          | t              | 8250         | 1.4E-05          | t         | t             | 477          | 1.5E-09          |
| MC_12000  | 12000  | 4493420  | 50.7        | 6.9E-07         | -          | -              | -            | -                | -         | -             | -            | -                |
| MC_14000  | 14000  | 5291481  | 41.8        | 1.6E-06         | -          | -              | -            | -                | -         | -             | -            | -                |
| MC_16000  | 16000  | 6089963  | 89.6        | 3.1E-06         | -          | -              | -            | -                | -         | -             | -            | -                |
| MC_18000  | 18000  | 6889768  | 95.5        | 6.3E-06         | -          | -              | -            | -                | -         | -             | -            | -                |
| MC_20000  | 20000  | 7688309  | 74.6        | 8.2E-07         | -          | -              | -            | -                | -         | -             | -            | -                |
| MC_40000  | 40000  | 15684167 | 282         | 7.2E-07         | -          | -              | -            | -                | -         | -             | -            | -                |
| MC_60000  | 60000  | 23683563 | 1072        | 3.6E-07         | -          | -              | -            | -                | -         | -             | -            | -                |
| MC_80000  | 80000  | 31682246 | 938         | 1.9E-07         | -          | -              | -            | -                | -         | -             | -            | -                |
| MC_100000 | 100000 | 39682090 | 344         | 4.1E-06         | -          | -              | -            | -                | -         | -             | -            | -                |
| MC_120000 | 120000 | 47682387 | 2291        | 3.7E-07         | -          | -              | -            | -                | -         | -             | -            | -                |
| MC_140000 | 140000 | 55682003 | 1514        | 3.5E-06         | -          | -              | -            | -                | -         | -             | -            | -                |
| MC_160000 | 160000 | 63681433 | 2059        | 2.3E-06         | -          | -              | -            | -                | -         | -             | -            | -                |
| MC_180000 | 180000 | 71681424 | 2023        | 7.4E-06         | -          | -              | -            | -                | -         | -             | -            | -                |

**Notes:**

- [1] For instances beyond `MC_10000`, only **LoRADS** is evaluated due to the significantly long solving time of other solvers. We use `-` as placeholder.
- [2] For **LoRADS**, we apply a heuristic factor of 10 for all problems tested in this table.
- [3] For **ManiSDP**, we use the specific parameter settings and function provided in their matrix completion example on GitHub.

#### **Results on Problems from Mittelmann's Benchmark**

| Problem     | n    | m     | LoRADS time | LoRADS errorₘₐₓ | SDPLR time | SDPLR errorₘₐₓ | SDPNAL+ time | SDPNAL+ errorₘₐₓ | COPT time | COPT errorₘₐₓ |
| ----------- | ---- | ----- | ----------- | --------------- | ---------- | -------------- | ------------ | ---------------- | --------- | ------------- |
| 1zc.1024    | 1024 | 16641 | 421         | 1.0E-05         | 1060       | 2.6E-05        | 56.0         | 3.05E-06         | 65.17     | 3.0E-08       |
| AlH         | 5990 | 7230  | 273         | 2.2E-05         | 2290       | 4.9E-04        | 1034         | 7.66E-06         | 1339      | 2.5E-09       |
| Bex2        | 8096 | 3002  | 262         | 6.8E-04         | 294        | 3.4E-05        | 22.8         | 4.49E-06         | 5.3       | 2.5E-07       |
| BH2         | 2166 | 1743  | 47.6        | 4.0E-05         | 142        | 1.2E-04        | 43.5         | 1.70E-06         | 22.5      | 2.5E-09       |
| cancer_100  | 569  | 10470 | 209         | 8.7E-06         | t          | t              | 56.6         | 5.07E-05         | 31.4      | 2.4E-05       |
| CH2         | 2166 | 1743  | 30.2        | 3.4E-05         | 59         | 2.5E-04        | 31.8         | 3.48E-04         | 19.6      | 3.6E-09       |
| checker_1.5 | 3970 | 3971  | 4.8         | 2.7E-07         | 46.4       | 1.4E-04        | 6984         | 5.37E-04         | 31.5      | 4.5E-08       |
| cphil12     | 363  | 12376 | 2.0         | 2.3E-05         | 0.4        | 3.3E-03        | 0.2          | 1.38E-16         | 13.1      | 1.6E-11       |
| G40_mb      | 2000 | 2001  | 11.1        | 5.7E-06         | 60.1       | 3.4E-08        | 665          | 3.33E-07         | 5.9       | 7.6E-06       |
| G48_mb      | 3000 | 3001  | 17.2        | 8.6E-07         | 29.9       | 7.7E-04        | 1816         | 7.15E-02         | 8.4       | 6.4E-07       |
| G48mc       | 3000 | 3000  | 0.1         | 1.4E-06         | 0.7        | 1.9E-07        | 1508         | 4.00E-03         | 2.9       | 5.3E-09       |
| G55mc       | 5000 | 5000  | 0.5         | 7.3E-06         | 3.4        | 4.3E-07        | 6005         | 1.44E-06         | 20.6      | 7.3E-09       |
| G59mc       | 5000 | 5000  | 3.5         | 6.1E-06         | 26.3       | 1.0E-06        | 7819         | 1.28E-05         | 29.4      | 8.5E-09       |
| G60_mb      | 7000 | 7001  | 152         | 1.7E-06         | 1140       | 1.0E-08        | t            | t                | 123       | 8.8E-08       |
| H3O         | 3162 | 2964  | 27.8        | 9.9E-06         | 206        | 7.1E-05        | 115          | 5.39E-06         | 67.1      | 3.5E-09       |
| hand        | 1296 | 1297  | 6.0         | 1.4E-06         | 32.4       | 1.5E-08        | 281          | 2.14E-07         | 2.0       | 8.9E-05       |
| ice_2.0     | 8113 | 8113  | 8.0         | 1.8E-06         | 38.6       | 1.0E-05        | t            | t                | 201       | 2.0E-08       |
| neosfbr25   | 577  | 14376 | 3694        | 1.0E-05         | 2400       | 1.0E-05        | 402          | 7.60E-05         | 102       | 1.5E-09       |
| neosfbr30e8 | 842  | 25201 | 7126        | 1.0E-05         | 7400       | 1.0E-05        | 966          | 9.80E-05         | 512       | 9.5E-09       |
| NH2         | 2046 | 1743  | 5.1         | 4.6E-05         | 22         | 8.5E-05        | 12.2         | 9.10E-06         | 18.3      | 2.5E-09       |
| NH3         | 3162 | 2964  | 63.9        | 9.5E-06         | 189        | 2.4E-04        | 63.2         | 2.66E-05         | 72.8      | 3.0E-09       |
| NH4         | 4426 | 4743  | 46.2        | 2.5E-05         | 224        | 3.1E-04        | 122          | 7.47E-04         | 366       | 5.3E-09       |
| p_auss2_3.0 | 9115 | 9115  | 5.6         | 6.8E-06         | 69.8       | 3.1E-05        | t            | t                | 235       | 4.7E-06       |
| rendl1_2000 | 2000 | 2001  | 55.7        | 8.9E-07         | 359        | 1.5E-08        | 634          | 4.12E-07         | 5.2       | 1.4E-05       |
| shmup4      | 799  | 4962  | 117         | 9.2E-06         | 80.8       | 2.2E-05        | 8417         | 5.40E-01         | 145       | 4.5E-06       |
| theta102    | 500  | 37467 | 197         | 3.0E-05         | 224        | 2.5E-05        | 6.7          | 1.12E-06         | 3.3       | 2.8E-09       |
| theta12     | 600  | 17979 | 125         | 1.3E-06         | 170        | 1.8E-05        | 13.6         | 5.31E-06         | 4.5       | 1.7E-08       |
| theta123    | 600  | 90020 | 590         | 9.8E-06         | 497        | 5.7E-05        | 12.2         | 3.86E-07         | 4.9       | 1.9E-08       |

**Notes:**

- [1] For all the problems tested in this table, we apply the default setting of **LoRADS** as detailed in Section 5.1.

#### Contributing

LoRADS is still in its preliminary release and will start accepting pull requests in a future release.

#### Developers

LoRADS is developed by

- Zhenwei Lin: zhenweilin@163.sufe.edu.cn
- Qiushi Han: joshhan2@illinois.edu

#### Reference

- Qiushi Han, Chenxi Li, Zhenwei Lin, Caihua Chen, Qi Deng, Dongdong Ge, Huikang Liu, and Yinyu Ye. "A Low-Rank ADMM Splitting Approach for Semidefinite Programming." _arXiv preprint arXiv:2403.09133_ (2024).
- Burer, Samuel, R. Monteiro, and Changhui Choi. "SDPLR 1.03-beta User’s Guide (short version)(2009)." _URL http://sburer.github.io/files/SDPLR-1.03-beta-usrguide.pdf_.
