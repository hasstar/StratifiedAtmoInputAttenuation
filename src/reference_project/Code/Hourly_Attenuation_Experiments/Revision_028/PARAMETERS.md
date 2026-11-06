# Parameters — why / how / source

All numeric choices are frozen before new test scoring. No value is claimed a guaranteed optimum.

|Parameter|Value / construction|Why and source|
|---|---|---|
|Data and target|2404 train,343 validation,687 test; clear_sky_attenuation_db|Existing Signal_Detectability_Revision_003; no favorable resplit or changed target|
|Inputs|base50 / full156, same list asRev023|Controlled addition of106 pressure columns, no new feature selection|
|Weather endpoints|overall,rain,dry_overcast|User requires all three to improve; rain unifies all rain intensities|
|Selection S|sqrt(mean of the three MSEs)|Equal consideration of requested endpoints; an engineering loss, not a physical law|
|Training weights|uniform1 or balanced(1+n*rain/nrain+n*cloud/ncloud)/3|Balanced loss equals the three-MSE mean exactly; counts from fitting hours only; every hour retains positive weight|
|Search budget|8 per family/input:4 original controls+4 extensions|Same opportunities for base/full and all10 families; original4 grids fromRev014 models.py; cannot remove losing candidates|
|Extension elastic|alpha x10, original l1_ratio|Bounded stronger coefficient shrinkage than control; both pools receive identical choices|
|Extension SVR|C /10, original epsilon/gamma|Stronger regularization via smaller penalty; sample weights rescale per-sample C per official API|
|Extension forest|min_samples_leaf at least5; original depth/cap|Discourage tiny leaves; computational engineering choice|
|Extension XGB|reg_lambda10,min_child_weight5,colsample_bytree0.6|Greater shrinkage/less fragile splits; originals remain controls|
|Extension LGB|reg_lambda10,min_child_samples40|Greater shrinkage/larger leaves; originals remain controls|
|Extension Cat|l2_leaf_reg10|More regularization; originals remain controls|
|Neural cap|160 epochs in all8 configs|Double earlier maximum80 to permit weighted objective convergence; bounded cap, no test-driven extension|
|Neural controls|lr.001/.0003,dropout.1/.2,wd.0001/.001|Original4 neural candidates, expanded uniform cap|
|Neural extensions|same lr,dropout+0.1,weight_decay x10|Predeclared stronger regularization, not claimed physical derivation|
|Fusion gate|32-component sigmoid(Linear(base32)), learned weights|Suppress unneeded pressure representation;0 bias gives neutral.5 at zero input; no imposed physical contribution. TFT gating motivation, not full TFT implementation|
|Architecture|FlatMLP64/32; existing vertical projection32;BiLSTM hidden32 bidirectional;Transformer2layers/4heads;GCN2layers|Rev014 architecture reused. Vertical axis is pressure, not an hourly future time sequence|
|Neural optimization|AdamW,batch128,2CPUthreads,target mean/std train only|Legacy implementation continuity; coefficients from fitting target only|
|Patience/tolerance|20 no-improvement epochs; improvement >1e-12 dB in S|Bounded early-stop engineering choice; earliest minimum selected, avoids numerical ties|
|Tree/solver rounds|Original caps300/500;Elasticmax_iter20000,tol1e-4;SVRconvergence|Reuse4 grids. Tree prefix selected AFTER cap-fit; solver iterations are not neural epochs|
|Seeds|17 tuning;17/29/43 formal|Legacy reproducibility; algorithm repetitions, not independent meteorological samples|
|Aggregation|Mean seed-specific RMSE/MAE/Bias|Same fair ranking asRev026/027; not ensemble RMSE|
|Bootstrap|4096 observed-date draws,seed20261011|Fixed computational precision and repeatability; whole dates preserve within-day correlation; no correction of multi-day dependence|
|Input imputation/scaling|Training median; StandardScaler for scaled models; empty neutral slots|ReuseRev026, fit train only. Full raw156 inputs remain intact; no projection/PCA introduced in this stage|
|Continuity guard|New winner overall <=0.28567316472555604 dB|OldQA-passed benchmark retained; cannot achieve a desired story by worsening all competitors|
|QA tolerances|Prediction replay1e-10; scalar metric1e-12|Numerical equivalence, not scientific effect-size threshold|
|Monitoring|Every15 minutes; unchanged healthy state quiet|Explicit user approval2026-10-11; matching live worker prohibits duplicate launch|

Training sample counts/weight distribution and configuration expansion are serialized in run_manifest.json. All failed numerical/convergence states and warnings are preserved. Validation/test labels do not enter the model input tensor. Original and new complete curves permit identifying the true fit ending and selected checkpoint separately.
