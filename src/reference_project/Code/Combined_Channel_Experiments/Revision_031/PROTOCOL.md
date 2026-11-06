# 英国 E-SST 有限输入采集与质量核验

本版执行2026年10月18日新的人工授权，只补齐Revision030中421104个明确缺失的ERA5再分析数值并做输入QA。不计算英国衰减或排名。当前主比较已改为五类，RMOP与NWP–PRM按用户决定移除，MTS继续未计算。详细依据见Outputs/Automation_Changes/TVT_Comparison_20261018_001/CHANGE_RECORD.md。

## 固定输入与采集边界

沿用030的input_batches.json、scope_hours.json、reused_geopotential.json和来源保护链。只允许reanalysis-era5-pressure-levels、两个单点、原30个精确批次。温度37层、位势缺失层、700hPa的u/v，UTC每日00至23时瞬时分析；不新增雨量、上下文日期或场站，不借用训练或测试标签。净新增421104值，旧位势9360值复用。全部5664小时保留，获取不承诺全部可算。

使用现有官方cdsapi配置与ecmwf-datastores-client SDK，不安装依赖、不运行或导入旧weather_download入口。凭据仅在内存供指定CDS认证，不写日志或输出。TLS校验开启，不接受新条款、不换源、不购买。官方说明：https://cds.climate.copernicus.eu/how-to-api 和 https://ecmwf.github.io/ecmwf-datastores-client/README.html 。原产品许可由服务端校验；若未接受或授权不足立即停止。

单worker、单个CDS任务顺序执行。每批提交前保存意图，返回后保存作业ID；意图存在但ID丢失时不重提。恢复只查询已记录ID，不自动删除服务端作业。获取前查服务返回大小，累计HTTP响应体设255MiB限额，另留1MiB传输开销余量，整体授权256MiB；超限停。原始字节、失败和部分文件保留，不能覆盖重试。

## 验收

生产用netCDF4核对变量、单位、唯一格点、UTC小时集合、压力层集合、数组形状、有限性、再分析版本和源SHA。独立QA用h5py直接读取全部原数组，独立展开030原始目标键核对完整性、无额外数据、无重复，复查SHA。输入QA不是模型QA；未知英国垂直基准、原生功率采样和物理适配仍须另行确定。

历史保护源前后SHA不变，唯一允许的既有文件变化是用户授权的tvt提示更新，旧新配置在变更目录存档。不改STOP。每次运行创建独立run记录，OS锁防双启动。中断与服务阻塞必须保留状态，不把请求提交或代码测试称为数据补齐。

## Preflight 与复用

06:14Z真实进程无相关worker，031目录未占用；5040历史源与23封存文件核验一致，约165GB余量。README、T012、001/PROGRESS、030协议交接QA及最新人工决定已读。复用030精确计划与已有SDK，新增仅限有界请求、日志与输入验收的薄适配。旧下载器因STOP不可执行；它们不是本次可复用入口。

C1/C2/C3=POTENTIAL LOW：031新写域与SHA保护，历史只读。C4=CONFIRMED：五类范围及本次天气例外由最新人工决定支持，不覆盖LOCAL批准内容。C5=NONE：原日期全量，不按标签筛选。C6=POTENTIAL LOW：单请求、累计上限、无GPU。C7=CONFIRMED：数据服务许可/网络待实际检查；UK物理适配尚不放行。C8=NONE：新授权的缺项未采集，已有9360值不重采。Decision=PARALLEL_ALLOWED仅限本隔离任务，不授权代理或其他任务。

## Execution Lock

Owner=/root；Session=01a0bda1-92d1-7832-9508-60f4e0ec1634；Started=2026-10-18T06:14Z；Lock Status=LOCKED。Path Base=项目根；READ=030及其历史来源、现有SDK；WRITE=031代码与输出及本次Automation_Changes目录；SHARED=本机有限CPU与CDS单请求。worker_os.lock提供运行期互斥，实际进程/每次run结果为准。文档不是进程证明，未知外部写者UNKNOWN；最终通过HANDOFF与状态记录交接，不移除历史锁或失败。
